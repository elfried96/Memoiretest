"""SAM2 Segmentation for precise object segmentation."""

import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
from dataclasses import dataclass
import time
from enum import Enum

logger = logging.getLogger(__name__)

class PromptType(Enum):
    """Types de prompts supportés par SAM2."""
    BOUNDING_BOX = "bbox"
    POINTS = "points"  
    MASK = "mask"
    COMBINED = "combined"

@dataclass
class SegmentationPrompt:
    """Unified prompt structure for SAM2."""
    prompt_type: PromptType
    bounding_boxes: Optional[List[List[float]]] = None
    points: Optional[List[List[List[float]]]] = None
    point_labels: Optional[List[List[int]]] = None
    input_masks: Optional[np.ndarray] = None
    
@dataclass
class SegmentationResult:
    """Result from SAM2 segmentation."""
    masks: np.ndarray
    scores: np.ndarray
    boxes: np.ndarray
    processing_time: float
    prompt_type: PromptType
    metadata: Optional[Dict[str, Any]] = None
    
class SAM2Segmentator:
    """SAM2 segmentation for enhanced object detection precision."""
    
    def __init__(self, model_path: str = "facebook/sam2-hiera-large", device: str = "auto", lazy_loading: bool = True):
        self.device = self._select_device(device)
        self.model = None
        self.processor = None
        self.model_path = model_path
        self.lazy_loading = lazy_loading
        self._model_loaded = False
        
        # Chargement immédiat ou lazy selon paramètre
        if not lazy_loading:
            self._load_model()
        
    def _select_device(self, device: str) -> torch.device:
        """Select optimal device for computation."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    def _ensure_model_loaded(self):
        """Ensure model is loaded (lazy loading)."""
        if not self._model_loaded:
            self._load_model()
    
    def _load_model(self):
        """Load SAM2 model with optimizations."""
        if self._model_loaded:
            return
            
        try:
            from transformers import Sam2Model, Sam2Processor
            
            logger.info(f"Loading SAM2 model {self.model_path}...")
            
            # Chargement avec gestion mémoire
            self.processor = Sam2Processor.from_pretrained(self.model_path)
            
            # Chargement avec optimisations selon le device
            if self.device.type == "cuda":
                # GPU: utiliser bfloat16 pour économiser VRAM
                self.model = Sam2Model.from_pretrained(self.model_path).to(self.device, dtype=torch.bfloat16)
                logger.info(f"SAM2 model loaded on {self.device} with bfloat16")
            else:
                # CPU: utiliser float32 pour compatibilité
                self.model = Sam2Model.from_pretrained(self.model_path).to(self.device)
                logger.info(f"SAM2 model loaded on {self.device} with float32")
            
            self.model.eval()
            self._model_loaded = True
            
        except ImportError as e:
            logger.error(f"Transformers library not available: {e}")
            self.model = None
            self._model_loaded = False
        except Exception as e:
            logger.error(f"Failed to load SAM2 model: {e}")
            self.model = None  
            self._model_loaded = False
    
    @torch.inference_mode()
    def segment_with_prompt(self, frame: np.ndarray, prompt: SegmentationPrompt, 
                           confidence_threshold: float = 0.8) -> SegmentationResult:
        """Universal segmentation method supporting all prompt types."""
        start_time = time.perf_counter()
        
        # Ensure model is loaded
        self._ensure_model_loaded()
        
        if self.model is None:
            return self._fallback_segmentation(frame, prompt.bounding_boxes or [], start_time)
        
        try:
            # Prepare inputs based on prompt type
            inputs = self._prepare_inputs(frame, prompt)
            
            # Forward pass
            dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
            with torch.autocast(self.device.type, dtype=dtype):
                outputs = self.model(**inputs)
            
            # Post-process results
            result = self._post_process_outputs(outputs, inputs, prompt, start_time)
            return result
            
        except Exception as e:
            logger.error(f"SAM2 segmentation failed: {e}")
            return self._fallback_segmentation(frame, prompt.bounding_boxes or [], start_time)
    
    def _prepare_inputs(self, frame: np.ndarray, prompt: SegmentationPrompt) -> Dict[str, torch.Tensor]:
        """Prepare inputs for SAM2 based on prompt type."""
        
        base_inputs = {"images": frame, "return_tensors": "pt"}
        
        # Add prompt-specific inputs
        if prompt.prompt_type == PromptType.BOUNDING_BOX and prompt.bounding_boxes:
            base_inputs["input_boxes"] = [[prompt.bounding_boxes]]
            
        elif prompt.prompt_type == PromptType.POINTS and prompt.points and prompt.point_labels:
            base_inputs["input_points"] = [prompt.points]
            base_inputs["input_labels"] = [prompt.point_labels]
            
        elif prompt.prompt_type == PromptType.MASK and prompt.input_masks is not None:
            # Convert numpy mask to tensor if needed
            mask_tensor = torch.from_numpy(prompt.input_masks) if isinstance(prompt.input_masks, np.ndarray) else prompt.input_masks
            base_inputs["input_masks"] = mask_tensor.unsqueeze(0) if len(mask_tensor.shape) == 2 else mask_tensor
            
        elif prompt.prompt_type == PromptType.COMBINED:
            # Multiple prompt types combined
            if prompt.bounding_boxes:
                base_inputs["input_boxes"] = [[prompt.bounding_boxes]]
            if prompt.points and prompt.point_labels:
                base_inputs["input_points"] = [prompt.points]
                base_inputs["input_labels"] = [prompt.point_labels]
            if prompt.input_masks is not None:
                mask_tensor = torch.from_numpy(prompt.input_masks) if isinstance(prompt.input_masks, np.ndarray) else prompt.input_masks
                base_inputs["input_masks"] = mask_tensor.unsqueeze(0) if len(mask_tensor.shape) == 2 else mask_tensor
        
        # Process inputs and move to device
        inputs = self.processor(**base_inputs).to(self.device)
        return inputs
    
    def _post_process_outputs(self, outputs, inputs: Dict[str, torch.Tensor], 
                            prompt: SegmentationPrompt, start_time: float) -> SegmentationResult:
        """Post-process SAM2 outputs."""
        
        # Post-process masks using official HF method
        masks = self.processor.post_process_masks(
            outputs.pred_masks.cpu(), 
            inputs["original_sizes"]
        )[0]
        
        # Extract IoU scores
        scores = outputs.iou_scores.cpu().numpy().squeeze()
        
        # Handle multiple masks - take best by default
        if len(scores.shape) > 0 and scores.shape[0] > 1:
            best_mask_idx = torch.argmax(outputs.iou_scores.squeeze())
            masks = masks[best_mask_idx:best_mask_idx+1]
            scores = np.array([scores[best_mask_idx]])
        else:
            scores = np.array([scores.item()] if scores.ndim == 0 else scores)
        
        # Convert masks to numpy if tensor
        if isinstance(masks, torch.Tensor):
            masks = masks.numpy()
        
        processing_time = time.perf_counter() - start_time
        
        return SegmentationResult(
            masks=masks,
            scores=scores,
            boxes=np.array(prompt.bounding_boxes) if prompt.bounding_boxes else np.array([]),
            processing_time=processing_time,
            prompt_type=prompt.prompt_type,
            metadata={
                "model_path": self.model_path,
                "device": str(self.device),
                "num_masks": len(masks)
            }
        )
    
    # Méthodes de convenance pour rétrocompatibilité
    @torch.inference_mode()
    def segment_objects(self, frame: np.ndarray, bounding_boxes: List[List[float]], 
                       confidence_threshold: float = 0.8) -> SegmentationResult:
        """Segment objects using SAM2 with bounding box prompts."""
        prompt = SegmentationPrompt(
            prompt_type=PromptType.BOUNDING_BOX,
            bounding_boxes=bounding_boxes
        )
        return self.segment_with_prompt(frame, prompt, confidence_threshold)
    
    @torch.inference_mode()  
    def segment_objects_old(self, frame: np.ndarray, bounding_boxes: List[List[float]], 
                       confidence_threshold: float = 0.8) -> SegmentationResult:
        """Legacy method - kept for backward compatibility."""
        start_time = time.perf_counter()
        
        self._ensure_model_loaded()
        
        if self.model is None:
            # Fallback to simple mask generation
            return self._fallback_segmentation(frame, bounding_boxes, start_time)
        
        try:
            # Format bounding boxes selon le standard HF : [[[x1, y1, x2, y2], ...]]
            formatted_boxes = [[bounding_boxes]]  # Wrap en 3 dimensions
            
            # Prepare inputs selon documentation HF
            inputs = self.processor(
                images=frame,
                input_boxes=formatted_boxes,
                return_tensors="pt"
            ).to(self.device)
            
            # Forward pass avec torch.inference_mode (déjà appliqué via décorateur)
            with torch.autocast(self.device.type, dtype=torch.bfloat16):
                outputs = self.model(**inputs)
            
            # Post-processing selon documentation HF
            masks = self.processor.post_process_masks(
                outputs.pred_masks.cpu(), 
                inputs["original_sizes"]
            )[0]
            
            # Extraction des scores IoU
            scores = outputs.iou_scores.cpu().numpy().squeeze()
            
            # Filter by confidence (si plusieurs masques générés)
            if len(scores.shape) > 0 and scores.shape[0] > 1:
                # Prendre le meilleur masque par défaut ou filtrer
                best_mask_idx = torch.argmax(outputs.iou_scores.squeeze())
                masks = masks[best_mask_idx:best_mask_idx+1]  # Garder dimension batch
                scores = np.array([scores[best_mask_idx]])
            else:
                scores = np.array([scores.item()] if scores.ndim == 0 else scores)
            
            # Convertir masks en numpy si tensor
            if isinstance(masks, torch.Tensor):
                masks = masks.numpy()
            
            processing_time = time.perf_counter() - start_time
            
            return SegmentationResult(
                masks=masks,
                scores=scores,
                boxes=np.array(bounding_boxes),
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"SAM2 segmentation failed: {e}")
            return self._fallback_segmentation(frame, bounding_boxes, start_time)
    
    @torch.inference_mode()
    def segment_objects_with_points(self, frame: np.ndarray, 
                                   input_points: List[List[List[float]]], 
                                   input_labels: List[List[int]],
                                   confidence_threshold: float = 0.8) -> SegmentationResult:
        """Segment objects using SAM2 with point prompts (following HF documentation)."""
        start_time = time.perf_counter()
        
        if self.model is None:
            return self._fallback_segmentation(frame, [], start_time)
        
        try:
            # Format selon documentation HF: input_points = [[[[500, 375]]]]
            formatted_points = [input_points]  # Wrap pour batch dimension
            formatted_labels = [input_labels]  # Wrap pour batch dimension
            
            # Prepare inputs selon documentation HF
            inputs = self.processor(
                images=frame,
                input_points=formatted_points,
                input_labels=formatted_labels,
                return_tensors="pt"
            ).to(self.device)
            
            # Forward pass
            with torch.autocast(self.device.type, dtype=torch.bfloat16):
                outputs = self.model(**inputs)
            
            # Post-processing selon documentation HF
            masks = self.processor.post_process_masks(
                outputs.pred_masks.cpu(), 
                inputs["original_sizes"]
            )[0]
            
            # Extraction des scores IoU
            scores = outputs.iou_scores.cpu().numpy().squeeze()
            
            # Prendre le meilleur masque
            if len(scores.shape) > 0 and scores.shape[0] > 1:
                best_mask_idx = torch.argmax(outputs.iou_scores.squeeze())
                masks = masks[best_mask_idx:best_mask_idx+1]
                scores = np.array([scores[best_mask_idx]])
            else:
                scores = np.array([scores.item()] if scores.ndim == 0 else scores)
            
            if isinstance(masks, torch.Tensor):
                masks = masks.numpy()
            
            processing_time = time.perf_counter() - start_time
            
            return SegmentationResult(
                masks=masks,
                scores=scores,
                boxes=np.array([]),  # Pas de boxes avec points
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"SAM2 point segmentation failed: {e}")
            return self._fallback_segmentation(frame, [], start_time)
    
    def _fallback_segmentation(self, frame: np.ndarray, bounding_boxes: List[List[float]], 
                             start_time: float) -> SegmentationResult:
        """Fallback segmentation using traditional CV methods."""
        masks = []
        scores = []
        
        for box in bounding_boxes:
            x1, y1, x2, y2 = map(int, box)
            roi = frame[y1:y2, x1:x2]
            
            # Simple edge-based segmentation
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # Create mask
            mask = np.zeros_like(gray)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                cv2.fillPoly(mask, [largest_contour], 255)
            
            # Resize mask to full frame
            full_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            full_mask[y1:y2, x1:x2] = mask
            
            masks.append(full_mask)
            scores.append(0.5)  # Default confidence
        
        processing_time = time.perf_counter() - start_time
        
        return SegmentationResult(
            masks=np.array(masks),
            scores=np.array(scores),
            boxes=np.array(bounding_boxes),
            processing_time=processing_time,
            prompt_type=PromptType.BOUNDING_BOX
        )
    
    def get_mask_properties(self, mask: np.ndarray) -> Dict[str, Any]:
        """Extract properties from segmentation mask."""
        area = np.sum(mask > 0)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {"area": 0, "perimeter": 0, "compactness": 0, "solidity": 0}
        
        largest_contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(largest_contour, True)
        
        # Compactness (4π * area / perimeter²)
        compactness = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
        
        # Solidity (area / convex hull area)
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        return {
            "area": float(area),
            "perimeter": float(perimeter),
            "compactness": float(compactness),
            "solidity": float(solidity)
        }
    
    # Helper methods pour créer des prompts facilement
    @staticmethod
    def create_bbox_prompt(bounding_boxes: List[List[float]]) -> SegmentationPrompt:
        """Create bounding box prompt."""
        return SegmentationPrompt(
            prompt_type=PromptType.BOUNDING_BOX,
            bounding_boxes=bounding_boxes
        )
    
    @staticmethod
    def create_point_prompt(points: List[List[List[float]]], labels: List[List[int]]) -> SegmentationPrompt:
        """Create point prompt with labels (1=positive, 0=negative)."""
        return SegmentationPrompt(
            prompt_type=PromptType.POINTS,
            points=points,
            point_labels=labels
        )
    
    @staticmethod
    def create_mask_prompt(input_mask: np.ndarray) -> SegmentationPrompt:
        """Create mask prompt from previous segmentation."""
        return SegmentationPrompt(
            prompt_type=PromptType.MASK,
            input_masks=input_mask
        )
    
    @staticmethod
    def create_combined_prompt(bounding_boxes: Optional[List[List[float]]] = None,
                              points: Optional[List[List[List[float]]]] = None,
                              point_labels: Optional[List[List[int]]] = None,
                              input_masks: Optional[np.ndarray] = None) -> SegmentationPrompt:
        """Create combined prompt with multiple input types."""
        return SegmentationPrompt(
            prompt_type=PromptType.COMBINED,
            bounding_boxes=bounding_boxes,
            points=points,
            point_labels=point_labels,
            input_masks=input_masks
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded model."""
        return {
            "model_name": self.model_path,
            "device": str(self.device),
            "model_loaded": self._model_loaded,
            "lazy_loading": self.lazy_loading,
            "processor_available": self.processor is not None,
            "model_available": self.model is not None
        }