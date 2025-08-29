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
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            
            logger.info(f"Loading SAM2 model...")
            
            # Utilisation du modèle SAM2 officiel
            # Mapping des noms de modèles
            model_cfg_map = {
                "facebook/sam2-hiera-large": "sam2_hiera_l.yaml",
                "sam2-hiera-large": "sam2_hiera_l.yaml",
                "sam2-hiera-base": "sam2_hiera_b.yaml", 
                "sam2-hiera-small": "sam2_hiera_s.yaml",
                "sam2-hiera-tiny": "sam2_hiera_t.yaml"
            }
            
            # Sélection du config
            config_name = model_cfg_map.get(self.model_path, "sam2_hiera_l.yaml")
            
            # Construction du modèle SAM2
            sam2_model = build_sam2(config_name, device=self.device)
            self.model = SAM2ImagePredictor(sam2_model)
            
            # Pas besoin de processor avec l'API SAM2 native
            self.processor = None
            
            logger.info(f"SAM2 model loaded on {self.device}")
            self._model_loaded = True
            
        except ImportError as e:
            logger.error(f"SAM2 library not available: {e}")
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
            # Set image for SAM2
            self.model.set_image(frame)
            
            # Prepare prompts based on type
            if prompt.prompt_type == PromptType.BOUNDING_BOX and prompt.bounding_boxes:
                # Use first bounding box
                bbox = prompt.bounding_boxes[0]
                input_box = np.array([[bbox[0], bbox[1], bbox[2], bbox[3]]])
                masks, scores, logits = self.model.predict(
                    point_coords=None,
                    point_labels=None, 
                    box=input_box,
                    multimask_output=True,
                )
                
            elif prompt.prompt_type == PromptType.POINTS and prompt.points and prompt.point_labels:
                # Use points
                point_coords = np.array(prompt.points[0])
                point_labels = np.array(prompt.point_labels[0])
                masks, scores, logits = self.model.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    box=None,
                    multimask_output=True,
                )
                
            else:
                # Fallback: use center point
                h, w = frame.shape[:2]
                center_point = np.array([[w//2, h//2]])
                center_label = np.array([1])
                masks, scores, logits = self.model.predict(
                    point_coords=center_point,
                    point_labels=center_label,
                    box=None,
                    multimask_output=True,
                )
            
            # Filter by confidence
            if confidence_threshold > 0:
                valid_mask = scores >= confidence_threshold
                masks = masks[valid_mask]
                scores = scores[valid_mask]
            
            # Convert to expected format
            if len(masks) == 0:
                # No valid masks
                return self._fallback_segmentation(frame, prompt.bounding_boxes or [], start_time)
                
            # Take best mask
            best_idx = np.argmax(scores)
            best_mask = masks[best_idx]
            best_score = scores[best_idx]
            
            # Generate dummy box from mask
            y_indices, x_indices = np.where(best_mask)
            if len(y_indices) > 0:
                x1, x2 = x_indices.min(), x_indices.max()
                y1, y2 = y_indices.min(), y_indices.max()
                boxes = np.array([[x1, y1, x2, y2]])
            else:
                boxes = np.array([[0, 0, frame.shape[1], frame.shape[0]]])
            
            return SegmentationResult(
                masks=masks,
                scores=scores,
                boxes=boxes,
                processing_time=time.perf_counter() - start_time,
                prompt_type=prompt.prompt_type
            )
            
        except Exception as e:
            logger.error(f"SAM2 segmentation failed: {e}")
            return self._fallback_segmentation(frame, prompt.bounding_boxes or [], start_time)
    
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
    
    def _fallback_segmentation(self, frame: np.ndarray, bounding_boxes: List[List[float]], 
                              start_time: float) -> SegmentationResult:
        """Fallback segmentation using basic CV techniques."""
        logger.info("Using fallback segmentation")
        
        # Create simple masks from bounding boxes
        if bounding_boxes:
            masks = []
            scores = []
            
            for bbox in bounding_boxes:
                x1, y1, x2, y2 = map(int, bbox)
                mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=bool)
                mask[y1:y2, x1:x2] = True
                masks.append(mask)
                scores.append(0.5)  # Default confidence
                
            return SegmentationResult(
                masks=np.array(masks),
                scores=np.array(scores),
                boxes=np.array(bounding_boxes),
                processing_time=time.perf_counter() - start_time,
                prompt_type=PromptType.BOUNDING_BOX
            )
        else:
            # Single mask covering entire frame
            mask = np.ones((frame.shape[0], frame.shape[1]), dtype=bool)
            return SegmentationResult(
                masks=np.array([mask]),
                scores=np.array([0.5]),
                boxes=np.array([[0, 0, frame.shape[1], frame.shape[0]]]),
                processing_time=time.perf_counter() - start_time,
                prompt_type=PromptType.BOUNDING_BOX
            )
    
    # Méthodes utilitaires
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
    
    @torch.inference_mode()
    def segment_everything(self, frame: np.ndarray, confidence_threshold: float = 0.8) -> SegmentationResult:
        """Segment all objects in the frame using automatic point grid."""
        start_time = time.perf_counter()
        
        self._ensure_model_loaded()
        
        if self.model is None:
            return self._fallback_segmentation(frame, [], start_time)
        
        try:
            # Set image for SAM2
            self.model.set_image(frame)
            
            # Generate automatic grid of points
            h, w = frame.shape[:2]
            grid_points = []
            grid_labels = []
            
            # Create a 3x3 grid of points
            for i in range(3):
                for j in range(3):
                    x = int(w * (j + 1) / 4)
                    y = int(h * (i + 1) / 4)
                    grid_points.append([x, y])
                    grid_labels.append(1)  # All positive points
            
            point_coords = np.array([grid_points])
            point_labels = np.array([grid_labels])
            
            # Predict with grid points
            masks, scores, logits = self.model.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=None,
                multimask_output=True,
            )
            
            # Filter by confidence
            if confidence_threshold > 0:
                valid_mask = scores >= confidence_threshold
                masks = masks[valid_mask]
                scores = scores[valid_mask]
            
            # Generate boxes from masks
            boxes = []
            for mask in masks:
                y_indices, x_indices = np.where(mask)
                if len(y_indices) > 0:
                    x1, x2 = x_indices.min(), x_indices.max()
                    y1, y2 = y_indices.min(), y_indices.max()
                    boxes.append([x1, y1, x2, y2])
                else:
                    boxes.append([0, 0, w, h])
            
            return SegmentationResult(
                masks=masks,
                scores=scores,
                boxes=np.array(boxes),
                processing_time=time.perf_counter() - start_time,
                prompt_type=PromptType.POINTS
            )
            
        except Exception as e:
            logger.error(f"SAM2 segment_everything failed: {e}")
            return self._fallback_segmentation(frame, [], start_time)

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded model."""
        return {
            "model_name": self.model_path,
            "device": str(self.device),
            "model_loaded": self._model_loaded,
            "lazy_loading": self.lazy_loading,
            "model_available": self.model is not None
        }