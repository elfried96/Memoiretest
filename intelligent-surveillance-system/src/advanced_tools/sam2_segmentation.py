"""SAM2 Segmentation for precise object segmentation."""

import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import List, Dict, Any, Optional, Tuple
import logging
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)

@dataclass
class SegmentationResult:
    """Result from SAM2 segmentation."""
    masks: np.ndarray
    scores: np.ndarray
    boxes: np.ndarray
    processing_time: float
    
class SAM2Segmentator:
    """SAM2 segmentation for enhanced object detection precision."""
    
    def __init__(self, model_path: str = "facebook/sam2-hiera-large", device: str = "auto"):
        self.device = self._select_device(device)
        self.model = None
        self.processor = None
        self.model_path = model_path
        self._load_model()
        
    def _select_device(self, device: str) -> torch.device:
        """Select optimal device for computation."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    def _load_model(self):
        """Load SAM2 model with optimizations."""
        try:
            from transformers import Sam2Model, Sam2Processor
            
            self.processor = Sam2Processor.from_pretrained(self.model_path)
            self.model = Sam2Model.from_pretrained(self.model_path).to(self.device)
            
            # Optimizations
            if self.device.type == "cuda":
                self.model = self.model.half()
                
            self.model.eval()
            logger.info(f"SAM2 model loaded on {self.device}")
            
        except ImportError:
            logger.warning("SAM2 not available, using fallback segmentation")
            self.model = None
    
    @torch.inference_mode()
    def segment_objects(self, frame: np.ndarray, bounding_boxes: List[List[float]], 
                       confidence_threshold: float = 0.8) -> SegmentationResult:
        """Segment objects using SAM2 with bounding box prompts."""
        start_time = time.perf_counter()
        
        if self.model is None:
            # Fallback to simple mask generation
            return self._fallback_segmentation(frame, bounding_boxes, start_time)
        
        try:
            # Prepare inputs
            inputs = self.processor(
                images=frame,
                input_boxes=[bounding_boxes],
                return_tensors="pt"
            ).to(self.device)
            
            # Forward pass
            with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
                outputs = self.model(**inputs)
            
            # Extract results
            masks = outputs.pred_masks.cpu().numpy()
            scores = outputs.iou_scores.cpu().numpy()
            
            # Filter by confidence
            valid_indices = scores > confidence_threshold
            masks = masks[valid_indices]
            scores = scores[valid_indices]
            boxes = np.array([box for i, box in enumerate(bounding_boxes) if valid_indices[i]])
            
            processing_time = time.perf_counter() - start_time
            
            return SegmentationResult(
                masks=masks,
                scores=scores,
                boxes=boxes,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"SAM2 segmentation failed: {e}")
            return self._fallback_segmentation(frame, bounding_boxes, start_time)
    
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
            processing_time=processing_time
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