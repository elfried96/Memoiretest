"""
Service de détection YOLO11 avec tracking ByteTrack.
Composant temps réel du pipeline de surveillance.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from loguru import logger

from src.core.config import settings


@dataclass
class Detection:
    """Résultat de détection unique."""
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox: list[float]  # [x1, y1, x2, y2] normalisé 0-1


@dataclass
class FrameResult:
    """Résultat complet pour un frame."""
    frame_number: int
    timestamp: datetime
    detections: list[Detection]
    inference_time_ms: float


class DetectionService:
    """
    Service de détection YOLO11 + ByteTrack.
    
    Usage:
        service = DetectionService()
        await service.initialize()
        result = service.process_frame(frame, frame_number)
    """
    
    # Mapping classes COCO
    COCO_CLASSES = {
        0: "person",
        24: "backpack", 
        25: "umbrella",
        26: "handbag",
        28: "suitcase",
        39: "bottle",
        67: "cell phone",
        73: "book"
    }
    
    def __init__(self):
        self.model = None
        self.device = settings.detection.device
        self.is_initialized = False
        
        # Config
        self.confidence = settings.detection.yolo_confidence
        self.iou_threshold = settings.detection.yolo_iou_threshold
        self.target_classes = settings.detection.target_classes
        self.tracker_type = settings.detection.tracker_type
        self.half = settings.detection.half_precision
    
    async def initialize(self) -> None:
        """Charge le modèle YOLO."""
        if self.is_initialized:
            return
        
        logger.info(f"Initializing DetectionService on {self.device}")
        
        try:
            from ultralytics import YOLO
            
            model_path = settings.detection.yolo_model
            
            # Télécharger si nécessaire
            if not Path(model_path).exists():
                logger.info(f"Downloading {model_path}...")
            
            self.model = YOLO(model_path)
            self.model.to(self.device)
            
            # Warmup
            logger.info("Warming up model...")
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            _ = self.model.track(dummy, persist=True, verbose=False)
            
            self.is_initialized = True
            logger.info(f"DetectionService ready - model: {model_path}, device: {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to initialize DetectionService: {e}")
            raise
    
    def process_frame(
        self,
        frame: np.ndarray,
        frame_number: int,
        persist_tracks: bool = True
    ) -> FrameResult:
        """
        Traite un frame et retourne les détections avec tracking.
        
        Args:
            frame: Image BGR (OpenCV format)
            frame_number: Numéro du frame
            persist_tracks: Maintenir les tracks entre frames
            
        Returns:
            FrameResult avec toutes les détections
        """
        if not self.is_initialized:
            raise RuntimeError("Service not initialized. Call initialize() first.")
        
        timestamp = datetime.utcnow()
        
        # Mesure du temps
        start = torch.cuda.Event(enable_timing=True) if "cuda" in self.device else None
        end = torch.cuda.Event(enable_timing=True) if "cuda" in self.device else None
        
        if start:
            start.record()
        
        # Exécuter tracking
        results = self.model.track(
            frame,
            persist=persist_tracks,
            conf=self.confidence,
            iou=self.iou_threshold,
            classes=self.target_classes,
            tracker=f"{self.tracker_type}.yaml",
            verbose=False,
            half=self.half
        )
        
        if end:
            end.record()
            torch.cuda.synchronize()
            inference_time = start.elapsed_time(end)
        else:
            inference_time = 0.0
        
        # Parser les résultats
        detections = self._parse_results(results, frame.shape)
        
        return FrameResult(
            frame_number=frame_number,
            timestamp=timestamp,
            detections=detections,
            inference_time_ms=inference_time
        )
    
    def _parse_results(self, results, frame_shape: tuple) -> list[Detection]:
        """Parse les résultats YOLO en objets Detection."""
        detections = []
        height, width = frame_shape[:2]
        
        if not results or len(results) == 0:
            return detections
        
        result = results[0]
        
        if result.boxes is None:
            return detections
        
        boxes = result.boxes
        
        for i in range(len(boxes)):
            # Track ID (-1 si pas de tracking)
            track_id = int(boxes.id[i]) if boxes.id is not None else -1
            
            # Classe et confiance
            class_id = int(boxes.cls[i])
            confidence = float(boxes.conf[i])
            
            # Bounding box normalisée
            xyxy = boxes.xyxy[i].cpu().numpy()
            bbox = [
                float(xyxy[0]) / width,
                float(xyxy[1]) / height,
                float(xyxy[2]) / width,
                float(xyxy[3]) / height
            ]
            
            # Nom de classe
            class_name = self.COCO_CLASSES.get(class_id, f"class_{class_id}")
            
            detections.append(Detection(
                track_id=track_id,
                class_id=class_id,
                class_name=class_name,
                confidence=confidence,
                bbox=bbox
            ))
        
        return detections
    
    def reset_tracks(self) -> None:
        """Réinitialise le tracker (entre vidéos)."""
        if self.model:
            self.model.predictor = None
        logger.debug("Tracks reset")
    
    async def shutdown(self) -> None:
        """Libère les ressources."""
        if self.model:
            del self.model
        
        if "cuda" in self.device:
            torch.cuda.empty_cache()
        
        self.is_initialized = False
        logger.info("DetectionService shut down")
    
    @property
    def stats(self) -> dict:
        """Statistiques du service."""
        gpu_memory = 0
        if "cuda" in self.device and torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**2  # MB
        
        return {
            "initialized": self.is_initialized,
            "device": self.device,
            "model": settings.detection.yolo_model,
            "tracker": self.tracker_type,
            "gpu_memory_mb": round(gpu_memory, 2),
            "target_classes": [self.COCO_CLASSES.get(c, c) for c in self.target_classes]
        }


# === SINGLETON ===

_detection_service: Optional[DetectionService] = None


def get_detection_service() -> DetectionService:
    """Retourne le singleton du service de détection."""
    global _detection_service
    if _detection_service is None:
        _detection_service = DetectionService()
    return _detection_service
