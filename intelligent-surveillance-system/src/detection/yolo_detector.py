"""
Détecteur YOLO pour la surveillance intelligente.
"""

import torch
import numpy as np
from typing import List, Tuple, Optional
from ultralytics import YOLO
import cv2
from loguru import logger


class YOLODetector:
    """Détecteur YOLO optimisé pour la surveillance."""
    
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        device: str = "auto",
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45
    ):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        
        # Configuration du device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Chargement du modèle
        try:
            self.model = YOLO(model_path)
            self.model.to(self.device)
            logger.info(f"✅ YOLO chargé: {model_path} sur {self.device}")
        except Exception as e:
            logger.error(f"❌ Erreur chargement YOLO: {e}")
            raise
        
        # Classes COCO importantes pour la surveillance
        self.surveillance_classes = {
            0: "person",
            1: "bicycle", 
            2: "car",
            3: "motorcycle",
            24: "handbag",
            25: "umbrella",
            26: "backpack",
            27: "tie",
            28: "suitcase"
        }
    
    def detect(
        self, 
        frame: np.ndarray,
        classes: Optional[List[int]] = None
    ) -> List:
        """
        Effectue la détection sur un frame.
        
        Args:
            frame: Frame OpenCV (BGR)
            classes: Classes spécifiques à détecter (None = toutes)
        
        Returns:
            Liste des résultats YOLO
        """
        try:
            # Paramètres de détection
            detect_params = {
                "conf": self.confidence_threshold,
                "iou": self.iou_threshold,
                "device": self.device,
                "verbose": False
            }
            
            # Classes spécifiques si demandées
            if classes is not None:
                detect_params["classes"] = classes
            
            # Détection
            results = self.model(frame, **detect_params)
            
            return results
            
        except Exception as e:
            logger.error(f"Erreur détection YOLO: {e}")
            return []
    
    def detect_persons_only(self, frame: np.ndarray) -> List:
        """Détection des personnes uniquement."""
        return self.detect(frame, classes=[0])  # Classe 0 = person
    
    def detect_vehicles(self, frame: np.ndarray) -> List:
        """Détection des véhicules."""
        vehicle_classes = [1, 2, 3, 5, 7]  # bicycle, car, motorcycle, bus, truck
        return self.detect(frame, classes=vehicle_classes)
    
    def get_detection_summary(self, results: List) -> dict:
        """Résumé des détections."""
        summary = {
            "total_detections": 0,
            "persons": 0,
            "objects": 0,
            "vehicles": 0,
            "confidence_avg": 0.0
        }
        
        if not results:
            return summary
        
        confidences = []
        
        for result in results:
            if hasattr(result, 'boxes') and result.boxes is not None:
                for box in result.boxes:
                    cls = int(box.cls[0].cpu().numpy())
                    conf = float(box.conf[0].cpu().numpy())
                    
                    summary["total_detections"] += 1
                    confidences.append(conf)
                    
                    if cls == 0:  # person
                        summary["persons"] += 1
                    elif cls in [1, 2, 3, 5, 7]:  # vehicles
                        summary["vehicles"] += 1
                    else:
                        summary["objects"] += 1
        
        if confidences:
            summary["confidence_avg"] = np.mean(confidences)
        
        return summary
    
    def draw_detections(self, frame: np.ndarray, results: List) -> np.ndarray:
        """Dessine les détections sur le frame."""
        if not results:
            return frame
        
        annotated_frame = frame.copy()
        
        for result in results:
            if hasattr(result, 'boxes') and result.boxes is not None:
                # Utilisation de l'annotation automatique d'ultralytics
                annotated_frame = result.plot()
        
        return annotated_frame
    
    def update_thresholds(self, confidence: float, iou: float):
        """Mise à jour des seuils de détection."""
        self.confidence_threshold = confidence
        self.iou_threshold = iou
        logger.info(f"Seuils mis à jour - Conf: {confidence}, IoU: {iou}")
    
    def get_model_info(self) -> dict:
        """Informations sur le modèle."""
        return {
            "model_path": self.model_path,
            "device": self.device,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
            "model_type": "YOLOv8"
        }