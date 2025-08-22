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
        model_path: str = "yolov11n.pt",
        device: str = "auto",
        confidence_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        environment: str = "supermarket"
    ):
        self.model_path = model_path
        self.environment = environment
        
        # Optimisation des seuils selon l'environnement
        self._setup_optimized_thresholds(confidence_threshold, iou_threshold)
        
        # Configuration du device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Chargement du modèle
        try:
            self.model = YOLO(model_path)
            self.model.to(self.device)
            logger.info(f"✅ YOLO chargé: {model_path} sur {self.device} - Mode: {environment}")
        except Exception as e:
            logger.error(f"❌ Erreur chargement YOLO: {e}")
            raise
        
        # Configuration des classes selon l'environnement
        self._setup_environment_classes()
    
    def _setup_optimized_thresholds(self, confidence: Optional[float], iou: Optional[float]):
        """Configure les seuils optimisés selon l'environnement."""
        
        if self.environment == "supermarket":
            # Seuils optimisés pour environnement retail
            self.confidence_threshold = confidence if confidence is not None else 0.6  # Plus strict pour réduire faux positifs
            self.iou_threshold = iou if iou is not None else 0.4  # Plus permissif pour objets proches
            
        elif self.environment == "outdoor":
            # Seuils pour surveillance extérieure
            self.confidence_threshold = confidence if confidence is not None else 0.5
            self.iou_threshold = iou if iou is not None else 0.45
            
        else:
            # Configuration par défaut
            self.confidence_threshold = confidence if confidence is not None else 0.5
            self.iou_threshold = iou if iou is not None else 0.45
        
        logger.info(f"Seuils {self.environment}: Conf={self.confidence_threshold}, IoU={self.iou_threshold}")
    
    def _setup_environment_classes(self):
        """Configure les classes de détection selon l'environnement."""
        
        if self.environment == "supermarket":
            # Classes COCO essentielles pour surveillance supermarché
            self.target_classes = {
                0: "person",      # Personnes
                25: "handbag",    # Sacs à main
                26: "backpack",   # Sacs à dos  
                28: "suitcase"    # Valises
            }
            
            # Classes prioritaires = toutes les classes pour supermarché
            self.priority_classes = [0, 25, 26, 28]
            
        elif self.environment == "outdoor":
            # Configuration pour surveillance extérieure
            self.target_classes = {
                0: "person",
                1: "bicycle", 
                2: "car",
                3: "motorcycle",
                5: "bus",
                7: "truck",
                25: "handbag",
                26: "backpack",
                28: "suitcase"
            }
            self.priority_classes = [0, 1, 2, 3, 5, 7]
            
        else:
            # Configuration générale (par défaut)
            self.target_classes = {
                0: "person",
                25: "handbag",
                26: "backpack", 
                28: "suitcase"
            }
            self.priority_classes = [0]
        
        logger.info(f"Configuration {self.environment}: {len(self.target_classes)} classes cibles")
    
    def get_target_classes_ids(self) -> List[int]:
        """Retourne la liste des IDs de classes cibles pour cet environnement."""
        return list(self.target_classes.keys())
    
    def get_priority_classes_ids(self) -> List[int]:
        """Retourne la liste des IDs de classes prioritaires."""
        return self.priority_classes
    
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
    
    def detect_target_classes(self, frame: np.ndarray) -> List:
        """Détection des classes cibles pour l'environnement actuel."""
        return self.detect(frame, classes=self.get_target_classes_ids())
    
    def detect_priority_objects(self, frame: np.ndarray) -> List:
        """Détection des objets prioritaires pour surveillance."""
        return self.detect(frame, classes=self.get_priority_classes_ids())
    
    def detect_surveillance_objects(self, frame: np.ndarray) -> List:
        """Détection des objets de surveillance (personnes + sacs)."""
        if self.environment == "supermarket":
            # Classes essentielles: personnes + sacs
            return self.detect(frame, classes=[0, 25, 26, 28])
        else:
            return self.detect_persons_only(frame)
    
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
            "environment": self.environment,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
            "model_type": "YOLO11",
            "target_classes_count": len(self.target_classes),
            "priority_classes": self.priority_classes,
            "optimized_for_retail": self.environment == "supermarket"
        }


class SimplifiedYOLODetector:
    """Version simplifiée du détecteur YOLO pour surveillance basique."""
    
    def __init__(self):
        self.model = YOLO('yolov11n.pt')
        self.surveillance_classes = [0, 25, 26, 28]  # person, handbag, backpack, suitcase
        self.confidence_threshold = 0.5
        
    def detect_objects(self, frame: np.ndarray):
        """Détection des objets de surveillance."""
        results = self.model(frame, 
                           classes=self.surveillance_classes,
                           conf=self.confidence_threshold)
        return self.process_detections(results)
    
    def process_detections(self, results):
        """Traitement simple des détections."""
        detections = []
        for result in results:
            if hasattr(result, 'boxes') and result.boxes is not None:
                for box in result.boxes:
                    detection = {
                        'class': int(box.cls[0].cpu().numpy()),
                        'confidence': float(box.conf[0].cpu().numpy()),
                        'bbox': box.xyxy[0].cpu().numpy().tolist()
                    }
                    detections.append(detection)
        return detections