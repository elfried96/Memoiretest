"""Détecteur d'objets basé sur YOLO v8 optimisé pour la surveillance."""

import time
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results
from loguru import logger

from ...core.types import DetectedObject, BoundingBox, Frame
from ...utils.exceptions import DetectionError
from ...utils.performance import measure_time


class YOLODetector:
    """
    Détecteur d'objets YOLO v8 optimisé pour la surveillance en grande distribution.
    
    Features:
    - Support multi-classes spécialisées
    - Filtrage par confiance adaptive
    - Optimisation GPU/CPU
    - Mise en cache des résultats
    """
    
    # Classes COCO utiles pour la surveillance
    SURVEILLANCE_CLASSES = {
        0: "person",
        24: "handbag",
        26: "suitcase",
        27: "sports ball",
        31: "backpack",
        39: "bottle",
        41: "cup",
        64: "mouse",
        67: "cell phone",
        73: "book",
        76: "scissors",
        77: "teddy bear",
        84: "hair drier"
    }
    
    # Seuils de confiance par classe
    CLASS_CONFIDENCE_THRESHOLDS = {
        "person": 0.3,  # Seuil bas pour détecter toutes les personnes
        "handbag": 0.4,
        "backpack": 0.4,
        "bottle": 0.5,
        "cell phone": 0.6,
        "default": 0.25
    }
    
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        device: str = "auto",
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_detections: int = 100,
        enable_tracking: bool = True
    ):
        self.model_path = model_path
        self.device = self._setup_device(device)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        self.enable_tracking = enable_tracking
        
        # État du modèle
        self.model: Optional[YOLO] = None
        self.is_loaded = False
        
        # Cache et optimisations
        self._detection_cache = {}
        self._cache_ttl = 0.1  # Cache pendant 100ms
        
        # Statistiques
        self.stats = {
            "total_detections": 0,
            "avg_inference_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        logger.info(f"Initialisation YOLODetector: {model_path} sur {device}")
    
    def _setup_device(self, device: str) -> str:
        """Configuration automatique du device."""
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        return device
    
    def load_model(self) -> None:
        """Chargement du modèle YOLO."""
        try:
            logger.info(f"Chargement du modèle YOLO: {self.model_path}")
            
            # Chargement du modèle
            self.model = YOLO(self.model_path)
            
            # Configuration du device
            if self.device != "cpu":
                self.model.to(self.device)
            
            # Test de fonctionnement
            dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
            _ = self.model(dummy_image, verbose=False)
            
            self.is_loaded = True
            logger.success(f"Modèle YOLO chargé avec succès sur {self.device}")
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement YOLO: {e}")
            raise DetectionError(f"Impossible de charger YOLO: {e}")
    
    @measure_time
    def detect(
        self, 
        frame: Frame,
        custom_confidence: Optional[float] = None,
        filter_classes: Optional[List[str]] = None
    ) -> List[DetectedObject]:
        """
        Détection d'objets dans un frame.
        
        Args:
            frame: Frame à analyser
            custom_confidence: Seuil de confiance personnalisé
            filter_classes: Classes à filtrer (None = toutes les classes surveillées)
        
        Returns:
            Liste des objets détectés
        """
        if not self.is_loaded:
            self.load_model()
        
        try:
            # Vérification cache
            cache_key = self._get_cache_key(frame)
            if cache_key in self._detection_cache:
                cached_result, timestamp = self._detection_cache[cache_key]
                if time.time() - timestamp < self._cache_ttl:
                    self.stats["cache_hits"] += 1
                    return cached_result
            
            self.stats["cache_misses"] += 1
            
            # Préparation de l'image
            image = self._prepare_image(frame.image)
            
            # Inférence YOLO
            start_time = time.time()
            
            results: List[Results] = self.model(
                image,
                conf=custom_confidence or self.confidence_threshold,
                iou=self.iou_threshold,
                max_det=self.max_detections,
                verbose=False,
                device=self.device
            )
            
            inference_time = time.time() - start_time
            self._update_stats(inference_time)
            
            # Traitement des résultats
            detections = self._process_results(
                results[0], 
                filter_classes=filter_classes,
                frame_shape=frame.shape
            )
            
            # Mise en cache
            self._detection_cache[cache_key] = (detections, time.time())
            self._cleanup_cache()
            
            logger.debug(
                f"YOLO détection: {len(detections)} objets en {inference_time:.3f}s"
            )
            
            return detections
            
        except Exception as e:
            logger.error(f"Erreur détection YOLO: {e}")
            raise DetectionError(f"Erreur lors de la détection: {e}")
    
    def detect_batch(
        self, 
        frames: List[Frame],
        custom_confidence: Optional[float] = None
    ) -> List[List[DetectedObject]]:
        """Détection en lot pour optimiser les performances."""
        
        if not self.is_loaded:
            self.load_model()
        
        try:
            # Préparation des images
            images = [self._prepare_image(frame.image) for frame in frames]
            
            # Inférence batch
            start_time = time.time()
            
            results = self.model(
                images,
                conf=custom_confidence or self.confidence_threshold,
                iou=self.iou_threshold,
                max_det=self.max_detections,
                verbose=False,
                device=self.device
            )
            
            inference_time = time.time() - start_time
            self._update_stats(inference_time / len(frames))
            
            # Traitement des résultats
            all_detections = []
            for result, frame in zip(results, frames):
                detections = self._process_results(result, frame_shape=frame.shape)
                all_detections.append(detections)
            
            logger.debug(
                f"YOLO batch: {len(frames)} frames en {inference_time:.3f}s"
            )
            
            return all_detections
            
        except Exception as e:
            logger.error(f"Erreur détection batch YOLO: {e}")
            raise DetectionError(f"Erreur lors de la détection batch: {e}")
    
    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        """Préparation de l'image pour YOLO."""
        
        # Conversion BGR -> RGB si nécessaire
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Vérification si l'image est en BGR (OpenCV)
            if np.mean(image[:, :, 2]) < np.mean(image[:, :, 0]):
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image
    
    def _process_results(
        self, 
        result: Results,
        filter_classes: Optional[List[str]] = None,
        frame_shape: Optional[Tuple[int, int, int]] = None
    ) -> List[DetectedObject]:
        """Traitement des résultats YOLO en objets détectés."""
        
        detections = []
        
        if result.boxes is None:
            return detections
        
        # Extraction des données
        boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        
        for i, (box, confidence, class_id) in enumerate(zip(boxes, confidences, class_ids)):
            # Vérification de la classe
            class_name = result.names.get(class_id, f"unknown_{class_id}")
            
            # Filtrage par classes surveillées
            if class_id not in self.SURVEILLANCE_CLASSES:
                continue
            
            # Filtrage par classes demandées
            if filter_classes and class_name not in filter_classes:
                continue
            
            # Vérification du seuil de confiance spécifique à la classe
            min_confidence = self.CLASS_CONFIDENCE_THRESHOLDS.get(
                class_name, 
                self.CLASS_CONFIDENCE_THRESHOLDS["default"]
            )
            
            if confidence < min_confidence:
                continue
            
            # Création de la boîte englobante
            x1, y1, x2, y2 = box.astype(int)
            width = x2 - x1
            height = y2 - y1
            
            # Validation des dimensions
            if width < 10 or height < 10:
                continue
            
            bbox = BoundingBox(
                x=x1,
                y=y1,
                width=width,
                height=height,
                confidence=float(confidence)
            )
            
            # Création de l'objet détecté
            detected_obj = DetectedObject(
                class_id=int(class_id),
                class_name=class_name,
                bbox=bbox,
                confidence=float(confidence)
            )
            
            detections.append(detected_obj)
        
        # Tri par confiance décroissante
        detections.sort(key=lambda x: x.confidence, reverse=True)
        
        self.stats["total_detections"] += len(detections)
        
        return detections
    
    def _get_cache_key(self, frame: Frame) -> str:
        """Génération d'une clé de cache pour un frame."""
        # Hash simple basé sur l'ID du frame et timestamp
        return f"{frame.stream_id}_{frame.frame_id}_{frame.timestamp.timestamp():.3f}"
    
    def _cleanup_cache(self, max_size: int = 100) -> None:
        """Nettoyage du cache de détection."""
        if len(self._detection_cache) > max_size:
            # Suppression des entrées les plus anciennes
            sorted_keys = sorted(
                self._detection_cache.keys(),
                key=lambda k: self._detection_cache[k][1]
            )
            
            for key in sorted_keys[:len(sorted_keys) - max_size // 2]:
                del self._detection_cache[key]
    
    def _update_stats(self, inference_time: float) -> None:
        """Mise à jour des statistiques de performance."""
        if self.stats["avg_inference_time"] == 0:
            self.stats["avg_inference_time"] = inference_time
        else:
            # Moyenne mobile exponentielle
            alpha = 0.1
            self.stats["avg_inference_time"] = (
                alpha * inference_time + 
                (1 - alpha) * self.stats["avg_inference_time"]
            )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Informations sur le modèle YOLO."""
        info = {
            "model_path": self.model_path,
            "device": self.device,
            "is_loaded": self.is_loaded,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
            "supported_classes": list(self.SURVEILLANCE_CLASSES.values()),
            "stats": self.stats.copy()
        }
        
        if self.model and self.is_loaded:
            info.update({
                "model_name": self.model.model_name if hasattr(self.model, 'model_name') else "YOLO",
                "input_size": getattr(self.model.model, 'imgsz', 640) if self.model.model else 640
            })
        
        return info
    
    def update_thresholds(
        self, 
        confidence: Optional[float] = None,
        iou: Optional[float] = None
    ) -> None:
        """Mise à jour des seuils de détection."""
        if confidence is not None:
            self.confidence_threshold = confidence
            logger.info(f"Seuil de confiance mis à jour: {confidence}")
        
        if iou is not None:
            self.iou_threshold = iou
            logger.info(f"Seuil IoU mis à jour: {iou}")
    
    def reset_stats(self) -> None:
        """Réinitialisation des statistiques."""
        self.stats = {
            "total_detections": 0,
            "avg_inference_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        logger.info("Statistiques YOLO réinitialisées")
    
    def cleanup(self) -> None:
        """Nettoyage des ressources."""
        if self.model is not None:
            del self.model
            self.model = None
            
        self._detection_cache.clear()
        
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        self.is_loaded = False
        logger.info("Ressources YOLO libérées")