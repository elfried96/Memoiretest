"""
Analyseur de frames pour le système de surveillance headless.
"""

import time
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from ..types import DetectedObject
from ..vlm.dynamic_model import DynamicVisionLanguageModel
from ..orchestrator.vlm_orchestrator import ModernVLMOrchestrator, OrchestrationConfig
from ...detection.yolo_detector import YOLODetector
from ...detection.tracking.byte_tracker import BYTETracker
from .result_models import SurveillanceResult, AlertLevel

logger = logging.getLogger(__name__)


class FrameAnalyzer:
    """Analyseur complet de frames avec détection, tracking et VLM."""

    def __init__(
        self,
        vlm_model: str = "kimi-vl-a3b-thinking",
        orchestration_config: Optional[OrchestrationConfig] = None,
        yolo_confidence: float = 0.5
    ):
        """
        Initialise l'analyseur de frames.
        
        Args:
            vlm_model: Modèle VLM à utiliser
            orchestration_config: Configuration de l'orchestrateur
            yolo_confidence: Seuil de confiance YOLO
        """
        self.vlm_model_name = vlm_model
        
        # Initialisation des composants
        self._initialize_components(orchestration_config, yolo_confidence)
        
        # Compteurs de performance
        self.total_analysis_time = 0.0
        self.analysis_count = 0

    def _initialize_components(
        self, 
        orchestration_config: Optional[OrchestrationConfig],
        yolo_confidence: float
    ) -> None:
        """Initialise les composants d'analyse."""
        try:
            # 1. Détecteur YOLO
            self.yolo_detector = YOLODetector(
                model_path="yolo11n.pt",
                confidence_threshold=yolo_confidence
            )
            logger.info("✅ YOLO détecteur initialisé")

            # 2. Tracker
            self.tracker = BYTETracker(
                frame_rate=30,
                track_thresh=0.5,
                track_buffer=30,
                match_thresh=0.8
            )
            logger.info("✅ BYTETracker initialisé")

            # 3. VLM dynamique
            self.vlm = DynamicVisionLanguageModel(
                preferred_model=self.vlm_model_name,
                fallback_model="qwen2-vl-7b-instruct"
            )
            logger.info(f"✅ VLM initialisé: {self.vlm_model_name}")

            # 4. Orchestrateur VLM
            config = orchestration_config or OrchestrationConfig()
            self.vlm_orchestrator = ModernVLMOrchestrator(
                vlm_model=self.vlm,
                config=config
            )
            logger.info("✅ Orchestrateur VLM initialisé")

        except Exception as e:
            logger.error(f"❌ Erreur initialisation composants: {e}")
            raise

    async def analyze_frame(
        self,
        frame: np.ndarray,
        frame_id: int,
        timestamp: float,
        vlm_analysis_mode: str = "smart"
    ) -> SurveillanceResult:
        """
        Analyse complète d'une frame.
        
        Args:
            frame: Frame à analyser
            frame_id: ID de la frame
            timestamp: Timestamp de la frame
            vlm_analysis_mode: Mode d'analyse VLM (continuous, periodic, smart)
            
        Returns:
            Résultat d'analyse complet
        """
        start_time = time.time()
        
        try:
            # 1. Détection YOLO
            detections = await self._detect_objects(frame)
            
            # 2. Tracking
            tracked_objects = self._track_objects(detections)
            
            # 3. Analyse de base
            persons_count = self._count_persons(tracked_objects)
            alert_level = self._determine_alert_level(tracked_objects, persons_count)
            
            # 4. Analyse VLM (conditionnelle)
            vlm_analysis = None
            if self._should_run_vlm_analysis(vlm_analysis_mode, alert_level, frame_id):
                vlm_analysis = await self._analyze_with_vlm(frame, tracked_objects)
            
            # 5. Actions recommandées
            actions = self._determine_actions(alert_level, vlm_analysis)
            
            # 6. Création du résultat
            processing_time = time.time() - start_time
            self._update_performance_stats(processing_time)
            
            result = SurveillanceResult(
                frame_id=frame_id,
                timestamp=timestamp,
                detections_count=len(tracked_objects),
                persons_detected=persons_count,
                alert_level=alert_level,
                vlm_analysis=vlm_analysis,
                actions_taken=actions,
                processing_time=processing_time
            )
            
            logger.debug(
                f"📊 Frame {frame_id}: {len(tracked_objects)} détections, "
                f"{persons_count} personnes, niveau {alert_level.value}, "
                f"{processing_time:.2f}s"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Erreur analyse frame {frame_id}: {e}")
            raise

    async def _detect_objects(self, frame: np.ndarray) -> List[DetectedObject]:
        """Détection d'objets avec YOLO."""
        try:
            detections = self.yolo_detector.detect(frame)
            logger.debug(f"🔍 YOLO: {len(detections)} détections")
            return detections
        except Exception as e:
            logger.warning(f"⚠️ Échec détection YOLO: {e}")
            return []

    def _track_objects(self, detections: List[DetectedObject]) -> List[DetectedObject]:
        """Tracking des objets détectés."""
        try:
            # Conversion pour BYTETracker
            bbox_array = np.array([
                [det.bbox.x, det.bbox.y, det.bbox.x + det.bbox.width, 
                 det.bbox.y + det.bbox.height, det.confidence]
                for det in detections
            ]) if detections else np.empty((0, 5))
            
            # Tracking
            tracked = self.tracker.update(bbox_array)
            
            # Reconversion en DetectedObject
            tracked_objects = []
            for track in tracked:
                x1, y1, x2, y2, track_id = track[:5]
                
                # Trouver la détection correspondante
                original_det = None
                for det in detections:
                    det_x2 = det.bbox.x + det.bbox.width
                    det_y2 = det.bbox.y + det.bbox.height
                    if abs(det.bbox.x - x1) < 5 and abs(det.bbox.y - y1) < 5:
                        original_det = det
                        break
                
                if original_det:
                    tracked_obj = DetectedObject(
                        label=original_det.label,
                        confidence=original_det.confidence,
                        bbox=original_det.bbox,
                        track_id=int(track_id)
                    )
                    tracked_objects.append(tracked_obj)
            
            logger.debug(f"🎯 Tracking: {len(tracked_objects)} objets suivis")
            return tracked_objects
            
        except Exception as e:
            logger.warning(f"⚠️ Échec tracking: {e}")
            return detections

    def _count_persons(self, objects: List[DetectedObject]) -> int:
        """Compte le nombre de personnes détectées."""
        return sum(1 for obj in objects if obj.label.lower() == 'person')

    def _determine_alert_level(
        self, 
        objects: List[DetectedObject], 
        persons_count: int
    ) -> AlertLevel:
        """Détermine le niveau d'alerte basé sur les détections."""
        if persons_count == 0:
            return AlertLevel.NORMAL
        elif persons_count == 1:
            return AlertLevel.ATTENTION
        elif persons_count <= 3:
            return AlertLevel.ALERTE
        else:
            return AlertLevel.CRITIQUE

    def _should_run_vlm_analysis(
        self, 
        mode: str, 
        alert_level: AlertLevel, 
        frame_id: int
    ) -> bool:
        """Détermine si une analyse VLM doit être effectuée."""
        if mode == "continuous":
            return True
        elif mode == "periodic":
            return frame_id % 10 == 0  # Toutes les 10 frames
        elif mode == "smart":
            return alert_level in [AlertLevel.ALERTE, AlertLevel.CRITIQUE]
        else:
            return False

    async def _analyze_with_vlm(
        self, 
        frame: np.ndarray, 
        objects: List[DetectedObject]
    ) -> Optional[Dict[str, Any]]:
        """Analyse VLM de la frame."""
        try:
            # Préparation du contexte
            context = {
                "detections": [
                    {
                        "label": obj.label,
                        "confidence": obj.confidence,
                        "track_id": getattr(obj, 'track_id', None)
                    }
                    for obj in objects
                ],
                "timestamp": time.time()
            }
            
            # Analyse orchestrée
            result = await self.vlm_orchestrator.analyze_frame(frame, context)
            
            logger.debug("🧠 Analyse VLM complétée")
            return result
            
        except Exception as e:
            logger.warning(f"⚠️ Échec analyse VLM: {e}")
            return None

    def _determine_actions(
        self, 
        alert_level: AlertLevel, 
        vlm_analysis: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Détermine les actions à prendre basées sur l'analyse."""
        actions = []
        
        if alert_level == AlertLevel.ATTENTION:
            actions.append("Surveillance accrue")
        elif alert_level == AlertLevel.ALERTE:
            actions.append("Notification sécurité")
            actions.append("Enregistrement activé")
        elif alert_level == AlertLevel.CRITIQUE:
            actions.append("Alerte immédiate")
            actions.append("Enregistrement prioritaire")
            actions.append("Notification urgente")
        
        # Actions basées sur l'analyse VLM
        if vlm_analysis and "recommended_actions" in vlm_analysis:
            actions.extend(vlm_analysis["recommended_actions"])
        
        return actions

    def _update_performance_stats(self, processing_time: float) -> None:
        """Met à jour les statistiques de performance."""
        self.total_analysis_time += processing_time
        self.analysis_count += 1

    def get_performance_stats(self) -> Dict[str, float]:
        """Retourne les statistiques de performance."""
        if self.analysis_count == 0:
            return {"average_processing_time": 0.0, "total_time": 0.0}
        
        return {
            "average_processing_time": self.total_analysis_time / self.analysis_count,
            "total_time": self.total_analysis_time,
            "analysis_count": self.analysis_count
        }

    async def cleanup(self) -> None:
        """Nettoyage des ressources."""
        try:
            if hasattr(self.vlm, 'cleanup'):
                await self.vlm.cleanup()
            logger.info("🧹 Analyseur nettoyé")
        except Exception as e:
            logger.warning(f"⚠️ Erreur nettoyage analyseur: {e}")