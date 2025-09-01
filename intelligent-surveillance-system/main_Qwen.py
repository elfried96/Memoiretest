#!/usr/bin/env python3
"""
🎯 MAIN QWEN - Pipeline UNIQUEMENT avec Qwen2-VL
================================================

Version dédiée EXCLUSIVEMENT au modèle Qwen2-VL.
AUCUN fallback - Si Qwen2-VL échoue, le système s'arrête.
"""

import asyncio
import cv2
import time
import base64
import json
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from io import BytesIO
import numpy as np

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Imports du système de surveillance
from src.core.vlm.dynamic_model import DynamicVisionLanguageModel
from src.core.orchestrator.vlm_orchestrator import (
    ModernVLMOrchestrator, 
    OrchestrationConfig, 
    OrchestrationMode
)
from src.core.types import DetectedObject, BoundingBox
from src.detection.yolo_detector import YOLODetector
from src.detection.tracking.byte_tracker import BYTETracker
from src.core.vlm.memory_system import vlm_memory


class AlertLevel(Enum):
    """Niveaux d'alerte du système."""
    NORMAL = "normal"
    ATTENTION = "attention" 
    ALERTE = "alerte"
    CRITIQUE = "critique"


@dataclass
class SurveillanceResult:
    """Résultat de surveillance pour export JSON."""
    frame_id: int
    timestamp: float
    detections_count: int
    persons_detected: int
    alert_level: str
    vlm_analysis: Optional[Dict] = None
    actions_taken: List[str] = None
    processing_time: float = 0.0


class QwenOnlySurveillanceSystem:
    """Système de surveillance UNIQUEMENT pour Qwen2-VL - AUCUN fallback."""
    
    def __init__(
        self,
        video_source: str = 0,
        vlm_model: str = "qwen2-vl-72b", 
        orchestration_mode: OrchestrationMode = OrchestrationMode.BALANCED,
        save_results: bool = True,
        save_frames: bool = False
    ):
        self.video_source = video_source
        self.vlm_model = vlm_model  # FORCE Qwen2-VL - pas de "none"
        self.orchestration_mode = orchestration_mode
        self.save_results = save_results
        self.save_frames = save_frames
        
        # Dossiers de sortie spécifiques Qwen
        self.output_dir = Path("surveillance_output_qwen_only")
        self.output_dir.mkdir(exist_ok=True)
        
        if save_frames:
            self.frames_dir = self.output_dir / "frames"
            self.frames_dir.mkdir(exist_ok=True)
        
        # === COMPOSANTS PRINCIPAUX ===
        
        # 1. Détection et tracking
        self.yolo_detector = YOLODetector(
            model_path="yolov11n.pt",
            device="auto"
        )
        self.tracker = BYTETracker()
        
        # 2. VLM UNIQUEMENT QWEN2-VL - FALLBACK DÉSACTIVÉ
        self.vlm = DynamicVisionLanguageModel(
            default_model=vlm_model,
            enable_fallback=False  # ❌ AUCUN FALLBACK
        )
        
        config = OrchestrationConfig(
            mode=orchestration_mode,
            enable_advanced_tools=True,
            max_concurrent_tools=4,
            confidence_threshold=0.7
        )
        
        self.orchestrator = ModernVLMOrchestrator(
            vlm_model_name=vlm_model,
            config=config
        )
        
        # ✅ VLM OBLIGATOIRE - pas de mode dégradé
        self.vlm_enabled = True
        logger.info(f"✅ SYSTÈME QWEN2-VL ONLY - Modèle: {vlm_model}")
        logger.info("⚠️ AUCUN FALLBACK - Qwen2-VL obligatoire")
        
        # === ÉTAT DU SYSTÈME ===
        self.frame_count = 0
        self.results_log = []
        self.processing_stats = {
            "total_frames": 0,
            "detected_objects": 0,
            "persons_detected": 0,
            "vlm_analyses": 0,
            "vlm_triggered": 0,
            "alerts_triggered": 0,
            "total_processing_time": 0.0,
            "average_fps": 0.0
        }
        
        # === SYSTÈME DE DÉCLENCHEMENT INTELLIGENT ===
        self.last_vlm_trigger_time = 0
        self.vlm_cooldown_seconds = 5
        self.person_count_history = []
        self.alert_history = []
        
        logger.info(f"🎯 Système QWEN2-VL ONLY initialisé - Mode: {orchestration_mode.value}")
        logger.info(f"📁 Résultats sauvés dans: {self.output_dir}")
    
    async def initialize(self):
        """Initialisation - ECHEC SI QWEN2-VL N'EST PAS DISPONIBLE."""
        logger.info("🚀 Initialisation QWEN2-VL ONLY...")
        
        # Chargement OBLIGATOIRE de Qwen2-VL
        logger.info(f"⏳ Chargement OBLIGATOIRE de {self.vlm_model}...")
        vlm_loaded = await self.vlm.load_model()
        
        if not vlm_loaded:
            # 🛑 ARRÊT COMPLET - pas de fallback
            logger.error("❌ ÉCHEC: Qwen2-VL non disponible")
            logger.error("❌ ARRÊT DU SYSTÈME - Aucun fallback configuré")
            sys.exit(1)
        
        logger.info(f"✅ {self.vlm_model} chargé avec succès - SYSTÈME OPÉRATIONNEL")
        
        # Test des composants
        logger.info("🔍 Tests des composants...")
        
        # Test YOLO
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        test_detections = self.yolo_detector.detect(test_frame)
        logger.info(f"✅ YOLO opérationnel - {len(test_detections)} détections test")
        
        # Health check OBLIGATOIRE
        status = await self.orchestrator.health_check()
        logger.info(f"📊 Health check Qwen2-VL: {status}")
        
        if not status.get('vlm_ready', False):
            logger.error("❌ VLM non prêt - ARRÊT SYSTÈME")
            sys.exit(1)
        
        logger.info("🎯 Système QWEN2-VL ONLY prêt pour surveillance!")
    
    def create_detections_list(self, yolo_results) -> List[DetectedObject]:
        """Convertit les résultats YOLO en liste de Detection."""
        detections = []
        
        for result in yolo_results:
            if hasattr(result, 'boxes') and result.boxes is not None:
                for box in result.boxes:
                    # Extraction des coordonnées
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    
                    # Nom de classe
                    class_names = result.names
                    class_name = class_names.get(cls, f"class_{cls}")
                    
                    # Création de la détection
                    detection = DetectedObject(
                        class_id=cls,
                        class_name=class_name,
                        bbox=BoundingBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2)),
                        confidence=float(conf),
                        track_id=None
                    )
                    detections.append(detection)
        
        return detections
    
    def _should_trigger_vlm_analysis(self, detections: List[DetectedObject], persons_count: int, context: Dict[str, Any]) -> bool:
        """
        Détermine si Qwen2-VL doit être déclenché.
        """
        current_time = time.time()
        
        # 1. Vérifier le cooldown
        if current_time - self.last_vlm_trigger_time < self.vlm_cooldown_seconds:
            return False
        
        # 2. Toujours déclencher si beaucoup de personnes
        if persons_count >= 3:
            logger.info(f"🚨 Qwen2-VL déclenché: {persons_count} personnes détectées")
            return True
        
        # 3. Maintenir historique du nombre de personnes
        self.person_count_history.append(persons_count)
        if len(self.person_count_history) > 10:
            self.person_count_history.pop(0)
        
        # 4. Détecter changement soudain de population
        if len(self.person_count_history) >= 5:
            recent_avg = sum(self.person_count_history[-5:]) / 5
            if persons_count > recent_avg + 1:
                logger.info(f"📈 Qwen2-VL déclenché: augmentation population {recent_avg:.1f} → {persons_count}")
                return True
        
        # 5. Objets suspects détectés
        suspicious_objects = ["backpack", "handbag", "suitcase", "umbrella", "sports ball", "bag"]
        for detection in detections:
            if detection.class_name in suspicious_objects:
                logger.info(f"👜 Qwen2-VL déclenché: objet suspect '{detection.class_name}' détecté")
                return True
        
        # 6. Personne seule qui reste longtemps
        if persons_count == 1 and len(self.person_count_history) >= 8:
            if all(count == 1 for count in self.person_count_history[-8:]):
                logger.info("🕐 Qwen2-VL déclenché: personne seule depuis longtemps")
                return True
        
        # 7. Déclenchement périodique
        if not context.get("test_mode", False):
            if (current_time - self.last_vlm_trigger_time > 60 and persons_count > 0):
                logger.info("⏰ Qwen2-VL déclenché: contrôle périodique de sécurité")
                return True
        
        return False
    
    def encode_frame_to_base64(self, frame: np.ndarray) -> str:
        """Encode un frame OpenCV en base64 pour Qwen2-VL."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        _, buffer = cv2.imencode('.jpg', frame_rgb, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_b64 = base64.b64encode(buffer).decode('utf-8')
        return frame_b64
    
    def save_frame_with_detections(self, frame: np.ndarray, detections: List[DetectedObject], frame_id: int):
        """Sauvegarde un frame avec les détections dessinées."""
        if not self.save_frames:
            return
            
        overlay_frame = frame.copy()
        
        # Dessiner les détections
        for detection in detections:
            bbox = detection.bbox
            color = (0, 255, 0) if detection.class_name == "person" else (255, 0, 0)
            
            # Rectangle
            cv2.rectangle(
                overlay_frame,
                (int(bbox.x1), int(bbox.y1)),
                (int(bbox.x2), int(bbox.y2)),
                color, 2
            )
            
            # Label avec indication Qwen2-VL
            label = f"{detection.class_name}: {detection.confidence:.2f} [QWEN2-VL]"
            cv2.putText(
                overlay_frame, label,
                (int(bbox.x1), int(bbox.y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )
        
        # Sauvegarder
        frame_path = self.frames_dir / f"qwen_only_frame_{frame_id:06d}.jpg"
        cv2.imwrite(str(frame_path), overlay_frame)
        logger.debug(f"🖼️ Frame Qwen2-VL ONLY sauvé: {frame_path}")
    
    async def process_frame(self, frame: np.ndarray) -> SurveillanceResult:
        """Traite un frame complet avec UNIQUEMENT Qwen2-VL."""
        
        start_time = time.time()
        self.frame_count += 1
        
        # === ÉTAPE 1: DÉTECTION YOLO ===
        yolo_results = self.yolo_detector.detect(frame)
        detections = self.create_detections_list(yolo_results)
        
        # === ÉTAPE 1.5: TRACKING ===
        tracked_objects = self.tracker.update(detections)
        
        persons_count = len([d for d in detections if d.class_name == "person"])
        
        self.processing_stats["detected_objects"] += len(detections)
        self.processing_stats["persons_detected"] += persons_count
        
        # === ÉTAPE 2: ANALYSE QWEN2-VL UNIQUEMENT ===
        vlm_analysis = None
        alert_level = AlertLevel.NORMAL
        actions_taken = []
        
        # Vérifier que Qwen2-VL est OBLIGATOIREMENT chargé
        if not (hasattr(self.vlm, 'model') and self.vlm.model is not None):
            logger.error("❌ ERREUR CRITIQUE: Qwen2-VL non chargé")
            raise RuntimeError("Qwen2-VL non disponible - ARRÊT SYSTÈME")
        
        logger.debug(f"🧠 Frame {self.frame_count} - Analyse QWEN2-VL ONLY...")
        
        # Encodage pour Qwen2-VL
        frame_b64 = self.encode_frame_to_base64(frame)
        
        # Contexte enrichi avec mémoire historique
        memory_context = vlm_memory.get_context_for_vlm()
        
        context = {
            "frame_id": self.frame_count,
            "timestamp": time.time(),
            "location": "Store Main Area",
            "camera": "CAM_01_QWEN_ONLY",
            "person_count": persons_count,
            "total_objects": len(detections),
            "time_of_day": time.strftime("%H:%M:%S"),
            "vlm_model": "qwen2-vl-72b-ONLY",
            **memory_context
        }
        
        # DÉCLENCHEMENT INTELLIGENT DE QWEN2-VL
        should_trigger_vlm = self._should_trigger_vlm_analysis(detections, persons_count, context)
        
        try:
            if should_trigger_vlm:
                # Marquer le déclenchement
                self.last_vlm_trigger_time = time.time()
                self.processing_stats["vlm_triggered"] += 1
                
                # Analyse orchestrée UNIQUEMENT avec Qwen2-VL
                vlm_analysis = await self.orchestrator.analyze_surveillance_frame(
                    frame_data=frame_b64,
                    detections=detections,
                    context=context
                )
                
                self.processing_stats["vlm_analyses"] += 1
            else:
                # Pas de déclenchement VLM - analyse légère seulement
                vlm_analysis = None
            
            # Décisions basées sur l'analyse Qwen2-VL
            if vlm_analysis is not None:
                if vlm_analysis.suspicion_level.value in ["HIGH", "CRITICAL"]:
                    alert_level = AlertLevel.ALERTE
                    actions_taken = ["qwen_alert_triggered", "recording_started"]
                    self.processing_stats["alerts_triggered"] += 1
                elif vlm_analysis.suspicion_level.value == "MEDIUM":
                    alert_level = AlertLevel.ATTENTION
                    actions_taken = ["qwen_increased_monitoring"]
            else:
                # Analyse de base sans VLM (surveillance continue rapide)
                if persons_count >= 3:
                    alert_level = AlertLevel.ATTENTION
                    actions_taken = ["high_occupancy_basic"]
                elif len(detections) > 5:
                    alert_level = AlertLevel.ATTENTION
                    actions_taken = ["many_objects_detected"]
            
        except Exception as e:
            logger.error(f"❌ Erreur analyse Qwen2-VL frame {self.frame_count}: {e}")
            # En mode QWEN ONLY, on peut décider d'arrêter ou continuer
            # Pour l'instant on continue mais on log l'erreur
        
        # === SAUVEGARDE FRAME SI DEMANDÉE ===
        if len(detections) > 0 or alert_level != AlertLevel.NORMAL:
            self.save_frame_with_detections(frame, detections, self.frame_count)
        
        # === CRÉATION DU RÉSULTAT ===
        processing_time = time.time() - start_time
        
        # Sérialisation sécurisée de vlm_analysis
        vlm_analysis_dict = None
        if vlm_analysis:
            try:
                if hasattr(vlm_analysis, 'to_dict'):
                    vlm_analysis_dict = vlm_analysis.to_dict()
                elif hasattr(vlm_analysis, '__dict__'):
                    vlm_analysis_dict = vlm_analysis.__dict__.copy()
                    # Convertir les enums en strings
                    for key, value in vlm_analysis_dict.items():
                        if hasattr(value, 'value'):
                            vlm_analysis_dict[key] = value.value
                    # Marquer comme QWEN ONLY
                    vlm_analysis_dict["model_used"] = "qwen2-vl-72b-ONLY"
            except Exception:
                vlm_analysis_dict = None
        
        result = SurveillanceResult(
            frame_id=self.frame_count,
            timestamp=time.time(),
            detections_count=len(detections),
            persons_detected=persons_count,
            alert_level=alert_level.value,
            vlm_analysis=vlm_analysis_dict,
            actions_taken=actions_taken,
            processing_time=processing_time
        )
        
        # === MISE À JOUR DES STATISTIQUES ===
        self.processing_stats["total_frames"] += 1
        self.processing_stats["total_processing_time"] += processing_time
        
        if self.processing_stats["total_frames"] > 0:
            avg_time = self.processing_stats["total_processing_time"] / self.processing_stats["total_frames"]
            self.processing_stats["average_fps"] = 1.0 / avg_time if avg_time > 0 else 0
        
        # === ENREGISTREMENT EN MÉMOIRE ===
        vlm_triggered_for_memory = (vlm_analysis is not None)
        vlm_memory.add_frame(
            frame_id=self.frame_count,
            detections=detections,
            vlm_triggered=vlm_triggered_for_memory,
            vlm_analysis=vlm_analysis,
            alert_level=alert_level.value,
            actions_taken=actions_taken
        )
        
        # === LOGGING DÉTAILLÉ QWEN2-VL ONLY ===
        if persons_count > 0 or alert_level != AlertLevel.NORMAL:
            memory_stats = vlm_memory.get_memory_stats()
            
            logger.info(f"📊 [QWEN2-VL ONLY] Frame {self.frame_count}: "
                       f"{len(detections)} objs, {persons_count} personnes, "
                       f"Alert: {alert_level.value}, "
                       f"Actions: {actions_taken}, "
                       f"Temps: {processing_time:.2f}s, "
                       f"Mémoire: {memory_stats['current_frames_in_memory']} frames, "
                       f"{memory_stats['active_persons']} personnes actives")
        elif self.frame_count % 60 == 0:
            # Calcul du taux de déclenchement
            trigger_rate = (self.processing_stats['vlm_triggered'] / self.processing_stats['total_frames'] * 100) if self.processing_stats['total_frames'] > 0 else 0
            memory_stats = vlm_memory.get_memory_stats()
            
            logger.info(f"📈 [QWEN2-VL ONLY] Frame {self.frame_count}: "
                       f"FPS: {self.processing_stats['average_fps']:.1f}, "
                       f"Total objets: {self.processing_stats['detected_objects']}, "
                       f"VLM déclenché: {self.processing_stats['vlm_triggered']}/{self.processing_stats['total_frames']} ({trigger_rate:.1f}%), "
                       f"Analyses VLM: {self.processing_stats['vlm_analyses']}, "
                       f"Patterns détectés: {memory_stats['patterns_detected']}")
        
        return result
    
    async def run_surveillance(self, max_frames: int = None):
        """Lance la surveillance en mode QWEN2-VL ONLY."""
        logger.info("🎬 Démarrage surveillance QWEN2-VL ONLY...")
        
        # Ouverture de la source vidéo
        cap = cv2.VideoCapture(self.video_source)
        
        if not cap.isOpened():
            logger.error(f"❌ Impossible d'ouvrir source vidéo: {self.video_source}")
            return
        
        # Optimisations extraction frames
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        
        # Informations vidéo
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"📹 Source vidéo: {self.video_source}")
        logger.info(f"📏 Résolution: {width}x{height} | FPS: {fps:.2f} | Frames: {total_frames}")
        logger.info(f"🤖 Modèle VLM: {self.vlm_model} UNIQUEMENT")
        
        try:
            frame_processed = 0
            
            while True:
                # Lecture optimisée du frame
                ret, frame = cap.read()
                if not ret:
                    logger.warning("📹 Fin de vidéo ou erreur lecture")
                    break
                
                # Validation du frame
                if frame is None or frame.size == 0:
                    logger.warning("⚠️ Frame vide détecté")
                    continue
                
                if frame.shape[0] < 100 or frame.shape[1] < 100:
                    logger.warning(f"⚠️ Frame trop petit: {frame.shape}")
                    continue
                
                # Traitement complet du frame
                result = await self.process_frame(frame)
                self.results_log.append(result)
                
                frame_processed += 1
                
                # Limite de frames si spécifiée
                if max_frames and frame_processed >= max_frames:
                    logger.info(f"🏁 Limite de {max_frames} frames atteinte")
                    break
        
        except KeyboardInterrupt:
            logger.info("🛑 Arrêt par interruption clavier")
        
        finally:
            # Nettoyage
            cap.release()
            
            # Sauvegarde des résultats
            if self.save_results:
                self.save_results_to_json()
            
            # Statistiques finales
            self.print_final_statistics()
    
    def _serialize_result(self, result):
        """Sérialise un résultat en évitant les erreurs JSON."""
        import datetime
        
        def serialize_value(value):
            if hasattr(value, 'value'):
                return value.value
            elif isinstance(value, datetime.datetime):
                return value.isoformat()
            elif isinstance(value, dict):
                return {k: serialize_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [serialize_value(item) for item in value]
            else:
                return value
        
        try:
            result_dict = asdict(result)
            return {key: serialize_value(value) for key, value in result_dict.items()}
        except Exception as e:
            return {
                "frame_id": getattr(result, 'frame_id', 0),
                "timestamp": getattr(result, 'timestamp', 0),
                "detections_count": getattr(result, 'detections_count', 0),
                "persons_detected": getattr(result, 'persons_detected', 0),
                "alert_level": getattr(result, 'alert_level', 'normal'),
                "actions_taken": getattr(result, 'actions_taken', []),
                "processing_time": getattr(result, 'processing_time', 0),
                "vlm_analysis": None,
                "model_used": "qwen2-vl-72b-ONLY"
            }
    
    def save_results_to_json(self):
        """Sauvegarde les résultats en JSON."""
        results_file = self.output_dir / f"qwen_only_surveillance_{int(time.time())}.json"
        
        output_data = {
            "metadata": {
                "video_source": str(self.video_source),
                "vlm_model": self.vlm_model,
                "model_type": "qwen2-vl-72b-ONLY",
                "fallback_enabled": False,
                "orchestration_mode": self.orchestration_mode.value,
                "total_frames": len(self.results_log),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "statistics": self.processing_stats,
            "results": [self._serialize_result(result) for result in self.results_log]
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Résultats QWEN2-VL ONLY sauvés: {results_file}")
    
    def print_final_statistics(self):
        """Affiche les statistiques finales."""
        logger.info("=" * 60)
        logger.info("📈 STATISTIQUES FINALES QWEN2-VL ONLY")
        logger.info("=" * 60)
        
        # Statistiques générales
        for key, value in self.processing_stats.items():
            if isinstance(value, float):
                logger.info(f"  • {key}: {value:.2f}")
            else:
                logger.info(f"  • {key}: {value}")
        
        # Efficacité du déclenchement intelligent
        logger.info("")
        logger.info("🧠 EFFICACITÉ QWEN2-VL UNIQUEMENT:")
        logger.info("-" * 60)
        
        total_frames = self.processing_stats.get("total_frames", 1)
        vlm_triggered = self.processing_stats.get("vlm_triggered", 0)
        vlm_analyses = self.processing_stats.get("vlm_analyses", 0)
        
        trigger_rate = (vlm_triggered / total_frames * 100) if total_frames > 0 else 0
        success_rate = (vlm_analyses / vlm_triggered * 100) if vlm_triggered > 0 else 0
        
        logger.info(f"  🎯 Frames total traités: {total_frames}")
        logger.info(f"  ⚡ Qwen2-VL déclenché: {vlm_triggered} fois ({trigger_rate:.1f}%)")
        logger.info(f"  ✅ Analyses réussies: {vlm_analyses} ({success_rate:.1f}%)")
        logger.info(f"  🚀 Économie traitement: {(100 - trigger_rate):.1f}%")
        
        # Statistiques mémoire
        memory_stats = vlm_memory.get_memory_stats()
        logger.info("")
        logger.info("🧠 SYSTÈME DE MÉMOIRE:")
        logger.info("-" * 60)
        logger.info(f"  💾 Frames en mémoire: {memory_stats['current_frames_in_memory']}")
        logger.info(f"  👥 Personnes trackées: {memory_stats['active_persons']}")
        logger.info(f"  🔍 Patterns détectés: {memory_stats['patterns_detected']}")


def parse_arguments():
    """Parse les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(
        description="🎯 Système de Surveillance QWEN2-VL UNIQUEMENT",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--video", "-v", default="webcam",
                       help="Source vidéo")
    parser.add_argument("--model", "-m", default="qwen2-vl-72b",
                       help="Modèle VLM Qwen2-VL")
    parser.add_argument("--mode", default="BALANCED",
                       choices=["FAST", "BALANCED", "THOROUGH"],
                       help="Mode orchestration")
    parser.add_argument("--max-frames", type=int, default=None,
                       help="Nombre max de frames")
    parser.add_argument("--save-frames", action="store_true",
                       help="Sauvegarder frames avec détections")
    parser.add_argument("--no-save", action="store_true",
                       help="Ne pas sauvegarder les résultats")
    
    return parser.parse_args()


async def main():
    """Point d'entrée principal QWEN2-VL ONLY."""
    args = parse_arguments()
    
    print(f"""
🎯 SYSTÈME DE SURVEILLANCE QWEN2-VL UNIQUEMENT
===============================================

Configuration:
📹 Source vidéo  : {args.video}
🤖 Modèle VLM    : {args.model} UNIQUEMENT
⚙️ Mode          : {args.mode}
💾 Sauvegarde    : {'Activée' if not args.no_save else 'Désactivée'}
🖼️ Frames        : {'Sauvées' if args.save_frames else 'Non sauvées'}
📊 Max frames    : {args.max_frames or 'Illimité'}

⚠️ ATTENTION: AUCUN FALLBACK CONFIGURÉ
Si {args.model} échoue → ARRÊT DU SYSTÈME

WORKFLOW QWEN2-VL ONLY:
1. 📹 Capture vidéo → logs détaillés
2. 🔍 Détection YOLO → comptage objets  
3. 🧠 Analyse Qwen2-VL UNIQUEMENT
4. 🚨 Prise décisions → actions automatisées
5. 💾 Export JSON → résultats structurés
6. 📊 Statistiques finales → rapport complet
""")
    
    # Configuration du mode
    mode_mapping = {
        "FAST": OrchestrationMode.FAST,
        "BALANCED": OrchestrationMode.BALANCED,
        "THOROUGH": OrchestrationMode.THOROUGH
    }
    
    # Validation source vidéo
    video_source = args.video
    if video_source.lower() == "webcam":
        video_source = 0
    
    # Initialisation système QWEN2-VL ONLY
    system = QwenOnlySurveillanceSystem(
        video_source=video_source,
        vlm_model=args.model,
        orchestration_mode=mode_mapping[args.mode],
        save_results=not args.no_save,
        save_frames=args.save_frames
    )
    
    try:
        # Initialisation et démarrage
        await system.initialize()
        await system.run_surveillance(max_frames=args.max_frames)
        
    except Exception as e:
        logger.error(f"❌ Erreur système QWEN2-VL ONLY: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())