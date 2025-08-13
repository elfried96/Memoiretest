#!/usr/bin/env python3
"""
🎯 MAIN.PY - Pipeline Complet de Surveillance Intelligente
==========================================================

WORKFLOW COMPLET :
1. Capture vidéo → 2. Détection YOLO → 3. Tracking → 4. VLM Kimi-VL → 5. Orchestration → 6. Décisions

Architecture :
- Traitement temps réel des flux vidéo
- Détection/tracking des objets et personnes
- Analyse VLM avec Kimi-VL (principal) + fallbacks LLaVA/Qwen
- Orchestration intelligente des 8 outils avancés
- Prise de décisions automatisée
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
from dataclasses import dataclass
from enum import Enum
import logging
from io import BytesIO
import numpy as np

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Imports du système de surveillance
from src.core.vlm.dynamic_model import DynamicVisionLanguageModel
from src.core.orchestrator.vlm_orchestrator import (
    ModernVLMOrchestrator, 
    OrchestrationConfig, 
    OrchestrationMode
)
from src.core.types import AnalysisRequest, AnalysisResponse, Detection, BoundingBox
from src.detection.yolo_detector import YOLODetector
from src.detection.tracking.byte_tracker import BYTETracker


class AlertLevel(Enum):
    """Niveaux d'alerte du système."""
    NORMAL = "normal"
    ATTENTION = "attention"
    ALERTE = "alerte"
    CRITIQUE = "critique"


@dataclass
class SurveillanceFrame:
    """Frame de surveillance enrichi."""
    frame_id: int
    timestamp: float
    frame: np.ndarray
    detections: List[Detection]
    tracked_objects: List[Dict]
    vlm_analysis: Optional[AnalysisResponse] = None
    alert_level: AlertLevel = AlertLevel.NORMAL
    actions_taken: List[str] = None


class SurveillanceDecisionEngine:
    """Moteur de décision pour les actions de surveillance."""
    
    def __init__(self):
        self.alert_history = []
        self.action_log = []
    
    def process_analysis(self, analysis: AnalysisResponse, frame_context: Dict) -> Dict[str, Any]:
        """Traite l'analyse VLM et prend des décisions."""
        
        decisions = {
            "alert_level": AlertLevel.NORMAL,
            "actions": [],
            "notifications": [],
            "recording_required": False,
            "human_review": False
        }
        
        # === LOGIQUE DE DÉCISION ===
        
        # 1. Niveau de suspicion élevé
        if analysis.suspicion_level.value in ["HIGH", "CRITICAL"]:
            decisions["alert_level"] = AlertLevel.ALERTE
            decisions["recording_required"] = True
            decisions["actions"].append("start_recording")
            decisions["notifications"].append(f"Suspicion {analysis.suspicion_level.value} détectée")
        
        # 2. Confiance faible nécessite révision humaine
        if analysis.confidence < 0.6:
            decisions["human_review"] = True
            decisions["actions"].append("request_human_review")
        
        # 3. Actions spécifiques selon le type d'action détecté
        action_responses = {
            "SUSPICIOUS_ACTIVITY": [
                "increase_monitoring", 
                "alert_security", 
                "track_individual"
            ],
            "THEFT_ATTEMPT": [
                "immediate_alert", 
                "contact_security", 
                "start_recording",
                "track_suspect"
            ],
            "NORMAL_SHOPPING": [
                "continue_monitoring"
            ],
            "LOITERING": [
                "extended_observation",
                "gentle_staff_intervention"
            ]
        }
        
        if analysis.action_type.value in action_responses:
            decisions["actions"].extend(action_responses[analysis.action_type.value])
        
        # 4. Outils utilisés influencent les décisions
        if "adversarial_detector" in analysis.tools_used:
            decisions["actions"].append("security_protocol_enhanced")
        
        if "trajectory_analyzer" in analysis.tools_used:
            decisions["actions"].append("movement_pattern_logged")
        
        # 5. Contexte temporel
        current_time = time.strftime("%H:%M")
        if current_time < "08:00" or current_time > "22:00":
            # Hors heures d'ouverture = plus strict
            if decisions["alert_level"] == AlertLevel.NORMAL:
                decisions["alert_level"] = AlertLevel.ATTENTION
            decisions["actions"].append("after_hours_protocol")
        
        return decisions


class IntelligentSurveillanceSystem:
    """Système de surveillance intelligente complet."""
    
    def __init__(
        self,
        video_source: str = 0,  # 0 pour webcam, ou chemin vers fichier
        vlm_model: str = "kimi-vl-a3b-thinking",
        orchestration_mode: OrchestrationMode = OrchestrationMode.BALANCED
    ):
        self.video_source = video_source
        self.vlm_model = vlm_model
        self.orchestration_mode = orchestration_mode
        
        # === COMPOSANTS PRINCIPAUX ===
        
        # 1. Détection et tracking
        self.yolo_detector = YOLODetector(
            model_path="yolov8n.pt",  # Modèle léger pour démo
            device="auto"
        )
        self.tracker = BYTETracker()
        
        # 2. VLM et orchestration
        self.vlm = DynamicVisionLanguageModel(
            default_model=vlm_model,
            enable_fallback=True
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
        
        # 3. Moteur de décision
        self.decision_engine = SurveillanceDecisionEngine()
        
        # === ÉTAT DU SYSTÈME ===
        self.frame_count = 0
        self.processing_stats = {
            "total_frames": 0,
            "detected_objects": 0,
            "vlm_analyses": 0,
            "alerts_triggered": 0,
            "average_fps": 0.0
        }
        
        logger.info(f"🎯 Système initialisé - VLM: {vlm_model}, Mode: {orchestration_mode.value}")
    
    async def initialize(self):
        """Initialisation asynchrone des composants."""
        logger.info("🚀 Initialisation du système...")
        
        # Chargement du VLM
        logger.info(f"⏳ Chargement {self.vlm_model}...")
        vlm_loaded = await self.vlm.load_model()
        
        if vlm_loaded:
            logger.info(f"✅ {self.vlm_model} chargé avec succès")
        else:
            logger.warning("⚠️ VLM principal échec, fallback activé")
        
        # Test des composants
        logger.info("🔍 Tests des composants...")
        
        # Test YOLO
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        test_detections = self.yolo_detector.detect(test_frame)
        logger.info(f"✅ YOLO opérationnel - {len(test_detections)} détections test")
        
        # Statut complet
        status = await self.orchestrator.health_check()
        logger.info(f"📊 Health check: {status}")
        
        logger.info("🎯 Système prêt pour surveillance!")
    
    def encode_frame_to_base64(self, frame: np.ndarray) -> str:
        """Encode un frame OpenCV en base64 pour le VLM."""
        # Conversion BGR (OpenCV) → RGB (PIL)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Encodage en JPEG puis base64
        _, buffer = cv2.imencode('.jpg', frame_rgb, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_b64 = base64.b64encode(buffer).decode('utf-8')
        
        return frame_b64
    
    def create_detections_list(self, yolo_results) -> List[Detection]:
        """Convertit les résultats YOLO en liste de Detection."""
        detections = []
        
        for result in yolo_results:
            if hasattr(result, 'boxes') and result.boxes is not None:
                for box in result.boxes:
                    # Extraction des coordonnées
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    
                    # Nom de classe (COCO classes)
                    class_names = result.names
                    class_name = class_names.get(cls, f"class_{cls}")
                    
                    # Création de la détection
                    detection = Detection(
                        bbox=BoundingBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2)),
                        confidence=float(conf),
                        class_name=class_name,
                        track_id=None  # Sera ajouté par le tracker
                    )
                    detections.append(detection)
        
        return detections
    
    async def process_frame(self, frame: np.ndarray) -> SurveillanceFrame:
        """Traite un frame complet : détection → tracking → VLM → décisions."""
        
        start_time = time.time()
        self.frame_count += 1
        
        # === ÉTAPE 1: DÉTECTION YOLO ===
        logger.debug(f"🔍 Frame {self.frame_count} - Détection YOLO...")
        yolo_results = self.yolo_detector.detect(frame)
        detections = self.create_detections_list(yolo_results)
        
        self.processing_stats["detected_objects"] += len(detections)
        
        # === ÉTAPE 2: TRACKING ===
        logger.debug(f"🎯 Frame {self.frame_count} - Tracking...")
        # Note: Implémentation simplifiée du tracking
        tracked_objects = []
        for i, detection in enumerate(detections):
            tracked_objects.append({
                "track_id": f"track_{i}_{self.frame_count}",
                "detection": detection,
                "age": 1,
                "status": "active"
            })
        
        # === ÉTAPE 3: ANALYSE VLM (conditionnelle) ===
        vlm_analysis = None
        
        # Analyse VLM si personnes détectées ou tous les N frames
        should_analyze = (
            len([d for d in detections if d.class_name == "person"]) > 0 or
            self.frame_count % 30 == 0  # Analyse périodique
        )
        
        if should_analyze:
            logger.debug(f"🧠 Frame {self.frame_count} - Analyse VLM...")
            
            # Encodage pour VLM
            frame_b64 = self.encode_frame_to_base64(frame)
            
            # Contexte enrichi
            context = {
                "frame_id": self.frame_count,
                "timestamp": time.time(),
                "location": "Store Main Area",
                "camera": "CAM_01",
                "person_count": len([d for d in detections if d.class_name == "person"]),
                "total_objects": len(detections),
                "time_of_day": time.strftime("%H:%M:%S")
            }
            
            # Analyse orchestrée
            vlm_analysis = await self.orchestrator.analyze_surveillance_frame(
                frame_data=frame_b64,
                detections=detections,
                context=context
            )
            
            self.processing_stats["vlm_analyses"] += 1
        
        # === ÉTAPE 4: PRISE DE DÉCISIONS ===
        alert_level = AlertLevel.NORMAL
        actions_taken = []
        
        if vlm_analysis:
            decisions = self.decision_engine.process_analysis(vlm_analysis, {
                "frame_count": self.frame_count,
                "detections_count": len(detections)
            })
            
            alert_level = decisions["alert_level"]
            actions_taken = decisions["actions"]
            
            # Logging des décisions importantes
            if alert_level != AlertLevel.NORMAL:
                logger.warning(f"⚠️ Frame {self.frame_count}: {alert_level.value} - Actions: {actions_taken}")
                self.processing_stats["alerts_triggered"] += 1
        
        # === CRÉATION DU FRAME DE SURVEILLANCE ===
        surveillance_frame = SurveillanceFrame(
            frame_id=self.frame_count,
            timestamp=time.time(),
            frame=frame,
            detections=detections,
            tracked_objects=tracked_objects,
            vlm_analysis=vlm_analysis,
            alert_level=alert_level,
            actions_taken=actions_taken
        )
        
        # === MISE À JOUR DES STATISTIQUES ===
        processing_time = time.time() - start_time
        self.processing_stats["total_frames"] += 1
        
        # Calcul FPS moyen
        if self.processing_stats["total_frames"] > 0:
            self.processing_stats["average_fps"] = 1.0 / processing_time
        
        logger.debug(f"✅ Frame {self.frame_count} traité en {processing_time:.2f}s")
        
        return surveillance_frame
    
    def draw_surveillance_overlay(self, frame: np.ndarray, surveillance_frame: SurveillanceFrame) -> np.ndarray:
        """Dessine les overlays de surveillance sur le frame."""
        overlay_frame = frame.copy()
        
        # === DÉTECTIONS ET TRACKING ===
        for detection in surveillance_frame.detections:
            bbox = detection.bbox
            
            # Couleur selon la classe
            color = (0, 255, 0) if detection.class_name == "person" else (255, 0, 0)
            
            # Rectangle de détection
            cv2.rectangle(
                overlay_frame,
                (int(bbox.x1), int(bbox.y1)),
                (int(bbox.x2), int(bbox.y2)),
                color, 2
            )
            
            # Label avec confiance
            label = f"{detection.class_name}: {detection.confidence:.2f}"
            cv2.putText(
                overlay_frame, label,
                (int(bbox.x1), int(bbox.y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )
        
        # === INFORMATIONS SYSTÈME ===
        info_y = 30
        
        # Frame ID et timestamp
        cv2.putText(overlay_frame, f"Frame: {surveillance_frame.frame_id}", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        info_y += 25
        
        # Alert level
        alert_color = {
            AlertLevel.NORMAL: (0, 255, 0),
            AlertLevel.ATTENTION: (0, 255, 255),
            AlertLevel.ALERTE: (0, 165, 255),
            AlertLevel.CRITIQUE: (0, 0, 255)
        }
        
        cv2.putText(overlay_frame, f"Alert: {surveillance_frame.alert_level.value.upper()}", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                   alert_color[surveillance_frame.alert_level], 2)
        info_y += 25
        
        # Analyse VLM si disponible
        if surveillance_frame.vlm_analysis:
            analysis = surveillance_frame.vlm_analysis
            cv2.putText(overlay_frame, f"VLM: {analysis.suspicion_level.value} ({analysis.confidence:.2f})", 
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            info_y += 20
            
            cv2.putText(overlay_frame, f"Action: {analysis.action_type.value}", 
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # === STATISTIQUES ===
        stats_y = overlay_frame.shape[0] - 100
        cv2.putText(overlay_frame, f"FPS: {self.processing_stats['average_fps']:.1f}", 
                   (10, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        stats_y += 20
        
        cv2.putText(overlay_frame, f"Detections: {len(surveillance_frame.detections)}", 
                   (10, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        stats_y += 20
        
        cv2.putText(overlay_frame, f"Alerts: {self.processing_stats['alerts_triggered']}", 
                   (10, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return overlay_frame
    
    async def run_surveillance(self, max_frames: int = None, display: bool = True):
        """Lance la surveillance en temps réel."""
        logger.info("🎬 Démarrage de la surveillance...")
        
        # Ouverture de la source vidéo
        cap = cv2.VideoCapture(self.video_source)
        
        if not cap.isOpened():
            logger.error(f"❌ Impossible d'ouvrir la source vidéo: {self.video_source}")
            return
        
        logger.info(f"📹 Source vidéo ouverte: {self.video_source}")
        
        try:
            frame_processed = 0
            
            while True:
                # Lecture du frame
                ret, frame = cap.read()
                if not ret:
                    logger.warning("📹 Fin de vidéo ou erreur lecture")
                    break
                
                # Traitement complet du frame
                surveillance_frame = await self.process_frame(frame)
                
                # Affichage si demandé
                if display:
                    overlay_frame = self.draw_surveillance_overlay(frame, surveillance_frame)
                    cv2.imshow("🎯 Surveillance Intelligente - Multi-VLM", overlay_frame)
                    
                    # ESC pour quitter
                    if cv2.waitKey(1) & 0xFF == 27:
                        logger.info("🛑 Arrêt demandé par utilisateur")
                        break
                
                frame_processed += 1
                
                # Limite de frames si spécifiée
                if max_frames and frame_processed >= max_frames:
                    logger.info(f"🏁 Limite de {max_frames} frames atteinte")
                    break
                
                # Logging périodique
                if frame_processed % 60 == 0:
                    logger.info(f"📊 {frame_processed} frames traités - "
                              f"FPS moyen: {self.processing_stats['average_fps']:.1f} - "
                              f"Alertes: {self.processing_stats['alerts_triggered']}")
        
        except KeyboardInterrupt:
            logger.info("🛑 Arrêt par interruption clavier")
        
        finally:
            # Nettoyage
            cap.release()
            if display:
                cv2.destroyAllWindows()
            
            # Statistiques finales
            logger.info("📈 Statistiques finales:")
            for key, value in self.processing_stats.items():
                logger.info(f"  • {key}: {value}")
    
    async def shutdown(self):
        """Arrêt propre du système."""
        logger.info("🛑 Arrêt du système de surveillance...")
        await self.orchestrator.shutdown()
        await self.vlm.shutdown()
        logger.info("✅ Système arrêté proprement")


def parse_arguments():
    """Parse les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(
        description="🎯 Système de Surveillance Intelligente Multi-VLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  python main.py                                    # Webcam (défaut)
  python main.py --video webcam                     # Webcam explicite
  python main.py --video /path/to/video.mp4         # Fichier vidéo
  python main.py --video rtsp://camera_ip:554/stream # Flux RTSP
  python main.py --model llava-v1.6-mistral-7b      # Modèle différent
  python main.py --mode FAST                        # Mode rapide
  python main.py --mode THOROUGH                    # Mode complet
        """
    )
    
    parser.add_argument(
        "--video", "-v",
        default="webcam",
        help="Source vidéo: 'webcam', chemin vers fichier, ou URL RTSP (défaut: webcam)"
    )
    
    parser.add_argument(
        "--model", "-m",
        default="kimi-vl-a3b-thinking",
        choices=["kimi-vl-a3b-thinking", "kimi-vl-a3b-instruct", "llava-v1.6-mistral-7b", "qwen2-vl-7b-instruct"],
        help="Modèle VLM principal (défaut: kimi-vl-a3b-thinking)"
    )
    
    parser.add_argument(
        "--mode",
        default="BALANCED",
        choices=["FAST", "BALANCED", "THOROUGH"],
        help="Mode d'orchestration (défaut: BALANCED)"
    )
    
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Nombre maximum de frames à traiter (défaut: infini)"
    )
    
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Désactiver l'affichage vidéo (mode headless)"
    )
    
    return parser.parse_args()


def validate_video_source(video_arg: str) -> Any:
    """Valide et convertit la source vidéo."""
    
    if video_arg.lower() == "webcam":
        logger.info("📹 Source vidéo: Webcam (device 0)")
        return 0
    
    # Vérifier si c'est un fichier local
    video_path = Path(video_arg)
    if video_path.exists():
        if video_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
            logger.info(f"📹 Source vidéo: Fichier local {video_path}")
            return str(video_path)
        else:
            logger.warning(f"⚠️ Extension de fichier non recommandée: {video_path.suffix}")
            return str(video_path)
    
    # Vérifier si c'est une URL (RTSP, HTTP, etc.)
    if video_arg.startswith(('rtsp://', 'http://', 'https://')):
        logger.info(f"📹 Source vidéo: Flux réseau {video_arg}")
        return video_arg
    
    # Sinon, considérer comme chemin de fichier (même si inexistant)
    logger.warning(f"⚠️ Fichier non trouvé, tentative d'ouverture: {video_arg}")
    return video_arg


async def main():
    """Point d'entrée principal du système."""
    
    # Parsing des arguments
    args = parse_arguments()
    
    print(f"""
🎯 SYSTÈME DE SURVEILLANCE INTELLIGENTE MULTI-VLM
==================================================

Configuration:
📹 Source vidéo  : {args.video}
🤖 Modèle VLM    : {args.model}
⚙️ Mode          : {args.mode}
🖥️ Affichage     : {'Désactivé' if args.no_display else 'Activé'}
📊 Max frames    : {args.max_frames or 'Illimité'}

WORKFLOW COMPLET:
1. 📹 Capture vidéo (webcam/fichier)
2. 🔍 Détection YOLO (objets/personnes)  
3. 🎯 Tracking multi-objets
4. 🧠 Analyse VLM {args.model} + 8 outils avancés
5. ⚙️ Orchestration intelligente
6. 🚨 Prise de décisions automatisée
7. 📊 Affichage temps réel + alertes

Appuyez sur ESC pour quitter (si affichage activé)
""")
    
    # Validation de la source vidéo
    try:
        video_source = validate_video_source(args.video)
    except Exception as e:
        logger.error(f"❌ Erreur validation source vidéo: {e}")
        sys.exit(1)
    
    # Configuration du mode d'orchestration
    mode_mapping = {
        "FAST": OrchestrationMode.FAST,
        "BALANCED": OrchestrationMode.BALANCED,
        "THOROUGH": OrchestrationMode.THOROUGH
    }
    orchestration_mode = mode_mapping[args.mode]
    
    # Initialisation du système
    system = IntelligentSurveillanceSystem(
        video_source=video_source,
        vlm_model=args.model,
        orchestration_mode=orchestration_mode
    )
    
    try:
        # Initialisation asynchrone
        await system.initialize()
        
        # Démarrage de la surveillance
        await system.run_surveillance(
            max_frames=args.max_frames,
            display=not args.no_display
        )
    
    except Exception as e:
        logger.error(f"❌ Erreur système: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Arrêt propre
        await system.shutdown()


if __name__ == "__main__":
    asyncio.run(main())