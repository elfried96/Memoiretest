#!/usr/bin/env python3
"""
🎯 MAIN HEADLESS - Pipeline Sans Interface Graphique
===================================================

Version adaptée du main.py pour environnements sans display (serveurs, SSH, etc.)
Génère des logs détaillés et sauvegarde optionnelle des frames avec détections.
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
from src.core.types import AnalysisRequest, AnalysisResponse, DetectedObject, BoundingBox
from src.detection.yolo_detector import YOLODetector
from src.detection.tracking.byte_tracker import BYTETracker


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


class HeadlessSurveillanceSystem:
    """Système de surveillance pour environnement sans interface graphique."""
    
    def __init__(
        self,
        video_source: str = 0,
        vlm_model: str = "kimi-vl-a3b-thinking", 
        orchestration_mode: OrchestrationMode = OrchestrationMode.BALANCED,
        save_results: bool = True,
        save_frames: bool = False
    ):
        self.video_source = video_source
        self.vlm_model = vlm_model
        self.orchestration_mode = orchestration_mode
        self.save_results = save_results
        self.save_frames = save_frames
        
        # Dossiers de sortie
        self.output_dir = Path("surveillance_output")
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
        
        # === ÉTAT DU SYSTÈME ===
        self.frame_count = 0
        self.results_log = []
        self.processing_stats = {
            "total_frames": 0,
            "detected_objects": 0,
            "persons_detected": 0,
            "vlm_analyses": 0,
            "alerts_triggered": 0,
            "total_processing_time": 0.0,
            "average_fps": 0.0
        }
        
        logger.info(f"🎯 Système headless initialisé - VLM: {vlm_model}, Mode: {orchestration_mode.value}")
        logger.info(f"📁 Résultats sauvés dans: {self.output_dir}")
    
    async def initialize(self):
        """Initialisation asynchrone des composants."""
        logger.info("🚀 Initialisation du système headless...")
        
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
        
        logger.info("🎯 Système prêt pour surveillance headless!")
    
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
    
    def encode_frame_to_base64(self, frame: np.ndarray) -> str:
        """Encode un frame OpenCV en base64 pour le VLM."""
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
            
            # Label
            label = f"{detection.class_name}: {detection.confidence:.2f}"
            cv2.putText(
                overlay_frame, label,
                (int(bbox.x1), int(bbox.y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )
        
        # Sauvegarder
        frame_path = self.frames_dir / f"frame_{frame_id:06d}.jpg"
        cv2.imwrite(str(frame_path), overlay_frame)
        logger.debug(f"🖼️ Frame sauvé: {frame_path}")
    
    async def process_frame(self, frame: np.ndarray) -> SurveillanceResult:
        """Traite un frame complet en mode headless."""
        
        start_time = time.time()
        self.frame_count += 1
        
        # === ÉTAPE 1: DÉTECTION YOLO ===
        yolo_results = self.yolo_detector.detect(frame)
        detections = self.create_detections_list(yolo_results)
        
        persons_count = len([d for d in detections if d.class_name == "person"])
        
        self.processing_stats["detected_objects"] += len(detections)
        self.processing_stats["persons_detected"] += persons_count
        
        # === ÉTAPE 2: ANALYSE VLM (conditionnelle) ===
        vlm_analysis = None
        alert_level = AlertLevel.NORMAL
        actions_taken = []
        
        # Analyse VLM si personnes détectées ou tous les N frames
        should_analyze = (
            persons_count > 0 or
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
                "person_count": persons_count,
                "total_objects": len(detections),
                "time_of_day": time.strftime("%H:%M:%S")
            }
            
            try:
                # Analyse orchestrée
                vlm_analysis = await self.orchestrator.analyze_surveillance_frame(
                    frame_data=frame_b64,
                    detections=detections,
                    context=context
                )
                
                self.processing_stats["vlm_analyses"] += 1
                
                # Décisions basées sur l'analyse
                if vlm_analysis.suspicion_level.value in ["HIGH", "CRITICAL"]:
                    alert_level = AlertLevel.ALERTE
                    actions_taken = ["alert_triggered", "recording_started"]
                    self.processing_stats["alerts_triggered"] += 1
                elif vlm_analysis.suspicion_level.value == "MEDIUM":
                    alert_level = AlertLevel.ATTENTION
                    actions_taken = ["increased_monitoring"]
                
            except Exception as e:
                logger.error(f"❌ Erreur analyse VLM frame {self.frame_count}: {e}")
        
        # === SAUVEGARDE FRAME SI DEMANDÉE ===
        if len(detections) > 0 or alert_level != AlertLevel.NORMAL:
            self.save_frame_with_detections(frame, detections, self.frame_count)
        
        # === CRÉATION DU RÉSULTAT ===
        processing_time = time.time() - start_time
        
        result = SurveillanceResult(
            frame_id=self.frame_count,
            timestamp=time.time(),
            detections_count=len(detections),
            persons_detected=persons_count,
            alert_level=alert_level.value,
            vlm_analysis=vlm_analysis.to_dict() if vlm_analysis and hasattr(vlm_analysis, 'to_dict') else (vlm_analysis.__dict__ if vlm_analysis else None),
            actions_taken=actions_taken,
            processing_time=processing_time
        )
        
        # === MISE À JOUR DES STATISTIQUES ===
        self.processing_stats["total_frames"] += 1
        self.processing_stats["total_processing_time"] += processing_time
        
        if self.processing_stats["total_frames"] > 0:
            avg_time = self.processing_stats["total_processing_time"] / self.processing_stats["total_frames"]
            self.processing_stats["average_fps"] = 1.0 / avg_time if avg_time > 0 else 0
        
        # === LOGGING DÉTAILLÉ ===
        if persons_count > 0 or alert_level != AlertLevel.NORMAL:
            logger.info(f"📊 Frame {self.frame_count}: "
                       f"{len(detections)} objs, {persons_count} personnes, "
                       f"Alert: {alert_level.value}, "
                       f"Actions: {actions_taken}, "
                       f"Temps: {processing_time:.2f}s")
        elif self.frame_count % 60 == 0:
            logger.info(f"📈 Frame {self.frame_count}: "
                       f"FPS: {self.processing_stats['average_fps']:.1f}, "
                       f"Total objets: {self.processing_stats['detected_objects']}, "
                       f"Analyses VLM: {self.processing_stats['vlm_analyses']}")
        
        return result
    
    async def run_surveillance(self, max_frames: int = None):
        """Lance la surveillance en mode headless."""
        logger.info("🎬 Démarrage surveillance headless...")
        
        # Ouverture de la source vidéo
        cap = cv2.VideoCapture(self.video_source)
        
        if not cap.isOpened():
            logger.error(f"❌ Impossible d'ouvrir source vidéo: {self.video_source}")
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
    
    def save_results_to_json(self):
        """Sauvegarde les résultats en JSON."""
        results_file = self.output_dir / f"surveillance_results_{int(time.time())}.json"
        
        output_data = {
            "metadata": {
                "video_source": str(self.video_source),
                "vlm_model": self.vlm_model,
                "orchestration_mode": self.orchestration_mode.value,
                "total_frames": len(self.results_log),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "statistics": self.processing_stats,
            "results": [asdict(result) for result in self.results_log]
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Résultats sauvés: {results_file}")
    
    def print_final_statistics(self):
        """Affiche les statistiques finales."""
        logger.info("=" * 60)
        logger.info("📈 STATISTIQUES FINALES DE SURVEILLANCE")
        logger.info("=" * 60)
        
        for key, value in self.processing_stats.items():
            if isinstance(value, float):
                logger.info(f"  • {key}: {value:.2f}")
            else:
                logger.info(f"  • {key}: {value}")
        
        # Analyse des alertes
        alerts = [r for r in self.results_log if r.alert_level != "normal"]
        if alerts:
            logger.info(f"🚨 ALERTES DÉTECTÉES: {len(alerts)}")
            for alert in alerts[-5:]:  # 5 dernières alertes
                logger.info(f"   Frame {alert.frame_id}: {alert.alert_level} - {alert.actions_taken}")
        
        # Analyse des personnes
        frames_with_persons = [r for r in self.results_log if r.persons_detected > 0]
        if frames_with_persons:
            max_persons = max(r.persons_detected for r in frames_with_persons)
            logger.info(f"👥 Max personnes simultanées: {max_persons}")
            logger.info(f"👥 Frames avec personnes: {len(frames_with_persons)}/{len(self.results_log)}")


def parse_arguments():
    """Parse les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(
        description="🎯 Système de Surveillance Headless",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--video", "-v", default="webcam",
                       help="Source vidéo")
    parser.add_argument("--model", "-m", default="kimi-vl-a3b-thinking",
                       help="Modèle VLM")
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
    """Point d'entrée principal."""
    args = parse_arguments()
    
    print(f"""
🎯 SYSTÈME DE SURVEILLANCE HEADLESS
====================================

Configuration:
📹 Source vidéo  : {args.video}
🤖 Modèle VLM    : {args.model}
⚙️ Mode          : {args.mode}
💾 Sauvegarde    : {'Activée' if not args.no_save else 'Désactivée'}
🖼️ Frames        : {'Sauvées' if args.save_frames else 'Non sauvées'}
📊 Max frames    : {args.max_frames or 'Illimité'}

WORKFLOW HEADLESS:
1. 📹 Capture vidéo → logs détaillés
2. 🔍 Détection YOLO → comptage objets  
3. 🧠 Analyse VLM → évaluation suspicion
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
    
    # Initialisation système
    system = HeadlessSurveillanceSystem(
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
        logger.error(f"❌ Erreur système: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())