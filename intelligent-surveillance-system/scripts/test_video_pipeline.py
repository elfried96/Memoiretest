#!/usr/bin/env python3
"""
🎬 Script de Test Pipeline Vidéo Complet avec Kimi-VL
=====================================================

Test l'architecture complète de surveillance sur des vidéos :
- Détection YOLO
- Tracking ByteTracker  
- Analyse VLM avec Kimi-VL uniquement
- Orchestration des 8 outils avancés
- Génération de rapports détaillés

Usage:
    python scripts/test_video_pipeline.py video.mp4
    python scripts/test_video_pipeline.py video.mp4 --profile thorough
"""

import sys
import asyncio
import cv2
import time
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import asdict
import numpy as np

# Configuration du path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Imports du système
from config.video_test_config import (
    get_video_test_config, 
    get_system_config_for_video_test,
    VideoTestConfig,
    VIDEO_OUTPUTS_ROOT
)
from src.core.vlm.dynamic_model import DynamicVisionLanguageModel
from src.core.orchestrator.vlm_orchestrator import ModernVLMOrchestrator, OrchestrationConfig
from src.detection.yolo_detector import YOLODetector
from src.detection.tracking.byte_tracker import BYTETracker
from src.core.types import AnalysisRequest, Detection, BoundingBox
from src.utils.performance import start_performance_monitoring, stop_performance_monitoring, get_performance_summary


class VideoTestResult:
    """Résultats d'un test vidéo."""
    
    def __init__(self):
        self.video_path: str = ""
        self.config_profile: str = ""
        self.total_frames: int = 0
        self.processed_frames: int = 0
        self.processing_time: float = 0.0
        self.fps_average: float = 0.0
        
        # Statistiques de détection
        self.detections_total: int = 0
        self.persons_detected: int = 0
        self.objects_detected: int = 0
        
        # Statistiques VLM
        self.vlm_analyses: int = 0
        self.vlm_success_rate: float = 0.0
        self.vlm_average_confidence: float = 0.0
        
        # Alertes générées
        self.alerts_normal: int = 0
        self.alerts_attention: int = 0
        self.alerts_critique: int = 0
        
        # Performance système
        self.memory_peak_mb: float = 0.0
        self.gpu_memory_peak_mb: float = 0.0
        
        # Erreurs
        self.errors: List[str] = []


class VideoTestPipeline:
    """Pipeline de test vidéo complet."""
    
    def __init__(self, config: VideoTestConfig, system_config):
        self.config = config
        self.system_config = system_config
        
        # Composants du pipeline
        self.vlm_orchestrator: Optional[ModernVLMOrchestrator] = None
        self.yolo_detector: Optional[YOLODetector] = None
        self.tracker: Optional[BYTETracker] = None
        
        # Résultats
        self.result = VideoTestResult()
        
        # État
        self.is_initialized = False
    
    async def initialize(self) -> bool:
        """Initialisation du pipeline."""
        print("🔧 Initialisation du pipeline de test vidéo...")
        
        try:
            # VLM Orchestrator avec Kimi-VL uniquement
            print("🤖 Chargement Kimi-VL...")
            orchestration_config = OrchestrationConfig(
                mode=self.system_config.orchestration.mode,
                max_concurrent_tools=self.system_config.orchestration.max_concurrent_tools,
                timeout_seconds=self.system_config.orchestration.timeout_seconds,
                confidence_threshold=self.system_config.orchestration.confidence_threshold,
                enable_advanced_tools=True
            )
            
            self.vlm_orchestrator = ModernVLMOrchestrator(
                vlm_model_name=self.system_config.vlm.primary_model,
                config=orchestration_config
            )
            
            # YOLO Detector
            print("🔍 Initialisation YOLO...")
            self.yolo_detector = YOLODetector(
                model_path=self.system_config.yolo.model_path,
                confidence_threshold=self.config.confidence_threshold,
                device=self.system_config.yolo.device.value
            )
            await asyncio.to_thread(self.yolo_detector.load_model)
            
            # ByteTracker
            print("🎯 Initialisation Tracker...")
            self.tracker = BYTETracker(
                frame_rate=30,
                track_thresh=self.config.detection_threshold
            )
            
            self.is_initialized = True
            print("✅ Pipeline initialisé avec succès")
            return True
            
        except Exception as e:
            print(f"❌ Erreur d'initialisation: {e}")
            self.result.errors.append(f"Initialization error: {str(e)}")
            return False
    
    def _create_detections_from_yolo(self, yolo_results, frame_shape) -> List[Detection]:
        """Conversion des résultats YOLO en objets Detection."""
        detections = []
        
        if yolo_results and len(yolo_results) > 0:
            boxes = yolo_results[0].boxes
            if boxes is not None:
                for box in boxes:
                    # Extraction des coordonnées
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = self.yolo_detector.model.names[class_id]
                    
                    # Création de la detection
                    detection = Detection(
                        bbox=BoundingBox(x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2)),
                        confidence=float(confidence),
                        class_name=class_name,
                        class_id=class_id
                    )
                    detections.append(detection)
        
        return detections
    
    def _encode_frame_to_base64(self, frame: np.ndarray) -> str:
        """Encodage d'une frame en base64 pour le VLM."""
        import base64
        from io import BytesIO
        from PIL import Image
        
        # Conversion BGR -> RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # Encodage base64
        buffer = BytesIO()
        pil_image.save(buffer, format='JPEG', quality=85)
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return img_str
    
    async def process_video(self, video_path: str) -> VideoTestResult:
        """Traitement complet d'une vidéo."""
        self.result.video_path = video_path
        
        if not self.is_initialized:
            if not await self.initialize():
                return self.result
        
        print(f"🎬 Traitement vidéo: {video_path}")
        
        # Ouverture de la vidéo
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            error_msg = f"Impossible d'ouvrir la vidéo: {video_path}"
            print(f"❌ {error_msg}")
            self.result.errors.append(error_msg)
            return self.result
        
        # Informations vidéo
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.result.total_frames = min(total_frames, self.config.max_frames)
        
        print(f"📊 Vidéo: {total_frames} frames, {fps:.1f} FPS")
        print(f"⚙️ Configuration: {self.result.total_frames} frames max, skip={self.config.frame_skip}")
        
        # Préparation sortie vidéo si demandée
        output_writer = None
        if self.config.save_video_output:
            output_path = self.config.videos_dir / f"processed_{Path(video_path).stem}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_writer = cv2.VideoWriter(
                str(output_path), fourcc, fps/self.config.frame_skip,
                (self.config.resize_width, self.config.resize_height)
            )
        
        # Traitement frame par frame
        frame_count = 0
        processed_count = 0
        start_time = time.time()
        
        frame_results = []
        
        try:
            while frame_count < self.result.total_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Skip frames selon configuration
                if frame_count % self.config.frame_skip != 0:
                    continue
                
                processed_count += 1
                
                try:
                    # Redimensionnement
                    frame_resized = cv2.resize(frame, (self.config.resize_width, self.config.resize_height))
                    
                    # Détection YOLO
                    yolo_results = self.yolo_detector.detect(frame_resized)
                    detections = self._create_detections_from_yolo(yolo_results, frame_resized.shape)
                    
                    # Mise à jour des statistiques
                    self.result.detections_total += len(detections)
                    self.result.persons_detected += len([d for d in detections if d.class_name == "person"])
                    self.result.objects_detected += len([d for d in detections if d.class_name != "person"])
                    
                    # Tracking
                    if self.config.tracking_enabled and detections:
                        # Note: Ici on devrait utiliser le tracker, simplifié pour l'exemple
                        pass
                    
                    # Analyse VLM (une frame sur 10 pour économiser)
                    vlm_result = None
                    if processed_count % 10 == 0 and detections:
                        try:
                            frame_b64 = self._encode_frame_to_base64(frame_resized)
                            
                            analysis_request = AnalysisRequest(
                                frame_data=frame_b64,
                                context={
                                    "frame_number": frame_count,
                                    "detections_count": len(detections),
                                    "timestamp": time.time()
                                }
                            )
                            
                            vlm_result = await self.vlm_orchestrator.analyze_surveillance_frame(
                                frame_data=frame_b64,
                                detections=detections,
                                context=analysis_request.context
                            )
                            
                            self.result.vlm_analyses += 1
                            if vlm_result.confidence > 0.5:
                                self.result.vlm_average_confidence += vlm_result.confidence
                            
                            # Comptage des alertes selon le niveau de suspicion
                            if vlm_result.suspicion_level.value == "low":
                                self.result.alerts_normal += 1
                            elif vlm_result.suspicion_level.value == "medium":
                                self.result.alerts_attention += 1
                            elif vlm_result.suspicion_level.value == "high":
                                self.result.alerts_critique += 1
                                
                        except Exception as e:
                            print(f"⚠️ Erreur analyse VLM frame {frame_count}: {e}")
                            self.result.errors.append(f"VLM error frame {frame_count}: {str(e)}")
                    
                    # Sauvegarde frame si demandée
                    if self.config.save_processed_frames and processed_count % 30 == 0:
                        frame_path = self.config.frames_dir / f"frame_{frame_count:06d}.jpg"
                        cv2.imwrite(str(frame_path), frame_resized)
                    
                    # Sauvegarde dans vidéo de sortie
                    if output_writer:
                        # Ajout d'informations sur la frame (détections, etc.)
                        display_frame = frame_resized.copy()
                        
                        # Dessiner les détections
                        for det in detections:
                            cv2.rectangle(display_frame, 
                                        (det.bbox.x1, det.bbox.y1), 
                                        (det.bbox.x2, det.bbox.y2), 
                                        (0, 255, 0), 2)
                            cv2.putText(display_frame, 
                                      f"{det.class_name} {det.confidence:.2f}",
                                      (det.bbox.x1, det.bbox.y1-10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        
                        # Info globale
                        info_text = f"Frame: {frame_count} | Dets: {len(detections)}"
                        cv2.putText(display_frame, info_text, (10, 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
                        output_writer.write(display_frame)
                    
                    # Stockage des résultats de frame
                    frame_result = {
                        "frame_number": frame_count,
                        "detections": len(detections),
                        "persons": len([d for d in detections if d.class_name == "person"]),
                        "vlm_analysis": vlm_result.description if vlm_result else None,
                        "vlm_confidence": vlm_result.confidence if vlm_result else None
                    }
                    frame_results.append(frame_result)
                    
                    # Affichage progression
                    if processed_count % 30 == 0:
                        elapsed = time.time() - start_time
                        fps_current = processed_count / elapsed
                        print(f"📈 Frame {frame_count}/{self.result.total_frames} | "
                              f"Processed: {processed_count} | "
                              f"FPS: {fps_current:.1f} | "
                              f"Détections: {len(detections)}")
                
                except Exception as e:
                    print(f"⚠️ Erreur traitement frame {frame_count}: {e}")
                    self.result.errors.append(f"Frame {frame_count} error: {str(e)}")
                    continue
        
        finally:
            cap.release()
            if output_writer:
                output_writer.release()
        
        # Finalisation des statistiques
        self.result.processed_frames = processed_count
        self.result.processing_time = time.time() - start_time
        self.result.fps_average = processed_count / self.result.processing_time
        
        if self.result.vlm_analyses > 0:
            self.result.vlm_average_confidence /= self.result.vlm_analyses
            self.result.vlm_success_rate = (self.result.vlm_analyses / (processed_count // 10)) * 100
        
        # Sauvegarde des résultats détaillés
        if self.config.save_analysis_results:
            results_data = {
                "summary": asdict(self.result),
                "frame_results": frame_results,
                "config": asdict(self.config),
                "system_config": self.system_config.to_dict()
            }
            
            results_path = self.config.results_dir / f"test_results_{Path(video_path).stem}.json"
            with open(results_path, 'w') as f:
                json.dump(results_data, f, indent=2, default=str)
        
        print(f"✅ Traitement terminé: {processed_count} frames en {self.result.processing_time:.1f}s")
        return self.result
    
    def cleanup(self):
        """Nettoyage des ressources."""
        if self.vlm_orchestrator:
            # Note: Ajout éventuel de cleanup pour l'orchestrateur
            pass
        
        if self.yolo_detector:
            self.yolo_detector.cleanup()


def print_test_results(result: VideoTestResult, config_profile: str):
    """Affichage des résultats de test."""
    print("\n" + "="*60)
    print("🏆 RÉSULTATS DU TEST VIDÉO")
    print("="*60)
    
    print(f"📹 Vidéo: {Path(result.video_path).name}")
    print(f"⚙️ Profil: {config_profile}")
    print(f"📊 Frames: {result.processed_frames}/{result.total_frames}")
    print(f"⏱️ Durée: {result.processing_time:.1f}s")
    print(f"🎯 FPS moyen: {result.fps_average:.1f}")
    
    print(f"\n🔍 DÉTECTIONS:")
    print(f"  Total: {result.detections_total}")
    print(f"  Personnes: {result.persons_detected}")
    print(f"  Objets: {result.objects_detected}")
    
    print(f"\n🤖 ANALYSE VLM:")
    print(f"  Analyses: {result.vlm_analyses}")
    print(f"  Taux succès: {result.vlm_success_rate:.1f}%")
    print(f"  Confiance moyenne: {result.vlm_average_confidence:.2f}")
    
    print(f"\n🚨 ALERTES:")
    print(f"  Normal: {result.alerts_normal}")
    print(f"  Attention: {result.alerts_attention}")
    print(f"  Critique: {result.alerts_critique}")
    
    if result.errors:
        print(f"\n❌ ERREURS ({len(result.errors)}):")
        for error in result.errors[:5]:  # Afficher max 5 erreurs
            print(f"  • {error}")
        if len(result.errors) > 5:
            print(f"  ... et {len(result.errors) - 5} autres erreurs")


def create_argument_parser() -> argparse.ArgumentParser:
    """Créer le parser d'arguments."""
    parser = argparse.ArgumentParser(
        description="Test pipeline vidéo complet avec Kimi-VL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  python scripts/test_video_pipeline.py video.mp4                    # Test standard
  python scripts/test_video_pipeline.py video.mp4 --profile fast    # Test rapide
  python scripts/test_video_pipeline.py video.mp4 --profile thorough # Test complet
  python scripts/test_video_pipeline.py video.mp4 --no-save         # Sans sauvegarde
        """
    )
    
    parser.add_argument(
        "video_path",
        help="Chemin vers la vidéo à tester"
    )
    
    parser.add_argument(
        "--profile",
        type=str,
        default="standard",
        choices=["fast", "standard", "thorough", "demo"],
        help="Profil de configuration de test"
    )
    
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Ne pas sauvegarder les résultats"
    )
    
    parser.add_argument(
        "--performance",
        action="store_true",
        help="Activer le monitoring de performance"
    )
    
    return parser


async def main():
    """Point d'entrée principal."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Vérification du fichier vidéo
    video_path = Path(args.video_path)
    if not video_path.exists():
        print(f"❌ Fichier vidéo non trouvé: {video_path}")
        return 1
    
    print(f"🎬 Test Pipeline Vidéo avec Kimi-VL")
    print(f"📹 Vidéo: {video_path}")
    print(f"⚙️ Profil: {args.profile}")
    print("="*60)
    
    # Configuration
    video_config = get_video_test_config(args.profile)
    system_config = get_system_config_for_video_test()
    
    # Override sauvegarde si demandé
    if args.no_save:
        video_config.save_processed_frames = False
        video_config.save_analysis_results = False
        video_config.save_video_output = False
    
    # Création des dossiers de sortie
    for path in [video_config.output_dir, video_config.frames_dir, 
                 video_config.results_dir, video_config.videos_dir]:
        path.mkdir(parents=True, exist_ok=True)
    
    # Monitoring de performance
    if args.performance:
        start_performance_monitoring()
    
    try:
        # Création et exécution du pipeline
        pipeline = VideoTestPipeline(video_config, system_config)
        result = await pipeline.process_video(str(video_path))
        result.config_profile = args.profile
        
        # Nettoyage
        pipeline.cleanup()
        
        # Affichage des résultats
        print_test_results(result, args.profile)
        
        # Performance système
        if args.performance:
            perf_summary = get_performance_summary()
            print(f"\n📊 PERFORMANCE SYSTÈME:")
            if perf_summary.get('system_metrics'):
                metrics = perf_summary['system_metrics']
                print(f"  CPU moyen: {metrics.get('avg_cpu_percent', 0):.1f}%")
                print(f"  RAM moyenne: {metrics.get('avg_memory_percent', 0):.1f}%")
                print(f"  RAM utilisée: {metrics.get('avg_memory_used_mb', 0):.0f} MB")
        
        # Sortie selon les résultats
        if len(result.errors) == 0:
            print(f"\n🎉 Test réussi ! Résultats sauvés dans {video_config.results_dir}")
            return 0
        else:
            print(f"\n⚠️ Test terminé avec {len(result.errors)} erreurs")
            return 1
            
    except KeyboardInterrupt:
        print("\n⚠️ Interruption utilisateur")
        return 1
    except Exception as e:
        print(f"\n❌ Erreur inattendue: {e}")
        return 1
    finally:
        if args.performance:
            stop_performance_monitoring()


if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        sys.exit(result)
    except Exception as e:
        print(f"Erreur fatale: {e}")
        sys.exit(1)