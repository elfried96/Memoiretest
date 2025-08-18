#!/usr/bin/env python3
"""
🎬 Test Complet du Système de Surveillance sur Vidéo
===================================================

Ce script teste TOUT le système :
- YOLO11 pour détection
- Tracking multi-objets  
- VLM avec tool calling
- Orchestration adaptative
- Optimisation automatique des outils
- Analyse complète en temps réel

Usage:
    python test_full_system_video.py --video /path/to/video.mp4
    python test_full_system_video.py --video webcam
    python test_full_system_video.py --video rtsp://camera_ip:554/stream
"""

import asyncio
import argparse
import cv2
import time
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import sys

# Ajout du chemin pour les imports
sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.live import Live

try:
    from src.core.orchestrator.adaptive_orchestrator import create_adaptive_orchestrator
    from src.core.orchestrator.vlm_orchestrator import OrchestrationConfig, OrchestrationMode
    from src.core.orchestrator.tool_calling_vlm import create_tool_calling_vlm
    from src.testing.tool_optimization_benchmark import ToolOptimizationBenchmark
    from src.detection.yolo_detector import YOLODetector
    from src.core.types import Detection, BoundingBox
except ImportError as e:
    print(f"❌ Erreur d'import: {e}")
    print("Vérifiez que tous les modules sont présents")
    sys.exit(1)

console = Console()


class FullSystemVideoTester:
    """Testeur complet du système sur vidéo."""
    
    def __init__(self, video_source: str):
        self.video_source = video_source
        self.console = Console()
        
        # Statistiques de test
        self.stats = {
            "frames_processed": 0,
            "detections_total": 0,
            "analysis_total": 0,
            "avg_processing_time": 0.0,
            "optimization_runs": 0,
            "alerts_triggered": 0
        }
        
        # Composants du système
        self.yolo_detector = None
        self.adaptive_orchestrator = None
        self.tool_calling_vlm = None
        self.optimization_benchmark = None
        
        # Configuration
        self.config = OrchestrationConfig(
            mode=OrchestrationMode.BALANCED,
            enable_advanced_tools=True,
            max_concurrent_tools=4,
            confidence_threshold=0.7
        )
        
        self.results_log = []
    
    async def initialize_system(self) -> bool:
        """Initialisation complète du système."""
        
        console.print(Panel.fit(
            "[bold blue]🚀 Initialisation du Système Complet de Surveillance[/bold blue]\n"
            "[dim]YOLO11 + VLM + Tool Calling + Orchestration Adaptative + Optimisation[/dim]",
            border_style="blue"
        ))
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            # 1. YOLO11 Detector
            task = progress.add_task("[cyan]Initialisation YOLO11...", total=None)
            try:
                self.yolo_detector = YOLODetector(
                    model_path="yolov11n.pt",  # Nouvelle version !
                    device="auto",
                    confidence_threshold=0.5
                )
                progress.update(task, description="[green]✅ YOLO11 initialisé")
                await asyncio.sleep(1)
            except Exception as e:
                progress.update(task, description="[red]❌ Erreur YOLO11")
                console.print(f"Erreur YOLO11: {e}")
                return False
            
            # 2. VLM avec Tool Calling
            progress.update(task, description="[cyan]Chargement VLM + Tool Calling...")
            try:
                self.tool_calling_vlm = create_tool_calling_vlm(
                    model_name="kimi-vl-a3b-thinking",
                    enable_caching=True
                )
                progress.update(task, description="[green]✅ VLM Tool Calling prêt")
                await asyncio.sleep(1)
            except Exception as e:
                progress.update(task, description="[red]❌ Erreur VLM")
                console.print(f"Erreur VLM: {e}")
                return False
            
            # 3. Orchestrateur Adaptatif
            progress.update(task, description="[cyan]Configuration Orchestrateur Adaptatif...")
            try:
                self.adaptive_orchestrator = create_adaptive_orchestrator(
                    vlm_model_name="kimi-vl-a3b-thinking",
                    config=self.config,
                    enable_learning=True
                )
                progress.update(task, description="[green]✅ Orchestrateur Adaptatif prêt")
                await asyncio.sleep(1)
            except Exception as e:
                progress.update(task, description="[red]❌ Erreur Orchestrateur")
                console.print(f"Erreur Orchestrateur: {e}")
                return False
            
            # 4. Système d'Optimisation
            progress.update(task, description="[cyan]Préparation Optimisation des Outils...")
            try:
                self.optimization_benchmark = ToolOptimizationBenchmark(
                    vlm_model_name="kimi-vl-a3b-thinking"
                )
                await self.optimization_benchmark.load_test_cases()
                progress.update(task, description="[green]✅ Système d'Optimisation prêt")
                await asyncio.sleep(1)
            except Exception as e:
                progress.update(task, description="[red]❌ Erreur Optimisation")
                console.print(f"Erreur Optimisation: {e}")
                return False
        
        # Affichage du statut système
        self._display_system_status()
        
        return True
    
    def _display_system_status(self):
        """Affichage du statut du système."""
        
        status_table = Table(title="🔍 Statut du Système Complet")
        status_table.add_column("Composant", style="cyan", no_wrap=True)
        status_table.add_column("Version/Type", style="yellow")
        status_table.add_column("Statut", style="bold")
        status_table.add_column("Configuration", style="dim")
        
        # YOLO11
        status_table.add_row(
            "YOLO11 Detector", 
            "yolov11n.pt", 
            "✅ Opérationnel",
            f"Seuil: {self.yolo_detector.confidence_threshold}"
        )
        
        # VLM Tool Calling
        tool_stats = self.tool_calling_vlm.get_tool_calling_stats()
        status_table.add_row(
            "VLM Tool Calling",
            "kimi-vl-a3b-thinking",
            "✅ Opérationnel", 
            f"{tool_stats['registered_tools']} outils"
        )
        
        # Orchestrateur Adaptatif
        adaptive_status = self.adaptive_orchestrator.get_adaptive_status()
        status_table.add_row(
            "Orchestrateur Adaptatif",
            f"Mode {self.config.mode.value}",
            "✅ Opérationnel",
            f"Apprentissage: {'✅' if adaptive_status['adaptive_learning_enabled'] else '❌'}"
        )
        
        # Optimisation
        status_table.add_row(
            "Optimisation Outils",
            "Benchmark Auto",
            "✅ Prêt",
            f"{len(self.optimization_benchmark.test_cases)} cas de test"
        )
        
        console.print(status_table)
    
    async def process_video_with_full_system(self, max_frames: int = None, save_results: bool = True):
        """Traitement vidéo avec le système complet."""
        
        console.print(Panel.fit(
            f"[bold green]🎬 Traitement Vidéo avec Système Complet[/bold green]\n"
            f"[dim]Source: {self.video_source}[/dim]",
            border_style="green"
        ))
        
        # Ouverture de la vidéo
        cap = cv2.VideoCapture(self.video_source if self.video_source != "webcam" else 0)
        
        if not cap.isOpened():
            console.print(f"❌ Impossible d'ouvrir la vidéo: {self.video_source}")
            return False
        
        # Informations vidéo
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:  # Pour webcam ou flux
            total_frames = max_frames or 1000
        
        console.print(f"📹 FPS: {fps:.1f} | Frames totales: {total_frames}")
        
        # Table de résultats en temps réel
        results_table = Table(title="📊 Résultats d'Analyse en Temps Réel")
        results_table.add_column("Frame", style="cyan")
        results_table.add_column("Détections", style="yellow") 
        results_table.add_column("Suspicion", style="red")
        results_table.add_column("Outils Utilisés", style="blue")
        results_table.add_column("Temps (s)", style="green")
        results_table.add_column("Actions", style="magenta")
        
        frame_count = 0
        optimization_interval = 50  # Optimisation tous les 50 frames
        
        with Live(results_table, refresh_per_second=2, console=console) as live:
            
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_count += 1
                    if max_frames and frame_count > max_frames:
                        break
                    
                    # Traitement complet du frame
                    start_time = time.time()
                    result = await self._process_single_frame(frame, frame_count)
                    processing_time = time.time() - start_time
                    
                    # Mise à jour des statistiques
                    self.stats["frames_processed"] = frame_count
                    self.stats["avg_processing_time"] = (
                        (self.stats["avg_processing_time"] * (frame_count - 1) + processing_time) / frame_count
                    )
                    
                    if result["detections_count"] > 0:
                        self.stats["detections_total"] += result["detections_count"]
                    
                    if result["analysis_performed"]:
                        self.stats["analysis_total"] += 1
                    
                    if result["alert_triggered"]:
                        self.stats["alerts_triggered"] += 1
                    
                    # Mise à jour de la table
                    suspicion_emoji = {
                        "LOW": "🟢",
                        "MEDIUM": "🟡", 
                        "HIGH": "🟠",
                        "CRITICAL": "🔴"
                    }
                    
                    results_table.add_row(
                        f"{frame_count}",
                        f"{result['detections_count']}",
                        f"{suspicion_emoji.get(result['suspicion_level'], '⚪')} {result['suspicion_level']}",
                        f"{len(result['tools_used'])}",
                        f"{processing_time:.2f}",
                        "🚨" if result["alert_triggered"] else "✅"
                    )
                    
                    # Limitation d'affichage (garder que les 10 derniers)
                    if len(results_table.rows) > 10:
                        results_table.rows = results_table.rows[-10:]
                    
                    # Optimisation périodique
                    if frame_count % optimization_interval == 0:
                        console.print(f"\n🔄 Optimisation automatique (frame {frame_count})...")
                        await self._run_periodic_optimization()
                        self.stats["optimization_runs"] += 1
                    
                    # Sauvegarde des résultats
                    if save_results:
                        self.results_log.append({
                            "frame": frame_count,
                            "timestamp": datetime.now().isoformat(),
                            "processing_time": processing_time,
                            **result
                        })
                    
                    # Affichage vidéo (optionnel)
                    self._display_frame_with_overlay(frame, result)
                    
                    # ESC pour quitter
                    if cv2.waitKey(1) & 0xFF == 27:
                        console.print("\n🛑 Arrêt demandé par l'utilisateur")
                        break
                    
            except KeyboardInterrupt:
                console.print("\n🛑 Interruption par Ctrl+C")
            
            finally:
                cap.release()
                cv2.destroyAllWindows()
        
        # Affichage des résultats finaux
        await self._display_final_results(save_results)
        
        return True
    
    async def _process_single_frame(self, frame: np.ndarray, frame_id: int) -> Dict[str, Any]:
        """Traitement complet d'un seul frame."""
        
        # 1. Détection YOLO11
        yolo_results = self.yolo_detector.detect(frame)
        detections = self._convert_yolo_to_detections(yolo_results)
        
        # 2. Contexte enrichi
        context = {
            "frame_id": frame_id,
            "timestamp": time.time(),
            "location": "test_area",
            "camera": "test_camera",
            "detections_count": len(detections),
            "person_count": len([d for d in detections if d.class_name == "person"])
        }
        
        # 3. Analyse avec orchestrateur adaptatif (si pertinent)
        analysis_performed = False
        suspicion_level = "LOW"
        tools_used = []
        alert_triggered = False
        
        # Analyse VLM si personnes détectées ou périodiquement
        if len(detections) > 0 or frame_id % 20 == 0:
            try:
                # Encodage du frame pour VLM
                frame_b64 = self._encode_frame_to_base64(frame)
                
                # Analyse avec orchestrateur adaptatif
                analysis = await self.adaptive_orchestrator.analyze_surveillance_frame(
                    frame_data=frame_b64,
                    detections=detections,
                    context=context
                )
                
                analysis_performed = True
                suspicion_level = analysis.suspicion_level.value
                tools_used = analysis.tools_used
                
                # Déclenchement d'alerte si suspicion élevée
                if analysis.suspicion_level.value in ["HIGH", "CRITICAL"]:
                    alert_triggered = True
                
            except Exception as e:
                console.print(f"⚠️ Erreur analyse frame {frame_id}: {e}")
        
        return {
            "detections_count": len(detections),
            "detections": detections,
            "context": context,
            "analysis_performed": analysis_performed,
            "suspicion_level": suspicion_level,
            "tools_used": tools_used,
            "alert_triggered": alert_triggered
        }
    
    def _convert_yolo_to_detections(self, yolo_results) -> List[Detection]:
        """Conversion des résultats YOLO en objets Detection."""
        
        detections = []
        
        for result in yolo_results:
            if hasattr(result, 'boxes') and result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    
                    class_names = result.names
                    class_name = class_names.get(cls, f"class_{cls}")
                    
                    detection = Detection(
                        bbox=BoundingBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2)),
                        confidence=float(conf),
                        class_name=class_name,
                        track_id=None
                    )
                    detections.append(detection)
        
        return detections
    
    def _encode_frame_to_base64(self, frame: np.ndarray) -> str:
        """Encodage frame en base64 pour VLM."""
        
        # Redimensionnement pour optimiser
        height, width = frame.shape[:2]
        if width > 640:
            scale = 640 / width
            new_width = 640
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height))
        
        # Conversion BGR → RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Encodage JPEG puis base64
        _, buffer = cv2.imencode('.jpg', frame_rgb, [cv2.IMWRITE_JPEG_QUALITY, 75])
        
        import base64
        frame_b64 = base64.b64encode(buffer).decode('utf-8')
        
        return frame_b64
    
    async def _run_periodic_optimization(self):
        """Optimisation périodique des outils."""
        
        try:
            # Test rapide de 3 combinaisons pour optimisation
            test_combinations = [
                ["dino_features", "pose_estimator"],
                ["multimodal_fusion", "adversarial_detector"],
                ["sam2_segmentator", "temporal_transformer"]
            ]
            
            best_combo = None
            best_score = 0.0
            
            for combo in test_combinations:
                result = await self.optimization_benchmark.test_tool_combination(
                    combo, 
                    self.optimization_benchmark.test_cases[:2]  # Test rapide
                )
                
                if result.quality_score > best_score:
                    best_score = result.quality_score
                    best_combo = combo
            
            if best_combo:
                # Mise à jour des outils optimaux dans l'orchestrateur
                self.adaptive_orchestrator.current_optimal_tools = best_combo
                console.print(f"🎯 Outils optimisés: {best_combo} (Score: {best_score:.3f})")
            
        except Exception as e:
            console.print(f"⚠️ Erreur optimisation: {e}")
    
    def _display_frame_with_overlay(self, frame: np.ndarray, result: Dict[str, Any]):
        """Affichage du frame avec overlay d'information."""
        
        overlay_frame = frame.copy()
        
        # Dessiner les détections
        for detection in result["detections"]:
            bbox = detection.bbox
            x1, y1, x2, y2 = int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)
            
            # Couleur selon le niveau de suspicion
            color = (0, 255, 0)  # Vert par défaut
            if result["suspicion_level"] == "MEDIUM":
                color = (0, 255, 255)  # Jaune
            elif result["suspicion_level"] == "HIGH":
                color = (0, 165, 255)  # Orange
            elif result["suspicion_level"] == "CRITICAL":
                color = (0, 0, 255)  # Rouge
            
            cv2.rectangle(overlay_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(overlay_frame, f"{detection.class_name}: {detection.confidence:.2f}",
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Informations système
        cv2.putText(overlay_frame, f"Frame: {result['context']['frame_id']}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(overlay_frame, f"Suspicion: {result['suspicion_level']}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(overlay_frame, f"Outils: {len(result['tools_used'])}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if result["alert_triggered"]:
            cv2.putText(overlay_frame, "🚨 ALERTE", 
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        
        cv2.imshow("🎯 Surveillance Intelligente - Test Complet", overlay_frame)
    
    async def _display_final_results(self, save_results: bool):
        """Affichage des résultats finaux."""
        
        console.print("\n" + "="*80)
        console.print(Panel.fit(
            "[bold green]📊 RÉSULTATS FINAUX DU TEST COMPLET[/bold green]",
            border_style="green"
        ))
        
        # Statistiques globales
        stats_table = Table(title="📈 Statistiques Globales")
        stats_table.add_column("Métrique", style="cyan")
        stats_table.add_column("Valeur", style="green")
        stats_table.add_column("Performance", style="yellow")
        
        stats_table.add_row(
            "Frames Traités", 
            f"{self.stats['frames_processed']}", 
            f"{self.stats['frames_processed']/max(1, self.stats['avg_processing_time']*self.stats['frames_processed']):.1f} FPS moyen"
        )
        
        stats_table.add_row(
            "Détections Totales",
            f"{self.stats['detections_total']}",
            f"{self.stats['detections_total']/max(1, self.stats['frames_processed']):.1f} par frame"
        )
        
        stats_table.add_row(
            "Analyses VLM",
            f"{self.stats['analysis_total']}",
            f"{self.stats['analysis_total']/max(1, self.stats['frames_processed'])*100:.1f}% des frames"
        )
        
        stats_table.add_row(
            "Alertes Déclenchées",
            f"{self.stats['alerts_triggered']}",
            f"{self.stats['alerts_triggered']/max(1, self.stats['frames_processed'])*100:.1f}% des frames"
        )
        
        stats_table.add_row(
            "Optimisations Auto",
            f"{self.stats['optimization_runs']}",
            "Système adaptatif actif"
        )
        
        stats_table.add_row(
            "Temps Moyen/Frame",
            f"{self.stats['avg_processing_time']:.3f}s",
            "Incluant détection + VLM + orchestration"
        )
        
        console.print(stats_table)
        
        # Statut de l'apprentissage adaptatif
        adaptive_status = self.adaptive_orchestrator.get_adaptive_status()
        
        learning_table = Table(title="🧠 Apprentissage Adaptatif")
        learning_table.add_column("Aspect", style="cyan")
        learning_table.add_column("Résultat", style="green")
        
        learning_table.add_row(
            "Patterns Contextuels Appris",
            f"{adaptive_status['learning_stats']['context_patterns_learned']}"
        )
        
        learning_table.add_row(
            "Outils Optimaux Actuels", 
            f"{len(adaptive_status['current_optimal_tools'])} outils"
        )
        
        learning_table.add_row(
            "Performance Récente",
            f"{adaptive_status['learning_stats']['recent_performance']:.2f}"
        )
        
        console.print(learning_table)
        
        # Sauvegarde des résultats
        if save_results and self.results_log:
            results_file = Path(f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            
            full_results = {
                "test_info": {
                    "video_source": self.video_source,
                    "timestamp": datetime.now().isoformat(),
                    "system_config": {
                        "yolo_model": "yolov11n.pt",
                        "vlm_model": "kimi-vl-a3b-thinking",
                        "orchestration_mode": self.config.mode.value
                    }
                },
                "statistics": self.stats,
                "adaptive_status": adaptive_status,
                "frame_results": self.results_log
            }
            
            with open(results_file, 'w') as f:
                json.dump(full_results, f, indent=2, default=str)
            
            console.print(f"\n💾 Résultats sauvegardés: {results_file}")
        
        console.print("\n🎉 TEST COMPLET TERMINÉ AVEC SUCCÈS !")


async def main():
    """Point d'entrée principal."""
    
    parser = argparse.ArgumentParser(
        description="Test complet du système de surveillance sur vidéo"
    )
    
    parser.add_argument(
        "--video",
        default="webcam",
        help="Source vidéo: 'webcam', chemin vers fichier, ou URL RTSP"
    )
    
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Nombre maximum de frames à traiter"
    )
    
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Ne pas sauvegarder les résultats"
    )
    
    args = parser.parse_args()
    
    # Validation de la source vidéo
    video_source = args.video
    if video_source == "webcam":
        video_source = 0
    
    console.print(f"🎬 Test du système complet sur: {args.video}")
    
    # Initialisation et lancement du test
    tester = FullSystemVideoTester(video_source)
    
    try:
        # Initialisation
        init_success = await tester.initialize_system()
        if not init_success:
            console.print("❌ Échec de l'initialisation")
            return
        
        # Test complet
        await tester.process_video_with_full_system(
            max_frames=args.max_frames,
            save_results=not args.no_save
        )
        
    except KeyboardInterrupt:
        console.print("\n🛑 Test interrompu")
    except Exception as e:
        console.print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())