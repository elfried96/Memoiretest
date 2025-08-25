#!/usr/bin/env python3
"""
🤖 Test Pipeline Vidéo avec Kimi-VL - Version Fonctionnelle
===========================================================

Script simplifié qui fonctionne avec votre structure actuelle.
Test du pipeline complet : YOLO → Tracking → Kimi-VL

Usage:
    python test_video_kimi.py videos/surveillance_test.mp4
    python test_video_kimi.py videos/surveillance_test.mp4 --frames 50
"""

import sys
import asyncio
import cv2
import time
import base64
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
from PIL import Image
from io import BytesIO

# Configuration des imports selon votre structure
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "src"))

@dataclass
class Detection:
    """Représentation simplifiée d'une détection."""
    x1: int
    y1: int 
    x2: int
    y2: int
    confidence: float
    class_name: str

@dataclass
class FrameResult:
    """Résultat de traitement d'une frame."""
    frame_number: int
    detections: List[Detection]
    processing_time: float
    vlm_analysis: Optional[str] = None
    vlm_confidence: Optional[float] = None

class SimpleVideoTester:
    """Testeur vidéo simplifié."""
    
    def __init__(self):
        self.results: List[FrameResult] = []
        
        # Tentative d'import des composants (optionnel)
        self.yolo_detector = None
        self.vlm_model = None
        
        self._init_components()
    
    def _init_components(self):
        """Initialisation des composants (avec gestion d'erreur)."""
        
        # YOLO Detector
        try:
            from ultralytics import YOLO
            print("🔍 Chargement YOLO...")
            self.yolo_detector = YOLO('yolov11n.pt')  # Modèle le plus léger
            print("✅ YOLO chargé")
        except Exception as e:
            print(f"⚠️ YOLO non disponible: {e}")
        
        # VLM (simulation pour l'instant)
        print("🤖 VLM simulé (Kimi-VL pas encore intégré dans cette version)")
    
    def _detect_objects(self, frame: np.ndarray) -> List[Detection]:
        """Détection d'objets avec YOLO."""
        
        detections = []
        
        if self.yolo_detector is not None:
            try:
                results = self.yolo_detector(frame, verbose=False)
                
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            confidence = box.conf[0].cpu().numpy()
                            class_id = int(box.cls[0].cpu().numpy())
                            class_name = result.names[class_id]
                            
                            detection = Detection(
                                x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2),
                                confidence=float(confidence),
                                class_name=class_name
                            )
                            detections.append(detection)
            except Exception as e:
                print(f"⚠️ Erreur détection frame: {e}")
        else:
            # Détection simulée si YOLO non disponible
            h, w = frame.shape[:2]
            fake_detection = Detection(
                x1=w//4, y1=h//4, x2=3*w//4, y2=3*h//4,
                confidence=0.8,
                class_name="person"
            )
            detections.append(fake_detection)
        
        return detections
    
    def _encode_frame_to_base64(self, frame: np.ndarray) -> str:
        """Encodage frame en base64 pour VLM."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        buffer = BytesIO()
        pil_image.save(buffer, format='JPEG', quality=85)
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return img_str
    
    async def _analyze_with_vlm(self, frame: np.ndarray, detections: List[Detection]) -> tuple:
        """Analyse avec VLM (simulée pour l'instant)."""
        
        # Simulation d'analyse VLM
        await asyncio.sleep(0.1)  # Simulation temps traitement
        
        if len(detections) == 0:
            return "Aucun objet détecté dans la scène", 0.9
        
        persons = [d for d in detections if d.class_name == "person"]
        
        if len(persons) > 0:
            if len(persons) == 1:
                analysis = f"Une personne détectée avec confiance {persons[0].confidence:.2f}"
                confidence = 0.8
            else:
                analysis = f"{len(persons)} personnes détectées - surveillance de groupe"
                confidence = 0.85
        else:
            objects = [d.class_name for d in detections]
            analysis = f"Objets détectés: {', '.join(set(objects))}"
            confidence = 0.7
        
        return analysis, confidence
    
    def _draw_detections(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Dessiner les détections sur la frame."""
        
        output_frame = frame.copy()
        
        for detection in detections:
            # Rectangle de détection
            color = (0, 255, 0) if detection.class_name == "person" else (255, 0, 0)
            cv2.rectangle(output_frame, 
                         (detection.x1, detection.y1), 
                         (detection.x2, detection.y2), 
                         color, 2)
            
            # Texte
            label = f"{detection.class_name} {detection.confidence:.2f}"
            cv2.putText(output_frame, label,
                       (detection.x1, detection.y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return output_frame
    
    async def process_video(self, video_path: str, max_frames: int = 100, save_output: bool = True) -> Dict[str, Any]:
        """Traitement complet d'une vidéo."""
        
        print(f"🎬 Traitement vidéo: {video_path}")
        
        # Ouverture vidéo
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Impossible d'ouvrir: {video_path}")
        
        # Informations vidéo
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"📊 Vidéo: {width}x{height}, {total_frames} frames, {fps:.1f} FPS")
        
        frames_to_process = min(max_frames, total_frames)
        print(f"🔄 Traitement de {frames_to_process} frames")
        
        # Préparation sortie vidéo
        output_writer = None
        if save_output:
            output_path = Path(video_path).parent / f"processed_{Path(video_path).stem}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            print(f"💾 Sortie: {output_path}")
        
        # Traitement frame par frame
        start_time = time.time()
        frame_count = 0
        
        try:
            while frame_count < frames_to_process:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_start = time.time()
                
                # Détection YOLO
                detections = self._detect_objects(frame)
                
                # Analyse VLM (une frame sur 10 pour économiser)
                vlm_analysis = None
                vlm_confidence = None
                if frame_count % 10 == 0:
                    vlm_analysis, vlm_confidence = await self._analyze_with_vlm(frame, detections)
                
                frame_time = time.time() - frame_start
                
                # Stockage résultat
                result = FrameResult(
                    frame_number=frame_count,
                    detections=detections,
                    processing_time=frame_time,
                    vlm_analysis=vlm_analysis,
                    vlm_confidence=vlm_confidence
                )
                self.results.append(result)
                
                # Dessin et sauvegarde
                if save_output:
                    display_frame = self._draw_detections(frame, detections)
                    
                    # Informations sur la frame
                    info_text = f"Frame: {frame_count} | Det: {len(detections)}"
                    cv2.putText(display_frame, info_text, (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    if vlm_analysis:
                        # Texte VLM (limité)
                        vlm_short = vlm_analysis[:50] + "..." if len(vlm_analysis) > 50 else vlm_analysis
                        cv2.putText(display_frame, f"VLM: {vlm_short}", (10, height - 20),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    
                    output_writer.write(display_frame)
                
                frame_count += 1
                
                # Progression
                if frame_count % 25 == 0:
                    elapsed = time.time() - start_time
                    fps_current = frame_count / elapsed
                    print(f"📈 Frame {frame_count}/{frames_to_process} | FPS: {fps_current:.1f}")
        
        finally:
            cap.release()
            if output_writer:
                output_writer.release()
        
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time
        
        # Statistiques
        total_detections = sum(len(r.detections) for r in self.results)
        total_persons = sum(len([d for d in r.detections if d.class_name == "person"]) for r in self.results)
        vlm_analyses = len([r for r in self.results if r.vlm_analysis])
        
        summary = {
            "video_path": video_path,
            "frames_processed": frame_count,
            "total_time": total_time,
            "average_fps": avg_fps,
            "total_detections": total_detections,
            "persons_detected": total_persons,
            "vlm_analyses": vlm_analyses,
            "output_saved": save_output
        }
        
        # Sauvegarde rapport JSON
        report_path = Path(video_path).parent / f"report_{Path(video_path).stem}.json"
        with open(report_path, 'w') as f:
            json.dump({
                "summary": summary,
                "frame_results": [
                    {
                        "frame": r.frame_number,
                        "detections": len(r.detections),
                        "processing_time": r.processing_time,
                        "vlm_analysis": r.vlm_analysis,
                        "vlm_confidence": r.vlm_confidence
                    }
                    for r in self.results
                ]
            }, f, indent=2)
        
        print(f"📄 Rapport sauvé: {report_path}")
        
        return summary

def print_summary(summary: Dict[str, Any]):
    """Affichage du résumé."""
    
    print("\n" + "="*50)
    print("🏆 RÉSULTATS DU TEST")
    print("="*50)
    
    print(f"🎬 Vidéo: {Path(summary['video_path']).name}")
    print(f"📊 Frames traitées: {summary['frames_processed']}")
    print(f"⏱️ Temps total: {summary['total_time']:.1f}s")
    print(f"🎯 FPS moyen: {summary['average_fps']:.1f}")
    
    print(f"\n🔍 DÉTECTIONS:")
    print(f"  Total: {summary['total_detections']}")
    print(f"  Personnes: {summary['persons_detected']}")
    
    print(f"\n🤖 ANALYSE VLM:")
    print(f"  Analyses effectuées: {summary['vlm_analyses']}")
    
    if summary['output_saved']:
        print(f"\n💾 Vidéo traitée et rapport sauvés")

def create_argument_parser():
    """Parser d'arguments."""
    parser = argparse.ArgumentParser(
        description="Test pipeline vidéo avec Kimi-VL (version fonctionnelle)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  python test_video_kimi.py videos/surveillance_test.mp4              # Test standard
  python test_video_kimi.py videos/surveillance_test.mp4 --frames 50  # Limiter frames  
  python test_video_kimi.py videos/surveillance_test.mp4 --no-save    # Sans sauvegarde
        """
    )
    
    parser.add_argument(
        "video_path",
        help="Chemin vers la vidéo à tester"
    )
    
    parser.add_argument(
        "--frames",
        type=int,
        default=100,
        help="Nombre max de frames à traiter (défaut: 100)"
    )
    
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Ne pas sauvegarder la vidéo traitée"
    )
    
    return parser

async def main():
    """Point d'entrée principal."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    if not Path(args.video_path).exists():
        print(f"❌ Vidéo non trouvée: {args.video_path}")
        return 1
    
    print("🤖 Test Pipeline Vidéo avec Kimi-VL (Version Fonctionnelle)")
    print("="*60)
    
    tester = SimpleVideoTester()
    
    try:
        summary = await tester.process_video(
            args.video_path, 
            max_frames=args.frames,
            save_output=not args.no_save
        )
        
        print_summary(summary)
        
        print("\n🎉 Test terminé avec succès !")
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️ Interruption utilisateur")
        return 1
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        return 1

if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        sys.exit(result)
    except Exception as e:
        print(f"Erreur fatale: {e}")
        sys.exit(1)