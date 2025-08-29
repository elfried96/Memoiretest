#!/usr/bin/env python3
"""
🎯 SYSTÈME DE SURVEILLANCE HEADLESS - Version Corrigée
====================================================
Version sans outils avancés pour éviter les erreurs
"""

import argparse
import asyncio
import time
import cv2
import json
import numpy as np
from pathlib import Path
from typing import List
from dataclasses import dataclass, asdict

# Imports principaux
from src.detection.yolo_detector import YOLODetector
from src.detection.tracking.byte_tracker import BYTETracker
from src.core.types import DetectedObject, BoundingBox


@dataclass
class SurveillanceResult:
    """Résultat d'analyse d'un frame."""
    frame_id: int
    timestamp: float
    detections_count: int
    persons_detected: int
    alert_level: str
    actions_taken: List[str]
    processing_time: float
    vlm_analysis: dict = None


class HeadlessSurveillanceSystemFixed:
    """Système de surveillance headless corrigé - sans outils avancés."""
    
    def __init__(self, source="webcam", vlm_model="none", mode="SIMPLE", 
                 save_frames=False, output_dir="surveillance_output"):
        self.source = source
        self.vlm_model = vlm_model
        self.mode = mode
        self.save_frames = save_frames
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
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
        
        # Initialisation composants principaux
        print("🔧 Initialisation système corrigé...")
        self.detector = YOLODetector()
        self.tracker = BYTETracker()
        print("✅ Système corrigé prêt !")
    
    def create_detections_list(self, yolo_results) -> List[DetectedObject]:
        """Convertit résultats YOLO en DetectedObjects."""
        detections = []
        
        for result in yolo_results:
            if hasattr(result, 'boxes') and result.boxes:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    
                    class_names = result.names
                    class_name = class_names.get(cls, f"class_{cls}")
                    
                    detection = DetectedObject(
                        class_id=cls,
                        class_name=class_name,
                        bbox=BoundingBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2)),
                        confidence=conf,
                        track_id=None
                    )
                    detections.append(detection)
        
        return detections
    
    async def process_frame(self, frame):
        """Traite un frame - version simplifiée."""
        start_time = time.time()
        
        # 1. Détection YOLO
        yolo_results = self.detector.detect(frame)
        detections = self.create_detections_list(yolo_results)
        
        # 2. Tracking
        tracked_objects = self.tracker.update(detections)
        
        # 3. Analyse simple (sans VLM)
        persons_count = sum(1 for det in detections if det.class_name == "person")
        
        # 4. Niveau d'alerte simple
        if persons_count > 3:
            alert_level = "high"
        elif persons_count > 1:
            alert_level = "medium"  
        else:
            alert_level = "normal"
        
        actions_taken = []
        if persons_count > 2:
            actions_taken.append("surveillance_increased")
        
        processing_time = time.time() - start_time
        
        # 5. Création résultat
        result = SurveillanceResult(
            frame_id=self.frame_count,
            timestamp=time.time(),
            detections_count=len(detections),
            persons_detected=persons_count,
            alert_level=alert_level,
            actions_taken=actions_taken,
            processing_time=processing_time
        )
        
        # 6. Mise à jour statistiques
        self.processing_stats["total_frames"] += 1
        self.processing_stats["detected_objects"] += len(detections)
        self.processing_stats["persons_detected"] += persons_count
        self.processing_stats["total_processing_time"] += processing_time
        
        if self.processing_stats["total_frames"] > 0:
            avg_time = self.processing_stats["total_processing_time"] / self.processing_stats["total_frames"]
            self.processing_stats["average_fps"] = 1.0 / avg_time if avg_time > 0 else 0
        
        if alert_level in ["medium", "high"]:
            self.processing_stats["alerts_triggered"] += 1
        
        # 7. Logging
        if persons_count > 0 or alert_level != "normal":
            print(f"📊 Frame {self.frame_count}: "
                  f"{len(detections)} objs, {persons_count} personnes, "
                  f"Alert: {alert_level}, Actions: {actions_taken}, "
                  f"Temps: {processing_time:.2f}s")
        elif self.frame_count % 60 == 0:
            fps = self.processing_stats['average_fps']
            total_objs = self.processing_stats['detected_objects']
            print(f"📈 Frame {self.frame_count}: FPS: {fps:.1f}, Total objets: {total_objs}")
        
        self.results_log.append(result)
        return result
    
    async def run_surveillance(self, max_frames=None):
        """Lance la surveillance."""
        
        # Ouverture source vidéo
        if self.source == "webcam":
            cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(self.source)
        
        if not cap.isOpened():
            print(f"❌ Impossible d'ouvrir source vidéo: {self.source}")
            return
        
        print(f"📹 Source vidéo ouverte: {self.source}")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("📺 Fin de vidéo")
                    break
                
                self.frame_count += 1
                
                # Traitement frame
                result = await self.process_frame(frame)
                
                # Limite frames
                if max_frames and self.frame_count >= max_frames:
                    print(f"🏁 Limite de {max_frames} frames atteinte")
                    break
        
        except KeyboardInterrupt:
            print("🛑 Arrêt par interruption clavier")
        
        finally:
            cap.release()
            
            # Sauvegarde et statistiques
            self.save_results_to_json()
            self.print_final_statistics()
    
    def save_results_to_json(self):
        """Sauvegarde résultats JSON - version corrigée."""
        results_file = self.output_dir / f"surveillance_results_{int(time.time())}.json"
        
        session_info = {
            "source": self.source,
            "vlm_model": self.vlm_model,
            "mode": self.mode,
            "total_frames": len(self.results_log),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Conversion sécurisée des résultats
        serialized_results = []
        for result in self.results_log:
            try:
                result_dict = asdict(result)
                serialized_results.append(result_dict)
            except Exception as e:
                # Fallback manual
                serialized_results.append({
                    "frame_id": result.frame_id,
                    "timestamp": result.timestamp,
                    "detections_count": result.detections_count,
                    "persons_detected": result.persons_detected,
                    "alert_level": result.alert_level,
                    "actions_taken": result.actions_taken,
                    "processing_time": result.processing_time
                })
        
        output_data = {
            "session_info": session_info,
            "statistics": self.processing_stats,
            "results": serialized_results
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Résultats sauvés: {results_file}")
    
    def print_final_statistics(self):
        """Affiche statistiques finales."""
        stats = self.processing_stats
        
        print("=" * 60)
        print("📈 STATISTIQUES FINALES DE SURVEILLANCE")
        print("=" * 60)
        
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"   • {key}: {value:.2f}")
            else:
                print(f"   • {key}: {value}")
        
        print("=" * 60)


async def main():
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(
        description="🎯 Système de Surveillance Headless Corrigé"
    )
    
    parser.add_argument("--video", "-v", default="webcam", help="Source vidéo")
    parser.add_argument("--model", "-m", default="none", help="Modèle VLM (désactivé)")
    parser.add_argument("--mode", default="SIMPLE", help="Mode (SIMPLE uniquement)")
    parser.add_argument("--max-frames", type=int, help="Nombre max de frames")
    parser.add_argument("--save-frames", action="store_true", help="Sauvegarder frames")
    parser.add_argument("--no-save", action="store_true", help="Ne pas sauvegarder")
    
    args = parser.parse_args()
    
    print("🎯 SYSTÈME DE SURVEILLANCE HEADLESS CORRIGÉ")
    print("=" * 50)
    print(f"📹 Source vidéo  : {args.video}")
    print(f"🤖 Modèle VLM    : {args.model} (désactivé)")
    print(f"⚙️ Mode          : {args.mode}")
    print(f"💾 Sauvegarde    : {'Désactivée' if args.no_save else 'Activée'}")
    print(f"🖼️ Frames        : {'Sauvées' if args.save_frames else 'Non sauvées'}")
    print(f"📊 Max frames    : {args.max_frames or 'Illimité'}")
    print()
    print("WORKFLOW CORRIGÉ:")
    print("1. 📹 Capture vidéo → logs détaillés")
    print("2. 🔍 Détection YOLO → comptage objets")
    print("3. 🏃 Tracking → suivi objets")
    print("4. ⚡ Analyse simple → évaluation basique")
    print("5. 💾 Export JSON → résultats structurés")
    print("6. 📊 Statistiques finales → rapport complet")
    print()
    
    system = HeadlessSurveillanceSystemFixed(
        source=args.video,
        vlm_model=args.model,
        mode=args.mode,
        save_frames=args.save_frames
    )
    
    await system.run_surveillance(max_frames=args.max_frames)


if __name__ == "__main__":
    asyncio.run(main())