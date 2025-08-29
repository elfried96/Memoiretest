#!/usr/bin/env python3
"""
🎯 Mode Simple - YOLO + Tracking seulement 
==========================================
Version simplifiée sans outils avancés pour tests rapides
"""

import argparse
import asyncio
import time
import cv2
import json
import numpy as np
from pathlib import Path
from typing import List

# Imports simplifiés
from src.detection.yolo_detector import YOLODetector
from src.detection.tracking.byte_tracker import BYTETracker
from src.core.types import DetectedObject, BoundingBox


class SimpleSurveillanceSystem:
    """Système de surveillance simplifié - YOLO + Tracking uniquement."""
    
    def __init__(self, source="webcam", max_frames=None):
        self.source = source
        self.max_frames = max_frames
        self.frame_count = 0
        self.results = []
        
        # Initialisation YOLO + Tracker
        print("🔧 Initialisation système simple...")
        self.detector = YOLODetector()
        self.tracker = BYTETracker()
        print("✅ Système simple prêt !")
    
    def process_frame(self, frame):
        """Traite un frame avec YOLO + tracking."""
        start_time = time.time()
        
        # Détection YOLO
        detections = []
        yolo_results = self.detector.detect(frame)
        
        # Conversion en DetectedObject
        for result in yolo_results:
            if hasattr(result, 'boxes') and result.boxes:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_name = result.names.get(cls, f"class_{cls}")
                    
                    detection = DetectedObject(
                        class_id=cls,
                        class_name=class_name,
                        bbox=BoundingBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2)),
                        confidence=conf,
                        track_id=None
                    )
                    detections.append(detection)
        
        # Tracking
        tracked_objects = self.tracker.update(detections)
        
        processing_time = time.time() - start_time
        
        # Résultat simple
        result = {
            "frame_id": self.frame_count,
            "timestamp": time.time(),
            "detections_count": len(detections),
            "tracked_objects_count": len(tracked_objects),
            "processing_time": processing_time,
            "detections": [
                {
                    "class_name": det.class_name,
                    "confidence": det.confidence,
                    "bbox": [det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2]
                } for det in detections
            ]
        }
        
        self.results.append(result)
        
        # Log périodique
        if self.frame_count % 30 == 0:
            print(f"📊 Frame {self.frame_count}: {len(detections)} détections, "
                  f"{len(tracked_objects)} trackés, {processing_time:.3f}s")
        
        return result
    
    async def run_surveillance(self):
        """Lance la surveillance simplifiée."""
        
        # Ouverture source vidéo
        if self.source == "webcam":
            cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(self.source)
        
        if not cap.isOpened():
            print(f"❌ Impossible d'ouvrir: {self.source}")
            return
        
        print(f"📹 Source ouverte: {self.source}")
        print("🚀 Démarrage surveillance simple...")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("📺 Fin de vidéo ou erreur lecture")
                    break
                
                self.frame_count += 1
                
                # Traitement frame
                result = self.process_frame(frame)
                
                # Limite frames si spécifiée
                if self.max_frames and self.frame_count >= self.max_frames:
                    print(f"🏁 Limite {self.max_frames} frames atteinte")
                    break
        
        except KeyboardInterrupt:
            print("🛑 Arrêt par utilisateur")
        
        finally:
            cap.release()
            
            # Sauvegarde résultats
            self.save_results()
            self.print_statistics()
    
    def save_results(self):
        """Sauvegarde résultats JSON."""
        output_file = f"simple_surveillance_{int(time.time())}.json"
        
        summary = {
            "total_frames": len(self.results),
            "total_detections": sum(r["detections_count"] for r in self.results),
            "avg_processing_time": sum(r["processing_time"] for r in self.results) / len(self.results) if self.results else 0,
            "source": self.source,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        output_data = {
            "summary": summary,
            "results": self.results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Résultats sauvés: {output_file}")
    
    def print_statistics(self):
        """Affiche statistiques finales."""
        if not self.results:
            return
        
        total_detections = sum(r["detections_count"] for r in self.results)
        avg_time = sum(r["processing_time"] for r in self.results) / len(self.results)
        fps = 1 / avg_time if avg_time > 0 else 0
        
        print("\n" + "="*50)
        print("📊 STATISTIQUES SURVEILLANCE SIMPLE")
        print("="*50)
        print(f"Frames traités: {len(self.results)}")
        print(f"Total détections: {total_detections}")
        print(f"Temps moyen/frame: {avg_time:.3f}s")
        print(f"FPS moyen: {fps:.1f}")
        print("="*50)


async def main():
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(description="🎯 Surveillance Simple - YOLO + Tracking")
    parser.add_argument("--video", "-v", default="webcam", help="Source vidéo")
    parser.add_argument("--max-frames", type=int, help="Limite frames")
    
    args = parser.parse_args()
    
    print("🎯 SURVEILLANCE SIMPLE - YOLO + TRACKING")
    print("="*50)
    print(f"📹 Source: {args.video}")
    print(f"🎬 Max frames: {args.max_frames or 'illimité'}")
    print()
    
    system = SimpleSurveillanceSystem(
        source=args.video,
        max_frames=args.max_frames
    )
    
    await system.run_surveillance()


if __name__ == "__main__":
    asyncio.run(main())