#!/usr/bin/env python3
"""Test simple d'analyse vidéo avec VLM."""

import cv2
import asyncio
import sys
from pathlib import Path
from PIL import Image
import numpy as np
import argparse
from datetime import datetime

# Path
sys.path.append(str(Path(__file__).parent / "src"))

from src.core.vlm.dynamic_model import DynamicVisionLanguageModel
from src.core.types import AnalysisRequest


class VideoAnalyzer:
    """Analyseur vidéo simple avec VLM."""
    
    def __init__(self, model_name: str = "kimi-vl-a3b-thinking"):
        self.vlm = DynamicVisionLanguageModel(
            default_model=model_name,
            enable_fallback=False
        )
        self.results = []
    
    async def setup(self):
        """Initialiser le VLM."""
        print(f"⏳ Chargement du modèle VLM...")
        success = await self.vlm.switch_model(self.vlm.model_registry.get_recommended_model("surveillance"))
        if success:
            print(f"✅ VLM chargé: {self.vlm.current_model_id}")
            return True
        else:
            print("❌ Échec du chargement VLM")
            return False
    
    def frame_to_pil(self, frame) -> Image.Image:
        """Convertir frame OpenCV vers PIL."""
        # OpenCV utilise BGR, PIL utilise RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_frame)
    
    async def analyze_frame(self, frame, frame_number: int, timestamp: float):
        """Analyser une frame avec le VLM."""
        
        # Convertir en PIL
        pil_image = self.frame_to_pil(frame)
        
        # Prompt surveillance
        prompt = f"""Analysez cette frame de surveillance (frame #{frame_number}, t={timestamp:.1f}s).

Détectez et décrivez:
1. Personnes présentes (nombre, positions, actions)
2. Objets suspects ou inhabituels
3. Comportements potentiellement dangereux
4. Niveau de risque global (LOW/MEDIUM/HIGH/CRITICAL)

Soyez concis et précis."""
        
        # Requête VLM
        request = AnalysisRequest(
            image=pil_image,
            prompt=prompt,
            enable_advanced_tools=True,
            max_tokens=200
        )
        
        try:
            response = await self.vlm.analyze_image(request)
            
            result = {
                "frame_number": frame_number,
                "timestamp": timestamp,
                "suspicion": response.suspicion_level,
                "action": response.recommended_action,
                "confidence": response.confidence_score,
                "description": response.description,
                "tools_used": len(response.tools_used),
                "success": True
            }
            
            return result
            
        except Exception as e:
            return {
                "frame_number": frame_number,
                "timestamp": timestamp,
                "error": str(e),
                "success": False
            }
    
    async def process_video(self, video_path: str, max_frames: int = None, frame_skip: int = 30):
        """Traiter une vidéo complète."""
        
        print(f"🎬 Analyse vidéo: {video_path}")
        print(f"📊 Paramètres: max_frames={max_frames}, frame_skip={frame_skip}")
        
        # Ouvrir la vidéo
        if video_path == "webcam":
            cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"❌ Impossible d'ouvrir: {video_path}")
            return
        
        # Propriétés vidéo
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if video_path != "webcam":
            print(f"📹 Vidéo: {total_frames} frames, {fps:.1f} FPS")
        
        frame_count = 0
        analyzed_frames = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Skip frames pour accélérer
                if frame_count % frame_skip != 0:
                    continue
                
                # Limite max frames
                if max_frames and analyzed_frames >= max_frames:
                    break
                
                # Calculer timestamp
                timestamp = frame_count / fps if fps > 0 else frame_count
                
                print(f"\n🔍 Analyse frame {frame_count} (t={timestamp:.1f}s)...")
                
                # Afficher frame (optionnel)
                cv2.imshow('Video Analysis', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("⏹️  Analyse interrompue par l'utilisateur")
                    break
                
                # Analyser avec VLM
                result = await self.analyze_frame(frame, frame_count, timestamp)
                self.results.append(result)
                
                if result["success"]:
                    print(f"  🎯 Suspicion: {result['suspicion']}")
                    print(f"  🔧 Action: {result['action']}")
                    print(f"  📊 Confiance: {result['confidence']:.2f}")
                    print(f"  📝 {result['description'][:100]}...")
                else:
                    print(f"  ❌ Erreur: {result['error']}")
                
                analyzed_frames += 1
                
                # Pause pour voir les résultats
                if video_path != "webcam":
                    await asyncio.sleep(0.1)
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        print(f"\n✅ Analyse terminée: {analyzed_frames} frames analysées")
    
    def generate_summary(self):
        """Générer un résumé de l'analyse."""
        
        if not self.results:
            print("❌ Aucun résultat à résumer")
            return
        
        print("\n" + "="*50)
        print("📊 RÉSUMÉ D'ANALYSE VIDÉO")
        print("="*50)
        
        successful_analyses = [r for r in self.results if r.get("success", False)]
        
        print(f"📈 Statistiques:")
        print(f"  • Frames analysées: {len(self.results)}")
        print(f"  • Analyses réussies: {len(successful_analyses)}")
        print(f"  • Taux de succès: {len(successful_analyses)/len(self.results)*100:.1f}%")
        
        if successful_analyses:
            # Distribution des niveaux de suspicion
            suspicion_counts = {}
            for r in successful_analyses:
                level = r.get("suspicion", "unknown")
                suspicion_counts[level] = suspicion_counts.get(level, 0) + 1
            
            print(f"\n🎯 Niveaux de suspicion:")
            for level, count in suspicion_counts.items():
                print(f"  • {level.upper()}: {count} frames")
            
            # Confiance moyenne
            confidences = [r.get("confidence", 0) for r in successful_analyses]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            print(f"\n📊 Confiance moyenne: {avg_confidence:.2f}")
            
            # Moments critiques
            critical_moments = [r for r in successful_analyses if r.get("suspicion") == "high"]
            if critical_moments:
                print(f"\n⚠️  Moments critiques détectés:")
                for moment in critical_moments[:5]:  # Top 5
                    print(f"  • Frame {moment['frame_number']} (t={moment['timestamp']:.1f}s): {moment['description'][:80]}...")
        
        # Sauvegarder résultats
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"video_analysis_{timestamp}.json"
        
        import json
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n💾 Résultats sauvés: {filename}")


async def main():
    """Point d'entrée principal."""
    
    parser = argparse.ArgumentParser(description="Analyse vidéo avec VLM")
    parser.add_argument("--video", required=True, help="Chemin vidéo ou 'webcam'")
    parser.add_argument("--model", default="kimi-vl-a3b-thinking", help="Modèle VLM à utiliser")
    parser.add_argument("--max-frames", type=int, help="Nombre max de frames à analyser")
    parser.add_argument("--frame-skip", type=int, default=30, help="Analyser 1 frame toutes les N frames")
    
    args = parser.parse_args()
    
    analyzer = VideoAnalyzer(model_name=args.model)
    
    # Setup
    if not await analyzer.setup():
        return
    
    try:
        # Analyse vidéo
        await analyzer.process_video(
            args.video,
            max_frames=args.max_frames,
            frame_skip=args.frame_skip
        )
        
        # Résumé
        analyzer.generate_summary()
        
    except KeyboardInterrupt:
        print("\n⏹️  Analyse interrompue")
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())