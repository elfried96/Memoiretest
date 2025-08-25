#!/usr/bin/env python3
"""
🏗️ Test de l'Architecture Sans Dépendances Vidéo
==============================================

Test de la logique métier de l'AdaptiveVLMOrchestrator sans nécessiter de vraies vidéos.
"""

import sys
import asyncio
from pathlib import Path
from typing import List, Dict, Any
import time
import json

# Configuration du path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

def simulate_frame_data(frame_number: int) -> Dict[str, Any]:
    """Simule les données d'une frame de surveillance."""
    
    # Simulation d'objets détectés
    detections = []
    
    # Frame 10-30: Personne entre dans le magasin
    if 10 <= frame_number <= 30:
        detections.append({
            "bbox": [100, 100, 200, 300],
            "confidence": 0.85,
            "class_name": "person",
            "track_id": 1
        })
    
    # Frame 40-60: Personne près des étagères (comportement suspect)
    if 40 <= frame_number <= 60:
        detections.extend([
            {
                "bbox": [150, 120, 250, 320],
                "confidence": 0.88,
                "class_name": "person", 
                "track_id": 1
            },
            {
                "bbox": [300, 50, 400, 150],
                "confidence": 0.75,
                "class_name": "bottle",
                "track_id": 2
            }
        ])
    
    # Frame 70-90: Action de prise d'objet
    if 70 <= frame_number <= 90:
        detections.extend([
            {
                "bbox": [140, 110, 240, 310],
                "confidence": 0.90,
                "class_name": "person",
                "track_id": 1
            },
            {
                "bbox": [180, 180, 220, 220],  # Objet près de la personne
                "confidence": 0.70,
                "class_name": "bottle", 
                "track_id": 2
            }
        ])
    
    return {
        "frame_number": frame_number,
        "detections": detections,
        "timestamp": time.time(),
        "resolution": (640, 480)
    }

class ArchitectureTester:
    """Testeur de l'architecture sans dépendances vidéo."""
    
    def __init__(self):
        self.orchestrator = None
        self.results = []
        
    async def initialize_architecture(self):
        """Initialise l'AdaptiveVLMOrchestrator."""
        
        try:
            # Import et initialisation de l'orchestrateur
            from src.core.orchestrator.adaptive_orchestrator import AdaptiveVLMOrchestrator
            from config.app_config import load_config
            
            # Configuration test avec Kimi-VL seulement
            config = load_config("testing_kimi")
            
            print("🤖 Initialisation AdaptiveVLMOrchestrator...")
            self.orchestrator = AdaptiveVLMOrchestrator(config)
            
            print(f"✅ Orchestrateur initialisé")
            print(f"📋 Mode: {config.orchestration.mode.value}")
            print(f"🛠️ Outils disponibles: {len(config.orchestration.enabled_tools)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Erreur initialisation: {e}")
            return False
    
    async def test_frame_analysis(self, frame_data: Dict[str, Any]) -> Dict[str, Any]:
        """Test l'analyse d'une frame."""
        
        if not self.orchestrator:
            return {"error": "Orchestrateur non initialisé"}
        
        try:
            # Préparation des données pour l'orchestrateur
            analysis_request = {
                "frame_number": frame_data["frame_number"],
                "detections": frame_data["detections"],
                "context": "surveillance_magasin",
                "timestamp": frame_data["timestamp"]
            }
            
            start_time = time.time()
            
            # Analyse avec l'AdaptiveVLMOrchestrator
            result = await self.orchestrator.analyze_scene(analysis_request)
            
            processing_time = time.time() - start_time
            
            return {
                "frame_number": frame_data["frame_number"],
                "processing_time": processing_time,
                "detections_count": len(frame_data["detections"]),
                "analysis": result,
                "success": True
            }
            
        except Exception as e:
            return {
                "frame_number": frame_data["frame_number"],
                "error": str(e),
                "success": False
            }
    
    async def run_surveillance_scenario(self, total_frames: int = 100):
        """Exécute un scénario de surveillance complet."""
        
        print(f"🎬 Démarrage scénario surveillance ({total_frames} frames)")
        print("=" * 60)
        
        if not await self.initialize_architecture():
            return
        
        # Variables de tracking
        total_detections = 0
        suspicious_frames = 0
        alerts_generated = 0
        tools_used = set()
        
        start_time = time.time()
        
        # Traitement frame par frame
        for frame_num in range(1, total_frames + 1):
            
            # Génération données simulées
            frame_data = simulate_frame_data(frame_num)
            
            # Analyse avec orchestrateur
            result = await self.test_frame_analysis(frame_data)
            
            self.results.append(result)
            
            if result["success"]:
                total_detections += result["detections_count"]
                
                # Analyse des résultats
                analysis = result.get("analysis", {})
                
                if isinstance(analysis, dict):
                    # Détection de comportements suspects
                    if analysis.get("suspicion_level", 0) > 0.5:
                        suspicious_frames += 1
                    
                    # Comptage des alertes
                    if analysis.get("security_alert"):
                        alerts_generated += 1
                    
                    # Outils utilisés
                    if "tools_used" in analysis:
                        tools_used.update(analysis["tools_used"])
            
            # Progression
            if frame_num % 25 == 0:
                elapsed = time.time() - start_time
                fps = frame_num / elapsed
                print(f"📈 Frame {frame_num}/{total_frames} | FPS: {fps:.1f} | Détections: {total_detections}")
        
        total_time = time.time() - start_time
        avg_fps = total_frames / total_time
        
        # Rapport final
        self.print_surveillance_report({
            "total_frames": total_frames,
            "total_time": total_time,
            "average_fps": avg_fps,
            "total_detections": total_detections,
            "suspicious_frames": suspicious_frames,
            "alerts_generated": alerts_generated,
            "tools_used": list(tools_used),
            "success_rate": len([r for r in self.results if r["success"]]) / total_frames
        })
        
        # Sauvegarde rapport détaillé
        self.save_detailed_report()
    
    def print_surveillance_report(self, summary: Dict[str, Any]):
        """Affiche le rapport de surveillance."""
        
        print("\n" + "=" * 60)
        print("🛡️ RAPPORT DE SURVEILLANCE")
        print("=" * 60)
        
        print(f"📊 Frames analysées: {summary['total_frames']}")
        print(f"⏱️ Temps total: {summary['total_time']:.2f}s")
        print(f"🎯 FPS moyen: {summary['average_fps']:.1f}")
        print(f"✅ Taux de succès: {summary['success_rate']:.1%}")
        
        print(f"\n🔍 DÉTECTIONS:")
        print(f"  Total détections: {summary['total_detections']}")
        print(f"  Frames suspectes: {summary['suspicious_frames']}")
        print(f"  Alertes générées: {summary['alerts_generated']}")
        
        print(f"\n🛠️ OUTILS UTILISÉS:")
        if summary['tools_used']:
            for tool in summary['tools_used']:
                print(f"  ✓ {tool}")
        else:
            print("  ⚠️ Aucun outil avancé détecté")
        
        print(f"\n📈 PERFORMANCE:")
        print(f"  Frames/sec: {summary['average_fps']:.1f}")
        print(f"  Détections/frame: {summary['total_detections']/summary['total_frames']:.2f}")
        
        # Évaluation qualitative
        if summary['alerts_generated'] > 0:
            print(f"\n🚨 SÉCURITÉ: {summary['alerts_generated']} alertes générées - Système actif")
        else:
            print(f"\n🔒 SÉCURITÉ: Aucune alerte - Surveillance normale")
    
    def save_detailed_report(self):
        """Sauvegarde un rapport détaillé."""
        
        report_data = {
            "test_type": "architecture_simulation",
            "timestamp": time.time(),
            "total_results": len(self.results),
            "frame_analysis": [
                {
                    "frame": r["frame_number"],
                    "success": r["success"],
                    "processing_time": r.get("processing_time", 0),
                    "detections": r.get("detections_count", 0),
                    "analysis_summary": str(r.get("analysis", {}))[:200] + "..." if len(str(r.get("analysis", {}))) > 200 else str(r.get("analysis", {}))
                }
                for r in self.results
            ]
        }
        
        report_path = PROJECT_ROOT / "architecture_test_report.json"
        
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n💾 Rapport détaillé sauvé: {report_path}")

async def main():
    """Point d'entrée principal."""
    
    print("🏗️ Test de l'Architecture de Surveillance Intelligente")
    print("=" * 65)
    
    tester = ArchitectureTester()
    
    try:
        await tester.run_surveillance_scenario(100)
        print("\n🎉 Test architectural terminé avec succès !")
        
    except KeyboardInterrupt:
        print("\n⚠️ Test interrompu par l'utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur durant le test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())