#!/usr/bin/env python3
"""Test du système de déclenchement intelligent du VLM."""

import sys
import time
from pathlib import Path

# Ajouter le répertoire source au PYTHON_PATH
sys.path.append(str(Path(__file__).parent / "src"))

from main_headless import HeadlessSurveillanceSystem
from src.core.orchestrator.vlm_orchestrator import OrchestrationMode
from src.core.types import DetectedObject, BoundingBox

def test_intelligent_triggering():
    """Test du système de déclenchement intelligent."""
    print("🧪 Test du système de déclenchement intelligent VLM")
    print("=" * 60)
    
    # Créer une instance du système de surveillance
    surveillance = HeadlessSurveillanceSystem(
        video_source=0,  # Pas important pour le test
        vlm_model="kimi-vl-a3b-thinking",
        orchestration_mode=OrchestrationMode.FAST,
        save_results=False,  # Pas de sauvegarde pour le test
        save_frames=False
    )
    
    # Simuler différents scénarios de détection
    test_scenarios = [
        {
            "name": "Scénario normal: 1 personne",
            "detections": [
                DetectedObject(
                    class_name="person",
                    class_id=0,
                    confidence=0.9,
                    bbox=BoundingBox(x1=100, y1=100, x2=200, y2=300),
                    track_id=1
                )
            ],
            "expected": False  # Ne devrait pas déclencher immédiatement
        },
        {
            "name": "Scénario suspect: 3 personnes",
            "detections": [
                DetectedObject(class_name="person", class_id=0, confidence=0.9, bbox=BoundingBox(x1=100, y1=100, x2=200, y2=300), track_id=1),
                DetectedObject(class_name="person", class_id=0, confidence=0.8, bbox=BoundingBox(x1=300, y1=100, x2=400, y2=300), track_id=2),
                DetectedObject(class_name="person", class_id=0, confidence=0.85, bbox=BoundingBox(x1=500, y1=100, x2=600, y2=300), track_id=3)
            ],
            "expected": True  # Devrait déclencher (3+ personnes)
        },
        {
            "name": "Scénario objet suspect: sac à dos",
            "detections": [
                DetectedObject(class_name="person", class_id=0, confidence=0.9, bbox=BoundingBox(x1=100, y1=100, x2=200, y2=300), track_id=1),
                DetectedObject(class_name="backpack", class_id=1, confidence=0.7, bbox=BoundingBox(x1=150, y1=120, x2=180, y2=180), track_id=2)
            ],
            "expected": True  # Devrait déclencher (objet suspect)
        }
    ]
    
    print("\n🔍 Test des différents scénarios:")
    print("-" * 40)
    
    for i, scenario in enumerate(test_scenarios):
        print(f"\n{i+1}. {scenario['name']}")
        
        # Simuler le contexte
        context = {
            "frame_id": i,
            "timestamp": time.time(),
            "test_mode": True
        }
        
        # Tester la logique de déclenchement
        persons_count = len([d for d in scenario['detections'] if d.class_name == "person"])
        should_trigger = surveillance._should_trigger_vlm_analysis(
            scenario['detections'], 
            persons_count, 
            context
        )
        
        # Vérifier le résultat
        status = "✅ CORRECT" if should_trigger == scenario['expected'] else "❌ ERREUR"
        trigger_text = "DÉCLENCHÉ" if should_trigger else "PAS DÉCLENCHÉ"
        expected_text = "devrait" if scenario['expected'] else "ne devrait pas"
        
        print(f"   Résultat: {trigger_text} ({expected_text} déclencher)")
        print(f"   Status: {status}")
        
        # Réinitialiser le cooldown pour chaque test
        surveillance.last_vlm_trigger_time = 0
    
    print("\n" + "=" * 60)
    print("🏁 Test terminé")

if __name__ == "__main__":
    test_intelligent_triggering()