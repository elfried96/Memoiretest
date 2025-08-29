#!/usr/bin/env python3
"""Test du système de mémoire contextuelle VLM."""

import sys
import time
from pathlib import Path

# Ajouter le répertoire source au PYTHON_PATH
sys.path.append(str(Path(__file__).parent / "src"))

from src.core.vlm.memory_system import VLMMemorySystem
from src.core.types import DetectedObject, BoundingBox, AnalysisResponse, SuspicionLevel, ActionType

def test_memory_system():
    """Test du système de mémoire contextuelle."""
    print("🧪 Test du système de mémoire contextuelle VLM")
    print("=" * 60)
    
    # Créer une instance du système de mémoire
    memory = VLMMemorySystem(max_frames=10, max_persons=5)
    
    print("\n1. 🧠 Test d'ajout de frames en mémoire")
    print("-" * 40)
    
    # Simuler plusieurs frames avec différents scénarios
    test_scenarios = [
        {
            "frame_id": 1,
            "detections": [
                DetectedObject(class_name="person", class_id=0, confidence=0.9, 
                              bbox=BoundingBox(x1=100, y1=100, x2=200, y2=300), track_id=1)
            ],
            "vlm_triggered": False,
            "description": "1 personne normale"
        },
        {
            "frame_id": 2,
            "detections": [
                DetectedObject(class_name="person", class_id=0, confidence=0.9, 
                              bbox=BoundingBox(x1=105, y1=105, x2=205, y2=305), track_id=1),
                DetectedObject(class_name="person", class_id=0, confidence=0.8, 
                              bbox=BoundingBox(x1=300, y1=100, x2=400, y2=300), track_id=2)
            ],
            "vlm_triggered": False,
            "description": "2 personnes"
        },
        {
            "frame_id": 3,
            "detections": [
                DetectedObject(class_name="person", class_id=0, confidence=0.9, 
                              bbox=BoundingBox(x1=110, y1=110, x2=210, y2=310), track_id=1),
                DetectedObject(class_name="person", class_id=0, confidence=0.8, 
                              bbox=BoundingBox(x1=305, y1=105, x2=405, y2=305), track_id=2),
                DetectedObject(class_name="person", class_id=0, confidence=0.85, 
                              bbox=BoundingBox(x1=500, y1=100, x2=600, y2=300), track_id=3)
            ],
            "vlm_triggered": True,
            "vlm_analysis": AnalysisResponse(
                suspicion_level=SuspicionLevel.MEDIUM,
                action_type=ActionType.SUSPICIOUS_MOVEMENT,
                confidence=0.75,
                description="3 personnes détectées - situation à surveiller",
                tools_used=["pose_estimator", "dino_features"],
                recommendations=["Continuer surveillance"]
            ),
            "description": "3 personnes - VLM déclenché"
        }
    ]
    
    # Ajouter les frames à la mémoire
    for scenario in test_scenarios:
        print(f"  📝 Frame {scenario['frame_id']}: {scenario['description']}")
        
        memory.add_frame(
            frame_id=scenario['frame_id'],
            detections=scenario['detections'],
            vlm_triggered=scenario['vlm_triggered'],
            vlm_analysis=scenario.get('vlm_analysis'),
            alert_level="normal" if not scenario['vlm_triggered'] else "attention",
            actions_taken=["monitoring"] if scenario['vlm_triggered'] else []
        )
        
        time.sleep(0.1)  # Petite pause pour simuler le temps
    
    print("\n2. 📊 Statistiques de mémoire")
    print("-" * 40)
    
    stats = memory.get_memory_stats()
    print(f"  • Frames traitées: {stats['total_frames_processed']}")
    print(f"  • Frames en mémoire: {stats['current_frames_in_memory']}")
    print(f"  • Analyses VLM stockées: {stats['vlm_analyses_stored']}")
    print(f"  • Personnes actives: {stats['active_persons']}")
    print(f"  • Patterns détectés: {stats['patterns_detected']}")
    print(f"  • Taille mémoire: {stats['memory_size_mb']:.3f} MB")
    
    print("\n3. 🧠 Contexte pour le VLM")
    print("-" * 40)
    
    context = memory.get_context_for_vlm()
    print(f"  • Frames d'historique: {len(context['previous_detections'])}")
    print(f"  • Personnes actives: {len(context['active_persons'])}")
    print(f"  • Patterns suspects: {len(context['suspicious_patterns'])}")
    
    print("\n  📋 Résumé de mémoire:")
    for key, value in context['memory_summary'].items():
        print(f"    - {key}: {value}")
    
    if context['previous_detections']:
        print("\n  📖 Dernières détections:")
        for detection in context['previous_detections'][-2:]:  # 2 dernières
            print(f"    Frame {detection['frame_id']}: {detection['persons_count']} personnes, "
                  f"Alert: {detection['alert_level']}")
    
    print("\n4. 🔍 Test de détection de patterns")
    print("-" * 40)
    
    # Ajouter plusieurs frames avec augmentation de population pour déclencher un pattern
    for i in range(4, 15):  # Frames 4 à 14
        person_count = min(i - 2, 5)  # Augmentation progressive
        detections = []
        
        for p in range(person_count):
            detections.append(
                DetectedObject(
                    class_name="person", class_id=0, confidence=0.8, 
                    bbox=BoundingBox(x1=100+p*50, y1=100, x2=150+p*50, y2=300), 
                    track_id=10+p
                )
            )
        
        memory.add_frame(
            frame_id=i,
            detections=detections,
            vlm_triggered=(person_count >= 3),
            vlm_analysis=None,
            alert_level="attention" if person_count >= 3 else "normal",
            actions_taken=["monitoring"] if person_count >= 3 else []
        )
        
        time.sleep(0.05)
    
    # Vérifier les patterns détectés
    final_stats = memory.get_memory_stats()
    final_context = memory.get_context_for_vlm()
    
    print(f"  🎯 Patterns détectés au total: {final_stats['patterns_detected']}")
    
    if final_context['suspicious_patterns']:
        print("  📈 Patterns suspects identifiés:")
        for pattern in final_context['suspicious_patterns']:
            print(f"    - {pattern['type']}: {pattern['description']} (Gravité: {pattern['severity']})")
    else:
        print("  ✅ Aucun pattern suspect détecté")
    
    print("\n5. 🗂️ Export de dump mémoire")
    print("-" * 40)
    
    dump = memory.export_memory_dump()
    print(f"  📝 Taille du dump: {len(dump)} caractères")
    
    # Sauvegarder le dump pour inspection
    with open("memory_dump_test.json", "w") as f:
        f.write(dump)
    print("  💾 Dump sauvé: memory_dump_test.json")
    
    print("\n" + "=" * 60)
    print("🏁 Test du système de mémoire terminé")
    print(f"✅ {final_stats['total_frames_processed']} frames traitées")
    print(f"🧠 {final_stats['current_frames_in_memory']} frames en mémoire")
    print(f"👥 {final_stats['active_persons']} personnes trackées")
    print(f"🔍 {final_stats['patterns_detected']} patterns détectés")

if __name__ == "__main__":
    test_memory_system()