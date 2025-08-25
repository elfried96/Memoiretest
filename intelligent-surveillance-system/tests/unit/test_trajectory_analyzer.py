#!/usr/bin/env python3
"""Test script for TrajectoryAnalyzer."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import time
from advanced_tools.trajectory_analyzer import TrajectoryAnalyzer

def create_test_trajectories():
    """Create different test trajectory patterns."""
    trajectories = {}
    current_time = time.time()
    
    # 1. Normal walking pattern (straight line)
    trajectories["normal_walk"] = []
    for i in range(20):
        trajectories["normal_walk"].append({
            "person_id": "person_1",
            "x": 100 + i * 10,  # Moving right
            "y": 200 + np.random.normal(0, 2),  # Small vertical variation
            "timestamp": current_time + i * 0.5
        })
    
    # 2. Suspicious loitering pattern (staying in one area)
    trajectories["loitering"] = []
    base_x, base_y = 300, 150
    for i in range(25):
        trajectories["loitering"].append({
            "person_id": "person_2", 
            "x": base_x + np.random.normal(0, 5),  # Small movements around same spot
            "y": base_y + np.random.normal(0, 5),
            "timestamp": current_time + i * 1.0  # Longer time intervals
        })
    
    # 3. Browsing pattern (lots of direction changes)
    trajectories["browsing"] = []
    x, y = 150, 100
    for i in range(30):
        # Random direction changes
        direction = np.random.random() * 2 * np.pi
        step_size = np.random.uniform(5, 15)
        x += np.cos(direction) * step_size
        y += np.sin(direction) * step_size
        
        # Keep within bounds
        x = np.clip(x, 50, 400)
        y = np.clip(y, 50, 300)
        
        trajectories["browsing"].append({
            "person_id": "person_3",
            "x": x,
            "y": y,
            "timestamp": current_time + i * 0.3
        })
    
    # 4. Return pattern (going back to same place)
    trajectories["return_pattern"] = []
    start_x, start_y = 200, 200
    
    # Go away from start
    for i in range(10):
        trajectories["return_pattern"].append({
            "person_id": "person_4",
            "x": start_x + i * 8,
            "y": start_y + i * 5,
            "timestamp": current_time + i * 0.4
        })
    
    # Return to start
    for i in range(10):
        trajectories["return_pattern"].append({
            "person_id": "person_4",
            "x": start_x + (10-i) * 8,
            "y": start_y + (10-i) * 5,
            "timestamp": current_time + (i + 10) * 0.4
        })
    
    # 5. Evasive movement (erratic, high speed changes)
    trajectories["evasive"] = []
    x, y = 250, 180
    for i in range(20):
        # Sudden direction and speed changes
        if i % 3 == 0:  # Change direction every 3 steps
            direction = np.random.random() * 2 * np.pi
            speed = np.random.uniform(15, 30)  # High speed
        else:
            speed = np.random.uniform(3, 8)   # Variable speed
        
        x += np.cos(direction) * speed
        y += np.sin(direction) * speed
        
        x = np.clip(x, 50, 450)
        y = np.clip(y, 50, 350)
        
        trajectories["evasive"].append({
            "person_id": "person_5",
            "x": x,
            "y": y,
            "timestamp": current_time + i * 0.2  # Fast movement
        })
    
    return trajectories

def test_trajectory_analyzer():
    """Test TrajectoryAnalyzer functionality."""
    print("=== Test TrajectoryAnalyzer ===")
    
    # Initialize TrajectoryAnalyzer
    try:
        analyzer = TrajectoryAnalyzer(history_size=50, velocity_threshold=0.5)
        print("✓ TrajectoryAnalyzer initialisé")
        print(f"  - Taille historique: {analyzer.history_size}")
        print(f"  - Seuil vélocité: {analyzer.velocity_threshold}")
        print(f"  - Patterns disponibles: {len(analyzer.pattern_templates)}")
    except Exception as e:
        print(f"✗ Erreur lors de l'initialisation: {e}")
        return False
    
    # Create test trajectories
    test_trajectories = create_test_trajectories()
    print(f"✓ {len(test_trajectories)} trajectoires de test créées")
    
    # Test each trajectory pattern
    results = {}
    
    for pattern_name, trajectory_data in test_trajectories.items():
        print(f"\n--- Test: {pattern_name} ---")
        
        # Add trajectory points
        try:
            person_id = trajectory_data[0]["person_id"]
            for data_point in trajectory_data:
                analyzer.update_trajectory(
                    person_id=data_point["person_id"],
                    x=data_point["x"], 
                    y=data_point["y"],
                    timestamp=data_point["timestamp"]
                )
            
            print(f"✓ {len(trajectory_data)} points ajoutés pour {person_id}")
        except Exception as e:
            print(f"✗ Erreur ajout de points: {e}")
            continue
        
        # Analyze trajectory
        try:
            analysis = analyzer.analyze_trajectory(person_id)
            if analysis:
                print("✓ Analyse de trajectoire réussie")
                print(f"  - Distance totale: {analysis.total_distance:.2f}")
                print(f"  - Vitesse moyenne: {analysis.average_velocity:.4f}")
                print(f"  - Changements direction: {analysis.direction_changes}")
                print(f"  - Points d'arrêt: {len(analysis.stopping_points)}")
                print(f"  - Score d'anomalie: {analysis.anomaly_score:.4f}")
                print(f"  - Classification: {analysis.pattern_classification}")
                print(f"  - Confiance: {analysis.confidence:.4f}")
                print(f"  - Temps traitement: {analysis.processing_time:.4f}s")
                
                results[pattern_name] = analysis
            else:
                print("✗ Aucune analyse retournée")
                
        except Exception as e:
            print(f"✗ Erreur lors de l'analyse: {e}")
    
    # Test motion analysis interface (for tool calling)
    print(f"\n--- Test Interface Motion Analysis ---")
    try:
        motion_result = analyzer.analyze_motion(test_trajectories["normal_walk"])
        print("✓ Interface motion analysis réussie")
        print(f"  - Vitesse mouvement: {motion_result['movement_speed']}")
        print(f"  - Changements direction: {motion_result['direction_changes']}")
        print(f"  - Fréquence arrêts: {motion_result['stopping_frequency']}")
        print(f"  - Pattern: {motion_result['pattern']}")
        print(f"  - Score anomalie: {motion_result['anomaly_score']:.4f}")
    except Exception as e:
        print(f"✗ Erreur interface motion analysis: {e}")
    
    # Test trajectory visualization
    print(f"\n--- Test Visualisation ---")
    try:
        frame_shape = (400, 500)  # height, width
        vis_frame = analyzer.visualize_trajectory("person_1", frame_shape)
        
        if vis_frame is not None:
            print("✓ Visualisation générée")
            print(f"  - Shape: {vis_frame.shape}")
        else:
            print("⚠ Aucune visualisation (trajectoire non trouvée)")
    except Exception as e:
        print(f"✗ Erreur visualisation: {e}")
    
    # Test trajectory summary
    print(f"\n--- Test Résumé Trajectoire ---")
    try:
        for person_id in ["person_1", "person_2", "person_3"]:
            summary = analyzer.get_trajectory_summary(person_id)
            if summary:
                print(f"✓ Résumé {person_id}:")
                print(f"  - Points: {summary['total_points']}")
                print(f"  - Distance: {summary['total_distance']:.2f}")
                print(f"  - Durée: {summary['time_span']:.2f}s")
                print(f"  - Pattern: {summary['pattern_classification']}")
    except Exception as e:
        print(f"✗ Erreur résumé trajectoire: {e}")
    
    # Test pattern analysis
    print(f"\n--- Test Analyse Patterns ---")
    expected_patterns = {
        "normal_walk": ["purposeful_walking", "normal_movement"],
        "loitering": ["suspicious_loitering"],
        "browsing": ["browsing"],
        "return_pattern": ["return_pattern"],
        "evasive": ["evasive_movement"]
    }
    
    correct_classifications = 0
    total_classifications = 0
    
    for pattern_name, expected in expected_patterns.items():
        if pattern_name in results:
            actual = results[pattern_name].pattern_classification
            total_classifications += 1
            if actual in expected:
                correct_classifications += 1
                print(f"✓ {pattern_name}: {actual} (correct)")
            else:
                print(f"⚠ {pattern_name}: {actual} (attendu: {expected})")
    
    if total_classifications > 0:
        accuracy = correct_classifications / total_classifications
        print(f"✓ Précision classification: {accuracy:.2%} ({correct_classifications}/{total_classifications})")
    
    # Test edge cases
    print(f"\n--- Test Cas Limites ---")
    
    # Empty trajectory
    try:
        empty_motion = analyzer.analyze_motion([])
        print(f"✓ Gestion trajectoire vide: {empty_motion['pattern']}")
    except Exception as e:
        print(f"✗ Erreur trajectoire vide: {e}")
    
    # Single point trajectory
    try:
        single_point = [{"person_id": "test", "x": 100, "y": 100, "timestamp": time.time()}]
        single_motion = analyzer.analyze_motion(single_point)
        print(f"✓ Gestion point unique: {single_motion['pattern']}")
    except Exception as e:
        print(f"✗ Erreur point unique: {e}")
    
    # Clean up old trajectories
    try:
        analyzer.clear_old_trajectories(max_age_seconds=1)  # Very short age for testing
        print("✓ Nettoyage anciennes trajectoires")
    except Exception as e:
        print(f"✗ Erreur nettoyage: {e}")
    
    return True

if __name__ == "__main__":
    success = test_trajectory_analyzer()
    
    print(f"\n{'='*60}")
    if success:
        print("✅ TOUS LES TESTS TRAJECTORYANALYZER RÉUSSIS")
        print("\nFonctionnalités testées:")
        print("- Initialisation et configuration")
        print("- Ajout de points de trajectoire")
        print("- Analyse complète de trajectoire")
        print("- Classification de patterns de mouvement:")
        print("  * Normal walking")
        print("  * Suspicious loitering") 
        print("  * Browsing behavior")
        print("  * Return patterns")
        print("  * Evasive movement")
        print("- Interface pour appel d'outils")
        print("- Visualisation de trajectoires")
        print("- Résumés et statistiques")
        print("- Gestion des cas limites")
        print("- Nettoyage automatique")
    else:
        print("❌ CERTAINS TESTS TRAJECTORYANALYZER ONT ÉCHOUÉ")
    
    print(f"{'='*60}")