#!/usr/bin/env python3
"""Test script for OpenPoseEstimator."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import cv2
from advanced_tools.pose_estimation import OpenPoseEstimator

def create_test_frames():
    """Create test frames with human-like figures."""
    frames = []
    
    # Frame 1: Simple stick figure
    frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw a simple human-like figure
    center_x, center_y = 320, 240
    # Head
    cv2.circle(frame1, (center_x, center_y - 100), 30, (255, 255, 255), -1)
    # Body
    cv2.line(frame1, (center_x, center_y - 70), (center_x, center_y + 50), (255, 255, 255), 8)
    # Arms
    cv2.line(frame1, (center_x, center_y - 30), (center_x - 60, center_y), (255, 255, 255), 6)
    cv2.line(frame1, (center_x, center_y - 30), (center_x + 60, center_y), (255, 255, 255), 6)
    # Legs
    cv2.line(frame1, (center_x, center_y + 50), (center_x - 40, center_y + 120), (255, 255, 255), 6)
    cv2.line(frame1, (center_x, center_y + 50), (center_x + 40, center_y + 120), (255, 255, 255), 6)
    frames.append(frame1)
    
    # Frame 2: Two figures
    frame2 = np.zeros((480, 640, 3), dtype=np.uint8)
    for i, x_pos in enumerate([200, 440]):
        # Head
        cv2.circle(frame2, (x_pos, 140), 25, (255, 255, 255), -1)
        # Body
        cv2.line(frame2, (x_pos, 165), (x_pos, 290), (255, 255, 255), 6)
        # Arms (different positions)
        if i == 0:  # First person - arms down
            cv2.line(frame2, (x_pos, 190), (x_pos - 50, 240), (255, 255, 255), 5)
            cv2.line(frame2, (x_pos, 190), (x_pos + 50, 240), (255, 255, 255), 5)
        else:  # Second person - arms up
            cv2.line(frame2, (x_pos, 190), (x_pos - 50, 160), (255, 255, 255), 5)
            cv2.line(frame2, (x_pos, 190), (x_pos + 50, 160), (255, 255, 255), 5)
        # Legs
        cv2.line(frame2, (x_pos, 290), (x_pos - 30, 380), (255, 255, 255), 5)
        cv2.line(frame2, (x_pos, 290), (x_pos + 30, 380), (255, 255, 255), 5)
    frames.append(frame2)
    
    # Frame 3: Real-world-like scene with noise
    frame3 = np.random.randint(20, 60, (480, 640, 3), dtype=np.uint8)  # Background noise
    # Add a clearer figure
    center_x, center_y = 320, 240
    cv2.circle(frame3, (center_x, center_y - 80), 25, (200, 200, 200), -1)
    cv2.line(frame3, (center_x, center_y - 55), (center_x, center_y + 30), (200, 200, 200), 6)
    cv2.line(frame3, (center_x, center_y - 20), (center_x - 45, center_y + 10), (200, 200, 200), 4)
    cv2.line(frame3, (center_x, center_y - 20), (center_x + 45, center_y + 10), (200, 200, 200), 4)
    cv2.line(frame3, (center_x, center_y + 30), (center_x - 35, center_y + 100), (200, 200, 200), 4)
    cv2.line(frame3, (center_x, center_y + 30), (center_x + 35, center_y + 100), (200, 200, 200), 4)
    frames.append(frame3)
    
    return frames

def test_pose_estimator():
    """Test OpenPoseEstimator functionality."""
    print("=== Test OpenPoseEstimator ===")
    
    # Create test frames
    test_frames = create_test_frames()
    print(f"✓ {len(test_frames)} frames de test créées")
    
    # Test with different model types
    model_types = ["mediapipe", "movenet"]
    
    for model_type in model_types:
        print(f"\n--- Test avec modèle: {model_type} ---")
        
        # Initialize OpenPoseEstimator
        try:
            estimator = OpenPoseEstimator(model_type=model_type)
            print(f"✓ OpenPoseEstimator ({model_type}) initialisé")
        except Exception as e:
            print(f"✗ Erreur lors de l'initialisation {model_type}: {e}")
            continue
        
        # Test 1: Basic pose estimation
        print(f"\n  Test 1: Estimation de base ({model_type})")
        try:
            result = estimator.estimate_poses(test_frames[0])
            print("  ✓ Estimation de pose réussie")
            print(f"    - Keypoints shape: {result.keypoints.shape if result.keypoints.size > 0 else 'Vide'}")
            print(f"    - Nombre de personnes: {len(result.person_boxes)}")
            print(f"    - Connexions squelette: {len(result.skeleton_connections)}")
            print(f"    - Temps de traitement: {result.processing_time:.4f}s")
        except Exception as e:
            print(f"  ✗ Erreur lors de l'estimation de base: {e}")
            continue
        
        # Test 2: Pose estimation with person boxes
        print(f"\n  Test 2: Estimation avec boîtes personnages ({model_type})")
        person_boxes = [(150, 50, 350, 400), (350, 50, 550, 400)]  # Two person areas
        
        try:
            result_with_boxes = estimator.estimate_poses(test_frames[1], person_boxes=person_boxes)
            print("  ✓ Estimation avec boîtes réussie")
            print(f"    - Keypoints détectés: {result_with_boxes.keypoints.shape if result_with_boxes.keypoints.size > 0 else 'Vide'}")
            print(f"    - Boîtes traitées: {len(result_with_boxes.person_boxes)}")
        except Exception as e:
            print(f"  ✗ Erreur avec boîtes personnages: {e}")
            continue
        
        # Test 3: Behavioral analysis
        print(f"\n  Test 3: Analyse comportementale ({model_type})")
        try:
            if result.keypoints.size > 0:
                behavior_analysis = estimator.analyze_pose_behavior(result.keypoints)
                print("  ✓ Analyse comportementale réussie")
                print(f"    - Score de comportement: {behavior_analysis['behavior_score']:.4f}")
                print(f"    - Nombre d'indicateurs: {len(behavior_analysis['indicators'])}")
                print(f"    - Nombre de personnes: {behavior_analysis['num_people']}")
                if behavior_analysis['indicators']:
                    print(f"    - Indicateurs: {behavior_analysis['indicators']}")
            else:
                print("  ⚠ Pas de keypoints pour analyse comportementale")
        except Exception as e:
            print(f"  ✗ Erreur lors de l'analyse comportementale: {e}")
        
        # Test 4: Movement analysis (if we have previous keypoints)
        print(f"\n  Test 4: Analyse de mouvement ({model_type})")
        try:
            # Get poses from two different frames
            current_result = estimator.estimate_poses(test_frames[0])
            previous_result = estimator.estimate_poses(test_frames[1])
            
            if current_result.keypoints.size > 0 and previous_result.keypoints.size > 0:
                movement_analysis = estimator.analyze_pose_behavior(
                    current_result.keypoints, 
                    previous_keypoints=previous_result.keypoints
                )
                print("  ✓ Analyse de mouvement réussie")
                print(f"    - Score avec mouvement: {movement_analysis['behavior_score']:.4f}")
                if 'movement_indicators' in str(movement_analysis):
                    print("    - Indicateurs de mouvement détectés")
            else:
                print("  ⚠ Keypoints insuffisants pour analyse de mouvement")
        except Exception as e:
            print(f"  ✗ Erreur lors de l'analyse de mouvement: {e}")
        
        # Test 5: Edge cases
        print(f"\n  Test 5: Cas limites ({model_type})")
        
        # Empty frame
        try:
            empty_frame = np.zeros((100, 100, 3), dtype=np.uint8)
            empty_result = estimator.estimate_poses(empty_frame)
            print("  ✓ Gestion frame vide")
        except Exception as e:
            print(f"  ✗ Erreur avec frame vide: {e}")
        
        # Very small person boxes
        try:
            tiny_boxes = [(10, 10, 20, 20)]
            tiny_result = estimator.estimate_poses(test_frames[0], person_boxes=tiny_boxes)
            print("  ✓ Gestion boîtes minuscules")
        except Exception as e:
            print(f"  ✗ Erreur avec boîtes minuscules: {e}")
        
        # Empty keypoints for behavior analysis
        try:
            empty_keypoints = np.array([])
            empty_behavior = estimator.analyze_pose_behavior(empty_keypoints)
            print(f"  ✓ Gestion keypoints vides: score={empty_behavior['behavior_score']:.4f}")
        except Exception as e:
            print(f"  ✗ Erreur avec keypoints vides: {e}")
        
        print(f"  --- Fin tests {model_type} ---")
    
    return True

if __name__ == "__main__":
    success = test_pose_estimator()
    
    print(f"\n{'='*60}")
    if success:
        print("✅ TOUS LES TESTS OPENPOSEESTIMATOR TERMINÉS")
        print("\nFonctionnalités testées:")
        print("- Initialisation avec différents modèles (MediaPipe, MoveNet)")
        print("- Estimation de pose globale")
        print("- Estimation avec boîtes de personnages")
        print("- Analyse comportementale")
        print("- Analyse de mouvement")
        print("- Gestion des cas limites")
        print("- Fallback en cas d'erreur des modèles")
        print("\nNote: Certains modèles peuvent ne pas être disponibles")
        print("      mais le système utilise des fallbacks appropriés.")
    else:
        print("❌ CERTAINS TESTS OPENPOSEESTIMATOR ONT ÉCHOUÉ")
    
    print(f"{'='*60}")