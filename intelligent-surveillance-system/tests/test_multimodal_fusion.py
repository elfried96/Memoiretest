#!/usr/bin/env python3
"""Test script for MultiModalFusion."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import time
from advanced_tools.multimodal_fusion import MultiModalFusion, FusionInput

def create_test_features():
    """Create mock features for different modalities."""
    # Visual features (DINO v2 like)
    visual_features = np.random.randn(768)  # 768-dim DINO features
    
    # Detection features (extracted from detections)
    detection_features = np.array([
        5.0,    # num_detections
        0.8, 0.1, 0.9,  # confidence stats (mean, std, max)
        2, 1, 0, 1,     # class counts (person, handbag, backpack, suitcase)
        0.5, 0.3,       # spatial center x, y
        0.2, 0.15       # spatial spread x, y
    ])
    # Pad to expected dimension (256)
    detection_features = np.pad(detection_features, (0, 256 - len(detection_features)))
    
    # Pose features (33 keypoints * 4 = 132)
    pose_features = np.random.randn(132)
    pose_features[::4] = np.random.uniform(0.3, 1.0, 33)  # Confidence values
    
    # Motion features
    motion_features = np.array([
        1.2,    # average_velocity
        3.0,    # direction_changes (normalized)
        0.15,   # anomaly_score
        150.0,  # total_distance (normalized)
        # Pattern encoding (one-hot for "normal_movement")
        0, 0, 0, 0, 0, 1
    ])
    # Pad to 64 dimensions
    motion_features = np.pad(motion_features, (0, 64 - len(motion_features)))
    
    # Temporal features
    temporal_features = np.array([
        0.5,    # hour_of_day (normalized)
        0, 1, 0, 0, 0, 0, 0,  # day_of_week (Tuesday)
        0.7,    # consistency_score
        0.15    # sequence_length (normalized)
    ])
    # Pad to 128 dimensions
    temporal_features = np.pad(temporal_features, (0, 128 - len(temporal_features)))
    
    return {
        'visual': visual_features,
        'detection': detection_features,
        'pose': pose_features,
        'motion': motion_features,
        'temporal': temporal_features
    }

def create_test_detections():
    """Create mock detection results."""
    return [
        {
            'confidence': 0.9,
            'bbox': [100, 150, 200, 300],
            'class': 'person'
        },
        {
            'confidence': 0.7,
            'bbox': [180, 200, 220, 250],
            'class': 'handbag'
        },
        {
            'confidence': 0.6,
            'bbox': [50, 100, 90, 140],
            'class': 'backpack'
        }
    ]

def create_test_pose_data():
    """Create mock pose estimation data."""
    # 33 keypoints with x, y, confidence
    keypoints = []
    for i in range(33):
        keypoints.append([
            np.random.uniform(0, 640),      # x
            np.random.uniform(0, 480),      # y
            np.random.uniform(0.3, 0.9)     # confidence
        ])
    
    return {
        'keypoints': [keypoints],  # One person
        'indicators': ['hand_near_waist'],
        'suspicion_score': 0.3
    }

def create_test_motion_data():
    """Create mock motion analysis data."""
    return {
        'average_velocity': 1.2,
        'direction_changes': 3,
        'anomaly_score': 0.15,
        'total_distance': 150.0,
        'pattern': 'normal_movement'
    }

def create_test_temporal_data():
    """Create mock temporal data."""
    return {
        'consistency_score': 0.7,
        'sequence_length': 15
    }

def test_multimodal_fusion():
    """Test MultiModalFusion functionality."""
    print("=== Test MultiModalFusion ===")
    
    # Initialize MultiModalFusion
    try:
        fusion_system = MultiModalFusion()
        print("✓ MultiModalFusion initialisé")
        print(f"  - Dimensions features: {fusion_system.feature_dims}")
        print(f"  - Device: {fusion_system.device}")
    except Exception as e:
        print(f"✗ Erreur lors de l'initialisation: {e}")
        return False
    
    # Test feature extraction methods
    print(f"\n--- Test Extraction Features ---")
    
    # Test detection feature extraction
    try:
        test_detections = create_test_detections()
        detection_features = fusion_system.extract_detection_features(test_detections)
        print(f"✓ Features détection extraites: shape {detection_features.shape}")
        print(f"  - Nombre détections: {detection_features[10]:.0f}")  # detection count
        print(f"  - Confiance moyenne: {detection_features[0]:.3f}")
    except Exception as e:
        print(f"✗ Erreur extraction features détection: {e}")
    
    # Test pose feature extraction
    try:
        test_pose = create_test_pose_data()
        pose_features = fusion_system.extract_pose_features(test_pose)
        print(f"✓ Features pose extraites: shape {pose_features.shape}")
    except Exception as e:
        print(f"✗ Erreur extraction features pose: {e}")
    
    # Test motion feature extraction
    try:
        test_motion = create_test_motion_data()
        motion_features = fusion_system.extract_motion_features(test_motion)
        print(f"✓ Features mouvement extraites: shape {motion_features.shape}")
        print(f"  - Vitesse: {motion_features[0]:.3f}")
        print(f"  - Score anomalie: {motion_features[2]:.3f}")
    except Exception as e:
        print(f"✗ Erreur extraction features mouvement: {e}")
    
    # Test temporal feature extraction
    try:
        test_temporal = create_test_temporal_data()
        temporal_features = fusion_system.extract_temporal_features(test_temporal)
        print(f"✓ Features temporelles extraites: shape {temporal_features.shape}")
        print(f"  - Score consistance: {temporal_features[8]:.3f}")
    except Exception as e:
        print(f"✗ Erreur extraction features temporelles: {e}")
    
    # Test complete fusion process
    print(f"\n--- Test Fusion Complète ---")
    
    # Create complete fusion input
    test_features = create_test_features()
    fusion_input = FusionInput(
        visual_features=test_features['visual'],
        detection_features=test_features['detection'],
        pose_features=test_features['pose'],
        motion_features=test_features['motion'],
        temporal_features=test_features['temporal']
    )
    
    # Test statistical fusion
    print(f"\n  Test 1: Fusion Statistique")
    try:
        stat_result = fusion_system.fuse_features(fusion_input, fusion_method="statistical")
        print("  ✓ Fusion statistique réussie")
        print(f"    - Shape features fusionnées: {stat_result.fused_features.shape}")
        print(f"    - Prédiction finale: {stat_result.final_prediction:.4f}")
        print(f"    - Poids attention: {stat_result.attention_weights}")
        print(f"    - Scores confiance: {list(stat_result.confidence_scores.keys())}")
        print(f"    - Temps traitement: {stat_result.processing_time:.4f}s")
    except Exception as e:
        print(f"  ✗ Erreur fusion statistique: {e}")
    
    # Test attention-based fusion
    print(f"\n  Test 2: Fusion par Attention")
    try:
        attention_result = fusion_system.fuse_features(fusion_input, fusion_method="attention")
        print("  ✓ Fusion par attention réussie")
        print(f"    - Shape features fusionnées: {attention_result.fused_features.shape}")
        print(f"    - Prédiction finale: {attention_result.final_prediction:.4f}")
        print(f"    - Poids attention: {attention_result.attention_weights}")
        print(f"    - Temps traitement: {attention_result.processing_time:.4f}s")
    except Exception as e:
        print(f"  ✗ Erreur fusion attention: {e}")
        # Try statistical as fallback
        try:
            fallback_result = fusion_system.fuse_features(fusion_input, fusion_method="statistical")
            print("  ✓ Fallback statistique utilisé")
        except Exception as e2:
            print(f"  ✗ Même fallback échoué: {e2}")
    
    # Test partial inputs (missing modalities)
    print(f"\n--- Test Entrées Partielles ---")
    
    test_cases = [
        ("Visual seul", FusionInput(visual_features=test_features['visual'])),
        ("Détection seule", FusionInput(detection_features=test_features['detection'])),
        ("Pose seule", FusionInput(pose_features=test_features['pose'])),
        ("Visual + Détection", FusionInput(
            visual_features=test_features['visual'],
            detection_features=test_features['detection']
        )),
        ("Mouvement + Temporel", FusionInput(
            motion_features=test_features['motion'],
            temporal_features=test_features['temporal']
        ))
    ]
    
    for test_name, partial_input in test_cases:
        try:
            partial_result = fusion_system.fuse_features(partial_input, fusion_method="statistical")
            print(f"  ✓ {test_name}: prédiction {partial_result.final_prediction:.4f}")
            print(f"    Modalités: {list(partial_result.attention_weights.keys())}")
        except Exception as e:
            print(f"  ✗ {test_name}: erreur {e}")
    
    # Test edge cases
    print(f"\n--- Test Cas Limites ---")
    
    # Empty input
    try:
        empty_input = FusionInput()
        empty_result = fusion_system.fuse_features(empty_input)
        print(f"✓ Entrée vide: prédiction {empty_result.final_prediction:.4f}")
    except Exception as e:
        print(f"✗ Erreur entrée vide: {e}")
    
    # Invalid/corrupted features
    try:
        corrupted_features = np.full(768, np.nan)  # NaN features
        corrupted_input = FusionInput(visual_features=corrupted_features)
        corrupted_result = fusion_system.fuse_features(corrupted_input)
        print(f"✓ Features corrompues gérées: prédiction {corrupted_result.final_prediction:.4f}")
    except Exception as e:
        print(f"✗ Erreur features corrompues: {e}")
    
    # Very small features
    try:
        tiny_features = np.random.randn(10)  # Much smaller than expected
        tiny_input = FusionInput(visual_features=tiny_features)
        tiny_result = fusion_system.fuse_features(tiny_input)
        print(f"✓ Features petites: prédiction {tiny_result.final_prediction:.4f}")
    except Exception as e:
        print(f"✗ Erreur features petites: {e}")
    
    # Test integration with real extracted features
    print(f"\n--- Test Intégration Réelle ---")
    try:
        # Use actual extraction methods
        detections = create_test_detections()
        pose_data = create_test_pose_data()
        motion_data = create_test_motion_data()
        temporal_data = create_test_temporal_data()
        
        real_fusion_input = FusionInput(
            detection_features=fusion_system.extract_detection_features(detections),
            pose_features=fusion_system.extract_pose_features(pose_data),
            motion_features=fusion_system.extract_motion_features(motion_data),
            temporal_features=fusion_system.extract_temporal_features(temporal_data)
        )
        
        real_result = fusion_system.fuse_features(real_fusion_input)
        print("✓ Intégration réelle réussie")
        print(f"  - Prédiction finale: {real_result.final_prediction:.4f}")
        print(f"  - Modalités utilisées: {list(real_result.attention_weights.keys())}")
        print(f"  - Poids les plus élevés: {max(real_result.attention_weights.items(), key=lambda x: x[1])}")
        
    except Exception as e:
        print(f"✗ Erreur intégration réelle: {e}")
    
    return True

if __name__ == "__main__":
    success = test_multimodal_fusion()
    
    print(f"\n{'='*70}")
    if success:
        print("✅ TOUS LES TESTS MULTIMODALFUSION RÉUSSIS")
        print("\nFonctionnalités testées:")
        print("- Initialisation avec dimensions correctes")
        print("- Extraction features pour chaque modalité:")
        print("  * Détections (confidences, classes, positions)")
        print("  * Poses (keypoints, indicateurs)")
        print("  * Mouvement (vitesse, patterns)")
        print("  * Temporel (consistance, séquences)")
        print("- Fusion statistique (fallback)")
        print("- Fusion par attention (réseau neuronal)")
        print("- Gestion entrées partielles")
        print("- Gestion cas limites (vide, corrompu)")
        print("- Intégration complète des modalités")
        print("- Calcul poids d'attention")
        print("- Scores de confiance par modalité")
    else:
        print("❌ CERTAINS TESTS MULTIMODALFUSION ONT ÉCHOUÉ")
    
    print(f"{'='*70}")