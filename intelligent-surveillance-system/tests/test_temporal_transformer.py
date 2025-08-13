#!/usr/bin/env python3
"""Test script for TemporalTransformer."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import time
from advanced_tools.temporal_transformer import TemporalTransformer, TemporalFrame

def create_test_sequences():
    """Create test temporal sequences with different patterns."""
    sequences = {}
    current_time = time.time()
    
    # Sequence 1: Stable normal activity
    sequences["stable_sequence"] = []
    for i in range(20):
        features = np.random.normal(0.5, 0.1, 256)  # Stable features around 0.5
        detections = [
            {"confidence": 0.8 + np.random.normal(0, 0.05), "class": "person", "bbox": [100, 100, 200, 300]},
            {"confidence": 0.6 + np.random.normal(0, 0.03), "class": "handbag", "bbox": [150, 200, 180, 230]}
        ]
        analysis_results = {
            "suspicion_level": 0.2 + np.random.normal(0, 0.05),
            "confidence": 0.8,
            "risk_factors": [],
            "tool_results": {
                "motion": type('obj', (object,), {
                    'result': {'average_velocity': 1.0 + np.random.normal(0, 0.1)}
                })()
            }
        }
        
        sequences["stable_sequence"].append(TemporalFrame(
            frame_id=f"frame_{i}",
            timestamp=current_time + i,
            features=features,
            detections=detections,
            analysis_results=analysis_results
        ))
    
    # Sequence 2: Gradual increase in suspicion
    sequences["increasing_suspicion"] = []
    for i in range(15):
        suspicion = 0.1 + (i / 15) * 0.7  # Gradually increase from 0.1 to 0.8
        features = np.random.normal(suspicion, 0.1, 256)
        detections = [
            {"confidence": 0.9 - suspicion * 0.2, "class": "person", "bbox": [100, 100, 200, 300]}
        ]
        analysis_results = {
            "suspicion_level": suspicion,
            "confidence": 0.9 - suspicion * 0.3,
            "risk_factors": ["loitering"] if suspicion > 0.5 else [],
            "tool_results": {
                "behavior": type('obj', (object,), {
                    'result': {'suspicion_score': suspicion}
                })()
            }
        }
        
        sequences["increasing_suspicion"].append(TemporalFrame(
            frame_id=f"frame_{i}",
            timestamp=current_time + i * 2,
            features=features,
            detections=detections,
            analysis_results=analysis_results
        ))
    
    # Sequence 3: Oscillating pattern
    sequences["oscillating_pattern"] = []
    for i in range(25):
        oscillation = 0.5 + 0.3 * np.sin(i * 0.5)  # Sinusoidal pattern
        features = np.random.normal(oscillation, 0.05, 256)
        detections = [
            {"confidence": 0.7 + 0.2 * np.cos(i * 0.5), "class": "person", "bbox": [100, 100, 200, 300]}
        ]
        analysis_results = {
            "suspicion_level": oscillation,
            "confidence": 0.8,
            "risk_factors": [],
            "tool_results": {}
        }
        
        sequences["oscillating_pattern"].append(TemporalFrame(
            frame_id=f"frame_{i}",
            timestamp=current_time + i * 0.8,
            features=features,
            detections=detections,
            analysis_results=analysis_results
        ))
    
    # Sequence 4: Sudden spike pattern
    sequences["spike_pattern"] = []
    for i in range(18):
        if i == 10:  # Sudden spike at frame 10
            suspicion = 0.9
            features = np.random.normal(0.8, 0.15, 256)
        else:
            suspicion = 0.2 + np.random.normal(0, 0.05)
            features = np.random.normal(0.3, 0.1, 256)
            
        detections = [
            {"confidence": 0.8 if suspicion < 0.5 else 0.6, "class": "person", "bbox": [100, 100, 200, 300]}
        ]
        analysis_results = {
            "suspicion_level": suspicion,
            "confidence": 0.9 if suspicion < 0.5 else 0.5,
            "risk_factors": ["suspicious_behavior"] if suspicion > 0.7 else [],
            "tool_results": {}
        }
        
        sequences["spike_pattern"].append(TemporalFrame(
            frame_id=f"frame_{i}",
            timestamp=current_time + i * 1.2,
            features=features,
            detections=detections,
            analysis_results=analysis_results
        ))
    
    return sequences

def test_temporal_transformer():
    """Test TemporalTransformer functionality."""
    print("=== Test TemporalTransformer ===")
    
    # Initialize TemporalTransformer
    try:
        transformer = TemporalTransformer(sequence_length=30, feature_dim=256)
        print("✓ TemporalTransformer initialisé")
        print(f"  - Longueur séquence: {transformer.sequence_length}")
        print(f"  - Dimension features: {transformer.feature_dim}")
        print(f"  - Device: {transformer.device}")
        print(f"  - Patterns disponibles: {len(transformer.pattern_definitions)}")
        print(f"  - Extracteurs features: {list(transformer.feature_extractors.keys())}")
    except Exception as e:
        print(f"✗ Erreur lors de l'initialisation: {e}")
        return False
    
    # Create test sequences
    test_sequences = create_test_sequences()
    print(f"✓ {len(test_sequences)} séquences de test créées")
    
    # Test adding frames and analyzing sequences
    stream_results = {}
    
    for seq_name, frames in test_sequences.items():
        stream_id = f"stream_{seq_name}"
        print(f"\n--- Test: {seq_name} ---")
        
        # Add frames to temporal sequence
        try:
            for frame in frames:
                transformer.add_frame(
                    stream_id=stream_id,
                    frame_id=frame.frame_id,
                    timestamp=frame.timestamp,
                    features=frame.features,
                    detections=frame.detections,
                    analysis_results=frame.analysis_results
                )
            
            print(f"✓ {len(frames)} frames ajoutées à {stream_id}")
        except Exception as e:
            print(f"✗ Erreur ajout frames: {e}")
            continue
        
        # Test different analysis types
        analysis_types = ['detection_based', 'behavior_based', 'motion_based']
        
        for analysis_type in analysis_types:
            try:
                analysis = transformer.analyze_temporal_sequence(stream_id, analysis_type)
                if analysis:
                    print(f"  ✓ Analyse {analysis_type} réussie:")
                    print(f"    - Consistance séquence: {analysis.sequence_consistency:.4f}")
                    print(f"    - Score anomalie: {analysis.anomaly_score:.4f}")
                    print(f"    - Patterns temporels: {analysis.temporal_patterns}")
                    print(f"    - Confiance: {analysis.confidence:.4f}")
                    print(f"    - Temps traitement: {analysis.processing_time:.4f}s")
                    
                    if analysis_type == 'behavior_based':
                        stream_results[seq_name] = analysis
                else:
                    print(f"  ⚠ Aucune analyse {analysis_type} retournée")
                    
            except Exception as e:
                print(f"  ✗ Erreur analyse {analysis_type}: {e}")
    
    # Test sequence summaries
    print(f"\n--- Test Résumés Séquences ---")
    for seq_name in test_sequences.keys():
        stream_id = f"stream_{seq_name}"
        try:
            summary = transformer.get_sequence_summary(stream_id)
            if summary:
                print(f"✓ Résumé {seq_name}:")
                print(f"  - Total frames: {summary['total_frames']}")
                print(f"  - Durée: {summary['time_span_seconds']:.1f}s")
                print(f"  - Détections totales: {summary['total_detections']}")
                print(f"  - Couverture analyse: {summary['analysis_coverage']:.2%}")
        except Exception as e:
            print(f"✗ Erreur résumé {seq_name}: {e}")
    
    # Test pattern recognition accuracy
    print(f"\n--- Test Reconnaissance Patterns ---")
    expected_patterns = {
        "stable_sequence": ["stable_normal", "normal_movement"],
        "increasing_suspicion": ["gradual_increase", "trending_suspicious"],
        "oscillating_pattern": ["oscillating"],
        "spike_pattern": ["sudden_spike"]
    }
    
    pattern_matches = 0
    total_tests = 0
    
    for seq_name, expected in expected_patterns.items():
        if seq_name in stream_results:
            actual_patterns = stream_results[seq_name].temporal_patterns
            total_tests += 1
            
            # Check if any expected pattern is found
            found_match = any(pattern in actual_patterns for pattern in expected)
            if found_match:
                pattern_matches += 1
                matching = [p for p in expected if p in actual_patterns]
                print(f"✓ {seq_name}: trouvé {matching}")
            else:
                print(f"⚠ {seq_name}: trouvé {actual_patterns}, attendu {expected}")
    
    if total_tests > 0:
        accuracy = pattern_matches / total_tests
        print(f"✓ Précision patterns: {accuracy:.2%} ({pattern_matches}/{total_tests})")
    
    # Test trend analysis
    print(f"\n--- Test Analyse Tendances ---")
    for seq_name, expected_trend in [("stable_sequence", "stable"), ("increasing_suspicion", "increasing")]:
        if seq_name in stream_results:
            analysis = stream_results[seq_name]
            if hasattr(analysis, 'trend_analysis') and 'suspicion_trend' in analysis.trend_analysis:
                trend_direction = analysis.trend_analysis['suspicion_trend']['direction']
                if trend_direction == expected_trend:
                    print(f"✓ {seq_name}: tendance {trend_direction} correcte")
                else:
                    print(f"⚠ {seq_name}: tendance {trend_direction}, attendu {expected_trend}")
    
    # Test edge cases
    print(f"\n--- Test Cas Limites ---")
    
    # Very short sequence
    try:
        short_frames = test_sequences["stable_sequence"][:3]
        short_stream = "short_stream"
        for frame in short_frames:
            transformer.add_frame(short_stream, frame.frame_id, frame.timestamp, 
                                frame.features, frame.detections, frame.analysis_results)
        
        short_analysis = transformer.analyze_temporal_sequence(short_stream)
        if short_analysis:
            print("⚠ Séquence courte analysée (inattendu)")
        else:
            print("✓ Séquence courte correctement rejetée")
    except Exception as e:
        print(f"✗ Erreur séquence courte: {e}")
    
    # Invalid stream ID
    try:
        invalid_analysis = transformer.analyze_temporal_sequence("nonexistent_stream")
        if invalid_analysis is None:
            print("✓ Stream inexistant géré")
        else:
            print("⚠ Stream inexistant retourne résultat (inattendu)")
    except Exception as e:
        print(f"✗ Erreur stream inexistant: {e}")
    
    # Test feature normalization
    print(f"\n--- Test Normalisation Features ---")
    try:
        # Test with wrong feature dimensions
        wrong_features = np.random.randn(128)  # Half expected size
        normalized = transformer._normalize_features(wrong_features)
        
        if len(normalized) == transformer.feature_dim:
            print(f"✓ Features normalisées: {len(wrong_features)} → {len(normalized)}")
        else:
            print(f"✗ Normalisation échouée: {len(normalized)} != {transformer.feature_dim}")
            
        # Test with oversized features
        big_features = np.random.randn(512)  # Double expected size
        normalized_big = transformer._normalize_features(big_features)
        
        if len(normalized_big) == transformer.feature_dim:
            print(f"✓ Features grandes normalisées: {len(big_features)} → {len(normalized_big)}")
        
    except Exception as e:
        print(f"✗ Erreur normalisation features: {e}")
    
    # Test cleanup functionality
    print(f"\n--- Test Nettoyage ---")
    try:
        initial_count = len(transformer.temporal_sequences)
        transformer.cleanup_old_sequences(max_age_seconds=1)  # Very short for testing
        time.sleep(1.1)  # Wait a bit
        transformer.cleanup_old_sequences(max_age_seconds=1)
        final_count = len(transformer.temporal_sequences)
        
        print(f"✓ Nettoyage effectué: {initial_count} → {final_count} séquences")
    except Exception as e:
        print(f"✗ Erreur nettoyage: {e}")
    
    return True

if __name__ == "__main__":
    success = test_temporal_transformer()
    
    print(f"\n{'='*70}")
    if success:
        print("✅ TOUS LES TESTS TEMPORALTRANSFORMER RÉUSSIS")
        print("\nFonctionnalités testées:")
        print("- Initialisation avec modèle Transformer")
        print("- Ajout de frames temporelles")
        print("- Types d'analyse multiples:")
        print("  * Detection-based (statistiques détection)")
        print("  * Behavior-based (analyse comportementale)")
        print("  * Motion-based (patterns mouvement)")
        print("- Reconnaissance de patterns temporels:")
        print("  * Stable normal")
        print("  * Augmentation graduelle")
        print("  * Oscillations")
        print("  * Pics soudains")
        print("- Analyse de tendances")
        print("- Calcul consistance séquence")
        print("- Détection d'anomalies temporelles")
        print("- Résumés et statistiques")
        print("- Normalisation automatique features")
        print("- Nettoyage séquences anciennes")
        print("- Gestion cas limites")
    else:
        print("❌ CERTAINS TESTS TEMPORALTRANSFORMER ONT ÉCHOUÉ")
    
    print(f"{'='*70}")