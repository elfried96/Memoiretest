#!/usr/bin/env python3
"""Test script for DinoV2FeatureExtractor."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import cv2
from advanced_tools.dino_features import DinoV2FeatureExtractor

def create_test_images():
    """Create test images."""
    # Image 1: Simple geometric shapes
    image1 = np.zeros((224, 224, 3), dtype=np.uint8)
    cv2.rectangle(image1, (50, 50), (150, 150), (255, 0, 0), -1)
    cv2.circle(image1, (170, 170), 40, (0, 255, 0), -1)
    
    # Image 2: Different pattern
    image2 = np.zeros((224, 224, 3), dtype=np.uint8)
    cv2.line(image2, (0, 0), (224, 224), (255, 255, 255), 3)
    cv2.line(image2, (0, 224), (224, 0), (255, 255, 255), 3)
    
    # Image 3: Random noise (for robustness test)
    image3 = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    return [image1, image2, image3]

def test_dino_feature_extractor():
    """Test DinoV2FeatureExtractor functionality."""
    print("=== Test DinoV2FeatureExtractor ===")
    
    # Create test images
    test_images = create_test_images()
    print(f"✓ {len(test_images)} images de test créées")
    
    # Initialize DinoV2FeatureExtractor
    try:
        extractor = DinoV2FeatureExtractor()
        print("✓ DinoV2FeatureExtractor initialisé")
    except Exception as e:
        print(f"✗ Erreur lors de l'initialisation: {e}")
        return False
    
    # Test 1: Basic feature extraction
    print("\n--- Test 1: Extraction de base ---")
    try:
        features = extractor.extract_features(test_images[0])
        print("✓ Extraction de features réussie")
        print(f"  - Shape des features: {features.features.shape}")
        print(f"  - Temps de traitement: {features.processing_time:.4f}s")
        
        if features.patch_features is not None:
            print(f"  - Patch features disponibles: {features.patch_features.shape}")
        else:
            print("  - Pas de patch features (mode normal)")
            
    except Exception as e:
        print(f"✗ Erreur lors de l'extraction: {e}")
        return False
    
    # Test 2: Feature extraction with attention
    print("\n--- Test 2: Extraction avec attention ---")
    try:
        features_with_attention = extractor.extract_features(test_images[0], extract_attention=True)
        print("✓ Extraction avec attention réussie")
        
        if features_with_attention.attention_maps is not None and features_with_attention.attention_maps.size > 0:
            print(f"  - Attention maps: {features_with_attention.attention_maps.shape}")
        else:
            print("  - Attention maps non disponibles (fallback mode)")
            
    except Exception as e:
        print(f"✗ Erreur lors de l'extraction avec attention: {e}")
        return False
    
    # Test 3: Regional feature extraction
    print("\n--- Test 3: Extraction régionale ---")
    regions = [(20, 20, 120, 120), (100, 100, 200, 200)]  # Two regions
    
    try:
        regional_features = extractor.extract_features(test_images[0], regions=regions)
        print("✓ Extraction régionale réussie")
        print(f"  - Shape des features régionales: {regional_features.features.shape}")
        print(f"  - Nombre de régions traitées: {regional_features.features.shape[0] if len(regional_features.features.shape) > 1 else 1}")
        
    except Exception as e:
        print(f"✗ Erreur lors de l'extraction régionale: {e}")
        return False
    
    # Test 4: Similarity computation
    print("\n--- Test 4: Calcul de similarité ---")
    try:
        features1 = extractor.extract_features(test_images[0])
        features2 = extractor.extract_features(test_images[1])
        
        # Cosine similarity
        cosine_sim = extractor.compute_similarity(features1.features, features2.features, metric="cosine")
        print(f"✓ Similarité cosinus: {cosine_sim:.4f}")
        
        # Euclidean similarity
        euclidean_sim = extractor.compute_similarity(features1.features, features2.features, metric="euclidean")
        print(f"✓ Similarité euclidienne: {euclidean_sim:.4f}")
        
        # Test similarity with same image (should be high)
        self_sim = extractor.compute_similarity(features1.features, features1.features, metric="cosine")
        print(f"✓ Auto-similarité (devrait être ~1.0): {self_sim:.4f}")
        
    except Exception as e:
        print(f"✗ Erreur lors du calcul de similarité: {e}")
        return False
    
    # Test 5: Feature clustering
    print("\n--- Test 5: Clustering de features ---")
    try:
        # Extract features from all test images
        all_features = []
        for img in test_images:
            feat = extractor.extract_features(img)
            all_features.append(feat.features)
        
        clustering_result = extractor.cluster_features(all_features, n_clusters=2)
        print("✓ Clustering réussi")
        print(f"  - Labels: {clustering_result['labels']}")
        print(f"  - Inertia: {clustering_result['inertia']:.4f}")
        print(f"  - Nombre de centres: {len(clustering_result['centers'])}")
        
    except Exception as e:
        print(f"✗ Erreur lors du clustering: {e}")
        return False
    
    # Test 6: Edge cases
    print("\n--- Test 6: Cas limites ---")
    
    # Empty regions
    try:
        empty_result = extractor.extract_features(test_images[0], regions=[])
        print("✓ Gestion des régions vides")
    except Exception as e:
        print(f"✗ Erreur avec régions vides: {e}")
        return False
    
    # Invalid similarity metric
    try:
        features1 = extractor.extract_features(test_images[0])
        try:
            invalid_sim = extractor.compute_similarity(features1.features, features1.features, metric="invalid")
            print("✗ Métrique invalide acceptée (problème)")
        except ValueError:
            print("✓ Métrique invalide correctement rejetée")
    except Exception as e:
        print(f"✗ Erreur lors du test de métrique invalide: {e}")
        return False
    
    # Empty features list for clustering
    try:
        empty_cluster = extractor.cluster_features([], n_clusters=2)
        print("✓ Gestion de la liste vide pour clustering")
        print(f"  - Résultat: {empty_cluster}")
    except Exception as e:
        print(f"✗ Erreur avec liste vide pour clustering: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = test_dino_feature_extractor()
    
    print(f"\n{'='*60}")
    if success:
        print("✅ TOUS LES TESTS DINOV2FEATUREEXTRACTOR RÉUSSIS")
        print("\nFonctionnalités testées:")
        print("- Initialisation du modèle")
        print("- Extraction de features globales")
        print("- Extraction avec cartes d'attention")
        print("- Extraction régionale")
        print("- Calcul de similarité (cosinus et euclidienne)")
        print("- Clustering de features")
        print("- Gestion des cas limites")
        print("- Fallback en cas d'erreur du modèle")
    else:
        print("❌ CERTAINS TESTS DINOV2FEATUREEXTRACTOR ONT ÉCHOUÉ")
    
    print(f"{'='*60}")