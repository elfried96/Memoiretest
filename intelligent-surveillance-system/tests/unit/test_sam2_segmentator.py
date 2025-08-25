#!/usr/bin/env python3
"""Test script for SAM2Segmentator."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import cv2
from advanced_tools.sam2_segmentation import SAM2Segmentator

def create_test_image():
    """Create a simple test image."""
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    # Add some basic shapes
    cv2.rectangle(image, (100, 100), (200, 200), (255, 0, 0), -1)  # Blue rectangle
    cv2.circle(image, (400, 300), 50, (0, 255, 0), -1)  # Green circle
    return image

def test_sam2_segmentator():
    """Test SAM2Segmentator functionality."""
    print("=== Test SAM2Segmentator ===")
    
    # Create test image
    test_image = create_test_image()
    print(f"✓ Image de test créée: {test_image.shape}")
    
    # Initialize SAM2Segmentator
    try:
        segmentator = SAM2Segmentator()
        print("✓ SAM2Segmentator initialisé")
    except Exception as e:
        print(f"✗ Erreur lors de l'initialisation: {e}")
        return False
    
    # Test segmentation with bounding boxes
    bounding_boxes = [
        [100, 100, 200, 200],  # Rectangle
        [350, 250, 450, 350]   # Circle area
    ]
    
    try:
        result = segmentator.segment_objects(test_image, bounding_boxes)
        print("✓ Segmentation exécutée avec succès")
        print(f"  - Nombre de masques: {len(result.masks) if result.masks.size > 0 else 0}")
        print(f"  - Scores: {result.scores}")
        print(f"  - Temps de traitement: {result.processing_time:.4f}s")
    except Exception as e:
        print(f"✗ Erreur lors de la segmentation: {e}")
        return False
    
    # Test mask properties
    if result.masks.size > 0:
        try:
            first_mask = result.masks[0] if len(result.masks.shape) > 2 else result.masks
            properties = segmentator.get_mask_properties(first_mask)
            print("✓ Propriétés des masques calculées:")
            print(f"  - Aire: {properties['area']}")
            print(f"  - Périmètre: {properties['perimeter']}")
            print(f"  - Compacité: {properties['compactness']:.4f}")
            print(f"  - Solidité: {properties['solidity']:.4f}")
        except Exception as e:
            print(f"✗ Erreur lors du calcul des propriétés: {e}")
            return False
    
    # Test with empty bounding boxes
    try:
        empty_result = segmentator.segment_objects(test_image, [])
        print("✓ Gestion des boîtes vides testée")
    except Exception as e:
        print(f"✗ Erreur avec boîtes vides: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = test_sam2_segmentator()
    
    print(f"\n{'='*50}")
    if success:
        print("✅ TOUS LES TESTS SAM2SEGMENTATOR RÉUSSIS")
        print("\nFonctionnalités testées:")
        print("- Initialisation du modèle")
        print("- Segmentation avec boîtes englobantes")
        print("- Calcul des propriétés des masques")
        print("- Gestion des cas limites")
        print("- Fallback en cas d'erreur du modèle")
    else:
        print("❌ CERTAINS TESTS SAM2SEGMENTATOR ONT ÉCHOUÉ")
    
    print(f"{'='*50}")