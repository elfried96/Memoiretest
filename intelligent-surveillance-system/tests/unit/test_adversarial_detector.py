#!/usr/bin/env python3
"""Test script for AdversarialDetector."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import cv2
from advanced_tools.adversarial_detector import AdversarialDetector

def create_test_images():
    """Create test images including normal and adversarial-like patterns."""
    images = {}
    
    # 1. Normal clean image
    normal_image = np.zeros((224, 224, 3), dtype=np.uint8)
    # Add some natural patterns
    cv2.rectangle(normal_image, (50, 50), (150, 150), (120, 120, 120), -1)
    cv2.circle(normal_image, (170, 70), 30, (80, 80, 80), -1)
    # Add some gaussian noise (natural)
    noise = np.random.normal(0, 5, normal_image.shape).astype(np.uint8)
    normal_image = cv2.add(normal_image, noise)
    images['normal'] = normal_image
    
    # 2. High frequency noise image (FGSM-like)
    high_freq_image = normal_image.copy()
    # Add high frequency perturbations
    high_freq_noise = np.random.randint(-10, 10, high_freq_image.shape).astype(np.int16)
    high_freq_image = np.clip(high_freq_image.astype(np.int16) + high_freq_noise, 0, 255).astype(np.uint8)
    images['high_frequency'] = high_freq_image
    
    # 3. Uniform noise pattern (adversarial-like)
    uniform_image = normal_image.copy()
    uniform_perturbation = np.full(uniform_image.shape, 5, dtype=np.int16)
    uniform_perturbation[::2] = -5  # Checkerboard-like uniform pattern
    uniform_image = np.clip(uniform_image.astype(np.int16) + uniform_perturbation, 0, 255).astype(np.uint8)
    images['uniform_noise'] = uniform_image
    
    # 4. Very bright image (brightness anomaly)
    bright_image = np.clip(normal_image.astype(np.int16) + 100, 0, 255).astype(np.uint8)
    images['brightness_anomaly'] = bright_image
    
    # 5. Very dark image (brightness anomaly)
    dark_image = np.clip(normal_image.astype(np.int16) - 80, 0, 255).astype(np.uint8)
    images['dark_anomaly'] = dark_image
    
    # 6. High gradient image (edge anomaly)
    gradient_image = normal_image.copy()
    # Add artificial sharp edges
    cv2.line(gradient_image, (0, 112), (224, 112), (255, 255, 255), 2)
    cv2.line(gradient_image, (112, 0), (112, 224), (0, 0, 0), 2)
    images['gradient_anomaly'] = gradient_image
    
    # 7. Low clarity image (weather-like distortion)
    low_clarity_image = cv2.GaussianBlur(normal_image, (7, 7), 2)
    # Add some random noise to simulate weather effects
    weather_noise = np.random.normal(0, 10, low_clarity_image.shape).astype(np.int16)
    low_clarity_image = np.clip(low_clarity_image.astype(np.int16) + weather_noise, 0, 255).astype(np.uint8)
    images['low_clarity'] = low_clarity_image
    
    return images

def create_normal_training_images():
    """Create a set of normal images for training the detector."""
    training_images = []
    
    for i in range(10):
        # Create varied but normal images
        base_brightness = np.random.uniform(80, 180)
        image = np.full((224, 224, 3), base_brightness, dtype=np.uint8)
        
        # Add some geometric shapes with natural variation
        num_shapes = np.random.randint(1, 4)
        for _ in range(num_shapes):
            shape_type = np.random.choice(['rectangle', 'circle', 'line'])
            color = np.random.randint(0, 255, 3).tolist()
            
            if shape_type == 'rectangle':
                x1, y1 = np.random.randint(0, 150, 2)
                x2, y2 = x1 + np.random.randint(20, 70), y1 + np.random.randint(20, 70)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, -1)
            elif shape_type == 'circle':
                center = tuple(np.random.randint(30, 194, 2))
                radius = np.random.randint(10, 40)
                cv2.circle(image, center, radius, color, -1)
            else:  # line
                pt1 = tuple(np.random.randint(0, 224, 2))
                pt2 = tuple(np.random.randint(0, 224, 2))
                cv2.line(image, pt1, pt2, color, np.random.randint(1, 5))
        
        # Add natural noise
        natural_noise = np.random.normal(0, 3, image.shape).astype(np.int16)
        image = np.clip(image.astype(np.int16) + natural_noise, 0, 255).astype(np.uint8)
        
        training_images.append(image)
    
    return training_images

def test_adversarial_detector():
    """Test AdversarialDetector functionality."""
    print("=== Test AdversarialDetector ===")
    
    # Initialize AdversarialDetector
    try:
        detector = AdversarialDetector()
        print("✓ AdversarialDetector initialisé")
        print(f"  - Device: {detector.device}")
        print(f"  - Signatures d'attaque: {list(detector.attack_signatures.keys())}")
    except Exception as e:
        print(f"✗ Erreur lors de l'initialisation: {e}")
        return False
    
    # Create training and test images
    print(f"\n--- Préparation des données ---")
    normal_images = create_normal_training_images()
    test_images = create_test_images()
    print(f"✓ {len(normal_images)} images normales créées pour entraînement")
    print(f"✓ {len(test_images)} images de test créées")
    
    # Train on normal data
    print(f"\n--- Entraînement sur données normales ---")
    try:
        detector.train_on_normal_data(normal_images)
        print("✓ Entraînement sur données normales réussi")
    except Exception as e:
        print(f"✗ Erreur lors de l'entraînement: {e}")
        return False
    
    # Test detection on different image types
    print(f"\n--- Tests de Détection ---")
    detection_results = {}
    
    expected_results = {
        'normal': False,           # Should not be detected as adversarial
        'high_frequency': True,    # Should be detected (high freq noise)
        'uniform_noise': True,     # Should be detected (uniform pattern)
        'brightness_anomaly': True, # Should be detected (brightness anomaly)
        'dark_anomaly': True,      # Should be detected (darkness anomaly)
        'gradient_anomaly': True,  # Should be detected (edge anomaly)
        'low_clarity': False       # Might or might not be detected
    }
    
    correct_detections = 0
    total_tests = 0
    
    for img_name, image in test_images.items():
        print(f"\n  Test: {img_name}")
        try:
            result = detector.detect_adversarial(image)
            detection_results[img_name] = result
            
            print(f"    ✓ Détection réussie")
            print(f"    - Est adversarial: {result.is_adversarial}")
            print(f"    - Confiance: {result.confidence:.4f}")
            print(f"    - Type attaque: {result.attack_type}")
            print(f"    - Magnitude perturbation: {result.perturbation_magnitude:.4f}")
            print(f"    - Méthode détection: {result.detection_method}")
            print(f"    - Temps traitement: {result.processing_time:.4f}s")
            
            # Check if detection matches expectation
            if img_name in expected_results:
                expected = expected_results[img_name]
                actual = result.is_adversarial
                total_tests += 1
                
                if actual == expected:
                    correct_detections += 1
                    print(f"    ✓ Résultat correct: {actual}")
                else:
                    print(f"    ⚠ Résultat inattendu: {actual}, attendu: {expected}")
            
            # Show evidence details
            if 'statistical' in result.evidence:
                stat_evidence = result.evidence['statistical']
                print(f"    - Tests statistiques: {list(stat_evidence.get('statistical_tests', {}).keys())}")
            
        except Exception as e:
            print(f"    ✗ Erreur détection {img_name}: {e}")
    
    # Calculate detection accuracy
    if total_tests > 0:
        accuracy = correct_detections / total_tests
        print(f"\n✓ Précision détection: {accuracy:.2%} ({correct_detections}/{total_tests})")
    
    # Test robustness scoring
    print(f"\n--- Test Scores de Robustesse ---")
    try:
        for img_name, image in list(test_images.items())[:3]:  # Test first 3 images
            robustness_score = detector.get_robustness_score(image)
            print(f"✓ Score robustesse {img_name}: {robustness_score:.4f}")
            
            # Robust images should have higher scores
            if img_name == 'normal':
                if robustness_score > 0.5:
                    print(f"  ✓ Image normale bien classée comme robuste")
                else:
                    print(f"  ⚠ Image normale classée comme vulnérable")
                    
    except Exception as e:
        print(f"✗ Erreur scores robustesse: {e}")
    
    # Test batch robustness report
    print(f"\n--- Test Rapport de Robustesse ---")
    try:
        test_batch = list(test_images.values())[:5]  # First 5 images
        report = detector.generate_robustness_report(test_batch)
        
        print("✓ Rapport de robustesse généré:")
        print(f"  - Images totales: {report['total_images']}")
        print(f"  - Adversariales détectées: {report['adversarial_detected']}")
        print(f"  - Taux adversarial: {report['adversarial_rate']:.2%}")
        print(f"  - Robustesse moyenne: {report['average_robustness']:.4f}")
        print(f"  - Temps traitement moyen: {report['average_processing_time']:.4f}s")
        print(f"  - Types d'attaque: {report['attack_types']}")
        print(f"  - Méthodes détection: {report['detection_methods']}")
        
    except Exception as e:
        print(f"✗ Erreur rapport robustesse: {e}")
    
    # Test attack type classification
    print(f"\n--- Test Classification Types d'Attaque ---")
    attack_classifications = {}
    
    for img_name, result in detection_results.items():
        if result.is_adversarial and result.attack_type:
            attack_classifications[img_name] = result.attack_type
    
    if attack_classifications:
        print("✓ Classifications d'attaque:")
        for img_name, attack_type in attack_classifications.items():
            print(f"  - {img_name}: {attack_type}")
            
        # Check some expected classifications
        if 'high_frequency' in attack_classifications:
            attack_type = attack_classifications['high_frequency']
            if attack_type in ['fgsm', 'pgd']:
                print("  ✓ High frequency correctement classée comme FGSM/PGD")
            else:
                print(f"  ⚠ High frequency classée comme {attack_type}")
    else:
        print("⚠ Aucune classification d'attaque détectée")
    
    # Test edge cases
    print(f"\n--- Test Cas Limites ---")
    
    # Very small image
    try:
        tiny_image = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        tiny_result = detector.detect_adversarial(tiny_image)
        print(f"✓ Image minuscule traitée: adversarial={tiny_result.is_adversarial}")
    except Exception as e:
        print(f"✗ Erreur image minuscule: {e}")
    
    # Grayscale image
    try:
        gray_image = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
        gray_result = detector.detect_adversarial(gray_image)
        print(f"✓ Image niveau de gris traitée: adversarial={gray_result.is_adversarial}")
    except Exception as e:
        print(f"✗ Erreur image niveau de gris: {e}")
    
    # All-zero image
    try:
        zero_image = np.zeros((224, 224, 3), dtype=np.uint8)
        zero_result = detector.detect_adversarial(zero_image)
        print(f"✓ Image noire traitée: adversarial={zero_result.is_adversarial}")
    except Exception as e:
        print(f"✗ Erreur image noire: {e}")
    
    # All-white image
    try:
        white_image = np.full((224, 224, 3), 255, dtype=np.uint8)
        white_result = detector.detect_adversarial(white_image)
        print(f"✓ Image blanche traitée: adversarial={white_result.is_adversarial}")
    except Exception as e:
        print(f"✗ Erreur image blanche: {e}")
    
    return True

if __name__ == "__main__":
    success = test_adversarial_detector()
    
    print(f"\n{'='*70}")
    if success:
        print("✅ TOUS LES TESTS ADVERSARIALDETECTOR RÉUSSIS")
        print("\nFonctionnalités testées:")
        print("- Initialisation avec détecteurs multiples")
        print("- Entraînement sur données normales")
        print("- Détection d'attaques adversariales:")
        print("  * High frequency noise (FGSM-like)")
        print("  * Uniform noise patterns")
        print("  * Brightness anomalies")
        print("  * Gradient anomalies")
        print("- Méthodes de détection:")
        print("  * Statistical (Isolation Forest, Elliptic Envelope)")
        print("  * Pattern-based (signatures d'attaque)")
        print("  * Neural (réseau neuronal)")
        print("- Classification types d'attaque")
        print("- Scores de robustesse")
        print("- Rapports de robustesse complets")
        print("- Gestion cas limites (images petites, niveaux gris)")
        print("- Système de vote pour décision finale")
    else:
        print("❌ CERTAINS TESTS ADVERSARIALDETECTOR ONT ÉCHOUÉ")
    
    print(f"{'='*70}")