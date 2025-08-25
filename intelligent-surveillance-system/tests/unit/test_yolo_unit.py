"""
🎯 Tests Unitaires YOLO - Validation isolation du détecteur d'objets
====================================================================

Tests spécialisés pour valider :
- Performance de détection YOLO isolée
- Précision sur différentes classes d'objets
- Robustesse avec différentes résolutions d'images
- Optimisation GPU pour inférence rapide
"""

import pytest
import torch
import time
import numpy as np
import cv2
from typing import List, Dict, Tuple
from PIL import Image, ImageDraw
import tempfile
import json

from src.detection.yolo_detector import YOLODetector
from src.core.types import DetectedObject, BoundingBox
from src.utils.performance import measure_time, get_current_performance


class TestYOLOUnitIsolated:
    """Tests unitaires isolés pour le détecteur YOLO."""
    
    @pytest.fixture
    def yolo_detector(self) -> YOLODetector:
        """Instance YOLO optimisée pour tests."""
        return YOLODetector(
            model_name="yolov8n.pt",  # Modèle léger pour tests rapides
            device="auto",
            confidence_threshold=0.5,
            iou_threshold=0.45
        )
    
    @pytest.fixture
    def test_images(self) -> Dict[str, np.ndarray]:
        """Images de test synthétiques avec objets connus."""
        images = {}
        
        # Image avec personne simulée (rectangle + cercle pour tête)
        img_person = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(img_person, (200, 150), (300, 400), (100, 100, 100), -1)  # Corps
        cv2.circle(img_person, (250, 120), 30, (150, 150, 150), -1)  # Tête
        images["person"] = img_person
        
        # Image avec objets multiples
        img_multi = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(img_multi, (50, 200), (150, 300), (0, 255, 0), -1)  # Objet 1
        cv2.rectangle(img_multi, (400, 100), (500, 250), (255, 0, 0), -1)  # Objet 2
        cv2.circle(img_multi, (300, 350), 50, (0, 0, 255), -1)  # Objet 3
        images["multi_objects"] = img_multi
        
        # Image vide (test cas limite)
        img_empty = np.ones((480, 640, 3), dtype=np.uint8) * 128  # Gris uniforme
        images["empty"] = img_empty
        
        # Image haute résolution
        img_hd = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        images["hd"] = img_hd
        
        return images
    
    def test_yolo_initialization(self, yolo_detector):
        """Test 2.1: Initialisation et chargement du modèle YOLO."""
        # Vérification état initial
        assert yolo_detector is not None
        assert yolo_detector.model is not None
        assert yolo_detector.device in ['cpu', 'cuda']
        assert 0.0 <= yolo_detector.confidence_threshold <= 1.0
        assert 0.0 <= yolo_detector.iou_threshold <= 1.0
        
        # Test des classes supportées
        class_names = yolo_detector.get_class_names()
        assert "person" in class_names
        assert len(class_names) > 10  # COCO dataset a 80 classes
        
        print(f"✅ YOLO initialisé: {len(class_names)} classes, device={yolo_detector.device}")
    
    @measure_time
    def test_detection_performance_single(self, yolo_detector, test_images):
        """Test 2.2: Performance détection image unique."""
        start_time = time.time()
        
        detections = yolo_detector.detect(test_images["person"])
        detection_time = time.time() - start_time
        
        # Validations performance
        assert detection_time < 1.0, f"Détection trop lente: {detection_time:.3f}s"
        assert isinstance(detections, list)
        
        # Validation structure des détections
        for detection in detections:
            assert isinstance(detection, DetectedObject)
            assert hasattr(detection, 'class_name')
            assert hasattr(detection, 'confidence') 
            assert hasattr(detection, 'bbox')
            assert 0.0 <= detection.confidence <= 1.0
        
        print(f"✅ Détection: {len(detections)} objets en {detection_time:.3f}s")
        return {
            "detection_time": detection_time,
            "objects_detected": len(detections),
            "average_confidence": np.mean([d.confidence for d in detections]) if detections else 0.0
        }
    
    def test_detection_batch_performance(self, yolo_detector, test_images):
        """Test 2.3: Performance détection batch."""
        image_batch = [
            test_images["person"],
            test_images["multi_objects"],
            test_images["empty"]
        ]
        
        start_time = time.time()
        batch_results = yolo_detector.detect_batch(image_batch)
        batch_time = time.time() - start_time
        
        # Validations
        assert len(batch_results) == len(image_batch)
        assert batch_time < 2.0, f"Batch trop lent: {batch_time:.3f}s"
        
        # Vérification cohérence avec détections individuelles
        individual_time = 0
        for img in image_batch:
            start = time.time()
            yolo_detector.detect(img)
            individual_time += time.time() - start
        
        speedup = individual_time / batch_time if batch_time > 0 else 1
        print(f"✅ Batch: {len(image_batch)} images en {batch_time:.3f}s (speedup: {speedup:.2f}x)")
        
        return {
            "batch_time": batch_time,
            "individual_time": individual_time,
            "speedup": speedup,
            "images_processed": len(image_batch)
        }
    
    def test_detection_accuracy_validation(self, yolo_detector, test_images):
        """Test 2.4: Validation précision détection."""
        results = {}
        
        # Test image avec personne
        person_detections = yolo_detector.detect(test_images["person"])
        person_found = any(d.class_name == "person" for d in person_detections)
        results["person_detection"] = {
            "found": person_found,
            "total_objects": len(person_detections),
            "confidences": [d.confidence for d in person_detections if d.class_name == "person"]
        }
        
        # Test image multi-objets
        multi_detections = yolo_detector.detect(test_images["multi_objects"])
        results["multi_objects"] = {
            "total_detected": len(multi_detections),
            "classes_detected": [d.class_name for d in multi_detections]
        }
        
        # Test image vide
        empty_detections = yolo_detector.detect(test_images["empty"])
        results["empty_image"] = {
            "false_positives": len(empty_detections),
            "should_be_zero": len(empty_detections) == 0
        }
        
        print(f"✅ Précision: Personne détectée={person_found}, "
              f"Multi-objets={len(multi_detections)}, "
              f"Faux positifs={len(empty_detections)}")
        
        return results
    
    def test_resolution_robustness(self, yolo_detector, test_images):
        """Test 2.5: Robustesse selon résolution."""
        resolutions_test = [
            (320, 240),   # Très petite
            (640, 480),   # Standard
            (1280, 720),  # HD
            (1920, 1080), # Full HD
        ]
        
        base_image = test_images["person"]
        results = {}
        
        for width, height in resolutions_test:
            # Redimensionnement
            resized = cv2.resize(base_image, (width, height))
            
            # Détection avec mesure de temps
            start_time = time.time()
            detections = yolo_detector.detect(resized)
            detection_time = time.time() - start_time
            
            results[f"{width}x{height}"] = {
                "detection_time": detection_time,
                "objects_detected": len(detections),
                "resolution_pixels": width * height
            }
        
        # Analyse de la scalabilité
        for res, data in results.items():
            print(f"📐 {res}: {data['objects_detected']} objets en {data['detection_time']:.3f}s")
        
        return results
    
    def test_confidence_threshold_impact(self, yolo_detector, test_images):
        """Test 2.6: Impact du seuil de confiance."""
        thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
        original_threshold = yolo_detector.confidence_threshold
        
        results = {}
        
        for threshold in thresholds:
            yolo_detector.confidence_threshold = threshold
            detections = yolo_detector.detect(test_images["multi_objects"])
            
            results[threshold] = {
                "total_detections": len(detections),
                "high_confidence": len([d for d in detections if d.confidence > 0.8]),
                "average_confidence": np.mean([d.confidence for d in detections]) if detections else 0.0
            }
        
        # Restauration seuil original
        yolo_detector.confidence_threshold = original_threshold
        
        # Validation cohérence : seuil plus élevé = moins de détections
        detection_counts = [results[t]["total_detections"] for t in thresholds]
        is_decreasing = all(detection_counts[i] >= detection_counts[i+1] for i in range(len(detection_counts)-1))
        
        assert is_decreasing, "Le nombre de détections devrait diminuer avec le seuil"
        
        print(f"✅ Seuils testés: {dict(zip(thresholds, detection_counts))}")
        return results


@pytest.mark.gpu
class TestYOLOGPUOptimized:
    """Tests optimisés GPU pour YOLO."""
    
    def test_gpu_vs_cpu_performance(self):
        """Test GPU: Comparaison performance CPU vs GPU."""
        if not torch.cuda.is_available():
            pytest.skip("GPU non disponible")
        
        # Création détecteurs CPU/GPU
        yolo_cpu = YOLODetector(device="cpu")
        yolo_gpu = YOLODetector(device="cuda")
        
        # Image test
        test_image = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
        
        # Benchmark CPU
        cpu_times = []
        for _ in range(10):
            start = time.time()
            yolo_cpu.detect(test_image)
            cpu_times.append(time.time() - start)
        
        # Benchmark GPU
        gpu_times = []
        for _ in range(10):
            start = time.time()
            yolo_gpu.detect(test_image)
            gpu_times.append(time.time() - start)
        
        avg_cpu = np.mean(cpu_times)
        avg_gpu = np.mean(gpu_times)
        speedup = avg_cpu / avg_gpu if avg_gpu > 0 else 1
        
        print(f"🚀 Performance: CPU={avg_cpu:.3f}s, GPU={avg_gpu:.3f}s")
        print(f"📈 Speedup GPU: {speedup:.2f}x")
        
        return {
            "cpu_time": avg_cpu,
            "gpu_time": avg_gpu,
            "speedup": speedup
        }
    
    def test_memory_usage_gpu(self):
        """Test GPU: Utilisation mémoire GPU."""
        if not torch.cuda.is_available():
            pytest.skip("GPU non disponible")
        
        yolo_detector = YOLODetector(device="cuda")
        
        # Mesure mémoire initiale
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()
        
        # Test avec images de différentes tailles
        sizes = [(320, 240), (640, 480), (1280, 720), (1920, 1080)]
        memory_usage = {}
        
        for width, height in sizes:
            torch.cuda.empty_cache()
            img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            
            # Détection
            _ = yolo_detector.detect(img)
            memory_used = torch.cuda.memory_allocated() - initial_memory
            
            memory_usage[f"{width}x{height}"] = memory_used / (1024**2)  # MB
        
        print("💾 Utilisation mémoire GPU:")
        for res, mem in memory_usage.items():
            print(f"   {res}: {mem:.1f} MB")
        
        return memory_usage


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])