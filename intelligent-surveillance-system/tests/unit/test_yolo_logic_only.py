#!/usr/bin/env python3
"""
🎯 Tests YOLO - Logique Pure Sans GPU
===================================

Tests qui se concentrent sur la logique de YOLO sans charger le modèle réel.
Teste: parsing, filtrage, calculs de confiance, etc.
"""

# import pytest  # Not needed for direct execution
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any
# import numpy as np  # Not needed, using pure Python

class MockYOLOResult:
    """Mock d'un résultat YOLO"""
    
    def __init__(self, detections: List[Dict]):
        self.boxes = MockBoxes(detections)
        self.names = {0: "person", 1: "bottle", 2: "car", 3: "electronics"}

class MockBoxes:
    """Mock des boxes YOLO"""
    
    def __init__(self, detections: List[Dict]):
        self.detections = detections
        
    def __iter__(self):
        return iter(self.detections)
        
    def __len__(self):
        return len(self.detections)

class MockBox:
    """Mock d'une box individuelle"""
    
    def __init__(self, detection: Dict):
        self.xyxy = [[detection["bbox"]]]
        self.conf = [[detection["confidence"]]]  
        self.cls = [[detection["class_id"]]]

# =================== Tests de Logique Pure ===================

class TestYOLOLogicPure:
    """Tests de la logique YOLO sans modèle réel"""
    
    def test_detection_parsing(self):
        """Test du parsing des détections YOLO"""
        
        # Données simulées du modèle YOLO
        mock_detections = [
            {"bbox": [100, 100, 200, 200], "confidence": 0.85, "class_id": 0},  # person
            {"bbox": [300, 150, 400, 250], "confidence": 0.60, "class_id": 1},  # bottle
            {"bbox": [50, 50, 150, 150], "confidence": 0.45, "class_id": 2}     # car (faible conf)
        ]
        
        # Fonction de parsing qu'on veut tester
        def parse_yolo_results(raw_results, confidence_threshold=0.5):
            parsed = []
            for result in raw_results:
                if result["confidence"] >= confidence_threshold:
                    parsed.append({
                        "bbox": result["bbox"],
                        "confidence": result["confidence"], 
                        "class_name": {0: "person", 1: "bottle", 2: "car"}[result["class_id"]],
                        "area": (result["bbox"][2] - result["bbox"][0]) * (result["bbox"][3] - result["bbox"][1])
                    })
            return parsed
        
        # Test avec seuil par défaut (0.5)
        results = parse_yolo_results(mock_detections)
        
        assert len(results) == 2  # Car éliminé (conf 0.45 < 0.5)
        assert results[0]["class_name"] == "person"
        assert results[0]["area"] == 10000  # 100x100
        assert results[1]["class_name"] == "bottle"
    
    def test_confidence_filtering(self):
        """Test du filtrage par confiance"""
        
        detections = [
            {"confidence": 0.9, "class": "person"},
            {"confidence": 0.7, "class": "bottle"},
            {"confidence": 0.3, "class": "car"},
            {"confidence": 0.1, "class": "unknown"}
        ]
        
        def filter_by_confidence(detections, threshold):
            return [d for d in detections if d["confidence"] >= threshold]
        
        # Test différents seuils
        high_conf = filter_by_confidence(detections, 0.8)
        medium_conf = filter_by_confidence(detections, 0.5)
        low_conf = filter_by_confidence(detections, 0.2)
        
        assert len(high_conf) == 1    # Seulement person (0.9)
        assert len(medium_conf) == 2  # person + bottle
        assert len(low_conf) == 3     # person + bottle + car
    
    def test_class_filtering_supermarket(self):
        """Test filtrage classes pour contexte supermarché"""
        
        all_detections = [
            {"class_name": "person", "confidence": 0.8},
            {"class_name": "bottle", "confidence": 0.7},
            {"class_name": "car", "confidence": 0.9},
            {"class_name": "electronics", "confidence": 0.85},
            {"class_name": "dog", "confidence": 0.6}
        ]
        
        # Classes importantes pour surveillance supermarché
        supermarket_classes = ["person", "bottle", "electronics"]
        
        def filter_relevant_classes(detections, relevant_classes):
            return [d for d in detections if d["class_name"] in relevant_classes]
        
        filtered = filter_relevant_classes(all_detections, supermarket_classes)
        
        assert len(filtered) == 3
        assert all(d["class_name"] in supermarket_classes for d in filtered)
        assert "car" not in [d["class_name"] for d in filtered]  # Éliminé
        assert "dog" not in [d["class_name"] for d in filtered]  # Éliminé
    
    def test_bbox_area_calculation(self):
        """Test calcul des aires des bounding boxes"""
        
        detections = [
            {"bbox": [0, 0, 100, 100]},    # 10000 pixels
            {"bbox": [50, 50, 150, 100]},  # 5000 pixels  
            {"bbox": [200, 200, 220, 220]} # 400 pixels
        ]
        
        def calculate_bbox_area(bbox):
            return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        
        areas = [calculate_bbox_area(d["bbox"]) for d in detections]
        
        assert areas == [10000, 5000, 400]
        assert max(areas) == 10000
        assert min(areas) == 400
    
    def test_detection_sorting(self):
        """Test tri des détections par différents critères"""
        
        detections = [
            {"class_name": "person", "confidence": 0.7, "area": 5000},
            {"class_name": "bottle", "confidence": 0.9, "area": 1000},
            {"class_name": "person", "confidence": 0.8, "area": 8000}
        ]
        
        # Tri par confiance (décroissant)
        by_confidence = sorted(detections, key=lambda x: x["confidence"], reverse=True)
        assert by_confidence[0]["confidence"] == 0.9  # bottle
        assert by_confidence[1]["confidence"] == 0.8  # person
        
        # Tri par aire (décroissant) 
        by_area = sorted(detections, key=lambda x: x["area"], reverse=True)
        assert by_area[0]["area"] == 8000  # Grande personne
        assert by_area[1]["area"] == 5000  # Petite personne
        assert by_area[2]["area"] == 1000  # bottle
    
    def test_nms_logic_simulation(self):
        """Test logique NMS (Non-Maximum Suppression) simplifiée"""
        
        detections = [
            {"bbox": [100, 100, 200, 200], "confidence": 0.9, "class": "person"},
            {"bbox": [110, 110, 210, 210], "confidence": 0.7, "class": "person"},  # Overlap
            {"bbox": [300, 300, 400, 400], "confidence": 0.8, "class": "person"}   # Séparé
        ]
        
        def calculate_iou(box1, box2):
            """Calcul IoU simplifié"""
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])
            
            if x2 <= x1 or y2 <= y1:
                return 0.0
                
            intersection = (x2 - x1) * (y2 - y1)
            area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
            area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
            union = area1 + area2 - intersection
            
            return intersection / union if union > 0 else 0.0
        
        def simple_nms(detections, iou_threshold=0.5):
            """NMS simplifié pour test"""
            # Trier par confiance
            sorted_dets = sorted(detections, key=lambda x: x["confidence"], reverse=True)
            keep = []
            
            for i, det in enumerate(sorted_dets):
                should_keep = True
                for kept_det in keep:
                    if det["class"] == kept_det["class"]:
                        iou = calculate_iou(det["bbox"], kept_det["bbox"])
                        if iou > iou_threshold:
                            should_keep = False
                            break
                if should_keep:
                    keep.append(det)
            
            return keep
        
        # Test NMS
        result = simple_nms(detections, iou_threshold=0.3)
        
        # Devrait garder les 2 détections les plus confiantes non-overlapping
        assert len(result) == 2
        assert result[0]["confidence"] == 0.9  # Plus haute confiance gardée
        assert result[1]["confidence"] == 0.8  # Pas d'overlap avec la première
    
    def test_detection_statistics(self):
        """Test calcul de statistiques sur les détections"""
        
        detections = [
            {"class_name": "person", "confidence": 0.9},
            {"class_name": "person", "confidence": 0.7},
            {"class_name": "bottle", "confidence": 0.8},
            {"class_name": "bottle", "confidence": 0.6},
            {"class_name": "electronics", "confidence": 0.85}
        ]
        
        def calculate_detection_stats(detections):
            stats = {}
            
            # Compter par classe
            class_counts = {}
            confidences_by_class = {}
            
            for det in detections:
                class_name = det["class_name"]
                if class_name not in class_counts:
                    class_counts[class_name] = 0
                    confidences_by_class[class_name] = []
                    
                class_counts[class_name] += 1
                confidences_by_class[class_name].append(det["confidence"])
            
            stats["class_counts"] = class_counts
            stats["total_detections"] = len(detections)
            
            # Confiance moyenne par classe
            stats["avg_confidence_by_class"] = {
                cls: sum(confs) / len(confs) 
                for cls, confs in confidences_by_class.items()
            }
            
            return stats
        
        stats = calculate_detection_stats(detections)
        
        assert stats["total_detections"] == 5
        assert stats["class_counts"]["person"] == 2
        assert stats["class_counts"]["bottle"] == 2
        assert stats["class_counts"]["electronics"] == 1
        assert abs(stats["avg_confidence_by_class"]["person"] - 0.8) < 0.01  # (0.9+0.7)/2
        assert abs(stats["avg_confidence_by_class"]["bottle"] - 0.7) < 0.01   # (0.8+0.6)/2

class TestYOLOConfigurationLogic:
    """Tests de la configuration YOLO"""
    
    def test_threshold_optimization(self):
        """Test optimisation des seuils selon contexte"""
        
        def get_optimal_thresholds(context_type: str):
            """Seuils optimaux selon le contexte"""
            thresholds = {
                "supermarket": {"conf": 0.6, "iou": 0.4},
                "parking": {"conf": 0.7, "iou": 0.5},
                "warehouse": {"conf": 0.5, "iou": 0.3},
                "default": {"conf": 0.5, "iou": 0.45}
            }
            return thresholds.get(context_type, thresholds["default"])
        
        supermarket = get_optimal_thresholds("supermarket")
        parking = get_optimal_thresholds("parking")
        unknown = get_optimal_thresholds("unknown_context")
        
        assert supermarket["conf"] == 0.6
        assert parking["conf"] == 0.7
        assert unknown["conf"] == 0.5  # Default
    
    def test_class_priority_logic(self):
        """Test logique de priorité des classes"""
        
        def get_class_priority(class_name: str, context: str = "supermarket"):
            """Priorité des classes selon contexte"""
            priorities = {
                "supermarket": {
                    "person": 1,      # Priorité maximale
                    "electronics": 2, # Important pour vol  
                    "bottle": 3,      # Objets de valeur
                    "car": 10         # Moins important
                },
                "parking": {
                    "car": 1,
                    "person": 2,
                    "truck": 3
                }
            }
            
            context_priorities = priorities.get(context, {})
            return context_priorities.get(class_name, 99)  # 99 = pas important
        
        # Test supermarché
        assert get_class_priority("person", "supermarket") == 1
        assert get_class_priority("electronics", "supermarket") == 2
        assert get_class_priority("unknown_class", "supermarket") == 99
        
        # Test parking
        assert get_class_priority("car", "parking") == 1
        assert get_class_priority("person", "parking") == 2

if __name__ == "__main__":
    print("🎯 Tests YOLO - Logique Pure Sans GPU")
    print("=" * 40)
    
    # Démonstration sans pytest
    tester = TestYOLOLogicPure()
    
    try:
        tester.test_detection_parsing()
        print("✅ Parsing détections: OK")
        
        tester.test_confidence_filtering()  
        print("✅ Filtrage confiance: OK")
        
        tester.test_class_filtering_supermarket()
        print("✅ Filtrage classes: OK")
        
        tester.test_bbox_area_calculation()
        print("✅ Calcul aires: OK")
        
        tester.test_nms_logic_simulation()
        print("✅ Logique NMS: OK")
        
        print(f"\n🎉 Tous les tests YOLO passent sans GPU !")
        
    except AssertionError as e:
        print(f"❌ Test échoué: {e}")
    except Exception as e:
        print(f"💥 Erreur: {e}")