#!/usr/bin/env python3
"""
Script de test complet pour tous les outils du système de surveillance.
Usage: python test_tools.py
"""

import numpy as np
import cv2
import time
import logging
from pathlib import Path

# Import des outils
from src.advanced_tools.sam2_segmentation import SAM2Segmentator
from src.advanced_tools.dino_features import DinoV2FeatureExtractor  
from src.advanced_tools.pose_estimation import OpenPoseEstimator
from src.advanced_tools.trajectory_analyzer import TrajectoryAnalyzer
from src.advanced_tools.multimodal_fusion import MultiModalFusion, FusionInput

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ToolTester:
    """Testeur complet des outils avancés."""
    
    def __init__(self):
        self.test_results = {}
        self.test_image = self.create_test_image()
        
    def create_test_image(self) -> np.ndarray:
        """Crée une image de test simple."""
        # Image 640x480 avec un rectangle (simule une personne)
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        image.fill(128)  # Fond gris
        
        # Rectangle simulant une personne
        cv2.rectangle(image, (200, 150), (400, 450), (255, 0, 0), -1)
        
        # Quelques formes pour tester
        cv2.circle(image, (320, 200), 30, (0, 255, 0), -1)  # Tête
        cv2.rectangle(image, (280, 220), (360, 300), (0, 0, 255), -1)  # Torse
        
        return image
    
    def test_sam2_segmentation(self) -> bool:
        """Test SAM2 Segmentator."""
        print("\n🔍 Test SAM2 Segmentator...")
        
        try:
            # Test avec lazy loading
            sam2 = SAM2Segmentator(lazy_loading=True)
            
            # Info du modèle
            info = sam2.get_model_info()
            print(f"   Model: {info['model_name']}")
            print(f"   Loaded: {info['model_loaded']}")
            print(f"   VRAM: ~1GB si GPU")
            
            # Test avec bounding boxes
            bboxes = [[200, 150, 400, 450]]  # [x1, y1, x2, y2]
            
            start_time = time.time()
            result = sam2.segment_objects(self.test_image, bboxes)
            processing_time = time.time() - start_time
            
            print(f"   ✅ Segmentation réussie")
            print(f"   📊 Masks: {result.masks.shape}")
            print(f"   📊 Scores: {result.scores}")
            print(f"   ⏱️  Temps: {processing_time:.2f}s")
            
            self.test_results['sam2'] = {
                'success': True,
                'processing_time': processing_time,
                'masks_count': len(result.masks)
            }
            
            return True
            
        except Exception as e:
            print(f"   ❌ Erreur: {e}")
            self.test_results['sam2'] = {'success': False, 'error': str(e)}
            return False
    
    def test_dino_features(self) -> bool:
        """Test DINO v2 Feature Extractor."""
        print("\n🧠 Test DINO v2 Features...")
        
        try:
            # Test avec lazy loading
            dino = DinoV2FeatureExtractor(lazy_loading=True)
            
            # Info du modèle
            info = dino.get_model_info()
            print(f"   Model: {info['model_name']}")
            print(f"   Dimensions: {info['feature_dimension']}")
            print(f"   Paramètres: {info['parameters']}")
            
            start_time = time.time()
            result = dino.extract_features(self.test_image)
            processing_time = time.time() - start_time
            
            print(f"   ✅ Extraction réussie")
            print(f"   📊 Features shape: {result.features.shape}")
            print(f"   📊 Feature sample: {result.features[:5]}...")
            print(f"   ⏱️  Temps: {processing_time:.2f}s")
            
            self.test_results['dino'] = {
                'success': True,
                'processing_time': processing_time,
                'feature_dim': len(result.features)
            }
            
            return True
            
        except Exception as e:
            print(f"   ❌ Erreur: {e}")
            self.test_results['dino'] = {'success': False, 'error': str(e)}
            return False
    
    def test_pose_estimation(self) -> bool:
        """Test Pose Estimation."""
        print("\n🤸 Test Pose Estimation...")
        
        try:
            # Test MediaPipe
            pose_estimator = OpenPoseEstimator(model_type="mediapipe", lazy_loading=True)
            
            # Info du modèle
            info = pose_estimator.get_model_info()
            print(f"   Model: {info['model_type']}")
            print(f"   Keypoints: {info['keypoints_count']}")
            
            start_time = time.time()
            result = pose_estimator.estimate_poses(self.test_image)
            processing_time = time.time() - start_time
            
            print(f"   ✅ Estimation réussie")
            print(f"   📊 Keypoints shape: {result.keypoints.shape}")
            print(f"   📊 Personnes détectées: {len(result.person_boxes)}")
            print(f"   ⏱️  Temps: {processing_time:.2f}s")
            
            self.test_results['pose'] = {
                'success': True,
                'processing_time': processing_time,
                'people_detected': len(result.person_boxes)
            }
            
            return True
            
        except Exception as e:
            print(f"   ❌ Erreur: {e}")
            self.test_results['pose'] = {'success': False, 'error': str(e)}
            return False
    
    def test_trajectory_analyzer(self) -> bool:
        """Test Trajectory Analyzer."""
        print("\n🚶 Test Trajectory Analyzer...")
        
        try:
            analyzer = TrajectoryAnalyzer()
            
            # Simuler une trajectoire
            timestamps = [time.time() + i for i in range(10)]
            positions = [(100 + i*10, 200 + i*5) for i in range(10)]
            
            # Ajouter les points
            for i, (timestamp, (x, y)) in enumerate(zip(timestamps, positions)):
                analyzer.update_trajectory("person_1", x, y, timestamp)
            
            start_time = time.time()
            result = analyzer.analyze_trajectory("person_1", time_window=60)
            processing_time = time.time() - start_time
            
            if result:
                print(f"   ✅ Analyse réussie")
                print(f"   📊 Distance totale: {result.total_distance:.1f}")
                print(f"   📊 Vitesse moyenne: {result.average_velocity:.2f}")
                print(f"   📊 Pattern: {result.pattern_classification}")
                print(f"   📊 Score anomalie: {result.anomaly_score:.2f}")
                print(f"   ⏱️  Temps: {processing_time:.4f}s")
                
                self.test_results['trajectory'] = {
                    'success': True,
                    'processing_time': processing_time,
                    'pattern': result.pattern_classification
                }
                
                return True
            else:
                print("   ❌ Pas assez de données pour l'analyse")
                return False
            
        except Exception as e:
            print(f"   ❌ Erreur: {e}")
            self.test_results['trajectory'] = {'success': False, 'error': str(e)}
            return False
    
    def test_multimodal_fusion(self) -> bool:
        """Test Multimodal Fusion."""
        print("\n🔗 Test Multimodal Fusion...")
        
        try:
            fusion = MultiModalFusion()
            
            # Créer des features fictives
            fusion_input = FusionInput(
                visual_features=np.random.rand(768),      # DINO features
                detection_features=np.random.rand(256),   # YOLO features
                pose_features=np.random.rand(132),        # Pose features  
                motion_features=np.random.rand(64),       # Motion features
                temporal_features=np.random.rand(128)     # Temporal features
            )
            
            start_time = time.time()
            result = fusion.fuse_features(fusion_input)
            processing_time = time.time() - start_time
            
            print(f"   ✅ Fusion réussie")
            print(f"   📊 Prediction: {result.final_prediction:.3f}")
            print(f"   📊 Attention weights: {result.attention_weights}")
            print(f"   📊 Fused features shape: {result.fused_features.shape}")
            print(f"   ⏱️  Temps: {processing_time:.4f}s")
            
            self.test_results['fusion'] = {
                'success': True,
                'processing_time': processing_time,
                'prediction': float(result.final_prediction)
            }
            
            return True
            
        except Exception as e:
            print(f"   ❌ Erreur: {e}")
            self.test_results['fusion'] = {'success': False, 'error': str(e)}
            return False
    
    def test_integration(self) -> bool:
        """Test d'intégration complète."""
        print("\n🔄 Test d'Intégration Complète...")
        
        try:
            # Charger tous les outils
            sam2 = SAM2Segmentator(lazy_loading=False)
            dino = DinoV2FeatureExtractor(lazy_loading=False) 
            pose_estimator = OpenPoseEstimator(lazy_loading=False)
            trajectory_analyzer = TrajectoryAnalyzer()
            fusion = MultiModalFusion()
            
            start_time = time.time()
            
            # Pipeline complet
            # 1. Extraction DINO features
            dino_result = dino.extract_features(self.test_image)
            
            # 2. Estimation pose
            pose_result = pose_estimator.estimate_poses(self.test_image)
            
            # 3. Trajectory (simulée)
            trajectory_analyzer.update_trajectory("test_person", 320, 240, time.time())
            trajectory_analyzer.update_trajectory("test_person", 330, 250, time.time() + 1)
            
            # 4. Fusion
            detections = [{"confidence": 0.9, "bbox": [200, 150, 400, 450], "class": "person"}]
            
            fusion_input = FusionInput(
                visual_features=dino_result.features,
                detection_features=fusion.extract_detection_features(detections),
                pose_features=fusion.extract_pose_features({"keypoints": pose_result.keypoints}),
                motion_features=np.random.rand(64),  # Simplifié
                temporal_features=fusion.extract_temporal_features({"consistency_score": 0.8})
            )
            
            fusion_result = fusion.fuse_features(fusion_input)
            
            total_time = time.time() - start_time
            
            print(f"   ✅ Pipeline complet réussi")
            print(f"   📊 Score final: {fusion_result.final_prediction:.3f}")
            print(f"   ⏱️  Temps total: {total_time:.2f}s")
            
            self.test_results['integration'] = {
                'success': True,
                'total_time': total_time,
                'final_score': float(fusion_result.final_prediction)
            }
            
            return True
            
        except Exception as e:
            print(f"   ❌ Erreur d'intégration: {e}")
            self.test_results['integration'] = {'success': False, 'error': str(e)}
            return False
    
    def run_all_tests(self):
        """Exécute tous les tests."""
        print("🚀 Démarrage des tests du système de surveillance")
        print("=" * 60)
        
        tests = [
            ("SAM2 Segmentation", self.test_sam2_segmentation),
            ("DINO v2 Features", self.test_dino_features), 
            ("Pose Estimation", self.test_pose_estimation),
            ("Trajectory Analysis", self.test_trajectory_analyzer),
            ("Multimodal Fusion", self.test_multimodal_fusion),
            ("Integration Test", self.test_integration)
        ]
        
        total_start = time.time()
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            success = test_func()
            if success:
                passed += 1
        
        total_time = time.time() - total_start
        
        print("\n" + "=" * 60)
        print("📊 RÉSULTATS FINAUX")
        print("=" * 60)
        print(f"✅ Tests réussis: {passed}/{total}")
        print(f"⏱️  Temps total: {total_time:.2f}s")
        print(f"📈 Taux de réussite: {passed/total*100:.1f}%")
        
        # Détail des résultats
        for tool, result in self.test_results.items():
            status = "✅ OK" if result['success'] else "❌ FAIL"
            time_info = f" ({result.get('processing_time', 0):.2f}s)" if result['success'] else ""
            error_info = f" - {result.get('error', '')}" if not result['success'] else ""
            print(f"   {tool}: {status}{time_info}{error_info}")
        
        if passed == total:
            print("\n🎉 TOUS LES TESTS SONT PASSÉS ! Le système fonctionne.")
        else:
            print(f"\n⚠️  {total-passed} tests ont échoué. Vérifiez les erreurs ci-dessus.")
        
        return passed == total

if __name__ == "__main__":
    tester = ToolTester()
    success = tester.run_all_tests()
    exit(0 if success else 1)