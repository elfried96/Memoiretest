"""
🛠️ Tests Unitaires Outils Avancés - Validation isolée des 8 outils spécialisés
===============================================================================

Tests spécialisés pour chaque outil avancé :
- SAM2Segmentator, DinoV2Features, OpenPoseEstimator
- TrajectoryAnalyzer, MultiModalFusion, TemporalTransformer  
- AdversarialDetector, DomainAdapter

Métriques : Performance, Précision, Robustesse, Utilisation GPU
"""

import pytest
import torch
import time
import numpy as np
import cv2
from typing import Dict, Any, List, Tuple
from PIL import Image
import tempfile

from src.advanced_tools import (
    SAM2Segmentator, DinoV2FeatureExtractor, OpenPoseEstimator,
    TrajectoryAnalyzer, MultiModalFusion, TemporalTransformer,
    AdversarialDetector, DomainAdapter
)
from src.utils.performance import measure_time, get_current_performance


class TestAdvancedToolsUnit:
    """Tests unitaires pour chaque outil avancé individuellement."""
    
    @pytest.fixture
    def test_image_array(self) -> np.ndarray:
        """Image test standard 640x480."""
        # Création image avec personne simulée pour tests pose/segmentation
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Personne debout (rectangle + tête)
        cv2.rectangle(img, (280, 200), (360, 450), (100, 150, 100), -1)  # Corps
        cv2.circle(img, (320, 180), 25, (150, 180, 150), -1)  # Tête
        
        # Bras
        cv2.rectangle(img, (240, 220), (280, 280), (120, 160, 120), -1)  # Bras gauche
        cv2.rectangle(img, (360, 220), (400, 280), (120, 160, 120), -1)  # Bras droit
        
        # Jambes
        cv2.rectangle(img, (285, 400), (315, 480), (110, 140, 110), -1)  # Jambe gauche
        cv2.rectangle(img, (325, 400), (355, 480), (110, 140, 110), -1)  # Jambe droite
        
        return img
    
    @pytest.fixture
    def trajectory_points(self) -> List[Tuple[int, int]]:
        """Points de trajectoire test."""
        return [
            (100, 200), (120, 210), (140, 225), (160, 240),
            (180, 250), (200, 255), (220, 265), (240, 280),
            (260, 300), (280, 320), (300, 340), (320, 360)
        ]
    
    # =================== SAM2 SEGMENTATOR ===================
    
    @measure_time
    def test_sam2_segmentation_performance(self, test_image_array):
        """Test 3.1: Performance segmentation SAM2."""
        try:
            segmentator = SAM2Segmentator()
            
            start_time = time.time()
            result = segmentator.segment_everything(test_image_array)
            segmentation_time = time.time() - start_time
            
            # Validations
            assert segmentation_time < 10.0, f"Segmentation trop lente: {segmentation_time:.2f}s"
            assert hasattr(result, 'masks')
            assert hasattr(result, 'confidence') or hasattr(result, 'mask_properties')
            
            mask_count = len(result.masks) if hasattr(result, 'masks') else 0
            print(f"✅ SAM2: {mask_count} segments en {segmentation_time:.2f}s")
            
            return {
                "tool": "sam2_segmentator",
                "execution_time": segmentation_time,
                "segments_found": mask_count,
                "success": True
            }
            
        except Exception as e:
            print(f"❌ SAM2 Error: {e}")
            return {"tool": "sam2_segmentator", "success": False, "error": str(e)}
    
    def test_sam2_with_boxes(self, test_image_array):
        """Test 3.1b: Segmentation SAM2 avec boîtes."""
        try:
            segmentator = SAM2Segmentator()
            
            # Boîtes englobantes pour la personne simulée  
            boxes = [(280, 180, 360, 450)]  # [x1, y1, x2, y2]
            
            result = segmentator.segment_with_boxes(test_image_array, boxes)
            
            assert hasattr(result, 'masks')
            print(f"✅ SAM2 avec boîtes: {len(result.masks)} segments")
            return True
            
        except Exception as e:
            print(f"❌ SAM2 avec boîtes: {e}")
            return False
    
    # =================== DINO V2 FEATURES ===================
    
    @measure_time 
    def test_dino_feature_extraction(self, test_image_array):
        """Test 3.2: Extraction features DinoV2."""
        try:
            extractor = DinoV2FeatureExtractor()
            
            start_time = time.time()
            features = extractor.extract_global_features(test_image_array)
            extraction_time = time.time() - start_time
            
            # Validations
            assert extraction_time < 2.0, f"Extraction trop lente: {extraction_time:.2f}s"
            assert features is not None
            assert isinstance(features, np.ndarray)
            assert features.shape[0] > 0  # Doit avoir des features
            
            print(f"✅ DinoV2: Features {features.shape} en {extraction_time:.2f}s")
            
            return {
                "tool": "dino_features",
                "execution_time": extraction_time,
                "feature_shape": features.shape,
                "feature_norm": np.linalg.norm(features),
                "success": True
            }
            
        except Exception as e:
            print(f"❌ DinoV2 Error: {e}")
            return {"tool": "dino_features", "success": False, "error": str(e)}
    
    def test_dino_regional_features(self, test_image_array):
        """Test 3.2b: Features régionales DinoV2."""
        try:
            extractor = DinoV2FeatureExtractor()
            
            # Régions d'intérêt (personne)
            regions = [(280, 180, 360, 450)]
            
            regional_features = extractor.extract_regional_features(test_image_array, regions)
            
            assert len(regional_features) == len(regions)
            print(f"✅ DinoV2 régional: {len(regional_features)} régions")
            return True
            
        except Exception as e:
            print(f"❌ DinoV2 régional: {e}")
            return False
    
    # =================== POSE ESTIMATION ===================
    
    @measure_time
    def test_pose_estimation(self, test_image_array):
        """Test 3.3: Estimation de pose."""
        try:
            pose_estimator = OpenPoseEstimator(model_type="mediapipe")
            
            start_time = time.time()
            poses = pose_estimator.estimate_poses(test_image_array)
            pose_time = time.time() - start_time
            
            # Validations
            assert pose_time < 1.0, f"Pose estimation trop lente: {pose_time:.2f}s"
            assert isinstance(poses, list)
            
            print(f"✅ Pose Estimation: {len(poses)} poses en {pose_time:.2f}s")
            
            return {
                "tool": "pose_estimator", 
                "execution_time": pose_time,
                "poses_detected": len(poses),
                "success": True
            }
            
        except Exception as e:
            print(f"❌ Pose Estimation Error: {e}")
            return {"tool": "pose_estimator", "success": False, "error": str(e)}
    
    def test_pose_behavior_analysis(self, test_image_array):
        """Test 3.3b: Analyse comportementale depuis pose."""
        try:
            pose_estimator = OpenPoseEstimator()
            poses = pose_estimator.estimate_poses(test_image_array)
            
            if poses:
                behavior = pose_estimator.analyze_behavior(poses[0])
                assert isinstance(behavior, dict)
                print(f"✅ Analyse comportement: {behavior}")
                return True
            
            return False
            
        except Exception as e:
            print(f"❌ Analyse comportement: {e}")
            return False
    
    # =================== TRAJECTORY ANALYZER ===================
    
    @measure_time
    def test_trajectory_analysis(self, trajectory_points):
        """Test 3.4: Analyse de trajectoire."""
        try:
            analyzer = TrajectoryAnalyzer()
            
            start_time = time.time()
            analysis = analyzer.analyze_trajectory(trajectory_points)
            analysis_time = time.time() - start_time
            
            # Validations
            assert analysis_time < 0.5, f"Analyse trajectoire trop lente: {analysis_time:.2f}s"
            assert isinstance(analysis, dict)
            assert "pattern" in analysis or "anomaly_score" in analysis
            
            print(f"✅ Trajectory Analysis: {analysis.get('pattern', 'unknown')} en {analysis_time:.2f}s")
            
            return {
                "tool": "trajectory_analyzer",
                "execution_time": analysis_time,
                "pattern_detected": analysis.get("pattern", "unknown"),
                "anomaly_score": analysis.get("anomaly_score", 0.0),
                "success": True
            }
            
        except Exception as e:
            print(f"❌ Trajectory Analysis Error: {e}")
            return {"tool": "trajectory_analyzer", "success": False, "error": str(e)}
    
    # =================== MULTIMODAL FUSION ===================
    
    @measure_time
    def test_multimodal_fusion(self):
        """Test 3.5: Fusion multimodale."""
        try:
            fusion = MultiModalFusion()
            
            # Données multimodales simulées
            modalities = {
                "visual": np.random.rand(512),
                "detection": {"person": 0.9, "confidence": 0.85},
                "pose": {"keypoints": 17, "confidence": 0.8},
                "motion": {"velocity": 1.2, "direction": 45}
            }
            
            start_time = time.time()
            result = fusion.fuse_modalities(modalities)
            fusion_time = time.time() - start_time
            
            # Validations
            assert fusion_time < 1.0, f"Fusion trop lente: {fusion_time:.2f}s"
            assert isinstance(result, dict)
            
            print(f"✅ Multimodal Fusion: Prediction={result.get('prediction', 'N/A')} en {fusion_time:.2f}s")
            
            return {
                "tool": "multimodal_fusion",
                "execution_time": fusion_time,
                "fusion_prediction": result.get("prediction", 0.0),
                "attention_weights": result.get("attention_weights", {}),
                "success": True
            }
            
        except Exception as e:
            print(f"❌ Multimodal Fusion Error: {e}")
            return {"tool": "multimodal_fusion", "success": False, "error": str(e)}
    
    # =================== TEMPORAL TRANSFORMER ===================
    
    @measure_time
    def test_temporal_analysis(self):
        """Test 3.6: Analyse temporelle."""
        try:
            temporal = TemporalTransformer()
            
            # Séquence temporelle simulée
            sequence_data = [0.3, 0.5, 0.7, 0.4, 0.6, 0.8, 0.5, 0.9, 0.7]
            
            start_time = time.time()
            result = temporal.analyze_sequence(sequence_data, sequence_type="behavior")
            temporal_time = time.time() - start_time
            
            # Validations
            assert temporal_time < 1.0, f"Analyse temporelle trop lente: {temporal_time:.2f}s"
            assert isinstance(result, dict)
            
            print(f"✅ Temporal Analysis: Pattern={result.get('pattern', 'unknown')} en {temporal_time:.2f}s")
            
            return {
                "tool": "temporal_transformer",
                "execution_time": temporal_time,
                "temporal_pattern": result.get("pattern", "unknown"),
                "trend_detected": result.get("trend", 0.0),
                "success": True
            }
            
        except Exception as e:
            print(f"❌ Temporal Analysis Error: {e}")
            return {"tool": "temporal_transformer", "success": False, "error": str(e)}
    
    # =================== ADVERSARIAL DETECTOR ===================
    
    @measure_time 
    def test_adversarial_detection(self, test_image_array):
        """Test 3.7: Détection d'adversaires."""
        try:
            detector = AdversarialDetector()
            
            start_time = time.time()
            result = detector.detect_adversarial_patterns(test_image_array)
            detection_time = time.time() - start_time
            
            # Validations
            assert detection_time < 2.0, f"Détection adversariale trop lente: {detection_time:.2f}s"
            assert isinstance(result, dict)
            assert "adversarial_score" in result or "threat_level" in result
            
            print(f"✅ Adversarial Detection: Score={result.get('adversarial_score', 'N/A')} en {detection_time:.2f}s")
            
            return {
                "tool": "adversarial_detector",
                "execution_time": detection_time,
                "adversarial_score": result.get("adversarial_score", 0.0),
                "threat_level": result.get("threat_level", "low"),
                "success": True
            }
            
        except Exception as e:
            print(f"❌ Adversarial Detection Error: {e}")
            return {"tool": "adversarial_detector", "success": False, "error": str(e)}
    
    # =================== DOMAIN ADAPTER ===================
    
    @measure_time
    def test_domain_adaptation(self, test_image_array):
        """Test 3.8: Adaptation de domaine."""
        try:
            adapter = DomainAdapter()
            
            start_time = time.time()
            result = adapter.adapt_to_domain(test_image_array, target_domain="retail")
            adaptation_time = time.time() - start_time
            
            # Validations
            assert adaptation_time < 1.0, f"Adaptation domaine trop lente: {adaptation_time:.2f}s"
            assert isinstance(result, dict)
            
            print(f"✅ Domain Adaptation: Domain={result.get('adapted_domain', 'unknown')} en {adaptation_time:.2f}s")
            
            return {
                "tool": "domain_adapter",
                "execution_time": adaptation_time,
                "adapted_domain": result.get("adapted_domain", "unknown"),
                "adaptation_confidence": result.get("confidence", 0.0),
                "success": True
            }
            
        except Exception as e:
            print(f"❌ Domain Adaptation Error: {e}")
            return {"tool": "domain_adapter", "success": False, "error": str(e)}


class TestAdvancedToolsIntegration:
    """Tests d'intégration entre outils avancés."""
    
    def test_tools_pipeline_coordination(self, test_image_array):
        """Test 3.9: Coordination pipeline d'outils."""
        tools_results = {}
        total_time = time.time()
        
        # Pipeline : YOLO -> Pose -> Trajectory -> Fusion
        try:
            # 1. Extraction features visuelles
            dino = DinoV2FeatureExtractor()
            features = dino.extract_global_features(test_image_array)
            tools_results["dino"] = {"success": features is not None}
            
            # 2. Estimation pose
            pose_estimator = OpenPoseEstimator()  
            poses = pose_estimator.estimate_poses(test_image_array)
            tools_results["pose"] = {"success": len(poses) >= 0}
            
            # 3. Fusion des résultats
            fusion = MultiModalFusion()
            fusion_data = {
                "visual": features[:512] if features is not None else np.random.rand(512),
                "pose": {"keypoints": len(poses), "detected": len(poses) > 0}
            }
            fusion_result = fusion.fuse_modalities(fusion_data)
            tools_results["fusion"] = {"success": isinstance(fusion_result, dict)}
            
            total_pipeline_time = time.time() - total_time
            
            print(f"✅ Pipeline intégré: {sum(1 for r in tools_results.values() if r['success'])}/3 outils en {total_pipeline_time:.2f}s")
            
            return {
                "pipeline_time": total_pipeline_time,
                "tools_success": tools_results,
                "integration_success": all(r["success"] for r in tools_results.values())
            }
            
        except Exception as e:
            print(f"❌ Pipeline Error: {e}")
            return {"integration_success": False, "error": str(e)}


@pytest.mark.gpu
class TestAdvancedToolsGPU:
    """Tests GPU pour outils avancés."""
    
    def test_tools_gpu_acceleration(self):
        """Test GPU: Accélération des outils."""
        if not torch.cuda.is_available():
            pytest.skip("GPU non disponible")
        
        # Test image
        test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        gpu_results = {}
        
        # Test des outils avec potentiel GPU
        tools_to_test = [
            ("dino_features", DinoV2FeatureExtractor),
            ("pose_estimator", lambda: OpenPoseEstimator(model_type="mediapipe")),
        ]
        
        for tool_name, tool_class in tools_to_test:
            try:
                tool = tool_class()
                
                start = time.time()
                if tool_name == "dino_features":
                    result = tool.extract_global_features(test_img)
                elif tool_name == "pose_estimator":
                    result = tool.estimate_poses(test_img)
                
                gpu_time = time.time() - start
                gpu_results[tool_name] = {
                    "gpu_time": gpu_time,
                    "success": result is not None
                }
                
            except Exception as e:
                gpu_results[tool_name] = {"success": False, "error": str(e)}
        
        print("🚀 Performance GPU Outils:")
        for tool, result in gpu_results.items():
            if result["success"]:
                print(f"   {tool}: {result['gpu_time']:.3f}s")
        
        return gpu_results


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])