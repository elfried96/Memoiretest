"""
🔄 Tests d'Intégration Progressive - Validation Pipeline Complet
================================================================

Tests d'intégration en plusieurs étapes :
1. Intégration VLM + YOLO (détection + analyse)
2. Intégration Outils Avancés + Orchestrateur 
3. Pipeline complet avec orchestration adaptative
4. Validation modes opérationnels (FAST, BALANCED, THOROUGH)
5. Tests de charge et performance sous stress

Objectif : Valider que tous les composants fonctionnent ensemble
"""

import pytest
import asyncio
import time
import numpy as np
import cv2
import base64
import io
from typing import Dict, List, Any, Tuple
from PIL import Image
import json
from pathlib import Path

from src.core.vlm.dynamic_model import DynamicVisionLanguageModel
from src.detection.yolo_detector import YOLODetector  
from src.core.orchestrator.adaptive_orchestrator import AdaptiveVLMOrchestrator
from src.core.orchestrator.vlm_orchestrator import ModernVLMOrchestrator, OrchestrationConfig, OrchestrationMode
from src.core.types import AnalysisRequest, AnalysisResponse, SuspicionLevel, ActionType, DetectedObject
from src.utils.performance import get_current_performance


class TestIntegrationProgressive:
    """Tests d'intégration progressive du système complet."""
    
    @pytest.fixture
    def test_images_b64(self) -> Dict[str, str]:
        """Images test en base64 pour différents scénarios."""
        images = {}
        
        # Scénario 1: Personne normale
        img_normal = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(img_normal, (300, 150), (380, 400), (100, 100, 100), -1)  # Personne
        cv2.circle(img_normal, (340, 130), 25, (150, 150, 150), -1)  # Tête
        images["normal_person"] = self._numpy_to_b64(img_normal)
        
        # Scénario 2: Comportement suspect (personne près d'un objet)
        img_suspect = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(img_suspect, (200, 200), (250, 350), (80, 80, 80), -1)  # Personne
        cv2.rectangle(img_suspect, (260, 300), (320, 360), (255, 0, 0), -1)  # Objet rouge (suspect)
        images["suspicious_behavior"] = self._numpy_to_b64(img_suspect)
        
        # Scénario 3: Foule (plusieurs personnes)
        img_crowd = np.zeros((480, 640, 3), dtype=np.uint8)
        positions = [(150, 200), (300, 180), (450, 220), (250, 350), (400, 320)]
        for i, (x, y) in enumerate(positions):
            cv2.rectangle(img_crowd, (x-20, y), (x+20, y+100), (100+i*20, 100, 100), -1)
            cv2.circle(img_crowd, (x, y-20), 15, (150, 150, 150), -1)
        images["crowd_scene"] = self._numpy_to_b64(img_crowd)
        
        # Scénario 4: Zone vide
        img_empty = np.ones((480, 640, 3), dtype=np.uint8) * 128  # Gris uniforme
        images["empty_area"] = self._numpy_to_b64(img_empty)
        
        return images
    
    def _numpy_to_b64(self, img: np.ndarray) -> str:
        """Conversion numpy array vers base64."""
        _, buffer = cv2.imencode('.jpg', img)
        img_b64 = base64.b64encode(buffer).decode('utf-8')
        return img_b64
    
    # =================== NIVEAU 1: VLM + YOLO ===================
    
    @pytest.mark.asyncio
    async def test_vlm_yolo_integration(self, test_images_b64):
        """Test 4.1: Intégration VLM + YOLO."""
        print("\\n🔗 Test Intégration VLM + YOLO")
        
        # Initialisation composants
        vlm = DynamicVisionLanguageModel(default_model="kimi-vl-a3b-thinking")
        yolo = YOLODetector()
        
        results = {}
        
        for scenario, image_b64 in test_images_b64.items():
            start_time = time.time()
            
            # 1. Détection YOLO
            img_array = self._b64_to_numpy(image_b64)
            detections = yolo.detect(img_array)
            
            # 2. Analyse VLM avec contexte YOLO
            request = AnalysisRequest(
                frame_data=image_b64,
                context={
                    "detections": len(detections),
                    "scenario": scenario,
                    "detected_objects": [d.class_name for d in detections]
                },
                tools_available=[]  # Test sans outils avancés
            )
            
            vlm_result = await vlm.analyze_with_tools(request, use_advanced_tools=False)
            
            integration_time = time.time() - start_time
            
            # Validations
            assert vlm_result is not None
            assert isinstance(vlm_result, AnalysisResponse)
            assert integration_time < 5.0, f"Intégration trop lente: {integration_time:.2f}s"
            
            results[scenario] = {
                "yolo_detections": len(detections),
                "vlm_confidence": vlm_result.confidence,
                "suspicion_level": vlm_result.suspicion_level.value,
                "integration_time": integration_time,
                "success": True
            }
            
            print(f"   {scenario}: {len(detections)} objets, confiance={vlm_result.confidence:.2f}, {integration_time:.2f}s")
        
        # Validation globale
        assert all(r["success"] for r in results.values())
        avg_time = np.mean([r["integration_time"] for r in results.values()])
        print(f"✅ Intégration VLM+YOLO: {len(results)} scénarios en {avg_time:.2f}s moyenne")
        
        return results
    
    def _b64_to_numpy(self, img_b64: str) -> np.ndarray:
        """Conversion base64 vers numpy array."""
        img_bytes = base64.b64decode(img_b64)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img
    
    # =================== NIVEAU 2: ORCHESTRATEUR MODERNE ===================
    
    @pytest.mark.asyncio
    async def test_modern_orchestrator_integration(self, test_images_b64):
        """Test 4.2: Intégration orchestrateur moderne."""
        print("\\n🎭 Test Orchestrateur Moderne")
        
        # Test des 3 modes d'orchestration
        modes = [
            OrchestrationMode.FAST,
            OrchestrationMode.BALANCED, 
            OrchestrationMode.THOROUGH
        ]
        
        results = {}
        
        for mode in modes:
            config = OrchestrationConfig(mode=mode, max_concurrent_tools=6)
            orchestrator = ModernVLMOrchestrator(
                vlm_model_name="kimi-vl-a3b-thinking",
                config=config
            )
            
            mode_results = {}
            
            for scenario, image_b64 in test_images_b64.items():
                start_time = time.time()
                
                # Analyse avec orchestration
                result = await orchestrator.analyze_surveillance_frame(
                    frame_data=image_b64,
                    detections=[],
                    context={"scenario": scenario, "test_mode": mode.value}
                )
                
                orchestration_time = time.time() - start_time
                
                # Validations
                assert result is not None
                assert isinstance(result, AnalysisResponse)
                
                mode_results[scenario] = {
                    "orchestration_time": orchestration_time,
                    "tools_used": len(result.tools_used),
                    "confidence": result.confidence,
                    "suspicion": result.suspicion_level.value
                }
            
            # Statistiques par mode
            avg_time = np.mean([r["orchestration_time"] for r in mode_results.values()])
            avg_tools = np.mean([r["tools_used"] for r in mode_results.values()])
            
            results[mode.value] = {
                "scenarios": mode_results,
                "avg_time": avg_time,
                "avg_tools_used": avg_tools,
                "success": True
            }
            
            print(f"   Mode {mode.value}: {avg_time:.2f}s, {avg_tools:.1f} outils moyens")
        
        # Validation cohérence des modes
        fast_time = results["fast"]["avg_time"]
        thorough_time = results["thorough"]["avg_time"] 
        
        assert fast_time < thorough_time, "Mode FAST devrait être plus rapide que THOROUGH"
        
        print(f"✅ Orchestrateur: FAST={fast_time:.2f}s < THOROUGH={thorough_time:.2f}s")
        return results
    
    # =================== NIVEAU 3: ORCHESTRATEUR ADAPTATIF ===================
    
    @pytest.mark.asyncio
    async def test_adaptive_orchestrator_integration(self, test_images_b64):
        """Test 4.3: Intégration orchestrateur adaptatif."""
        print("\\n🧠 Test Orchestrateur Adaptatif")
        
        config = OrchestrationConfig(
            mode=OrchestrationMode.BALANCED,
            max_concurrent_tools=4,
            enable_advanced_tools=True
        )
        
        adaptive_orchestrator = AdaptiveVLMOrchestrator(
            vlm_model_name="kimi-vl-a3b-thinking",
            config=config,
            enable_adaptive_learning=True
        )
        
        results = {}
        learning_progression = []
        
        # Test apprentissage progressif - 2 passes
        for pass_num in range(2):
            pass_results = {}
            
            for scenario, image_b64 in test_images_b64.items():
                start_time = time.time()
                
                # Analyse adaptative
                result = await adaptive_orchestrator.analyze_surveillance_frame(
                    frame_data=image_b64,
                    detections=[],
                    context={
                        "scenario": scenario, 
                        "pass": pass_num,
                        "location": "test_lab"
                    }
                )
                
                adaptive_time = time.time() - start_time
                
                pass_results[scenario] = {
                    "time": adaptive_time,
                    "tools_used": len(result.tools_used),
                    "confidence": result.confidence,
                    "selected_tools": result.tools_used
                }
            
            avg_time = np.mean([r["time"] for r in pass_results.values()])
            learning_progression.append(avg_time)
            
            results[f"pass_{pass_num + 1}"] = {
                "scenarios": pass_results,
                "avg_time": avg_time
            }
            
            print(f"   Pass {pass_num + 1}: {avg_time:.2f}s moyenne")
        
        # Validation apprentissage
        if len(learning_progression) >= 2:
            improvement = (learning_progression[0] - learning_progression[1]) / learning_progression[0]
            print(f"   Amélioration apprentissage: {improvement*100:.1f}%")
        
        # Test statut adaptatif
        adaptive_status = adaptive_orchestrator.get_adaptive_status()
        assert "adaptive_learning_enabled" in adaptive_status
        assert "current_optimal_tools" in adaptive_status
        
        print(f"✅ Orchestrateur Adaptatif: {len(adaptive_status['current_optimal_tools'])} outils optimaux")
        return results
    
    # =================== NIVEAU 4: PIPELINE COMPLET ===================
    
    @pytest.mark.asyncio
    async def test_complete_pipeline_integration(self, test_images_b64):
        """Test 4.4: Pipeline complet end-to-end."""
        print("\\n🚀 Test Pipeline Complet")
        
        # Configuration pipeline complet
        config = OrchestrationConfig(
            mode=OrchestrationMode.BALANCED,
            max_concurrent_tools=5,
            confidence_threshold=0.7,
            enable_advanced_tools=True
        )
        
        orchestrator = AdaptiveVLMOrchestrator(
            vlm_model_name="kimi-vl-a3b-thinking",
            config=config
        )
        
        pipeline_results = {}
        total_start = time.time()
        
        for scenario, image_b64 in test_images_b64.items():
            scenario_start = time.time()
            
            # Pipeline complet : Détection + VLM + Outils + Orchestration + Adaptation
            result = await orchestrator.analyze_surveillance_frame(
                frame_data=image_b64,
                detections=[],  # L'orchestrateur gère la détection
                context={
                    "scenario": scenario,
                    "pipeline_test": True,
                    "location": "surveillance_zone",
                    "timestamp": time.time()
                }
            )
            
            scenario_time = time.time() - scenario_start
            
            # Validations complètes
            assert result is not None
            assert isinstance(result, AnalysisResponse)
            assert result.suspicion_level in SuspicionLevel
            assert result.action_type in ActionType
            assert 0.0 <= result.confidence <= 1.0
            assert scenario_time < 10.0, f"Pipeline trop lent: {scenario_time:.2f}s"
            
            pipeline_results[scenario] = {
                "pipeline_time": scenario_time,
                "tools_used_count": len(result.tools_used),
                "tools_used": result.tools_used,
                "confidence": result.confidence,
                "suspicion_level": result.suspicion_level.value,
                "action_type": result.action_type.value,
                "recommendations": result.recommendations,
                "success": True
            }
            
            print(f"   {scenario}: {len(result.tools_used)} outils, confiance={result.confidence:.2f}, {scenario_time:.2f}s")
        
        total_time = time.time() - total_start
        
        # Statistiques globales
        successful_scenarios = sum(1 for r in pipeline_results.values() if r["success"])
        avg_pipeline_time = np.mean([r["pipeline_time"] for r in pipeline_results.values()])
        avg_tools_used = np.mean([r["tools_used_count"] for r in pipeline_results.values()])
        avg_confidence = np.mean([r["confidence"] for r in pipeline_results.values()])
        
        # Validations finales
        assert successful_scenarios == len(test_images_b64), "Tous les scénarios doivent réussir"
        assert avg_pipeline_time < 5.0, f"Pipeline moyen trop lent: {avg_pipeline_time:.2f}s"
        
        print(f"✅ Pipeline Complet: {successful_scenarios}/{len(test_images_b64)} scénarios")
        print(f"   Performance: {avg_pipeline_time:.2f}s, {avg_tools_used:.1f} outils, confiance={avg_confidence:.2f}")
        
        return {
            "pipeline_results": pipeline_results,
            "total_time": total_time,
            "avg_pipeline_time": avg_pipeline_time,
            "success_rate": successful_scenarios / len(test_images_b64),
            "avg_tools_used": avg_tools_used,
            "avg_confidence": avg_confidence
        }
    
    # =================== NIVEAU 5: TESTS DE CHARGE ===================
    
    @pytest.mark.asyncio
    async def test_load_testing_integration(self, test_images_b64):
        """Test 4.5: Tests de charge système."""
        print("\\n⚡ Test de Charge Système")
        
        orchestrator = AdaptiveVLMOrchestrator(
            vlm_model_name="kimi-vl-a3b-thinking",
            config=OrchestrationConfig(mode=OrchestrationMode.FAST)  # Mode rapide pour charge
        )
        
        # Test concurrent - simulation de plusieurs caméras
        concurrent_scenarios = list(test_images_b64.items()) * 3  # 3x chaque scénario
        
        async def process_scenario(scenario_name, image_b64, scenario_id):
            """Traitement d'un scénario individuel."""
            start = time.time()
            try:
                result = await orchestrator.analyze_surveillance_frame(
                    frame_data=image_b64,
                    context={"scenario": scenario_name, "concurrent_id": scenario_id}
                )
                processing_time = time.time() - start
                return {
                    "scenario": scenario_name,
                    "id": scenario_id,
                    "time": processing_time,
                    "success": True,
                    "confidence": result.confidence
                }
            except Exception as e:
                return {
                    "scenario": scenario_name,
                    "id": scenario_id,
                    "success": False,
                    "error": str(e)
                }
        
        # Exécution concurrente
        start_load = time.time()
        
        tasks = []
        for i, (scenario_name, image_b64) in enumerate(concurrent_scenarios):
            task = process_scenario(scenario_name, image_b64, i)
            tasks.append(task)
        
        # Limitation concurrence pour éviter surcharge
        batch_size = 5
        load_results = []
        
        for i in range(0, len(tasks), batch_size):
            batch_tasks = tasks[i:i+batch_size]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            load_results.extend([r for r in batch_results if not isinstance(r, Exception)])
        
        total_load_time = time.time() - start_load
        
        # Analyse des résultats
        successful_loads = [r for r in load_results if r.get("success", False)]
        failed_loads = [r for r in load_results if not r.get("success", False)]
        
        if successful_loads:
            avg_concurrent_time = np.mean([r["time"] for r in successful_loads])
            max_concurrent_time = np.max([r["time"] for r in successful_loads])
            
            print(f"   Concurrent: {len(successful_loads)}/{len(load_results)} réussies")
            print(f"   Performance: {avg_concurrent_time:.2f}s moyenne, {max_concurrent_time:.2f}s max")
            print(f"   Échecs: {len(failed_loads)} ({len(failed_loads)/len(load_results)*100:.1f}%)")
        
        # Validations charge
        success_rate = len(successful_loads) / len(load_results) if load_results else 0
        assert success_rate > 0.8, f"Taux de succès trop faible: {success_rate:.2f}"
        
        print(f"✅ Test de Charge: {success_rate:.2%} succès, {total_load_time:.2f}s total")
        
        return {
            "total_scenarios": len(load_results),
            "successful": len(successful_loads),
            "failed": len(failed_loads),
            "success_rate": success_rate,
            "total_load_time": total_load_time,
            "avg_concurrent_time": avg_concurrent_time if successful_loads else 0
        }


@pytest.mark.gpu 
class TestIntegrationGPU:
    """Tests d'intégration optimisés GPU."""
    
    @pytest.mark.asyncio
    async def test_gpu_pipeline_performance(self):
        """Test GPU: Performance pipeline complet."""
        if not torch.cuda.is_available():
            pytest.skip("GPU non disponible")
        
        print("\\n🚀 Test Pipeline GPU")
        
        # Configuration GPU optimisée
        config = OrchestrationConfig(
            mode=OrchestrationMode.BALANCED,
            max_concurrent_tools=6  # Plus d'outils avec GPU
        )
        
        orchestrator = AdaptiveVLMOrchestrator(
            vlm_model_name="kimi-vl-a3b-thinking",
            config=config
        )
        
        # Image test haute résolution pour GPU
        hd_img = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        hd_b64 = self._numpy_to_b64(hd_img)
        
        # Mesure performance GPU
        gpu_start = time.time()
        
        result = await orchestrator.analyze_surveillance_frame(
            frame_data=hd_b64,
            context={"resolution": "1080p", "gpu_test": True}
        )
        
        gpu_time = time.time() - gpu_start
        
        # Validation
        assert result is not None
        assert gpu_time < 8.0, f"Pipeline GPU trop lent: {gpu_time:.2f}s"
        
        print(f"✅ Pipeline GPU HD: {gpu_time:.2f}s, {len(result.tools_used)} outils")
        
        return {
            "gpu_pipeline_time": gpu_time,
            "resolution": "1080p", 
            "tools_used": len(result.tools_used),
            "gpu_memory_used": torch.cuda.memory_allocated() / (1024**2)  # MB
        }
    
    def _numpy_to_b64(self, img: np.ndarray) -> str:
        """Conversion numpy vers base64."""
        _, buffer = cv2.imencode('.jpg', img)
        return base64.b64encode(buffer).decode('utf-8')


if __name__ == "__main__":
    # Exécution avec rapports détaillés
    pytest.main([__file__, "-v", "--tb=short", "--capture=no"])