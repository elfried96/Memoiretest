"""
🧠 Tests Unitaires VLM - Évaluation isolée des modèles de langage visuel
=======================================================================

Tests spécialisés pour valider individuellement :
- Chargement et switch des modèles (Kimi-VL, Qwen2-VL)
- Performance d'inférence avec métriques précises
- Robustesse et fallback automatique
- Utilisation GPU optimisée
"""

import pytest
import torch
import time
import numpy as np
from typing import Dict, Any, List
import asyncio
from pathlib import Path
import base64
from PIL import Image
import io

from src.core.vlm.dynamic_model import DynamicVisionLanguageModel
from src.core.vlm.model_registry import VLMModelRegistry, VLMModelType
from src.core.types import AnalysisRequest, SuspicionLevel, ActionType
from src.utils.performance import measure_time, get_current_performance


class TestVLMUnitIsolated:
    """Tests unitaires isolés pour les modèles VLM."""
    
    @pytest.fixture
    def sample_image_b64(self) -> str:
        """Image d'exemple encodée en base64."""
        # Création d'une image test 224x224
        img = Image.new('RGB', (224, 224), color='red')
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG')
        img_bytes = buffer.getvalue()
        return base64.b64encode(img_bytes).decode('utf-8')
    
    @pytest.fixture
    def vlm_model(self) -> DynamicVisionLanguageModel:
        """Instance VLM pour tests."""
        return DynamicVisionLanguageModel(
            default_model="kimi-vl-a3b-thinking",
            device="auto",
            enable_fallback=True
        )
    
    @pytest.mark.asyncio
    async def test_model_loading_performance(self, vlm_model):
        """Test 1.1: Performance de chargement des modèles."""
        start_time = time.time()
        
        # Test chargement Kimi-VL
        success_kimi = await vlm_model.load_model("kimi-vl-a3b-thinking")
        kimi_load_time = time.time() - start_time
        
        assert success_kimi, "Échec chargement Kimi-VL-A3B-Thinking"
        assert kimi_load_time < 30.0, f"Chargement Kimi trop lent: {kimi_load_time:.2f}s"
        
        # Vérification état du modèle
        assert vlm_model.is_loaded
        assert vlm_model.current_model_id == "kimi-vl-a3b-thinking"
        assert vlm_model.model is not None
        assert vlm_model.processor is not None
        
        print(f"✅ Kimi-VL chargé en {kimi_load_time:.2f}s")
    
    @pytest.mark.asyncio
    async def test_model_switching_performance(self, vlm_model):
        """Test 1.2: Performance de switch entre modèles."""
        # Chargement initial
        await vlm_model.load_model("kimi-vl-a3b-thinking")
        
        # Switch vers Qwen2-VL
        start_time = time.time()
        success_switch = await vlm_model.switch_model("qwen2-vl-7b-instruct")
        switch_time = time.time() - start_time
        
        if success_switch:
            assert vlm_model.current_model_id == "qwen2-vl-7b-instruct"
            assert switch_time < 45.0, f"Switch trop lent: {switch_time:.2f}s"
            print(f"✅ Switch Kimi→Qwen réussi en {switch_time:.2f}s")
        else:
            print("⚠️ Switch échoué - Fallback testé")
    
    @pytest.mark.asyncio
    async def test_inference_performance_isolated(self, vlm_model, sample_image_b64):
        """Test 1.3: Performance d'inférence isolée."""
        await vlm_model.load_model("kimi-vl-a3b-thinking")
        
        request = AnalysisRequest(
            frame_data=sample_image_b64,
            context={"location": "test_lab", "mode": "unit_test"},
            tools_available=[]  # Pas d'outils = test VLM pur
        )
        
        # Test inférence multiple pour moyenner
        inference_times = []
        for i in range(5):
            start_time = time.time()
            result = await vlm_model.analyze_with_tools(request, use_advanced_tools=False)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            # Validations
            assert result is not None
            assert result.confidence >= 0.0
            assert result.suspicion_level in SuspicionLevel
            assert result.action_type in ActionType
        
        avg_inference_time = np.mean(inference_times)
        std_inference_time = np.std(inference_times)
        
        assert avg_inference_time < 3.0, f"Inférence trop lente: {avg_inference_time:.2f}s"
        
        print(f"✅ Inférence VLM: {avg_inference_time:.2f}±{std_inference_time:.2f}s")
        
        return {
            "avg_inference_time": avg_inference_time,
            "std_inference_time": std_inference_time,
            "model_used": vlm_model.current_model_id,
            "gpu_used": torch.cuda.is_available()
        }
    
    @pytest.mark.asyncio
    async def test_fallback_mechanism(self, vlm_model):
        """Test 1.4: Mécanisme de fallback."""
        # Test avec modèle inexistant pour déclencher fallback
        success = await vlm_model.load_model("modele-inexistant-test")
        
        if vlm_model.enable_fallback:
            assert vlm_model.is_loaded, "Fallback devrait charger un modèle"
            assert vlm_model.current_model_id in ["qwen2-vl-7b-instruct"], "Mauvais modèle fallback"
            print(f"✅ Fallback activé: {vlm_model.current_model_id}")
        else:
            assert not success, "Sans fallback, le chargement devrait échouer"
            print("✅ Fallback désactivé fonctionne correctement")
    
    @pytest.mark.asyncio
    async def test_memory_management(self, vlm_model):
        """Test 1.5: Gestion mémoire GPU/CPU."""
        initial_memory = get_current_performance()
        
        # Chargement modèle
        await vlm_model.load_model("kimi-vl-a3b-thinking")
        post_load_memory = get_current_performance()
        
        # Multiple inférences
        request = AnalysisRequest(
            frame_data="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",  # pixel rouge 1x1
            context={},
            tools_available=[]
        )
        
        for _ in range(10):
            await vlm_model.analyze_with_tools(request, use_advanced_tools=False)
        
        # Déchargement
        vlm_model._unload_current_model()
        final_memory = get_current_performance()
        
        # Vérifications mémoire
        memory_growth = post_load_memory.memory_used_mb - initial_memory.memory_used_mb
        memory_cleaned = final_memory.memory_used_mb < post_load_memory.memory_used_mb
        
        print(f"📊 Mémoire: Initial={initial_memory.memory_used_mb:.1f}MB, "
              f"Après chargement={post_load_memory.memory_used_mb:.1f}MB, "
              f"Final={final_memory.memory_used_mb:.1f}MB")
        
        assert memory_growth < 8000, f"Croissance mémoire excessive: {memory_growth:.1f}MB"
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    @pytest.mark.asyncio
    async def test_model_registry_validation(self):
        """Test 1.6: Validation du registre de modèles."""
        registry = VLMModelRegistry()
        
        # Test modèles disponibles
        available_models = registry.list_available_models()
        assert len(available_models) >= 2, "Registry doit contenir au moins 2 modèles"
        
        # Test validation Kimi-VL
        is_available, message = registry.validate_model_availability("kimi-vl-a3b-thinking")
        print(f"Kimi-VL disponible: {is_available} - {message}")
        
        # Test validation Qwen2-VL
        is_available_qwen, message_qwen = registry.validate_model_availability("qwen2-vl-7b-instruct")
        print(f"Qwen2-VL disponible: {is_available_qwen} - {message_qwen}")
        
        # Test modèle recommandé
        recommended = registry.get_recommended_model("surveillance")
        assert recommended == "kimi-vl-a3b-thinking", f"Modèle recommandé incorrect: {recommended}"
        
        # Test comparaison de modèles
        comparison = registry.get_model_comparison()
        assert "models" in comparison
        assert "summary" in comparison
        print(f"📊 {comparison['summary']['total_models']} modèles dans le registry")


@pytest.mark.gpu
class TestVLMGPUOptimized:
    """Tests spécialement optimisés pour environnement GPU."""
    
    @pytest.mark.asyncio
    async def test_gpu_acceleration(self):
        """Test GPU: Accélération matérielle."""
        if not torch.cuda.is_available():
            pytest.skip("GPU non disponible")
        
        vlm_cpu = DynamicVisionLanguageModel(device="cpu")
        vlm_gpu = DynamicVisionLanguageModel(device="cuda")
        
        # Test chargement
        await vlm_cpu.load_model("kimi-vl-a3b-thinking")
        await vlm_gpu.load_model("kimi-vl-a3b-thinking")
        
        # Comparaison performance CPU vs GPU
        request = AnalysisRequest(
            frame_data="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
            context={},
            tools_available=[]
        )
        
        # CPU timing
        start_cpu = time.time()
        await vlm_cpu.analyze_with_tools(request, use_advanced_tools=False)
        cpu_time = time.time() - start_cpu
        
        # GPU timing
        start_gpu = time.time()
        await vlm_gpu.analyze_with_tools(request, use_advanced_tools=False)
        gpu_time = time.time() - start_gpu
        
        print(f"🚀 Performance: CPU={cpu_time:.2f}s, GPU={gpu_time:.2f}s")
        print(f"📈 Accélération GPU: {cpu_time/gpu_time:.2f}x")
        
        # Nettoyage
        vlm_cpu._unload_current_model()
        vlm_gpu._unload_current_model()


if __name__ == "__main__":
    # Exécution directe pour développement
    pytest.main([__file__, "-v", "--tb=short"])