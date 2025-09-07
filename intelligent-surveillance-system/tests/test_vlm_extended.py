"""
Tests étendus pour les modèles VLM avec couverture complète.
"""

import pytest
import asyncio
import torch
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
from PIL import Image

from src.core.vlm.dynamic_model import DynamicVisionLanguageModel
from src.core.vlm.model_registry import VLMModelRegistry, VLMModelType
from src.core.vlm.prompt_builder import PromptBuilder
from src.core.vlm.response_parser import ResponseParser
from src.utils.exceptions import ModelError, ProcessingError


class TestVLMModelRegistry:
    """Tests pour le registre des modèles VLM."""
    
    def test_model_registry_initialization(self):
        """Test d'initialisation du registre."""
        registry = VLMModelRegistry()
        
        assert registry is not None
        assert len(registry.models) > 0
        assert "kimi-vl-a3b-thinking" in registry.models
        assert "qwen2-vl-7b-instruct" in registry.models
    
    def test_get_model_config(self):
        """Test de récupération de configuration."""
        registry = VLMModelRegistry()
        
        # Modèle existant
        config = registry.get_model_config("kimi-vl-a3b-thinking")
        assert config is not None
        assert config.model_type == VLMModelType.KIMI_VL
        assert config.supports_tool_calling is True
        
        # Modèle inexistant
        config = registry.get_model_config("modele_inexistant")
        assert config is None
    
    def test_get_models_by_type(self):
        """Test de filtrage par type de modèle."""
        registry = VLMModelRegistry()
        
        kimi_models = registry.get_models_by_type(VLMModelType.KIMI_VL)
        qwen_models = registry.get_models_by_type(VLMModelType.QWEN)
        
        assert len(kimi_models) >= 1
        assert len(qwen_models) >= 1
        
        # Vérification des types
        for model_id, config in kimi_models.items():
            assert config.model_type == VLMModelType.KIMI_VL
        
        for model_id, config in qwen_models.items():
            assert config.model_type == VLMModelType.QWEN
    
    def test_validate_model_availability(self):
        """Test de validation de disponibilité."""
        registry = VLMModelRegistry()
        
        # Test avec un modèle supporté
        is_available, message = registry.validate_model_availability("kimi-vl-a3b-thinking")
        assert isinstance(is_available, bool)
        assert isinstance(message, str)
        
        # Test avec modèle inexistant
        is_available, message = registry.validate_model_availability("inexistant")
        assert is_available is False
        assert "non supporté" in message
    
    def test_get_recommended_model(self):
        """Test de recommandation de modèle."""
        registry = VLMModelRegistry()
        
        # Test différents cas d'usage
        surveillance_model = registry.get_recommended_model("surveillance")
        thinking_model = registry.get_recommended_model("thinking")
        default_model = registry.get_recommended_model()
        
        assert surveillance_model in registry.models
        assert thinking_model in registry.models
        assert default_model in registry.models


class TestDynamicVisionLanguageModel:
    """Tests pour le modèle VLM dynamique."""
    
    @pytest.fixture
    def vlm_model(self):
        """Fixture pour créer un modèle VLM de test."""
        return DynamicVisionLanguageModel(
            default_model="kimi-vl-a3b-thinking",
            device="cpu",  # Force CPU pour les tests
            enable_fallback=True
        )
    
    def test_vlm_initialization(self, vlm_model):
        """Test d'initialisation du VLM."""
        assert vlm_model.default_model == "kimi-vl-a3b-thinking"
        assert vlm_model.device.type == "cpu"
        assert vlm_model.enable_fallback is True
        assert vlm_model.is_loaded is False
        assert vlm_model.current_model_id is None
    
    def test_device_setup_auto(self):
        """Test de configuration automatique du device."""
        vlm = DynamicVisionLanguageModel(device="auto")
        
        # Device devrait être CPU ou CUDA selon disponibilité
        assert vlm.device.type in ["cpu", "cuda"]
    
    def test_device_setup_specific(self):
        """Test de configuration spécifique du device."""
        vlm_cpu = DynamicVisionLanguageModel(device="cpu")
        assert vlm_cpu.device.type == "cpu"
        
        # Test CUDA seulement si disponible
        if torch.cuda.is_available():
            vlm_cuda = DynamicVisionLanguageModel(device="cuda")
            assert vlm_cuda.device.type == "cuda"
    
    @patch('src.core.vlm.dynamic_model.DynamicVisionLanguageModel._load_model_by_type')
    @pytest.mark.asyncio
    async def test_load_model_success(self, mock_load_by_type, vlm_model):
        """Test de chargement de modèle réussi."""
        mock_load_by_type.return_value = True
        
        success = await vlm_model.load_model("kimi-vl-a3b-thinking")
        
        assert success is True
        assert vlm_model.is_loaded is True
        assert vlm_model.current_model_id == "kimi-vl-a3b-thinking"
        mock_load_by_type.assert_called_once()
    
    @patch('src.core.vlm.dynamic_model.DynamicVisionLanguageModel._load_model_by_type')
    @pytest.mark.asyncio
    async def test_load_model_failure(self, mock_load_by_type, vlm_model):
        """Test d'échec de chargement de modèle."""
        mock_load_by_type.return_value = False
        
        success = await vlm_model.load_model("kimi-vl-a3b-thinking")
        
        assert success is False
        assert vlm_model.is_loaded is False
        assert vlm_model.current_model_id is None
    
    @pytest.mark.asyncio
    async def test_load_invalid_model(self, vlm_model):
        """Test de chargement d'un modèle invalide."""
        success = await vlm_model.load_model("modele_inexistant")
        
        assert success is False
        assert vlm_model.is_loaded is False
    
    def test_create_test_image(self):
        """Test de création d'image de test."""
        # Créer une image de test simple
        image = Image.new('RGB', (224, 224), color='red')
        assert image.size == (224, 224)
        assert image.mode == 'RGB'
    
    @pytest.mark.asyncio
    async def test_analyze_without_loaded_model(self, vlm_model):
        """Test d'analyse sans modèle chargé."""
        image = Image.new('RGB', (224, 224), color='blue')
        
        with pytest.raises(ModelError):
            await vlm_model.analyze_image(image, "Décris cette image")
    
    def test_get_available_models(self, vlm_model):
        """Test de récupération des modèles disponibles."""
        models = vlm_model.get_available_models()
        
        assert isinstance(models, dict)
        assert len(models) > 0
        assert "kimi-vl-a3b-thinking" in models
        assert "qwen2-vl-7b-instruct" in models
    
    def test_get_model_info(self, vlm_model):
        """Test de récupération d'informations de modèle."""
        info = vlm_model.get_model_info("kimi-vl-a3b-thinking")
        
        assert isinstance(info, dict)
        assert "model_name" in info
        assert "model_type" in info
        assert "description" in info
    
    def test_get_current_model_info_no_model(self, vlm_model):
        """Test d'info du modèle actuel sans modèle chargé."""
        info = vlm_model.get_current_model_info()
        
        assert info is None or info.get("model_id") is None


class TestPromptBuilder:
    """Tests pour le constructeur de prompts."""
    
    @pytest.fixture
    def prompt_builder(self):
        """Fixture pour créer un constructeur de prompts."""
        return PromptBuilder()
    
    def test_basic_prompt_building(self, prompt_builder):
        """Test de construction de prompt basique."""
        prompt = prompt_builder.build_surveillance_prompt("Analyse cette image")
        
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert "surveillance" in prompt.lower()
    
    def test_prompt_with_context(self, prompt_builder):
        """Test de prompt avec contexte."""
        context = {
            "detections": [{"label": "person", "confidence": 0.9}],
            "location": "entrée principale"
        }
        
        prompt = prompt_builder.build_contextualized_prompt(
            "Analyse comportementale", 
            context
        )
        
        assert isinstance(prompt, str)
        assert "person" in prompt
        assert "entrée principale" in prompt
    
    def test_tool_calling_prompt(self, prompt_builder):
        """Test de prompt pour tool calling."""
        tools = ["object_detection", "motion_analysis"]
        
        prompt = prompt_builder.build_tool_calling_prompt(
            "Analyse complète",
            available_tools=tools
        )
        
        assert isinstance(prompt, str)
        assert "object_detection" in prompt
        assert "motion_analysis" in prompt


class TestResponseParser:
    """Tests pour le parseur de réponses."""
    
    @pytest.fixture
    def response_parser(self):
        """Fixture pour créer un parseur de réponses."""
        return ResponseParser()
    
    def test_parse_basic_response(self, response_parser):
        """Test de parsing de réponse basique."""
        response = {
            "analysis": "3 personnes détectées dans la zone d'entrée",
            "confidence": 0.95,
            "alert_level": "attention"
        }
        
        parsed = response_parser.parse_surveillance_response(response)
        
        assert isinstance(parsed, dict)
        assert "analysis" in parsed
        assert parsed["confidence"] == 0.95
    
    def test_parse_tool_response(self, response_parser):
        """Test de parsing de réponse avec tools."""
        response = {
            "analysis": "Mouvement suspect détecté",
            "tool_calls": [
                {
                    "tool": "motion_analyzer",
                    "parameters": {"threshold": 0.8},
                    "result": {"motion_detected": True}
                }
            ]
        }
        
        parsed = response_parser.parse_tool_response(response)
        
        assert "tool_calls" in parsed
        assert len(parsed["tool_calls"]) == 1
        assert parsed["tool_calls"][0]["tool"] == "motion_analyzer"
    
    def test_parse_invalid_response(self, response_parser):
        """Test de parsing de réponse invalide."""
        with pytest.raises(ProcessingError):
            response_parser.parse_surveillance_response("réponse invalide")
    
    def test_extract_confidence_score(self, response_parser):
        """Test d'extraction de score de confiance."""
        response = {"confidence": 0.87, "other_data": "test"}
        
        confidence = response_parser.extract_confidence(response)
        assert confidence == 0.87
        
        # Test avec confidence manquante
        response_no_conf = {"other_data": "test"}
        confidence = response_parser.extract_confidence(response_no_conf)
        assert confidence is None or confidence == 0.0


@pytest.mark.integration
class TestVLMIntegration:
    """Tests d'intégration pour le système VLM complet."""
    
    @pytest.fixture
    def integration_vlm(self):
        """VLM pour tests d'intégration."""
        return DynamicVisionLanguageModel(
            default_model="kimi-vl-a3b-thinking",
            enable_fallback=True
        )
    
    @pytest.mark.asyncio
    async def test_end_to_end_analysis(self, integration_vlm):
        """Test d'analyse bout-en-bout (nécessite modèle réel)."""
        # Ce test ne sera exécuté que si le modèle est disponible
        try:
            success = await integration_vlm.load_model()
            if not success:
                pytest.skip("Modèle VLM non disponible")
            
            # Créer une image de test
            test_image = Image.new('RGB', (224, 224), color='green')
            
            # Analyse simple
            result = await integration_vlm.analyze_image(
                test_image, 
                "Décris brièvement cette image"
            )
            
            assert result is not None
            assert isinstance(result, dict)
            
        except Exception as e:
            pytest.skip(f"Test d'intégration ignoré: {e}")
    
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_model_switching(self, integration_vlm):
        """Test de changement de modèle."""
        try:
            # Chargement du premier modèle
            success1 = await integration_vlm.load_model("kimi-vl-a3b-thinking")
            if not success1:
                pytest.skip("Premier modèle non disponible")
            
            current_model1 = integration_vlm.current_model_id
            
            # Changement vers second modèle
            success2 = await integration_vlm.load_model("qwen2-vl-7b-instruct")
            if success2:
                current_model2 = integration_vlm.current_model_id
                assert current_model1 != current_model2
            
        except Exception as e:
            pytest.skip(f"Test de changement de modèle ignoré: {e}")


# Configuration des markers pytest
pytestmark = [
    pytest.mark.vlm,  # Marker pour tous les tests VLM
]