"""
Tests unitaires pour le système VLM avec corrections des erreurs.
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch
import sys
import os

# Ajout du chemin src pour les imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.vlm.dynamic_model import DynamicVisionLanguageModel
from core.vlm.mock_models import (
    MockVLMModel, MockAdvancedToolsManager, MockModelRegistry,
    MockResponseParser, create_mock_analysis_request, create_mock_image_data
)
from core.types import AnalysisRequest, SuspicionLevel, ActionType
from core.orchestrator.vlm_orchestrator import ModernVLMOrchestrator, OrchestrationConfig, OrchestrationMode


class TestDynamicVLMModel:
    """Tests pour DynamicVisionLanguageModel."""
    
    @pytest.fixture
    def mock_vlm(self):
        """VLM avec composants mockés."""
        with patch('core.vlm.dynamic_model.VLMModelRegistry') as mock_registry, \
             patch('core.vlm.dynamic_model.AdvancedToolsManager') as mock_tools, \
             patch('core.vlm.dynamic_model.ResponseParser') as mock_parser:
            
            # Configuration des mocks
            mock_registry.return_value = MockModelRegistry()
            mock_tools.return_value = MockAdvancedToolsManager()
            mock_parser.return_value = MockResponseParser()
            
            vlm = DynamicVisionLanguageModel(
                default_model="mock-vlm-model",
                enable_fallback=True
            )
            
            # Mock des méthodes de chargement pour éviter les vrais modèles
            vlm._load_model_by_type = AsyncMock(return_value=True)
            vlm.model = MockVLMModel()
            vlm.processor = MagicMock()
            
            return vlm
    
    @pytest.mark.asyncio
    async def test_load_model_success(self, mock_vlm):
        """Test chargement réussi d'un modèle."""
        success = await mock_vlm.load_model("mock-vlm-model")
        
        assert success is True
        assert mock_vlm.is_loaded is True
        assert mock_vlm.current_model_id == "mock-vlm-model"
    
    @pytest.mark.asyncio
    async def test_load_model_fallback_prevention(self, mock_vlm):
        """Test prévention de la récursion infinie."""
        # Forcer l'échec du modèle principal
        mock_vlm._load_model_by_type = AsyncMock(return_value=False)
        
        # Le fallback ne doit pas créer de récursion
        success = await mock_vlm.load_model("mock-vlm-model")
        
        # Même si ça échoue, ça ne doit pas boucler infiniment
        assert isinstance(success, bool)
    
    @pytest.mark.asyncio
    async def test_analyze_with_tools(self, mock_vlm):
        """Test analyse avec outils avancés."""
        # Préparer le VLM
        mock_vlm.is_loaded = True
        mock_vlm.current_model_id = "mock-vlm-model"
        mock_vlm.current_config = MockModelRegistry().get_model_config("mock-vlm-model")
        
        # Mock de la génération
        mock_vlm._generate_response = AsyncMock(return_value="""
        Analyse de surveillance:
        Niveau de suspicion: MEDIUM
        Action détectée: SUSPICIOUS_ACTIVITY  
        Confiance: 0.82
        Description: Comportement suspect détecté
        """)
        
        # Requête de test
        request = create_mock_analysis_request()
        
        # Test
        result = await mock_vlm.analyze_with_tools(request, use_advanced_tools=True)
        
        assert result is not None
        assert result.suspicion_level == SuspicionLevel.MEDIUM
        assert result.confidence > 0.0
        assert len(result.tools_used) > 0
    
    @pytest.mark.asyncio
    async def test_switch_model(self, mock_vlm):
        """Test switch entre modèles."""
        # Charger un modèle initial
        await mock_vlm.load_model("mock-vlm-model")
        initial_model = mock_vlm.current_model_id
        
        # Switch vers un autre modèle
        success = await mock_vlm.switch_model("mock-kimi-vl")
        
        assert success is True
        assert mock_vlm.current_model_id != initial_model
    
    def test_device_setup(self, mock_vlm):
        """Test configuration du device."""
        assert mock_vlm.device is not None
        # Le device doit être valide (cpu, cuda, mps)
        assert str(mock_vlm.device) in ["cpu", "cuda", "mps", "cuda:0"]
    
    def test_system_status(self, mock_vlm):
        """Test récupération du statut système."""
        status = mock_vlm.get_system_status()
        
        assert "current_model" in status
        assert "available_models" in status
        assert "tools_status" in status
        assert "system" in status
    
    def test_model_recommendations(self, mock_vlm):
        """Test recommandations de modèles."""
        recommendations = mock_vlm.get_model_recommendations()
        
        assert "surveillance_principal" in recommendations
        assert "haute_performance" in recommendations
        assert "economie_memoire" in recommendations


class TestOrchestrator:
    """Tests pour ModernVLMOrchestrator."""
    
    @pytest.fixture
    def mock_orchestrator(self):
        """Orchestrateur avec VLM mocké."""
        config = OrchestrationConfig(
            mode=OrchestrationMode.BALANCED,
            enable_advanced_tools=True,
            max_concurrent_tools=4
        )
        
        with patch('core.orchestrator.vlm_orchestrator.DynamicVisionLanguageModel') as mock_vlm_class:
            mock_vlm = AsyncMock()
            mock_vlm.is_loaded = True
            mock_vlm.analyze_with_tools = AsyncMock()
            mock_vlm_class.return_value = mock_vlm
            
            orchestrator = ModernVLMOrchestrator(
                vlm_model_name="mock-vlm-model",
                config=config
            )
            orchestrator.vlm = mock_vlm
            
            return orchestrator
    
    @pytest.mark.asyncio
    async def test_analyze_surveillance_frame(self, mock_orchestrator):
        """Test analyse complète d'un frame."""
        # Mock de la réponse VLM
        mock_response = MagicMock()
        mock_response.suspicion_level = SuspicionLevel.LOW
        mock_response.action_type = ActionType.NORMAL_SHOPPING
        mock_response.confidence = 0.85
        mock_response.description = "Test analysis"
        mock_response.tools_used = ["sam2_segmentator"]
        mock_response.recommendations = ["Continue monitoring"]
        
        mock_orchestrator.vlm.analyze_with_tools.return_value = mock_response
        
        # Test
        result = await mock_orchestrator.analyze_surveillance_frame(
            frame_data=create_mock_image_data(),
            detections=[],
            context={"test": True}
        )
        
        assert result is not None
        assert result.confidence > 0.0
        assert len(result.tools_used) > 0
    
    def test_tool_selection_modes(self, mock_orchestrator):
        """Test sélection des outils selon les modes."""
        # Test mode FAST
        mock_orchestrator.config.mode = OrchestrationMode.FAST
        fast_tools = mock_orchestrator._select_tools_for_mode()
        assert len(fast_tools) == 3
        assert "dino_features" in fast_tools
        
        # Test mode BALANCED
        mock_orchestrator.config.mode = OrchestrationMode.BALANCED
        balanced_tools = mock_orchestrator._select_tools_for_mode()
        assert len(balanced_tools) == 6
        assert "sam2_segmentator" in balanced_tools
        
        # Test mode THOROUGH
        mock_orchestrator.config.mode = OrchestrationMode.THOROUGH
        thorough_tools = mock_orchestrator._select_tools_for_mode()
        assert len(thorough_tools) == 8
    
    @pytest.mark.asyncio
    async def test_batch_analyze(self, mock_orchestrator):
        """Test analyse par batch."""
        # Mock des frames
        frames_data = [
            {
                "frame_data": create_mock_image_data(),
                "detections": [],
                "context": {"frame_id": i}
            }
            for i in range(3)
        ]
        
        # Mock de la réponse
        mock_response = MagicMock()
        mock_response.confidence = 0.8
        mock_orchestrator.vlm.analyze_with_tools.return_value = mock_response
        mock_orchestrator.analyze_surveillance_frame = AsyncMock(return_value=mock_response)
        
        # Test
        results = await mock_orchestrator.batch_analyze(frames_data, max_concurrent=2)
        
        assert len(results) == 3
        assert all(r.confidence > 0.0 for r in results)
    
    @pytest.mark.asyncio
    async def test_health_check(self, mock_orchestrator):
        """Test vérification de santé."""
        health = await mock_orchestrator.health_check()
        
        assert "vlm_loaded" in health
        assert "tools_available" in health
        assert "system_responsive" in health
    
    def test_config_update(self, mock_orchestrator):
        """Test mise à jour de configuration."""
        new_config = OrchestrationConfig(
            mode=OrchestrationMode.FAST,
            max_concurrent_tools=2
        )
        
        mock_orchestrator.update_config(new_config)
        
        assert mock_orchestrator.config.mode == OrchestrationMode.FAST
        assert mock_orchestrator.config.max_concurrent_tools == 2


class TestErrorHandling:
    """Tests de gestion d'erreurs."""
    
    @pytest.mark.asyncio
    async def test_model_loading_failure(self):
        """Test gestion d'échec de chargement."""
        with patch('core.vlm.dynamic_model.VLMModelRegistry') as mock_registry:
            mock_registry.return_value.validate_model_availability.return_value = (False, "Model not found")
            
            vlm = DynamicVisionLanguageModel(enable_fallback=False)
            success = await vlm.load_model("nonexistent-model")
            
            assert success is False
    
    @pytest.mark.asyncio
    async def test_analysis_error_handling(self):
        """Test gestion d'erreurs d'analyse."""
        with patch('core.vlm.dynamic_model.VLMModelRegistry') as mock_registry, \
             patch('core.vlm.dynamic_model.AdvancedToolsManager') as mock_tools, \
             patch('core.vlm.dynamic_model.ResponseParser') as mock_parser:
            
            mock_registry.return_value = MockModelRegistry()
            mock_tools.return_value = MockAdvancedToolsManager()
            mock_parser.return_value = MockResponseParser()
            
            vlm = DynamicVisionLanguageModel()
            vlm.is_loaded = False  # Forcer l'état non chargé
            
            request = create_mock_analysis_request()
            
            # Mock l'échec de chargement
            vlm.load_model = AsyncMock(return_value=False)
            
            with pytest.raises(Exception):
                await vlm.analyze_with_tools(request)
    
    def test_torch_dtype_handling(self):
        """Test gestion des types torch."""
        with patch('core.vlm.dynamic_model.VLMModelRegistry') as mock_registry:
            mock_config = MockModelRegistry().get_model_config("mock-vlm-model")
            mock_config.default_params["torch_dtype"] = "auto"  # Cas problématique
            
            mock_registry.return_value.get_model_config.return_value = mock_config
            
            vlm = DynamicVisionLanguageModel()
            
            # Le VLM doit gérer le cas "auto" sans erreur
            assert vlm is not None


class TestIntegrationBasic:
    """Tests d'intégration basiques."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_analysis(self):
        """Test pipeline complet d'analyse."""
        with patch('core.vlm.dynamic_model.VLMModelRegistry') as mock_registry, \
             patch('core.vlm.dynamic_model.AdvancedToolsManager') as mock_tools, \
             patch('core.vlm.dynamic_model.ResponseParser') as mock_parser:
            
            # Setup des mocks
            mock_registry.return_value = MockModelRegistry()
            mock_tools.return_value = MockAdvancedToolsManager()
            mock_parser.return_value = MockResponseParser()
            
            # Création des composants
            vlm = DynamicVisionLanguageModel(default_model="mock-vlm-model")
            vlm._load_model_by_type = AsyncMock(return_value=True)
            vlm.model = MockVLMModel()
            vlm.processor = MagicMock()
            vlm._generate_response = AsyncMock(return_value="Mock analysis response")
            
            config = OrchestrationConfig(mode=OrchestrationMode.BALANCED)
            orchestrator = ModernVLMOrchestrator("mock-vlm-model", config)
            orchestrator.vlm = vlm
            
            # Test pipeline complet
            await vlm.load_model()
            
            frame_data = create_mock_image_data()
            result = await orchestrator.analyze_surveillance_frame(
                frame_data=frame_data,
                detections=[],
                context={"test": "integration"}
            )
            
            assert result is not None
            assert result.confidence > 0.0


if __name__ == "__main__":
    # Exécution des tests
    pytest.main([__file__, "-v", "--tb=short"])