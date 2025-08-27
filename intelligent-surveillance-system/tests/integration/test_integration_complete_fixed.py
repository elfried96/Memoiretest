"""
Tests d'intégration complets pour le système de surveillance.
"""

import pytest
import asyncio
import numpy as np
import cv2
import base64
from unittest.mock import AsyncMock, MagicMock, patch
import sys
import os
from io import BytesIO
from PIL import Image

# Ajout du chemin src pour les imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from src.core.vlm.dynamic_model import DynamicVisionLanguageModel
from src.core.vlm.mock_models import MockModelRegistry, MockAdvancedToolsManager, MockResponseParser
from src.core.orchestrator.vlm_orchestrator import ModernVLMOrchestrator, OrchestrationConfig, OrchestrationMode
from src.core.types import Detection, BoundingBox, AnalysisRequest, SuspicionLevel, ActionType
from src.detection.yolo_detector import YOLODetector
from src.detection.tracking.byte_tracker import BYTETracker


class TestSystemIntegration:
    """Tests d'intégration du système complet."""
    
    @pytest.fixture
    def mock_surveillance_system(self):
        """Système de surveillance complet avec mocks."""
        # Mock VLM
        with patch('core.vlm.dynamic_model.VLMModelRegistry') as mock_registry, \
             patch('core.vlm.dynamic_model.AdvancedToolsManager') as mock_tools, \
             patch('core.vlm.dynamic_model.ResponseParser') as mock_parser:
            
            mock_registry.return_value = MockModelRegistry()
            mock_tools.return_value = MockAdvancedToolsManager()
            mock_parser.return_value = MockResponseParser()
            
            # VLM avec mocks
            vlm = DynamicVisionLanguageModel(default_model="mock-vlm-model")
            vlm._load_model_by_type = AsyncMock(return_value=True)
            vlm._generate_response = AsyncMock(return_value="""
            Analyse de surveillance:
            Niveau de suspicion: MEDIUM
            Action détectée: SUSPICIOUS_ACTIVITY
            Confiance: 0.82
            Description: Comportement suspect détecté près des produits
            """)
            
            # Orchestrateur
            config = OrchestrationConfig(
                mode=OrchestrationMode.BALANCED,
                enable_advanced_tools=True,
                max_concurrent_tools=4
            )
            orchestrator = ModernVLMOrchestrator("mock-vlm-model", config)
            orchestrator.vlm = vlm
            
            # Mock YOLO (pour éviter le téléchargement du modèle)
            mock_yolo = MagicMock()
            mock_yolo.detect.return_value = self._create_mock_yolo_results()
            
            # Mock Tracker
            mock_tracker = MagicMock()
            mock_tracker.update.return_value = self._create_mock_tracked_objects()
            
            return {
                'vlm': vlm,
                'orchestrator': orchestrator,
                'yolo': mock_yolo,
                'tracker': mock_tracker
            }
    
    def _create_test_frame(self) -> np.ndarray:
        """Crée un frame de test."""
        # Image 640x480 avec du contenu simple
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Ajouter un rectangle "personne" simulé
        cv2.rectangle(frame, (100, 100), (200, 400), (255, 255, 255), -1)
        cv2.putText(frame, "PERSON", (110, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        return frame
    
    def _create_mock_yolo_results(self):
        """Crée des résultats YOLO mockés."""
        mock_result = MagicMock()
        
        # Mock boxes
        mock_box = MagicMock()
        mock_box.xyxy = [np.array([100, 100, 200, 400])]  # [x1, y1, x2, y2]
        mock_box.conf = [np.array([0.85])]  # Confiance
        mock_box.cls = [np.array([0])]  # Classe 0 = person
        
        mock_result.boxes = [mock_box]
        mock_result.names = {0: "person", 1: "bicycle", 2: "car"}
        
        return [mock_result]
    
    def _create_mock_tracked_objects(self):
        """Crée des objets trackés mockés."""
        return [
            {
                "track_id": "track_1",
                "detection": Detection(
                    bbox=BoundingBox(x1=100, y1=100, x2=200, y2=400),
                    confidence=0.85,
                    class_name="person",
                    track_id=1
                ),
                "age": 5,
                "status": "active"
            }
        ]
    
    def _frame_to_base64(self, frame: np.ndarray) -> str:
        """Convertit un frame en base64."""
        # Conversion BGR → RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Encodage JPEG puis base64
        _, buffer = cv2.imencode('.jpg', frame_rgb, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_b64 = base64.b64encode(buffer).decode('utf-8')
        
        return frame_b64
    
    def _create_detections_from_yolo(self, yolo_results) -> list:
        """Convertit les résultats YOLO en détections."""
        detections = []
        
        for result in yolo_results:
            if hasattr(result, 'boxes') and result.boxes:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    
                    class_name = result.names.get(cls, f"class_{cls}")
                    
                    detection = Detection(
                        bbox=BoundingBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2)),
                        confidence=conf,
                        class_name=class_name,
                        track_id=None
                    )
                    detections.append(detection)
        
        return detections
    
    @pytest.mark.asyncio
    async def test_complete_surveillance_pipeline(self, mock_surveillance_system):
        """Test du pipeline complet de surveillance."""
        vlm = mock_surveillance_system['vlm']
        orchestrator = mock_surveillance_system['orchestrator'] 
        yolo = mock_surveillance_system['yolo']
        tracker = mock_surveillance_system['tracker']
        
        # Étape 1: Chargement du VLM
        vlm_success = await vlm.load_model()
        assert vlm_success is True
        
        # Étape 2: Création d'un frame de test
        frame = self._create_test_frame()
        assert frame.shape == (480, 640, 3)
        
        # Étape 3: Détection YOLO
        yolo_results = yolo.detect(frame)
        detections = self._create_detections_from_yolo(yolo_results)
        
        assert len(detections) > 0
        assert detections[0].class_name == "person"
        assert detections[0].confidence > 0.8
        
        # Étape 4: Tracking
        tracked_objects = tracker.update(detections)
        assert len(tracked_objects) > 0
        assert tracked_objects[0]["track_id"] == "track_1"
        
        # Étape 5: Analyse VLM si personne détectée
        person_detected = any(d.class_name == "person" for d in detections)
        if person_detected:
            frame_b64 = self._frame_to_base64(frame)
            
            context = {
                "frame_id": 12345,
                "timestamp": 1704886800.0,
                "location": "Test Store",
                "camera": "TEST_CAM_01",
                "person_count": len([d for d in detections if d.class_name == "person"]),
                "total_objects": len(detections)
            }
            
            vlm_analysis = await orchestrator.analyze_surveillance_frame(
                frame_data=frame_b64,
                detections=detections,
                context=context
            )
            
            # Vérifications
            assert vlm_analysis is not None
            assert vlm_analysis.suspicion_level in [SuspicionLevel.LOW, SuspicionLevel.MEDIUM, SuspicionLevel.HIGH]
            assert vlm_analysis.confidence > 0.0
            assert vlm_analysis.action_type in [ActionType.NORMAL_SHOPPING, ActionType.SUSPICIOUS_MOVEMENT, ActionType.POTENTIAL_THEFT]
            assert len(vlm_analysis.tools_used) > 0
        
        print("✅ Pipeline complet validé avec succès")
    
    @pytest.mark.asyncio
    async def test_multiple_frames_processing(self, mock_surveillance_system):
        """Test traitement de plusieurs frames."""
        orchestrator = mock_surveillance_system['orchestrator']
        
        # Préparer plusieurs frames
        frames_data = []
        for i in range(3):
            frame = self._create_test_frame()
            frame_b64 = self._frame_to_base64(frame)
            
            frames_data.append({
                "frame_data": frame_b64,
                "detections": self._create_detections_from_yolo(
                    mock_surveillance_system['yolo'].detect(frame)
                ),
                "context": {
                    "frame_id": 1000 + i,
                    "timestamp": 1704886800.0 + i,
                    "location": "Test Store"
                }
            })
        
        # Traitement par batch
        results = await orchestrator.batch_analyze(frames_data, max_concurrent=2)
        
        assert len(results) == 3
        assert all(r.confidence > 0.0 for r in results)
        assert all(r.suspicion_level is not None for r in results)
        
        print("✅ Traitement multi-frames validé")
    
    @pytest.mark.asyncio
    async def test_orchestration_modes(self, mock_surveillance_system):
        """Test des différents modes d'orchestration."""
        orchestrator = mock_surveillance_system['orchestrator']
        frame_b64 = self._frame_to_base64(self._create_test_frame())
        
        modes = [OrchestrationMode.FAST, OrchestrationMode.BALANCED, OrchestrationMode.THOROUGH]
        
        for mode in modes:
            # Mise à jour du mode
            config = OrchestrationConfig(mode=mode, enable_advanced_tools=True)
            orchestrator.update_config(config)
            
            # Test avec ce mode
            result = await orchestrator.analyze_surveillance_frame(
                frame_data=frame_b64,
                detections=[],
                context={"test_mode": mode.value}
            )
            
            assert result is not None
            assert result.confidence > 0.0
            
            # Vérifier que le nombre d'outils varie selon le mode
            tools_used = len(result.tools_used)
            if mode == OrchestrationMode.FAST:
                assert tools_used >= 1  # Au moins un outil
            elif mode == OrchestrationMode.THOROUGH:
                assert tools_used >= 3  # Plus d'outils
        
        print("✅ Modes d'orchestration validés")
    
    @pytest.mark.asyncio
    async def test_error_recovery(self, mock_surveillance_system):
        """Test récupération d'erreurs."""
        orchestrator = mock_surveillance_system['orchestrator']
        vlm = mock_surveillance_system['vlm']
        
        # Simuler une erreur dans le VLM
        original_analyze = vlm.analyze_with_tools
        vlm.analyze_with_tools = AsyncMock(side_effect=Exception("Test error"))
        
        # Test que l'orchestrateur gère l'erreur
        frame_b64 = self._frame_to_base64(self._create_test_frame())
        
        result = await orchestrator.analyze_surveillance_frame(
            frame_data=frame_b64,
            detections=[],
            context={"error_test": True}
        )
        
        # L'orchestrateur doit retourner une réponse d'erreur valide
        assert result is not None
        assert result.confidence == 0.0
        assert "Erreur d'analyse" in result.description
        
        # Restaurer la fonction normale
        vlm.analyze_with_tools = original_analyze
        
        print("✅ Récupération d'erreurs validée")
    
    @pytest.mark.asyncio 
    async def test_performance_monitoring(self, mock_surveillance_system):
        """Test monitoring des performances."""
        orchestrator = mock_surveillance_system['orchestrator']
        
        # Effectuer plusieurs analyses pour générer des stats
        frame_b64 = self._frame_to_base64(self._create_test_frame())
        
        for i in range(5):
            await orchestrator.analyze_surveillance_frame(
                frame_data=frame_b64,
                detections=[],
                context={"perf_test": i}
            )
        
        # Vérifier les statistiques
        status = orchestrator.get_system_status()
        
        assert "performance" in status
        assert status["performance"]["total_analyses"] == 5
        assert status["performance"]["average_response_time"] > 0.0
        assert "tools_usage" in status["performance"]
        
        print("✅ Monitoring des performances validé")
    
    @pytest.mark.asyncio
    async def test_health_check(self, mock_surveillance_system):
        """Test vérification de santé du système."""
        orchestrator = mock_surveillance_system['orchestrator']
        vlm = mock_surveillance_system['vlm']
        
        # Charger le VLM
        await vlm.load_model()
        
        # Health check
        health = await orchestrator.health_check()
        
        assert "vlm_loaded" in health
        assert "tools_available" in health  
        assert "system_responsive" in health
        
        # Le VLM doit être opérationnel
        assert health["vlm_loaded"] is True
        assert health["tools_available"] is True
        
        print("✅ Health check validé")


class TestRealWorldScenarios:
    """Tests de scénarios réels."""
    
    @pytest.fixture
    def surveillance_scenarios(self):
        """Scénarios de surveillance réalistes."""
        return {
            "normal_shopping": {
                "description": "Client normal qui fait ses courses",
                "person_count": 1,
                "expected_suspicion": SuspicionLevel.LOW,
                "expected_action": ActionType.NORMAL_SHOPPING
            },
            "suspicious_behavior": {
                "description": "Personne qui traîne près des produits chers",
                "person_count": 1,
                "expected_suspicion": SuspicionLevel.MEDIUM,
                "expected_action": ActionType.SUSPICIOUS_MOVEMENT
            },
            "multiple_people": {
                "description": "Plusieurs personnes dans le magasin",
                "person_count": 3,
                "expected_suspicion": SuspicionLevel.LOW,
                "expected_action": ActionType.NORMAL_SHOPPING
            },
            "empty_store": {
                "description": "Magasin vide",
                "person_count": 0,
                "expected_suspicion": SuspicionLevel.LOW,
                "expected_action": ActionType.NORMAL_SHOPPING
            }
        }
    
    @pytest.mark.asyncio
    async def test_scenario_normal_shopping(self, surveillance_scenarios):
        """Test scénario shopping normal."""
        with patch('core.vlm.dynamic_model.VLMModelRegistry') as mock_registry, \
             patch('core.vlm.dynamic_model.AdvancedToolsManager') as mock_tools, \
             patch('core.vlm.dynamic_model.ResponseParser') as mock_parser:
            
            # Setup
            mock_registry.return_value = MockModelRegistry()
            mock_tools.return_value = MockAdvancedToolsManager()
            mock_parser.return_value = MockResponseParser()
            
            vlm = DynamicVisionLanguageModel()
            vlm._load_model_by_type = AsyncMock(return_value=True)
            vlm._generate_response = AsyncMock(return_value="""
            Analyse de surveillance:
            Niveau de suspicion: LOW
            Action détectée: NORMAL_SHOPPING
            Confiance: 0.90
            Description: Client effectue ses achats normalement
            """)
            
            config = OrchestrationConfig(mode=OrchestrationMode.BALANCED)
            orchestrator = ModernVLMOrchestrator("mock-vlm-model", config)
            orchestrator.vlm = vlm
            
            # Test
            await vlm.load_model()
            
            # Créer un frame simple
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame_b64 = base64.b64encode(cv2.imencode('.jpg', frame)[1]).decode()
            
            result = await orchestrator.analyze_surveillance_frame(
                frame_data=frame_b64,
                detections=[],
                context=surveillance_scenarios["normal_shopping"]
            )
            
            assert result.suspicion_level == SuspicionLevel.LOW
            assert result.confidence > 0.8
            
            print("✅ Scénario shopping normal validé")


if __name__ == "__main__":
    # Exécution des tests
    pytest.main([__file__, "-v", "--tb=short", "-s"])