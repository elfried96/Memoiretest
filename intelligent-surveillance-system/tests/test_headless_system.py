"""
Tests pour le système de surveillance headless refactorisé.
"""

import pytest
import asyncio
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from PIL import Image

from src.core.headless import (
    HeadlessSurveillanceSystem,
    SurveillanceResult,
    AlertLevel,
    VideoProcessor,
    FrameAnalyzer
)
from src.core.orchestrator.vlm_orchestrator import OrchestrationMode


class TestSurveillanceResult:
    """Tests pour le modèle de résultats de surveillance."""
    
    def test_surveillance_result_creation(self):
        """Test de création d'un résultat de surveillance."""
        result = SurveillanceResult(
            frame_id=1,
            timestamp=1234567890.0,
            detections_count=3,
            persons_detected=2,
            alert_level=AlertLevel.ATTENTION,
            processing_time=0.5
        )
        
        assert result.frame_id == 1
        assert result.timestamp == 1234567890.0
        assert result.detections_count == 3
        assert result.persons_detected == 2
        assert result.alert_level == AlertLevel.ATTENTION
        assert result.processing_time == 0.5
        assert result.actions_taken == []  # valeur par défaut
    
    def test_surveillance_result_to_dict(self):
        """Test de conversion en dictionnaire."""
        result = SurveillanceResult(
            frame_id=5,
            timestamp=1234567890.0,
            detections_count=1,
            persons_detected=1,
            alert_level=AlertLevel.NORMAL,
            actions_taken=["log_event"]
        )
        
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert result_dict["frame_id"] == 5
        assert result_dict["alert_level"] == "normal"
        assert result_dict["actions_taken"] == ["log_event"]
    
    def test_alert_level_enum(self):
        """Test de l'énumération AlertLevel."""
        assert AlertLevel.NORMAL.value == "normal"
        assert AlertLevel.ATTENTION.value == "attention"
        assert AlertLevel.ALERTE.value == "alerte"
        assert AlertLevel.CRITIQUE.value == "critique"


class TestVideoProcessor:
    """Tests pour le processeur vidéo."""
    
    @pytest.fixture
    def video_processor(self):
        """Fixture pour créer un processeur vidéo de test."""
        return VideoProcessor(
            source="0",  # Webcam fictive
            frame_skip=1,
            max_frames=10
        )
    
    def test_video_processor_initialization(self, video_processor):
        """Test d'initialisation du processeur vidéo."""
        assert video_processor.source == "0"
        assert video_processor.frame_skip == 1
        assert video_processor.max_frames == 10
        assert video_processor.frame_count == 0
        assert video_processor.processed_frames == 0
    
    @patch('cv2.VideoCapture')
    def test_initialize_capture_webcam(self, mock_cv2_cap, video_processor):
        """Test d'initialisation de capture webcam."""
        # Mock de VideoCapture
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_WIDTH: 640.0,
            cv2.CAP_PROP_FRAME_HEIGHT: 480.0
        }.get(prop, 0.0)
        mock_cv2_cap.return_value = mock_cap
        
        # Note: _initialize_capture est appelé dans __init__
        # On teste via une nouvelle instance
        processor = VideoProcessor(source="0")
        
        mock_cv2_cap.assert_called_with(0)
        mock_cap.set.assert_called()  # Vérifier que des paramètres ont été définis
    
    @patch('cv2.VideoCapture')
    def test_initialize_capture_failure(self, mock_cv2_cap):
        """Test d'échec d'initialisation de capture."""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = False
        mock_cv2_cap.return_value = mock_cap
        
        with pytest.raises(RuntimeError):
            VideoProcessor(source="invalid_source")
    
    def test_get_video_info_no_cap(self):
        """Test d'info vidéo sans capture."""
        processor = VideoProcessor.__new__(VideoProcessor)
        processor.cap = None
        
        info = processor.get_video_info()
        assert info == {}
    
    @patch('cv2.VideoCapture')
    def test_get_video_info_with_cap(self, mock_cv2_cap):
        """Test d'info vidéo avec capture."""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: 25.0,
            cv2.CAP_PROP_FRAME_WIDTH: 1920.0,
            cv2.CAP_PROP_FRAME_HEIGHT: 1080.0,
            cv2.CAP_PROP_FRAME_COUNT: 1000.0
        }.get(prop, 0.0)
        mock_cv2_cap.return_value = mock_cap
        
        processor = VideoProcessor(source="test.mp4")
        info = processor.get_video_info()
        
        assert info["fps"] == 25.0
        assert info["width"] == 1920
        assert info["height"] == 1080
        assert info["total_frames"] == 1000
    
    def test_save_frame(self, tmp_path):
        """Test de sauvegarde de frame."""
        processor = VideoProcessor(source="0")
        
        # Créer une frame de test
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        with patch('cv2.imwrite') as mock_imwrite:
            mock_imwrite.return_value = True
            
            saved_path = processor.save_frame(frame, 1, tmp_path)
            
            assert saved_path.parent == tmp_path
            assert "frame_000001.jpg" in saved_path.name
            mock_imwrite.assert_called_once()


class TestFrameAnalyzer:
    """Tests pour l'analyseur de frames."""
    
    @pytest.fixture
    def frame_analyzer(self):
        """Fixture pour créer un analyseur de frames."""
        with patch('src.core.headless.frame_analyzer.YOLODetector'), \
             patch('src.core.headless.frame_analyzer.BYTETracker'), \
             patch('src.core.headless.frame_analyzer.DynamicVisionLanguageModel'), \
             patch('src.core.headless.frame_analyzer.ModernVLMOrchestrator'):
            
            analyzer = FrameAnalyzer(
                vlm_model="kimi-vl-a3b-thinking",
                yolo_confidence=0.5
            )
            return analyzer
    
    def test_frame_analyzer_initialization(self, frame_analyzer):
        """Test d'initialisation de l'analyseur."""
        assert frame_analyzer.vlm_model_name == "kimi-vl-a3b-thinking"
        assert frame_analyzer.total_analysis_time == 0.0
        assert frame_analyzer.analysis_count == 0
    
    def test_count_persons(self, frame_analyzer):
        """Test de comptage de personnes."""
        from src.core.types import DetectedObject, BoundingBox
        
        objects = [
            DetectedObject("person", 0.9, BoundingBox(0, 0, 100, 100)),
            DetectedObject("car", 0.8, BoundingBox(100, 100, 200, 200)),
            DetectedObject("person", 0.7, BoundingBox(200, 200, 300, 300))
        ]
        
        count = frame_analyzer._count_persons(objects)
        assert count == 2
    
    def test_determine_alert_level(self, frame_analyzer):
        """Test de détermination du niveau d'alerte."""
        # 0 personne = NORMAL
        level = frame_analyzer._determine_alert_level([], 0)
        assert level == AlertLevel.NORMAL
        
        # 1 personne = ATTENTION
        level = frame_analyzer._determine_alert_level([], 1)
        assert level == AlertLevel.ATTENTION
        
        # 2-3 personnes = ALERTE
        level = frame_analyzer._determine_alert_level([], 2)
        assert level == AlertLevel.ALERTE
        
        level = frame_analyzer._determine_alert_level([], 3)
        assert level == AlertLevel.ALERTE
        
        # 4+ personnes = CRITIQUE
        level = frame_analyzer._determine_alert_level([], 4)
        assert level == AlertLevel.CRITIQUE
    
    def test_should_run_vlm_analysis(self, frame_analyzer):
        """Test de décision d'analyse VLM."""
        # Mode continuous
        assert frame_analyzer._should_run_vlm_analysis("continuous", AlertLevel.NORMAL, 1) is True
        
        # Mode periodic (toutes les 10 frames)
        assert frame_analyzer._should_run_vlm_analysis("periodic", AlertLevel.NORMAL, 10) is True
        assert frame_analyzer._should_run_vlm_analysis("periodic", AlertLevel.NORMAL, 5) is False
        
        # Mode smart (seulement sur alertes)
        assert frame_analyzer._should_run_vlm_analysis("smart", AlertLevel.NORMAL, 1) is False
        assert frame_analyzer._should_run_vlm_analysis("smart", AlertLevel.ALERTE, 1) is True
        assert frame_analyzer._should_run_vlm_analysis("smart", AlertLevel.CRITIQUE, 1) is True
    
    def test_determine_actions(self, frame_analyzer):
        """Test de détermination des actions."""
        # NORMAL - pas d'action
        actions = frame_analyzer._determine_actions(AlertLevel.NORMAL, None)
        assert len(actions) == 0
        
        # ATTENTION
        actions = frame_analyzer._determine_actions(AlertLevel.ATTENTION, None)
        assert "Surveillance accrue" in actions
        
        # ALERTE
        actions = frame_analyzer._determine_actions(AlertLevel.ALERTE, None)
        assert "Notification sécurité" in actions
        assert "Enregistrement activé" in actions
        
        # CRITIQUE
        actions = frame_analyzer._determine_actions(AlertLevel.CRITIQUE, None)
        assert "Alerte immédiate" in actions
        assert "Notification urgente" in actions
    
    def test_update_performance_stats(self, frame_analyzer):
        """Test de mise à jour des statistiques."""
        assert frame_analyzer.analysis_count == 0
        assert frame_analyzer.total_analysis_time == 0.0
        
        frame_analyzer._update_performance_stats(1.5)
        
        assert frame_analyzer.analysis_count == 1
        assert frame_analyzer.total_analysis_time == 1.5
        
        frame_analyzer._update_performance_stats(0.5)
        
        assert frame_analyzer.analysis_count == 2
        assert frame_analyzer.total_analysis_time == 2.0
    
    def test_get_performance_stats(self, frame_analyzer):
        """Test de récupération des statistiques."""
        # Sans analyses
        stats = frame_analyzer.get_performance_stats()
        assert stats["average_processing_time"] == 0.0
        assert stats["total_time"] == 0.0
        
        # Avec analyses
        frame_analyzer._update_performance_stats(2.0)
        frame_analyzer._update_performance_stats(1.0)
        
        stats = frame_analyzer.get_performance_stats()
        assert stats["average_processing_time"] == 1.5
        assert stats["total_time"] == 3.0
        assert stats["analysis_count"] == 2


class TestHeadlessSurveillanceSystem:
    """Tests pour le système de surveillance headless."""
    
    @pytest.fixture
    def surveillance_system(self, tmp_path):
        """Fixture pour créer un système de surveillance."""
        with patch('src.core.headless.surveillance_system.VideoProcessor'), \
             patch('src.core.headless.surveillance_system.FrameAnalyzer'):
            
            system = HeadlessSurveillanceSystem(
                video_source="test_video.mp4",
                vlm_model="kimi-vl-a3b-thinking",
                orchestration_mode=OrchestrationMode.BALANCED,
                output_dir=str(tmp_path)
            )
            return system
    
    def test_surveillance_system_initialization(self, surveillance_system, tmp_path):
        """Test d'initialisation du système."""
        assert surveillance_system.video_source == "test_video.mp4"
        assert surveillance_system.vlm_model == "kimi-vl-a3b-thinking"
        assert surveillance_system.orchestration_mode == OrchestrationMode.BALANCED
        assert surveillance_system.output_dir == Path(tmp_path)
        assert surveillance_system.is_running is False
        assert len(surveillance_system.results) == 0
    
    def test_update_metrics(self, surveillance_system):
        """Test de mise à jour des métriques."""
        result = SurveillanceResult(
            frame_id=1,
            timestamp=1234567890.0,
            detections_count=2,
            persons_detected=1,
            alert_level=AlertLevel.ATTENTION
        )
        
        surveillance_system._update_metrics(result)
        
        assert surveillance_system.alert_counts["attention"] == 1
        assert len(surveillance_system.recent_results) == 1
    
    def test_generate_session_summary_empty(self, surveillance_system):
        """Test de génération de résumé avec session vide."""
        summary = surveillance_system._generate_session_summary()
        
        assert summary.total_frames == 0
        assert summary.total_detections == 0
        assert summary.total_persons == 0
        assert summary.average_processing_time == 0.0
        assert len(summary.key_events) == 0
    
    def test_generate_session_summary_with_data(self, surveillance_system):
        """Test de génération de résumé avec données."""
        # Ajouter des résultats de test
        results = [
            SurveillanceResult(1, 1234567890.0, 2, 1, AlertLevel.ATTENTION, processing_time=0.5),
            SurveillanceResult(2, 1234567891.0, 3, 2, AlertLevel.ALERTE, processing_time=0.7),
            SurveillanceResult(3, 1234567892.0, 1, 0, AlertLevel.NORMAL, processing_time=0.3)
        ]
        
        surveillance_system.results = results
        surveillance_system.alert_counts = {"attention": 1, "alerte": 1, "normal": 1}
        
        summary = surveillance_system._generate_session_summary()
        
        assert summary.total_frames == 3
        assert summary.total_detections == 6
        assert summary.total_persons == 3
        assert summary.average_processing_time == 0.5  # (0.5 + 0.7 + 0.3) / 3
        assert len(summary.key_events) == 2  # ATTENTION et ALERTE
    
    def test_get_real_time_stats_empty(self, surveillance_system):
        """Test de statistiques temps réel sans données."""
        stats = surveillance_system.get_real_time_stats()
        assert stats == {}
    
    def test_get_real_time_stats_with_data(self, surveillance_system):
        """Test de statistiques temps réel avec données."""
        # Ajouter des résultats récents
        result = SurveillanceResult(1, 1234567890.0, 2, 1, AlertLevel.ATTENTION, processing_time=0.5)
        surveillance_system.recent_results.append(result)
        surveillance_system.alert_counts["attention"] = 1
        surveillance_system.results.append(result)
        
        stats = surveillance_system.get_real_time_stats()
        
        assert stats["is_running"] is False
        assert stats["frames_processed"] == 1
        assert stats["recent_avg_processing_time"] == 0.5
        assert stats["alert_counts"]["attention"] == 1


@pytest.mark.integration
class TestHeadlessIntegration:
    """Tests d'intégration pour le système headless."""
    
    @pytest.mark.asyncio
    async def test_mock_surveillance_run(self, tmp_path):
        """Test de run de surveillance mocké."""
        with patch('src.core.headless.surveillance_system.VideoProcessor') as mock_video, \
             patch('src.core.headless.surveillance_system.FrameAnalyzer') as mock_analyzer:
            
            # Mock du processeur vidéo
            mock_video_instance = Mock()
            mock_video_instance.frames_generator.return_value = [
                (1, np.zeros((480, 640, 3), dtype=np.uint8)),
                (2, np.zeros((480, 640, 3), dtype=np.uint8))
            ]
            mock_video_instance.get_video_info.return_value = {"fps": 30, "total_frames": 2}
            mock_video.return_value = mock_video_instance
            
            # Mock de l'analyseur
            mock_analyzer_instance = Mock()
            mock_analyzer_instance.analyze_frame = AsyncMock(return_value=SurveillanceResult(
                frame_id=1,
                timestamp=1234567890.0,
                detections_count=1,
                persons_detected=1,
                alert_level=AlertLevel.NORMAL,
                processing_time=0.1
            ))
            mock_analyzer_instance.get_performance_stats.return_value = {
                "average_processing_time": 0.1,
                "total_time": 0.2
            }
            mock_analyzer_instance.cleanup = AsyncMock()
            mock_analyzer.return_value = mock_analyzer_instance
            
            # Système de surveillance
            system = HeadlessSurveillanceSystem(
                video_source="test.mp4",
                max_frames=2,
                output_dir=str(tmp_path)
            )
            
            # Exécution
            summary = await system.run_surveillance()
            
            # Vérifications
            assert summary.total_frames == 2
            assert system.is_running is False
            mock_analyzer_instance.cleanup.assert_called_once()


# Configuration pytest pour ce module
pytestmark = [
    pytest.mark.headless,
    pytest.mark.unit
]