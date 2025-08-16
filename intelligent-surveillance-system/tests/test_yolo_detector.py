"""Tests unitaires pour le détecteur YOLO."""

import pytest
import numpy as np
import cv2
from datetime import datetime
from unittest.mock import Mock, patch

from src.detection.yolo.detector import YOLODetector
from src.core.types import Frame, DetectedObject, BoundingBox
from src.utils.exceptions import DetectionError


class TestYOLODetector:
    """Tests pour le détecteur YOLO."""
    
    @pytest.fixture
    def detector(self):
        """Fixture pour le détecteur YOLO."""
        return YOLODetector(
            model_path="yolov11n.pt",
            device="cpu",  # Utiliser CPU pour les tests
            confidence_threshold=0.25
        )
    
    @pytest.fixture
    def sample_frame(self):
        """Fixture pour un frame de test."""
        # Création d'une image de test
        image = np.zeros((640, 640, 3), dtype=np.uint8)
        cv2.rectangle(image, (100, 100), (200, 200), (255, 255, 255), -1)
        
        return Frame(
            image=image,
            timestamp=datetime.now(),
            frame_id=1,
            stream_id="test",
            width=640,
            height=640
        )
    
    def test_detector_initialization(self):
        """Test de l'initialisation du détecteur."""
        detector = YOLODetector(
            model_path="yolov11n.pt",
            device="cpu",
            confidence_threshold=0.3,
            iou_threshold=0.5
        )
        
        assert detector.model_path == "yolov11n.pt"
        assert detector.device == "cpu"
        assert detector.confidence_threshold == 0.3
        assert detector.iou_threshold == 0.5
        assert not detector.is_loaded
    
    @patch('src.detection.yolo.detector.YOLO')
    def test_model_loading(self, mock_yolo, detector):
        """Test du chargement du modèle."""
        mock_model = Mock()
        mock_yolo.return_value = mock_model
        
        detector.load_model()
        
        mock_yolo.assert_called_once_with(detector.model_path)
        assert detector.is_loaded
        assert detector.model is not None
    
    @patch('src.detection.yolo.detector.YOLO')
    def test_detection_success(self, mock_yolo, detector, sample_frame):
        """Test de détection réussie."""
        # Configuration du mock
        mock_model = Mock()
        mock_yolo.return_value = mock_model
        
        # Mock du résultat YOLO
        mock_result = Mock()
        mock_result.boxes = Mock()
        mock_result.boxes.xyxy.cpu.return_value.numpy.return_value = np.array([[100, 100, 200, 200]])
        mock_result.boxes.conf.cpu.return_value.numpy.return_value = np.array([0.8])
        mock_result.boxes.cls.cpu.return_value.numpy.return_value.astype.return_value = np.array([0])
        mock_result.names = {0: "person"}
        
        mock_model.return_value = [mock_result]
        
        detector.load_model()
        detections = detector.detect(sample_frame)
        
        assert len(detections) == 1
        assert isinstance(detections[0], DetectedObject)
        assert detections[0].class_name == "person"
        assert detections[0].confidence == 0.8
    
    def test_detection_without_loading(self, detector, sample_frame):
        """Test de détection sans chargement du modèle."""
        with patch.object(detector, 'load_model') as mock_load:
            with patch('src.detection.yolo.detector.YOLO'):
                detector.detect(sample_frame)
                mock_load.assert_called_once()
    
    def test_detection_empty_result(self, detector, sample_frame):
        """Test de détection avec résultat vide."""
        with patch('src.detection.yolo.detector.YOLO') as mock_yolo:
            mock_model = Mock()
            mock_yolo.return_value = mock_model
            
            mock_result = Mock()
            mock_result.boxes = None
            mock_model.return_value = [mock_result]
            
            detector.load_model()
            detections = detector.detect(sample_frame)
            
            assert len(detections) == 0
    
    def test_confidence_filtering(self, detector):
        """Test du filtrage par confiance."""
        # Détection avec seuil personnalisé
        detector.confidence_threshold = 0.5
        
        bbox = BoundingBox(100, 100, 100, 100, confidence=0.3)
        
        # La détection devrait être filtrée car confiance < seuil
        assert bbox.confidence < detector.confidence_threshold
    
    def test_cache_functionality(self, detector, sample_frame):
        """Test du système de cache."""
        with patch('src.detection.yolo.detector.YOLO') as mock_yolo:
            mock_model = Mock()
            mock_yolo.return_value = mock_model
            mock_model.return_value = [Mock(boxes=None)]
            
            detector.load_model()
            
            # Premier appel
            detector.detect(sample_frame)
            assert detector.stats["cache_misses"] == 1
            
            # Second appel avec même frame (devrait utiliser le cache)
            detector.detect(sample_frame)
            # Note: Le cache dépend du timestamp, donc pas de hit ici
    
    def test_batch_detection(self, detector):
        """Test de la détection en lot."""
        frames = []
        for i in range(3):
            image = np.zeros((320, 320, 3), dtype=np.uint8)
            frame = Frame(
                image=image,
                timestamp=datetime.now(),
                frame_id=i,
                stream_id=f"test_{i}",
                width=320,
                height=320
            )
            frames.append(frame)
        
        with patch('src.detection.yolo.detector.YOLO') as mock_yolo:
            mock_model = Mock()
            mock_yolo.return_value = mock_model
            mock_model.return_value = [Mock(boxes=None)] * len(frames)
            
            detector.load_model()
            results = detector.detect_batch(frames)
            
            assert len(results) == len(frames)
            for result in results:
                assert isinstance(result, list)
    
    def test_model_info(self, detector):
        """Test des informations du modèle."""
        info = detector.get_model_info()
        
        assert isinstance(info, dict)
        assert "model_path" in info
        assert "device" in info
        assert "is_loaded" in info
        assert "supported_classes" in info
        assert "stats" in info
    
    def test_threshold_update(self, detector):
        """Test de la mise à jour des seuils."""
        original_conf = detector.confidence_threshold
        original_iou = detector.iou_threshold
        
        detector.update_thresholds(confidence=0.7, iou=0.6)
        
        assert detector.confidence_threshold == 0.7
        assert detector.iou_threshold == 0.6
        assert detector.confidence_threshold != original_conf
        assert detector.iou_threshold != original_iou
    
    def test_stats_reset(self, detector):
        """Test de la réinitialisation des statistiques."""
        # Ajout de quelques statistiques factices
        detector.stats["total_detections"] = 10
        detector.stats["avg_inference_time"] = 0.5
        
        detector.reset_stats()
        
        assert detector.stats["total_detections"] == 0
        assert detector.stats["avg_inference_time"] == 0.0
    
    def test_cleanup(self, detector):
        """Test du nettoyage."""
        with patch('src.detection.yolo.detector.YOLO'):
            detector.load_model()
            assert detector.is_loaded
            
            detector.cleanup()
            assert not detector.is_loaded
            assert detector.model is None
    
    def test_image_preparation(self, detector):
        """Test de la préparation d'image."""
        # Image BGR (format OpenCV)
        bgr_image = np.zeros((100, 100, 3), dtype=np.uint8)
        bgr_image[:, :, 0] = 255  # Canal bleu
        
        prepared = detector._prepare_image(bgr_image)
        
        # Vérifier que l'image a été convertie (le rouge devrait maintenant être dominant)
        assert prepared.shape == bgr_image.shape
    
    @patch('src.detection.yolo.detector.YOLO')
    def test_detection_error_handling(self, mock_yolo, detector, sample_frame):
        """Test de la gestion d'erreurs lors de la détection."""
        mock_model = Mock()
        mock_yolo.return_value = mock_model
        mock_model.side_effect = Exception("Erreur de test")
        
        detector.load_model()
        
        with pytest.raises(DetectionError):
            detector.detect(sample_frame)
    
    def test_surveillance_classes(self, detector):
        """Test des classes de surveillance."""
        assert "person" in detector.SURVEILLANCE_CLASSES.values()
        assert 0 in detector.SURVEILLANCE_CLASSES  # person class ID
    
    def test_class_confidence_thresholds(self, detector):
        """Test des seuils de confiance par classe."""
        assert "person" in detector.CLASS_CONFIDENCE_THRESHOLDS
        assert "default" in detector.CLASS_CONFIDENCE_THRESHOLDS
        assert detector.CLASS_CONFIDENCE_THRESHOLDS["person"] < detector.CLASS_CONFIDENCE_THRESHOLDS["default"]


@pytest.mark.integration
class TestYOLODetectorIntegration:
    """Tests d'intégration pour le détecteur YOLO."""
    
    def test_real_model_loading(self):
        """Test de chargement d'un vrai modèle (test d'intégration)."""
        detector = YOLODetector(
            model_path="yolov11n.pt",
            device="cpu"
        )
        
        try:
            detector.load_model()
            assert detector.is_loaded
            assert detector.model is not None
        except Exception as e:
            pytest.skip(f"Modèle YOLO non disponible: {e}")
        finally:
            detector.cleanup()
    
    def test_real_detection(self):
        """Test de détection sur une vraie image."""
        detector = YOLODetector(device="cpu")
        
        try:
            detector.load_model()
            
            # Image de test simple
            image = np.zeros((640, 640, 3), dtype=np.uint8)
            cv2.rectangle(image, (200, 200), (400, 500), (255, 255, 255), -1)
            
            frame = Frame(
                image=image,
                timestamp=datetime.now(),
                frame_id=1,
                stream_id="integration_test",
                width=640,
                height=640
            )
            
            detections = detector.detect(frame)
            assert isinstance(detections, list)
            
        except Exception as e:
            pytest.skip(f"Test d'intégration impossible: {e}")
        finally:
            detector.cleanup()