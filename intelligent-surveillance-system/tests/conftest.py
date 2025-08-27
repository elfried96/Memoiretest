"""
⚙️ Configuration pytest pour Tests Mémoire - Environnement GPU optimisé
========================================================================

Configuration centralisée pour :
- Gestion environnement GPU/CPU
- Fixtures partagées entre tests
- Markers personnalisés pour classification tests
- Collecte métriques performance automatique
- Setup/teardown optimisé pour mémoire
"""

import pytest
import torch
import numpy as np
import cv2
import base64
import time
import json
import asyncio
from typing import Dict, Any, List
from pathlib import Path
import logging
import sys
import os

# Configuration logging pour tests
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('tests_memoire/test_results.log')
    ]
)

logger = logging.getLogger(__name__)

# =================== CONFIGURATION PYTEST ===================

def pytest_configure(config):
    """Configuration globale pytest."""
    # Markers personnalisés
    config.addinivalue_line("markers", "gpu: tests nécessitant GPU")
    config.addinivalue_line("markers", "slow: tests lents (>10s)")
    config.addinivalue_line("markers", "integration: tests d'intégration")
    config.addinivalue_line("markers", "unit: tests unitaires")
    config.addinivalue_line("markers", "performance: tests de performance")
    
    # Configuration environnement
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    
    logger.info("🧪 Configuration tests mémoire initialisée")

def pytest_collection_modifyitems(config, items):
    """Modification de la collection de tests."""
    # Skip des tests GPU si pas disponible
    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="GPU non disponible")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)
    
    # Ordre des tests : unit -> integration -> performance
    def test_priority(item):
        if "unit" in item.keywords:
            return 1
        elif "integration" in item.keywords:
            return 2
        elif "performance" in item.keywords:
            return 3
        else:
            return 4
    
    items.sort(key=test_priority)

# =================== FIXTURES GLOBALES ===================

@pytest.fixture(scope="session")
def gpu_info():
    """Informations GPU pour la session."""
    if torch.cuda.is_available():
        gpu_props = torch.cuda.get_device_properties(0)
        return {
            "available": True,
            "name": gpu_props.name,
            "memory_total": gpu_props.total_memory / (1024**3),  # GB
            "compute_capability": f"{gpu_props.major}.{gpu_props.minor}",
            "device_count": torch.cuda.device_count()
        }
    else:
        return {"available": False}

@pytest.fixture(scope="session")
def test_data_dir():
    """Répertoire des données de test."""
    data_dir = Path("tests_memoire/data")
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir

@pytest.fixture(scope="function")
def cleanup_gpu():
    """Nettoyage GPU avant/après chaque test."""
    # Setup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()
    
    yield
    
    # Teardown
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        final_memory = torch.cuda.memory_allocated()
        memory_leak = final_memory - initial_memory
        
        if memory_leak > 100 * (1024**2):  # 100MB
            logger.warning(f"Possible fuite mémoire GPU: {memory_leak / (1024**2):.1f}MB")

@pytest.fixture
def performance_monitor():
    """Moniteur de performance pour tests."""
    from src.utils.performance import PerformanceMonitor
    
    monitor = PerformanceMonitor(collection_interval=0.5)
    monitor.start_monitoring()
    
    yield monitor
    
    monitor.stop_monitoring()

# =================== FIXTURES DONNÉES TEST ===================

@pytest.fixture
def sample_images():
    """Images test standardisées."""
    images = {}
    
    # Image normale 640x480
    img_normal = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(img_normal, (300, 150), (380, 400), (100, 100, 100), -1)
    cv2.circle(img_normal, (340, 130), 25, (150, 150, 150), -1)
    images["normal"] = img_normal
    
    # Image haute résolution 1080p
    img_hd = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    # Ajout d'objets pour tests
    cv2.rectangle(img_hd, (800, 400), (1000, 800), (120, 120, 120), -1)
    cv2.circle(img_hd, (900, 350), 50, (180, 180, 180), -1)
    images["hd"] = img_hd
    
    # Image multi-objets
    img_multi = np.zeros((480, 640, 3), dtype=np.uint8)
    positions = [(150, 200), (350, 180), (500, 220)]
    for i, (x, y) in enumerate(positions):
        cv2.rectangle(img_multi, (x-30, y), (x+30, y+120), (100+i*50, 100, 100), -1)
        cv2.circle(img_multi, (x, y-30), 20, (150, 150, 150), -1)
    images["multi"] = img_multi
    
    return images

@pytest.fixture  
def sample_images_b64(sample_images):
    """Images test en base64."""
    b64_images = {}
    for name, img in sample_images.items():
        _, buffer = cv2.imencode('.jpg', img)
        b64_images[name] = base64.b64encode(buffer).decode('utf-8')
    return b64_images

@pytest.fixture
def detection_test_data():
    """Données test pour détection."""
    return {
        "person_bbox": (280, 150, 380, 400),  # x1, y1, x2, y2
        "object_classes": ["person", "chair", "table", "book"],
        "confidence_thresholds": [0.3, 0.5, 0.7, 0.9],
        "trajectory_points": [
            (100, 200), (120, 210), (140, 225), (160, 240),
            (180, 250), (200, 255), (220, 265), (240, 280)
        ]
    }

# =================== FIXTURES MODÈLES ===================

@pytest.fixture(scope="session")
async def vlm_model_session():
    """Modèle VLM pour la session (évite rechargements)."""
    from src.core.vlm.dynamic_model import DynamicVisionLanguageModel
    
    vlm = DynamicVisionLanguageModel(
        default_model="kimi-vl-a3b-thinking",
        device="auto",
        enable_fallback=True
    )
    
    # Chargement initial
    success = await vlm.load_model()
    if not success:
        pytest.skip("Impossible de charger le modèle VLM")
    
    yield vlm
    
    # Cleanup
    vlm._unload_current_model()

@pytest.fixture(scope="session")
def yolo_detector_session():
    """Détecteur YOLO pour la session."""
    from src.detection.yolo_detector import YOLODetector
    
    try:
        detector = YOLODetector(
            model_name="yolov8n.pt",  # Modèle léger pour tests
            device="auto",
            confidence_threshold=0.5
        )
        return detector
    except Exception as e:
        pytest.skip(f"Impossible de charger YOLO: {e}")

# =================== UTILITAIRES TEST ===================

@pytest.fixture
def test_utils():
    """Utilitaires pour tests."""
    
    class TestUtils:
        @staticmethod
        def numpy_to_b64(img: np.ndarray) -> str:
            """Conversion numpy vers base64."""
            _, buffer = cv2.imencode('.jpg', img)
            return base64.b64encode(buffer).decode('utf-8')
        
        @staticmethod  
        def b64_to_numpy(img_b64: str) -> np.ndarray:
            """Conversion base64 vers numpy."""
            img_bytes = base64.b64decode(img_b64)
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        @staticmethod
        def measure_inference_time(func, *args, **kwargs):
            """Mesure temps d'inférence."""
            start = time.time()
            result = func(*args, **kwargs)
            return result, time.time() - start
        
        @staticmethod
        async def measure_async_time(func, *args, **kwargs):
            """Mesure temps fonction async."""
            start = time.time()
            result = await func(*args, **kwargs)
            return result, time.time() - start
    
    return TestUtils()

# =================== MÉTRIQUES COLLECTÉES ===================

@pytest.fixture(scope="session")
def metrics_collector():
    """Collecteur de métriques pour le mémoire."""
    
    class MetricsCollector:
        def __init__(self):
            self.metrics = {
                "test_results": [],
                "performance_data": {},
                "gpu_usage": [],
                "model_performance": {},
                "integration_metrics": {}
            }
        
        def add_test_result(self, test_name: str, result: Dict[str, Any]):
            """Ajouter résultat de test."""
            self.metrics["test_results"].append({
                "test_name": test_name,
                "timestamp": time.time(),
                **result
            })
        
        def add_performance_data(self, component: str, data: Dict[str, Any]):
            """Ajouter données performance."""
            if component not in self.metrics["performance_data"]:
                self.metrics["performance_data"][component] = []
            
            self.metrics["performance_data"][component].append({
                "timestamp": time.time(),
                **data
            })
        
        def save_metrics(self, filepath: str = "tests_memoire/metrics_results.json"):
            """Sauvegarder métriques."""
            with open(filepath, 'w') as f:
                json.dump(self.metrics, f, indent=2, default=str)
        
        def get_summary(self) -> Dict[str, Any]:
            """Résumé des métriques."""
            return {
                "total_tests": len(self.metrics["test_results"]),
                "components_tested": list(self.metrics["performance_data"].keys()),
                "gpu_samples": len(self.metrics["gpu_usage"]),
                "test_success_rate": sum(1 for t in self.metrics["test_results"] 
                                       if t.get("success", False)) / max(len(self.metrics["test_results"]), 1)
            }
    
    collector = MetricsCollector()
    yield collector
    
    # Sauvegarde finale
    collector.save_metrics()
    logger.info(f"📊 Métriques sauvegardées: {collector.get_summary()}")

# =================== HOOKS PYTEST ===================

def pytest_runtest_setup(item):
    """Setup avant chaque test."""
    # Log du début de test
    logger.info(f"🧪 Début test: {item.name}")
    
    # Vérification pré-requis GPU si nécessaire
    if "gpu" in item.keywords and not torch.cuda.is_available():
        pytest.skip("Test GPU mais GPU non disponible")

def pytest_runtest_teardown(item, nextitem):
    """Teardown après chaque test."""
    # Nettoyage mémoire
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    logger.info(f"✅ Fin test: {item.name}")

def pytest_sessionfinish(session, exitstatus):
    """Fin de session de tests."""
    logger.info(f"🏁 Session terminée avec status: {exitstatus}")
    
    # Rapport final GPU
    if torch.cuda.is_available():
        memory_used = torch.cuda.memory_allocated() / (1024**2)
        logger.info(f"💾 Mémoire GPU finale: {memory_used:.1f}MB")

# =================== CONFIGURATION ASYNCIO ===================

@pytest.fixture(scope="session")
def event_loop():
    """Event loop pour tests async."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def benchmark_images(sample_images):
    """Images standardisées pour benchmarks."""
    return sample_images