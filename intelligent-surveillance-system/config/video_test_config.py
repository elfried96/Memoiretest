"""Configuration spécifique pour les tests vidéo avec architecture complète."""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any
from .app_config import SystemConfig, VLMConfig, OrchestrationConfig, OrchestrationMode, DeviceType

# Chemins pour les tests vidéo
PROJECT_ROOT = Path(__file__).parent.parent
VIDEO_TEST_ROOT = PROJECT_ROOT / "data" / "video_tests"
VIDEO_OUTPUTS_ROOT = PROJECT_ROOT / "data" / "video_outputs"
VIDEO_DATASETS_ROOT = PROJECT_ROOT / "data" / "video_datasets"

# Création des dossiers
for path in [VIDEO_TEST_ROOT, VIDEO_OUTPUTS_ROOT, VIDEO_DATASETS_ROOT]:
    path.mkdir(parents=True, exist_ok=True)


@dataclass
class VideoTestConfig:
    """Configuration pour les tests vidéo."""
    
    # Paramètres vidéo
    max_frames: int = 300  # 10s à 30fps
    frame_skip: int = 1    # Traiter chaque frame (1) ou une sur N
    resize_width: int = 640
    resize_height: int = 480
    
    # Paramètres de traitement
    save_processed_frames: bool = True
    save_analysis_results: bool = True
    save_video_output: bool = True
    
    # Paramètres d'analyse
    confidence_threshold: float = 0.3
    detection_threshold: float = 0.5
    tracking_enabled: bool = True
    
    # Dossiers de sortie
    output_dir: Path = VIDEO_OUTPUTS_ROOT
    frames_dir: Path = VIDEO_OUTPUTS_ROOT / "frames"
    results_dir: Path = VIDEO_OUTPUTS_ROOT / "results"
    videos_dir: Path = VIDEO_OUTPUTS_ROOT / "processed_videos"


# Configuration système optimisée pour tests vidéo
VIDEO_TEST_SYSTEM_CONFIG = SystemConfig(
    vlm=VLMConfig(
        primary_model="kimi-vl-a3b-thinking",
        fallback_models=[],  # Pas de fallback
        enable_fallback=False,
        device=DeviceType.AUTO,
        load_in_4bit=True,  # Économie mémoire
        max_new_tokens=256,  # Plus court pour vidéo
        temperature=0.1
    ),
    orchestration=OrchestrationConfig(
        mode=OrchestrationMode.BALANCED,  # Bon compromis vitesse/qualité
        max_concurrent_tools=3,  # Limité pour éviter surcharge
        timeout_seconds=15,  # Plus court pour vidéo
        confidence_threshold=0.6,
        enable_advanced_tools=True
    ),
    performance=PerformanceConfig(
        enable_monitoring=True,
        collection_interval=2.0,  # Moins fréquent
        log_slow_functions=True,
        slow_function_threshold=5.0
    ),
    debug=True,
    log_level="INFO",
    save_processed_frames=True,
    output_directory=str(VIDEO_OUTPUTS_ROOT)
)


# Configurations prédéfinies pour différents types de tests vidéo
VIDEO_TEST_CONFIGS = {
    "fast": VideoTestConfig(
        max_frames=150,  # 5s
        frame_skip=2,    # Une frame sur deux
        resize_width=480,
        resize_height=360
    ),
    
    "standard": VideoTestConfig(
        max_frames=300,  # 10s
        frame_skip=1,
        resize_width=640,
        resize_height=480
    ),
    
    "thorough": VideoTestConfig(
        max_frames=600,  # 20s
        frame_skip=1,
        resize_width=720,
        resize_height=576,
        save_processed_frames=True,
        save_analysis_results=True
    ),
    
    "demo": VideoTestConfig(
        max_frames=90,   # 3s
        frame_skip=3,    # Une frame sur trois
        resize_width=320,
        resize_height=240,
        save_processed_frames=False
    )
}


def get_video_test_config(profile: str = "standard") -> VideoTestConfig:
    """Récupération d'une configuration de test vidéo."""
    if profile not in VIDEO_TEST_CONFIGS:
        raise ValueError(f"Profil '{profile}' non trouvé. Disponibles: {list(VIDEO_TEST_CONFIGS.keys())}")
    return VIDEO_TEST_CONFIGS[profile]


def get_system_config_for_video_test() -> SystemConfig:
    """Configuration système optimisée pour les tests vidéo."""
    return VIDEO_TEST_SYSTEM_CONFIG


# Datasets de test recommandés
RECOMMENDED_VIDEO_DATASETS = {
    "surveillance_basic": {
        "description": "Vidéos de surveillance basiques",
        "files": ["person_walking.mp4", "object_detection.mp4"],
        "duration": "30s each"
    },
    
    "theft_scenarios": {
        "description": "Scénarios de vol simulés",
        "files": ["shoplifting_simulation.mp4", "bag_snatching.mp4"],
        "duration": "45s each"
    },
    
    "crowded_scenes": {
        "description": "Scènes avec plusieurs personnes",
        "files": ["mall_crowd.mp4", "street_busy.mp4"],
        "duration": "60s each"
    },
    
    "edge_cases": {
        "description": "Cas limites (faible éclairage, occlusions)",
        "files": ["low_light.mp4", "occlusion_test.mp4"],
        "duration": "30s each"
    }
}