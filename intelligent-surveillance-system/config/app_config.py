"""Configuration centralisée pour le système de surveillance intelligente."""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

# Chemins de base
PROJECT_ROOT = Path(__file__).parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
TESTS_ROOT = PROJECT_ROOT / "tests"
DATA_ROOT = PROJECT_ROOT / "data"
MODELS_ROOT = DATA_ROOT / "models"
LOGS_ROOT = PROJECT_ROOT / "logs"

# Création des dossiers si inexistants
for path in [MODELS_ROOT, LOGS_ROOT]:
    path.mkdir(parents=True, exist_ok=True)


class DeviceType(Enum):
    """Types de devices supportés."""
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


class OrchestrationMode(Enum):
    """Modes d'orchestration."""
    FAST = "fast"
    BALANCED = "balanced"
    THOROUGH = "thorough"


@dataclass
class VLMConfig:
    """Configuration pour les modèles VLM."""
    primary_model: str = "kimi-vl-a3b-thinking"
    fallback_models: list = field(default_factory=list)  # Vide par défaut pour économiser la mémoire
    enable_fallback: bool = False  # Désactivé par défaut
    device: DeviceType = DeviceType.AUTO
    load_in_4bit: bool = True
    max_new_tokens: int = 512
    temperature: float = 0.1
    top_p: float = 0.9


@dataclass
class YOLOConfig:
    """Configuration pour YOLO."""
    model_path: str = "yolov11n.pt"
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    max_detections: int = 1000
    device: DeviceType = DeviceType.AUTO


@dataclass
class TrackerConfig:
    """Configuration pour le tracking."""
    tracker_type: str = "bytetrack"
    track_buffer: int = 30
    match_threshold: float = 0.8
    frame_rate: int = 30


@dataclass
class OrchestrationConfig:
    """Configuration pour l'orchestration."""
    mode: OrchestrationMode = OrchestrationMode.BALANCED
    max_concurrent_tools: int = 4
    timeout_seconds: int = 30
    confidence_threshold: float = 0.7
    enable_advanced_tools: bool = True


@dataclass
class PerformanceConfig:
    """Configuration pour le monitoring de performance."""
    enable_monitoring: bool = True
    collection_interval: float = 1.0
    max_history_size: int = 3600
    log_slow_functions: bool = True
    slow_function_threshold: float = 2.0


@dataclass
class SystemConfig:
    """Configuration système complète."""
    vlm: VLMConfig = field(default_factory=VLMConfig)
    yolo: YOLOConfig = field(default_factory=YOLOConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    orchestration: OrchestrationConfig = field(default_factory=OrchestrationConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    
    # Paramètres généraux
    debug: bool = False
    log_level: str = "INFO"
    save_processed_frames: bool = False
    output_directory: str = str(PROJECT_ROOT / "output")
    
    @classmethod
    def from_env(cls) -> 'SystemConfig':
        """Création de la configuration à partir des variables d'environnement."""
        config = cls()
        
        # VLM
        if os.getenv('VLM_MODEL'):
            config.vlm.primary_model = os.getenv('VLM_MODEL')
        if os.getenv('VLM_DEVICE'):
            config.vlm.device = DeviceType(os.getenv('VLM_DEVICE'))
        if os.getenv('VLM_4BIT') == 'false':
            config.vlm.load_in_4bit = False
            
        # YOLO
        if os.getenv('YOLO_MODEL'):
            config.yolo.model_path = os.getenv('YOLO_MODEL')
        if os.getenv('YOLO_CONFIDENCE'):
            config.yolo.confidence_threshold = float(os.getenv('YOLO_CONFIDENCE'))
            
        # Orchestration
        if os.getenv('ORCHESTRATION_MODE'):
            config.orchestration.mode = OrchestrationMode(os.getenv('ORCHESTRATION_MODE'))
        if os.getenv('MAX_CONCURRENT_TOOLS'):
            config.orchestration.max_concurrent_tools = int(os.getenv('MAX_CONCURRENT_TOOLS'))
            
        # Performance
        if os.getenv('DISABLE_MONITORING') == 'true':
            config.performance.enable_monitoring = False
            
        # Général
        if os.getenv('DEBUG') == 'true':
            config.debug = True
            config.log_level = "DEBUG"
        if os.getenv('LOG_LEVEL'):
            config.log_level = os.getenv('LOG_LEVEL')
            
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Conversion en dictionnaire."""
        return {
            'vlm': {
                'primary_model': self.vlm.primary_model,
                'fallback_models': self.vlm.fallback_models,
                'enable_fallback': self.vlm.enable_fallback,
                'device': self.vlm.device.value,
                'load_in_4bit': self.vlm.load_in_4bit,
                'max_new_tokens': self.vlm.max_new_tokens,
                'temperature': self.vlm.temperature,
                'top_p': self.vlm.top_p
            },
            'yolo': {
                'model_path': self.yolo.model_path,
                'confidence_threshold': self.yolo.confidence_threshold,
                'iou_threshold': self.yolo.iou_threshold,
                'max_detections': self.yolo.max_detections,
                'device': self.yolo.device.value
            },
            'tracker': {
                'tracker_type': self.tracker.tracker_type,
                'track_buffer': self.tracker.track_buffer,
                'match_threshold': self.tracker.match_threshold,
                'frame_rate': self.tracker.frame_rate
            },
            'orchestration': {
                'mode': self.orchestration.mode.value,
                'max_concurrent_tools': self.orchestration.max_concurrent_tools,
                'timeout_seconds': self.orchestration.timeout_seconds,
                'confidence_threshold': self.orchestration.confidence_threshold,
                'enable_advanced_tools': self.orchestration.enable_advanced_tools
            },
            'performance': {
                'enable_monitoring': self.performance.enable_monitoring,
                'collection_interval': self.performance.collection_interval,
                'max_history_size': self.performance.max_history_size,
                'log_slow_functions': self.performance.log_slow_functions,
                'slow_function_threshold': self.performance.slow_function_threshold
            },
            'general': {
                'debug': self.debug,
                'log_level': self.log_level,
                'save_processed_frames': self.save_processed_frames,
                'output_directory': self.output_directory
            }
        }


# Instance globale de configuration
_global_config: Optional[SystemConfig] = None


def get_config() -> SystemConfig:
    """Récupération de la configuration globale."""
    global _global_config
    if _global_config is None:
        _global_config = SystemConfig.from_env()
    return _global_config


def set_config(config: SystemConfig) -> None:
    """Définition de la configuration globale."""
    global _global_config
    _global_config = config


def reset_config() -> None:
    """Réinitialisation de la configuration."""
    global _global_config
    _global_config = None


# Configurations prédéfinies
CONFIGS = {
    "development": SystemConfig(
        vlm=VLMConfig(
            primary_model="kimi-vl-a3b-thinking",  # Kimi-VL uniquement
            fallback_models=[],
            enable_fallback=False,
            load_in_4bit=True
        ),
        orchestration=OrchestrationConfig(
            mode=OrchestrationMode.FAST,
            max_concurrent_tools=2
        ),
        performance=PerformanceConfig(
            enable_monitoring=True,
            log_slow_functions=True
        ),
        debug=True,
        log_level="DEBUG"
    ),
    
    "production": SystemConfig(
        vlm=VLMConfig(
            primary_model="kimi-vl-a3b-thinking",
            fallback_models=[],
            enable_fallback=False,
            load_in_4bit=False  # Meilleure qualité
        ),
        orchestration=OrchestrationConfig(
            mode=OrchestrationMode.BALANCED,
            max_concurrent_tools=6
        ),
        performance=PerformanceConfig(
            enable_monitoring=True,
            collection_interval=0.5
        ),
        debug=False,
        log_level="INFO"
    ),
    
    "testing": SystemConfig(
        vlm=VLMConfig(
            primary_model="microsoft/git-base-coco",  # Modèle léger pour tests
            fallback_models=[],  # Pas de fallback pour économiser mémoire
            enable_fallback=False,
            load_in_4bit=True
        ),
        orchestration=OrchestrationConfig(
            mode=OrchestrationMode.FAST,
            max_concurrent_tools=1,
            timeout_seconds=10
        ),
        performance=PerformanceConfig(
            enable_monitoring=False
        ),
        debug=True,
        log_level="DEBUG"
    ),
    
    "testing_kimi": SystemConfig(
        vlm=VLMConfig(
            primary_model="kimi-vl-a3b-thinking",  # Pour tests avec Kimi-VL
            fallback_models=[],
            enable_fallback=False,
            load_in_4bit=True
        ),
        orchestration=OrchestrationConfig(
            mode=OrchestrationMode.FAST,
            max_concurrent_tools=1,
            timeout_seconds=15
        ),
        performance=PerformanceConfig(
            enable_monitoring=False
        ),
        debug=True,
        log_level="DEBUG"
    ),
    
    "testing_qwen": SystemConfig(
        vlm=VLMConfig(
            primary_model="qwen2-vl-7b-instruct",  # Pour tests avec Qwen uniquement
            fallback_models=[],
            enable_fallback=False,
            load_in_4bit=True
        ),
        orchestration=OrchestrationConfig(
            mode=OrchestrationMode.FAST,
            max_concurrent_tools=1,
            timeout_seconds=15
        ),
        performance=PerformanceConfig(
            enable_monitoring=False
        ),
        debug=True,
        log_level="DEBUG"
    )
}


def load_config(profile: str = "development") -> SystemConfig:
    """Chargement d'un profil de configuration."""
    if profile not in CONFIGS:
        raise ValueError(f"Profil '{profile}' non trouvé. Disponibles: {list(CONFIGS.keys())}")
    
    config = CONFIGS[profile]
    set_config(config)
    return config