"""
Configuration de base pour le système de surveillance.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, List
from pathlib import Path


class LogLevel(Enum):
    """Niveaux de logging."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class VLMAnalysisMode(Enum):
    """Modes d'analyse VLM."""
    CONTINUOUS = "continuous"  # Chaque frame
    PERIODIC = "periodic"      # Périodique (ex: toutes les 10 frames)
    SMART = "smart"            # Uniquement sur alertes
    DISABLED = "disabled"      # Désactivé


@dataclass
class VideoConfig:
    """Configuration vidéo."""
    source: str = "0"  # Webcam par défaut
    frame_skip: int = 1
    max_frames: Optional[int] = None
    target_fps: Optional[int] = None
    buffer_size: int = 1
    
    def validate(self) -> None:
        """Valide la configuration vidéo."""
        if self.frame_skip < 1:
            raise ValueError("frame_skip doit être >= 1")
        if self.max_frames is not None and self.max_frames <= 0:
            raise ValueError("max_frames doit être > 0")


@dataclass
class VLMConfig:
    """Configuration des modèles VLM."""
    primary_model: str = "kimi-vl-a3b-thinking"
    fallback_model: str = "qwen2-vl-7b-instruct"
    analysis_mode: VLMAnalysisMode = VLMAnalysisMode.SMART
    max_retries: int = 3
    timeout: float = 30.0
    batch_size: int = 1
    
    # Paramètres spécifiques par modèle
    model_params: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "kimi-vl-a3b-thinking": {
            "max_new_tokens": 768,
            "temperature": 0.8,
            "do_sample": True
        },
        "qwen2-vl-7b-instruct": {
            "max_new_tokens": 512,
            "temperature": 0.1,
            "do_sample": True
        }
    })
    
    def get_model_params(self, model_name: str) -> Dict[str, Any]:
        """Récupère les paramètres pour un modèle donné."""
        return self.model_params.get(model_name, {})


@dataclass
class DetectionConfig:
    """Configuration de la détection."""
    yolo_model: str = "yolo11n.pt"
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.45
    max_detections: int = 100
    
    # Tracking
    track_thresh: float = 0.5
    track_buffer: int = 30
    match_thresh: float = 0.8
    frame_rate: int = 30
    
    def validate(self) -> None:
        """Valide la configuration de détection."""
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError("confidence_threshold doit être entre 0 et 1")
        if not 0 <= self.nms_threshold <= 1:
            raise ValueError("nms_threshold doit être entre 0 et 1")


@dataclass
class AlertConfig:
    """Configuration des alertes."""
    # Seuils de personnes pour les niveaux d'alerte
    attention_threshold: int = 1
    alerte_threshold: int = 2
    critique_threshold: int = 4
    
    # Actions par niveau
    actions_map: Dict[str, List[str]] = field(default_factory=lambda: {
        "attention": ["Surveillance accrue"],
        "alerte": ["Notification sécurité", "Enregistrement activé"],
        "critique": ["Alerte immédiate", "Enregistrement prioritaire", "Notification urgente"]
    })
    
    # Notifications
    enable_notifications: bool = True
    notification_cooldown: int = 30  # secondes


@dataclass
class OutputConfig:
    """Configuration des sorties."""
    base_dir: str = "surveillance_output"
    save_results: bool = True
    save_frames: bool = False
    save_critical_frames_only: bool = True
    
    # Formats de sortie
    json_indent: int = 2
    image_format: str = "jpg"
    image_quality: int = 90
    
    # Nettoyage automatique
    max_files: int = 1000
    max_age_days: int = 30
    
    @property
    def output_dir(self) -> Path:
        """Répertoire de sortie."""
        return Path(self.base_dir)
    
    @property
    def frames_dir(self) -> Path:
        """Répertoire des frames."""
        return self.output_dir / "frames"


@dataclass
class PerformanceConfig:
    """Configuration de performance."""
    # GPU
    gpu_memory_fraction: float = 0.8
    allow_growth: bool = True
    
    # Threading
    max_workers: int = 4
    
    # Cache
    enable_cache: bool = True
    cache_size: int = 100
    
    # Monitoring
    performance_monitoring: bool = True
    log_performance_every: int = 30  # frames


@dataclass
class BaseConfig:
    """Configuration de base du système."""
    
    # Logging
    log_level: LogLevel = LogLevel.INFO
    log_format: str = "%(asctime)s | %(levelname)s | %(message)s"
    log_date_format: str = "%H:%M:%S"
    
    # Configurations spécialisées
    video: VideoConfig = field(default_factory=VideoConfig)
    vlm: VLMConfig = field(default_factory=VLMConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    
    # Métadonnées
    version: str = "2.0.0"
    description: str = "Système de surveillance intelligent multimodale"
    
    def validate(self) -> None:
        """Valide toute la configuration."""
        self.video.validate()
        self.detection.validate()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit la configuration en dictionnaire."""
        return {
            "log_level": self.log_level.value,
            "log_format": self.log_format,
            "log_date_format": self.log_date_format,
            "video": {
                "source": self.video.source,
                "frame_skip": self.video.frame_skip,
                "max_frames": self.video.max_frames,
                "target_fps": self.video.target_fps
            },
            "vlm": {
                "primary_model": self.vlm.primary_model,
                "fallback_model": self.vlm.fallback_model,
                "analysis_mode": self.vlm.analysis_mode.value,
                "max_retries": self.vlm.max_retries,
                "timeout": self.vlm.timeout
            },
            "detection": {
                "yolo_model": self.detection.yolo_model,
                "confidence_threshold": self.detection.confidence_threshold,
                "nms_threshold": self.detection.nms_threshold
            },
            "alerts": {
                "attention_threshold": self.alerts.attention_threshold,
                "alerte_threshold": self.alerts.alerte_threshold,
                "critique_threshold": self.alerts.critique_threshold
            },
            "output": {
                "base_dir": self.output.base_dir,
                "save_results": self.output.save_results,
                "save_frames": self.output.save_frames
            },
            "performance": {
                "gpu_memory_fraction": self.performance.gpu_memory_fraction,
                "max_workers": self.performance.max_workers,
                "performance_monitoring": self.performance.performance_monitoring
            },
            "version": self.version,
            "description": self.description
        }