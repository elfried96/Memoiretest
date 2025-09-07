"""
Module de configuration unifié pour le système de surveillance.
"""

from .base_config import (
    BaseConfig,
    VideoConfig,
    VLMConfig,
    DetectionConfig,
    AlertConfig,
    OutputConfig,
    PerformanceConfig,
    LogLevel,
    VLMAnalysisMode
)
from .config_manager import ConfigManager
from .config_loader import load_config, save_config

__all__ = [
    "BaseConfig",
    "VideoConfig", 
    "VLMConfig",
    "DetectionConfig",
    "AlertConfig",
    "OutputConfig",
    "PerformanceConfig",
    "LogLevel",
    "VLMAnalysisMode",
    "ConfigManager",
    "load_config",
    "save_config"
]