"""Configuration centrale du système de surveillance."""

import os
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseSettings, Field


class ModelConfig(BaseSettings):
    """Configuration des modèles IA."""
    
    # VLM Configuration
    vlm_model_name: str = "microsoft/kosmos-2-patch14-224"
    vlm_device: str = "cuda" if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu"
    vlm_max_tokens: int = 512
    vlm_temperature: float = 0.1
    
    # YOLO Configuration
    yolo_model_path: str = "yolov8n.pt"
    yolo_confidence: float = 0.25
    yolo_iou_threshold: float = 0.45
    yolo_device: str = "cuda" if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu"
    
    # Tracking Configuration
    tracker_type: str = "bytetrack"  # "bytetrack" ou "deepsort"
    max_disappeared: int = 50
    max_distance: float = 100.0


class SystemConfig(BaseSettings):
    """Configuration système."""
    
    # Performance
    max_concurrent_streams: int = Field(10, description="Nombre max de flux simultanés")
    processing_fps: int = Field(15, description="FPS de traitement")
    buffer_size: int = Field(30, description="Taille buffer vidéo")
    
    # Seuils de détection
    theft_confidence_threshold: float = Field(0.85, description="Seuil confiance vol")
    false_positive_threshold: float = Field(0.03, description="Seuil faux positifs acceptés")
    
    # Cache et stockage
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    cache_ttl: int = 3600  # 1 heure
    
    # Logging
    log_level: str = "INFO"
    log_rotation: str = "1 day"
    log_retention: str = "30 days"


class WebConfig(BaseSettings):
    """Configuration interface web."""
    
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    workers: int = 1
    
    # WebSocket
    websocket_timeout: int = 30
    max_connections: int = 100
    
    # CORS
    cors_origins: List[str] = ["*"]
    cors_methods: List[str] = ["*"]
    cors_headers: List[str] = ["*"]


class PathConfig(BaseSettings):
    """Configuration des chemins."""
    
    # Répertoires principaux
    base_dir: Path = Path(__file__).parent.parent
    data_dir: Path = base_dir / "data"
    models_dir: Path = data_dir / "models"
    logs_dir: Path = base_dir / "logs"
    temp_dir: Path = base_dir / "temp"
    
    # Fichiers de configuration
    tools_config_path: Path = base_dir / "config" / "tools.yaml"
    prompts_dir: Path = base_dir / "config" / "prompts"
    
    def __post_init__(self):
        """Créer les répertoires nécessaires."""
        for directory in [self.data_dir, self.models_dir, self.logs_dir, self.temp_dir]:
            directory.mkdir(parents=True, exist_ok=True)


class Settings(BaseSettings):
    """Configuration globale de l'application."""
    
    model: ModelConfig = ModelConfig()
    system: SystemConfig = SystemConfig()
    web: WebConfig = WebConfig()
    paths: PathConfig = PathConfig()
    
    # Meta information
    app_name: str = "Intelligent Surveillance System"
    app_version: str = "1.0.0"
    debug: bool = False
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Instance globale de configuration
settings = Settings()


def get_settings() -> Settings:
    """Récupérer les paramètres de configuration."""
    return settings


def update_settings(**kwargs) -> None:
    """Mettre à jour les paramètres de configuration."""
    global settings
    for key, value in kwargs.items():
        if hasattr(settings, key):
            setattr(settings, key, value)


# Configuration spécifique pour les tests
class TestSettings(Settings):
    """Configuration pour les tests."""
    
    debug: bool = True
    system: SystemConfig = SystemConfig(
        max_concurrent_streams=2,
        processing_fps=5,
        redis_host="localhost",
        redis_port=6380,
        redis_db=1
    )
    web: WebConfig = WebConfig(
        port=8001,
        reload=False
    )


def get_test_settings() -> TestSettings:
    """Récupérer la configuration de test."""
    return TestSettings()