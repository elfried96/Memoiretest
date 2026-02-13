"""
Configuration centrale du système IntelliSurveillance.
Utilise Pydantic Settings pour validation et chargement depuis .env
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """Configuration PostgreSQL."""
    
    model_config = SettingsConfigDict(env_prefix="DB_")
    
    host: str = "localhost"
    port: int = 5432
    user: str = "intelli"
    password: str = "intelli_secure_pwd"
    name: str = "intelli_surveillance"
    
    @property
    def async_url(self) -> str:
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"
    
    @property
    def sync_url(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


class RedisSettings(BaseSettings):
    """Configuration Redis."""
    
    model_config = SettingsConfigDict(env_prefix="REDIS_")
    
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str | None = None
    
    @property
    def url(self) -> str:
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"


class DetectionSettings(BaseSettings):
    """Configuration du pipeline de détection YOLO + ByteTrack."""
    
    model_config = SettingsConfigDict(env_prefix="DETECTION_")
    
    # Modèle YOLO
    yolo_model: str = "yolo11m.pt"
    yolo_confidence: float = 0.25
    yolo_iou_threshold: float = 0.45
    
    # Classes COCO à détecter
    # 0=person, 24=backpack, 26=handbag, 28=suitcase, 39=bottle, 67=cell phone
    target_classes: list[int] = [0, 24, 26, 28, 39, 67]
    
    # Tracking
    tracker_type: Literal["bytetrack", "botsort"] = "bytetrack"
    track_buffer: int = 30
    
    # Performance
    device: str = "cuda:0"
    half_precision: bool = True


class VLMSettings(BaseSettings):
    """Configuration du VLM orchestrateur."""
    
    model_config = SettingsConfigDict(env_prefix="VLM_")
    
    # Modèle
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    quantization: Literal["4bit", "8bit", "none"] = "4bit"
    device_map: str = "auto"
    
    # Inference
    max_new_tokens: int = 1024
    temperature: float = 0.7
    
    # Seuils
    suspicion_threshold: float = 0.6
    min_frames_suspicious: int = 15


class AlertSettings(BaseSettings):
    """Configuration des alertes."""
    
    model_config = SettingsConfigDict(env_prefix="ALERT_")
    
    level_info: float = 0.3
    level_attention: float = 0.6
    level_critical: float = 0.85
    cooldown_seconds: int = 30


class APISettings(BaseSettings):
    """Configuration de l'API FastAPI."""
    
    model_config = SettingsConfigDict(env_prefix="API_")
    
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    
    # CORS
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:8501"]
    
    # Auth
    secret_key: str = "change-this-in-production"
    access_token_expire_minutes: int = 1440  # 24h


class Settings(BaseSettings):
    """Configuration globale."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    # Environnement
    env: Literal["development", "staging", "production"] = "development"
    debug: bool = True
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    
    # Projet
    project_name: str = "IntelliSurveillance"
    version: str = "0.1.0"
    
    # Sous-configurations
    db: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    detection: DetectionSettings = Field(default_factory=DetectionSettings)
    vlm: VLMSettings = Field(default_factory=VLMSettings)
    alert: AlertSettings = Field(default_factory=AlertSettings)
    api: APISettings = Field(default_factory=APISettings)


@lru_cache
def get_settings() -> Settings:
    """Singleton pour la configuration."""
    return Settings()


# Export pour usage direct
settings = get_settings()
