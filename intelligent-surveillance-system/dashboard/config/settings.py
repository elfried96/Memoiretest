"""Configuration centralisée du dashboard Streamlit."""

from dataclasses import dataclass
from typing import Dict, List, Optional
from pathlib import Path
import os

@dataclass
class DashboardConfig:
    """Configuration principale du dashboard."""
    
    # Interface
    page_title: str = " Surveillance Intelligente"
    layout: str = "wide"
    initial_sidebar_state: str = "expanded"
    
    # Caméras
    max_cameras: int = 9
    default_camera_grid: tuple = (3, 3)
    camera_resolution: tuple = (640, 480)
    default_fps: int = 30
    
    # Alertes
    alert_threshold_default: int = 70
    alert_cooldown_seconds: int = 5
    max_alerts_display: int = 10
    
    # VLM
    max_chat_history: int = 50
    vlm_response_timeout: int = 30
    
    # Fichiers
    upload_max_size: int = 200  # MB
    supported_video_formats: List[str] = None
    export_formats: List[str] = None
    
    # Performance
    cache_ttl: int = 300  # 5 minutes
    auto_refresh_interval: int = 2  # secondes
    
    def __post_init__(self):
        if self.supported_video_formats is None:
            self.supported_video_formats = ['mp4', 'avi', 'mov', 'mkv', 'webm']
        
        if self.export_formats is None:
            self.export_formats = ['json', 'csv', 'pdf']

@dataclass
class AudioConfig:
    """Configuration des alertes audio."""
    
    enabled: bool = True
    volume: float = 0.8
    sounds_dir: Path = Path(__file__).parent.parent / "assets" / "alert_sounds"
    
    # Types d'alertes
    alert_sounds: Dict[str, str] = None
    
    def __post_init__(self):
        if self.alert_sounds is None:
            self.alert_sounds = {
                "LOW": "beep_low.wav",
                "MEDIUM": "alert_medium.wav", 
                "HIGH": "alarm_high.wav",
                "CRITICAL": "emergency.wav"
            }

@dataclass
class SecurityConfig:
    """Configuration sécurité."""
    
    enable_auth: bool = False
    session_timeout: int = 3600  # 1 heure
    max_concurrent_sessions: int = 5
    log_user_actions: bool = True

# Instance globale
dashboard_config = DashboardConfig()
audio_config = AudioConfig()
security_config = SecurityConfig()

def get_dashboard_config() -> DashboardConfig:
    """Récupère la configuration dashboard."""
    return dashboard_config

def get_audio_config() -> AudioConfig:
    """Récupère la configuration audio."""
    return audio_config

def get_security_config() -> SecurityConfig:
    """Récupère la configuration sécurité."""
    return security_config

def load_config_from_env():
    """Charge la configuration depuis les variables d'environnement."""
    
    # Dashboard
    dashboard_config.max_cameras = int(os.getenv('MAX_CAMERAS', dashboard_config.max_cameras))
    dashboard_config.alert_threshold_default = int(os.getenv('ALERT_THRESHOLD', dashboard_config.alert_threshold_default))
    
    # Audio
    audio_config.enabled = os.getenv('AUDIO_ENABLED', 'true').lower() == 'true'
    audio_config.volume = float(os.getenv('AUDIO_VOLUME', audio_config.volume))
    
    # Sécurité
    security_config.enable_auth = os.getenv('ENABLE_AUTH', 'false').lower() == 'true'
    security_config.session_timeout = int(os.getenv('SESSION_TIMEOUT', security_config.session_timeout))

# Chargement auto des variables d'environnement
load_config_from_env()