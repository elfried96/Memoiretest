"""
Gestionnaire de configuration centralisé.
"""

import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path

from .base_config import BaseConfig, VLMAnalysisMode, LogLevel

logger = logging.getLogger(__name__)


class ConfigManager:
    """Gestionnaire centralisé de configuration."""
    
    _instance: Optional['ConfigManager'] = None
    _config: Optional[BaseConfig] = None
    
    def __new__(cls) -> 'ConfigManager':
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialisation du gestionnaire."""
        if self._config is None:
            self._config = self._load_default_config()
    
    @property
    def config(self) -> BaseConfig:
        """Accès à la configuration actuelle."""
        if self._config is None:
            self._config = self._load_default_config()
        return self._config
    
    def _load_default_config(self) -> BaseConfig:
        """Charge la configuration par défaut."""
        config = BaseConfig()
        
        # Override avec variables d'environnement
        self._apply_env_overrides(config)
        
        # Validation
        config.validate()
        
        logger.info("✅ Configuration par défaut chargée")
        return config
    
    def _apply_env_overrides(self, config: BaseConfig) -> None:
        """Applique les overrides des variables d'environnement."""
        
        # Logging
        if env_log_level := os.getenv("SURVEILLANCE_LOG_LEVEL"):
            try:
                config.log_level = LogLevel(env_log_level.upper())
            except ValueError:
                logger.warning(f"⚠️ Niveau de log invalide: {env_log_level}")
        
        # Vidéo
        if env_video_source := os.getenv("SURVEILLANCE_VIDEO_SOURCE"):
            config.video.source = env_video_source
        
        if env_frame_skip := os.getenv("SURVEILLANCE_FRAME_SKIP"):
            try:
                config.video.frame_skip = int(env_frame_skip)
            except ValueError:
                logger.warning(f"⚠️ frame_skip invalide: {env_frame_skip}")
        
        if env_max_frames := os.getenv("SURVEILLANCE_MAX_FRAMES"):
            try:
                config.video.max_frames = int(env_max_frames)
            except ValueError:
                logger.warning(f"⚠️ max_frames invalide: {env_max_frames}")
        
        # VLM
        if env_primary_model := os.getenv("SURVEILLANCE_PRIMARY_MODEL"):
            config.vlm.primary_model = env_primary_model
        
        if env_fallback_model := os.getenv("SURVEILLANCE_FALLBACK_MODEL"):
            config.vlm.fallback_model = env_fallback_model
        
        if env_vlm_mode := os.getenv("SURVEILLANCE_VLM_MODE"):
            try:
                config.vlm.analysis_mode = VLMAnalysisMode(env_vlm_mode.lower())
            except ValueError:
                logger.warning(f"⚠️ Mode VLM invalide: {env_vlm_mode}")
        
        # Détection
        if env_confidence := os.getenv("SURVEILLANCE_CONFIDENCE"):
            try:
                config.detection.confidence_threshold = float(env_confidence)
            except ValueError:
                logger.warning(f"⚠️ Seuil de confiance invalide: {env_confidence}")
        
        # Output
        if env_output_dir := os.getenv("SURVEILLANCE_OUTPUT_DIR"):
            config.output.base_dir = env_output_dir
        
        if env_save_frames := os.getenv("SURVEILLANCE_SAVE_FRAMES"):
            config.output.save_frames = env_save_frames.lower() in ['true', '1', 'yes', 'on']
        
        if env_save_results := os.getenv("SURVEILLANCE_SAVE_RESULTS"):
            config.output.save_results = env_save_results.lower() in ['true', '1', 'yes', 'on']
        
        # Performance
        if env_gpu_memory := os.getenv("SURVEILLANCE_GPU_MEMORY_FRACTION"):
            try:
                config.performance.gpu_memory_fraction = float(env_gpu_memory)
            except ValueError:
                logger.warning(f"⚠️ Fraction mémoire GPU invalide: {env_gpu_memory}")
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """Met à jour la configuration avec des valeurs spécifiques."""
        try:
            # Mise à jour récursive
            self._update_nested_dict(self.config.__dict__, updates)
            
            # Revalidation
            self.config.validate()
            
            logger.info("✅ Configuration mise à jour")
            
        except Exception as e:
            logger.error(f"❌ Erreur mise à jour configuration: {e}")
            raise
    
    def _update_nested_dict(self, base_dict: Dict[str, Any], updates: Dict[str, Any]) -> None:
        """Mise à jour récursive de dictionnaire."""
        for key, value in updates.items():
            if key in base_dict:
                if isinstance(value, dict) and hasattr(base_dict[key], '__dict__'):
                    # Mise à jour d'un objet nested
                    self._update_nested_dict(base_dict[key].__dict__, value)
                else:
                    # Mise à jour directe
                    setattr(base_dict, key, value) if hasattr(base_dict, key) else None
            else:
                logger.warning(f"⚠️ Clé de configuration inconnue: {key}")
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Récupère la configuration pour un modèle spécifique."""
        base_params = self.config.vlm.get_model_params(model_name)
        
        # Configuration commune
        common_config = {
            "timeout": self.config.vlm.timeout,
            "max_retries": self.config.vlm.max_retries,
            "batch_size": self.config.vlm.batch_size
        }
        
        return {**base_params, **common_config}
    
    def get_detection_config(self) -> Dict[str, Any]:
        """Récupère la configuration de détection."""
        return {
            "model_path": self.config.detection.yolo_model,
            "confidence_threshold": self.config.detection.confidence_threshold,
            "nms_threshold": self.config.detection.nms_threshold,
            "max_detections": self.config.detection.max_detections
        }
    
    def get_tracker_config(self) -> Dict[str, Any]:
        """Récupère la configuration du tracker."""
        return {
            "frame_rate": self.config.detection.frame_rate,
            "track_thresh": self.config.detection.track_thresh,
            "track_buffer": self.config.detection.track_buffer,
            "match_thresh": self.config.detection.match_thresh
        }
    
    def setup_logging(self) -> None:
        """Configure le système de logging basé sur la configuration."""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level.value),
            format=self.config.log_format,
            datefmt=self.config.log_date_format,
            force=True  # Override configuration existante
        )
        
        logger.info(f"📝 Logging configuré: niveau {self.config.log_level.value}")
    
    def create_output_directories(self) -> None:
        """Crée les répertoires de sortie nécessaires."""
        try:
            # Répertoire principal
            self.config.output.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Répertoire des frames si nécessaire
            if self.config.output.save_frames:
                self.config.output.frames_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"📁 Répertoires créés: {self.config.output.output_dir}")
            
        except Exception as e:
            logger.error(f"❌ Erreur création répertoires: {e}")
            raise
    
    def get_summary(self) -> Dict[str, Any]:
        """Retourne un résumé de la configuration actuelle."""
        return {
            "version": self.config.version,
            "video_source": self.config.video.source,
            "primary_model": self.config.vlm.primary_model,
            "vlm_mode": self.config.vlm.analysis_mode.value,
            "output_dir": str(self.config.output.output_dir),
            "log_level": self.config.log_level.value,
            "performance_monitoring": self.config.performance.performance_monitoring
        }


# Instance globale pour faciliter l'accès
config_manager = ConfigManager()