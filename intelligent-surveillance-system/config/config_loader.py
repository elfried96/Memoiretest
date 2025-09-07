"""
Utilitaires de chargement et sauvegarde de configuration.
"""

import json
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union

from .base_config import BaseConfig

logger = logging.getLogger(__name__)


def load_config(config_path: Union[str, Path]) -> BaseConfig:
    """
    Charge une configuration depuis un fichier.
    
    Args:
        config_path: Chemin vers le fichier de configuration (JSON ou YAML)
        
    Returns:
        Configuration chargée
        
    Raises:
        FileNotFoundError: Si le fichier n'existe pas
        ValueError: Si le format est invalide
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Fichier de configuration non trouvé: {config_path}")
    
    try:
        # Détection du format
        if config_path.suffix.lower() in ['.yml', '.yaml']:
            data = _load_yaml(config_path)
        elif config_path.suffix.lower() == '.json':
            data = _load_json(config_path)
        else:
            # Tentative JSON par défaut
            data = _load_json(config_path)
        
        # Création de la configuration
        config = _dict_to_config(data)
        
        logger.info(f"✅ Configuration chargée: {config_path}")
        return config
        
    except Exception as e:
        logger.error(f"❌ Erreur chargement configuration: {e}")
        raise


def save_config(config: BaseConfig, config_path: Union[str, Path], format: str = "json") -> None:
    """
    Sauvegarde une configuration dans un fichier.
    
    Args:
        config: Configuration à sauvegarder
        config_path: Chemin de sauvegarde
        format: Format de sortie ('json' ou 'yaml')
        
    Raises:
        ValueError: Si le format est invalide
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        data = config.to_dict()
        
        if format.lower() == "json":
            _save_json(data, config_path)
        elif format.lower() in ["yaml", "yml"]:
            _save_yaml(data, config_path)
        else:
            raise ValueError(f"Format non supporté: {format}")
        
        logger.info(f"✅ Configuration sauvée: {config_path}")
        
    except Exception as e:
        logger.error(f"❌ Erreur sauvegarde configuration: {e}")
        raise


def _load_json(path: Path) -> Dict[str, Any]:
    """Charge un fichier JSON."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Charge un fichier YAML."""
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML requis pour charger les fichiers YAML")
    
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def _save_json(data: Dict[str, Any], path: Path) -> None:
    """Sauvegarde en JSON."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _save_yaml(data: Dict[str, Any], path: Path) -> None:
    """Sauvegarde en YAML."""
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML requis pour sauvegarder en YAML")
    
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


def _dict_to_config(data: Dict[str, Any]) -> BaseConfig:
    """Convertit un dictionnaire en configuration."""
    # Pour l'instant, création d'une config par défaut
    # puis mise à jour avec les données
    config = BaseConfig()
    
    # Mise à jour basique - à améliorer pour la conversion complète
    if "video" in data:
        video_data = data["video"]
        if "source" in video_data:
            config.video.source = video_data["source"]
        if "frame_skip" in video_data:
            config.video.frame_skip = video_data["frame_skip"]
        if "max_frames" in video_data:
            config.video.max_frames = video_data["max_frames"]
    
    if "vlm" in data:
        vlm_data = data["vlm"]
        if "primary_model" in vlm_data:
            config.vlm.primary_model = vlm_data["primary_model"]
        if "fallback_model" in vlm_data:
            config.vlm.fallback_model = vlm_data["fallback_model"]
    
    if "detection" in data:
        det_data = data["detection"]
        if "confidence_threshold" in det_data:
            config.detection.confidence_threshold = det_data["confidence_threshold"]
    
    if "output" in data:
        out_data = data["output"]
        if "base_dir" in out_data:
            config.output.base_dir = out_data["base_dir"]
        if "save_results" in out_data:
            config.output.save_results = out_data["save_results"]
        if "save_frames" in out_data:
            config.output.save_frames = out_data["save_frames"]
    
    # Validation
    config.validate()
    
    return config


def create_default_config_files() -> None:
    """Crée des fichiers de configuration par défaut."""
    config_dir = Path("config/presets")
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # Configuration par défaut
    default_config = BaseConfig()
    save_config(default_config, config_dir / "default.json", "json")
    save_config(default_config, config_dir / "default.yml", "yaml")
    
    # Configuration performance
    perf_config = BaseConfig()
    perf_config.vlm.analysis_mode = VLMAnalysisMode.PERIODIC
    perf_config.video.frame_skip = 3
    perf_config.detection.confidence_threshold = 0.6
    save_config(perf_config, config_dir / "performance.json", "json")
    
    # Configuration qualité
    quality_config = BaseConfig()
    quality_config.vlm.analysis_mode = VLMAnalysisMode.CONTINUOUS
    quality_config.video.frame_skip = 1
    quality_config.detection.confidence_threshold = 0.3
    quality_config.output.save_frames = True
    save_config(quality_config, config_dir / "quality.json", "json")
    
    logger.info(f"✅ Fichiers de configuration créés dans: {config_dir}")


if __name__ == "__main__":
    # Création des fichiers par défaut
    from .base_config import VLMAnalysisMode
    create_default_config_files()