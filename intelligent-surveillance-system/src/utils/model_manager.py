"""
🚀 Gestionnaire de Modèles Automatique
=====================================

Gère le téléchargement et l'initialisation automatique de tous les modèles
sauf les VLM qui sont chargés seulement à la demande.
"""

import os
import sys
import requests
import torch
from pathlib import Path
from typing import Dict, List, Optional, Callable
from loguru import logger
import hashlib
import json
from tqdm import tqdm

class ModelManager:
    """Gestionnaire automatique des modèles IA"""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True, parents=True)
        self.models_config = self._load_models_config()
        self.downloaded_models = {}
        
    def _load_models_config(self) -> Dict:
        """Configuration des modèles à télécharger automatiquement"""
        return {
            "yolo": {
                "name": "YOLOv11 Nano",
                "filename": "yolo11n.pt",
                "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
                "size_mb": 5.4,
                "sha256": None,  # Optionnel pour vérification
                "auto_download": True,
                "required_for": ["object_detection", "surveillance"]
            },
            "yolo_medium": {
                "name": "YOLOv11 Medium",
                "filename": "yolo11m.pt", 
                "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt",
                "size_mb": 49.7,
                "auto_download": False,  # Plus lourd, pas automatique
                "required_for": ["high_accuracy_detection"]
            },
            "sam2_base": {
                "name": "SAM2 Base",
                "filename": "sam2_base.pt",
                "url": None,  # Via transformers hub
                "huggingface_id": "facebook/sam2-hiera-base-plus",
                "auto_download": True,
                "required_for": ["segmentation", "advanced_tools"]
            },
            "dino_v2": {
                "name": "DINO v2 Features", 
                "filename": "dinov2_vitb14.pth",
                "huggingface_id": "facebook/dinov2-base",
                "auto_download": True,
                "required_for": ["feature_extraction", "advanced_tools"]
            }
        }
    
    def download_file(self, url: str, filepath: Path, description: str = "") -> bool:
        """Télécharge un fichier avec barre de progression"""
        try:
            logger.info(f"📥 Téléchargement {description}...")
            
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filepath, 'wb') as file, tqdm(
                desc=f"📦 {description}",
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)
                        pbar.update(len(chunk))
            
            logger.success(f"✅ {description} téléchargé: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur téléchargement {description}: {e}")
            if filepath.exists():
                filepath.unlink()  # Supprimer fichier partiel
            return False
    
    def download_from_huggingface(self, model_id: str, filename: str, description: str = "") -> bool:
        """Télécharge un modèle depuis Hugging Face Hub"""
        try:
            from transformers import AutoModel, AutoTokenizer
            logger.info(f"📥 Téléchargement {description} depuis HF Hub...")
            
            # Cela va télécharger et cacher automatiquement
            model = AutoModel.from_pretrained(model_id)
            
            # Marquer comme téléchargé
            cache_file = self.models_dir / f"{filename}.downloaded"
            cache_file.write_text(f"Downloaded from {model_id}")
            
            logger.success(f"✅ {description} téléchargé et mis en cache")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur téléchargement {description} depuis HF: {e}")
            return False
    
    def is_model_available(self, model_key: str) -> bool:
        """Vérifie si un modèle est disponible localement"""
        config = self.models_config.get(model_key)
        if not config:
            return False
            
        if config.get("huggingface_id"):
            # Vérifier cache file HF
            cache_file = self.models_dir / f"{config['filename']}.downloaded"
            return cache_file.exists()
        else:
            # Vérifier fichier direct
            model_path = self.models_dir / config["filename"]
            return model_path.exists()
    
    def get_model_path(self, model_key: str) -> Optional[Path]:
        """Retourne le chemin vers un modèle s'il existe"""
        config = self.models_config.get(model_key)
        if not config:
            return None
            
        model_path = self.models_dir / config["filename"]
        return model_path if model_path.exists() else None
    
    def download_model(self, model_key: str, force: bool = False) -> bool:
        """Télécharge un modèle spécifique"""
        config = self.models_config.get(model_key)
        if not config:
            logger.error(f"❌ Configuration modèle inconnue: {model_key}")
            return False
        
        # Vérifier si déjà téléchargé
        if not force and self.is_model_available(model_key):
            logger.info(f"✅ Modèle {config['name']} déjà disponible")
            return True
        
        # Téléchargement selon le type
        if config.get("huggingface_id"):
            return self.download_from_huggingface(
                config["huggingface_id"],
                config["filename"],
                config["name"]
            )
        elif config.get("url"):
            model_path = self.models_dir / config["filename"]
            return self.download_file(
                config["url"],
                model_path,
                config["name"]
            )
        else:
            logger.error(f"❌ Aucune source de téléchargement pour {model_key}")
            return False
    
    def download_essential_models(self) -> Dict[str, bool]:
        """Télécharge tous les modèles marqués comme auto_download"""
        results = {}
        
        logger.info("🚀 Téléchargement des modèles essentiels...")
        
        for model_key, config in self.models_config.items():
            if config.get("auto_download", False):
                logger.info(f"📦 Traitement {config['name']}...")
                results[model_key] = self.download_model(model_key)
            else:
                logger.debug(f"⏭️ Ignore {config['name']} (pas en auto-download)")
                
        # Résumé
        successful = sum(1 for success in results.values() if success)
        total = len(results)
        
        if successful == total:
            logger.success(f"🎉 Tous les modèles téléchargés avec succès ({successful}/{total})")
        else:
            logger.warning(f"⚠️ Certains modèles ont échoué ({successful}/{total} réussis)")
            
        return results
    
    def check_system_requirements(self) -> Dict[str, bool]:
        """Vérifie les prérequis système"""
        requirements = {}
        
        # CUDA disponible
        requirements["cuda_available"] = torch.cuda.is_available()
        if requirements["cuda_available"]:
            requirements["gpu_count"] = torch.cuda.device_count()
            requirements["gpu_memory"] = torch.cuda.get_device_properties(0).total_memory
        else:
            requirements["gpu_count"] = 0
            requirements["gpu_memory"] = 0
            
        # Espace disque
        try:
            import shutil
            disk_usage = shutil.disk_usage(self.models_dir)
            requirements["disk_free_gb"] = disk_usage.free / (1024**3)
        except:
            requirements["disk_free_gb"] = 0
            
        # RAM système  
        try:
            import psutil
            requirements["ram_total_gb"] = psutil.virtual_memory().total / (1024**3)
            requirements["ram_available_gb"] = psutil.virtual_memory().available / (1024**3)
        except:
            requirements["ram_total_gb"] = 0
            requirements["ram_available_gb"] = 0
            
        return requirements
    
    def generate_setup_report(self) -> str:
        """Génère un rapport de setup complet"""
        report = []
        report.append("🎯 RAPPORT SETUP MODÈLES")
        report.append("=" * 50)
        
        # Prérequis système
        requirements = self.check_system_requirements()
        report.append(f"\n🖥️ SYSTÈME:")
        report.append(f"   CUDA: {'✅' if requirements['cuda_available'] else '❌'}")
        report.append(f"   GPU: {requirements['gpu_count']} device(s)")
        if requirements['gpu_memory'] > 0:
            report.append(f"   GPU Memory: {requirements['gpu_memory'] / (1024**3):.1f} GB")
        report.append(f"   RAM disponible: {requirements['ram_available_gb']:.1f} GB")
        report.append(f"   Espace disque: {requirements['disk_free_gb']:.1f} GB")
        
        # Status des modèles
        report.append(f"\n📦 MODÈLES:")
        for model_key, config in self.models_config.items():
            available = self.is_model_available(model_key)
            auto_dl = config.get("auto_download", False)
            status = "✅" if available else ("🔄" if auto_dl else "⏭️")
            
            size_info = f"({config.get('size_mb', 0):.1f}MB)" if config.get('size_mb') else ""
            report.append(f"   {status} {config['name']} {size_info}")
            
            if auto_dl and not available:
                report.append(f"      → Sera téléchargé automatiquement")
        
        return "\n".join(report)
    
    def initialize_yolo_with_auto_download(self):
        """Initialise YOLO avec téléchargement automatique si nécessaire"""
        try:
            from ultralytics import YOLO
            
            # Vérifier si modèle YOLO existe
            if not self.is_model_available("yolo"):
                logger.info("🔄 YOLO model manquant, téléchargement automatique...")
                success = self.download_model("yolo")
                if not success:
                    logger.error("❌ Échec téléchargement YOLO")
                    return None
            
            # Charger le modèle
            model_path = self.get_model_path("yolo")
            if model_path and model_path.exists():
                logger.info(f"📥 Chargement YOLO depuis {model_path}")
                return YOLO(str(model_path))
            else:
                # Fallback : laisser ultralytics télécharger
                logger.info("🔄 Fallback: YOLO téléchargement via ultralytics")
                return YOLO("yolo11n.pt")
                
        except Exception as e:
            logger.error(f"❌ Erreur initialisation YOLO: {e}")
            return None


# Instance globale
model_manager = ModelManager()


def ensure_models_ready() -> bool:
    """Assure que tous les modèles essentiels sont prêts"""
    logger.info("🚀 Vérification des modèles essentiels...")
    
    # Afficher rapport
    print(model_manager.generate_setup_report())
    
    # Télécharger les modèles essentiels
    results = model_manager.download_essential_models()
    
    # Vérifier succès
    all_success = all(results.values()) if results else True
    
    if all_success:
        logger.success("✅ Tous les modèles essentiels sont prêts!")
    else:
        failed_models = [k for k, v in results.items() if not v]
        logger.warning(f"⚠️ Modèles échoués: {failed_models}")
    
    return all_success


if __name__ == "__main__":
    # Test du gestionnaire
    ensure_models_ready()