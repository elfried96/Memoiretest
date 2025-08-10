#!/usr/bin/env python3
"""
Script de configuration rapide pour le système de surveillance intelligente.

Ce script automatise:
- Installation des dépendances
- Téléchargement des modèles pré-entraînés
- Configuration des répertoires
- Tests d'intégrité du système
- Génération des fichiers de configuration
"""

import os
import sys
import subprocess
import json
import yaml
import argparse
import requests
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional
import logging
from datetime import datetime
import platform
import shutil

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SurveillanceSystemSetup:
    """Configurateur automatique du système de surveillance."""
    
    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path).resolve()
        self.system_info = self._get_system_info()
        self.setup_config = self._load_setup_config()
        
    def _get_system_info(self) -> Dict:
        """Détecte les informations système."""
        
        info = {
            'platform': platform.system(),
            'architecture': platform.architecture()[0],
            'python_version': platform.python_version(),
            'cpu_count': os.cpu_count(),
            'memory_gb': 0,
            'gpu_available': False,
            'gpu_memory_gb': 0,
            'docker_available': False,
            'git_available': False
        }
        
        # Détection GPU
        try:
            import torch
            info['gpu_available'] = torch.cuda.is_available()
            if info['gpu_available']:
                info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1e9
                info['gpu_name'] = torch.cuda.get_device_name(0)
        except ImportError:
            pass
        
        # Détection mémoire système
        try:
            import psutil
            info['memory_gb'] = psutil.virtual_memory().total / 1e9
        except ImportError:
            pass
        
        # Détection Docker
        try:
            subprocess.run(['docker', '--version'], 
                         capture_output=True, check=True)
            info['docker_available'] = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        
        # Détection Git
        try:
            subprocess.run(['git', '--version'], 
                         capture_output=True, check=True)
            info['git_available'] = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        
        return info
    
    def _load_setup_config(self) -> Dict:
        """Charge la configuration de setup."""
        
        return {
            'dependencies': {
                'core': [
                    'torch>=2.0.0',
                    'torchvision>=0.15.0',
                    'ultralytics>=8.0.0',
                    'transformers>=4.30.0',
                    'accelerate>=0.20.0',
                    'peft>=0.4.0',
                    'datasets>=2.12.0'
                ],
                'vision': [
                    'opencv-python>=4.8.0',
                    'pillow>=9.5.0',
                    'albumentations>=1.3.0',
                    'scikit-image>=0.20.0'
                ],
                'ml': [
                    'scikit-learn>=1.3.0',
                    'scipy>=1.10.0',
                    'numpy>=1.24.0',
                    'pandas>=2.0.0'
                ],
                'monitoring': [
                    'psutil>=5.9.0',
                    'tqdm>=4.65.0',
                    'matplotlib>=3.7.0',
                    'seaborn>=0.12.0'
                ],
                'optional': [
                    'wandb>=0.15.0',
                    'tensorboard>=2.13.0',
                    'jupyter>=1.0.0',
                    'onnxruntime-gpu>=1.15.0',
                    'pynvml>=11.5.0'
                ]
            },
            'models': {
                'yolo': {
                    'yolov8n.pt': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt',
                    'yolov8s.pt': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt'
                },
                'vlm': {
                    'model_name': 'llava-hf/llava-v1.6-mistral-7b-hf',
                    'cache_dir': './models/vlm_cache'
                }
            },
            'directories': [
                'data/datasets/surveillance_detection/images/train',
                'data/datasets/surveillance_detection/images/val',
                'data/datasets/surveillance_detection/images/test',
                'data/datasets/surveillance_detection/labels/train',
                'data/datasets/surveillance_detection/labels/val',
                'data/datasets/surveillance_detection/labels/test',
                'data/datasets/vlm_surveillance/images',
                'data/datasets/behavior_analysis/sequences',
                'data/datasets/synthetic/generated_scenes',
                'models/yolo',
                'models/vlm',
                'models/validation',
                'logs',
                'outputs/training',
                'outputs/inference',
                'outputs/evaluation'
            ]
        }
    
    def run_complete_setup(self, install_optional: bool = False, 
                          skip_models: bool = False) -> bool:
        """Exécute la configuration complète."""
        
        logger.info("🚀 Démarrage de la configuration complète du système de surveillance")
        logger.info(f"📋 Système détecté: {self.system_info}")
        
        setup_steps = [
            ('create_directories', 'Création des répertoires'),
            ('install_dependencies', 'Installation des dépendances', install_optional),
            ('download_models', 'Téléchargement des modèles', skip_models),
            ('generate_configs', 'Génération des configurations'),
            ('run_system_tests', 'Tests d\'intégrité système'),
            ('create_scripts', 'Création des scripts utilitaires'),
            ('finalize_setup', 'Finalisation de la configuration')
        ]
        
        for i, step_info in enumerate(setup_steps, 1):
            step_name = step_info[0]
            step_desc = step_info[1]
            
            logger.info(f"[{i}/{len(setup_steps)}] {step_desc}...")
            
            try:
                if len(step_info) > 2:
                    # Étape avec paramètre
                    result = getattr(self, step_name)(step_info[2])
                else:
                    result = getattr(self, step_name)()
                
                if not result:
                    logger.error(f"❌ Échec de l'étape: {step_desc}")
                    return False
                
                logger.info(f"✅ {step_desc} terminé")
                
            except Exception as e:
                logger.error(f"❌ Erreur lors de {step_desc}: {e}")
                return False
        
        logger.info("🎉 Configuration complète terminée avec succès!")
        self._print_setup_summary()
        
        return True
    
    def create_directories(self) -> bool:
        """Crée la structure de répertoires."""
        
        try:
            for directory in self.setup_config['directories']:
                dir_path = self.base_path / directory
                dir_path.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Répertoire créé: {dir_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur création répertoires: {e}")
            return False
    
    def install_dependencies(self, install_optional: bool = False) -> bool:
        """Installe les dépendances Python."""
        
        try:
            # Installation des dépendances core
            for category in ['core', 'vision', 'ml', 'monitoring']:
                packages = self.setup_config['dependencies'][category]
                logger.info(f"Installation des packages {category}: {packages}")
                
                cmd = [sys.executable, '-m', 'pip', 'install'] + packages
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    logger.warning(f"Erreur installation {category}: {result.stderr}")
                    # Continue quand même, certains packages peuvent être optionnels
            
            # Installation des dépendances optionnelles
            if install_optional:
                optional_packages = self.setup_config['dependencies']['optional']
                logger.info(f"Installation des packages optionnels: {optional_packages}")
                
                for package in optional_packages:
                    try:
                        cmd = [sys.executable, '-m', 'pip', 'install', package]
                        subprocess.run(cmd, check=True, capture_output=True)
                        logger.debug(f"Package optionnel installé: {package}")
                    except subprocess.CalledProcessError as e:
                        logger.warning(f"Package optionnel non installé: {package}")
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur installation dépendances: {e}")
            return False
    
    def download_models(self, skip_download: bool = False) -> bool:
        """Télécharge les modèles pré-entraînés."""
        
        if skip_download:
            logger.info("⏭️ Téléchargement des modèles ignoré")
            return True
        
        try:
            models_config = self.setup_config['models']
            
            # Téléchargement des modèles YOLO
            yolo_dir = self.base_path / 'models' / 'yolo'
            yolo_dir.mkdir(parents=True, exist_ok=True)
            
            for model_name, model_url in models_config['yolo'].items():
                model_path = yolo_dir / model_name
                
                if model_path.exists():
                    logger.info(f"Modèle déjà présent: {model_name}")
                    continue
                
                logger.info(f"Téléchargement de {model_name}...")
                response = requests.get(model_url, stream=True)
                response.raise_for_status()
                
                with open(model_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                logger.info(f"✅ {model_name} téléchargé")
            
            # Préparation du cache VLM
            vlm_config = models_config['vlm']
            vlm_cache_dir = self.base_path / vlm_config['cache_dir']
            vlm_cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Le modèle VLM sera téléchargé automatiquement lors du premier usage
            logger.info(f"Cache VLM préparé: {vlm_cache_dir}")
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur téléchargement modèles: {e}")
            return False
    
    def generate_configs(self) -> bool:
        """Génère les fichiers de configuration."""
        
        try:
            # Configuration principale du système
            system_config = {
                'system': {
                    'base_path': str(self.base_path),
                    'data_path': str(self.base_path / 'data'),
                    'models_path': str(self.base_path / 'models'),
                    'logs_path': str(self.base_path / 'logs'),
                    'outputs_path': str(self.base_path / 'outputs')
                },
                'hardware': {
                    'gpu_available': self.system_info['gpu_available'],
                    'gpu_memory_gb': self.system_info.get('gpu_memory_gb', 0),
                    'cpu_count': self.system_info['cpu_count'],
                    'memory_gb': self.system_info.get('memory_gb', 8)
                },
                'models': {
                    'yolo': {
                        'model_path': 'models/yolo/yolov8n.pt',
                        'device': 'cuda' if self.system_info['gpu_available'] else 'cpu',
                        'confidence_threshold': 0.25,
                        'iou_threshold': 0.45
                    },
                    'vlm': {
                        'model_name': 'llava-hf/llava-v1.6-mistral-7b-hf',
                        'cache_dir': 'models/vlm_cache',
                        'load_in_4bit': True,
                        'max_new_tokens': 256
                    }
                },
                'pipeline': {
                    'max_concurrent_streams': min(4, self.system_info['cpu_count']),
                    'frame_buffer_size': 100,
                    'detection_threshold': 0.25,
                    'validation_threshold': 0.7
                }
            }
            
            config_path = self.base_path / 'config' / 'system_config.yaml'
            config_path.parent.mkdir(exist_ok=True)
            
            with open(config_path, 'w') as f:
                yaml.dump(system_config, f, default_flow_style=False, indent=2)
            
            logger.info(f"Configuration système générée: {config_path}")
            
            # Configuration dataset YOLO
            dataset_config = {
                'path': str(self.base_path / 'data' / 'datasets' / 'surveillance_detection'),
                'train': 'images/train',
                'val': 'images/val',
                'test': 'images/test',
                'names': {
                    0: 'person', 1: 'handbag', 2: 'backpack', 3: 'suitcase',
                    4: 'bottle', 5: 'cup', 6: 'cell_phone', 7: 'book',
                    8: 'shopping_cart', 9: 'suspicious_object'
                }
            }
            
            dataset_path = self.base_path / 'data' / 'datasets' / 'surveillance_detection' / 'dataset.yaml'
            dataset_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(dataset_path, 'w') as f:
                yaml.dump(dataset_config, f, default_flow_style=False, indent=2)
            
            logger.info(f"Configuration dataset générée: {dataset_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur génération configurations: {e}")
            return False
    
    def run_system_tests(self) -> bool:
        """Exécute les tests d'intégrité système."""
        
        try:
            test_results = {
                'python_imports': self._test_python_imports(),
                'model_loading': self._test_model_loading(),
                'gpu_functionality': self._test_gpu_functionality(),
                'file_structure': self._test_file_structure()
            }
            
            all_passed = all(test_results.values())
            
            logger.info("📊 Résultats des tests:")
            for test_name, passed in test_results.items():
                status = "✅ PASS" if passed else "❌ FAIL"
                logger.info(f"  {test_name}: {status}")
            
            if not all_passed:
                logger.warning("⚠️ Certains tests ont échoué, vérifiez la configuration")
            
            return all_passed
            
        except Exception as e:
            logger.error(f"Erreur tests système: {e}")
            return False
    
    def _test_python_imports(self) -> bool:
        """Test des imports Python critiques."""
        
        critical_modules = [
            'torch', 'torchvision', 'ultralytics', 
            'transformers', 'cv2', 'PIL', 'numpy'
        ]
        
        for module in critical_modules:
            try:
                __import__(module)
            except ImportError as e:
                logger.error(f"Import échoué: {module} - {e}")
                return False
        
        return True
    
    def _test_model_loading(self) -> bool:
        """Test de chargement des modèles."""
        
        try:
            # Test YOLO
            yolo_path = self.base_path / 'models' / 'yolo' / 'yolov8n.pt'
            if yolo_path.exists():
                from ultralytics import YOLO
                model = YOLO(str(yolo_path))
                logger.debug("Modèle YOLO chargé avec succès")
            else:
                logger.warning("Modèle YOLO non trouvé")
            
            # Test des imports VLM (sans charger le modèle complet)
            from transformers import AutoTokenizer, AutoModelForCausalLM
            logger.debug("Modules VLM importés avec succès")
            
            return True
            
        except Exception as e:
            logger.error(f"Test modèles échoué: {e}")
            return False
    
    def _test_gpu_functionality(self) -> bool:
        """Test de fonctionnalité GPU."""
        
        if not self.system_info['gpu_available']:
            logger.info("GPU non disponible, test ignoré")
            return True
        
        try:
            import torch
            
            # Test basique GPU
            device = torch.device('cuda')
            x = torch.randn(10, 10, device=device)
            y = torch.randn(10, 10, device=device)
            z = x @ y
            
            logger.debug("Test GPU basique réussi")
            return True
            
        except Exception as e:
            logger.error(f"Test GPU échoué: {e}")
            return False
    
    def _test_file_structure(self) -> bool:
        """Test de la structure de fichiers."""
        
        critical_dirs = [
            'data/datasets',
            'models',
            'logs',
            'config'
        ]
        
        for dir_name in critical_dirs:
            dir_path = self.base_path / dir_name
            if not dir_path.exists():
                logger.error(f"Répertoire critique manquant: {dir_path}")
                return False
        
        return True
    
    def create_scripts(self) -> bool:
        """Crée les scripts utilitaires."""
        
        try:
            scripts_dir = self.base_path / 'scripts'
            scripts_dir.mkdir(exist_ok=True)
            
            # Script de démarrage rapide
            quick_start_script = '''#!/usr/bin/env python3
"""Script de démarrage rapide du système de surveillance."""

import sys
import os
from pathlib import Path

# Ajout du chemin du projet
project_path = Path(__file__).parent.parent
sys.path.append(str(project_path / "src"))

def main():
    print("🚀 Démarrage du système de surveillance intelligente...")
    
    try:
        # Import des modules principaux
        from detection.yolo.detector import YOLODetector
        from core.vlm.model import VisionLanguageModel
        
        print("✅ Modules importés avec succès")
        
        # Initialisation basique
        detector = YOLODetector()
        print("✅ Détecteur YOLO initialisé")
        
        print("🎉 Système prêt à fonctionner!")
        print("📚 Consultez les notebooks dans le dossier 'notebooks' pour commencer")
        
    except Exception as e:
        print(f"❌ Erreur lors du démarrage: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
'''
            
            quick_start_path = scripts_dir / 'quick_start.py'
            with open(quick_start_path, 'w') as f:
                f.write(quick_start_script)
            
            # Rendre exécutable
            quick_start_path.chmod(0o755)
            
            # Script de génération de données synthétiques
            synthetic_script = f'''#!/usr/bin/env python3
"""Script de génération rapide de données synthétiques."""

import sys
from pathlib import Path

project_path = Path(__file__).parent.parent
sys.path.append(str(project_path))

from scripts.synthetic_data_generator import SyntheticDatasetGenerator, SyntheticConfig

def main():
    print("🎭 Génération de données synthétiques...")
    
    config = SyntheticConfig(
        output_dir="{self.base_path}/data/synthetic",
        num_scenes=1000,
        image_size=(640, 640)
    )
    
    generator = SyntheticDatasetGenerator(config)
    stats = generator.generate_dataset()
    
    print(f"✅ Généré: {{stats['generated_scenes']}} scènes")
    print(f"📊 Annotations: {{stats['total_annotations']}}")
    
if __name__ == "__main__":
    main()
'''
            
            synthetic_path = scripts_dir / 'generate_synthetic_data.py'
            with open(synthetic_path, 'w') as f:
                f.write(synthetic_script)
            
            synthetic_path.chmod(0o755)
            
            logger.info(f"Scripts utilitaires créés dans {scripts_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur création scripts: {e}")
            return False
    
    def finalize_setup(self) -> bool:
        """Finalise la configuration."""
        
        try:
            # Création du fichier de version
            version_info = {
                'version': '1.0.0',
                'setup_date': datetime.now().isoformat(),
                'system_info': self.system_info,
                'components': [
                    'YOLO Detection',
                    'VLM Analysis', 
                    'Cross Validation',
                    'Performance Monitoring'
                ]
            }
            
            version_path = self.base_path / '.system_info.json'
            with open(version_path, 'w') as f:
                json.dump(version_info, f, indent=2)
            
            # Création du README de démarrage
            readme_content = f"""# Système de Surveillance Intelligente

## ✅ Configuration Terminée

Ce système a été configuré automatiquement le {datetime.now().strftime('%d/%m/%Y à %H:%M')}.

## 🚀 Démarrage Rapide

1. **Test du système:**
   ```bash
   python scripts/quick_start.py
   ```

2. **Génération de données d'entraînement:**
   ```bash
   python scripts/generate_synthetic_data.py
   ```

3. **Notebooks d'entraînement:**
   - `notebooks/1_fine_tuning_vlm.ipynb` - Fine-tuning du modèle VLM
   - `notebooks/2_transfer_learning_yolo.ipynb` - Transfer learning YOLO
   - `notebooks/3_end_to_end_training.ipynb` - Entraînement complet

## 📁 Structure du Projet

```
{self.base_path.name}/
├── src/                    # Code source principal
├── data/                   # Données et datasets  
├── models/                 # Modèles entraînés
├── notebooks/              # Notebooks Jupyter
├── scripts/                # Scripts utilitaires
├── config/                 # Fichiers de configuration
└── docs/                   # Documentation

```

## ⚙️ Configuration Système

- **GPU**: {'Disponible' if self.system_info['gpu_available'] else 'Non disponible'}
- **Mémoire**: {self.system_info.get('memory_gb', 0):.1f} GB
- **CPUs**: {self.system_info['cpu_count']}

## 📚 Documentation

Consultez `docs/training_methodologies.md` pour la documentation complète des méthodologies d'entraînement.

## 🔧 Support

En cas de problème, vérifiez:
1. Les logs dans `logs/`
2. La configuration dans `config/`
3. Les tests avec `python scripts/quick_start.py`
"""
            
            readme_path = self.base_path / 'QUICK_START.md'
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write(readme_content)
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur finalisation: {e}")
            return False
    
    def _print_setup_summary(self):
        """Affiche le résumé de la configuration."""
        
        print("\n" + "="*60)
        print("🎉 CONFIGURATION SYSTÈME TERMINÉE")
        print("="*60)
        
        print(f"📁 Répertoire: {self.base_path}")
        print(f"🖥️  Système: {self.system_info['platform']} {self.system_info['architecture']}")
        print(f"🐍 Python: {self.system_info['python_version']}")
        print(f"💾 Mémoire: {self.system_info.get('memory_gb', 0):.1f} GB")
        print(f"⚡ CPUs: {self.system_info['cpu_count']}")
        
        if self.system_info['gpu_available']:
            print(f"🎮 GPU: {self.system_info.get('gpu_name', 'Disponible')}")
            print(f"💾 Mémoire GPU: {self.system_info.get('gpu_memory_gb', 0):.1f} GB")
        else:
            print("🎮 GPU: Non disponible")
        
        print("\n📚 PROCHAINES ÉTAPES:")
        print("1. Testez le système: python scripts/quick_start.py")
        print("2. Générez des données: python scripts/generate_synthetic_data.py") 
        print("3. Consultez les notebooks dans notebooks/")
        print("4. Lisez la documentation dans docs/")
        
        print("\n🔗 FICHIERS IMPORTANTS:")
        print("- config/system_config.yaml - Configuration principale")
        print("- QUICK_START.md - Guide de démarrage")
        print("- .system_info.json - Informations système")
        
        print("\n" + "="*60)


def main():
    """Fonction principale avec interface en ligne de commande."""
    
    parser = argparse.ArgumentParser(
        description='Configuration automatique du système de surveillance intelligente'
    )
    
    parser.add_argument('--path', '-p', default='.',
                       help='Répertoire de base du projet')
    parser.add_argument('--install-optional', action='store_true',
                       help='Installer les dépendances optionnelles')
    parser.add_argument('--skip-models', action='store_true',
                       help='Ignorer le téléchargement des modèles')
    parser.add_argument('--test-only', action='store_true',
                       help='Exécuter uniquement les tests système')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Mode verbose')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialisation du configurateur
    setup = SurveillanceSystemSetup(args.path)
    
    if args.test_only:
        # Tests uniquement
        logger.info("🧪 Exécution des tests système uniquement")
        success = setup.run_system_tests()
        return 0 if success else 1
    
    # Configuration complète
    success = setup.run_complete_setup(
        install_optional=args.install_optional,
        skip_models=args.skip_models
    )
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())