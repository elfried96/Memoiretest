# Installation

Ce guide vous accompagne dans l'installation complète du système de surveillance intelligente.

## 🖥️ Prérequis Système

### Configuration Minimale

- **OS**: Linux Ubuntu 20.04+, macOS 12+, Windows 10+
- **Python**: 3.9 ou supérieur
- **RAM**: 8 Go minimum, 16 Go recommandé
- **Stockage**: 10 Go d'espace libre
- **GPU**: Optionnel mais recommandé (CUDA 11.8+)

### Configuration Recommandée

- **OS**: Linux Ubuntu 22.04 LTS
- **Python**: 3.11
- **RAM**: 32 Go
- **GPU**: NVIDIA RTX 3080/4080 ou supérieur
- **VRAM**: 12 Go minimum
- **Stockage**: SSD 50 Go

## 🐍 Installation Python

### Via Package Manager

=== "Ubuntu/Debian"

    ```bash
    # Mise à jour du système
    sudo apt update && sudo apt upgrade -y
    
    # Installation Python et dépendances
    sudo apt install python3.11 python3.11-venv python3.11-dev
    sudo apt install python3-pip build-essential
    
    # Installation des bibliothèques système
    sudo apt install libopencv-dev python3-opencv
    sudo apt install libgl1-mesa-glx libglib2.0-0
    ```

=== "macOS"

    ```bash
    # Avec Homebrew
    brew install python@3.11
    brew install opencv
    
    # Avec MacPorts
    sudo port install python311
    sudo port install opencv4 +python311
    ```

=== "Windows"

    ```powershell
    # Avec Chocolatey
    choco install python --version=3.11.0
    
    # Ou télécharger depuis python.org
    # https://www.python.org/downloads/windows/
    ```

### Vérification de l'installation

```bash
python3 --version
# Python 3.11.x

pip3 --version
# pip 23.x.x
```

## 🔧 Installation CUDA (Optionnel)

Pour accélérer les performances GPU :

### Linux

```bash
# Installation des drivers NVIDIA
sudo apt install nvidia-driver-525
sudo reboot

# Vérification
nvidia-smi

# Installation CUDA Toolkit 11.8
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-11-8

# Variables d'environnement
echo 'export PATH=/usr/local/cuda-11.8/bin${PATH:+:${PATH}}' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
source ~/.bashrc
```

### Windows

1. Télécharger [CUDA Toolkit 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive)
2. Installer le driver NVIDIA compatible
3. Vérifier avec `nvcc --version`

## 📦 Installation du Projet

### Méthode 1: Clone Git + Pip

```bash
# Clonage du repository
git clone https://github.com/elfried-kinzoun/intelligent-surveillance-system.git
cd intelligent-surveillance-system

# Création de l'environnement virtuel
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# ou
venv\Scripts\activate  # Windows

# Mise à jour pip
pip install --upgrade pip

# Installation des dépendances
pip install -r requirements.txt

# Installation du projet
pip install -e .
```

### Méthode 2: Poetry (Recommandé)

```bash
# Installation de Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Clone et installation
git clone https://github.com/elfried-kinzoun/intelligent-surveillance-system.git
cd intelligent-surveillance-system

# Installation avec Poetry
poetry install

# Activation de l'environnement
poetry shell
```

### Méthode 3: Docker

```bash
# Pull de l'image
docker pull ghcr.io/elfried-kinzoun/intelligent-surveillance-system:latest

# Ou build local
git clone https://github.com/elfried-kinzoun/intelligent-surveillance-system.git
cd intelligent-surveillance-system
docker build -t surveillance-system .

# Lancement
docker run -p 8000:8000 -v $(pwd)/data:/app/data surveillance-system
```

## 🧪 Installation Spécifique GPU

### PyTorch avec CUDA

```bash
# Installation PyTorch CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Vérification CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA devices: {torch.cuda.device_count()}')"
```

### Pour Google Colab

```python
# Dans un notebook Colab
!nvidia-smi
!python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Installation automatique
%pip install -q torch torchvision torchaudio
%pip install -q transformers ultralytics opencv-python
```

## 📁 Structure des Répertoires

Après installation, votre structure devrait ressembler à :

```
intelligent-surveillance-system/
├── src/                    # Code source principal
│   ├── core/              # Modules centraux
│   ├── detection/         # Détection et suivi
│   ├── validation/        # Validation croisée
│   └── utils/             # Utilitaires
├── config/                # Fichiers de configuration
├── data/                  # Données et modèles
├── docs/                  # Documentation
├── tests/                 # Tests unitaires
├── notebooks/             # Notebooks Colab/Jupyter
├── docker/               # Fichiers Docker
├── requirements.txt       # Dépendances Python
├── pyproject.toml        # Configuration Poetry
└── README.md             # Documentation principale
```

## ✅ Vérification de l'Installation

### Test Rapide

```python
# Test d'importation
python -c "
from src.core.vlm.model import VisionLanguageModel
from src.detection.yolo.detector import YOLODetector
from src.detection.tracking.tracker import MultiObjectTracker
print('✅ Toutes les importations réussies!')
"
```

### Test Complet

```bash
# Lancement des tests
python -m pytest tests/ -v

# Test de performance
python scripts/benchmark.py

# Test de configuration
python scripts/check_setup.py
```

### Vérification GPU

```python
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"
```

## 🚨 Problèmes Courants

### Erreur CUDA

```bash
# Erreur: CUDA out of memory
# Solution: Réduire la taille des batches
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

### Erreur OpenCV

```bash
# Erreur: ImportError: libGL.so.1
sudo apt install libgl1-mesa-glx libglib2.0-0

# macOS: Permission denied
xcode-select --install
```

### Erreur Transformers

```bash
# Erreur: Cannot load model
# Solution: Vérifier connexion internet et espace disque
huggingface-cli login
```

### Erreur Mémoire

```python
# Configuration pour machines avec RAM limitée
export TRANSFORMERS_CACHE=/tmp/transformers_cache
export HF_DATASETS_OFFLINE=1
```

## 📊 Monitoring d'Installation

### Script de Diagnostic

```bash
# Créer le script
cat > check_installation.py << 'EOF'
#!/usr/bin/env python3
import sys
import importlib
import torch

def check_module(module_name):
    try:
        importlib.import_module(module_name)
        print(f"✅ {module_name}")
        return True
    except ImportError as e:
        print(f"❌ {module_name}: {e}")
        return False

modules = [
    'torch', 'torchvision', 'transformers', 'ultralytics',
    'opencv-python', 'numpy', 'pillow', 'fastapi',
    'redis', 'pydantic', 'loguru'
]

print("🔍 Vérification des modules...")
success = all(check_module(m.replace('-', '_')) for m in modules)

print(f"\n🐍 Python: {sys.version}")
print(f"🔥 PyTorch: {torch.__version__}")
print(f"🎯 CUDA: {torch.cuda.is_available()}")

if success:
    print("\n🎉 Installation complète réussie!")
else:
    print("\n⚠️  Certains modules manquent")
EOF

python check_installation.py
```

## ⚡ Installation Rapide

### One-liner pour développeurs

```bash
curl -sSL https://raw.githubusercontent.com/elfried-kinzoun/intelligent-surveillance-system/main/scripts/quick_install.sh | bash
```

### Google Colab One-click

```python
# Cellule Colab pour installation complète
!curl -sSL https://raw.githubusercontent.com/elfried-kinzoun/intelligent-surveillance-system/main/scripts/colab_setup.py | python
```

## 🔄 Mise à Jour

### Via Git

```bash
cd intelligent-surveillance-system
git pull origin main
pip install -r requirements.txt --upgrade
```

### Via Poetry

```bash
poetry update
```

## 🧹 Désinstallation

```bash
# Suppression de l'environnement virtuel
deactivate
rm -rf venv/

# Suppression du projet
rm -rf intelligent-surveillance-system/

# Nettoyage cache pip
pip cache purge
```

---

**Prochaine étape**: [Configuration du système](configuration.md)