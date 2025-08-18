# 🚀 GUIDE COMPLET UV POUR SERVEUR GPU

## ⚡ POURQUOI UV ?

UV est le gestionnaire de packages Python le plus rapide et moderne :
- **10-100x plus rapide** que pip
- **Résolution de dépendances garantie**
- **Environnements reproductibles**
- **Compatible avec PyPI**
- **Créé par Astral (les créateurs de Ruff)**

---

## 📋 INSTALLATION AUTOMATIQUE UV (3 COMMANDES)

### 1. 🔧 Installation UV + Configuration

```bash
# Installation UV (si pas déjà fait)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env

# Configuration projet + GPU
bash setup_gpu_server_uv.sh
```

### 2. 🧪 Validation Système UV

```bash
# Diagnostic complet UV
uv run python check_gpu_system_uv.py

# Tests complets UV  
uv run python run_gpu_tests_uv.py
```

### 3. 🎯 Test Surveillance UV

```bash
# Test surveillance avec UV
uv run python test_full_system_video.py --video webcam --max-frames 50
```

**C'est tout ! Votre système UV est prêt.**

---

## 🛠️ COMMANDES UV ESSENTIELLES

### Gestion de Projet
```bash
# Synchroniser l'environnement
uv sync

# Ajouter une dépendance
uv add package-name

# Ajouter dépendance de développement
uv add --dev pytest

# Supprimer une dépendance
uv remove package-name

# Mettre à jour toutes les dépendances
uv sync --upgrade
```

### Exécution avec UV
```bash
# Exécuter un script Python
uv run python script.py

# Lancer les tests
uv run pytest

# Lancer avec arguments
uv run python main.py --video webcam

# Shell interactif
uv shell
```

### Gestion des Environnements
```bash
# Créer nouvel environnement
uv venv

# Activer environnement (optionnel avec uv run)
source .venv/bin/activate

# Informations environnement
uv pip list
uv tree
```

---

## 🎬 SURVEILLANCE AVEC UV

### Lancement Production
```bash
# Surveillance webcam
uv run python main.py --video webcam

# Surveillance fichier vidéo
uv run python main.py --video /path/to/video.mp4

# Avec script dédié (créé automatiquement)
./run_surveillance.sh --video webcam
```

### Tests et Validation
```bash
# Test basique
uv run python test_basic_corrections.py

# Test système complet
uv run python test_system_fixed.py

# Test surveillance temps réel
uv run python test_full_system_video.py --video webcam --max-frames 100

# Tests de performance
uv run python run_gpu_tests_uv.py

# Ou avec script dédié
./run_tests.sh
```

---

## 🔥 INSTALLATION GPU OPTIMISÉE

### PyTorch GPU avec UV
```bash
# Installation PyTorch GPU automatique (fait par le script)
uv add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Vérification GPU
uv run python -c "
import torch
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"
```

### Dépendances Spécifiques
```bash
# Ajouter support GPU
uv add --group gpu nvidia-ml-py

# Dépendances développement
uv add --group dev pytest black ruff

# Installation complète avec extras
uv sync --extra gpu --extra dev
```

---

## 📊 MONITORING ET PERFORMANCE

### Diagnostic UV
```bash
# État du projet
uv tree

# Packages installés
uv pip list

# Informations environnement
uv run python -c "
import sys
print(f'Python: {sys.version}')
print(f'Executable: {sys.executable}')
"

# Diagnostic complet
uv run python check_gpu_system_uv.py
```

### Performance GPU
```bash
# Test performance YOLO11
uv run python -c "
import time
import numpy as np
from ultralytics import YOLO

model = YOLO('yolov11n.pt')
images = [np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8) for _ in range(10)]

start = time.time()
for img in images:
    results = model(img, verbose=False)
fps = 10 / (time.time() - start)

print(f'Performance YOLO11: {fps:.1f} FPS')
"

# Monitoring GPU
watch -n 1 nvidia-smi
```

---

## 🔧 DÉVELOPPEMENT AVEC UV

### Workflow Développement
```bash
# 1. Ajouter nouvelle dépendance
uv add new-package

# 2. Développer et tester
uv run python -m pytest

# 3. Formater code
uv run black src/
uv run ruff check src/

# 4. Lancer application
uv run python main.py
```

### Scripts Personnalisés
```bash
# Ajouter script dans pyproject.toml
[project.scripts]
surveillance = "src.main:main"

# Puis utiliser
uv run surveillance --video webcam
```

---

## 🚨 RÉSOLUTION PROBLÈMES UV

### Problèmes Courants
```bash
# 1. Synchronisation ratée
uv sync --reinstall

# 2. Cache corrompu
uv cache clean

# 3. Environnement cassé
rm -rf .venv
uv sync

# 4. Dépendances conflictuelles
uv add package --resolution=lowest-direct

# 5. Réinstallation complète
uv sync --reinstall-package torch
```

### Debug UV
```bash
# Mode verbose
uv -v sync

# Logs détaillés
uv --verbose run python script.py

# Informations système
uv --version
uv python list
```

---

## 📈 OPTIMISATIONS UV

### Performance Maximale
```bash
# Variables d'environnement UV
export UV_CACHE_DIR=/tmp/uv-cache
export UV_PYTHON_PREFERENCE=managed

# Installation parallèle
export UV_CONCURRENT_DOWNLOADS=10

# Cache global
export UV_SYSTEM_PYTHON=1
```

### Configuration Projet
```toml
# uv.toml
[tool.uv]
# Cache partagé
cache-dir = "/tmp/uv-cache"

# Python préféré
python-preference = "managed"

# Index personnalisé
index-url = "https://pypi.org/simple"
extra-index-url = ["https://download.pytorch.org/whl/cu121"]
```

---

## 🎯 COMPARAISON UV vs PIP

| Fonctionnalité | UV | pip + venv |
|----------------|-----|------------|
| **Vitesse installation** | ⚡ 10-100x plus rapide | 🐌 Standard |
| **Résolution dépendances** | ✅ Garantie | ⚠️ Meilleur effort |
| **Reproductibilité** | ✅ Fichier lock automatique | ❌ Manuel |
| **Gestion environnements** | ✅ Intégrée | ❌ Séparée |
| **Interface** | ✅ Moderne | ❌ Fragmentée |
| **Support PyPI** | ✅ 100% compatible | ✅ Natif |

---

## 🎉 AVANTAGES POUR SURVEILLANCE

### Pour notre Projet
```bash
# Installation ultra-rapide (< 2 min vs 20 min)
bash setup_gpu_server_uv.sh

# Environnements reproductibles
uv export > requirements.txt
uv pip install -r requirements.txt

# Développement fluide
uv run python test_full_system_video.py --video webcam

# Déploiement simplifié
uv sync && uv run python main.py
```

### Performance Production
- **Démarrage plus rapide** : `uv run` vs `python`
- **Dépendances optimisées** : Résolution automatique
- **Moins d'erreurs** : Environnements cohérents
- **Maintenance simple** : `uv sync` met tout à jour

---

## 📋 CHECKLIST UV FINALE

Avant production, vérifiez :

- [ ] `uv --version` fonctionne
- [ ] `bash setup_gpu_server_uv.sh` réussi
- [ ] `uv run python check_gpu_system_uv.py` tout vert
- [ ] `uv run python run_gpu_tests_uv.py` 100% réussi
- [ ] `uv run python test_full_system_video.py --video webcam` fonctionne
- [ ] GPU détecté : `uv run python -c "import torch; print(torch.cuda.is_available())"`
- [ ] YOLO11 OK : `uv run python -c "from ultralytics import YOLO; print('OK')"`
- [ ] Types OK : `uv run python test_basic_corrections.py`

---

## 🚀 CONCLUSION UV

**UV transforme complètement l'expérience Python pour notre projet de surveillance :**

✅ **Installation 10-100x plus rapide**  
✅ **Zéro conflit de dépendances**  
✅ **Environnements parfaitement reproductibles**  
✅ **Workflow de développement moderne**  
✅ **Compatible 100% avec l'écosystème Python**

**Votre système de surveillance avec UV sera plus rapide, plus fiable et plus facile à maintenir !**

---

*Guide UV optimisé pour surveillance intelligente - 2025*