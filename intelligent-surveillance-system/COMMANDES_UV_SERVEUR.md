# ⚡ COMMANDES UV POUR SERVEUR GPU

## 🚀 INSTALLATION ULTRA-RAPIDE (2 COMMANDES)

```bash
# 1. 🔧 Installation UV + Configuration complète
bash setup_gpu_server_uv.sh

# 2. 🧪 Validation système
uv run python run_gpu_tests_uv.py
```

**Votre système UV est prêt en 2-5 minutes !**

---

## 🎯 TESTS GARANTIS UV

```bash
# Test diagnostic
uv run python check_gpu_system_uv.py

# Test basique
uv run python test_basic_corrections.py

# Test surveillance (50 frames)
uv run python test_full_system_video.py --video webcam --max-frames 50

# Tests complets
uv run python run_gpu_tests_uv.py
```

---

## 🚀 DÉPLOIEMENT PRODUCTION UV

```bash
# Surveillance webcam
uv run python main.py --video webcam

# Surveillance fichier vidéo
uv run python main.py --video /path/to/video.mp4

# Avec script dédié (auto-créé)
./run_surveillance.sh --video webcam
```

---

## 🛠️ GESTION UV

```bash
# Synchroniser environnement
uv sync

# Ajouter dépendance
uv add package-name

# Voir packages installés
uv pip list

# Arbre des dépendances
uv tree
```

---

## 📊 MONITORING UV

```bash
# Performance GPU
uv run python -c "
import torch
print(f'GPU: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device: {torch.cuda.get_device_name(0)}')
"

# Test YOLO11 performance
uv run python -c "
import time, numpy as np
from ultralytics import YOLO
model = YOLO('yolov11n.pt')
img = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
start = time.time()
for _ in range(10):
    results = model(img, verbose=False)
fps = 10 / (time.time() - start)
print(f'YOLO11: {fps:.1f} FPS')
"
```

---

## 🔧 DÉPANNAGE UV

```bash
# Réinstallation environnement
rm -rf .venv && uv sync

# Cache clean
uv cache clean

# Réinstallation dépendance
uv sync --reinstall-package torch
```

---

## ⚡ AVANTAGES UV

- **10-100x plus rapide** que pip
- **Zéro conflit** de dépendances  
- **Environnements reproductibles**
- **Interface moderne**

**UV transforme votre workflow Python !** 🚀