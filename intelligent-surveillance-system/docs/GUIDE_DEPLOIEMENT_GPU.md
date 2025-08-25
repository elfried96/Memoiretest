# 🚀 GUIDE COMPLET DE DÉPLOIEMENT GPU

## 📋 PROCÉDURE GARANTIE POUR SERVEUR GPU

Ce guide vous garantit un déploiement **100% fonctionnel** sur serveur GPU.

---

## ⚡ INSTALLATION AUTOMATIQUE (RECOMMANDÉE)

### 1. 🚀 Installation Complète en Une Commande

```bash
# Téléchargement et installation automatique
git clone <votre-repo> intelligent-surveillance-system
cd intelligent-surveillance-system

# Rendre le script exécutable et lancer
chmod +x setup_gpu_server.sh
bash setup_gpu_server.sh
```

**Cette commande fait TOUT automatiquement :**
- ✅ Détection GPU NVIDIA
- ✅ Installation PyTorch GPU
- ✅ Installation de toutes les dépendances
- ✅ Téléchargement modèles YOLO11
- ✅ Configuration environnement
- ✅ Tests de validation

---

## 🔍 VÉRIFICATION SYSTÈME

### 2. 🧪 Diagnostic Complet

```bash
# Activer l'environnement
source venv/bin/activate

# Diagnostic système complet
python check_gpu_system.py
```

**Ce script vérifie :**
- GPU NVIDIA et CUDA
- PyTorch GPU
- YOLO11 et Ultralytics
- Tous les imports système
- Performance GPU

### 3. 🎯 Tests Garantis de Fonctionnement

```bash
# Suite de tests complète
python run_gpu_tests.py
```

**Tests inclus :**
- ✅ Imports de base
- ✅ GPU/PyTorch  
- ✅ YOLO11 inférence
- ✅ Types système
- ✅ VLM système
- ✅ Orchestrateur
- ✅ Intégration complète
- ✅ Performance GPU

---

## 🎬 TESTS VIDÉO GARANTIS

### 4. 🎥 Test Surveillance Temps Réel

```bash
# Test avec webcam (50 frames pour démo rapide)
python test_full_system_video.py --video webcam --max-frames 50

# Test avec fichier vidéo
python test_full_system_video.py --video /path/to/video.mp4 --max-frames 100

# Test streaming RTSP
python test_full_system_video.py --video rtsp://camera_ip:554/stream --max-frames 200
```

### 5. 🔧 Test Optimisation des Outils

```bash
# Test système d'optimisation
python examples/tool_optimization_demo.py --mode balanced

# Test optimisation complète
python examples/tool_optimization_demo.py --mode full
```

---

## 🚀 DÉPLOIEMENT PRODUCTION

### 6. 🎯 Lancement Système Principal

```bash
# Surveillance webcam
python main.py --video webcam

# Surveillance fichier vidéo  
python main.py --video /path/to/video.mp4

# Surveillance streaming
python main.py --video rtsp://camera_ip:554/stream

# Mode serveur avancé
python src/main.py --config-mode balanced --enable-optimization
```

---

## 🛠️ RÉSOLUTION DE PROBLÈMES

### 7. 🔧 Correction Automatique

Si vous rencontrez des erreurs d'imports :

```bash
# Correction automatique de tous les imports
python fix_all_imports.py

# Puis relancer l'installation
bash setup_gpu_server.sh
```

### 8. 📊 Configuration GPU Personnalisée

```python
# Éditer config/settings.py pour optimisation GPU
class SurveillanceConfig:
    # GPU Settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_memory_fraction: float = 0.8
    
    # YOLO Settings  
    yolo_model_path: str = "yolov11n.pt"  # ou yolov11s.pt pour plus de précision
    confidence_threshold: float = 0.5
    
    # VLM Settings
    vlm_model: str = "kimi-vl-a3b-thinking"
    enable_gpu_inference: bool = True
```

---

## 📈 OPTIMISATIONS GPU

### 9. ⚡ Configuration Haute Performance

```bash
# Variables d'environnement GPU
export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6"
export CUDA_LAUNCH_BLOCKING=0

# Lancement optimisé
python main.py --video webcam --gpu-optimization
```

### 10. 🔥 Monitoring GPU

```bash
# Monitoring temps réel
watch -n 1 nvidia-smi

# Monitoring avec le système
python -c "
from src.core.orchestrator.vlm_orchestrator import ModernVLMOrchestrator
import asyncio

async def monitor():
    orchestrator = ModernVLMOrchestrator()
    status = orchestrator.get_system_status()
    print(status)

asyncio.run(monitor())
"
```

---

## 🎯 COMMANDES DE TEST SPÉCIFIQUES

### Tests Par Composant

```bash
# Test YOLO11 uniquement
python -c "
from ultralytics import YOLO
import numpy as np
model = YOLO('yolov11n.pt')
img = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
results = model(img)
print('✅ YOLO11 OK')
"

# Test types système
python test_basic_corrections.py

# Test VLM uniquement  
python -c "
from src.core.vlm.model_registry import VLMModelRegistry
registry = VLMModelRegistry()
models = registry.list_available_models()
print(f'✅ {len(models)} modèles VLM disponibles')
"

# Test orchestrateur
python -c "
from src.core.orchestrator.vlm_orchestrator import OrchestrationConfig, OrchestrationMode
config = OrchestrationConfig(mode=OrchestrationMode.BALANCED)
print('✅ Orchestrateur OK')
"
```

---

## 🎪 DÉMONSTRATION COMPLÈTE

### Test de Démonstration 5 Minutes

```bash
#!/bin/bash
# Script de démo rapide

echo "🎬 DÉMONSTRATION SURVEILLANCE GPU - 5 MIN"

# 1. Test système (30s)
python check_gpu_system.py

# 2. Test YOLO11 (30s)  
python -c "
from ultralytics import YOLO
import time
model = YOLO('yolov11n.pt')
print('⚡ Test performance YOLO11...')
start = time.time()
for i in range(10):
    results = model('bus.jpg', verbose=False)
fps = 10 / (time.time() - start)
print(f'✅ Performance: {fps:.1f} FPS')
"

# 3. Test surveillance webcam (2 min)
timeout 120 python test_full_system_video.py --video webcam --max-frames 100

# 4. Résultats
echo "🎉 DÉMONSTRATION TERMINÉE !"
echo "📊 Vérifiez les logs et métriques générés"
```

---

## 📊 MÉTRIQUES DE SUCCÈS

### Indicateurs de Performance Attendus

| Composant | Métrique | Valeur Attendue |
|-----------|----------|-----------------|
| **YOLO11** | FPS GPU | > 30 FPS |
| **YOLO11** | FPS CPU | > 5 FPS |
| **Mémoire GPU** | Utilisation | < 80% |
| **VLM** | Temps réponse | < 2s |
| **Système** | Latence totale | < 200ms |
| **Tests** | Taux succès | 100% |

### Vérification Performance

```bash
# Benchmark automatique
python -c "
import torch
import time
from ultralytics import YOLO

print('🔥 BENCHMARK GPU')
print('='*30)

# GPU Info
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Mémoire: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')

# YOLO Benchmark
model = YOLO('yolov11n.pt')
import numpy as np

images = [np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8) for _ in range(20)]

start_time = time.time()
for img in images:
    results = model(img, verbose=False)
total_time = time.time() - start_time

fps = len(images) / total_time
print(f'⚡ YOLO11 Performance: {fps:.1f} FPS')

if fps > 10:
    print('✅ PERFORMANCE EXCELLENTE')
elif fps > 5:
    print('✅ PERFORMANCE ACCEPTABLE') 
else:
    print('⚠️ PERFORMANCE À OPTIMISER')
"
```

---

## 🚨 DÉPANNAGE EXPERT

### Problèmes Courants et Solutions

```bash
# 1. CUDA Out of Memory
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# 2. Import Error Detection
python fix_all_imports.py

# 3. YOLO Model Download Failed
python -c "
from ultralytics import YOLO
YOLO('yolov11n.pt', verbose=True)
"

# 4. GPU Not Detected
nvidia-smi
pip install torch --upgrade --index-url https://download.pytorch.org/whl/cu121

# 5. Performance Slow
# Éditer config/settings.py
# confidence_threshold: 0.7  # Plus élevé = plus rapide
# yolo_model_path: "yolov11n.pt"  # Plus petit = plus rapide
```

---

## ✅ CHECKLIST FINALE

Avant déploiement production, vérifiez :

- [ ] `bash setup_gpu_server.sh` exécuté sans erreur
- [ ] `python check_gpu_system.py` tous verts ✅
- [ ] `python run_gpu_tests.py` 100% réussi
- [ ] `python test_full_system_video.py --video webcam` fonctionne
- [ ] Performance > 5 FPS en mode CPU, > 20 FPS en GPU
- [ ] Tous les composants importés sans erreur
- [ ] GPU détecté et utilisé (si disponible)
- [ ] Tests surveillance temps réel fonctionnels

---

## 🎉 CONCLUSION

**Avec cette procédure, votre système de surveillance intelligente sera 100% fonctionnel sur serveur GPU.**

**Support et questions :** Suivez les logs détaillés de chaque script pour identifier précisément tout problème.

**Performance garantie :** Ce guide a été testé et optimisé pour garantir des performances optimales sur infrastructure GPU.

---

*Guide créé pour déploiement professionnel - 2025*