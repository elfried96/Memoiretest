# Tests sur Google Colab

Ce guide vous permet de tester rapidement le système de surveillance sur Google Colab avec GPU gratuit.

## 🚀 Lancement Rapide

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elfried-kinzoun/intelligent-surveillance-system/blob/main/notebooks/demo.ipynb)

## 📋 Prérequis Colab

### Activation GPU

1. **Menu Runtime** → **Change runtime type**
2. **Hardware accelerator** → **GPU** (T4 gratuit)
3. **Click Save**

### Vérification GPU

```python
# Vérifier GPU disponible
!nvidia-smi

import torch
print(f"CUDA disponible: {torch.cuda.is_available()}")
print(f"Nombre de GPU: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU actuel: {torch.cuda.get_device_name(0)}")
    print(f"Mémoire GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

## 🔧 Installation Complète

### Cellule 1: Configuration Environnement

```python
# Configuration pour éviter les timeouts
import os
os.environ['TRANSFORMERS_CACHE'] = '/content/cache'
os.environ['HF_HOME'] = '/content/hf_cache'

# Installation des dépendances système
!apt-get update -qq
!apt-get install -y -qq libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1

# Création des répertoires
!mkdir -p /content/cache /content/hf_cache /content/data
```

### Cellule 2: Clone et Installation

```python
# Clone du repository
!git clone https://github.com/elfried-kinzoun/intelligent-surveillance-system.git
%cd intelligent-surveillance-system

# Installation des requirements optimisés pour Colab
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install -q transformers[torch] ultralytics opencv-python-headless
!pip install -q fastapi uvicorn redis pydantic loguru
!pip install -q matplotlib seaborn plotly ipywidgets

# Installation du projet
!pip install -q -e .

print("✅ Installation terminée!")
```

### Cellule 3: Vérification Installation

```python
# Test des importations critiques
try:
    from src.core.vlm.model import VisionLanguageModel
    from src.detection.yolo.detector import YOLODetector  
    from src.detection.tracking.tracker import MultiObjectTracker
    from src.validation.cross_validator import CrossValidator
    print("✅ Tous les modules importés avec succès!")
except ImportError as e:
    print(f"❌ Erreur d'import: {e}")

# Test GPU PyTorch
import torch
if torch.cuda.is_available():
    x = torch.randn(10, 10).cuda()
    print(f"✅ GPU test réussi: {x.device}")
else:
    print("⚠️  GPU non disponible, utilisation CPU")
```

## 🎮 Démonstration Rapide

### Demo 1: Détection YOLO

```python
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Téléchargement d'une image de test
!wget -q "https://ultralytics.com/images/bus.jpg" -O test_image.jpg

# Initialisation du détecteur
from src.detection.yolo.detector import YOLODetector

detector = YOLODetector(model_path="yolov8n.pt")
detector.load_model()

# Chargement et préparation de l'image
image = cv2.imread('test_image.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Création d'un frame factice
from src.core.types import Frame
from datetime import datetime

frame = Frame(
    image=image_rgb,
    timestamp=datetime.now(),
    frame_id=1,
    stream_id="demo",
    width=image_rgb.shape[1],
    height=image_rgb.shape[0]
)

# Détection
detections = detector.detect(frame)

# Affichage des résultats
print(f"🎯 {len(detections)} objets détectés:")
for i, det in enumerate(detections[:5]):
    print(f"  {i+1}. {det.class_name}: {det.confidence:.3f}")

# Visualisation
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

ax1.imshow(image_rgb)
ax1.set_title("Image Originale")
ax1.axis('off')

# Image avec détections
image_with_det = image_rgb.copy()
for det in detections:
    bbox = det.bbox
    cv2.rectangle(image_with_det, 
                 (bbox.x, bbox.y), 
                 (bbox.x + bbox.width, bbox.y + bbox.height),
                 (255, 0, 0), 2)
    cv2.putText(image_with_det, 
               f"{det.class_name}: {det.confidence:.2f}",
               (bbox.x, bbox.y - 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

ax2.imshow(image_with_det)
ax2.set_title(f"Détections ({len(detections)} objets)")
ax2.axis('off')

plt.tight_layout()
plt.show()
```

### Demo 2: VLM Analysis (Version Allégée)

```python
# Configuration pour modèle léger (compatible Colab gratuit)
from src.core.vlm.model import VisionLanguageModel

# Utilisation d'un modèle plus léger pour Colab
vlm = VisionLanguageModel(
    model_name="microsoft/git-base-coco",  # Modèle plus léger
    device="cuda" if torch.cuda.is_available() else "cpu",
    load_in_4bit=True,  # Quantization pour économiser la mémoire
    max_tokens=128
)

print("⏳ Chargement du modèle VLM (peut prendre 2-3 minutes)...")
await vlm.load_model()
print("✅ Modèle VLM chargé!")

# Test d'analyse simple
import base64
from io import BytesIO

# Conversion image en base64
pil_image = Image.fromarray(image_rgb)
buffer = BytesIO()
pil_image.save(buffer, format="JPEG")
image_b64 = base64.b64encode(buffer.getvalue()).decode()

# Requête d'analyse
from src.core.types import AnalysisRequest

request = AnalysisRequest(
    frame_data=image_b64,
    context={
        "location": "test_zone",
        "timestamp": datetime.now().isoformat()
    },
    tools_available=["object_detector"]
)

# Analyse (peut prendre 30s-1min sur Colab gratuit)
print("⏳ Analyse VLM en cours...")
try:
    analysis = await vlm.analyze_frame(request)
    
    print("🧠 Résultat VLM:")
    print(f"  Niveau de suspicion: {analysis.suspicion_level.value}")
    print(f"  Type d'action: {analysis.action_type.value}")
    print(f"  Confiance: {analysis.confidence:.3f}")
    print(f"  Description: {analysis.description}")
    
except Exception as e:
    print(f"⚠️ Erreur VLM (normal sur Colab gratuit): {e}")
    print("💡 Utilisez un modèle plus léger ou Colab Pro pour des tests complets")
```

### Demo 3: Pipeline Complet Simplifié

```python
# Version allégée du pipeline pour Colab
import asyncio
from src.core.types import SurveillanceEvent, DetectionStatus

class ColabSurveillanceDemo:
    def __init__(self):
        self.detector = YOLODetector(model_path="yolov8n.pt")
        self.detector.load_model()
    
    async def analyze_image(self, image_path):
        """Analyse simplifiée d'une image."""
        
        # Chargement image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        frame = Frame(
            image=image_rgb,
            timestamp=datetime.now(),
            frame_id=1,
            stream_id="colab_demo",
            width=image_rgb.shape[1],
            height=image_rgb.shape[0]
        )
        
        # Détection
        detections = self.detector.detect(frame)
        
        # Analyse simple basée sur les détections
        person_count = len([d for d in detections if d.class_name == "person"])
        bag_count = len([d for d in detections if "bag" in d.class_name.lower()])
        
        # Logique de suspicion simplifiée
        if person_count > 0 and bag_count > 0:
            suspicion = "MEDIUM"
            action = "suspicious_movement"
            confidence = 0.7
        elif person_count > 3:
            suspicion = "HIGH"
            action = "crowded_area"
            confidence = 0.8
        else:
            suspicion = "LOW"
            action = "normal_shopping"
            confidence = 0.5
        
        return {
            "detections": len(detections),
            "persons": person_count,
            "bags": bag_count,
            "suspicion_level": suspicion,
            "action_type": action,
            "confidence": confidence,
            "objects": [d.class_name for d in detections[:10]]
        }

# Test du pipeline simplifié
demo = ColabSurveillanceDemo()

# Test sur plusieurs images
test_images = ["test_image.jpg"]

for img_path in test_images:
    if os.path.exists(img_path):
        print(f"\n🔍 Analyse de {img_path}:")
        result = await demo.analyze_image(img_path)
        
        print(f"  📊 Détections: {result['detections']}")
        print(f"  👥 Personnes: {result['persons']}")
        print(f"  🎒 Sacs: {result['bags']}")
        print(f"  ⚠️  Suspicion: {result['suspicion_level']}")
        print(f"  🎯 Action: {result['action_type']}")
        print(f"  📈 Confiance: {result['confidence']:.3f}")
        print(f"  🏷️  Objets: {result['objects'][:5]}")
```

## 📊 Benchmarking Colab

### Test de Performance

```python
import time
import psutil
import matplotlib.pyplot as plt

def benchmark_colab():
    """Benchmark des performances sur Colab."""
    
    results = {
        "yolo_times": [],
        "memory_usage": [],
        "gpu_memory": []
    }
    
    print("🏃‍♂️ Benchmark en cours...")
    
    # Test YOLO sur 10 images
    for i in range(10):
        start_time = time.time()
        detections = detector.detect(frame)
        end_time = time.time()
        
        results["yolo_times"].append(end_time - start_time)
        results["memory_usage"].append(psutil.virtual_memory().percent)
        
        if torch.cuda.is_available():
            results["gpu_memory"].append(torch.cuda.memory_allocated() / 1e9)
        
        print(f"  Test {i+1}/10: {end_time - start_time:.3f}s")
    
    # Affichage des résultats
    print("\n📊 Résultats Benchmark:")
    print(f"  ⏱️  Temps YOLO moyen: {np.mean(results['yolo_times']):.3f}s")
    print(f"  ⏱️  Temps YOLO min: {np.min(results['yolo_times']):.3f}s")
    print(f"  ⏱️  Temps YOLO max: {np.max(results['yolo_times']):.3f}s")
    print(f"  🧠 Mémoire RAM moyenne: {np.mean(results['memory_usage']):.1f}%")
    
    if results["gpu_memory"]:
        print(f"  🎮 Mémoire GPU moyenne: {np.mean(results['gpu_memory']):.2f} GB")
    
    # Graphique des performances
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    axes[0,0].plot(results["yolo_times"])
    axes[0,0].set_title("Temps YOLO par itération")
    axes[0,0].set_ylabel("Temps (s)")
    
    axes[0,1].hist(results["yolo_times"], bins=5)
    axes[0,1].set_title("Distribution des temps YOLO")
    axes[0,1].set_xlabel("Temps (s)")
    
    axes[1,0].plot(results["memory_usage"])
    axes[1,0].set_title("Usage Mémoire RAM")
    axes[1,0].set_ylabel("Usage (%)")
    
    if results["gpu_memory"]:
        axes[1,1].plot(results["gpu_memory"])
        axes[1,1].set_title("Usage Mémoire GPU")
        axes[1,1].set_ylabel("Usage (GB)")
    else:
        axes[1,1].text(0.5, 0.5, "GPU non disponible", 
                      ha='center', va='center', transform=axes[1,1].transAxes)
    
    plt.tight_layout()
    plt.show()
    
    return results

# Lancement du benchmark
benchmark_results = benchmark_colab()
```

## 🎯 Tests Spécifiques Surveillance

### Test Anti-Faux Positifs

```python
# Simulation de scénarios de test
test_scenarios = [
    {
        "name": "Foule normale",
        "persons": 5,
        "bags": 2,
        "expected_suspicion": "LOW"
    },
    {
        "name": "Personne avec sac dans zone sensible",
        "persons": 1,
        "bags": 1,
        "expected_suspicion": "MEDIUM"
    },
    {
        "name": "Mouvement erratique",
        "persons": 1,
        "bags": 0,
        "expected_suspicion": "HIGH"
    }
]

def simulate_scenario(scenario):
    """Simulation d'un scénario de test."""
    print(f"\n🎬 Scénario: {scenario['name']}")
    
    # Ici, on simulerait le comportement selon le scénario
    # Dans un vrai test, on utiliserait de vraies images
    
    # Logique simplifiée pour la démo
    persons = scenario["persons"]
    bags = scenario["bags"]
    
    if persons > 0 and bags > 0:
        suspicion = "MEDIUM"
    elif persons > 3:
        suspicion = "HIGH"
    else:
        suspicion = "LOW"
    
    expected = scenario["expected_suspicion"]
    match = suspicion == expected
    
    print(f"  👥 Personnes: {persons}")
    print(f"  🎒 Sacs: {bags}")
    print(f"  ⚠️  Suspicion détectée: {suspicion}")
    print(f"  ✅ Suspicion attendue: {expected}")
    print(f"  🎯 Match: {'✅' if match else '❌'}")
    
    return match

# Test des scénarios
print("🧪 Tests Anti-Faux Positifs")
matches = [simulate_scenario(scenario) for scenario in test_scenarios]
accuracy = sum(matches) / len(matches) * 100

print(f"\n📊 Résultat Global:")
print(f"  🎯 Précision: {accuracy:.1f}%")
print(f"  ✅ Tests réussis: {sum(matches)}/{len(matches)}")
```

## 💾 Sauvegarde des Résultats

```python
# Sauvegarde des résultats de test
import json
from datetime import datetime

test_results = {
    "timestamp": datetime.now().isoformat(),
    "environment": "Google Colab",
    "gpu_available": torch.cuda.is_available(),
    "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "benchmark": benchmark_results,
    "scenario_tests": {
        "scenarios": test_scenarios,
        "accuracy": accuracy,
        "matches": matches
    }
}

# Sauvegarde en JSON
with open('/content/test_results.json', 'w') as f:
    json.dump(test_results, f, indent=2, default=str)

print("💾 Résultats sauvegardés dans /content/test_results.json")

# Téléchargement des résultats
from google.colab import files
files.download('/content/test_results.json')
```

## 🐛 Debug et Optimisation Colab

### Monitoring Mémoire

```python
import gc
import psutil

def monitor_memory():
    """Monitoring de la mémoire en temps réel."""
    
    # Mémoire RAM
    ram = psutil.virtual_memory()
    print(f"🧠 RAM: {ram.percent:.1f}% ({ram.used/1e9:.1f}/{ram.total/1e9:.1f} GB)")
    
    # Mémoire GPU
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.memory_allocated() / 1e9
        gpu_max = torch.cuda.max_memory_allocated() / 1e9
        gpu_cached = torch.cuda.memory_reserved() / 1e9
        
        print(f"🎮 GPU: {gpu_mem:.1f} GB utilisé, {gpu_max:.1f} GB max, {gpu_cached:.1f} GB cache")
    
    # Suggestions d'optimisation
    if ram.percent > 80:
        print("⚠️  RAM élevée - Redémarrer runtime recommandé")
    
    if torch.cuda.is_available() and gpu_mem > 12:
        print("⚠️  GPU mémoire élevée - Réduire batch size")

# Nettoyage mémoire
def cleanup_memory():
    """Nettoyage de la mémoire."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("🧹 Mémoire nettoyée")

# Monitoring initial
monitor_memory()
```

### Configuration Optimale Colab

```python
# Configuration optimisée pour Colab gratuit
COLAB_CONFIG = {
    "yolo_model": "yolov8n.pt",  # Modèle le plus léger
    "vlm_model": "microsoft/git-base-coco",  # VLM léger
    "batch_size": 1,  # Batch minimal
    "max_tokens": 128,  # Tokens limités
    "load_in_4bit": True,  # Quantization agressive
    "processing_fps": 5,  # FPS réduit
    "cache_ttl": 60,  # Cache court
}

print("⚙️ Configuration Colab:")
for key, value in COLAB_CONFIG.items():
    print(f"  {key}: {value}")
```

## 🎓 Exercices et Défis

### Défi 1: Optimisation de Performance

```python
# Mission: Améliorer les temps de détection
# Objectif: < 0.5s par image sur Colab

def optimization_challenge():
    """Défi d'optimisation pour étudiants."""
    
    print("🏆 DÉFI: Optimiser la détection sous 0.5s/image")
    print("💡 Indices:")
    print("  - Ajuster la résolution d'entrée")
    print("  - Optimiser les seuils de confiance")
    print("  - Utiliser le cache intelligemment")
    print("  - Paralléliser si possible")
    
    # TODO: Implémenter vos optimisations ici
    
    # Test de votre optimisation
    start_time = time.time()
    detections = detector.detect(frame)
    execution_time = time.time() - start_time
    
    if execution_time < 0.5:
        print(f"🎉 RÉUSSI! Temps: {execution_time:.3f}s")
    else:
        print(f"❌ Échec. Temps: {execution_time:.3f}s (objectif: < 0.5s)")

optimization_challenge()
```

### Défi 2: Détection de Patterns Suspects

```python
# Mission: Créer un détecteur de comportements suspects
# Utilisez votre créativité!

def create_behavior_detector():
    """Créez votre propre détecteur de comportements."""
    
    print("🕵️ DÉFI: Créer un détecteur de comportements suspects")
    print("📝 À implémenter:")
    print("  - Détection de mouvements erratiques")
    print("  - Identification de zones à risque")
    print("  - Corrélation entre objets et comportements")
    
    # TODO: Votre code ici
    
    pass

create_behavior_detector()
```

---

**🔗 Liens Utiles pour Colab:**

- [Notebook Demo Principal](https://colab.research.google.com/github/elfried-kinzoun/intelligent-surveillance-system/blob/main/notebooks/demo.ipynb)
- [Notebook Benchmark](https://colab.research.google.com/github/elfried-kinzoun/intelligent-surveillance-system/blob/main/notebooks/benchmark.ipynb)
- [Notebook Tests Unitaires](https://colab.research.google.com/github/elfried-kinzoun/intelligent-surveillance-system/blob/main/notebooks/tests.ipynb)

**Prochaine étape**: [Configuration Avancée](configuration.md)