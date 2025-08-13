# 🚀 Guide de Démarrage Rapide - Multi-VLM

Guide pour commencer rapidement avec le système de surveillance **KIM + LLaVA + Qwen2-VL**.

## ⚡ Installation Express

```bash
# 1. Installation des dépendances
uv sync
# ou
pip install -r requirements.txt

# 2. Test du système  
python tests/test_model_switching.py
```

## 🎯 Utilisation Basique

### **1. VLM Simple (LLaVA)**

```python
from src.core.vlm.dynamic_model import DynamicVisionLanguageModel
from src.core.types import AnalysisRequest

# Initialisation avec LLaVA (stable)
vlm = DynamicVisionLanguageModel(
    default_model="llava-v1.6-mistral-7b",
    enable_fallback=True
)

# Chargement
await vlm.load_model()

# Analyse basique
request = AnalysisRequest(
    frame_data="<base64_image>",
    context={"location": "Store", "camera": "CAM_01"},
    tools_available=["dino_features", "pose_estimator"]
)

result = await vlm.analyze_with_tools(request)
print(f"Suspicion: {result.suspicion_level.value}")
```

### **2. Switch vers KIM (Recommandé)**

```python
# Switch vers KIM pour surveillance optimisée
success = await vlm.switch_model("kim-7b-instruct")

if success:
    result = await vlm.analyze_with_tools(request)
    print(f"Analyse KIM: {result.description}")
else:
    print("KIM indisponible, LLaVA utilisé en fallback")
```

### **3. Switch vers Qwen2-VL (Raisonnement)**

```python
# Switch vers Qwen2-VL pour raisonnement avancé
await vlm.switch_model("qwen2-vl-7b-instruct")

# Analyse avec plus d'outils pour raisonnement complexe
request.tools_available = [
    "sam2_segmentator", "dino_features", "pose_estimator",
    "trajectory_analyzer", "multimodal_fusion"
]

result = await vlm.analyze_with_tools(request)
```

## 🎮 Orchestrateur Moderne

### **Mode Rapide (FAST)**

```python
from src.core.orchestrator.vlm_orchestrator import (
    ModernVLMOrchestrator, OrchestrationConfig, OrchestrationMode
)

# Configuration rapide
config = OrchestrationConfig(
    mode=OrchestrationMode.FAST,  # 3 outils essentiels
    enable_advanced_tools=True
)

orchestrator = ModernVLMOrchestrator(
    vlm_model_name="llava-v1.6-mistral-7b",  # Stable et rapide
    config=config
)

# Analyse rapide (~1.2s)
result = await orchestrator.analyze_surveillance_frame(
    frame_data=image_base64,
    detections=yolo_detections,
    context={"location": "Aisle 1"}
)
```

### **Mode Complet (THOROUGH)**

```python
# Configuration complète avec KIM
config = OrchestrationConfig(
    mode=OrchestrationMode.THOROUGH,  # 8 outils complets
    enable_advanced_tools=True
)

orchestrator = ModernVLMOrchestrator(
    vlm_model_name="kim-7b-instruct",  # KIM pour performance
    config=config
)

# Analyse complète (~4.8s)
result = await orchestrator.analyze_surveillance_frame(
    frame_data=image_base64,
    detections=yolo_detections,
    context={"location": "High Security Zone"}
)
```

## 📊 Comparaison des Modèles

### **Quand utiliser chaque modèle ?**

| Situation | Modèle Recommandé | Raison |
|-----------|------------------|--------|
| 🎯 **Surveillance standard** | `kim-7b-instruct` | Optimisé pour surveillance |
| 🛡️ **Système stable/prod** | `llava-v1.6-mistral-7b` | Très stable, fallback fiable |
| 🧠 **Analyse complexe** | `qwen2-vl-7b-instruct` | Excellence en raisonnement |
| 🚀 **Performance max** | `kim-14b-instruct` | Version haute performance |
| 🏆 **Flagship** | `qwen2-vl-72b-instruct` | Performance ultime (GPU++) |

### **Test de tous les modèles**

```python
models = [
    "kim-7b-instruct",
    "llava-v1.6-mistral-7b", 
    "qwen2-vl-7b-instruct"
]

for model_id in models:
    print(f"Test {model_id}...")
    success = await vlm.switch_model(model_id)
    
    if success:
        result = await vlm.analyze_with_tools(request)
        print(f"  ✅ Suspicion: {result.suspicion_level.value}")
    else:
        print(f"  ❌ Modèle indisponible")
```

## 🧪 Tests Recommandés

### **1. Test Système Complet**
```bash
python tests/test_integration_complete.py
```
- Orchestration 3 modes
- 8 outils avancés
- Batch processing

### **2. Test Multi-VLM**
```bash
python tests/test_model_switching.py
```
- Switching KIM/LLaVA/Qwen
- Comparaisons performance
- Fallbacks automatiques

### **3. Tests Individuels**
```bash
python tests/test_sam2_segmentator.py
python tests/test_dino_features.py
python tests/test_pose_estimation.py
# etc.
```

## ⚙️ Configuration Recommandée

### **Production (Stabilité)**
```python
config = OrchestrationConfig(
    mode=OrchestrationMode.BALANCED,
    max_concurrent_tools=4,
    timeout_seconds=30,
    confidence_threshold=0.8,
    enable_advanced_tools=True
)

vlm = DynamicVisionLanguageModel(
    default_model="llava-v1.6-mistral-7b",  # Stable
    enable_fallback=True
)
```

### **Recherche (Performance)**
```python
config = OrchestrationConfig(
    mode=OrchestrationMode.THOROUGH,
    max_concurrent_tools=6,
    timeout_seconds=60,
    confidence_threshold=0.7,
    enable_advanced_tools=True
)

vlm = DynamicVisionLanguageModel(
    default_model="kim-7b-instruct",  # Principal
    enable_fallback=True
)
```

### **Edge/Mobile (Économique)**
```python
config = OrchestrationConfig(
    mode=OrchestrationMode.FAST,
    max_concurrent_tools=2,
    timeout_seconds=15,
    confidence_threshold=0.6,
    enable_advanced_tools=True
)

vlm = DynamicVisionLanguageModel(
    default_model="llava-v1.6-mistral-7b",
    enable_fallback=True
)
```

## 🔧 Dépannage Rapide

### **KIM Indisponible**
```python
# Le système utilise automatiquement LLaVA en fallback
# KIM nécessite possiblement une version spécialisée
print("KIM non disponible, utilisation de LLaVA")
```

### **GPU Insuffisant**
```python
# Pour Qwen2-VL-72B ou modèles volumineux
# Utiliser les versions 7B à la place
await vlm.switch_model("qwen2-vl-7b-instruct")  # Au lieu de 72B
```

### **Erreurs Dépendances**
```bash
# Réinstallation complète
uv sync
pip install torch torchvision transformers
```

## 🎯 Prochaines Étapes

1. **Explorez** les différents modèles avec vos données
2. **Testez** les 3 modes d'orchestration
3. **Optimisez** la configuration selon vos besoins
4. **Intégrez** dans votre pipeline de production

## 💡 Conseils Pro

- **Commencez** avec LLaVA pour la stabilité
- **Utilisez** KIM pour la surveillance optimisée  
- **Testez** Qwen2-VL pour des analyses complexes
- **Activez** les fallbacks automatiques
- **Surveillez** les performances avec `get_system_status()`

---

**🚀 Votre système multi-VLM est prêt ! Commencez par un test simple avec LLaVA puis explorez KIM et Qwen2-VL !**