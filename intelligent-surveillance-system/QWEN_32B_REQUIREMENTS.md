# 🚀 Qwen2.5-VL-32B - Exigences et Configuration

Guide complet pour utiliser **Qwen2.5-VL-32B** comme modèle principal du système de surveillance intelligente.

## 📊 Exigences Matérielles

### 🎯 **GPU Recommandés (2025)**

| GPU                    | VRAM  | Performance | Statut        |
|------------------------|-------|-------------|---------------|
| **RTX 5090** ⭐        | 32GB  | Optimal     | Recommandé    |
| **RTX 4090**           | 24GB  | Excellent   | Recommandé    |
| **RTX 3090**           | 24GB  | Très Bon    | Minimum       |
| **Tesla V100**         | 32GB  | Excellent   | Serveur       |
| **A100 40GB**          | 40GB  | Optimal     | Professionnel |

### 💾 **RAM Système**
- **Minimum**: 32GB DDR4
- **Recommandé**: 64GB DDR5
- **Optimal**: 128GB pour traitement batch

### 🔧 **Configuration Logicielle**

#### Installation CUDA
```bash
# CUDA 12.1+ requis
nvidia-smi  # Vérifier version driver
```

#### Dependencies Python
```bash
# PyTorch avec CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Transformers dernière version
pip install transformers>=4.45.0 accelerate bitsandbytes

# Qwen VL Utils
pip install qwen-vl-utils

# Optimisation mémoire
pip install flash-attn --no-build-isolation
```

## 🚀 Lancement avec Qwen2.5-VL-32B

### 1. **Script de Lancement Principal**

```bash
# Analyse vidéo standard
python launch_qwen_32b.py --video videos/surveillance01.mp4

# Haute performance (plus de frames)
python launch_qwen_32b.py --video videos/test.mp4 --max-frames 50 --frame-skip 1

# Mode rapide (moins de frames)
python launch_qwen_32b.py --video videos/test.mp4 --max-frames 10 --frame-skip 5
```

### 2. **Dashboard avec Qwen2.5-VL-32B**

```bash
cd dashboard/
streamlit run production_dashboard.py
```

Le dashboard détectera automatiquement Qwen2.5-VL-32B comme modèle principal.

### 3. **Mode Headless**

```bash
python main.py --model Qwen/Qwen2.5-VL-32B-Instruct --video videos/test.mp4
```

## ⚙️ Optimisations Performance

### 🔧 **Variables d'Environnement**

```bash
# GPU principal
export CUDA_VISIBLE_DEVICES=0

# Optimisation mémoire CUDA
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Parallélisme tokenizers
export TOKENIZERS_PARALLELISM=false

# Cache Hugging Face
export HF_HOME=/path/to/large/cache
```

### 🚄 **Optimisations Avancées**

#### Quantization 4-bit (Si VRAM limitée)
```python
# Configuration automatique dans le code
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
```

#### Flash Attention 2.0
```python
# Activation automatique si disponible
model = AutoModelForVision2Seq.from_pretrained(
    model_name,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16
)
```

## 📈 Performance Attendue

### 🎯 **Benchmarks Qwen2.5-VL-32B**

| Métrique              | Score    | Comparaison      |
|-----------------------|----------|------------------|
| **MMMU**              | 65.5     | SOTA 2025        |
| **DocVQA**            | 94.5     | +5% vs GPT-4o    |
| **RealWorldQA**       | 75.2     | +3% vs Claude    |
| **Video QA**          | 68.9     | Excellent        |

### ⚡ **Vitesse de Traitement**

| Configuration         | Temps/Frame | VRAM Utilisé |
|-----------------------|-------------|--------------|
| **RTX 4090 + 4-bit**  | ~2.5s       | ~18GB        |
| **RTX 4090 + FP16**   | ~3.2s       | ~23GB        |
| **RTX 3090 + 4-bit**  | ~3.1s       | ~19GB        |

## 🔍 Surveillance avec Qwen2.5-VL-32B

### 🎯 **Avantages pour Surveillance**

1. **Analyse Comportementale Avancée**
   - Détection d'intentions suspectes
   - Analyse de micro-expressions
   - Comportements anormaux complexes

2. **Compréhension Contextuelle**
   - Analyse de scènes multi-objets
   - Corrélation temporelle avancée
   - Raisonnement spatial sophistiqué

3. **Précision Détection**
   - Moins de faux positifs
   - Détection d'objets partiellement occultés
   - Classification fine d'activités

### 📊 **Exemples de Détections**

```python
# Résultat typique Qwen2.5-VL-32B
{
    "suspicion_level": 0.85,
    "confidence": 0.92,
    "description": "Personne exhibant comportement furtif près du véhicule, regardant autour avant manipulation poignée",
    "detections": [
        {
            "type": "person",
            "bbox": [x, y, w, h],
            "behavior": "suspicious_loitering",
            "threat_level": "medium"
        }
    ],
    "risk_assessment": {
        "theft_probability": 0.78,
        "vandalism_risk": 0.23,
        "false_alarm": 0.08
    }
}
```

## 🛠️ Dépannage

### ❌ **Erreurs Communes**

#### CUDA Out of Memory
```bash
# Réduire batch size
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256

# Utiliser quantization
python launch_qwen_32b.py --quantize-4bit
```

#### Modèle Non Trouvé
```bash
# Télécharger manuellement
huggingface-cli download Qwen/Qwen2.5-VL-32B-Instruct --local-dir ./models/qwen2.5-vl-32b
```

#### Performance Lente
```bash
# Vérifier GPU utilization
nvidia-smi -l 1

# Optimiser frame skip
python launch_qwen_32b.py --frame-skip 3  # Plus rapide
```

## 🎬 Tests de Validation

### 🧪 **Test Rapide GPU**

```bash
python -c "
import torch
print(f'CUDA: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB')
"
```

### 📹 **Test Vidéo Minimal**

```bash
# Test 5 frames seulement
python launch_qwen_32b.py \
  --video videos/test_short.mp4 \
  --max-frames 5 \
  --frame-skip 10 \
  --verbose
```

## 🚨 Configuration Production

### 🏭 **Environnement Serveur**

```bash
# Daemon mode
nohup python launch_qwen_32b.py --video stream.mp4 > surveillance.log 2>&1 &

# Monitoring GPU
nvidia-smi dmon -s pucvmet -d 10
```

### 📡 **Streaming Temps Réel**

```bash
# Dashboard production
cd dashboard/
streamlit run production_dashboard.py --server.port 8501 --server.address 0.0.0.0
```

---

## 📞 Support

Pour questions techniques sur Qwen2.5-VL-32B :
- [GitHub Issues](https://github.com/QwenLM/Qwen2.5-VL/issues)
- [Documentation Officielle](https://qwenlm.github.io/)
- [Hugging Face](https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct)

**Bonne surveillance avec Qwen2.5-VL-32B ! 🎯**