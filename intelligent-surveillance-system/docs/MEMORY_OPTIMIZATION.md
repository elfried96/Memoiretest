# 💾 Guide d'Optimisation Mémoire

Guide pour économiser la mémoire lors des tests et de l'utilisation du système de surveillance.

## 🎯 Problème de Mémoire

Le système utilise des modèles VLM volumineux (Kimi-VL, Qwen2-VL) qui peuvent consommer beaucoup de mémoire RAM/VRAM, surtout avec les modèles de fallback chargés simultanément.

## ✅ Solutions Implémentées

### 1. **Configuration Sans Fallback par Défaut**

```python
# Configuration par défaut (économe en mémoire)
vlm_config = VLMConfig(
    primary_model="kimi-vl-a3b-thinking",
    fallback_models=[],  # Pas de modèles de fallback
    enable_fallback=False,
    load_in_4bit=True  # Quantization pour économiser mémoire
)
```

### 2. **Profils de Configuration Dédiés**

```bash
# Tests avec modèle léger
ENV=testing python run_system.py

# Tests avec Kimi-VL uniquement  
ENV=testing_kimi python run_system.py

# Tests avec Qwen2-VL uniquement
ENV=testing_qwen python run_system.py
```

### 3. **Nettoyage Mémoire Optimisé**

Le système inclut un nettoyage mémoire avancé :
- Garbage collection Python
- Nettoyage cache GPU/CUDA
- Synchronisation GPU
- Libération explicite des tensors

## 🚀 Utilisation Optimisée

### Tests avec Modèle Unique

```bash
# Test Kimi-VL seulement
python scripts/test_single_model.py --kimi --memory-stats

# Test Qwen2-VL seulement  
python scripts/test_single_model.py --qwen --memory-stats

# Test modèle léger
python scripts/test_single_model.py --git-base --memory-stats
```

### Tests Automatisés Optimisés

```bash
# Tests unitaires (modèle léger par défaut)
python scripts/run_tests.py --unit

# Tests complets sans modèles lourds
ENV=testing python scripts/run_tests.py --all
```

### Système Principal avec Kimi-VL Uniquement

```bash
# Mode développement (Kimi-VL uniquement)
ENV=development python run_system.py

# Mode production (Kimi-VL optimisé)
ENV=production python run_system.py
```

## 📊 Monitoring Mémoire

### Script de Test avec Stats Mémoire

```bash
python scripts/test_single_model.py --kimi --memory-stats
```

Affiche :
- Utilisation RAM (RSS/VMS)
- Mémoire GPU allouée/réservée
- Avant/après chargement

### Variables d'Environnement

```bash
# Forcer mode économe
export ENABLE_FALLBACK=false
export VLM_MODEL=microsoft/git-base-coco  # Modèle léger
export LOAD_IN_4BIT=true

# Monitoring mémoire
export DISABLE_MONITORING=true  # Économise ressources
```

## ⚙️ Configuration par Cas d'Usage

### 🧪 Tests Unitaires (Mémoire Minimale)
```python
config = load_config("testing")
# → Git-base, pas de fallback, quantization 4-bit
```

### 🔬 Tests avec Kimi-VL
```python  
config = load_config("testing_kimi")
# → Kimi-VL uniquement, pas de fallback
```

### 🌟 Tests avec Qwen2-VL
```python
config = load_config("testing_qwen")
# → Qwen2-VL uniquement, pas de fallback
```

### 🚀 Production
```python
config = load_config("production")
# → Kimi-VL, pas de fallback, qualité optimale
```

## 🔧 Paramètres d'Optimisation

### Quantization 4-bit
```python
vlm_config.load_in_4bit = True  # Réduit utilisation mémoire ~50%
```

### Device Management
```python
vlm_config.device = DeviceType.CPU  # Force CPU si GPU saturé
```

### Concurrent Tools
```python
orchestration_config.max_concurrent_tools = 1  # Limite parallélisme
```

## 📈 Métriques de Performance

### Utilisation Mémoire Typique

| Configuration | RAM | VRAM | Temps Chargement |
|---------------|-----|------|------------------|
| Git-base + CPU | ~2GB | 0GB | ~5s |
| Git-base + GPU | ~1GB | ~2GB | ~3s |
| Kimi-VL + CPU | ~4GB | 0GB | ~15s |
| Kimi-VL + GPU | ~2GB | ~4GB | ~8s |
| Qwen2-VL + GPU | ~3GB | ~6GB | ~12s |

### Avec Fallback (NON Recommandé)
| Configuration | RAM | VRAM | Temps Chargement |
|---------------|-----|------|------------------|
| Kimi + Qwen fallback | ~6GB | ~10GB | ~25s |

## ⚠️ Bonnes Pratiques

### ✅ DO
- Utilisez un seul modèle à la fois
- Activez la quantization 4-bit
- Libérez la mémoire après tests
- Utilisez les profils de configuration appropriés

### ❌ DON'T  
- Ne chargez pas multiple modèles simultanément
- N'utilisez pas de fallback en développement
- Ne gardez pas les modèles en mémoire inutilement
- N'oubliez pas le nettoyage GPU

## 🛠️ Dépannage

### Erreur "Out of Memory"
```bash
# Solution 1: Forcer CPU
export VLM_DEVICE=cpu

# Solution 2: Modèle plus léger
export VLM_MODEL=microsoft/git-base-coco

# Solution 3: Quantization 4-bit
export LOAD_IN_4BIT=true
```

### Système Lent après Tests
```python
# Nettoyage manuel
vlm._unload_current_model()
torch.cuda.empty_cache()
```

### Monitoring Continu
```bash
# Surveillance mémoire en temps réel
watch -n 1 'nvidia-smi && free -h'
```

---

**Optimisation Mémoire v1.0.0**  
*Système de Surveillance Intelligente - Tests Efficaces*