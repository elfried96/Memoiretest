# 🧪 Tests du Système de Surveillance Intelligente

Guide complet des tests organisés et structurés pour le système de surveillance.

## 📋 Structure des Tests

```
tests/
├── README.md                    # Ce guide
├── conftest.py                  # Configuration pytest commune
├── pytest.ini                  # Configuration pytest
├── unit/                        # Tests unitaires
│   ├── test_*_unit.py          # Tests des modules individuels
│   ├── test_vlm_unit.py        # Tests VLM
│   └── test_yolo_unit.py       # Tests YOLO
├── integration/                 # Tests d'intégration
│   ├── test_integration_*.py   # Tests d'intégration système
│   └── test_model_switching.py # Tests changement de modèles
└── performance/                 # Tests de performance
    └── test_performance_*.py   # Benchmarks et tests de performance
```

## 🚀 Exécution des Tests

### Tests Rapides (Unités)
```bash
# Tous les tests unitaires
pytest tests/unit/ -v

# Test spécifique
pytest tests/unit/test_vlm_unit.py -v

# Tests avec couverture
pytest tests/unit/ --cov=src --cov-report=term-missing
```

### Tests Complets (Intégration)
```bash
# Tests d'intégration
pytest tests/integration/ -v

# Test complet du système
pytest tests/integration/test_integration_complete.py -v
```

### Tests de Performance
```bash
# Benchmarks de performance
pytest tests/performance/ -v

# Tests avec marqueurs
pytest -m "not slow" -v  # Éviter les tests lents
pytest -m "gpu" -v       # Tests GPU uniquement
```

### Tests par Catégorie
```bash
# Tests par marqueurs
pytest -m unit -v           # Tests unitaires seulement
pytest -m integration -v    # Tests d'intégration seulement
pytest -m "not gpu" -v      # Éviter les tests GPU
```

## ⚙️ Configuration

### Variables d'Environnement de Test
```bash
# Configuration pour tests
export TEST_MODE=true
export LOG_LEVEL=DEBUG
export VLM_MODEL=microsoft/git-base-coco  # Modèle léger pour tests
export YOLO_MODEL=yolov11n.pt
export DISABLE_GPU=true  # Pour forcer CPU en tests
```

### Profils de Configuration
```python
# Utilisation du profil de test
from config.app_config import load_config
config = load_config("testing")  # Charge configuration optimisée tests
```

## 📊 Marqueurs de Tests

Les tests sont organisés avec des marqueurs pytest :

- `@pytest.mark.unit` - Tests unitaires rapides
- `@pytest.mark.integration` - Tests d'intégration complets
- `@pytest.mark.slow` - Tests longs (> 30s)
- `@pytest.mark.gpu` - Tests nécessitant GPU
- `@pytest.mark.performance` - Tests de performance

## 🔍 Tests Disponibles

### Tests Unitaires (`tests/unit/`)

| Test | Description | Durée |
|------|-------------|-------|
| `test_vlm_unit.py` | Tests des modèles VLM | ~15s |
| `test_yolo_unit.py` | Tests détection YOLO | ~10s |
| `test_advanced_tools_unit.py` | Tests outils avancés | ~20s |

### Tests d'Intégration (`tests/integration/`)

| Test | Description | Durée |
|------|-------------|-------|
| `test_integration_complete.py` | Pipeline complet | ~45s |
| `test_model_switching.py` | Changement modèles | ~30s |
| `test_integration_progressive.py` | Tests progressifs | ~60s |

### Tests de Performance (`tests/performance/`)

| Test | Description | Durée |
|------|-------------|-------|
| `test_performance_benchmarks.py` | Benchmarks système | ~120s |

## 🛠️ Outils de Test

### Coverage (Couverture de Code)
```bash
# Génération rapport de couverture
pytest --cov=src --cov-report=html
# Rapport dans htmlcov/index.html
```

### Profiling (Performance)
```bash
# Profiling des tests lents
pytest --profile-svg
```

### Tests Parallèles
```bash
# Exécution parallèle avec pytest-xdist
pip install pytest-xdist
pytest -n 4  # 4 processus parallèles
```

## 🚨 Tests d'Échec Communs

### Problème : Modèles non téléchargés
```bash
# Solution : Pré-télécharger les modèles
python -c "
from transformers import AutoModel
AutoModel.from_pretrained('microsoft/git-base-coco')
"
```

### Problème : GPU non disponible
```bash
# Solution : Forcer CPU pour tests
export CUDA_VISIBLE_DEVICES=""
pytest tests/ -v
```

### Problème : Timeout sur tests longs
```bash
# Solution : Augmenter timeout ou éviter tests lents
pytest -m "not slow" -v
```

## 📈 Métriques de Qualité

### Objectifs de Couverture
- **Code Coverage** : > 80%
- **Tests unitaires** : > 95% des fonctions critiques
- **Tests d'intégration** : Tous les workflows principaux

### Critères de Réussite
- ✅ Tous les tests unitaires passent
- ✅ Tests d'intégration principaux passent  
- ✅ Pas de régression de performance > 20%
- ✅ Pas de fuites mémoire détectées

## 🔧 Développement de Tests

### Ajouter un Nouveau Test
```python
import pytest
from config.app_config import load_config

@pytest.mark.unit
def test_nouvelle_fonctionnalite():
    """Test de la nouvelle fonctionnalité."""
    config = load_config("testing")
    # Votre test ici
    assert True
```

### Convention de Nommage
- `test_<module>_unit.py` - Tests unitaires
- `test_<feature>_integration.py` - Tests d'intégration
- `test_<component>_performance.py` - Tests de performance

## 🎯 Commandes Utiles

```bash
# Tests complets avec rapport
pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

# Tests rapides seulement
pytest tests/unit/ -x -v  # Arrêt au premier échec

# Tests avec logs détaillés
pytest tests/ -v -s --log-cli-level=DEBUG

# Nettoyage après tests
find . -name "*.pyc" -delete
find . -name "__pycache__" -delete
```

---

**Guide des Tests v1.0.0**  
*Système de Surveillance Intelligente*