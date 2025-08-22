# 📊 Tests Mémoire - Système de Surveillance IA

## 🎯 **Objectif**

Suite de tests complète pour validation mémoire du système de surveillance intelligente avec architecture multi-VLM et orchestration adaptative.

## 🏗️ **Architecture Tests**

### **1. Tests Unitaires Isolés**
```
tests_memoire/
├── test_vlm_unit.py           # Tests VLM isolés (Kimi-VL, Qwen2-VL)
├── test_yolo_unit.py          # Tests YOLO isolés (performance, précision)
└── test_advanced_tools_unit.py # Tests 8 outils avancés individuels
```

### **2. Tests d'Intégration Progressive**
```
├── test_integration_progressive.py  # Pipeline complet niveau par niveau
```

### **3. Benchmarks Performance**
```
├── test_performance_benchmarks.py   # Métriques quantitatives détaillées
```

### **4. Configuration & Infrastructure**
```
├── conftest.py              # Configuration pytest & fixtures
├── pytest.ini              # Configuration pytest
├── run_gpu_tests.py         # Script exécution GPU optimisé
└── README_TESTS_MEMOIRE.md  # Documentation (ce fichier)
```

## 🚀 **Exécution Tests**

### **Environnement Standard**
```bash
# Tests complets
python tests_memoire/run_gpu_tests.py

# Tests unitaires uniquement
python tests_memoire/run_gpu_tests.py --unit-only

# Tests intégration uniquement  
python tests_memoire/run_gpu_tests.py --integration-only

# Tests performance uniquement
python tests_memoire/run_gpu_tests.py --performance-only

# Mode rapide (stop au premier échec)
python tests_memoire/run_gpu_tests.py --quick
```

### **Environnement GPU**
```bash
# Tests GPU spécifiques
python tests_memoire/run_gpu_tests.py --gpu-only

# Vérification GPU
python -c "import torch; print(f'GPU: {torch.cuda.is_available()}')"
```

### **Pytest Direct**
```bash
# Tests avec markers
pytest tests_memoire/ -m "unit"
pytest tests_memoire/ -m "integration" 
pytest tests_memoire/ -m "performance"
pytest tests_memoire/ -m "gpu"

# Tests avec coverage
pytest tests_memoire/ --cov=src --cov-report=html
```

## 📋 **Tests Détaillés**

### **🧠 1. Tests VLM Unitaires (`test_vlm_unit.py`)**

| Test | Description | Métrique Validée |
|------|-------------|-----------------|
| `test_model_loading_performance` | Temps chargement modèles | <30s pour Kimi-VL |
| `test_model_switching_performance` | Switch entre modèles | <45s Kimi→Qwen |
| `test_inference_performance_isolated` | Inférence VLM pure | <3s par inférence |
| `test_fallback_mechanism` | Mécanisme fallback | Robustesse système |
| `test_memory_management` | Gestion mémoire GPU/CPU | <8GB croissance |
| `test_model_registry_validation` | Validation registry | Cohérence modèles |

### **🎯 2. Tests YOLO Unitaires (`test_yolo_unit.py`)**

| Test | Description | Métrique Validée |
|------|-------------|-----------------|
| `test_yolo_initialization` | Initialisation YOLO | Classes COCO disponibles |
| `test_detection_performance_single` | Détection image unique | <1s par image |
| `test_detection_batch_performance` | Détection batch | Speedup batch vs individuel |
| `test_detection_accuracy_validation` | Précision détection | Détection personne simulée |
| `test_resolution_robustness` | Robustesse résolutions | 320x240 → 1920x1080 |
| `test_confidence_threshold_impact` | Impact seuils confiance | Cohérence filtrage |

### **🛠️ 3. Tests Outils Avancés (`test_advanced_tools_unit.py`)**

| Outil | Test Performance | Métrique Cible |
|-------|-----------------|----------------|
| **SAM2Segmentator** | `test_sam2_segmentation_performance` | <10s segmentation |
| **DinoV2Features** | `test_dino_feature_extraction` | <2s extraction |
| **OpenPoseEstimator** | `test_pose_estimation` | <1s estimation |
| **TrajectoryAnalyzer** | `test_trajectory_analysis` | <0.5s analyse |
| **MultiModalFusion** | `test_multimodal_fusion` | <1s fusion |
| **TemporalTransformer** | `test_temporal_analysis` | <1s séquence |
| **AdversarialDetector** | `test_adversarial_detection` | <2s détection |
| **DomainAdapter** | `test_domain_adaptation` | <1s adaptation |

### **🔄 4. Tests Intégration Progressive (`test_integration_progressive.py`)**

| Niveau | Test | Validation |
|--------|------|-----------|
| **Niveau 1** | VLM + YOLO | Intégration détection + analyse |
| **Niveau 2** | Orchestrateur Moderne | 3 modes (FAST/BALANCED/THOROUGH) |
| **Niveau 3** | Orchestrateur Adaptatif | Apprentissage et optimisation |
| **Niveau 4** | Pipeline Complet | End-to-end avec tous composants |
| **Niveau 5** | Tests de Charge | Résistance charge concurrente |

### **⚡ 5. Benchmarks Performance (`test_performance_benchmarks.py`)**

| Benchmark | Objectif | Métriques Collectées |
|-----------|----------|---------------------|
| **VLM Comparatif** | Kimi-VL vs Qwen2-VL | Temps, précision, mémoire |
| **YOLO Scalabilité** | Performance vs résolution | FPS, pixels/seconde |
| **Modes Orchestration** | FAST vs BALANCED vs THOROUGH | Trade-off vitesse/précision |
| **Stress Test** | Charge concurrente | Throughput, taux erreur |
| **GPU Memory** | Efficacité mémoire GPU | MB/image, batch scaling |

## 📊 **Métriques Collectées**

### **Performance Temporelle**
- **Temps chargement modèle** : <30s (Kimi-VL), <45s (Qwen2-VL)
- **Temps inférence VLM** : <3s par analyse  
- **Temps détection YOLO** : <1s par image
- **Temps orchestration** : <5s pipeline complet
- **Throughput système** : >1 FPS en mode FAST

### **Utilisation Ressources**
- **Mémoire GPU** : Monitoring allocation/libération
- **Mémoire CPU** : <8GB croissance par modèle
- **Scalabilité GPU** : Efficiency vs batch size
- **Concurrence** : Support 2-8 streams simultanés

### **Qualité & Robustesse**
- **Taux succès** : >95% analyses réussies
- **Précision détection** : Validation objets simulés
- **Fallback** : Switch automatique si échec modèle
- **Adaptabilité** : Apprentissage patterns contextuels

## 🎯 **Validation Mémoire**

### **Hypothèses Testées**
1. **Performance VLM** : Kimi-VL-A3B-Thinking plus rapide que Qwen2-VL-7B
2. **Scalabilité Orchestration** : Mode FAST < BALANCED < THOROUGH (temps)
3. **Efficacité GPU** : Accélération significative vs CPU
4. **Apprentissage Adaptatif** : Amélioration performance avec usage
5. **Robustesse Charge** : Maintien performance sous stress

### **Données pour Mémoire**
- **Tableaux comparatifs** : Performance modèles VLM
- **Graphiques scalabilité** : YOLO selon résolution
- **Trade-offs** : Vitesse vs précision orchestration
- **Courbes apprentissage** : Optimisation adaptative
- **Métriques système** : Utilisation GPU/CPU/Mémoire

## 🛠️ **Configuration Environnement**

### **Prérequis**
```bash
# Python packages
pip install pytest pytest-html pytest-cov pytest-asyncio
pip install torch torchvision  # GPU version
pip install opencv-python pillow numpy

# Modèles (téléchargement automatique)
# - YOLO: yolov8n.pt (~6MB)
# - VLM: Chargement via transformers
```

### **Environnement GPU**
```bash
# Vérification CUDA
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# Configuration optimale
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
export CUDA_LAUNCH_BLOCKING=0
```

### **Structure Rapports**
```
tests_memoire/
├── reports/
│   ├── test_report_<timestamp>.html      # Rapport HTML détaillé
│   ├── junit_<timestamp>.xml             # Rapport JUnit
│   ├── coverage_html/                    # Coverage HTML
│   ├── performance_report_<timestamp>.json # Métriques JSON
│   └── metrics_results.json              # Métriques agrégées
├── logs/
│   └── test_results.log                  # Logs détaillés
└── data/
    └── [données test générées]
```

## 🚀 **Usage Avancé**

### **Tests Personnalisés**
```python
# Ajout de nouveaux benchmarks
@pytest.mark.performance
def test_custom_benchmark(self, metrics_collector):
    # Votre test personnalisé
    metrics_collector.add_performance_data("custom", results)
```

### **Métriques Personnalisées**
```python
# Dans conftest.py
@pytest.fixture
def custom_metrics():
    # Collecteur personnalisé
    pass
```

### **Configuration GPU**
```python
# Tests conditionnels GPU
@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_gpu_feature():
    pass
```

## 📈 **Interprétation Résultats**

### **Succès Test**
- ✅ **Exit code 0** : Tous tests passés
- 📊 **Coverage >70%** : Code bien testé
- 🚀 **Performance targets** : Métriques dans limites

### **Échec Test**
- ❌ **Exit code ≠ 0** : Tests échoués
- 🐛 **Logs détaillés** : Analyse erreurs
- 📉 **Performance dégradée** : Optimisation requise

### **Rapports Mémoire**
- **Tableaux performance** : Temps, mémoire, throughput
- **Graphiques comparatifs** : Modèles, modes, résolutions
- **Analyses statistiques** : Moyennes, écarts-types, percentiles
- **Recommandations** : Optimisations suggérées

---

## 🎓 **Pour le Mémoire**

Ces tests fournissent les **données quantitatives** nécessaires pour :

1. **Validation Architecture** : Performance composants individuels
2. **Comparaison Modèles** : Kimi-VL vs Qwen2-VL benchmarkés
3. **Efficacité Orchestration** : Trade-offs modes opérationnels
4. **Scalabilité Système** : Comportement sous charge
5. **Optimisation GPU** : Accélération matérielle mesurée

**Résultats directement utilisables** pour tableaux, graphiques et analyses dans le mémoire ! 📊✨