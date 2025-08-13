# 🎯 Système de Surveillance Intelligente Multi-VLM

Système de surveillance avancé avec **support multi-modèles VLM** : **KIM**, **LLaVA** et **Qwen2-VL** + **8 outils avancés** intégrés.

## ⭐ Fonctionnalités Principales

### 🧠 **Support Multi-VLM Dynamique**
- **KIM** (Microsoft) - Modèle principal optimisé surveillance
- **LLaVA-NeXT** - Modèle éprouvé et stable  
- **Qwen2-VL** - Excellence en raisonnement visuel
- **Switch à chaud** entre modèles sans redémarrage

### 🛠️ **8 Outils Avancés Intégrés**
- `SAM2Segmentator` - Segmentation précise avec SAM2
- `DinoV2FeatureExtractor` - Features visuelles robustes
- `OpenPoseEstimator` - Analyse posturale et comportementale
- `TrajectoryAnalyzer` - Patterns de mouvement sophistiqués  
- `MultiModalFusion` - Fusion intelligente des données
- `TemporalTransformer` - Analyse temporelle avancée
- `AdversarialDetector` - Protection contre attaques
- `DomainAdapter` - Adaptation multi-environnements

### 🎮 **Orchestration Intelligente** 
- **3 modes** : `FAST`, `BALANCED`, `THOROUGH`
- **Tool-calling** natif optimisé par modèle
- **Traitement batch** avec contrôle de concurrence
- **Fallbacks** automatiques et robustes

## 🏗️ Architecture Modernisée

```
src/
├── core/
│   ├── vlm/
│   │   ├── dynamic_model.py      # VLM multi-modèles
│   │   ├── model_registry.py     # Registre KIM/LLaVA/Qwen
│   │   ├── prompt_builder.py     # Prompts optimisés
│   │   ├── response_parser.py    # Parsing intelligent  
│   │   └── tools_integration.py  # Gestionnaire 8 outils
│   ├── orchestrator/
│   │   └── vlm_orchestrator.py   # Orchestrateur moderne
│   └── types.py                  # Types partagés
├── advanced_tools/               # 8 outils avancés
│   ├── sam2_segmentation.py
│   ├── dino_features.py
│   ├── pose_estimation.py
│   ├── trajectory_analyzer.py
│   ├── multimodal_fusion.py
│   ├── temporal_transformer.py
│   ├── adversarial_detector.py
│   └── domain_adapter.py
└── detection/                    # YOLO et tracking
    ├── yolo/
    └── tracking/
```

## 🚀 Installation

```bash
# Créer l'environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows

# Installer les dépendances
pip install -r requirements.txt

# Configuration GPU (optionnel)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 📊 Performance Attendue

| Métrique | Objectif | Baseline |
|----------|----------|----------|
| Précision | > 90% | 75% |
| Faux positifs | < 3% | 15-20% |
| Latence | < 1.5s | 3-5s |
| Débit | > 10 flux | 2-3 flux |

## 📝 Licence

Projet académique - Mémoire de Licence en IA