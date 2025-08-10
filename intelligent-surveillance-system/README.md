# Système de Surveillance Intelligente Multimodale

Système de surveillance basé sur des modèles Vision-Language avec capacités d'orchestration d'outils pour la prévention du vol en grande distribution.

## Objectifs

- **Taux de faux positifs** < 3%
- **Précision de détection** > 90%
- **Traitement temps réel** < 1.5s
- **Support multi-flux** > 10 caméras simultanées

## 🏗️ Architecture

```
├── src/
│   ├── core/              # Modules centraux
│   │   ├── vlm/          # Vision-Language Model
│   │   ├── orchestrator/ # Orchestration d'outils
│   │   └── pipeline/     # Pipeline de traitement
│   ├── detection/        # Détection et suivi
│   │   ├── yolo/        # YOLO v8 détection
│   │   └── tracking/    # DeepSORT/ByteTrack
│   ├── validation/      # Anti-faux positifs
│   ├── monitoring/      # Interface et alertes
│   └── utils/          # Utilitaires communs
├── config/             # Configurations
├── data/              # Données et modèles
├── tests/             # Tests unitaires
└── docker/            # Conteneurisation
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