# 🎬 Guide Complet des Tests Vidéo avec Kimi-VL

Guide complet pour tester l'architecture de surveillance intelligente sur des vidéos avec le modèle Kimi-VL uniquement.

## 🎯 Vue d'Ensemble

Ce guide vous permet de tester l'**architecture complète** du système sur des vidéos réelles :
- **Pipeline complet** : YOLO → Tracking → VLM Kimi-VL → Orchestration → Décisions
- **Tests individuels** ou **en lot (batch)**
- **Analyses de performance** détaillées
- **Rapports automatisés**

## 🏗️ Architecture Testée

```
📹 VIDÉO INPUT
     ⬇️
🔍 YOLO DETECTOR (détection objets/personnes)
     ⬇️  
🎯 BYTE TRACKER (tracking multi-objets)
     ⬇️
🤖 KIMI-VL ANALYSIS (analyse VLM uniquement - pas de fallback)
     ⬇️
⚙️ ORCHESTRATION (8 outils avancés selon mode)
     ⬇️
🚨 DECISIONS & ALERTES
```

## 📋 Prérequis

### Matériel Recommandé
- **RAM** : 8GB minimum (16GB recommandé)
- **VRAM** : 6GB minimum pour Kimi-VL + YOLO
- **Stockage** : 2GB libre pour résultats
- **CPU** : Multi-core pour traitement vidéo

### Logiciels
```bash
pip install opencv-python pillow numpy
# Dépendances déjà dans requirements.txt
```

### Vidéos de Test
- **Format** : MP4, AVI, MOV
- **Résolution** : 480p-1080p
- **Durée** : 10s-2min recommandé
- **Contenu** : Scènes de surveillance

## 🚀 Démarrage Rapide

### 1. Préparer les Datasets
```bash
# Créer la structure de datasets
python scripts/manage_video_datasets.py --create-structure

# Générer des exemples synthétiques  
python scripts/manage_video_datasets.py --generate-examples

# Valider les datasets
python scripts/manage_video_datasets.py --validate
```

### 2. Test Simple
```bash
# Test avec vidéo générée
python scripts/test_video_pipeline.py data/video_datasets/synthetic/generated_basic.mp4

# Test avec vos vidéos
python scripts/test_video_pipeline.py votre_video.mp4 --profile standard
```

### 3. Tests en Lot
```bash
# Tester toutes les catégories
python scripts/batch_video_test.py --all-categories

# Tester une catégorie spécifique
python scripts/batch_video_test.py --category surveillance_basic
```

## ⚙️ Profils de Configuration

### 🚀 **Fast** (~1.2s/frame)
```bash
--profile fast
```
- **Usage** : Tests rapides, développement
- **Outils** : 3 outils essentiels seulement
- **Résolution** : 480x360
- **Frames** : 150 max (5s)

### ⚖️ **Standard** (~2.5s/frame) 
```bash
--profile standard  # Défaut
```
- **Usage** : Tests complets standards
- **Outils** : 6 outils principaux
- **Résolution** : 640x480  
- **Frames** : 300 max (10s)

### 🔬 **Thorough** (~4.8s/frame)
```bash
--profile thorough
```
- **Usage** : Analyse forensique, recherche
- **Outils** : 8 outils complets
- **Résolution** : 720x576
- **Frames** : 600 max (20s)

### 🎮 **Demo** (~0.8s/frame)
```bash
--profile demo
```
- **Usage** : Démonstrations rapides
- **Résolution** : 320x240
- **Frames** : 90 max (3s)

## 🛠️ Scripts Disponibles

### 1. **test_video_pipeline.py** - Test Individuel
```bash
# Test basique
python scripts/test_video_pipeline.py video.mp4

# Test avec profil spécifique
python scripts/test_video_pipeline.py video.mp4 --profile thorough

# Test sans sauvegarde
python scripts/test_video_pipeline.py video.mp4 --no-save

# Test avec monitoring performance
python scripts/test_video_pipeline.py video.mp4 --performance
```

**Sortie** :
- Vidéo traitée avec overlay détections
- Rapport JSON détaillé
- Frames sauvées (optionnel)
- Métriques de performance

### 2. **batch_video_test.py** - Tests en Lot
```bash
# Toutes les catégories
python scripts/batch_video_test.py --all-categories

# Catégorie spécifique
python scripts/batch_video_test.py --category theft_scenarios

# Comparaison de profils
python scripts/batch_video_test.py --compare-profiles

# Profil spécifique sur toutes catégories
python scripts/batch_video_test.py --all-categories --profile thorough
```

**Sortie** :
- Rapport consolidé multi-vidéos
- Statistiques agrégées par catégorie
- Comparaison de performance
- Recommandations

### 3. **manage_video_datasets.py** - Gestion Datasets
```bash
# Créer structure
python scripts/manage_video_datasets.py --create-structure

# Lister datasets
python scripts/manage_video_datasets.py --list

# Valider vidéos
python scripts/manage_video_datasets.py --validate

# Générer exemples
python scripts/manage_video_datasets.py --generate-examples
```

## 📊 Datasets Organisés

```
data/video_datasets/
├── surveillance_basic/     # Scènes normales
│   ├── person_walking.mp4
│   ├── empty_room.mp4  
│   └── object_detection.mp4
├── theft_scenarios/        # Simulations de vol
│   ├── shoplifting.mp4
│   └── pickpocket.mp4
├── crowded_scenes/         # Scènes complexes  
│   ├── mall_crowd.mp4
│   └── multiple_persons.mp4
├── edge_cases/             # Cas limites
│   ├── low_light.mp4
│   └── occlusion_test.mp4
└── synthetic/              # Exemples générés
    ├── generated_basic.mp4
    └── generated_theft.mp4
```

## 📈 Métriques Mesurées

### Détection & Tracking
- **Détections totales** par vidéo
- **Personnes détectées** vs objets
- **Précision YOLO** (confiance moyenne)
- **Efficacité tracking** (IDs persistants)

### Analyse VLM (Kimi-VL)
- **Analyses VLM** effectuées 
- **Confiance moyenne** des analyses
- **Taux de succès** VLM
- **Temps d'inférence** moyen

### Performance Système  
- **FPS moyen** de traitement
- **Utilisation mémoire** (RAM/VRAM)
- **Temps total** de traitement
- **Erreurs** rencontrées

### Alertes Générées
- **Normal** : Situations standards
- **Attention** : Surveillance renforcée  
- **Critique** : Intervention requise

## 🎛️ Configuration Avancée

### Variables d'Environnement
```bash
# Forcer modèle spécifique
export VLM_MODEL=kimi-vl-a3b-thinking

# Device management  
export VLM_DEVICE=cuda  # ou cpu
export YOLO_DEVICE=cuda

# Optimisation mémoire
export LOAD_IN_4BIT=true
export ENABLE_FALLBACK=false  # Pas de fallback (important)

# Mode debug
export DEBUG=true
export LOG_LEVEL=DEBUG
```

### Personnalisation des Configs
```python
# config/video_test_config.py
VIDEO_TEST_CONFIGS["custom"] = VideoTestConfig(
    max_frames=500,
    frame_skip=1,
    resize_width=800,
    resize_height=600,
    confidence_threshold=0.4,
    save_processed_frames=True
)
```

## 📋 Exemple de Workflow Complet

### Scenario : Test Architecture sur Vidéos de Vol

```bash
# 1. Préparer l'environnement
python scripts/manage_video_datasets.py --create-structure
python scripts/manage_video_datasets.py --generate-examples

# 2. Placer vos vidéos de vol dans le bon dossier
cp mes_videos_vol/*.mp4 data/video_datasets/theft_scenarios/

# 3. Valider les datasets
python scripts/manage_video_datasets.py --validate

# 4. Test individuel pour vérifier
python scripts/test_video_pipeline.py data/video_datasets/theft_scenarios/shoplifting.mp4 --profile standard --performance

# 5. Tests en lot complets
python scripts/batch_video_test.py --category theft_scenarios --profile thorough

# 6. Comparaison de profils
python scripts/batch_video_test.py --compare-profiles

# 7. Test toutes catégories
python scripts/batch_video_test.py --all-categories --profile balanced
```

## 📊 Interprétation des Résultats

### Résultats Excellents ✅
- **FPS** : > 5 fps
- **Détections** : > 80% des objets/personnes
- **Confiance VLM** : > 0.7
- **Taux succès** : > 90%
- **Erreurs** : < 5%

### Résultats Acceptables ⚠️
- **FPS** : 2-5 fps  
- **Détections** : 60-80%
- **Confiance VLM** : 0.5-0.7
- **Taux succès** : 70-90%
- **Erreurs** : 5-15%

### Résultats Problématiques ❌
- **FPS** : < 2 fps
- **Détections** : < 60%  
- **Confiance VLM** : < 0.5
- **Taux succès** : < 70%
- **Erreurs** : > 15%

## 🔧 Dépannage

### Problème : Mémoire Insuffisante
```bash
# Solutions
export VLM_DEVICE=cpu        # Forcer CPU
export LOAD_IN_4BIT=true     # Quantization 4-bit
--profile fast               # Profil léger
```

### Problème : FPS Trop Lent
```bash
# Solutions  
--profile fast               # Profil rapide
export YOLO_MODEL=yolov11n.pt # YOLO nano
frame_skip=2                 # Une frame sur deux
```

### Problème : Kimi-VL Échoue
```bash
# Vérifications
python scripts/test_single_model.py --kimi --memory-stats
# Fallback vers modèle léger pour tests
export VLM_MODEL=microsoft/git-base-coco
```

### Problème : Pas de Détections
```bash
# Ajustements  
confidence_threshold=0.3     # Seuil plus bas
detection_threshold=0.4      # Tracking moins strict
```

## 📁 Structure des Résultats

```
data/video_outputs/
├── frames/                  # Frames sauvées
├── results/                 # Rapports JSON  
├── processed_videos/        # Vidéos avec overlay
└── batch_results/           # Rapports batch
    ├── batch_test_report_20240125_143022.json
    └── validation_report.json
```

### Format Rapport JSON
```json
{
  "summary": {
    "video_path": "video.mp4",
    "processed_frames": 300,
    "fps_average": 4.2,
    "detections_total": 145,
    "vlm_analyses": 30,
    "vlm_average_confidence": 0.78,
    "alerts_critique": 2
  },
  "frame_results": [...],
  "config": {...},
  "system_config": {...}
}
```

## 💡 Conseils d'Optimisation

### Pour Tests de Développement
```bash
# Configuration rapide et économe  
--profile fast --no-save
export ENABLE_FALLBACK=false
export DISABLE_MONITORING=true
```

### Pour Tests de Production
```bash
# Configuration complète
--profile thorough --performance  
export VLM_DEVICE=cuda
export LOAD_IN_4BIT=false  # Qualité maximale
```

### Pour Tests de Recherche
```bash
# Analyse exhaustive
--profile thorough
python scripts/batch_video_test.py --compare-profiles
# Analyse manuelle des rapports JSON
```

## 🎯 Cas d'Usage Spécifiques

### Test Robustesse Système
```bash
# Test conditions dégradées
python scripts/batch_video_test.py --category edge_cases --profile thorough
```

### Validation Performance Temps Réel
```bash  
# Test vitesse maximale
python scripts/batch_video_test.py --category synthetic --profile fast --performance
```

### Analyse Précision Détection
```bash
# Test avec vérité terrain
python scripts/test_video_pipeline.py annotated_video.mp4 --profile thorough
# Comparer avec annotations manuelles
```

---

**Guide Tests Vidéo v1.0.0**  
*Système de Surveillance Intelligente - Architecture Complète avec Kimi-VL*