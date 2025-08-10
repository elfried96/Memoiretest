# Formats de Données d'Entraînement - Surveillance Intelligente

Ce document définit les formats de données standardisés pour l'entraînement des modèles du système de surveillance.

## 📁 Structure des Datasets

```
data/datasets/
├── surveillance_detection/     # Dataset détection YOLO
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── labels/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── dataset.yaml
├── vlm_surveillance/          # Dataset Vision-Language
│   ├── images/
│   ├── conversations.json
│   └── metadata.json
├── behavior_analysis/         # Dataset analyse comportementale
│   ├── sequences/
│   ├── annotations/
│   └── behaviors.json
└── synthetic/                # Données synthétiques
    ├── generated_scenes/
    └── augmented_data/
```

## 🎯 Format YOLO Detection

### Structure de Fichiers
```
surveillance_detection/
├── images/
│   ├── train/
│   │   ├── store_001_frame_001.jpg
│   │   ├── store_001_frame_002.jpg
│   │   └── ...
│   ├── val/ 
│   └── test/
├── labels/
│   ├── train/
│   │   ├── store_001_frame_001.txt
│   │   ├── store_001_frame_002.txt
│   │   └── ...
│   ├── val/
│   └── test/
└── dataset.yaml
```

### Format des Labels YOLO
```
# Fichier: labels/train/store_001_frame_001.txt
# Format: class_id center_x center_y width height confidence
0 0.5 0.3 0.2 0.4      # person
1 0.7 0.8 0.1 0.15     # handbag
2 0.2 0.6 0.05 0.08    # bottle
```

### Classes de Surveillance
```yaml
# dataset.yaml
path: ./surveillance_detection
train: images/train
val: images/val
test: images/test

names:
  0: person
  1: handbag
  2: backpack
  3: suitcase
  4: bottle
  5: cup
  6: cell_phone
  7: book
  8: shopping_cart
  9: suspicious_object

# Classes comportementales additionnelles
behaviors:
  10: normal_shopping
  11: suspicious_movement
  12: item_concealment
  13: potential_theft
  14: group_behavior
```

### Métadonnées Enrichies
```json
{
  "image_id": "store_001_frame_001",
  "timestamp": "2024-01-15T14:30:25Z",
  "store_id": "store_001",
  "zone": "electronics",
  "camera_id": "cam_003",
  "lighting": "normal",
  "crowd_density": "medium",
  "suspicious_activity": false,
  "verified_by_human": true,
  "annotations": [
    {
      "class_id": 0,
      "class_name": "person",
      "bbox": [0.5, 0.3, 0.2, 0.4],
      "confidence": 1.0,
      "track_id": 42,
      "behavior": "normal_shopping",
      "attributes": {
        "age_group": "adult",
        "carrying_items": true,
        "movement_speed": "normal"
      }
    }
  ]
}
```

## 🧠 Format VLM Surveillance

### Structure de Conversations
```json
{
  "conversations": [
    {
      "id": "conv_001",
      "image": "images/store_001_frame_001.jpg",
      "conversations": [
        {
          "from": "human",
          "value": "Analysez cette scène de surveillance. Y a-t-il des comportements suspects ?"
        },
        {
          "from": "gpt", 
          "value": "Je vois une personne dans la section électronique qui manipule des produits de manière normale. Elle porte un sac à dos mais son comportement indique un shopping classique. Niveau de suspicion: FAIBLE. Actions recommandées: surveillance passive."
        }
      ],
      "metadata": {
        "store_zone": "electronics",
        "time_of_day": "afternoon",
        "ground_truth": "normal_shopping",
        "tools_available": ["object_detector", "tracker", "behavior_analyzer"]
      }
    }
  ]
}
```

### Format Tool-Calling VLM
```json
{
  "tool_calling_examples": [
    {
      "id": "tool_001",
      "image": "images/suspicious_scene_001.jpg",
      "conversation": [
        {
          "from": "human",
          "value": "Analysez cette scène et utilisez les outils appropriés."
        },
        {
          "from": "gpt",
          "value": "Je détecte une activité potentiellement suspecte. Laissez-moi utiliser les outils d'analyse.",
          "tool_calls": [
            {
              "name": "object_detector",
              "arguments": {
                "image": "current_frame",
                "confidence_threshold": 0.3,
                "focus_classes": ["person", "handbag", "backpack"]
              }
            },
            {
              "name": "behavior_analyzer", 
              "arguments": {
                "sequence_length": 30,
                "analysis_type": "movement_pattern"
              }
            }
          ]
        },
        {
          "from": "tool",
          "tool_name": "object_detector",
          "value": {
            "detections": [
              {"class": "person", "confidence": 0.92, "bbox": [0.3, 0.2, 0.4, 0.7]},
              {"class": "handbag", "confidence": 0.78, "bbox": [0.45, 0.6, 0.1, 0.15]}
            ]
          }
        },
        {
          "from": "gpt",
          "value": "Basé sur l'analyse des outils, je détecte 1 personne avec un sac. Le mouvement semble erratique. Recommandation: surveillance accrue. Niveau de suspicion: MOYEN."
        }
      ]
    }
  ]
}
```

## 🎭 Format Analyse Comportementale

### Séquences Temporelles
```json
{
  "sequences": [
    {
      "sequence_id": "seq_001",
      "duration": 30.0,
      "fps": 15,
      "frames": [
        {
          "frame_id": 1,
          "timestamp": 0.0,
          "detections": [
            {
              "track_id": 42,
              "class": "person",
              "bbox": [0.5, 0.3, 0.2, 0.4],
              "pose_keypoints": [...],
              "velocity": [0.02, 0.01],
              "behavior_features": {
                "gaze_direction": "products",
                "hand_position": "reaching",
                "body_orientation": "towards_shelf"
              }
            }
          ]
        }
      ],
      "behavior_labels": {
        "overall_behavior": "item_concealment",
        "suspicion_level": "HIGH",
        "key_moments": [
          {
            "timestamp": 12.5,
            "event": "item_hidden_in_bag",
            "confidence": 0.87
          }
        ]
      }
    }
  ]
}
```

## 🤖 Données Synthétiques

### Configuration de Génération
```yaml
# synthetic/config.yaml
generation_params:
  num_scenes: 10000
  environments:
    - grocery_store
    - electronics_store
    - clothing_store
    - pharmacy
  
  lighting_conditions:
    - bright
    - normal  
    - dim
    - mixed
  
  crowd_densities:
    - empty
    - low
    - medium
    - high
  
  suspicious_behaviors:
    - normal_shopping: 70%
    - suspicious_movement: 15%
    - item_concealment: 10%
    - potential_theft: 5%

augmentation:
  brightness: [-0.3, 0.3]
  contrast: [0.7, 1.3]
  saturation: [0.8, 1.2]
  noise: [0, 0.05]
  blur: [0, 2.0]
```

## 📊 Validation et Métriques

### Format Annotations de Validation
```json
{
  "validation_set": {
    "total_images": 2000,
    "human_verified": true,
    "inter_annotator_agreement": 0.92,
    "class_distribution": {
      "person": 1850,
      "handbag": 420,
      "backpack": 320,
      "suspicious_activity": 180
    },
    "quality_metrics": {
      "image_quality_score": 0.87,
      "annotation_confidence": 0.94,
      "label_consistency": 0.89
    }
  }
}
```

### Métriques d'Évaluation
```python
evaluation_metrics = {
    "detection_metrics": {
        "mAP@0.5": 0.0,      # À calculer
        "mAP@0.5:0.95": 0.0, # À calculer
        "precision": 0.0,
        "recall": 0.0,
        "f1_score": 0.0
    },
    "surveillance_metrics": {
        "false_positive_rate": 0.0,
        "true_positive_rate": 0.0,
        "alert_accuracy": 0.0,
        "response_time": 0.0
    },
    "behavioral_metrics": {
        "behavior_classification_accuracy": 0.0,
        "temporal_consistency": 0.0,
        "suspicion_level_accuracy": 0.0
    }
}
```

## 🛠️ Scripts de Préparation

### Conversion de Formats
```bash
# COCO vers YOLO
python scripts/convert_coco_to_yolo.py --input coco_annotations.json --output labels/

# Validation du dataset
python scripts/validate_dataset.py --dataset_path ./surveillance_detection/

# Génération de statistiques
python scripts/dataset_stats.py --dataset_path ./surveillance_detection/
```

### Augmentation de Données
```python
# Configuration augmentation
augmentation_config = {
    "geometric": {
        "rotation": [-5, 5],
        "scale": [0.9, 1.1], 
        "translation": [-0.1, 0.1]
    },
    "photometric": {
        "brightness": [-0.2, 0.2],
        "contrast": [0.8, 1.2],
        "saturation": [0.9, 1.1]
    },
    "synthetic": {
        "occlusion": 0.1,
        "background_replacement": 0.05,
        "lighting_simulation": True
    }
}
```

## 📈 Bonnes Pratiques

### Qualité des Données
1. **Images haute résolution** (min 640x640)
2. **Annotations précises** (IoU > 0.8)
3. **Diversité des scénarios** 
4. **Équilibrage des classes**
5. **Validation croisée** par experts

### Stratégie d'Échantillonnage
1. **80% entraînement** / 15% validation / 5% test
2. **Stratification** par magasin/zone
3. **Temporal splitting** (éviter data leakage)
4. **Class balancing** avec techniques SMOTE

### Versioning des Données
```
v1.0.0 - Dataset initial (10k images)
v1.1.0 - Ajout données nocturnes (15k images)  
v1.2.0 - Amélioration annotations comportementales
v2.0.0 - Intégration données synthétiques (50k images)
```

---

**📝 Note**: Tous les formats sont conçus pour être compatibles avec les standards de l'industrie et optimisés pour l'entraînement de modèles de surveillance en temps réel.