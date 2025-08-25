#!/usr/bin/env python3
"""
📁 Gestionnaire de Datasets Vidéo pour Tests
===========================================

Script pour gérer les datasets vidéo de test :
- Création de datasets organisés
- Validation des vidéos
- Génération d'exemples synthétiques
- Préparation pour tests automatisés

Usage:
    python scripts/manage_video_datasets.py --create-structure
    python scripts/manage_video_datasets.py --validate
    python scripts/manage_video_datasets.py --generate-examples
"""

import sys
import os
import shutil
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import cv2
import numpy as np
from datetime import datetime

# Configuration du path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.video_test_config import (
    VIDEO_DATASETS_ROOT,
    VIDEO_TEST_ROOT,
    RECOMMENDED_VIDEO_DATASETS
)


class VideoDatasetManager:
    """Gestionnaire de datasets vidéo."""
    
    def __init__(self):
        self.datasets_root = VIDEO_DATASETS_ROOT
        self.test_root = VIDEO_TEST_ROOT
        
        # Structure recommandée
        self.dataset_structure = {
            "surveillance_basic": [
                "person_walking_indoor.mp4",
                "person_walking_outdoor.mp4", 
                "object_on_table.mp4",
                "empty_room.mp4"
            ],
            "theft_scenarios": [
                "shoplifting_simulation.mp4",
                "bag_snatching.mp4",
                "pickpocket_simulation.mp4",
                "theft_attempt.mp4"
            ],
            "crowded_scenes": [
                "mall_crowd.mp4",
                "street_busy.mp4",
                "multiple_persons.mp4",
                "group_interaction.mp4"
            ],
            "edge_cases": [
                "low_light.mp4",
                "occlusion_test.mp4",
                "fast_movement.mp4",
                "camera_shake.mp4"
            ],
            "synthetic": [
                "generated_basic.mp4",
                "generated_theft.mp4",
                "generated_crowd.mp4",
                "generated_edge.mp4"
            ]
        }
    
    def create_dataset_structure(self) -> bool:
        """Création de la structure de datasets."""
        print("📁 Création de la structure de datasets...")
        
        try:
            # Dossier principal
            self.datasets_root.mkdir(parents=True, exist_ok=True)
            
            # Dossiers par catégorie
            for category, files in self.dataset_structure.items():
                category_dir = self.datasets_root / category
                category_dir.mkdir(exist_ok=True)
                
                # README pour chaque catégorie
                readme_path = category_dir / "README.md"
                readme_content = self._generate_category_readme(category, files)
                with open(readme_path, 'w') as f:
                    f.write(readme_content)
                
                print(f"  ✅ {category}/ créé")
            
            # Fichier de métadonnées global
            metadata = {
                "created": datetime.now().isoformat(),
                "structure": self.dataset_structure,
                "total_categories": len(self.dataset_structure),
                "expected_files": sum(len(files) for files in self.dataset_structure.values())
            }
            
            metadata_path = self.datasets_root / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Guide d'utilisation
            guide_path = self.datasets_root / "USAGE_GUIDE.md"
            with open(guide_path, 'w') as f:
                f.write(self._generate_usage_guide())
            
            print(f"✅ Structure créée dans {self.datasets_root}")
            return True
            
        except Exception as e:
            print(f"❌ Erreur création structure: {e}")
            return False
    
    def _generate_category_readme(self, category: str, files: List[str]) -> str:
        """Génération du README pour une catégorie."""
        descriptions = {
            "surveillance_basic": "Vidéos de surveillance de base avec scénarios simples",
            "theft_scenarios": "Simulations de vol et comportements suspects",
            "crowded_scenes": "Scènes avec plusieurs personnes et interactions complexes",
            "edge_cases": "Cas limites : faible éclairage, occlusions, mouvements rapides",
            "synthetic": "Vidéos générées synthétiquement pour tests contrôlés"
        }
        
        content = f"""# {category.replace('_', ' ').title()}

## Description
{descriptions.get(category, "Catégorie de vidéos de test")}

## Fichiers Attendus
"""
        
        for file in files:
            content += f"- `{file}` - Durée recommandée: 30-60s\n"
        
        content += f"""
## Format Recommandé
- **Résolution**: 720p ou 1080p
- **Format**: MP4 (H.264)
- **Durée**: 30-120 secondes
- **FPS**: 25-30 fps

## Placement
Placez vos fichiers vidéo dans ce dossier :
```
{category}/
├── {files[0]}
├── {files[1]}
└── ...
```

## Test
Pour tester avec ces vidéos :
```bash
python scripts/test_video_pipeline.py {category}/{files[0]}
```
"""
        return content
    
    def _generate_usage_guide(self) -> str:
        """Génération du guide d'utilisation."""
        return """# 📖 Guide d'Utilisation des Datasets Vidéo

## 🎯 Vue d'Ensemble

Ce dossier contient les datasets vidéo organisés pour tester le système de surveillance intelligente avec Kimi-VL.

## 📁 Structure

```
video_datasets/
├── surveillance_basic/     # Scénarios de base
├── theft_scenarios/        # Simulations de vol  
├── crowded_scenes/         # Scènes complexes
├── edge_cases/             # Cas limites
├── synthetic/              # Vidéos générées
└── metadata.json           # Métadonnées
```

## 🚀 Comment Utiliser

### 1. Ajouter des Vidéos
Placez vos fichiers vidéo dans les dossiers appropriés selon leur contenu.

### 2. Valider les Datasets
```bash
python scripts/manage_video_datasets.py --validate
```

### 3. Tester un Pipeline Complet
```bash
# Test rapide
python scripts/test_video_pipeline.py video_datasets/surveillance_basic/person_walking.mp4 --profile fast

# Test complet
python scripts/test_video_pipeline.py video_datasets/theft_scenarios/shoplifting.mp4 --profile thorough
```

### 4. Tests Automatisés
```bash
# Test tous les datasets
python scripts/batch_video_test.py --all-categories

# Test catégorie spécifique
python scripts/batch_video_test.py --category surveillance_basic
```

## 📊 Formats Recommandés

| Paramètre | Valeur Recommandée |
|-----------|-------------------|
| Format | MP4 (H.264) |
| Résolution | 720p-1080p |
| Durée | 30-120 secondes |
| FPS | 25-30 fps |
| Taille | < 50MB par fichier |

## 🔍 Types de Contenu

### surveillance_basic
- Personnes marchant normalement
- Objets statiques
- Salles vides
- Mouvements simples

### theft_scenarios  
- Simulations de vol en magasin
- Pickpocket
- Comportements suspects
- Tentatives de vol

### crowded_scenes
- Centres commerciaux
- Rues fréquentées
- Interactions multiples
- Foules denses

### edge_cases
- Éclairage faible
- Occlusions partielles
- Mouvements rapides
- Qualité dégradée

### synthetic
- Vidéos générées par IA
- Scénarios contrôlés
- Tests de robustesse
- Cas spécifiques

## 🛠️ Outils Disponibles

- `manage_video_datasets.py` - Gestion des datasets
- `test_video_pipeline.py` - Test pipeline individuel
- `batch_video_test.py` - Tests en lot
- `generate_synthetic_videos.py` - Génération synthétique

## 📈 Métriques de Test

Le système mesure automatiquement :
- Détections YOLO (précision, rappel)
- Performance VLM (confiance, vitesse)
- Alertes générées
- Utilisation ressources
- Temps de traitement

## 💡 Conseils

1. **Commencez petit** : Testez avec `surveillance_basic` d'abord
2. **Variez les contenus** : Utilisez différentes catégories
3. **Surveillez la mémoire** : Utilisez `--performance` pour monitoring
4. **Documentez vos tests** : Les résultats sont sauvés automatiquement
5. **Itérez** : Affinez selon les résultats

---

**Datasets Vidéo v1.0.0**  
*Système de Surveillance Intelligente - Tests avec Kimi-VL*"""
    
    def validate_datasets(self) -> Dict[str, Any]:
        """Validation des datasets existants."""
        print("🔍 Validation des datasets...")
        
        validation_results = {
            "valid_categories": [],
            "invalid_categories": [],
            "total_files": 0,
            "valid_files": 0,
            "invalid_files": [],
            "missing_files": [],
            "file_details": {}
        }
        
        for category, expected_files in self.dataset_structure.items():
            category_dir = self.datasets_root / category
            category_valid = True
            
            print(f"  📁 Vérification {category}...")
            
            if not category_dir.exists():
                print(f"    ❌ Dossier manquant: {category}")
                validation_results["invalid_categories"].append(category)
                validation_results["missing_files"].extend([f"{category}/{f}" for f in expected_files])
                continue
            
            category_files = []
            for expected_file in expected_files:
                file_path = category_dir / expected_file
                file_info = {
                    "path": str(file_path),
                    "exists": file_path.exists(),
                    "valid": False,
                    "size_mb": 0,
                    "duration_s": 0,
                    "resolution": None,
                    "fps": 0
                }
                
                if file_path.exists():
                    try:
                        # Validation avec OpenCV
                        cap = cv2.VideoCapture(str(file_path))
                        
                        if cap.isOpened():
                            file_info["valid"] = True
                            file_info["size_mb"] = file_path.stat().st_size / (1024 * 1024)
                            
                            # Propriétés vidéo
                            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                            fps = cap.get(cv2.CAP_PROP_FPS)
                            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            
                            file_info["duration_s"] = total_frames / fps if fps > 0 else 0
                            file_info["resolution"] = f"{width}x{height}"
                            file_info["fps"] = fps
                            
                            validation_results["valid_files"] += 1
                            print(f"    ✅ {expected_file} - {file_info['resolution']}, {file_info['duration_s']:.1f}s")
                        else:
                            print(f"    ❌ {expected_file} - Impossible à ouvrir")
                            validation_results["invalid_files"].append(f"{category}/{expected_file}")
                            category_valid = False
                        
                        cap.release()
                        
                    except Exception as e:
                        print(f"    ❌ {expected_file} - Erreur: {e}")
                        validation_results["invalid_files"].append(f"{category}/{expected_file}")
                        category_valid = False
                else:
                    print(f"    ❌ {expected_file} - Fichier manquant")
                    validation_results["missing_files"].append(f"{category}/{expected_file}")
                    category_valid = False
                
                category_files.append(file_info)
                validation_results["total_files"] += 1
            
            validation_results["file_details"][category] = category_files
            
            if category_valid and len([f for f in category_files if f["valid"]]) > 0:
                validation_results["valid_categories"].append(category)
            else:
                validation_results["invalid_categories"].append(category)
        
        # Sauvegarde du rapport de validation
        report_path = self.datasets_root / "validation_report.json"
        with open(report_path, 'w') as f:
            json.dump(validation_results, f, indent=2, default=str)
        
        # Résumé
        print(f"\n📊 RÉSULTATS VALIDATION:")
        print(f"  Catégories valides: {len(validation_results['valid_categories'])}")
        print(f"  Fichiers valides: {validation_results['valid_files']}/{validation_results['total_files']}")
        print(f"  Fichiers manquants: {len(validation_results['missing_files'])}")
        print(f"  Rapport sauvé: {report_path}")
        
        return validation_results
    
    def generate_synthetic_examples(self) -> bool:
        """Génération d'exemples vidéo synthétiques pour tests."""
        print("🎬 Génération d'exemples synthétiques...")
        
        synthetic_dir = self.datasets_root / "synthetic"
        synthetic_dir.mkdir(exist_ok=True)
        
        examples = [
            {
                "name": "generated_basic.mp4",
                "description": "Vidéo basique avec rectangle mobile",
                "duration": 5,
                "fps": 30,
                "width": 640,
                "height": 480,
                "generator": self._generate_basic_video
            },
            {
                "name": "generated_theft.mp4", 
                "description": "Simulation de vol avec objets",
                "duration": 8,
                "fps": 30,
                "width": 640,
                "height": 480,
                "generator": self._generate_theft_simulation
            },
            {
                "name": "generated_crowd.mp4",
                "description": "Scène avec multiples objets mobiles",
                "duration": 10,
                "fps": 30,
                "width": 640,
                "height": 480,
                "generator": self._generate_crowd_scene
            }
        ]
        
        for example in examples:
            output_path = synthetic_dir / example["name"]
            print(f"  🎥 Génération {example['name']}...")
            
            try:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(
                    str(output_path), fourcc, example["fps"],
                    (example["width"], example["height"])
                )
                
                total_frames = example["duration"] * example["fps"]
                example["generator"](writer, total_frames, example["width"], example["height"])
                
                writer.release()
                print(f"    ✅ Généré: {example['name']}")
                
            except Exception as e:
                print(f"    ❌ Erreur génération {example['name']}: {e}")
                return False
        
        print("✅ Génération terminée")
        return True
    
    def _generate_basic_video(self, writer, total_frames: int, width: int, height: int):
        """Génération vidéo basique avec rectangle mobile."""
        for frame_idx in range(total_frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Rectangle mobile
            x = int((frame_idx / total_frames) * (width - 100))
            y = height // 2 - 25
            
            cv2.rectangle(frame, (x, y), (x + 100, y + 50), (0, 255, 0), -1)
            
            # Texte info
            cv2.putText(frame, f"Frame: {frame_idx}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            writer.write(frame)
    
    def _generate_theft_simulation(self, writer, total_frames: int, width: int, height: int):
        """Génération simulation de vol."""
        for frame_idx in range(total_frames):
            frame = np.ones((height, width, 3), dtype=np.uint8) * 50  # Fond gris foncé
            
            # Objet fixe (table/étagère)
            cv2.rectangle(frame, (width//2 - 50, height//2 - 20), 
                         (width//2 + 50, height//2 + 20), (139, 69, 19), -1)
            
            # Objet sur la table
            if frame_idx < total_frames * 0.7:
                cv2.circle(frame, (width//2, height//2), 15, (0, 0, 255), -1)
            
            # Personne qui s'approche
            if frame_idx > total_frames * 0.3:
                person_x = int(50 + (frame_idx - total_frames * 0.3) / (total_frames * 0.7) * (width//2 - 100))
                cv2.rectangle(frame, (person_x, height//2 - 40), 
                             (person_x + 30, height//2 + 40), (255, 255, 0), -1)
            
            # Texte scénario
            scenario = "Normal"
            if frame_idx > total_frames * 0.7:
                scenario = "THEFT DETECTED"
                cv2.putText(frame, scenario, (10, height - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            cv2.putText(frame, f"Frame: {frame_idx} | {scenario}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            writer.write(frame)
    
    def _generate_crowd_scene(self, writer, total_frames: int, width: int, height: int):
        """Génération scène de foule."""
        # Positions et vitesses aléatoires pour plusieurs objets
        np.random.seed(42)  # Pour reproductibilité
        num_objects = 5
        objects = []
        
        for i in range(num_objects):
            objects.append({
                "x": np.random.randint(0, width),
                "y": np.random.randint(50, height-50),
                "vx": np.random.randint(-3, 4),
                "vy": np.random.randint(-2, 3),
                "color": (np.random.randint(100, 255), np.random.randint(100, 255), np.random.randint(100, 255)),
                "size": np.random.randint(15, 25)
            })
        
        for frame_idx in range(total_frames):
            frame = np.ones((height, width, 3), dtype=np.uint8) * 30  # Fond sombre
            
            # Mise à jour et dessin des objets
            for obj in objects:
                # Mise à jour position
                obj["x"] += obj["vx"]
                obj["y"] += obj["vy"]
                
                # Rebond sur les bords
                if obj["x"] <= obj["size"] or obj["x"] >= width - obj["size"]:
                    obj["vx"] *= -1
                if obj["y"] <= obj["size"] or obj["y"] >= height - obj["size"]:
                    obj["vy"] *= -1
                
                # Contrainte dans l'écran
                obj["x"] = max(obj["size"], min(width - obj["size"], obj["x"]))
                obj["y"] = max(obj["size"], min(height - obj["size"], obj["y"]))
                
                # Dessin
                cv2.circle(frame, (int(obj["x"]), int(obj["y"])), obj["size"], obj["color"], -1)
            
            # Informations
            cv2.putText(frame, f"Frame: {frame_idx} | Objects: {num_objects}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            writer.write(frame)
    
    def list_datasets(self):
        """Listage des datasets disponibles."""
        print("📋 Datasets Disponibles:")
        
        if not self.datasets_root.exists():
            print("❌ Aucun dataset trouvé. Utilisez --create-structure d'abord.")
            return
        
        total_files = 0
        for category in self.dataset_structure.keys():
            category_dir = self.datasets_root / category
            if category_dir.exists():
                files = list(category_dir.glob("*.mp4"))
                print(f"  📁 {category}: {len(files)} fichiers")
                for file in files[:3]:  # Afficher max 3 fichiers
                    size_mb = file.stat().st_size / (1024 * 1024)
                    print(f"    • {file.name} ({size_mb:.1f} MB)")
                if len(files) > 3:
                    print(f"    ... et {len(files) - 3} autres fichiers")
                total_files += len(files)
            else:
                print(f"  📁 {category}: dossier manquant")
        
        print(f"\n📊 Total: {total_files} fichiers vidéo")


def create_argument_parser() -> argparse.ArgumentParser:
    """Créer le parser d'arguments."""
    parser = argparse.ArgumentParser(
        description="Gestionnaire de datasets vidéo pour tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  python scripts/manage_video_datasets.py --create-structure    # Créer structure
  python scripts/manage_video_datasets.py --validate           # Valider datasets
  python scripts/manage_video_datasets.py --generate-examples  # Générer exemples
  python scripts/manage_video_datasets.py --list               # Lister datasets
        """
    )
    
    parser.add_argument(
        "--create-structure",
        action="store_true",
        help="Créer la structure de datasets"
    )
    
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Valider les datasets existants"
    )
    
    parser.add_argument(
        "--generate-examples",
        action="store_true",
        help="Générer des exemples synthétiques"
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="Lister les datasets disponibles"
    )
    
    return parser


def main():
    """Point d'entrée principal."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    manager = VideoDatasetManager()
    
    if args.create_structure:
        manager.create_dataset_structure()
    
    if args.validate:
        manager.validate_datasets()
    
    if args.generate_examples:
        manager.generate_synthetic_examples()
    
    if args.list:
        manager.list_datasets()
    
    if not any([args.create_structure, args.validate, args.generate_examples, args.list]):
        parser.print_help()


if __name__ == "__main__":
    main()