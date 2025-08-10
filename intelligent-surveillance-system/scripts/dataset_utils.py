#!/usr/bin/env python3
"""
Utilitaires de préparation des datasets pour le système de surveillance.

Ce script fournit des outils pour:
- Conversion entre formats de données (COCO, YOLO, etc.)
- Validation et nettoyage des datasets
- Génération de statistiques
- Augmentation de données
- Création de datasets synthétiques
"""

import os
import sys
import json
import yaml
import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import logging
from datetime import datetime, timedelta
import random
from sklearn.model_selection import train_test_split

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Configuration pour la préparation des datasets."""
    base_path: str = "data/datasets"
    output_path: str = "data/processed"
    
    # Paramètres de split
    train_split: float = 0.8
    val_split: float = 0.15
    test_split: float = 0.05
    
    # Paramètres d'augmentation
    augmentation_factor: int = 3
    enable_synthetic: bool = True
    synthetic_samples: int = 5000
    
    # Validation
    min_bbox_area: float = 0.001  # 0.1% de l'image
    max_bbox_area: float = 0.8    # 80% de l'image
    min_image_size: Tuple[int, int] = (320, 320)
    max_image_size: Tuple[int, int] = (1920, 1080)


class COCOToYOLOConverter:
    """Convertit les annotations COCO vers le format YOLO."""
    
    def __init__(self):
        self.class_mapping = {}
        
    def convert_annotations(self, coco_json_path: str, output_dir: str, 
                          class_names: Optional[Dict[int, str]] = None) -> Dict:
        """Convertit les annotations COCO vers YOLO."""
        logger.info(f"Conversion COCO -> YOLO: {coco_json_path}")
        
        with open(coco_json_path, 'r') as f:
            coco_data = json.load(f)
        
        # Extraction des informations
        images = {img['id']: img for img in coco_data['images']}
        categories = {cat['id']: cat for cat in coco_data['categories']}
        annotations = coco_data['annotations']
        
        # Création du mapping des classes
        if class_names is None:
            self.class_mapping = {cat_id: idx for idx, cat_id in enumerate(categories.keys())}
        else:
            self.class_mapping = {cat_id: idx for idx, (cat_id, name) in enumerate(class_names.items())}
        
        # Création du répertoire de sortie
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        conversion_stats = {
            'total_images': len(images),
            'total_annotations': len(annotations),
            'converted_annotations': 0,
            'skipped_annotations': 0,
            'class_distribution': {}
        }
        
        # Groupement des annotations par image
        image_annotations = {}
        for ann in annotations:
            img_id = ann['image_id']
            if img_id not in image_annotations:
                image_annotations[img_id] = []
            image_annotations[img_id].append(ann)
        
        # Conversion
        for img_id, img_info in tqdm(images.items(), desc="Conversion annotations"):
            img_width = img_info['width']
            img_height = img_info['height']
            img_filename = Path(img_info['file_name']).stem
            
            yolo_annotations = []
            
            if img_id in image_annotations:
                for ann in image_annotations[img_id]:
                    # Extraction de la bounding box COCO (x, y, width, height)
                    x, y, w, h = ann['bbox']
                    
                    # Validation de la bbox
                    if w <= 0 or h <= 0:
                        conversion_stats['skipped_annotations'] += 1
                        continue
                    
                    # Conversion vers format YOLO (centre normalisé)
                    center_x = (x + w / 2) / img_width
                    center_y = (y + h / 2) / img_height
                    norm_width = w / img_width
                    norm_height = h / img_height
                    
                    # Validation des coordonnées normalisées
                    if (center_x < 0 or center_x > 1 or center_y < 0 or center_y > 1 or
                        norm_width > 1 or norm_height > 1):
                        conversion_stats['skipped_annotations'] += 1
                        continue
                    
                    # Classe YOLO
                    category_id = ann['category_id']
                    if category_id not in self.class_mapping:
                        conversion_stats['skipped_annotations'] += 1
                        continue
                    
                    yolo_class = self.class_mapping[category_id]
                    
                    # Création de l'annotation YOLO
                    yolo_annotation = f"{yolo_class} {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}"
                    yolo_annotations.append(yolo_annotation)
                    
                    # Statistiques
                    conversion_stats['converted_annotations'] += 1
                    class_name = categories[category_id]['name']
                    conversion_stats['class_distribution'][class_name] = \
                        conversion_stats['class_distribution'].get(class_name, 0) + 1
            
            # Sauvegarde du fichier d'annotations YOLO
            if yolo_annotations:
                label_file = output_path / f"{img_filename}.txt"
                with open(label_file, 'w') as f:
                    f.write('\\n'.join(yolo_annotations))
        
        logger.info(f"Conversion terminée: {conversion_stats['converted_annotations']} annotations converties")
        return conversion_stats


class DatasetValidator:
    """Validateur de datasets avec nettoyage automatique."""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        
    def validate_yolo_dataset(self, dataset_path: str) -> Dict:
        """Valide un dataset YOLO et génère un rapport."""
        logger.info(f"Validation du dataset YOLO: {dataset_path}")
        
        dataset_path = Path(dataset_path)
        
        validation_report = {
            'timestamp': datetime.now().isoformat(),
            'dataset_path': str(dataset_path),
            'total_images': 0,
            'total_labels': 0,
            'valid_samples': 0,
            'issues': {
                'missing_labels': [],
                'corrupted_images': [],
                'invalid_annotations': [],
                'size_issues': [],
                'bbox_issues': []
            },
            'class_distribution': {},
            'size_statistics': {
                'width': {'min': float('inf'), 'max': 0, 'mean': 0},
                'height': {'min': float('inf'), 'max': 0, 'mean': 0}
            },
            'bbox_statistics': {
                'area': {'min': float('inf'), 'max': 0, 'mean': 0},
                'aspect_ratio': {'min': float('inf'), 'max': 0, 'mean': 0}
            }
        }
        
        # Recherche des images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(dataset_path.glob(f"**/*{ext}"))
            image_files.extend(dataset_path.glob(f"**/*{ext.upper()}"))
        
        validation_report['total_images'] = len(image_files)
        
        width_sum, height_sum = 0, 0
        bbox_areas, aspect_ratios = [], []
        
        for img_path in tqdm(image_files, desc="Validation images"):
            try:
                # Vérification de l'image
                img = cv2.imread(str(img_path))
                if img is None:
                    validation_report['issues']['corrupted_images'].append(str(img_path))
                    continue
                
                h, w = img.shape[:2]
                
                # Vérification de la taille
                if (w < self.config.min_image_size[0] or h < self.config.min_image_size[1] or
                    w > self.config.max_image_size[0] or h > self.config.max_image_size[1]):
                    validation_report['issues']['size_issues'].append({
                        'file': str(img_path),
                        'size': [w, h],
                        'reason': 'Invalid size'
                    })
                
                # Statistiques de taille
                validation_report['size_statistics']['width']['min'] = min(
                    validation_report['size_statistics']['width']['min'], w)
                validation_report['size_statistics']['width']['max'] = max(
                    validation_report['size_statistics']['width']['max'], w)
                validation_report['size_statistics']['height']['min'] = min(
                    validation_report['size_statistics']['height']['min'], h)
                validation_report['size_statistics']['height']['max'] = max(
                    validation_report['size_statistics']['height']['max'], h)
                
                width_sum += w
                height_sum += h
                
                # Vérification du fichier label correspondant
                label_path = img_path.with_suffix('.txt')
                if not label_path.exists():
                    # Chercher dans un dossier labels
                    label_path = dataset_path / "labels" / img_path.relative_to(dataset_path).with_suffix('.txt')
                    
                if not label_path.exists():
                    validation_report['issues']['missing_labels'].append(str(img_path))
                    continue
                
                # Validation des annotations
                with open(label_path, 'r') as f:
                    lines = f.read().strip().split('\\n')
                    
                if not lines or lines == ['']:
                    continue  # Image sans annotations (acceptable)
                
                validation_report['total_labels'] += len(lines)
                
                for line_num, line in enumerate(lines, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) != 5:
                        validation_report['issues']['invalid_annotations'].append({
                            'file': str(label_path),
                            'line': line_num,
                            'content': line,
                            'reason': f'Expected 5 values, got {len(parts)}'
                        })
                        continue
                    
                    try:
                        class_id = int(parts[0])
                        center_x, center_y, width, height = map(float, parts[1:5])
                        
                        # Validation des coordonnées
                        if not (0 <= center_x <= 1 and 0 <= center_y <= 1 and
                               0 < width <= 1 and 0 < height <= 1):
                            validation_report['issues']['bbox_issues'].append({
                                'file': str(label_path),
                                'line': line_num,
                                'bbox': [center_x, center_y, width, height],
                                'reason': 'Coordinates out of bounds'
                            })
                            continue
                        
                        # Validation de la taille de bbox
                        bbox_area = width * height
                        if (bbox_area < self.config.min_bbox_area or 
                            bbox_area > self.config.max_bbox_area):
                            validation_report['issues']['bbox_issues'].append({
                                'file': str(label_path),
                                'line': line_num,
                                'bbox': [center_x, center_y, width, height],
                                'reason': f'Bbox area {bbox_area:.4f} out of range'
                            })
                            continue
                        
                        # Statistiques
                        bbox_areas.append(bbox_area)
                        aspect_ratios.append(width / height)
                        
                        # Distribution des classes
                        validation_report['class_distribution'][class_id] = \
                            validation_report['class_distribution'].get(class_id, 0) + 1
                        
                    except ValueError as e:
                        validation_report['issues']['invalid_annotations'].append({
                            'file': str(label_path),
                            'line': line_num,
                            'content': line,
                            'reason': f'Parse error: {str(e)}'
                        })
                
                validation_report['valid_samples'] += 1
                
            except Exception as e:
                validation_report['issues']['corrupted_images'].append({
                    'file': str(img_path),
                    'error': str(e)
                })
        
        # Calcul des statistiques finales
        if validation_report['total_images'] > 0:
            validation_report['size_statistics']['width']['mean'] = width_sum / validation_report['total_images']
            validation_report['size_statistics']['height']['mean'] = height_sum / validation_report['total_images']
        
        if bbox_areas:
            validation_report['bbox_statistics']['area'] = {
                'min': min(bbox_areas),
                'max': max(bbox_areas),
                'mean': np.mean(bbox_areas)
            }
            validation_report['bbox_statistics']['aspect_ratio'] = {
                'min': min(aspect_ratios),
                'max': max(aspect_ratios),
                'mean': np.mean(aspect_ratios)
            }
        
        # Résumé
        total_issues = sum(len(issues) for issues in validation_report['issues'].values())
        validation_report['summary'] = {
            'total_issues': total_issues,
            'success_rate': validation_report['valid_samples'] / max(validation_report['total_images'], 1),
            'issues_by_type': {k: len(v) for k, v in validation_report['issues'].items()}
        }
        
        logger.info(f"Validation terminée: {validation_report['valid_samples']}/{validation_report['total_images']} images valides")
        return validation_report
    
    def clean_dataset(self, dataset_path: str, validation_report: Dict, backup: bool = True) -> Dict:
        """Nettoie un dataset basé sur le rapport de validation."""
        logger.info(f"Nettoyage du dataset: {dataset_path}")
        
        dataset_path = Path(dataset_path)
        
        if backup:
            backup_path = dataset_path.parent / f"{dataset_path.name}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            logger.info(f"Création du backup: {backup_path}")
            shutil.copytree(dataset_path, backup_path)
        
        cleaning_stats = {
            'removed_files': 0,
            'fixed_annotations': 0,
            'resized_images': 0
        }
        
        # Suppression des images corrompues
        for corrupted in validation_report['issues']['corrupted_images']:
            file_path = Path(corrupted if isinstance(corrupted, str) else corrupted['file'])
            if file_path.exists():
                file_path.unlink()
                # Suppression du label correspondant s'il existe
                label_path = file_path.with_suffix('.txt')
                if label_path.exists():
                    label_path.unlink()
                cleaning_stats['removed_files'] += 1
        
        # Suppression des images sans labels (si requis)
        for missing_label in validation_report['issues']['missing_labels']:
            file_path = Path(missing_label)
            if file_path.exists():
                file_path.unlink()
                cleaning_stats['removed_files'] += 1
        
        logger.info(f"Nettoyage terminé: {cleaning_stats['removed_files']} fichiers supprimés")
        return cleaning_stats


class DatasetSplitter:
    """Gestionnaire pour le découpage train/val/test."""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        
    def split_dataset(self, dataset_path: str, output_path: str, 
                     stratify_by_class: bool = True) -> Dict:
        """Découpe un dataset en train/val/test."""
        logger.info(f"Découpage du dataset: {dataset_path}")
        
        dataset_path = Path(dataset_path)
        output_path = Path(output_path)
        
        # Création des répertoires de sortie
        for split in ['train', 'val', 'test']:
            (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
            (output_path / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        # Collecte des fichiers
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(dataset_path.glob(f"**/*{ext}"))
            image_files.extend(dataset_path.glob(f"**/*{ext.upper()}"))
        
        # Préparation des données pour stratification
        if stratify_by_class:
            # Analyse des classes pour chaque image
            image_classes = []
            valid_images = []
            
            for img_path in tqdm(image_files, desc="Analyse des classes"):
                label_path = img_path.with_suffix('.txt')
                if not label_path.exists():
                    label_path = dataset_path / "labels" / img_path.relative_to(dataset_path).with_suffix('.txt')
                
                if label_path.exists():
                    with open(label_path, 'r') as f:
                        lines = f.read().strip().split('\\n')
                    
                    classes = set()
                    for line in lines:
                        if line.strip():
                            parts = line.split()
                            if len(parts) >= 1:
                                try:
                                    classes.add(int(parts[0]))
                                except ValueError:
                                    continue
                    
                    if classes:
                        valid_images.append(img_path)
                        # Utilise la classe principale (première classe trouvée)
                        image_classes.append(list(classes)[0])
            
            # Découpage stratifié
            if len(valid_images) > 0:
                # Premier split: train vs (val+test)
                train_files, temp_files, train_classes, temp_classes = train_test_split(
                    valid_images, image_classes,
                    test_size=(self.config.val_split + self.config.test_split),
                    random_state=42,
                    stratify=image_classes
                )
                
                # Deuxième split: val vs test
                if len(temp_files) > 1:
                    val_files, test_files, _, _ = train_test_split(
                        temp_files, temp_classes,
                        test_size=self.config.test_split / (self.config.val_split + self.config.test_split),
                        random_state=42,
                        stratify=temp_classes
                    )
                else:
                    val_files = temp_files
                    test_files = []
            else:
                train_files, val_files, test_files = [], [], []
                
        else:
            # Découpage aléatoire simple
            random.shuffle(image_files)
            
            n_total = len(image_files)
            n_train = int(n_total * self.config.train_split)
            n_val = int(n_total * self.config.val_split)
            
            train_files = image_files[:n_train]
            val_files = image_files[n_train:n_train + n_val]
            test_files = image_files[n_train + n_val:]
        
        # Copie des fichiers dans les répertoires appropriés
        split_stats = {
            'train': self._copy_split_files(train_files, output_path / 'train', dataset_path),
            'val': self._copy_split_files(val_files, output_path / 'val', dataset_path),
            'test': self._copy_split_files(test_files, output_path / 'test', dataset_path)
        }
        
        # Création du fichier dataset.yaml
        dataset_yaml = {
            'path': str(output_path),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'names': {i: f"class_{i}" for i in range(10)}  # Classes par défaut
        }
        
        with open(output_path / 'dataset.yaml', 'w') as f:
            yaml.dump(dataset_yaml, f, default_flow_style=False)
        
        logger.info(f"Découpage terminé: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")
        return split_stats
    
    def _copy_split_files(self, files: List[Path], split_dir: Path, dataset_path: Path) -> Dict:
        """Copie les fichiers dans un répertoire de split."""
        stats = {'images': 0, 'labels': 0}
        
        for img_path in tqdm(files, desc=f"Copie {split_dir.name}"):
            # Copie de l'image
            dest_img = split_dir / 'images' / img_path.name
            shutil.copy2(img_path, dest_img)
            stats['images'] += 1
            
            # Copie du label
            label_path = img_path.with_suffix('.txt')
            if not label_path.exists():
                label_path = dataset_path / "labels" / img_path.relative_to(dataset_path).with_suffix('.txt')
            
            if label_path.exists():
                dest_label = split_dir / 'labels' / label_path.name
                shutil.copy2(label_path, dest_label)
                stats['labels'] += 1
        
        return stats


class DatasetAugmenter:
    """Générateur d'augmentations de données."""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        
    def augment_dataset(self, dataset_path: str, output_path: str, 
                       augmentation_config: Optional[Dict] = None) -> Dict:
        """Applique des augmentations à un dataset."""
        logger.info(f"Augmentation du dataset: {dataset_path}")
        
        if augmentation_config is None:
            augmentation_config = {
                'rotation': (-10, 10),
                'brightness': (0.8, 1.2),
                'contrast': (0.8, 1.2),
                'saturation': (0.8, 1.2),
                'blur': (0, 1.5),
                'noise': (0, 0.02),
                'flip_horizontal': 0.5,
                'flip_vertical': 0.1
            }
        
        dataset_path = Path(dataset_path)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Collecte des images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(dataset_path.glob(f"**/*{ext}"))
        
        augmentation_stats = {
            'original_images': len(image_files),
            'augmented_images': 0,
            'total_generated': 0
        }
        
        for img_path in tqdm(image_files, desc="Augmentation"):
            # Chargement de l'image et des annotations
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            
            label_path = img_path.with_suffix('.txt')
            annotations = []
            
            if label_path.exists():
                with open(label_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            parts = line.split()
                            if len(parts) == 5:
                                annotations.append([float(x) for x in parts])
            
            # Génération des augmentations
            for aug_idx in range(self.config.augmentation_factor):
                aug_image, aug_annotations = self._apply_augmentations(
                    image.copy(), annotations.copy(), augmentation_config
                )
                
                # Sauvegarde
                base_name = img_path.stem
                aug_img_name = f"{base_name}_aug_{aug_idx:03d}{img_path.suffix}"
                aug_label_name = f"{base_name}_aug_{aug_idx:03d}.txt"
                
                cv2.imwrite(str(output_path / aug_img_name), aug_image)
                
                if aug_annotations:
                    with open(output_path / aug_label_name, 'w') as f:
                        for ann in aug_annotations:
                            f.write(f"{int(ann[0])} {ann[1]:.6f} {ann[2]:.6f} {ann[3]:.6f} {ann[4]:.6f}\\n")
                
                augmentation_stats['augmented_images'] += 1
            
            augmentation_stats['total_generated'] += self.config.augmentation_factor
        
        logger.info(f"Augmentation terminée: {augmentation_stats['augmented_images']} images générées")
        return augmentation_stats
    
    def _apply_augmentations(self, image: np.ndarray, annotations: List, 
                           config: Dict) -> Tuple[np.ndarray, List]:
        """Applique des transformations aléatoires à une image et ses annotations."""
        h, w = image.shape[:2]
        
        # Flip horizontal
        if random.random() < config.get('flip_horizontal', 0):
            image = cv2.flip(image, 1)
            # Adaptation des annotations
            for ann in annotations:
                ann[1] = 1.0 - ann[1]  # Inversion de center_x
        
        # Flip vertical
        if random.random() < config.get('flip_vertical', 0):
            image = cv2.flip(image, 0)
            # Adaptation des annotations
            for ann in annotations:
                ann[2] = 1.0 - ann[2]  # Inversion de center_y
        
        # Rotation légère
        if 'rotation' in config:
            angle = random.uniform(*config['rotation'])
            if abs(angle) > 0.5:
                center = (w // 2, h // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                image = cv2.warpAffine(image, rotation_matrix, (w, h))
        
        # Ajustements de couleur
        if 'brightness' in config:
            brightness_factor = random.uniform(*config['brightness'])
            image = cv2.convertScaleAbs(image, alpha=brightness_factor, beta=0)
        
        if 'contrast' in config:
            contrast_factor = random.uniform(*config['contrast'])
            image = cv2.convertScaleAbs(image, alpha=contrast_factor, beta=0)
        
        # Bruit
        if 'noise' in config:
            noise_factor = random.uniform(*config['noise'])
            if noise_factor > 0:
                noise = np.random.normal(0, noise_factor * 255, image.shape).astype(np.uint8)
                image = cv2.add(image, noise)
        
        # Flou
        if 'blur' in config:
            blur_factor = random.uniform(*config['blur'])
            if blur_factor > 0.5:
                kernel_size = int(blur_factor * 2) * 2 + 1
                image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        
        return image, annotations


def main():
    """Fonction principale avec interface en ligne de commande."""
    parser = argparse.ArgumentParser(description='Utilitaires de préparation des datasets')
    parser.add_argument('command', choices=['convert', 'validate', 'split', 'augment', 'stats'],
                       help='Commande à exécuter')
    parser.add_argument('--input', '-i', required=True, help='Chemin du dataset d\'entrée')
    parser.add_argument('--output', '-o', help='Chemin de sortie')
    parser.add_argument('--config', '-c', help='Fichier de configuration YAML')
    parser.add_argument('--format', choices=['coco', 'yolo'], default='yolo',
                       help='Format du dataset')
    parser.add_argument('--backup', action='store_true', help='Créer un backup avant nettoyage')
    parser.add_argument('--clean', action='store_true', help='Nettoyer le dataset après validation')
    
    args = parser.parse_args()
    
    # Chargement de la configuration
    if args.config:
        with open(args.config, 'r') as f:
            config_data = yaml.safe_load(f)
        config = DatasetConfig(**config_data)
    else:
        config = DatasetConfig()
    
    # Exécution de la commande
    if args.command == 'convert':
        if args.format == 'coco':
            converter = COCOToYOLOConverter()
            result = converter.convert_annotations(args.input, args.output or 'converted_dataset')
            print(f"Conversion terminée: {result['converted_annotations']} annotations")
        else:
            print("Conversion YOLO->COCO pas encore implémentée")
    
    elif args.command == 'validate':
        validator = DatasetValidator(config)
        report = validator.validate_yolo_dataset(args.input)
        
        # Sauvegarde du rapport
        output_file = args.output or f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Rapport de validation sauvegardé: {output_file}")
        print(f"Images valides: {report['valid_samples']}/{report['total_images']}")
        
        if args.clean and report['summary']['total_issues'] > 0:
            cleaning_stats = validator.clean_dataset(args.input, report, backup=args.backup)
            print(f"Nettoyage effectué: {cleaning_stats['removed_files']} fichiers supprimés")
    
    elif args.command == 'split':
        splitter = DatasetSplitter(config)
        result = splitter.split_dataset(args.input, args.output or 'split_dataset')
        print(f"Découpage terminé: {result}")
    
    elif args.command == 'augment':
        augmenter = DatasetAugmenter(config)
        result = augmenter.augment_dataset(args.input, args.output or 'augmented_dataset')
        print(f"Augmentation terminée: {result['augmented_images']} images générées")
    
    elif args.command == 'stats':
        validator = DatasetValidator(config)
        report = validator.validate_yolo_dataset(args.input)
        
        print("\\n=== STATISTIQUES DU DATASET ===")
        print(f"Images totales: {report['total_images']}")
        print(f"Labels totaux: {report['total_labels']}")
        print(f"Images valides: {report['valid_samples']}")
        print(f"Taux de succès: {report['summary']['success_rate']:.2%}")
        
        print("\\n=== DISTRIBUTION DES CLASSES ===")
        for class_id, count in report['class_distribution'].items():
            print(f"Classe {class_id}: {count} annotations")
        
        print("\\n=== PROBLÈMES DÉTECTÉS ===")
        for issue_type, issues in report['issues'].items():
            if issues:
                print(f"{issue_type}: {len(issues)} problèmes")


if __name__ == '__main__':
    main()