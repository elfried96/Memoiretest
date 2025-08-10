#!/usr/bin/env python3
"""
Générateur de données synthétiques pour le système de surveillance.

Ce script génère des scènes de surveillance synthétiques avec:
- Environnements de magasins variés
- Objets et personnes réalistes
- Comportements normaux et suspects
- Annotations automatiques
- Métadonnées enrichies
"""

import os
import sys
import json
import yaml
import argparse
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import random
from tqdm.auto import tqdm
import logging
import math

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class SyntheticConfig:
    """Configuration pour la génération de données synthétiques."""
    
    # Paramètres généraux
    output_dir: str = "data/synthetic"
    num_scenes: int = 10000
    image_size: Tuple[int, int] = (640, 640)
    
    # Environnements de magasins
    store_types: List[str] = field(default_factory=lambda: [
        'grocery_store', 'electronics_store', 'clothing_store', 
        'pharmacy', 'bookstore', 'hardware_store'
    ])
    
    # Conditions d'éclairage
    lighting_conditions: List[str] = field(default_factory=lambda: [
        'bright', 'normal', 'dim', 'mixed', 'artificial'
    ])
    
    # Densités de foule
    crowd_densities: Dict[str, Tuple[int, int]] = field(default_factory=lambda: {
        'empty': (0, 1),
        'low': (1, 3),
        'medium': (3, 6),
        'high': (6, 10)
    })
    
    # Distribution des comportements
    behavior_distribution: Dict[str, float] = field(default_factory=lambda: {
        'normal_shopping': 0.70,
        'browsing': 0.15,
        'suspicious_movement': 0.10,
        'item_concealment': 0.04,
        'potential_theft': 0.01
    })
    
    # Classes d'objets
    object_classes: Dict[int, str] = field(default_factory=lambda: {
        0: 'person', 1: 'handbag', 2: 'backpack', 3: 'suitcase',
        4: 'bottle', 5: 'cup', 6: 'cell_phone', 7: 'book',
        8: 'shopping_cart', 9: 'suspicious_object'
    })
    
    # Paramètres de variation
    color_variation: float = 0.3
    size_variation: float = 0.4
    position_variation: float = 0.2
    
    # Paramètres de qualité
    noise_level: float = 0.02
    blur_probability: float = 0.1
    compression_quality: Tuple[int, int] = (85, 98)


class BackgroundGenerator:
    """Générateur de fonds de magasins."""
    
    def __init__(self, config: SyntheticConfig):
        self.config = config
        
    def generate_store_background(self, store_type: str, lighting: str) -> np.ndarray:
        """Génère un fond de magasin selon le type et l'éclairage."""
        width, height = self.config.image_size
        
        # Couleurs de base selon le type de magasin
        base_colors = {
            'grocery_store': [(240, 245, 250), (220, 225, 230)],
            'electronics_store': [(200, 210, 220), (180, 190, 200)],
            'clothing_store': [(250, 240, 235), (230, 220, 215)],
            'pharmacy': [(245, 250, 255), (235, 240, 245)],
            'bookstore': [(240, 235, 220), (220, 215, 200)],
            'hardware_store': [(210, 215, 200), (190, 195, 180)]
        }
        
        colors = base_colors.get(store_type, [(240, 240, 240), (220, 220, 220)])
        base_color = random.choice(colors)
        
        # Création du dégradé de fond
        background = np.zeros((height, width, 3), dtype=np.uint8)
        
        for y in range(height):
            gradient_factor = y / height
            color = [
                int(base_color[i] * (1 - gradient_factor * 0.2))
                for i in range(3)
            ]
            background[y, :] = color
        
        # Ajout de textures selon le type de magasin
        background = self._add_store_fixtures(background, store_type)
        
        # Ajustement de l'éclairage
        background = self._apply_lighting(background, lighting)
        
        return background
    
    def _add_store_fixtures(self, background: np.ndarray, store_type: str) -> np.ndarray:
        """Ajoute des éléments fixes du magasin (rayons, comptoirs, etc.)."""
        height, width = background.shape[:2]
        
        # Rayons verticaux (étagères)
        num_aisles = random.randint(2, 4)
        for i in range(num_aisles):
            x = int(width * (0.2 + i * 0.2))
            shelf_width = random.randint(15, 25)
            shelf_color = tuple(max(0, c - random.randint(30, 50)) for c in [180, 180, 180])
            
            cv2.rectangle(background, 
                         (x, 0), (x + shelf_width, height),
                         shelf_color, -1)
            
            # Produits sur les rayons
            for shelf_y in range(50, height - 100, 80):
                for item_x in range(x + 5, x + shelf_width - 5, 15):
                    item_color = tuple(random.randint(100, 250) for _ in range(3))
                    cv2.rectangle(background,
                                 (item_x, shelf_y), (item_x + 10, shelf_y + 30),
                                 item_color, -1)
        
        # Zone de caisse (si approprié)
        if store_type in ['grocery_store', 'electronics_store', 'pharmacy']:
            checkout_y = height - 80
            checkout_color = tuple(max(0, c - 40) for c in [200, 200, 200])
            cv2.rectangle(background,
                         (width - 150, checkout_y), (width, height),
                         checkout_color, -1)
        
        return background
    
    def _apply_lighting(self, image: np.ndarray, lighting: str) -> np.ndarray:
        """Applique des effets d'éclairage."""
        if lighting == 'bright':
            # Éclairage lumineux
            image = cv2.convertScaleAbs(image, alpha=1.2, beta=20)
        elif lighting == 'dim':
            # Éclairage tamisé
            image = cv2.convertScaleAbs(image, alpha=0.7, beta=-30)
        elif lighting == 'mixed':
            # Éclairage inégal
            height, width = image.shape[:2]
            for i in range(3):  # 3 zones d'éclairage
                x1 = int(width * i / 3)
                x2 = int(width * (i + 1) / 3)
                brightness_factor = random.uniform(0.6, 1.3)
                image[:, x1:x2] = cv2.convertScaleAbs(
                    image[:, x1:x2], alpha=brightness_factor, beta=0
                )
        elif lighting == 'artificial':
            # Éclairage artificiel avec teinte jaunâtre
            image[:, :, 0] = np.clip(image[:, :, 0] * 0.9, 0, 255)  # Réduire le bleu
            image[:, :, 1] = np.clip(image[:, :, 1] * 1.1, 0, 255)  # Augmenter le vert
        
        return image


class ObjectGenerator:
    """Générateur d'objets et de personnes."""
    
    def __init__(self, config: SyntheticConfig):
        self.config = config
        
    def generate_person(self, position: Tuple[int, int], size_factor: float = 1.0) -> Tuple[np.ndarray, Dict]:
        """Génère une personne avec ses caractéristiques."""
        base_width = int(60 * size_factor)
        base_height = int(160 * size_factor)
        
        # Variation de taille
        width = int(base_width * random.uniform(0.8, 1.2))
        height = int(base_height * random.uniform(0.8, 1.2))
        
        x, y = position
        
        # Création de la silhouette de personne
        person_mask = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Corps (rectangle arrondi)
        body_color = tuple(random.randint(80, 180) for _ in range(3))
        cv2.rectangle(person_mask, 
                     (width//4, height//3), (3*width//4, height),
                     body_color, -1)
        
        # Tête (cercle)
        head_center = (width//2, height//6)
        head_radius = width//6
        head_color = tuple(random.randint(150, 220) for _ in range(3))
        cv2.circle(person_mask, head_center, head_radius, head_color, -1)
        
        # Bras (rectangles)
        arm_color = body_color
        cv2.rectangle(person_mask,
                     (0, height//3), (width//4, 2*height//3),
                     arm_color, -1)
        cv2.rectangle(person_mask,
                     (3*width//4, height//3), (width, 2*height//3),
                     arm_color, -1)
        
        # Jambes (rectangles)
        leg_color = tuple(random.randint(50, 120) for _ in range(3))
        cv2.rectangle(person_mask,
                     (width//3, 2*height//3), (width//2, height),
                     leg_color, -1)
        cv2.rectangle(person_mask,
                     (width//2, 2*height//3), (2*width//3, height),
                     leg_color, -1)
        
        # Métadonnées de la personne
        person_metadata = {
            'type': 'person',
            'size': (width, height),
            'position': position,
            'attributes': {
                'age_group': random.choice(['child', 'adult', 'elderly']),
                'clothing_color': body_color,
                'carrying_items': random.choice([True, False]),
                'movement_speed': random.choice(['slow', 'normal', 'fast'])
            }
        }
        
        return person_mask, person_metadata
    
    def generate_object(self, obj_class: int, position: Tuple[int, int], 
                       size_factor: float = 1.0) -> Tuple[np.ndarray, Dict]:
        """Génère un objet selon sa classe."""
        class_name = self.config.object_classes.get(obj_class, 'unknown')
        
        # Dimensions de base selon la classe
        base_dimensions = {
            'handbag': (40, 30), 'backpack': (35, 45), 'suitcase': (60, 80),
            'bottle': (15, 40), 'cup': (20, 25), 'cell_phone': (15, 25),
            'book': (25, 35), 'shopping_cart': (80, 100), 'suspicious_object': (30, 30)
        }
        
        base_width, base_height = base_dimensions.get(class_name, (30, 30))
        
        # Application du facteur de taille et variation
        width = int(base_width * size_factor * random.uniform(0.7, 1.3))
        height = int(base_height * size_factor * random.uniform(0.7, 1.3))
        
        x, y = position
        
        # Création de l'objet
        object_mask = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Couleur selon le type d'objet
        color_ranges = {
            'handbag': [(100, 50, 50), (200, 100, 100)],
            'backpack': [(50, 50, 100), (100, 100, 200)],
            'suitcase': [(80, 80, 80), (150, 150, 150)],
            'bottle': [(0, 100, 0), (50, 200, 50)],
            'cup': [(200, 200, 200), (255, 255, 255)],
            'cell_phone': [(20, 20, 20), (60, 60, 60)],
            'book': [(150, 100, 50), (200, 150, 100)],
            'shopping_cart': [(180, 180, 180), (220, 220, 220)],
            'suspicious_object': [(100, 0, 0), (200, 50, 50)]
        }
        
        color_range = color_ranges.get(class_name, [(100, 100, 100), (200, 200, 200)])
        color = tuple(random.randint(color_range[0][i], color_range[1][i]) for i in range(3))
        
        # Forme selon le type
        if class_name in ['handbag', 'backpack', 'suitcase', 'book']:
            cv2.rectangle(object_mask, (2, 2), (width-2, height-2), color, -1)
        elif class_name in ['bottle', 'cup']:
            # Forme cylindrique approximative
            cv2.rectangle(object_mask, (width//4, 0), (3*width//4, height), color, -1)
            cv2.ellipse(object_mask, (width//2, height//8), (width//4, height//16), 0, 0, 360, color, -1)
        elif class_name == 'cell_phone':
            cv2.rectangle(object_mask, (1, 1), (width-1, height-1), color, -1)
            # Écran
            screen_color = (50, 50, 100)
            cv2.rectangle(object_mask, (3, 3), (width-3, height-3), screen_color, -1)
        elif class_name == 'shopping_cart':
            # Structure du caddie
            cv2.rectangle(object_mask, (10, height//2), (width-10, height-10), color, 2)
            cv2.rectangle(object_mask, (5, height//3), (width-5, height//2), color, -1)
        else:
            # Forme générique
            cv2.rectangle(object_mask, (0, 0), (width, height), color, -1)
        
        # Métadonnées de l'objet
        object_metadata = {
            'type': class_name,
            'class_id': obj_class,
            'size': (width, height),
            'position': position,
            'color': color,
            'attributes': {
                'suspicious': class_name == 'suspicious_object',
                'portable': class_name in ['handbag', 'backpack', 'bottle', 'cup', 'cell_phone', 'book']
            }
        }
        
        return object_mask, object_metadata


class BehaviorSimulator:
    """Simulateur de comportements dans les scènes."""
    
    def __init__(self, config: SyntheticConfig):
        self.config = config
        
    def generate_scene_behavior(self, num_people: int, store_type: str) -> Dict:
        """Génère le comportement global d'une scène."""
        
        # Sélection du comportement principal
        behaviors = list(self.config.behavior_distribution.keys())
        weights = list(self.config.behavior_distribution.values())
        main_behavior = np.random.choice(behaviors, p=weights)
        
        # Génération des trajectoires et interactions
        scene_behavior = {
            'main_behavior': main_behavior,
            'people_behaviors': [],
            'interactions': [],
            'suspicious_indicators': [],
            'temporal_events': []
        }
        
        for person_id in range(num_people):
            person_behavior = self._generate_person_behavior(person_id, main_behavior, store_type)
            scene_behavior['people_behaviors'].append(person_behavior)
            
            # Détection d'indicateurs suspects
            if person_behavior['behavior_type'] in ['suspicious_movement', 'item_concealment', 'potential_theft']:
                scene_behavior['suspicious_indicators'].append({
                    'person_id': person_id,
                    'indicator_type': person_behavior['behavior_type'],
                    'confidence': person_behavior['confidence'],
                    'timestamp': random.uniform(0, 30)  # Temps dans la séquence
                })
        
        # Génération d'interactions entre personnes
        if num_people > 1:
            num_interactions = random.randint(0, min(2, num_people // 2))
            for _ in range(num_interactions):
                person1, person2 = random.sample(range(num_people), 2)
                interaction = {
                    'participants': [person1, person2],
                    'interaction_type': random.choice(['conversation', 'passing_by', 'group_movement']),
                    'duration': random.uniform(2, 15),
                    'suspicious': random.choice([True, False]) if main_behavior != 'normal_shopping' else False
                }
                scene_behavior['interactions'].append(interaction)
        
        return scene_behavior
    
    def _generate_person_behavior(self, person_id: int, scene_behavior: str, store_type: str) -> Dict:
        """Génère le comportement d'une personne individuelle."""
        
        # Influence du comportement de scène sur le comportement individuel
        if scene_behavior == 'normal_shopping':
            individual_behaviors = ['normal_shopping', 'browsing']
            weights = [0.8, 0.2]
        elif scene_behavior == 'suspicious_movement':
            individual_behaviors = ['normal_shopping', 'suspicious_movement', 'browsing']
            weights = [0.5, 0.3, 0.2]
        else:
            # Comportement selon la distribution globale
            individual_behaviors = list(self.config.behavior_distribution.keys())
            weights = list(self.config.behavior_distribution.values())
        
        behavior_type = np.random.choice(individual_behaviors, p=weights)
        
        # Paramètres spécifiques au comportement
        behavior_params = self._get_behavior_parameters(behavior_type, store_type)
        
        person_behavior = {
            'person_id': person_id,
            'behavior_type': behavior_type,
            'confidence': behavior_params['confidence'],
            'movement_pattern': behavior_params['movement_pattern'],
            'interaction_level': behavior_params['interaction_level'],
            'focus_areas': behavior_params['focus_areas'],
            'duration': behavior_params['duration'],
            'attributes': behavior_params['attributes']
        }
        
        return person_behavior
    
    def _get_behavior_parameters(self, behavior_type: str, store_type: str) -> Dict:
        """Retourne les paramètres spécifiques à un type de comportement."""
        
        parameters = {
            'normal_shopping': {
                'confidence': random.uniform(0.8, 0.95),
                'movement_pattern': 'structured',
                'interaction_level': 'medium',
                'focus_areas': ['products', 'shelves', 'checkout'],
                'duration': random.uniform(300, 1800),  # 5-30 minutes
                'attributes': {
                    'shopping_list_following': True,
                    'price_checking': random.choice([True, False]),
                    'social_interaction': random.choice([True, False])
                }
            },
            'browsing': {
                'confidence': random.uniform(0.7, 0.9),
                'movement_pattern': 'wandering',
                'interaction_level': 'low',
                'focus_areas': ['products', 'displays'],
                'duration': random.uniform(180, 900),  # 3-15 minutes
                'attributes': {
                    'indecisive_behavior': True,
                    'prolonged_examining': True,
                    'frequent_direction_changes': True
                }
            },
            'suspicious_movement': {
                'confidence': random.uniform(0.6, 0.8),
                'movement_pattern': 'erratic',
                'interaction_level': 'low',
                'focus_areas': ['exits', 'cameras', 'staff'],
                'duration': random.uniform(60, 300),  # 1-5 minutes
                'attributes': {
                    'avoiding_staff': True,
                    'camera_awareness': True,
                    'nervous_behavior': True,
                    'frequent_looking_around': True
                }
            },
            'item_concealment': {
                'confidence': random.uniform(0.7, 0.9),
                'movement_pattern': 'deliberate',
                'interaction_level': 'very_low',
                'focus_areas': ['blind_spots', 'products', 'bags'],
                'duration': random.uniform(30, 180),  # 30s-3 minutes
                'attributes': {
                    'concealment_actions': True,
                    'bag_manipulation': True,
                    'position_strategic': True,
                    'staff_avoidance': True
                }
            },
            'potential_theft': {
                'confidence': random.uniform(0.8, 0.95),
                'movement_pattern': 'strategic',
                'interaction_level': 'minimal',
                'focus_areas': ['exits', 'high_value_items', 'blind_spots'],
                'duration': random.uniform(60, 240),  # 1-4 minutes
                'attributes': {
                    'theft_indicators': True,
                    'exit_planning': True,
                    'staff_monitoring': True,
                    'camera_avoidance': True
                }
            }
        }
        
        return parameters.get(behavior_type, parameters['normal_shopping'])


class SceneComposer:
    """Compositeur de scènes complètes."""
    
    def __init__(self, config: SyntheticConfig):
        self.config = config
        self.background_gen = BackgroundGenerator(config)
        self.object_gen = ObjectGenerator(config)
        self.behavior_sim = BehaviorSimulator(config)
        
    def compose_scene(self, scene_id: int) -> Tuple[np.ndarray, List[Dict], Dict]:
        """Compose une scène complète avec tous ses éléments."""
        
        # Sélection des paramètres de scène
        store_type = random.choice(self.config.store_types)
        lighting = random.choice(self.config.lighting_conditions)
        
        # Sélection de la densité de foule
        crowd_density = random.choice(list(self.config.crowd_densities.keys()))
        min_people, max_people = self.config.crowd_densities[crowd_density]
        num_people = random.randint(min_people, max_people)
        
        # Génération du fond
        background = self.background_gen.generate_store_background(store_type, lighting)
        
        # Génération du comportement de scène
        scene_behavior = self.behavior_sim.generate_scene_behavior(num_people, store_type)
        
        # Composition de la scène
        scene_image = background.copy()
        annotations = []
        object_metadata = []
        
        width, height = self.config.image_size
        
        # Placement des personnes
        occupied_areas = []  # Pour éviter les chevauchements
        
        for person_id in range(num_people):
            # Recherche d'une position libre
            max_attempts = 20
            for attempt in range(max_attempts):
                x = random.randint(50, width - 150)
                y = random.randint(100, height - 200)
                
                # Vérification des chevauchements
                person_rect = (x, y, x + 100, y + 160)
                overlap = any(self._rectangles_overlap(person_rect, occupied) for occupied in occupied_areas)
                
                if not overlap or attempt == max_attempts - 1:
                    break
            
            # Génération de la personne
            person_size_factor = random.uniform(0.8, 1.2)
            person_mask, person_meta = self.object_gen.generate_person((x, y), person_size_factor)
            
            # Intégration dans la scène
            self._blend_object(scene_image, person_mask, (x, y))
            
            # Annotation YOLO
            person_width, person_height = person_meta['size']
            center_x = (x + person_width / 2) / width
            center_y = (y + person_height / 2) / height
            norm_width = person_width / width
            norm_height = person_height / height
            
            annotations.append({
                'class_id': 0,  # person
                'center_x': center_x,
                'center_y': center_y,
                'width': norm_width,
                'height': norm_height,
                'confidence': random.uniform(0.8, 0.95)
            })
            
            object_metadata.append(person_meta)
            occupied_areas.append((x, y, x + person_width, y + person_height))
            
            # Objets associés à la personne (sacs, etc.)
            if person_meta['attributes']['carrying_items']:
                num_items = random.randint(1, 2)
                for _ in range(num_items):
                    item_class = random.choice([1, 2, 4, 5, 6, 7])  # handbag, backpack, etc.
                    
                    # Position relative à la personne
                    item_x = x + random.randint(-20, 20)
                    item_y = y + random.randint(person_height//2, person_height - 20)
                    
                    if 0 <= item_x < width - 50 and 0 <= item_y < height - 50:
                        item_mask, item_meta = self.object_gen.generate_object(
                            item_class, (item_x, item_y), 0.6
                        )
                        
                        self._blend_object(scene_image, item_mask, (item_x, item_y))
                        
                        # Annotation
                        item_width, item_height = item_meta['size']
                        center_x = (item_x + item_width / 2) / width
                        center_y = (item_y + item_height / 2) / height
                        norm_width = item_width / width
                        norm_height = item_height / height
                        
                        annotations.append({
                            'class_id': item_class,
                            'center_x': center_x,
                            'center_y': center_y,
                            'width': norm_width,
                            'height': norm_height,
                            'confidence': random.uniform(0.6, 0.9)
                        })
                        
                        object_metadata.append(item_meta)
        
        # Objets indépendants (caddie, objets suspects, etc.)
        num_independent_objects = random.randint(0, 3)
        for _ in range(num_independent_objects):
            obj_class = random.choice([8, 9])  # shopping_cart, suspicious_object
            
            # Position libre
            max_attempts = 15
            for attempt in range(max_attempts):
                x = random.randint(20, width - 100)
                y = random.randint(20, height - 120)
                
                obj_rect = (x, y, x + 80, y + 100)
                overlap = any(self._rectangles_overlap(obj_rect, occupied) for occupied in occupied_areas)
                
                if not overlap or attempt == max_attempts - 1:
                    break
            
            obj_mask, obj_meta = self.object_gen.generate_object(obj_class, (x, y))
            self._blend_object(scene_image, obj_mask, (x, y))
            
            # Annotation
            obj_width, obj_height = obj_meta['size']
            center_x = (x + obj_width / 2) / width
            center_y = (y + obj_height / 2) / height
            norm_width = obj_width / width
            norm_height = obj_height / height
            
            annotations.append({
                'class_id': obj_class,
                'center_x': center_x,
                'center_y': center_y,
                'width': norm_width,
                'height': norm_height,
                'confidence': random.uniform(0.7, 0.95)
            })
            
            object_metadata.append(obj_meta)
        
        # Application d'effets de post-traitement
        scene_image = self._apply_post_processing(scene_image)
        
        # Métadonnées de la scène
        scene_metadata = {
            'scene_id': scene_id,
            'timestamp': datetime.now().isoformat(),
            'store_type': store_type,
            'lighting_condition': lighting,
            'crowd_density': crowd_density,
            'num_people': num_people,
            'num_objects': len(annotations) - num_people,  # Objets non-personnes
            'behavior_analysis': scene_behavior,
            'objects': object_metadata,
            'image_size': self.config.image_size,
            'quality_metrics': {
                'noise_level': self.config.noise_level,
                'blur_applied': random.random() < self.config.blur_probability,
                'compression_quality': random.randint(*self.config.compression_quality)
            }
        }
        
        return scene_image, annotations, scene_metadata
    
    def _blend_object(self, scene: np.ndarray, obj_mask: np.ndarray, position: Tuple[int, int]):
        """Intègre un objet dans la scène avec mélange naturel."""
        x, y = position
        obj_height, obj_width = obj_mask.shape[:2]
        scene_height, scene_width = scene.shape[:2]
        
        # Calcul des limites de placement
        x_end = min(x + obj_width, scene_width)
        y_end = min(y + obj_height, scene_height)
        
        if x >= scene_width or y >= scene_height or x_end <= x or y_end <= y:
            return
        
        # Ajustement des dimensions si nécessaire
        obj_width_adj = x_end - x
        obj_height_adj = y_end - y
        
        obj_mask_adj = obj_mask[:obj_height_adj, :obj_width_adj]
        
        # Masque pour le mélange (pixels non-noirs)
        mask = np.any(obj_mask_adj > 10, axis=2)
        
        # Application avec mélange
        scene[y:y_end, x:x_end][mask] = obj_mask_adj[mask]
    
    def _rectangles_overlap(self, rect1: Tuple[int, int, int, int], 
                          rect2: Tuple[int, int, int, int]) -> bool:
        """Vérifie si deux rectangles se chevauchent."""
        x1, y1, x2, y2 = rect1
        x3, y3, x4, y4 = rect2
        
        return not (x2 <= x3 or x4 <= x1 or y2 <= y3 or y4 <= y1)
    
    def _apply_post_processing(self, image: np.ndarray) -> np.ndarray:
        """Applique des effets de post-traitement pour plus de réalisme."""
        
        # Bruit
        if self.config.noise_level > 0:
            noise = np.random.normal(0, self.config.noise_level * 255, image.shape)
            image = np.clip(image.astype(float) + noise, 0, 255).astype(np.uint8)
        
        # Flou occasionnel
        if random.random() < self.config.blur_probability:
            kernel_size = random.choice([3, 5])
            image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        
        # Ajustements de couleur subtils
        if random.random() < 0.3:
            # Ajustement de saturation
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            saturation_factor = random.uniform(0.9, 1.1)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_factor, 0, 255)
            image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        return image


class SyntheticDatasetGenerator:
    """Générateur principal de dataset synthétique."""
    
    def __init__(self, config: SyntheticConfig):
        self.config = config
        self.scene_composer = SceneComposer(config)
        
    def generate_dataset(self) -> Dict:
        """Génère un dataset complet."""
        logger.info(f"Génération de {self.config.num_scenes} scènes synthétiques...")
        
        # Création des répertoires
        output_path = Path(self.config.output_dir)
        images_dir = output_path / "images"
        labels_dir = output_path / "labels"
        metadata_dir = output_path / "metadata"
        
        for dir_path in [images_dir, labels_dir, metadata_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        generation_stats = {
            'total_scenes': self.config.num_scenes,
            'generated_scenes': 0,
            'total_annotations': 0,
            'class_distribution': {},
            'behavior_distribution': {},
            'store_type_distribution': {},
            'lighting_distribution': {},
            'errors': []
        }
        
        # Génération des scènes
        for scene_id in tqdm(range(self.config.num_scenes), desc="Génération scènes"):
            try:
                scene_image, annotations, scene_metadata = self.scene_composer.compose_scene(scene_id)
                
                # Sauvegarde de l'image
                image_filename = f"synthetic_scene_{scene_id:06d}.jpg"
                image_path = images_dir / image_filename
                
                # Application de la compression JPEG
                quality = scene_metadata['quality_metrics']['compression_quality']
                cv2.imwrite(str(image_path), scene_image, 
                           [cv2.IMWRITE_JPEG_QUALITY, quality])
                
                # Sauvegarde des annotations YOLO
                label_filename = f"synthetic_scene_{scene_id:06d}.txt"
                label_path = labels_dir / label_filename
                
                with open(label_path, 'w') as f:
                    for ann in annotations:
                        f.write(f"{ann['class_id']} {ann['center_x']:.6f} {ann['center_y']:.6f} "
                               f"{ann['width']:.6f} {ann['height']:.6f}\\n")
                
                # Sauvegarde des métadonnées
                metadata_filename = f"synthetic_scene_{scene_id:06d}.json"
                metadata_path = metadata_dir / metadata_filename
                
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(scene_metadata, f, ensure_ascii=False, indent=2)
                
                # Mise à jour des statistiques
                generation_stats['generated_scenes'] += 1
                generation_stats['total_annotations'] += len(annotations)
                
                # Distribution des classes
                for ann in annotations:
                    class_id = ann['class_id']
                    class_name = self.config.object_classes.get(class_id, f'class_{class_id}')
                    generation_stats['class_distribution'][class_name] = \
                        generation_stats['class_distribution'].get(class_name, 0) + 1
                
                # Distribution des comportements
                main_behavior = scene_metadata['behavior_analysis']['main_behavior']
                generation_stats['behavior_distribution'][main_behavior] = \
                    generation_stats['behavior_distribution'].get(main_behavior, 0) + 1
                
                # Distribution des types de magasins
                store_type = scene_metadata['store_type']
                generation_stats['store_type_distribution'][store_type] = \
                    generation_stats['store_type_distribution'].get(store_type, 0) + 1
                
                # Distribution de l'éclairage
                lighting = scene_metadata['lighting_condition']
                generation_stats['lighting_distribution'][lighting] = \
                    generation_stats['lighting_distribution'].get(lighting, 0) + 1
                
            except Exception as e:
                logger.error(f"Erreur lors de la génération de la scène {scene_id}: {e}")
                generation_stats['errors'].append({
                    'scene_id': scene_id,
                    'error': str(e)
                })
        
        # Création du fichier dataset.yaml
        dataset_yaml = {
            'path': str(output_path),
            'train': 'images',
            'val': 'images',  # À séparer ultérieurement
            'names': {i: name for i, name in self.config.object_classes.items()}
        }
        
        with open(output_path / 'dataset.yaml', 'w') as f:
            yaml.dump(dataset_yaml, f, default_flow_style=False)
        
        # Sauvegarde des statistiques de génération
        with open(output_path / 'generation_stats.json', 'w', encoding='utf-8') as f:
            json.dump(generation_stats, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Génération terminée: {generation_stats['generated_scenes']} scènes, "
                   f"{generation_stats['total_annotations']} annotations")
        
        return generation_stats


def main():
    """Fonction principale avec interface en ligne de commande."""
    parser = argparse.ArgumentParser(description='Générateur de données synthétiques de surveillance')
    parser.add_argument('--output', '-o', default='data/synthetic', 
                       help='Répertoire de sortie')
    parser.add_argument('--num-scenes', '-n', type=int, default=1000,
                       help='Nombre de scènes à générer')
    parser.add_argument('--config', '-c', help='Fichier de configuration YAML')
    parser.add_argument('--size', nargs=2, type=int, default=[640, 640],
                       help='Taille des images (largeur hauteur)')
    parser.add_argument('--store-types', nargs='+', 
                       default=['grocery_store', 'electronics_store', 'clothing_store'],
                       help='Types de magasins à générer')
    parser.add_argument('--seed', type=int, help='Graine aléatoire pour la reproductibilité')
    
    args = parser.parse_args()
    
    # Configuration de la graine aléatoire
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
    
    # Chargement de la configuration
    if args.config:
        with open(args.config, 'r') as f:
            config_data = yaml.safe_load(f)
        config = SyntheticConfig(**config_data)
    else:
        config = SyntheticConfig()
    
    # Application des arguments
    config.output_dir = args.output
    config.num_scenes = args.num_scenes
    config.image_size = tuple(args.size)
    config.store_types = args.store_types
    
    # Génération du dataset
    generator = SyntheticDatasetGenerator(config)
    stats = generator.generate_dataset()
    
    # Affichage du résumé
    print("\\n=== GÉNÉRATION TERMINÉE ===")
    print(f"Scènes générées: {stats['generated_scenes']}/{stats['total_scenes']}")
    print(f"Annotations totales: {stats['total_annotations']}")
    print(f"Erreurs: {len(stats['errors'])}")
    
    print("\\n=== DISTRIBUTION DES CLASSES ===")
    for class_name, count in stats['class_distribution'].items():
        print(f"{class_name}: {count}")
    
    print("\\n=== DISTRIBUTION DES COMPORTEMENTS ===")
    for behavior, count in stats['behavior_distribution'].items():
        print(f"{behavior}: {count}")
    
    print(f"\\n📁 Dataset sauvegardé dans: {args.output}")


if __name__ == '__main__':
    main()