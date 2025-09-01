"""
Tracker multi-objets BYTE pour la surveillance.
Version simplifiée pour intégration avec le système VLM.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from loguru import logger


@dataclass
class TrackedObject:
    """Objet suivi avec historique."""
    track_id: int
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float
    class_name: str
    age: int = 1
    hits: int = 1
    hit_streak: int = 1
    time_since_update: int = 0
    velocity: Optional[List[float]] = None  # [dx, dy]
    trajectory: List[List[float]] = None
    
    def __post_init__(self):
        if self.trajectory is None:
            self.trajectory = [self.bbox.copy()]


class BYTETracker:
    """
    Tracker BYTE simplifié pour surveillance intelligente.
    Maintient l'identité des objets détectés entre les frames.
    """
    
    def __init__(
        self,
        track_thresh: float = 0.3,  # Plus permissif pour détecter
        track_buffer: int = 60,     # Plus de frames avant de perdre un track
        match_thresh: float = 0.6,  # Seuil de correspondance plus permissif
        frame_rate: int = 30
    ):
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.frame_rate = frame_rate
        
        # État du tracker
        self.frame_id = 0
        self.tracked_objects: Dict[int, TrackedObject] = {}
        self.next_id = 1
        
        # Statistiques
        self.stats = {
            "total_tracks": 0,
            "active_tracks": 0,
            "lost_tracks": 0
        }
        
        logger.info("🎯 BYTE Tracker initialisé")
    
    def update(self, detections: List) -> List[TrackedObject]:
        """
        Met à jour le tracker avec les nouvelles détections.
        
        Args:
            detections: Liste des détections du frame actuel
            
        Returns:
            Liste des objets suivis mis à jour
        """
        self.frame_id += 1
        
        # Conversion des détections en format interne
        current_detections = self._convert_detections(detections)
        
        # Mise à jour des objets existants
        self._update_existing_tracks(current_detections)
        
        # Création de nouveaux tracks
        self._create_new_tracks(current_detections)
        
        # Nettoyage des tracks perdus
        self._cleanup_lost_tracks()
        
        # Mise à jour des statistiques
        self._update_stats()
        
        return list(self.tracked_objects.values())
    
    def _convert_detections(self, detections) -> List[Dict]:
        """Convertit les détections en format interne."""
        converted = []
        
        for detection in detections:
            if hasattr(detection, 'bbox'):
                # Format Detection de notre système
                bbox = [
                    detection.bbox.x1,
                    detection.bbox.y1, 
                    detection.bbox.x2,
                    detection.bbox.y2
                ]
                
                converted.append({
                    "bbox": bbox,
                    "confidence": detection.confidence,
                    "class_name": detection.class_name,
                    "matched": False
                })
        
        return converted
    
    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calcule l'IoU entre deux bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Intersection
        x1_inter = max(x1_1, x1_2)
        y1_inter = max(y1_1, y1_2)
        x2_inter = min(x2_1, x2_2)
        y2_inter = min(y2_1, y2_2)
        
        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0
        
        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        
        # Aire des bounding boxes
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # Union
        union_area = area1 + area2 - inter_area
        
        if union_area <= 0:
            return 0.0
        
        return inter_area / union_area
    
    def _calculate_distance(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calcule la distance entre les centres de deux bounding boxes."""
        center1 = [(bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2]
        center2 = [(bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2]
        
        dx = center1[0] - center2[0]
        dy = center1[1] - center2[1]
        
        return np.sqrt(dx*dx + dy*dy)
    
    def _update_existing_tracks(self, detections: List[Dict]):
        """Met à jour les tracks existants avec les détections."""
        
        # Incrémenter l'âge et time_since_update pour tous les tracks
        for track in self.tracked_objects.values():
            track.age += 1
            track.time_since_update += 1
        
        # Association détections <-> tracks existants
        for track_id, track in self.tracked_objects.items():
            best_match = None
            best_score = 0.0
            
            for detection in detections:
                if detection["matched"]:
                    continue
                
                # Filtre par classe
                if detection["class_name"] != track.class_name:
                    continue
                
                # Calcul du score de correspondance (IoU + distance)
                iou = self._calculate_iou(track.bbox, detection["bbox"])
                distance = self._calculate_distance(track.bbox, detection["bbox"])
                
                # Score combiné (privilégier IoU, pénaliser distance)
                score = iou - (distance / 1000)  # Normalisation empirique
                
                if score > best_score and score > self.match_thresh:
                    best_score = score
                    best_match = detection
            
            # Mise à jour du track si correspondance trouvée
            if best_match:
                # Calcul de la vélocité
                old_center = [(track.bbox[0] + track.bbox[2]) / 2, 
                             (track.bbox[1] + track.bbox[3]) / 2]
                new_center = [(best_match["bbox"][0] + best_match["bbox"][2]) / 2,
                             (best_match["bbox"][1] + best_match["bbox"][3]) / 2]
                
                velocity = [new_center[0] - old_center[0], 
                           new_center[1] - old_center[1]]
                
                # Mise à jour
                track.bbox = best_match["bbox"]
                track.confidence = best_match["confidence"]
                track.velocity = velocity
                track.hits += 1
                track.hit_streak += 1
                track.time_since_update = 0
                track.trajectory.append(track.bbox.copy())
                
                # Marquer la détection comme utilisée
                best_match["matched"] = True
            else:
                # Pas de correspondance trouvée
                track.hit_streak = 0
    
    def _create_new_tracks(self, detections: List[Dict]):
        """Crée de nouveaux tracks pour les détections non matchées."""
        
        for detection in detections:
            if detection["matched"]:
                continue
            
            # Créer un nouveau track
            new_track = TrackedObject(
                track_id=self.next_id,
                bbox=detection["bbox"],
                confidence=detection["confidence"],
                class_name=detection["class_name"]
            )
            
            self.tracked_objects[self.next_id] = new_track
            self.next_id += 1
            self.stats["total_tracks"] += 1
            
            logger.debug(f"Nouveau track créé: ID {new_track.track_id} ({new_track.class_name})")
    
    def _cleanup_lost_tracks(self):
        """Supprime les tracks perdus depuis trop longtemps."""
        
        tracks_to_remove = []
        
        for track_id, track in self.tracked_objects.items():
            # Supprimer si pas de correspondance depuis track_buffer frames
            if track.time_since_update > self.track_buffer:
                tracks_to_remove.append(track_id)
                self.stats["lost_tracks"] += 1
                logger.debug(f"Track perdu: ID {track_id} (age: {track.age})")
        
        # Suppression effective
        for track_id in tracks_to_remove:
            del self.tracked_objects[track_id]
    
    def _update_stats(self):
        """Met à jour les statistiques."""
        self.stats["active_tracks"] = len(self.tracked_objects)
    
    def get_track_by_id(self, track_id: int) -> Optional[TrackedObject]:
        """Récupère un track par son ID."""
        return self.tracked_objects.get(track_id)
    
    def get_tracks_by_class(self, class_name: str) -> List[TrackedObject]:
        """Récupère tous les tracks d'une classe donnée."""
        return [track for track in self.tracked_objects.values() 
                if track.class_name == class_name]
    
    def get_person_tracks(self) -> List[TrackedObject]:
        """Récupère tous les tracks de personnes."""
        return self.get_tracks_by_class("person")
    
    def get_trajectory_data(self) -> Dict[int, List[List[float]]]:
        """Récupère les données de trajectoire pour tous les tracks."""
        return {track_id: track.trajectory 
                for track_id, track in self.tracked_objects.items()}
    
    def get_stats(self) -> Dict[str, Any]:
        """Récupère les statistiques du tracker."""
        return {
            **self.stats,
            "frame_id": self.frame_id,
            "next_track_id": self.next_id
        }
    
    def reset(self):
        """Remet à zéro le tracker."""
        self.frame_id = 0
        self.tracked_objects.clear()
        self.next_id = 1
        self.stats = {
            "total_tracks": 0,
            "active_tracks": 0,
            "lost_tracks": 0
        }
        logger.info("🔄 Tracker réinitialisé")