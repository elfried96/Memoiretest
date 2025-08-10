"""Système de suivi d'objets multi-algorithmes (ByteTrack/DeepSORT)."""

import time
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import cv2
from loguru import logger

from ...core.types import DetectedObject, Frame, BoundingBox
from ...utils.exceptions import TrackingError
from ...utils.performance import measure_time


class TrackerType(Enum):
    """Types de trackers disponibles."""
    BYTETRACK = "bytetrack"
    DEEPSORT = "deepsort"
    CENTROID = "centroid"


@dataclass
class TrackingState:
    """État d'un objet suivi."""
    track_id: int
    class_name: str
    bbox: BoundingBox
    confidence: float
    age: int = 0
    hits: int = 0
    time_since_update: int = 0
    velocity: Optional[Tuple[float, float]] = None
    trajectory: List[Tuple[int, int]] = field(default_factory=list)
    features: Optional[np.ndarray] = None
    
    def update_position(self, new_bbox: BoundingBox) -> None:
        """Mise à jour de la position et calcul de vélocité."""
        if self.bbox is not None:
            # Calcul de la vélocité
            old_center = self.bbox.center
            new_center = new_bbox.center
            self.velocity = (
                new_center[0] - old_center[0],
                new_center[1] - old_center[1]
            )
            
            # Ajout à la trajectoire
            self.trajectory.append(new_center)
            
            # Limitation de la trajectoire (garder 30 derniers points)
            if len(self.trajectory) > 30:
                self.trajectory = self.trajectory[-30:]
        
        self.bbox = new_bbox
        self.hits += 1
        self.time_since_update = 0
    
    @property
    def is_confirmed(self) -> bool:
        """Vérifie si le track est confirmé (suffisamment de hits)."""
        return self.hits >= 3
    
    @property
    def is_lost(self) -> bool:
        """Vérifie si le track est perdu (trop de temps sans update)."""
        return self.time_since_update > 30


class CentroidTracker:
    """Tracker simple basé sur les centroïdes."""
    
    def __init__(self, max_disappeared: int = 50, max_distance: float = 100.0):
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.next_object_id = 0
        self.objects = {}
        self.disappeared = {}
    
    def update(self, detections: List[DetectedObject]) -> Dict[int, TrackingState]:
        """Mise à jour du suivi avec nouvelles détections."""
        
        # Si pas de détections, incrémenter les compteurs de disparition
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                
                # Supprimer les objets disparus trop longtemps
                if self.disappeared[object_id] > self.max_disappeared:
                    self._deregister(object_id)
            
            return self.objects
        
        # Extraire les centroïdes des détections
        input_centroids = []
        for detection in detections:
            input_centroids.append(detection.bbox.center)
        
        # Si pas d'objets existants, enregistrer toutes les détections
        if len(self.objects) == 0:
            for i, detection in enumerate(detections):
                self._register(detection, input_centroids[i])
        
        # Sinon, associer détections existantes aux nouvelles
        else:
            object_centroids = [obj.bbox.center for obj in self.objects.values()]
            
            # Calculer la matrice de distances
            D = self._compute_distance_matrix(
                np.array(object_centroids),
                np.array(input_centroids)
            )
            
            # Association basée sur la distance minimale
            self._associate_detections(D, detections, input_centroids)
        
        return self.objects
    
    def _register(self, detection: DetectedObject, centroid: Tuple[int, int]) -> None:
        """Enregistrement d'un nouvel objet."""
        track_state = TrackingState(
            track_id=self.next_object_id,
            class_name=detection.class_name,
            bbox=detection.bbox,
            confidence=detection.confidence
        )
        track_state.trajectory.append(centroid)
        
        self.objects[self.next_object_id] = track_state
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1
    
    def _deregister(self, object_id: int) -> None:
        """Suppression d'un objet suivi."""
        del self.objects[object_id]
        del self.disappeared[object_id]
    
    def _compute_distance_matrix(
        self, 
        existing_centroids: np.ndarray, 
        new_centroids: np.ndarray
    ) -> np.ndarray:
        """Calcul de la matrice de distances entre centroïdes."""
        D = np.linalg.norm(
            existing_centroids[:, np.newaxis] - new_centroids, 
            axis=2
        )
        return D
    
    def _associate_detections(
        self,
        D: np.ndarray,
        detections: List[DetectedObject],
        input_centroids: List[Tuple[int, int]]
    ) -> None:
        """Association des détections aux objets existants."""
        
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]
        
        used_row_indices = set()
        used_col_indices = set()
        
        for (row, col) in zip(rows, cols):
            if row in used_row_indices or col in used_col_indices:
                continue
            
            if D[row, col] > self.max_distance:
                continue
            
            # Mise à jour de l'objet existant
            object_id = list(self.objects.keys())[row]
            self.objects[object_id].update_position(detections[col].bbox)
            self.objects[object_id].confidence = detections[col].confidence
            self.disappeared[object_id] = 0
            
            used_row_indices.add(row)
            used_col_indices.add(col)
        
        # Gérer les objets non associés
        unused_row_indices = set(range(0, D.shape[0])).difference(used_row_indices)
        unused_col_indices = set(range(0, D.shape[1])).difference(used_col_indices)
        
        # Incrémenter les compteurs de disparition
        if D.shape[0] >= D.shape[1]:
            for row in unused_row_indices:
                object_id = list(self.objects.keys())[row]
                self.disappeared[object_id] += 1
                
                if self.disappeared[object_id] > self.max_disappeared:
                    self._deregister(object_id)
        
        # Enregistrer nouvelles détections
        else:
            for col in unused_col_indices:
                self._register(detections[col], input_centroids[col])


class ByteTracker:
    """Implementation simplifiée de ByteTrack."""
    
    def __init__(
        self,
        track_thresh: float = 0.5,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        frame_rate: int = 30
    ):
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.frame_rate = frame_rate
        
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        self.frame_id = 0
        self.track_id_count = 0
    
    def update(self, detections: List[DetectedObject]) -> Dict[int, TrackingState]:
        """Mise à jour ByteTrack."""
        
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        
        # Séparation détections haute/basse confiance
        high_conf_dets = []
        low_conf_dets = []
        
        for det in detections:
            if det.confidence >= self.track_thresh:
                high_conf_dets.append(det)
            else:
                low_conf_dets.append(det)
        
        # Conversion en tracking states
        if high_conf_dets:
            det_states = [self._detection_to_state(det) for det in high_conf_dets]
        else:
            det_states = []
        
        # Association avec tracks existants
        matched_pairs, unmatched_dets, unmatched_tracks = self._associate(
            self.tracked_stracks, det_states
        )
        
        # Mise à jour des tracks appariés
        for track_idx, det_idx in matched_pairs:
            track = self.tracked_stracks[track_idx]
            det = det_states[det_idx]
            track.update_position(det.bbox)
            track.confidence = det.confidence
            activated_stracks.append(track)
        
        # Gestion des détections non appariées
        for det_idx in unmatched_dets:
            det = det_states[det_idx]
            det.track_id = self._get_new_track_id()
            activated_stracks.append(det)
        
        # Gestion des tracks perdus
        for track_idx in unmatched_tracks:
            track = self.tracked_stracks[track_idx]
            track.time_since_update += 1
            if track.time_since_update <= self.track_buffer:
                lost_stracks.append(track)
            else:
                removed_stracks.append(track)
        
        # Mise à jour des listes
        self.tracked_stracks = activated_stracks
        self.lost_stracks = lost_stracks
        self.removed_stracks = removed_stracks
        
        # Conversion en dictionnaire
        result = {}
        for track in self.tracked_stracks:
            if track.is_confirmed:
                result[track.track_id] = track
        
        return result
    
    def _detection_to_state(self, detection: DetectedObject) -> TrackingState:
        """Conversion détection en état de tracking."""
        return TrackingState(
            track_id=-1,  # Sera assigné plus tard
            class_name=detection.class_name,
            bbox=detection.bbox,
            confidence=detection.confidence
        )
    
    def _associate(
        self,
        tracks: List[TrackingState],
        detections: List[TrackingState]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Association tracks-détections basée sur IoU."""
        
        if len(tracks) == 0:
            return [], list(range(len(detections))), []
        
        if len(detections) == 0:
            return [], [], list(range(len(tracks)))
        
        # Calcul matrice IoU
        iou_matrix = np.zeros((len(tracks), len(detections)))
        
        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                iou_matrix[i, j] = self._calculate_iou(track.bbox, det.bbox)
        
        # Association par IoU maximale
        matched_pairs = []
        unmatched_tracks = list(range(len(tracks)))
        unmatched_detections = list(range(len(detections)))
        
        # Tant qu'il y a des associations possibles
        while len(unmatched_tracks) > 0 and len(unmatched_detections) > 0:
            # Trouver la meilleure association
            max_iou = 0
            best_track_idx = -1
            best_det_idx = -1
            
            for track_idx in unmatched_tracks:
                for det_idx in unmatched_detections:
                    iou = iou_matrix[track_idx, det_idx]
                    if iou > max_iou and iou > self.match_thresh:
                        max_iou = iou
                        best_track_idx = track_idx
                        best_det_idx = det_idx
            
            # Si aucune bonne association trouvée, arrêter
            if best_track_idx == -1:
                break
            
            # Enregistrer l'association
            matched_pairs.append((best_track_idx, best_det_idx))
            unmatched_tracks.remove(best_track_idx)
            unmatched_detections.remove(best_det_idx)
        
        return matched_pairs, unmatched_detections, unmatched_tracks
    
    def _calculate_iou(self, bbox1: BoundingBox, bbox2: BoundingBox) -> float:
        """Calcul de l'IoU entre deux boîtes englobantes."""
        
        # Coordonnées des boîtes
        x1_1, y1_1 = bbox1.x, bbox1.y
        x2_1, y2_1 = bbox1.x + bbox1.width, bbox1.y + bbox1.height
        
        x1_2, y1_2 = bbox2.x, bbox2.y
        x2_2, y2_2 = bbox2.x + bbox2.width, bbox2.y + bbox2.height
        
        # Intersection
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        inter_area = (xi2 - xi1) * (yi2 - yi1)
        
        # Union
        box1_area = bbox1.area
        box2_area = bbox2.area
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def _get_new_track_id(self) -> int:
        """Génération d'un nouvel ID de track."""
        self.track_id_count += 1
        return self.track_id_count


class MultiObjectTracker:
    """
    Tracker principal supportant plusieurs algorithmes.
    
    Features:
    - Support ByteTrack et CentroidTracker
    - Gestion automatique des trajectoires
    - Analyse comportementale intégrée
    - Métriques de performance
    """
    
    def __init__(
        self,
        tracker_type: TrackerType = TrackerType.BYTETRACK,
        max_disappeared: int = 50,
        max_distance: float = 100.0,
        track_buffer: int = 30,
        **kwargs
    ):
        self.tracker_type = tracker_type
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.track_buffer = track_buffer
        
        # Initialisation du tracker
        self.tracker = self._initialize_tracker(**kwargs)
        
        # Métriques
        self.stats = {
            "total_tracks": 0,
            "active_tracks": 0,
            "avg_track_duration": 0.0,
            "tracking_accuracy": 0.0
        }
        
        # Historique des tracks pour analyse comportementale
        self.track_history = {}
        
        logger.info(f"MultiObjectTracker initialisé avec {tracker_type.value}")
    
    def _initialize_tracker(self, **kwargs):
        """Initialisation du tracker selon le type."""
        
        if self.tracker_type == TrackerType.BYTETRACK:
            return ByteTracker(
                track_buffer=self.track_buffer,
                **kwargs
            )
        elif self.tracker_type == TrackerType.CENTROID:
            return CentroidTracker(
                max_disappeared=self.max_disappeared,
                max_distance=self.max_distance
            )
        else:
            raise TrackingError(f"Type de tracker non supporté: {self.tracker_type}")
    
    @measure_time
    def update(
        self, 
        detections: List[DetectedObject],
        frame_info: Optional[Dict[str, Any]] = None
    ) -> Dict[int, TrackingState]:
        """
        Mise à jour du suivi avec nouvelles détections.
        
        Args:
            detections: Liste des objets détectés
            frame_info: Informations additionnelles du frame
            
        Returns:
            Dictionnaire des états de tracking
        """
        
        try:
            # Filtrage des détections (personnes principalement)
            filtered_detections = self._filter_detections(detections)
            
            # Mise à jour du tracker
            tracked_objects = self.tracker.update(filtered_detections)
            
            # Mise à jour de l'historique et analyse comportementale
            self._update_track_history(tracked_objects, frame_info)
            
            # Mise à jour des statistiques
            self._update_stats(tracked_objects)
            
            logger.debug(
                f"Tracking: {len(tracked_objects)} objets actifs, "
                f"{len(filtered_detections)} détections traitées"
            )
            
            return tracked_objects
            
        except Exception as e:
            logger.error(f"Erreur dans le suivi: {e}")
            raise TrackingError(f"Erreur de suivi: {e}")
    
    def _filter_detections(self, detections: List[DetectedObject]) -> List[DetectedObject]:
        """Filtrage des détections pour le suivi."""
        
        # Priorité aux personnes
        filtered = []
        for det in detections:
            if det.class_name == "person" and det.confidence > 0.3:
                filtered.append(det)
            elif det.class_name in ["handbag", "backpack"] and det.confidence > 0.5:
                filtered.append(det)
        
        return filtered
    
    def _update_track_history(
        self, 
        tracked_objects: Dict[int, TrackingState],
        frame_info: Optional[Dict[str, Any]]
    ) -> None:
        """Mise à jour de l'historique des tracks."""
        
        timestamp = frame_info.get("timestamp", time.time()) if frame_info else time.time()
        
        for track_id, track_state in tracked_objects.items():
            if track_id not in self.track_history:
                self.track_history[track_id] = {
                    "start_time": timestamp,
                    "positions": [],
                    "behaviors": [],
                    "class_name": track_state.class_name
                }
            
            # Ajout de la position actuelle
            self.track_history[track_id]["positions"].append({
                "timestamp": timestamp,
                "bbox": track_state.bbox.to_dict(),
                "confidence": track_state.confidence
            })
            
            # Nettoyage de l'historique (garder 5 minutes max)
            max_history_time = 300  # 5 minutes
            cutoff_time = timestamp - max_history_time
            
            self.track_history[track_id]["positions"] = [
                pos for pos in self.track_history[track_id]["positions"]
                if pos["timestamp"] > cutoff_time
            ]
    
    def _update_stats(self, tracked_objects: Dict[int, TrackingState]) -> None:
        """Mise à jour des statistiques de suivi."""
        
        self.stats["active_tracks"] = len(tracked_objects)
        
        # Calcul de la durée moyenne des tracks
        if self.track_history:
            current_time = time.time()
            durations = []
            
            for track_data in self.track_history.values():
                if track_data["positions"]:
                    start_time = track_data["start_time"]
                    last_time = track_data["positions"][-1]["timestamp"]
                    duration = last_time - start_time
                    durations.append(duration)
            
            if durations:
                self.stats["avg_track_duration"] = np.mean(durations)
    
    def get_track_behavior_analysis(self, track_id: int) -> Optional[Dict[str, Any]]:
        """Analyse comportementale d'un track spécifique."""
        
        if track_id not in self.track_history:
            return None
        
        track_data = self.track_history[track_id]
        positions = track_data["positions"]
        
        if len(positions) < 5:
            return None
        
        # Calcul de métriques comportementales
        analysis = {
            "track_id": track_id,
            "class_name": track_data["class_name"],
            "duration": positions[-1]["timestamp"] - positions[0]["timestamp"],
            "total_distance": self._calculate_total_distance(positions),
            "avg_speed": 0.0,
            "direction_changes": self._count_direction_changes(positions),
            "time_in_zones": {},
            "suspicious_indicators": []
        }
        
        # Vitesse moyenne
        if analysis["duration"] > 0:
            analysis["avg_speed"] = analysis["total_distance"] / analysis["duration"]
        
        # Détection d'indicateurs suspects
        if analysis["direction_changes"] > 10:
            analysis["suspicious_indicators"].append("Changements de direction fréquents")
        
        if analysis["avg_speed"] < 0.5:  # Très lent
            analysis["suspicious_indicators"].append("Mouvement très lent")
        
        return analysis
    
    def _calculate_total_distance(self, positions: List[Dict]) -> float:
        """Calcul de la distance totale parcourue."""
        
        if len(positions) < 2:
            return 0.0
        
        total_distance = 0.0
        
        for i in range(1, len(positions)):
            prev_bbox = positions[i-1]["bbox"]
            curr_bbox = positions[i]["bbox"]
            
            # Centres des boîtes
            prev_center = (
                prev_bbox["x"] + prev_bbox["width"] // 2,
                prev_bbox["y"] + prev_bbox["height"] // 2
            )
            curr_center = (
                curr_bbox["x"] + curr_bbox["width"] // 2,
                curr_bbox["y"] + curr_bbox["height"] // 2
            )
            
            # Distance euclidienne
            distance = np.sqrt(
                (curr_center[0] - prev_center[0])**2 + 
                (curr_center[1] - prev_center[1])**2
            )
            
            total_distance += distance
        
        return total_distance
    
    def _count_direction_changes(self, positions: List[Dict]) -> int:
        """Comptage des changements de direction significatifs."""
        
        if len(positions) < 3:
            return 0
        
        direction_changes = 0
        prev_direction = None
        
        for i in range(2, len(positions)):
            # Calcul de la direction entre 3 points consécutifs
            p1 = positions[i-2]["bbox"]
            p2 = positions[i-1]["bbox"]
            p3 = positions[i]["bbox"]
            
            # Vecteurs
            v1 = (p2["x"] - p1["x"], p2["y"] - p1["y"])
            v2 = (p3["x"] - p2["x"], p3["y"] - p2["y"])
            
            # Angle entre vecteurs
            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.arccos(cos_angle)
                
                # Changement significatif si angle > 45 degrés
                if angle > np.pi / 4:
                    direction_changes += 1
        
        return direction_changes
    
    def get_stats(self) -> Dict[str, Any]:
        """Statistiques du tracker."""
        stats = self.stats.copy()
        stats.update({
            "tracker_type": self.tracker_type.value,
            "total_track_history": len(self.track_history)
        })
        return stats
    
    def reset(self) -> None:
        """Réinitialisation du tracker."""
        self.tracker = self._initialize_tracker()
        self.track_history.clear()
        
        self.stats = {
            "total_tracks": 0,
            "active_tracks": 0,
            "avg_track_duration": 0.0,
            "tracking_accuracy": 0.0
        }
        
        logger.info("Tracker réinitialisé")
    
    def cleanup_old_tracks(self, max_age_seconds: int = 300) -> None:
        """Nettoyage des anciens tracks."""
        
        current_time = time.time()
        cutoff_time = current_time - max_age_seconds
        
        tracks_to_remove = []
        
        for track_id, track_data in self.track_history.items():
            if track_data["positions"]:
                last_update = track_data["positions"][-1]["timestamp"]
                if last_update < cutoff_time:
                    tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.track_history[track_id]
        
        if tracks_to_remove:
            logger.info(f"Nettoyage: {len(tracks_to_remove)} anciens tracks supprimés")