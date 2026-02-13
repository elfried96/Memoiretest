"""
Service d'analyse de trajectoires et calcul du score de suspicion.
Identifie les comportements suspects basés sur les patterns de mouvement.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
from loguru import logger

from src.core.config import settings


@dataclass
class TrajectoryPoint:
    """Point d'une trajectoire."""
    x: float
    y: float
    timestamp: datetime
    frame_number: int


@dataclass
class Trajectory:
    """Trajectoire complète d'une personne."""
    track_id: int
    class_name: str
    points: list[TrajectoryPoint] = field(default_factory=list)
    
    # Métriques
    total_distance: float = 0.0
    avg_speed: float = 0.0
    max_speed: float = 0.0
    direction_changes: int = 0
    
    # Zones visitées
    zones_visited: list[str] = field(default_factory=list)
    
    # Suspicion
    suspicion_score: float = 0.0
    suspicion_reasons: list[str] = field(default_factory=list)
    
    @property
    def duration_seconds(self) -> float:
        if len(self.points) < 2:
            return 0.0
        return (self.points[-1].timestamp - self.points[0].timestamp).total_seconds()
    
    @property
    def is_active(self) -> bool:
        if not self.points:
            return False
        return (datetime.utcnow() - self.points[-1].timestamp).total_seconds() < 5.0


@dataclass
class SuspicionFactors:
    """Facteurs de suspicion avec poids."""
    loitering: float = 0.0           # Stationnement prolongé
    erratic_movement: float = 0.0    # Mouvements nerveux
    zone_avoidance: float = 0.0      # Évitement caisses
    unusual_path: float = 0.0        # Trajet non linéaire
    repeated_visits: float = 0.0     # Retours fréquents au même endroit
    speed_anomaly: float = 0.0       # Vitesse anormale
    
    def total(self) -> float:
        """Score total pondéré."""
        weights = {
            "loitering": 0.25,
            "erratic_movement": 0.20,
            "zone_avoidance": 0.20,
            "unusual_path": 0.15,
            "repeated_visits": 0.10,
            "speed_anomaly": 0.10
        }
        
        score = sum(
            getattr(self, factor) * weight 
            for factor, weight in weights.items()
        )
        return min(1.0, max(0.0, score))


class TrajectoryAnalyzer:
    """
    Analyseur de trajectoires pour détection de comportements suspects.
    
    Usage:
        analyzer = TrajectoryAnalyzer()
        analyzer.set_zones(zones_config)
        trajectory = analyzer.update(track_id, "person", bbox, frame_number)
        suspicious = analyzer.get_suspicious_trajectories()
    """
    
    def __init__(self):
        # Trajectoires actives
        self.trajectories: dict[int, Trajectory] = {}
        
        # Historique (trajectoires terminées)
        self.history: list[Trajectory] = []
        self.max_history = 500
        
        # Zones par défaut (à configurer par magasin)
        self.zones: dict[str, dict] = {
            "entrance": {"x_range": (0.0, 0.15), "y_range": (0.0, 1.0)},
            "exit": {"x_range": (0.85, 1.0), "y_range": (0.0, 1.0)},
            "checkout": {"x_range": (0.7, 0.85), "y_range": (0.7, 1.0)},
            "high_value": {"x_range": (0.3, 0.5), "y_range": (0.2, 0.5)},
        }
        
        # Seuils
        self.loitering_threshold_seconds = 45
        self.erratic_direction_threshold = 6
        self.min_points_for_analysis = 15
    
    def update(
        self,
        track_id: int,
        class_name: str,
        bbox: list[float],
        frame_number: int
    ) -> Trajectory:
        """
        Met à jour une trajectoire avec une nouvelle détection.
        
        Args:
            track_id: ID du tracker
            class_name: Classe de l'objet ("person", etc.)
            bbox: Bounding box [x1, y1, x2, y2] normalisée
            frame_number: Numéro du frame
            
        Returns:
            Trajectoire mise à jour avec score de suspicion
        """
        # Centre de la bbox
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        
        point = TrajectoryPoint(
            x=center_x,
            y=center_y,
            timestamp=datetime.utcnow(),
            frame_number=frame_number
        )
        
        # Créer ou récupérer la trajectoire
        if track_id not in self.trajectories:
            self.trajectories[track_id] = Trajectory(
                track_id=track_id,
                class_name=class_name
            )
        
        trajectory = self.trajectories[track_id]
        trajectory.points.append(point)
        
        # Mettre à jour les métriques
        self._update_metrics(trajectory)
        
        # Calculer le score de suspicion
        if len(trajectory.points) >= self.min_points_for_analysis:
            self._calculate_suspicion(trajectory)
        
        return trajectory
    
    def _update_metrics(self, trajectory: Trajectory) -> None:
        """Met à jour les métriques de distance, vitesse, etc."""
        points = trajectory.points
        
        if len(points) < 2:
            return
        
        # Calcul des distances et vitesses
        distances = []
        speeds = []
        
        for i in range(1, len(points)):
            dx = points[i].x - points[i-1].x
            dy = points[i].y - points[i-1].y
            dist = np.sqrt(dx**2 + dy**2)
            distances.append(dist)
            
            dt = (points[i].timestamp - points[i-1].timestamp).total_seconds()
            if dt > 0:
                speeds.append(dist / dt)
        
        trajectory.total_distance = sum(distances)
        trajectory.avg_speed = np.mean(speeds) if speeds else 0.0
        trajectory.max_speed = max(speeds) if speeds else 0.0
        
        # Changements de direction
        if len(points) >= 3:
            direction_changes = 0
            for i in range(2, len(points)):
                v1 = (points[i-1].x - points[i-2].x, points[i-1].y - points[i-2].y)
                v2 = (points[i].x - points[i-1].x, points[i].y - points[i-1].y)
                
                mag1 = np.sqrt(v1[0]**2 + v1[1]**2)
                mag2 = np.sqrt(v2[0]**2 + v2[1]**2)
                
                if mag1 > 0.01 and mag2 > 0.01:
                    dot = v1[0]*v2[0] + v1[1]*v2[1]
                    cos_angle = np.clip(dot / (mag1 * mag2), -1, 1)
                    angle = np.arccos(cos_angle)
                    
                    # Changement > 45 degrés
                    if angle > np.pi / 4:
                        direction_changes += 1
            
            trajectory.direction_changes = direction_changes
        
        # Zones visitées
        current_zones = self._get_zones_at_point(points[-1].x, points[-1].y)
        for zone in current_zones:
            if zone not in trajectory.zones_visited:
                trajectory.zones_visited.append(zone)
    
    def _get_zones_at_point(self, x: float, y: float) -> list[str]:
        """Retourne les zones contenant le point."""
        zones = []
        for zone_name, zone_def in self.zones.items():
            x_range = zone_def["x_range"]
            y_range = zone_def["y_range"]
            
            if x_range[0] <= x <= x_range[1] and y_range[0] <= y <= y_range[1]:
                zones.append(zone_name)
        
        return zones
    
    def _calculate_suspicion(self, trajectory: Trajectory) -> None:
        """Calcule le score de suspicion basé sur les patterns."""
        factors = SuspicionFactors()
        reasons = []
        points = trajectory.points
        
        # 1. LOITERING - Stationnement prolongé
        if trajectory.duration_seconds > self.loitering_threshold_seconds:
            recent = points[-30:] if len(points) >= 30 else points
            if len(recent) >= 2:
                movement = sum(
                    np.sqrt((recent[i].x - recent[i-1].x)**2 + 
                           (recent[i].y - recent[i-1].y)**2)
                    for i in range(1, len(recent))
                )
                
                if movement < 0.05:  # Moins de 5% de mouvement
                    factors.loitering = min(1.0, trajectory.duration_seconds / 90)
                    reasons.append(f"Stationnement {trajectory.duration_seconds:.0f}s")
        
        # 2. ERRATIC - Mouvements nerveux
        if trajectory.direction_changes > self.erratic_direction_threshold:
            factors.erratic_movement = min(1.0, trajectory.direction_changes / 12)
            reasons.append(f"Mouvements erratiques ({trajectory.direction_changes}x)")
        
        # 3. ZONE AVOIDANCE - Évitement des caisses
        if trajectory.duration_seconds > 60:
            if "checkout" not in trajectory.zones_visited:
                if "exit" in trajectory.zones_visited or "entrance" in trajectory.zones_visited:
                    factors.zone_avoidance = 0.7
                    reasons.append("Évitement zone caisse")
        
        # 4. UNUSUAL PATH - Trajet non linéaire
        if trajectory.total_distance > 0 and len(points) >= 2:
            start, end = points[0], points[-1]
            direct_dist = np.sqrt((end.x - start.x)**2 + (end.y - start.y)**2)
            
            if direct_dist > 0.1 and trajectory.total_distance > 3 * direct_dist:
                factors.unusual_path = min(1.0, trajectory.total_distance / (5 * direct_dist))
                reasons.append("Trajet non linéaire")
        
        # 5. REPEATED VISITS - Retours fréquents zone sensible
        high_value_entries = 0
        in_high_value = False
        for p in points:
            zones = self._get_zones_at_point(p.x, p.y)
            if "high_value" in zones and not in_high_value:
                high_value_entries += 1
                in_high_value = True
            elif "high_value" not in zones:
                in_high_value = False
        
        if high_value_entries >= 3:
            factors.repeated_visits = min(1.0, high_value_entries / 5)
            reasons.append(f"Retours zone sensible ({high_value_entries}x)")
        
        # 6. SPEED ANOMALY - Vitesse anormale
        if trajectory.avg_speed < 0.005 and trajectory.duration_seconds > 30:
            factors.speed_anomaly = 0.5
            reasons.append("Vitesse très lente")
        elif trajectory.max_speed > 0.15:
            factors.speed_anomaly = 0.4
            reasons.append("Pic de vitesse")
        
        # Score final
        trajectory.suspicion_score = factors.total()
        trajectory.suspicion_reasons = reasons
    
    def get_trajectory(self, track_id: int) -> Optional[Trajectory]:
        """Récupère une trajectoire par ID."""
        return self.trajectories.get(track_id)
    
    def get_suspicious_trajectories(self, threshold: float = None) -> list[Trajectory]:
        """Retourne les trajectoires suspectes actives."""
        if threshold is None:
            threshold = settings.vlm.suspicion_threshold
        
        return [
            t for t in self.trajectories.values()
            if t.suspicion_score >= threshold and t.is_active
        ]
    
    def cleanup_inactive(self, max_inactive_seconds: float = 10.0) -> list[Trajectory]:
        """Archive les trajectoires inactives."""
        now = datetime.utcnow()
        archived = []
        
        inactive_ids = [
            tid for tid, traj in self.trajectories.items()
            if traj.points and 
               (now - traj.points[-1].timestamp).total_seconds() > max_inactive_seconds
        ]
        
        for tid in inactive_ids:
            traj = self.trajectories.pop(tid)
            archived.append(traj)
            self.history.append(traj)
        
        # Limite historique
        while len(self.history) > self.max_history:
            self.history.pop(0)
        
        if archived:
            logger.debug(f"Archived {len(archived)} inactive trajectories")
        
        return archived
    
    def set_zones(self, zones: dict[str, dict]) -> None:
        """Configure les zones de surveillance."""
        self.zones = zones
        logger.info(f"Zones updated: {list(zones.keys())}")
    
    def reset(self) -> None:
        """Réinitialise l'analyseur."""
        self.trajectories.clear()
        logger.info("TrajectoryAnalyzer reset")
    
    @property
    def stats(self) -> dict:
        """Statistiques de l'analyseur."""
        active = [t for t in self.trajectories.values() if t.is_active]
        suspicious = self.get_suspicious_trajectories()
        
        return {
            "active_trajectories": len(active),
            "suspicious_count": len(suspicious),
            "history_size": len(self.history),
            "zones": list(self.zones.keys()),
            "avg_suspicion": round(np.mean([t.suspicion_score for t in active]), 3) if active else 0
        }


# === SINGLETON ===

_analyzer: Optional[TrajectoryAnalyzer] = None


def get_trajectory_analyzer() -> TrajectoryAnalyzer:
    """Retourne le singleton de l'analyseur."""
    global _analyzer
    if _analyzer is None:
        _analyzer = TrajectoryAnalyzer()
    return _analyzer