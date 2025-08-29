"""Trajectory analysis for movement pattern detection."""

import numpy as np
import cv2
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from collections import deque
import time
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)

@dataclass
class TrajectoryPoint:
    """Single point in trajectory."""
    x: float
    y: float
    timestamp: float
    velocity: Optional[float] = None
    direction: Optional[float] = None

@dataclass
class TrajectoryAnalysis:
    """Complete trajectory analysis result."""
    person_id: str
    trajectory_points: List[TrajectoryPoint]
    total_distance: float
    average_velocity: float
    direction_changes: int
    stopping_points: List[TrajectoryPoint]
    anomaly_score: float
    pattern_classification: str
    processing_time: float

class TrajectoryAnalyzer:
    """Advanced trajectory analysis for behavioral assessment."""
    
    def __init__(self, history_size: int = 100, velocity_threshold: float = 0.5):
        self.history_size = history_size
        self.velocity_threshold = velocity_threshold
        
        # Trajectory storage
        self.trajectories = {}  # person_id -> deque of TrajectoryPoint
        
        # Pattern templates
        self.pattern_templates = self._initialize_pattern_templates()
        
    def _initialize_pattern_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize common movement pattern templates."""
        return {
            "browsing": {
                "description": "Casual browsing with frequent direction changes",
                "velocity_range": (0.1, 1.0),
                "direction_change_freq": "high",
                "stopping_freq": "medium"
            },
            "purposeful_walking": {
                "description": "Direct movement towards specific destination",
                "velocity_range": (0.8, 2.0),
                "direction_change_freq": "low",
                "stopping_freq": "low"
            },
            "suspicious_loitering": {
                "description": "Lingering in area with minimal movement",
                "velocity_range": (0.0, 0.3),
                "direction_change_freq": "low",
                "stopping_freq": "high"
            },
            "evasive_movement": {
                "description": "Irregular movement avoiding certain areas",
                "velocity_range": (0.5, 1.5),
                "direction_change_freq": "very_high",
                "stopping_freq": "low"
            },
            "return_pattern": {
                "description": "Returning to same area multiple times",
                "velocity_range": (0.3, 1.2),
                "direction_change_freq": "medium",
                "stopping_freq": "medium"
            }
        }
    
    def update_trajectory(self, person_id: str, x: float, y: float, timestamp: float) -> None:
        """Update trajectory for a person."""
        
        if person_id not in self.trajectories:
            self.trajectories[person_id] = deque(maxlen=self.history_size)
        
        trajectory = self.trajectories[person_id]
        
        # Calculate velocity and direction if previous point exists
        velocity = None
        direction = None
        
        if len(trajectory) > 0:
            prev_point = trajectory[-1]
            dx = x - prev_point.x
            dy = y - prev_point.y
            dt = timestamp - prev_point.timestamp
            
            if dt > 0:
                distance = np.sqrt(dx**2 + dy**2)
                velocity = distance / dt
                direction = np.arctan2(dy, dx)  # Radians
        
        # Add new point
        trajectory.append(TrajectoryPoint(
            x=x, y=y, timestamp=timestamp,
            velocity=velocity, direction=direction
        ))
    
    def analyze_trajectory(self, person_id: str, time_window: int = 60) -> Optional[TrajectoryAnalysis]:
        """Analyze trajectory for a specific person."""
        start_time = time.time()
        
        if person_id not in self.trajectories:
            return None
        
        trajectory = list(self.trajectories[person_id])
        
        if len(trajectory) < 2:
            return None
        
        # Filter by time window
        current_time = trajectory[-1].timestamp
        filtered_trajectory = [
            point for point in trajectory 
            if current_time - point.timestamp <= time_window
        ]
        
        if len(filtered_trajectory) < 2:
            return None
        
        # Calculate metrics
        total_distance = self._calculate_total_distance(filtered_trajectory)
        average_velocity = self._calculate_average_velocity(filtered_trajectory)
        direction_changes = self._count_direction_changes(filtered_trajectory)
        stopping_points = self._find_stopping_points(filtered_trajectory)
        anomaly_score = self._calculate_anomaly_score(filtered_trajectory)
        pattern_classification = self._classify_movement_pattern(filtered_trajectory)
        
        processing_time = time.time() - start_time
        
        return TrajectoryAnalysis(
            person_id=person_id,
            trajectory_points=filtered_trajectory,
            total_distance=total_distance,
            average_velocity=average_velocity,
            direction_changes=direction_changes,
            stopping_points=stopping_points,
            anomaly_score=anomaly_score,
            pattern_classification=pattern_classification,
            processing_time=processing_time
        )
    
    def _calculate_total_distance(self, trajectory: List[TrajectoryPoint]) -> float:
        """Calculate total distance traveled."""
        total_distance = 0.0
        
        for i in range(1, len(trajectory)):
            dx = trajectory[i].x - trajectory[i-1].x
            dy = trajectory[i].y - trajectory[i-1].y
            total_distance += np.sqrt(dx**2 + dy**2)
        
        return total_distance
    
    def _calculate_average_velocity(self, trajectory: List[TrajectoryPoint]) -> float:
        """Calculate average velocity."""
        velocities = [point.velocity for point in trajectory if point.velocity is not None]
        if not velocities:
            return 0.0
        return float(np.mean(velocities))
    
    def _count_direction_changes(self, trajectory: List[TrajectoryPoint], 
                               angle_threshold: float = np.pi/4) -> int:
        """Count significant direction changes."""
        direction_changes = 0
        
        directions = [point.direction for point in trajectory if point.direction is not None]
        
        for i in range(1, len(directions)):
            angle_diff = abs(directions[i] - directions[i-1])
            # Handle angle wrapping
            angle_diff = min(angle_diff, 2*np.pi - angle_diff)
            
            if angle_diff > angle_threshold:
                direction_changes += 1
        
        return direction_changes
    
    def _find_stopping_points(self, trajectory: List[TrajectoryPoint]) -> List[TrajectoryPoint]:
        """Find points where person stopped or moved very slowly."""
        stopping_points = []
        
        for point in trajectory:
            if point.velocity is not None and point.velocity < self.velocity_threshold:
                stopping_points.append(point)
        
        # Cluster nearby stopping points
        if len(stopping_points) > 1:
            stopping_points = self._cluster_stopping_points(stopping_points)
        
        return stopping_points
    
    def _cluster_stopping_points(self, stopping_points: List[TrajectoryPoint]) -> List[TrajectoryPoint]:
        """Cluster nearby stopping points."""
        if len(stopping_points) < 2:
            return stopping_points
        
        # Extract coordinates
        coords = np.array([[point.x, point.y] for point in stopping_points])
        
        # DBSCAN clustering
        clustering = DBSCAN(eps=20, min_samples=2).fit(coords)
        
        # Get representative point for each cluster
        clustered_points = []
        unique_labels = set(clustering.labels_)
        
        for label in unique_labels:
            if label == -1:  # Noise points
                continue
            
            cluster_indices = np.where(clustering.labels_ == label)[0]
            cluster_points = [stopping_points[i] for i in cluster_indices]
            
            # Use centroid of cluster
            centroid_x = np.mean([p.x for p in cluster_points])
            centroid_y = np.mean([p.y for p in cluster_points])
            avg_timestamp = np.mean([p.timestamp for p in cluster_points])
            
            clustered_points.append(TrajectoryPoint(
                x=centroid_x, y=centroid_y, timestamp=avg_timestamp, velocity=0.0
            ))
        
        return clustered_points
    
    def _calculate_anomaly_score(self, trajectory: List[TrajectoryPoint]) -> float:
        """Calculate anomaly score based on movement patterns."""
        
        if len(trajectory) < 3:
            return 0.0
        
        anomaly_score = 0.0
        
        # Velocity anomalies
        velocities = [p.velocity for p in trajectory if p.velocity is not None]
        if velocities:
            velocity_std = np.std(velocities)
            avg_velocity = np.mean(velocities)
            
            # High velocity variance indicates erratic movement
            if velocity_std > avg_velocity * 0.5:
                anomaly_score += 0.3
        
        # Direction change anomalies
        direction_changes = self._count_direction_changes(trajectory)
        trajectory_length = len(trajectory)
        change_ratio = direction_changes / max(trajectory_length, 1)
        
        # Too many direction changes
        if change_ratio > 0.3:
            anomaly_score += 0.4
        
        # Stopping pattern anomalies
        stopping_points = self._find_stopping_points(trajectory)
        stop_ratio = len(stopping_points) / trajectory_length
        
        # Too much stopping
        if stop_ratio > 0.5:
            anomaly_score += 0.2
        
        # Return pattern detection
        if self._detect_return_pattern(trajectory):
            anomaly_score += 0.1
        
        return min(1.0, anomaly_score)
    
    def _detect_return_pattern(self, trajectory: List[TrajectoryPoint], 
                             proximity_threshold: float = 30) -> bool:
        """Detect if person returns to same area multiple times."""
        
        if len(trajectory) < 10:
            return False
        
        # Divide trajectory into segments
        segment_size = len(trajectory) // 3
        segments = [
            trajectory[:segment_size],
            trajectory[segment_size:2*segment_size],
            trajectory[2*segment_size:]
        ]
        
        # Check if end segments are close to beginning
        first_segment_center = self._get_segment_center(segments[0])
        last_segment_center = self._get_segment_center(segments[2])
        
        distance = np.sqrt(
            (first_segment_center[0] - last_segment_center[0])**2 +
            (first_segment_center[1] - last_segment_center[1])**2
        )
        
        return distance < proximity_threshold
    
    def _get_segment_center(self, segment: List[TrajectoryPoint]) -> Tuple[float, float]:
        """Get center point of trajectory segment."""
        x_coords = [point.x for point in segment]
        y_coords = [point.y for point in segment]
        return (np.mean(x_coords), np.mean(y_coords))
    
    def _classify_movement_pattern(self, trajectory: List[TrajectoryPoint]) -> str:
        """Classify movement pattern based on characteristics."""
        
        if len(trajectory) < 3:
            return "insufficient_data"
        
        avg_velocity = self._calculate_average_velocity(trajectory)
        direction_changes = self._count_direction_changes(trajectory)
        stopping_points = self._find_stopping_points(trajectory)
        
        trajectory_length = len(trajectory)
        change_ratio = direction_changes / max(trajectory_length, 1)
        stop_ratio = len(stopping_points) / trajectory_length
        
        # Classification logic
        if avg_velocity < 0.3 and stop_ratio > 0.4:
            return "suspicious_loitering"
        elif change_ratio > 0.4:
            return "evasive_movement"
        elif avg_velocity > 1.0 and change_ratio < 0.2:
            return "purposeful_walking"
        elif self._detect_return_pattern(trajectory):
            return "return_pattern"
        elif change_ratio > 0.2 and stop_ratio > 0.2:
            return "browsing"
        else:
            return "normal_movement"
    
    def analyze_motion(self, trajectory_data: List[Dict], time_window: int = 60) -> Dict[str, Any]:
        """Public interface for motion analysis (for tool calling)."""
        
        if not trajectory_data:
            return {
                "movement_speed": "unknown",
                "direction_changes": 0,
                "stopping_frequency": "unknown",
                "pattern": "insufficient_data",
                "anomaly_score": 0.0
            }
        
        # Convert trajectory data to internal format
        person_id = trajectory_data[0].get('person_id', 'unknown')
        
        # Update trajectory with provided data
        for data_point in trajectory_data:
            self.update_trajectory(
                person_id=person_id,
                x=data_point.get('x', 0),
                y=data_point.get('y', 0),
                timestamp=data_point.get('timestamp', time.time())
            )
        
        # Analyze
        analysis = self.analyze_trajectory(person_id, time_window)
        
        if analysis is None:
            return {
                "movement_speed": "unknown",
                "direction_changes": 0,
                "stopping_frequency": "unknown", 
                "pattern": "insufficient_data",
                "anomaly_score": 0.0
            }
        
        # Convert to expected format
        speed_category = "slow" if analysis.average_velocity < 0.5 else \
                        "normal" if analysis.average_velocity < 1.5 else "fast"
        
        stop_freq = "low" if len(analysis.stopping_points) / len(analysis.trajectory_points) < 0.2 else \
                   "medium" if len(analysis.stopping_points) / len(analysis.trajectory_points) < 0.4 else "high"
        
        return {
            "movement_speed": speed_category,
            "direction_changes": analysis.direction_changes,
            "stopping_frequency": stop_freq,
            "pattern": analysis.pattern_classification,
            "anomaly_score": analysis.anomaly_score,
            "total_distance": analysis.total_distance,
            "average_velocity": analysis.average_velocity
        }
    
    def visualize_trajectory(self, person_id: str, frame_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """Visualize trajectory on a frame."""
        
        if person_id not in self.trajectories:
            return None
        
        trajectory = list(self.trajectories[person_id])
        if len(trajectory) < 2:
            return None
        
        # Create visualization frame
        vis_frame = np.zeros((frame_shape[0], frame_shape[1], 3), dtype=np.uint8)
        
        # Draw trajectory path
        points = [(int(point.x), int(point.y)) for point in trajectory]
        
        for i in range(1, len(points)):
            # Color based on velocity
            if trajectory[i].velocity is not None:
                velocity = trajectory[i].velocity
                color_intensity = min(255, int(velocity * 100))
                color = (0, color_intensity, 255 - color_intensity)  # Blue to red gradient
            else:
                color = (255, 255, 255)  # White for unknown velocity
            
            cv2.line(vis_frame, points[i-1], points[i], color, 2)
        
        # Mark stopping points
        analysis = self.analyze_trajectory(person_id)
        if analysis and analysis.stopping_points:
            for stop_point in analysis.stopping_points:
                cv2.circle(vis_frame, (int(stop_point.x), int(stop_point.y)), 5, (0, 0, 255), -1)
        
        # Mark start and end points
        if points:
            cv2.circle(vis_frame, points[0], 8, (0, 255, 0), -1)  # Green start
            cv2.circle(vis_frame, points[-1], 8, (255, 0, 0), -1)  # Blue end
        
        return vis_frame
    
    def get_trajectory_summary(self, person_id: str) -> Optional[Dict[str, Any]]:
        """Get summary statistics for a person's trajectory."""
        
        analysis = self.analyze_trajectory(person_id)
        if analysis is None:
            return None
        
        return {
            "person_id": person_id,
            "total_points": len(analysis.trajectory_points),
            "total_distance": analysis.total_distance,
            "average_velocity": analysis.average_velocity,
            "direction_changes": analysis.direction_changes,
            "stopping_points_count": len(analysis.stopping_points),
            "anomaly_score": analysis.anomaly_score,
            "pattern_classification": analysis.pattern_classification,
            "time_span": analysis.trajectory_points[-1].timestamp - analysis.trajectory_points[0].timestamp
        }
    
    def clear_old_trajectories(self, max_age_seconds: int = 300):
        """Clear trajectories older than specified age."""
        current_time = time.time()
        
        to_remove = []
        for person_id, trajectory in self.trajectories.items():
            if trajectory and (current_time - trajectory[-1].timestamp) > max_age_seconds:
                to_remove.append(person_id)
        
        for person_id in to_remove:
            del self.trajectories[person_id]
            
        logger.info(f"Cleared {len(to_remove)} old trajectories")