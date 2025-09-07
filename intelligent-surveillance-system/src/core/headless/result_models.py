"""
Modèles de données pour les résultats de surveillance.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional


class AlertLevel(Enum):
    """Niveaux d'alerte du système de surveillance."""
    NORMAL = "normal"
    ATTENTION = "attention" 
    ALERTE = "alerte"
    CRITIQUE = "critique"


@dataclass
class SurveillanceResult:
    """Résultat d'analyse de surveillance pour une frame."""
    frame_id: int
    timestamp: float
    detections_count: int
    persons_detected: int
    alert_level: AlertLevel
    vlm_analysis: Optional[Dict[str, Any]] = None
    actions_taken: List[str] = field(default_factory=list)
    processing_time: float = 0.0
    cumulative_summary: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire pour sérialisation JSON."""
        return {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "detections_count": self.detections_count,
            "persons_detected": self.persons_detected,
            "alert_level": self.alert_level.value,
            "vlm_analysis": self.vlm_analysis,
            "actions_taken": self.actions_taken,
            "processing_time": self.processing_time,
            "cumulative_summary": self.cumulative_summary
        }


@dataclass
class SessionSummary:
    """Résumé complet d'une session de surveillance."""
    total_frames: int
    total_detections: int
    total_persons: int
    alerts_by_level: Dict[str, int]
    average_processing_time: float
    session_duration: float
    key_events: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire pour sérialisation JSON."""
        return {
            "total_frames": self.total_frames,
            "total_detections": self.total_detections,
            "total_persons": self.total_persons,
            "alerts_by_level": self.alerts_by_level,
            "average_processing_time": self.average_processing_time,
            "session_duration": self.session_duration,
            "key_events": self.key_events
        }