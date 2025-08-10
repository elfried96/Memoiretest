"""Types de données centraux pour le système de surveillance."""

from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime
from dataclasses import dataclass, field
from pydantic import BaseModel, Field
import numpy as np


class DetectionStatus(Enum):
    """Statuts de détection."""
    PENDING = "pending"
    PROCESSING = "processing"
    DETECTED = "detected"
    VALIDATED = "validated"
    FALSE_POSITIVE = "false_positive"
    ALERT = "alert"


class SuspicionLevel(Enum):
    """Niveaux de suspicion."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ActionType(Enum):
    """Types d'actions détectées."""
    NORMAL_SHOPPING = "normal_shopping"
    SUSPICIOUS_MOVEMENT = "suspicious_movement"
    ITEM_CONCEALMENT = "item_concealment"
    POTENTIAL_THEFT = "potential_theft"
    CONFIRMED_THEFT = "confirmed_theft"


class ToolType(Enum):
    """Types d'outils disponibles."""
    OBJECT_DETECTOR = "object_detector"
    TRACKER = "tracker"
    BEHAVIOR_ANALYZER = "behavior_analyzer"
    CONTEXT_VALIDATOR = "context_validator"
    FALSE_POSITIVE_FILTER = "false_positive_filter"


@dataclass
class BoundingBox:
    """Boîte englobante pour détections."""
    x: int
    y: int
    width: int
    height: int
    confidence: float = 0.0
    
    @property
    def center(self) -> Tuple[int, int]:
        """Centre de la boîte."""
        return (self.x + self.width // 2, self.y + self.height // 2)
    
    @property
    def area(self) -> int:
        """Aire de la boîte."""
        return self.width * self.height
    
    def to_dict(self) -> Dict[str, Union[int, float]]:
        """Conversion en dictionnaire."""
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "confidence": self.confidence
        }


@dataclass
class DetectedObject:
    """Objet détecté dans une image."""
    class_id: int
    class_name: str
    bbox: BoundingBox
    confidence: float
    track_id: Optional[int] = None
    features: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Conversion en dictionnaire."""
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "bbox": self.bbox.to_dict(),
            "confidence": self.confidence,
            "track_id": self.track_id
        }


@dataclass
class Frame:
    """Frame de vidéo avec métadonnées."""
    image: np.ndarray
    timestamp: datetime
    frame_id: int
    stream_id: str
    width: int
    height: int
    detections: List[DetectedObject] = field(default_factory=list)
    
    @property
    def shape(self) -> Tuple[int, int, int]:
        """Dimensions de l'image (H, W, C)."""
        return self.image.shape


class AnalysisRequest(BaseModel):
    """Requête d'analyse VLM."""
    frame_data: str = Field(..., description="Image encodée en base64")
    context: Dict[str, Any] = Field(default_factory=dict)
    previous_detections: List[Dict[str, Any]] = Field(default_factory=list)
    tools_available: List[str] = Field(default_factory=list)
    max_tokens: int = Field(512, description="Nombre max de tokens")
    temperature: float = Field(0.1, description="Température pour génération")


class AnalysisResponse(BaseModel):
    """Réponse d'analyse VLM."""
    suspicion_level: SuspicionLevel
    action_type: ActionType
    confidence: float = Field(..., ge=0.0, le=1.0)
    description: str
    tools_used: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)


class SurveillanceEvent(BaseModel):
    """Événement de surveillance."""
    event_id: str
    stream_id: str
    timestamp: datetime
    location: Optional[Dict[str, float]] = None
    action_type: ActionType
    suspicion_level: SuspicionLevel
    confidence: float
    description: str
    frame_id: int
    detections: List[Dict[str, Any]] = Field(default_factory=list)
    validation_status: DetectionStatus
    false_positive_probability: float = 0.0
    
    class Config:
        use_enum_values = True


class StreamConfig(BaseModel):
    """Configuration d'un flux vidéo."""
    stream_id: str
    source_url: str
    location_name: str
    is_active: bool = True
    fps: int = 15
    resolution: Tuple[int, int] = (640, 480)
    roi: Optional[List[Tuple[int, int]]] = None  # Region of Interest
    sensitivity: float = 0.5
    
    class Config:
        validate_assignment = True


class SystemMetrics(BaseModel):
    """Métriques système."""
    timestamp: datetime
    active_streams: int
    processed_frames: int
    detected_events: int
    false_positives: int
    cpu_usage: float
    memory_usage: float
    gpu_usage: Optional[float] = None
    latency_ms: float
    throughput_fps: float
    
    @property
    def false_positive_rate(self) -> float:
        """Taux de faux positifs."""
        if self.detected_events == 0:
            return 0.0
        return self.false_positives / self.detected_events


class ToolResult(BaseModel):
    """Résultat d'utilisation d'un outil."""
    tool_type: ToolType
    success: bool
    data: Dict[str, Any] = Field(default_factory=dict)
    confidence: Optional[float] = None
    execution_time_ms: float
    error_message: Optional[str] = None
    
    class Config:
        use_enum_values = True


# Types pour la validation croisée
ValidationResult = Dict[str, Union[bool, float, str, List[Any]]]
ToolChain = List[Tuple[ToolType, Dict[str, Any]]]

# Types pour les templates de prompts
PromptTemplate = str
ContextData = Dict[str, Any]