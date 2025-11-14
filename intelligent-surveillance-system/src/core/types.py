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
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
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
    SAM2_SEGMENTATOR = "sam2_segmentator"
    DINO_FEATURES = "dino_features"
    POSE_ESTIMATOR = "pose_estimator"
    TRAJECTORY_ANALYZER = "trajectory_analyzer"
    MULTIMODAL_FUSION = "multimodal_fusion"
    ADVERSARIAL_DETECTOR = "adversarial_detector"
    DOMAIN_ADAPTER = "domain_adapter"
    TEMPORAL_TRANSFORMER = "temporal_transformer"


@dataclass
class BoundingBox:
    """Boîte englobante pour détections."""
    x1: float  # Coordonnée x du coin supérieur gauche
    y1: float  # Coordonnée y du coin supérieur gauche  
    x2: float  # Coordonnée x du coin inférieur droit
    y2: float  # Coordonnée y du coin inférieur droit
    confidence: float = 0.0
    
    @property
    def x(self) -> int:
        """Coordonnée x (pour compatibilité)."""
        return int(self.x1)
    
    @property 
    def y(self) -> int:
        """Coordonnée y (pour compatibilité)."""
        return int(self.y1)
        
    @property
    def width(self) -> int:
        """Largeur de la boîte."""
        return int(self.x2 - self.x1)
    
    @property
    def height(self) -> int:
        """Hauteur de la boîte.""" 
        return int(self.y2 - self.y1)
    
    @property
    def center(self) -> Tuple[float, float]:
        """Centre de la boîte."""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
    @property
    def area(self) -> float:
        """Aire de la boîte."""
        return (self.x2 - self.x1) * (self.y2 - self.y1)
    
    def to_dict(self) -> Dict[str, Union[int, float]]:
        """Conversion en dictionnaire."""
        return {
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
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
    preferred_model: Optional[str] = Field(None, description="Modèle préféré pour l'analyse")


class AnalysisResponse(BaseModel):
    """Réponse d'analyse VLM."""
    suspicion_level: SuspicionLevel
    action_type: ActionType
    confidence: float = Field(..., ge=0.0, le=1.0)
    description: str
    reasoning: str = Field(default="", description="Processus de raisonnement du modèle")
    detections: List[Dict[str, Any]] = Field(default_factory=list, description="Liste des détections d'objets")
    tools_used: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)


# Alias pour compatibilité
Detection = DetectedObject


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