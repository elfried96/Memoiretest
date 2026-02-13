"""
Schémas Pydantic pour l'API REST et WebSocket.
Validation des données entrantes et sortantes.
"""

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


# === ENUMS ===

class AlertLevel(str, Enum):
    INFO = "info"
    ATTENTION = "attention"
    CRITICAL = "critical"


class AnalysisStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# === STORE SCHEMAS ===

class StoreBase(BaseModel):
    name: str = Field(max_length=100)
    address: Optional[str] = Field(default=None, max_length=300)
    timezone: str = Field(default="Africa/Porto-Novo", max_length=50)


class StoreCreate(StoreBase):
    pass


class StoreResponse(StoreBase):
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    is_active: bool
    created_at: datetime
    updated_at: datetime


# === ZONE SCHEMAS ===

class ZoneBase(BaseModel):
    name: str = Field(max_length=100)
    zone_type: Optional[str] = Field(default=None, max_length=50)
    polygon: dict  # {"points": [[x1,y1], [x2,y2], ...]}
    suspicion_multiplier: float = Field(default=1.0, ge=0.1, le=5.0)
    max_loiter_seconds: int = Field(default=60, ge=10, le=600)


class ZoneCreate(ZoneBase):
    store_id: UUID


class ZoneResponse(ZoneBase):
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    store_id: UUID
    is_active: bool
    created_at: datetime


# === STREAM SCHEMAS ===

class StreamBase(BaseModel):
    name: str = Field(max_length=100)
    location: Optional[str] = Field(default=None, max_length=200)
    source_url: str = Field(max_length=500)
    fps_target: int = Field(default=15, ge=1, le=60)
    resolution: Optional[str] = Field(default=None, max_length=20)


class StreamCreate(StreamBase):
    store_id: UUID


class StreamUpdate(BaseModel):
    name: Optional[str] = Field(default=None, max_length=100)
    location: Optional[str] = Field(default=None, max_length=200)
    source_url: Optional[str] = Field(default=None, max_length=500)
    is_active: Optional[bool] = None
    fps_target: Optional[int] = Field(default=None, ge=1, le=60)


class StreamResponse(StreamBase):
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    store_id: UUID
    is_active: bool
    created_at: datetime
    updated_at: datetime


# === ALERT SCHEMAS ===

class AlertBase(BaseModel):
    level: AlertLevel
    alert_type: str = Field(max_length=50)
    title: str = Field(max_length=200)
    description: Optional[str] = None
    suspicion_score: float = Field(ge=0, le=1)


class AlertCreate(AlertBase):
    stream_id: UUID
    track_id: Optional[int] = None
    frame_path: Optional[str] = None
    video_clip_path: Optional[str] = None
    bbox: Optional[list[float]] = None


class AlertUpdate(BaseModel):
    is_acknowledged: Optional[bool] = None
    acknowledged_by: Optional[str] = Field(default=None, max_length=100)
    is_false_positive: Optional[bool] = None
    feedback_notes: Optional[str] = None


class AlertResponse(AlertBase):
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    stream_id: UUID
    track_id: Optional[int]
    frame_path: Optional[str]
    video_clip_path: Optional[str]
    bbox: Optional[list[float]]
    vlm_analysis_id: Optional[UUID]
    is_acknowledged: bool
    acknowledged_by: Optional[str]
    acknowledged_at: Optional[datetime]
    is_false_positive: Optional[bool]
    feedback_notes: Optional[str]
    created_at: datetime


class AlertStats(BaseModel):
    """Statistiques des alertes."""
    total: int
    by_level: dict[str, int]
    by_type: dict[str, int]
    acknowledged: int
    false_positives: int
    period_start: datetime
    period_end: datetime


# === TRAJECTORY SCHEMAS ===

class TrajectoryPoint(BaseModel):
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    t: datetime


class TrajectoryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    stream_id: UUID
    track_id: int
    start_time: datetime
    end_time: Optional[datetime]
    points: list[dict]
    total_distance: float
    avg_speed: float
    zones_visited: list[str]
    max_suspicion_score: float
    suspicion_reasons: list[str]


# === VLM ANALYSIS SCHEMAS ===

class VLMAnalysisRequest(BaseModel):
    """Requête d'analyse VLM."""
    stream_id: UUID
    track_ids: list[int]
    frame_paths: list[str]
    initial_suspicion_score: float = Field(ge=0, le=1)
    priority: int = Field(default=5, ge=1, le=10)


class VLMAnalysisResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    stream_id: UUID
    status: AnalysisStatus
    track_ids: list[int]
    frame_paths: list[str]
    initial_suspicion_score: float
    tools_used: list[str]
    tools_results: Optional[dict]
    vlm_response: Optional[str]
    final_suspicion_score: Optional[float]
    is_threat_confirmed: Optional[bool]
    reasoning: Optional[str]
    inference_time_ms: Optional[int]
    tokens_used: Optional[int]
    created_at: datetime
    completed_at: Optional[datetime]


# === DETECTION SCHEMAS (temps réel) ===

class Detection(BaseModel):
    """Détection unique dans un frame."""
    track_id: int
    class_id: int
    class_name: str
    confidence: float = Field(ge=0, le=1)
    bbox: list[float] = Field(min_length=4, max_length=4)  # [x1, y1, x2, y2]
    suspicion_score: float = Field(default=0.0, ge=0, le=1)


class FrameDetections(BaseModel):
    """Toutes les détections d'un frame."""
    stream_id: UUID
    frame_number: int
    timestamp: datetime
    detections: list[Detection]
    inference_time_ms: float


# === WEBSOCKET SCHEMAS ===

class WSMessage(BaseModel):
    """Message WebSocket générique."""
    type: str
    data: dict
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class WSSubscribe(BaseModel):
    """Abonnement à un stream."""
    action: str = "subscribe"
    stream_id: UUID


class WSDetectionEvent(BaseModel):
    """Événement détection temps réel."""
    type: str = "detection"
    stream_id: UUID
    frame_number: int
    detections: list[Detection]
    timestamp: datetime


class WSAlertEvent(BaseModel):
    """Événement alerte temps réel."""
    type: str = "alert"
    alert: AlertResponse


# === RESPONSE WRAPPERS ===

class PaginatedResponse(BaseModel):
    """Réponse paginée."""
    items: list
    total: int
    page: int
    page_size: int
    pages: int


class APIResponse(BaseModel):
    """Réponse API standard."""
    success: bool
    message: Optional[str] = None
    data: Optional[dict] = None


class HealthCheck(BaseModel):
    """Réponse health check."""
    status: str
    version: str
    environment: str
    database: bool
    redis: bool
    gpu_available: bool
    models_loaded: dict[str, bool]
