"""
Modèles SQLAlchemy pour PostgreSQL.
Définit les tables du système IntelliSurveillance.
"""

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID as PG_UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Classe de base pour tous les modèles."""
    pass


class AlertLevel(str, Enum):
    """Niveaux d'alerte."""
    INFO = "info"
    ATTENTION = "attention"
    CRITICAL = "critical"


class AnalysisStatus(str, Enum):
    """Statut d'une analyse VLM."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# === TABLE: STORES (Magasins) ===

class Store(Base):
    """Magasin surveillé."""
    
    __tablename__ = "stores"
    
    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    address: Mapped[Optional[str]] = mapped_column(String(300))
    timezone: Mapped[str] = mapped_column(String(50), default="Africa/Porto-Novo")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())
    
    # Relations
    streams: Mapped[list["Stream"]] = relationship(back_populates="store")
    zones: Mapped[list["Zone"]] = relationship(back_populates="store")


# === TABLE: ZONES (Zones de surveillance) ===

class Zone(Base):
    """Zone de surveillance dans un magasin."""
    
    __tablename__ = "zones"
    
    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    store_id: Mapped[UUID] = mapped_column(ForeignKey("stores.id"), nullable=False)
    
    name: Mapped[str] = mapped_column(String(100), nullable=False)  # "Rayon Électronique"
    zone_type: Mapped[str] = mapped_column(String(50))  # "high_value", "checkout", "exit"
    
    # Polygone de la zone (coordonnées normalisées)
    polygon: Mapped[dict] = mapped_column(JSONB)  # {"points": [[x1,y1], [x2,y2], ...]}
    
    # Paramètres spécifiques
    suspicion_multiplier: Mapped[float] = mapped_column(Float, default=1.0)
    max_loiter_seconds: Mapped[int] = mapped_column(Integer, default=60)
    
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    
    # Relations
    store: Mapped["Store"] = relationship(back_populates="zones")


# === TABLE: STREAMS (Caméras) ===

class Stream(Base):
    """Flux vidéo (caméra) surveillé."""
    
    __tablename__ = "streams"
    
    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    store_id: Mapped[UUID] = mapped_column(ForeignKey("stores.id"), nullable=False)
    
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    location: Mapped[str] = mapped_column(String(200))  # "Entrée principale"
    source_url: Mapped[str] = mapped_column(String(500), nullable=False)  # RTSP URL
    
    # Configuration
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    fps_target: Mapped[int] = mapped_column(Integer, default=15)
    resolution: Mapped[Optional[str]] = mapped_column(String(20))  # "1920x1080"
    
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())
    
    # Relations
    store: Mapped["Store"] = relationship(back_populates="streams")
    alerts: Mapped[list["Alert"]] = relationship(back_populates="stream")


# === TABLE: ALERTS ===

class Alert(Base):
    """Alerte générée par le système."""
    
    __tablename__ = "alerts"
    
    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    stream_id: Mapped[UUID] = mapped_column(ForeignKey("streams.id"), nullable=False)
    
    # Niveau et type
    level: Mapped[str] = mapped_column(String(20), nullable=False)  # info, attention, critical
    alert_type: Mapped[str] = mapped_column(String(50), nullable=False)  # "loitering", "concealment"
    
    # Contexte
    track_id: Mapped[Optional[int]] = mapped_column(Integer)
    suspicion_score: Mapped[float] = mapped_column(Float, nullable=False)
    
    # Description
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    
    # Preuves
    frame_path: Mapped[Optional[str]] = mapped_column(String(500))
    video_clip_path: Mapped[Optional[str]] = mapped_column(String(500))
    bbox: Mapped[Optional[list]] = mapped_column(ARRAY(Float))
    
    # Analyse VLM
    vlm_analysis_id: Mapped[Optional[UUID]] = mapped_column(ForeignKey("vlm_analyses.id"))
    
    # Feedback opérateur
    is_acknowledged: Mapped[bool] = mapped_column(Boolean, default=False)
    acknowledged_by: Mapped[Optional[str]] = mapped_column(String(100))
    acknowledged_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    is_false_positive: Mapped[Optional[bool]] = mapped_column(Boolean)
    feedback_notes: Mapped[Optional[str]] = mapped_column(Text)
    
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    
    # Relations
    stream: Mapped["Stream"] = relationship(back_populates="alerts")
    vlm_analysis: Mapped[Optional["VLMAnalysis"]] = relationship(back_populates="alerts")
    
    __table_args__ = (
        Index("ix_alerts_level_created", "level", "created_at"),
        Index("ix_alerts_stream_created", "stream_id", "created_at"),
    )


# === TABLE: TRAJECTORIES ===

class Trajectory(Base):
    """Trajectoire d'une personne trackée."""
    
    __tablename__ = "trajectories"
    
    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    stream_id: Mapped[UUID] = mapped_column(ForeignKey("streams.id"), nullable=False)
    track_id: Mapped[int] = mapped_column(Integer, nullable=False)
    
    # Timestamps
    start_time: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    end_time: Mapped[Optional[datetime]] = mapped_column(DateTime)
    
    # Points: [{"x": 0.5, "y": 0.3, "t": "2024-01-01T10:00:00"}, ...]
    points: Mapped[list] = mapped_column(JSONB, default=list)
    
    # Métriques
    total_distance: Mapped[float] = mapped_column(Float, default=0.0)
    avg_speed: Mapped[float] = mapped_column(Float, default=0.0)
    zones_visited: Mapped[list] = mapped_column(ARRAY(String), default=list)
    
    # Suspicion
    max_suspicion_score: Mapped[float] = mapped_column(Float, default=0.0)
    suspicion_reasons: Mapped[list] = mapped_column(ARRAY(String), default=list)
    
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    
    __table_args__ = (
        Index("ix_trajectories_stream_track", "stream_id", "track_id"),
    )


# === TABLE: VLM_ANALYSES ===

class VLMAnalysis(Base):
    """Analyse contextuelle par le VLM."""
    
    __tablename__ = "vlm_analyses"
    
    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    stream_id: Mapped[UUID] = mapped_column(ForeignKey("streams.id"), nullable=False)
    
    # Statut
    status: Mapped[str] = mapped_column(String(20), default="pending")
    
    # Input
    track_ids: Mapped[list] = mapped_column(ARRAY(Integer), nullable=False)
    frame_paths: Mapped[list] = mapped_column(ARRAY(String), nullable=False)
    initial_suspicion_score: Mapped[float] = mapped_column(Float, nullable=False)
    
    # Outils utilisés
    tools_used: Mapped[list] = mapped_column(ARRAY(String), default=list)
    tools_results: Mapped[Optional[dict]] = mapped_column(JSONB)
    
    # Output VLM
    vlm_response: Mapped[Optional[str]] = mapped_column(Text)
    final_suspicion_score: Mapped[Optional[float]] = mapped_column(Float)
    is_threat_confirmed: Mapped[Optional[bool]] = mapped_column(Boolean)
    reasoning: Mapped[Optional[str]] = mapped_column(Text)
    
    # Performance
    inference_time_ms: Mapped[Optional[int]] = mapped_column(Integer)
    tokens_used: Mapped[Optional[int]] = mapped_column(Integer)
    
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    
    # Relations
    alerts: Mapped[list["Alert"]] = relationship(back_populates="vlm_analysis")


# === TABLE: SYSTEM_METRICS ===

class SystemMetric(Base):
    """Métriques système pour monitoring."""
    
    __tablename__ = "system_metrics"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    
    metric_type: Mapped[str] = mapped_column(String(50), nullable=False)  # "fps", "latency", "gpu_usage"
    value: Mapped[float] = mapped_column(Float, nullable=False)
    unit: Mapped[Optional[str]] = mapped_column(String(20))
    
    stream_id: Mapped[Optional[UUID]] = mapped_column(ForeignKey("streams.id"))
    extra_data: Mapped[Optional[dict]] = mapped_column(JSONB)
    
    timestamp: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    
    __table_args__ = (
        Index("ix_metrics_type_time", "metric_type", "timestamp"),
    )