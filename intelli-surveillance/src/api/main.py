"""
API FastAPI principale - IntelliSurveillance.
Point d'entrée du backend avec REST et WebSocket.
"""

from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4

import torch
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse
from loguru import logger

from src.core.config import settings
from src.core.logging import setup_logging
from src.models.schemas import (
    AlertCreate, AlertResponse, AlertUpdate, AlertStats,
    StreamCreate, StreamResponse, StreamUpdate,
    VLMAnalysisRequest, VLMAnalysisResponse,
    APIResponse, HealthCheck, AnalysisStatus
)


# === LIFESPAN ===

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestion du cycle de vie."""
    # Startup
    setup_logging()
    logger.info(f"Starting {settings.project_name} v{settings.version}")
    logger.info(f"Environment: {settings.env}, Debug: {settings.debug}")
    
    # Initialiser le service de détection
    try:
        from src.services.detection.detector import get_detection_service
        detection = get_detection_service()
        await detection.initialize()
        logger.info("Detection service ready")
    except Exception as e:
        logger.warning(f"Detection service init failed: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")
    try:
        from src.services.detection.detector import get_detection_service
        from src.services.vlm.orchestrator import get_vlm_orchestrator
        
        await get_detection_service().shutdown()
        await get_vlm_orchestrator().shutdown()
    except Exception as e:
        logger.error(f"Shutdown error: {e}")
    
    logger.info("Shutdown complete")


# === APPLICATION ===

app = FastAPI(
    title=settings.project_name,
    version=settings.version,
    description="API de Surveillance Intelligente pour la Prévention du Vol",
    default_response_class=ORJSONResponse,
    lifespan=lifespan,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# === WEBSOCKET MANAGER ===

class ConnectionManager:
    """Gestionnaire des connexions WebSocket."""
    
    def __init__(self):
        self.connections: dict[str, WebSocket] = {}
        self.subscriptions: dict[UUID, set[str]] = {}
    
    async def connect(self, ws: WebSocket, client_id: str):
        await ws.accept()
        self.connections[client_id] = ws
        logger.info(f"WS connected: {client_id}")
    
    def disconnect(self, client_id: str):
        self.connections.pop(client_id, None)
        for subs in self.subscriptions.values():
            subs.discard(client_id)
        logger.info(f"WS disconnected: {client_id}")
    
    def subscribe(self, client_id: str, stream_id: UUID):
        if stream_id not in self.subscriptions:
            self.subscriptions[stream_id] = set()
        self.subscriptions[stream_id].add(client_id)
    
    async def broadcast_to_stream(self, stream_id: UUID, message: dict):
        if stream_id not in self.subscriptions:
            return
        for client_id in list(self.subscriptions[stream_id]):
            if client_id in self.connections:
                try:
                    await self.connections[client_id].send_json(message)
                except:
                    self.disconnect(client_id)
    
    async def broadcast_all(self, message: dict):
        for client_id, ws in list(self.connections.items()):
            try:
                await ws.send_json(message)
            except:
                self.disconnect(client_id)


ws_manager = ConnectionManager()


# === ROUTES: HEALTH ===

@app.get("/", response_model=APIResponse)
async def root():
    return APIResponse(
        success=True,
        message=f"{settings.project_name} API",
        data={"version": settings.version}
    )


@app.get("/health", response_model=HealthCheck)
async def health_check():
    from src.services.detection.detector import get_detection_service
    from src.services.vlm.orchestrator import get_vlm_orchestrator
    
    detection = get_detection_service()
    vlm = get_vlm_orchestrator()
    
    return HealthCheck(
        status="healthy",
        version=settings.version,
        environment=settings.env,
        database=True,  # TODO: vérifier connexion DB
        redis=True,     # TODO: vérifier connexion Redis
        gpu_available=torch.cuda.is_available(),
        models_loaded={
            "detection": detection.is_initialized,
            "vlm": vlm.is_initialized
        }
    )


# === ROUTES: STREAMS ===

@app.post("/api/v1/streams", response_model=StreamResponse, status_code=201)
async def create_stream(stream: StreamCreate):
    """Crée un nouveau flux de surveillance."""
    # TODO: Sauvegarder en DB
    logger.info(f"Creating stream: {stream.name}")
    
    return StreamResponse(
        id=uuid4(),
        store_id=stream.store_id,
        name=stream.name,
        location=stream.location,
        source_url=stream.source_url,
        fps_target=stream.fps_target,
        resolution=stream.resolution,
        is_active=True,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )


@app.get("/api/v1/streams", response_model=list[StreamResponse])
async def list_streams(
    store_id: Optional[UUID] = Query(None),
    is_active: Optional[bool] = Query(None),
    limit: int = Query(20, ge=1, le=100)
):
    """Liste les flux de surveillance."""
    # TODO: Requête DB
    return []


@app.get("/api/v1/streams/{stream_id}", response_model=StreamResponse)
async def get_stream(stream_id: UUID):
    # TODO: Requête DB
    raise HTTPException(status_code=404, detail="Stream not found")


# === ROUTES: ALERTS ===

@app.post("/api/v1/alerts", response_model=AlertResponse, status_code=201)
async def create_alert(alert: AlertCreate):
    """Crée une nouvelle alerte."""
    logger.info(f"Creating alert: {alert.title} (level: {alert.level})")
    
    response = AlertResponse(
        id=uuid4(),
        stream_id=alert.stream_id,
        level=alert.level,
        alert_type=alert.alert_type,
        title=alert.title,
        description=alert.description,
        suspicion_score=alert.suspicion_score,
        track_id=alert.track_id,
        frame_path=alert.frame_path,
        video_clip_path=alert.video_clip_path,
        bbox=alert.bbox,
        vlm_analysis_id=None,
        is_acknowledged=False,
        acknowledged_by=None,
        acknowledged_at=None,
        is_false_positive=None,
        feedback_notes=None,
        created_at=datetime.utcnow()
    )
    
    # Broadcast WebSocket
    await ws_manager.broadcast_to_stream(
        alert.stream_id,
        {"type": "alert", "data": response.model_dump(mode="json")}
    )
    
    return response


@app.get("/api/v1/alerts", response_model=list[AlertResponse])
async def list_alerts(
    stream_id: Optional[UUID] = Query(None),
    level: Optional[str] = Query(None),
    is_acknowledged: Optional[bool] = Query(None),
    limit: int = Query(50, ge=1, le=200)
):
    """Liste les alertes avec filtres."""
    # TODO: Requête DB
    return []


@app.patch("/api/v1/alerts/{alert_id}", response_model=AlertResponse)
async def update_alert(alert_id: UUID, update: AlertUpdate):
    """Met à jour une alerte (acknowledge, feedback)."""
    # TODO: Update DB
    raise HTTPException(status_code=404, detail="Alert not found")


@app.get("/api/v1/alerts/stats", response_model=AlertStats)
async def get_alert_stats(
    stream_id: Optional[UUID] = Query(None),
    days: int = Query(7, ge=1, le=90)
):
    """Statistiques des alertes."""
    now = datetime.utcnow()
    # TODO: Calculer depuis DB
    return AlertStats(
        total=0,
        by_level={"info": 0, "attention": 0, "critical": 0},
        by_type={},
        acknowledged=0,
        false_positives=0,
        period_start=now,
        period_end=now
    )


# === ROUTES: VLM ANALYSIS ===

@app.post("/api/v1/analysis", response_model=VLMAnalysisResponse, status_code=202)
async def request_analysis(request: VLMAnalysisRequest):
    """Demande une analyse VLM (async)."""
    logger.info(f"VLM analysis requested for stream {request.stream_id}")
    
    # TODO: Enqueue Celery task
    return VLMAnalysisResponse(
        id=uuid4(),
        stream_id=request.stream_id,
        status=AnalysisStatus.PENDING,
        track_ids=request.track_ids,
        frame_paths=request.frame_paths,
        initial_suspicion_score=request.initial_suspicion_score,
        tools_used=[],
        tools_results=None,
        vlm_response=None,
        final_suspicion_score=None,
        is_threat_confirmed=None,
        reasoning=None,
        inference_time_ms=None,
        tokens_used=None,
        created_at=datetime.utcnow(),
        completed_at=None
    )


@app.get("/api/v1/analysis/{analysis_id}", response_model=VLMAnalysisResponse)
async def get_analysis(analysis_id: UUID):
    """Récupère le résultat d'une analyse."""
    # TODO: Requête DB
    raise HTTPException(status_code=404, detail="Analysis not found")


# === ROUTES: STATS ===

@app.get("/api/v1/stats")
async def get_system_stats():
    """Statistiques système."""
    from src.services.detection.detector import get_detection_service
    from src.services.vlm.orchestrator import get_vlm_orchestrator
    from src.services.analysis.trajectory import get_trajectory_analyzer
    
    return {
        "detection": get_detection_service().stats,
        "vlm": get_vlm_orchestrator().stats,
        "trajectory": get_trajectory_analyzer().stats,
        "websocket_clients": len(ws_manager.connections)
    }


# === WEBSOCKET ===

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """
    WebSocket pour événements temps réel.
    
    Messages:
        → {"action": "subscribe", "stream_id": "..."}
        → {"action": "unsubscribe", "stream_id": "..."}
        → {"action": "ping"}
        
        ← {"type": "detection", "data": {...}}
        ← {"type": "alert", "data": {...}}
        ← {"type": "pong"}
    """
    await ws_manager.connect(websocket, client_id)
    
    try:
        while True:
            data = await websocket.receive_json()
            action = data.get("action")
            
            if action == "subscribe":
                stream_id = UUID(data["stream_id"])
                ws_manager.subscribe(client_id, stream_id)
                await websocket.send_json({"type": "subscribed", "stream_id": str(stream_id)})
            
            elif action == "unsubscribe":
                stream_id = UUID(data["stream_id"])
                if stream_id in ws_manager.subscriptions:
                    ws_manager.subscriptions[stream_id].discard(client_id)
                await websocket.send_json({"type": "unsubscribed", "stream_id": str(stream_id)})
            
            elif action == "ping":
                await websocket.send_json({"type": "pong"})
            
            else:
                await websocket.send_json({"type": "error", "message": f"Unknown action: {action}"})
    
    except WebSocketDisconnect:
        ws_manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error {client_id}: {e}")
        ws_manager.disconnect(client_id)


# === MAIN ===

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.api.main:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.debug,
        workers=1 if settings.debug else settings.api.workers
    )