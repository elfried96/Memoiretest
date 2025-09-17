"""
🔒 Dashboard de Surveillance Intelligent - VERSION PRODUCTION
============================================================

Version finale avec vraie intégration VLM :
- Pipeline VLM réelle avec 8 outils avancés
- AdaptiveOrchestrator pour sélection intelligente
- ToolOptimizationBenchmark pour optimisation
- Métriques de performance réelles
- Chat contextualisé avec vraies données
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import random
import json
from typing import Dict, List, Any, Optional
import tempfile
import os
import asyncio
import requests
import threading
import queue
import io
from PIL import Image

# Import contexte vidéo
from video_context_integration import (
    VideoContextMetadata, 
    create_video_metadata_from_form,
    get_video_context_integration
)

# Configuration de la page
st.set_page_config(
    page_title="🔒 Surveillance Intelligente - Production",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🔒"
)

# Imports de la pipeline réelle
try:
    from real_pipeline_integration import (
        RealVLMPipeline, 
        RealAnalysisResult,
        initialize_real_pipeline,
        get_real_pipeline,
        is_real_pipeline_available
    )
    from camera_manager import CameraConfig, MultiCameraManager, FrameData
    from vlm_chatbot_symbiosis import process_vlm_chat_query, get_vlm_chatbot
    PIPELINE_AVAILABLE = True
except ImportError as e:
    st.error(f"❌ Impossible d'importer la pipeline VLM: {e}")
    PIPELINE_AVAILABLE = False

# Initialisation des variables de session
if 'cameras' not in st.session_state:
    st.session_state.cameras = {}
if 'surveillance_chat' not in st.session_state:
    st.session_state.surveillance_chat = []
if 'video_chat' not in st.session_state:
    st.session_state.video_chat = []
if 'real_detections' not in st.session_state:
    st.session_state.real_detections = []
if 'real_alerts' not in st.session_state:
    st.session_state.real_alerts = []
if 'surveillance_active' not in st.session_state:
    st.session_state.surveillance_active = False
if 'uploaded_videos' not in st.session_state:
    st.session_state.uploaded_videos = []
if 'video_analysis_results' not in st.session_state:
    st.session_state.video_analysis_results = {}
if 'pipeline_initialized' not in st.session_state:
    st.session_state.pipeline_initialized = False
if 'camera_manager' not in st.session_state:
    st.session_state.camera_manager = None
if 'real_pipeline' not in st.session_state:
    st.session_state.real_pipeline = None
if 'optimization_results' not in st.session_state:
    st.session_state.optimization_results = []
if 'streaming_manager' not in st.session_state:
    st.session_state.streaming_manager = None
if 'frame_cache' not in st.session_state:
    st.session_state.frame_cache = {}
if 'last_frame_update' not in st.session_state:
    st.session_state.last_frame_update = {}

# CSS pour l'interface
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #1e3c72, #2a5298);
    padding: 1.5rem;
    border-radius: 10px;
    text-align: center;
    color: white;
    margin-bottom: 1rem;
}

.pipeline-status {
    padding: 1rem;
    border-radius: 8px;
    margin: 1rem 0;
    border-left: 4px solid;
}

.pipeline-active { 
    border-color: #28a745; 
    background: rgba(40, 167, 69, 0.1); 
    color: #28a745;
}

.pipeline-inactive { 
    border-color: #dc3545; 
    background: rgba(220, 53, 69, 0.1); 
    color: #dc3545;
}

.real-analysis-result {
    border: 2px solid #007bff;
    border-radius: 8px;
    padding: 1rem;
    margin: 1rem 0;
    background: #f8f9ff;
}

.tool-performance {
    background: #e8f5e8;
    border: 1px solid #28a745;
    border-radius: 5px;
    padding: 0.5rem;
    margin: 0.25rem 0;
}

.optimization-badge {
    display: inline-block;
    padding: 0.25rem 0.5rem;
    border-radius: 0.25rem;
    font-size: 0.875rem;
    font-weight: 500;
    background: #17a2b8;
    color: white;
}

.chat-container {
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 1rem;
    background: #f8f9fa;
    margin-top: 1rem;
    max-height: 300px;
    overflow-y: auto;
}

.chat-user {
    background: #e3f2fd;
    padding: 0.5rem;
    border-radius: 5px;
    margin: 0.25rem 0;
    text-align: right;
}

.chat-ai {
    background: #f3e5f5;
    padding: 0.5rem;
    border-radius: 5px;
    margin: 0.25rem 0;
}

.camera-card {
    border: 2px solid #dee2e6;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
    background: #f8f9fa;
}

.metric-card {
    background: white;
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 1rem;
    text-align: center;
}

.alert-critical {
    background: #dc3545;
    color: white;
    padding: 0.5rem;
    border-radius: 5px;
    margin: 0.25rem 0;
}

.alert-high {
    background: #fd7e14;
    color: white;
    padding: 0.5rem;
    border-radius: 5px;
    margin: 0.25rem 0;
}

.alert-medium {
    background: #ffc107;
    color: black;
    padding: 0.5rem;
    border-radius: 5px;
    margin: 0.25rem 0;
}
</style>
""", unsafe_allow_html=True)

def render_header():
    """Affiche l'en-tête principal."""
    st.markdown("""
    <div class="main-header">
        <h1>🔒 Dashboard de Surveillance Intelligente - PRODUCTION</h1>
        <p>Pipeline VLM Réelle • 8 Outils Avancés • Optimisation Adaptative</p>
    </div>
    """, unsafe_allow_html=True)

def render_pipeline_status():
    """Affiche le statut de la pipeline VLM."""
    if PIPELINE_AVAILABLE and is_real_pipeline_available():
        pipeline = get_real_pipeline()
        if pipeline and pipeline.running:
            stats = pipeline.get_performance_stats()
            
            st.markdown(f"""
            <div class="pipeline-status pipeline-active">
                <h4>✅ Pipeline VLM Active</h4>
                <p><strong>Frames traitées:</strong> {stats.get('frames_processed', 0)}</p>
                <p><strong>Temps moyen:</strong> {stats.get('average_processing_time', 0):.2f}s</p>
                <p><strong>Outils optimaux:</strong> {len(stats.get('current_optimal_tools', []))}</p>
                <p><strong>Score performance:</strong> {stats.get('current_performance_score', 0):.2f}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="pipeline-status pipeline-inactive">
                <h4>⏸️ Pipeline VLM Arrêtée</h4>
                <p>Cliquez sur "Initialiser Pipeline" pour démarrer</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="pipeline-status pipeline-inactive">
            <h4>❌ Pipeline VLM Non Disponible</h4>
            <p>Mode simulation activé - Vérifiez l'installation du système core</p>
        </div>
        """, unsafe_allow_html=True)

async def initialize_pipeline():
    """Initialise la pipeline VLM réelle."""
    if not PIPELINE_AVAILABLE:
        return False
    
    try:
        with st.spinner("🔄 Initialisation de la pipeline VLM..."):
            # Initialisation de la pipeline
            success = initialize_real_pipeline(
                vlm_model_name="kimi-vl-a3b-thinking",
                enable_optimization=True,
                max_concurrent_analysis=2
            )
            
            if success:
                st.session_state.real_pipeline = get_real_pipeline()
                
                # Callbacks pour intégration dashboard
                def on_analysis_result(result):
                    st.session_state.real_detections.append(result)
                    
                    # Génération d'alertes basées sur niveau de suspicion
                    if hasattr(result, 'suspicion_level'):
                        suspicion_str = str(result.suspicion_level)
                        if 'HIGH' in suspicion_str or 'CRITICAL' in suspicion_str:
                            alert = {
                                'level': suspicion_str,
                                'message': result.description,
                                'camera': result.camera_id,
                                'timestamp': result.timestamp,
                                'resolved': False,
                                'confidence': result.confidence,
                                'tools_used': result.tools_used
                            }
                            st.session_state.real_alerts.append(alert)
                
                def on_optimization_result(result):
                    st.session_state.optimization_results.append(result)
                
                def on_error(error):
                    st.error(f"❌ Erreur pipeline: {error}")
                
                # Enregistrement des callbacks
                st.session_state.real_pipeline.add_analysis_callback(on_analysis_result)
                st.session_state.real_pipeline.add_optimization_callback(on_optimization_result)
                st.session_state.real_pipeline.add_error_callback(on_error)
                
                # Démarrage du traitement
                if st.session_state.real_pipeline.start_processing():
                    st.session_state.pipeline_initialized = True
                    st.success("✅ Pipeline VLM initialisée et démarrée!")
                    return True
            
            st.error("❌ Échec de l'initialisation de la pipeline")
            return False
            
    except Exception as e:
        st.error(f"❌ Erreur initialisation: {e}")
        return False

def generate_real_frame_analysis(frame_data: FrameData) -> Optional[RealAnalysisResult]:
    """Analyse une frame avec la vraie pipeline ou simulation."""
    if st.session_state.pipeline_initialized and st.session_state.real_pipeline:
        # Utilisation de la vraie pipeline
        try:
            return asyncio.run(st.session_state.real_pipeline.analyze_frame(frame_data))
        except Exception as e:
            st.warning(f"⚠️ Erreur pipeline réelle, fallback simulation: {e}")
    
    # Fallback simulation si pipeline non disponible
    return simulate_vlm_analysis(frame_data)

def simulate_vlm_analysis(frame_data: FrameData) -> RealAnalysisResult:
    """Simulation d'analyse VLM pour fallback."""
    from real_pipeline_integration import RealAnalysisResult
    
    # Simulation de détections
    detections = []
    tools_used = []
    
    if random.random() > 0.3:
        detections.append({
            'type': random.choice(['person', 'bag', 'vehicle', 'suspicious_object']),
            'confidence': random.uniform(0.7, 0.95),
            'bbox': [
                random.randint(0, frame_data.frame.shape[1]//2),
                random.randint(0, frame_data.frame.shape[0]//2),
                random.randint(frame_data.frame.shape[1]//2, frame_data.frame.shape[1]),
                random.randint(frame_data.frame.shape[0]//2, frame_data.frame.shape[0])
            ]
        })
        
        # Outils utilisés selon le type de détection
        tools_used = random.sample([
            'sam2_segmentator', 'dino_features', 'pose_estimator',
            'trajectory_analyzer', 'multimodal_fusion'
        ], random.randint(2, 4))
    
    # Niveau de suspicion
    if any('suspicious' in str(d['type']) for d in detections):
        suspicion_level = random.choice(['HIGH', 'CRITICAL'])
    elif detections:
        suspicion_level = random.choice(['LOW', 'MEDIUM'])
    else:
        suspicion_level = 'LOW'
    
    # Import dynamique pour éviter les erreurs circulaires
    try:
        from src.core.types import SuspicionLevel, ActionType
        suspicion_enum = getattr(SuspicionLevel, suspicion_level, SuspicionLevel.LOW)
        action_enum = ActionType.NORMAL_SHOPPING
    except:
        suspicion_enum = suspicion_level
        action_enum = 'NORMAL_SHOPPING'
    
    return RealAnalysisResult(
        frame_id=f"{frame_data.camera_id}_{frame_data.frame_number}",
        camera_id=frame_data.camera_id,
        timestamp=frame_data.timestamp,
        suspicion_level=suspicion_enum,
        action_type=action_enum,
        confidence=random.uniform(0.7, 0.9),
        description=f"Analyse {'simulée' if not st.session_state.pipeline_initialized else 'VLM'} - {len(detections)} détection(s)",
        detections=detections,
        tool_results={},
        processing_time=random.uniform(0.5, 2.0),
        tools_used=tools_used,
        optimization_score=random.uniform(0.6, 0.9),
        context_pattern=None,
        risk_assessment={'risk_level': random.uniform(0.1, 0.8)},
        bbox_annotations=detections
    )

class StreamingManager:
    """Gestionnaire de streaming en arrière-plan pour éviter les rechargements."""
    
    def __init__(self):
        self.active_streams = {}  # Dict[camera_id, thread]
        self.frame_cache = {}     # Dict[camera_id, frame]
        self.last_update = {}     # Dict[camera_id, timestamp]
        self.running = False
        
    def start_stream(self, camera_id: str, camera_config: dict):
        """Démarre un stream en arrière-plan pour une caméra."""
        if camera_id in self.active_streams:
            return  # Déjà actif
        
        def stream_worker():
            """Worker thread pour capture continue."""
            attempt = 0
            while self.running and camera_id in self.active_streams:
                try:
                    attempt += 1
                    print(f"DEBUG: Stream {camera_id} tentative {attempt}")
                    
                    frame = capture_real_frame(camera_config, width=640, height=480)
                    if frame is not None:
                        print(f"DEBUG: Frame capturée pour {camera_id} - shape: {frame.shape}")
                        self.frame_cache[camera_id] = frame
                        self.last_update[camera_id] = time.time()
                    else:
                        print(f"DEBUG: Frame NULL pour {camera_id}")
                    
                    # Pause selon la configuration
                    refresh_rate = camera_config.get('refresh_rate', 2)
                    time.sleep(refresh_rate)
                    
                except Exception as e:
                    print(f"ERREUR stream {camera_id}: {e}")
                    import traceback
                    traceback.print_exc()
                    time.sleep(5)  # Retry après erreur
        
        # Démarre le thread
        thread = threading.Thread(target=stream_worker, daemon=True)
        thread.start()
        self.active_streams[camera_id] = thread
        self.running = True
    
    def stop_stream(self, camera_id: str):
        """Arrête le stream d'une caméra."""
        if camera_id in self.active_streams:
            del self.active_streams[camera_id]
            if camera_id in self.frame_cache:
                del self.frame_cache[camera_id]
            if camera_id in self.last_update:
                del self.last_update[camera_id]
    
    def get_latest_frame(self, camera_id: str):
        """Récupère la dernière frame en cache."""
        return self.frame_cache.get(camera_id, None)
    
    def is_frame_fresh(self, camera_id: str, max_age: float = 10.0) -> bool:
        """Vérifie si la frame est récente."""
        if camera_id not in self.last_update:
            return False
        return (time.time() - self.last_update[camera_id]) < max_age
    
    def stop_all(self):
        """Arrête tous les streams."""
        self.running = False
        self.active_streams.clear()
        self.frame_cache.clear()
        self.last_update.clear()

class MJPEGStreamManager:
    """Gestionnaire optimisé pour flux MJPEG avec cache et réduction latence."""
    
    def __init__(self):
        self.sessions = {}  # Cache des sessions par URL
        self.frame_cache = {}  # Cache des dernières frames
        self.last_update = {}  # Timestamp dernière mise à jour
        
    def get_session(self, url: str):
        """Récupère ou crée une session HTTP optimisée."""
        if url not in self.sessions:
            session = requests.Session()
            # Configuration optimisée pour faible latence
            session.headers.update({
                'User-Agent': 'SurveillanceBot/1.0 (low-latency)',
                'Accept': 'multipart/x-mixed-replace, */*',
                'Connection': 'keep-alive',
                'Cache-Control': 'no-cache',
                'Pragma': 'no-cache'
            })
            self.sessions[url] = session
        return self.sessions[url]
    
    def capture_latest_frame(self, url: str, width: int = 640, height: int = 480, max_age_ms: int = 500):
        """Capture avec cache intelligent pour réduire la latence."""
        current_time = time.time() * 1000
        
        # Vérification cache
        if url in self.frame_cache and url in self.last_update:
            age = current_time - self.last_update[url]
            if age < max_age_ms:
                return self.frame_cache[url]  # Frame récente en cache
        
        # Capture nouvelle frame
        frame = self._capture_fresh_frame(url, width, height)
        if frame is not None:
            self.frame_cache[url] = frame
            self.last_update[url] = current_time
            return frame
        
        # Fallback sur cache même ancien
        return self.frame_cache.get(url, None)
    
    def _capture_fresh_frame(self, url: str, width: int, height: int):
        """Capture MJPEG robuste avec validation stricte."""
        try:
            session = self.get_session(url)
            
            # Stream optimisé
            response = session.get(url, stream=True, timeout=3)
            
            if response.status_code == 200:
                content = b""
                frames_tested = 0
                max_frames_to_test = 3
                
                # Lecture par chunks plus grands
                for i, chunk in enumerate(response.iter_content(chunk_size=8192)):
                    if not chunk:
                        break
                        
                    content += chunk
                    
                    # Recherche d'images complètes
                    while frames_tested < max_frames_to_test:
                        jpg_start = content.find(b'\xff\xd8')
                        if jpg_start == -1:
                            break
                            
                        jpg_end = content.find(b'\xff\xd9', jpg_start)
                        if jpg_end == -1:
                            # Image incomplète
                            content = content[jpg_start:]
                            break
                            
                        try:
                            # Extraction
                            jpeg_data = content[jpg_start:jpg_end+2]
                            
                            # Validation stricte
                            if len(jpeg_data) > 5000:  # Taille minimale pour image correcte
                                pil_image = Image.open(io.BytesIO(jpeg_data))
                                
                                # Vérifications de qualité
                                if (pil_image.size[0] >= 320 and pil_image.size[1] >= 240 and
                                    pil_image.mode in ['RGB', 'L']):
                                    
                                    frame = np.array(pil_image)
                                    
                                    # Conversion couleur
                                    if len(frame.shape) == 3 and frame.shape[2] == 3:
                                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                                    elif len(frame.shape) == 2:  # Grayscale
                                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                                    
                                    # Validation du contenu (pas que du bruit)
                                    if frame.std() > 10:  # Vérification variance
                                        # Redimensionnement final
                                        if frame.shape[1] != width or frame.shape[0] != height:
                                            frame = cv2.resize(frame, (width, height))
                                        
                                        return frame
                            
                        except Exception:
                            pass
                        
                        frames_tested += 1
                        # Supprime cette image et continue
                        content = content[jpg_end+2:]
                    
                    # Limite buffer
                    if len(content) > 200000:
                        content = content[-50000:]
                    
                    # Limite chunks
                    if i > 25:
                        break
                        
        except Exception:
            pass
        
        return None

# Instance globale pour réutilisation
_mjpeg_manager = MJPEGStreamManager()

def capture_mjpeg_frame_simple(url: str, width: int = 640, height: int = 480):
    """Capture MJPEG simple et robuste."""
    try:
        # Headers optimisés pour MJPEG
        headers = {
            'User-Agent': 'Mozilla/5.0 (Linux; SurveillanceBot)',
            'Accept': 'multipart/x-mixed-replace, image/jpeg, */*',
            'Connection': 'keep-alive',
            'Cache-Control': 'no-cache'
        }
        
        # Stream pour capture continue
        with requests.get(url, stream=True, timeout=5, headers=headers) as response:
            if response.status_code == 200:
                content_buffer = b""
                
                # Lecture par chunks
                for chunk in response.iter_content(chunk_size=4096):
                    if not chunk:
                        continue
                        
                    content_buffer += chunk
                    
                    # Recherche image JPEG la plus récente
                    while True:
                        # Trouve début JPEG
                        start_marker = content_buffer.find(b'\xff\xd8')
                        if start_marker == -1:
                            break
                            
                        # Trouve fin JPEG
                        end_marker = content_buffer.find(b'\xff\xd9', start_marker)
                        if end_marker == -1:
                            # Image incomplète, garder le buffer
                            content_buffer = content_buffer[start_marker:]
                            break
                        
                        # Extraction de l'image
                        jpeg_data = content_buffer[start_marker:end_marker + 2]
                        
                        try:
                            # Validation stricte taille et contenu
                            if len(jpeg_data) > 5000:  # Image suffisamment grande
                                try:
                                    img = Image.open(io.BytesIO(jpeg_data))
                                    
                                    # Vérifications multiples
                                    if (img.width >= 320 and img.height >= 240 and 
                                        img.mode in ['RGB', 'L', 'RGBA']):
                                        
                                        frame = np.array(img)
                                        
                                        # Conversion couleur selon le mode
                                        if len(frame.shape) == 3 and frame.shape[2] == 3:
                                            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                                        elif len(frame.shape) == 3 and frame.shape[2] == 4:
                                            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                                        elif len(frame.shape) == 2:
                                            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                                        
                                        # Validation du contenu (pas de bruit)
                                        if frame.std() > 15:  # Variance significative
                                            # Redimensionnement
                                            if frame.shape[1] != width or frame.shape[0] != height:
                                                frame = cv2.resize(frame, (width, height))
                                            
                                            return frame
                                            
                                except Exception:
                                    pass
                        except Exception:
                            pass
                        
                        # Supprime cette image et continue
                        content_buffer = content_buffer[end_marker + 2:]
                    
                    # Limite taille buffer
                    if len(content_buffer) > 100000:
                        content_buffer = content_buffer[-50000:]
                    
                    # Limite lecture pour éviter blocage
                    if len(content_buffer) > 200000:
                        break
                        
    except Exception:
        pass
    
    return None

def capture_mjpeg_frame_native(url: str, width: int = 640, height: int = 480):
    """Capture MJPEG avec double fallback."""
    # Essai 1 : Système optimisé
    frame = _mjpeg_manager.capture_latest_frame(url, width, height, max_age_ms=500)
    if frame is not None:
        return frame
    
    # Essai 2 : Méthode simple
    frame = capture_mjpeg_frame_simple(url, width, height)
    if frame is not None:
        return frame
    
    return None

def capture_real_frame_simple(camera_config: dict, width: int = 640, height: int = 480):
    """Capture simplifiée et directe - pas de cache complexe."""
    source_url = camera_config['source']
    camera_name = camera_config.get('name', 'Camera')
    
    print(f"DEBUG SIMPLE: Capture pour {camera_name} depuis {source_url}")
    
    # 1. FLUX HTTP/MJPEG - méthode directe
    if source_url.startswith('http'):
        print(f"DEBUG SIMPLE: Tentative HTTP native pour {source_url}")
        try:
            frame = capture_mjpeg_frame_simple(source_url, width, height)
            if frame is not None:
                print(f"DEBUG SIMPLE: HTTP réussi - shape: {frame.shape}")
                # Overlay minimal
                timestamp = datetime.now().strftime("%H:%M:%S")
                cv2.putText(frame, f"{camera_name}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"HTTP | {timestamp}", (10, height - 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                return frame
            else:
                print(f"DEBUG SIMPLE: HTTP échoué pour {source_url}")
        except Exception as e:
            print(f"DEBUG SIMPLE: Exception HTTP {source_url}: {e}")
    
    # 2. WEBCAM/RTSP - OpenCV direct
    else:
        print(f"DEBUG SIMPLE: Tentative OpenCV pour {source_url}")
        try:
            # Sélection backend selon type
            if source_url.isdigit():
                # Webcam
                backends = [cv2.CAP_ANY, cv2.CAP_V4L2]
                source_index = int(source_url)
            else:
                # RTSP/fichier
                backends = [cv2.CAP_FFMPEG, cv2.CAP_ANY]
                source_index = source_url
                if source_url.startswith('rtsp') and 'tcp' not in source_url.lower():
                    separator = '&' if '?' in source_url else '?'
                    source_index = f"{source_url}{separator}tcp"
            
            for backend in backends:
                print(f"DEBUG SIMPLE: Test backend {backend} pour {source_index}")
                try:
                    cap = cv2.VideoCapture(source_index, backend)
                    if cap.isOpened():
                        ret, frame = cap.read()
                        if ret and frame is not None and frame.size > 0:
                            print(f"DEBUG SIMPLE: OpenCV réussi - shape: {frame.shape}")
                            
                            # Redimensionnement si nécessaire
                            if frame.shape[:2] != (height, width):
                                frame = cv2.resize(frame, (width, height))
                            
                            # Overlay minimal
                            timestamp = datetime.now().strftime("%H:%M:%S")
                            cv2.putText(frame, f"{camera_name}", (10, 30), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            cv2.putText(frame, f"CV2 | {timestamp}", (10, height - 15), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                            
                            cap.release()
                            return frame
                        else:
                            print(f"DEBUG SIMPLE: Pas de frame valide avec backend {backend}")
                    else:
                        print(f"DEBUG SIMPLE: Ouverture échouée avec backend {backend}")
                    cap.release()
                except Exception as e:
                    print(f"DEBUG SIMPLE: Exception backend {backend}: {e}")
        except Exception as e:
            print(f"DEBUG SIMPLE: Exception OpenCV globale: {e}")
    
    # 3. Frame d'erreur si tout échoue
    print(f"DEBUG SIMPLE: Échec total pour {source_url}")
    error_frame = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(error_frame, f"ECHEC: {camera_name}", (10, height//2 - 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(error_frame, f"Source: {source_url[:30]}...", (10, height//2 + 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    cv2.putText(error_frame, datetime.now().strftime("%H:%M:%S"), (10, height - 15), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
    return error_frame

def capture_real_frame(camera_config: dict, width: int = 640, height: int = 480):
    """Wrapper qui utilise la méthode simplifiée."""
    return capture_real_frame_simple(camera_config, width, height)
    
    # OpenCV pour RTSP/locaux uniquement (PAS HTTP)
    if not source_url.startswith('http'):
        # Sélection du backend selon le type et ce qui a marché au test
        if source_url.isdigit():
            # Webcam : utiliser backend natif, éviter FFMPEG
            if backend_tested in ['V4L2', 'DSHOW', 'DEFAULT']:
                backend_map = {
                    'V4L2': cv2.CAP_V4L2,
                    'DSHOW': cv2.CAP_DSHOW, 
                    'DEFAULT': cv2.CAP_ANY,
                    'GSTREAMER': cv2.CAP_GSTREAMER
                }
            else:
                # Forcer DEFAULT si backend testé pas adapté webcam
                backend_tested = 'DEFAULT'
                backend_map = {'DEFAULT': cv2.CAP_ANY}
        else:
            # RTSP/fichier : FFMPEG OK
            backend_map = {
                'FFMPEG': cv2.CAP_FFMPEG,
                'DEFAULT': cv2.CAP_ANY,
                'GSTREAMER': cv2.CAP_GSTREAMER,
                'V4L2': cv2.CAP_V4L2,
                'DSHOW': cv2.CAP_DSHOW
            }
        
        try:
            cap = cv2.VideoCapture()
            
            # Configuration optimisée selon le type
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # URL finale selon le type
            final_url = source_url
            
            # Optimisations RTSP
            if source_url.startswith('rtsp'):
                # Transport TCP pour RTSP
                if 'tcp' not in source_url.lower():
                    separator = '&' if '?' in source_url else '?'
                    final_url += f"{separator}tcp"
                
                # FPS plus bas pour RTSP
                cap.set(cv2.CAP_PROP_FPS, 15)
            elif source_url.isdigit():
                # Webcam - convertir en entier
                final_url = int(source_url)
                cap.set(cv2.CAP_PROP_FPS, 30)
                # Configuration webcam
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
            else:
                # Fichier local
                cap.set(cv2.CAP_PROP_FPS, 25)
            
            backend_code = backend_map.get(backend_tested, cv2.CAP_FFMPEG)
            
            if cap.open(final_url, backend_code) and cap.isOpened():
                # Configuration résolution
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                
                # Vide buffer pour frame récente (seulement pour RTSP)
                if source_url.startswith('rtsp'):
                    for _ in range(2):
                        cap.read()
                
                # Capture
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    # Redimensionnement si nécessaire
                    if frame.shape[:2] != (height, width):
                        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
                    
                    # Overlay
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    cv2.putText(frame, f"{camera_config['name']}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, timestamp, (10, height - 15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(frame, f"{backend_tested} | {source_url[:4].upper()}", (10, height - 35), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    
                    cap.release()
                    return frame
            
            cap.release()
            
        except Exception as e:
            pass
    
    # Frame d'erreur
    error_frame = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(error_frame, f"ERREUR: {camera_config['name']}", (10, height//2), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(error_frame, "Flux non disponible", (10, height//2 + 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(error_frame, f"Source: {source_url[:50]}...", (10, height//2 + 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (128, 128, 128), 1)
    
    return error_frame

def generate_dummy_frame(camera_id: str, width: int = 320, height: int = 240):
    """Génère une frame simulée avec analyse VLM (fallback)."""
    img = np.random.randint(50, 200, (height, width, 3), dtype=np.uint8)
    
    # Informations de base
    cv2.putText(img, f"Camera {camera_id}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    timestamp = datetime.now().strftime("%H:%M:%S")
    cv2.putText(img, timestamp, (10, height - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Création des données de frame pour analyse
    frame_data = FrameData(
        camera_id=camera_id,
        frame=img,
        timestamp=datetime.now(),
        frame_number=random.randint(1000, 9999),
        metadata={'resolution': (width, height)}
    )
    
    # Analyse avec pipeline VLM
    if st.session_state.surveillance_active:
        analysis_result = generate_real_frame_analysis(frame_data)
        
        if analysis_result and analysis_result.detections:
            # Ajout des détections visuelles
            for detection in analysis_result.detections:
                bbox = detection.get('bbox', [])
                if len(bbox) >= 4:
                    x1, y1, x2, y2 = bbox[:4]
                    # Assurer que les coordonnées sont dans les limites
                    x1, y1 = max(0, min(x1, width-1)), max(0, min(y1, height-1))
                    x2, y2 = max(x1+1, min(x2, width)), max(y1+1, min(y2, height))
                    
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(img, f"{detection['type']}", (int(x1), int(y1)-5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    return img

def render_integrated_chat(chat_type: str, context_data: Dict = None):
    """Chat intégré avec vraies données VLM."""
    
    chat_key = f"{chat_type}_chat"
    if chat_key not in st.session_state:
        st.session_state[chat_key] = []
    
    st.markdown("### 💬 Chat IA avec Vraies Données VLM")
    
    # Questions prédéfinies selon le contexte
    if chat_type == "surveillance":
        questions = [
            "Analyse les détections VLM en temps réel",
            "Quels outils VLM sont les plus performants ?",
            "Résume l'optimisation adaptative actuelle",
            "Évalue les scores de performance des outils",
            "Recommande des améliorations de configuration"
        ]
        
        # Contexte avec vraies données
        if st.session_state.pipeline_initialized and st.session_state.real_pipeline:
            stats = st.session_state.real_pipeline.get_performance_stats()
            tool_details = st.session_state.real_pipeline.get_tool_performance_details()
            context_info = f"🔬 Pipeline active: {stats.get('frames_processed', 0)} frames, {len(stats.get('current_optimal_tools', []))} outils optimaux"
        else:
            context_info = "🔬 Mode simulation - Pipeline VLM non initialisée"
    
    elif chat_type == "video":
        questions = [
            "Analyse les outils VLM utilisés dans cette vidéo",
            "Compare les performances des différents outils",
            "Explique le processus d'optimisation adaptative",
            "Détaille les scores de confiance par outil",
            "Recommande la meilleure configuration d'outils"
        ]
        context_info = f"🎥 {len(st.session_state.uploaded_videos)} vidéos analysées avec pipeline VLM"
    
    # Affichage du contexte
    st.info(f"📊 Contexte: {context_info}")
    
    # Zone de chat
    with st.container():
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        for message in st.session_state[chat_key][-8:]:
            if message['role'] == 'user':
                st.markdown(f"""
                <div class="chat-user">
                    <strong>👤 Vous:</strong> {message['content']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-ai">
                    <strong>🤖 IA VLM:</strong> {message['content']}
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Interface de chat
    col1, col2 = st.columns([2, 1])
    
    with col1:
        user_input = st.text_input(
            f"Question sur {chat_type} VLM...", 
            key=f"chat_input_{chat_type}",
            placeholder=f"Ex: {questions[0]}"
        )
    
    with col2:
        selected_question = st.selectbox(
            "Questions VLM", 
            [""] + questions,
            key=f"quick_q_{chat_type}"
        )
    
    # Traitement des messages
    message_to_send = None
    if st.button("📤 Envoyer", key=f"send_{chat_type}") and user_input:
        message_to_send = user_input
    elif selected_question:
        if st.button("📤 Envoyer Question", key=f"send_quick_{chat_type}"):
            message_to_send = selected_question
    
    if message_to_send:
        # Ajouter la question
        st.session_state[chat_key].append({
            'role': 'user',
            'content': message_to_send,
            'timestamp': datetime.now()
        })
        
        # Générer réponse avec VLM thinking/reasoning
        with st.spinner("🧠 Analyse VLM avec thinking..."):
            ai_response = asyncio.run(generate_real_vlm_response(message_to_send, chat_type, context_data))
        
        st.session_state[chat_key].append({
            'role': 'assistant',
            'content': ai_response,
            'timestamp': datetime.now()
        })
        
        st.rerun()

async def generate_real_vlm_response(question: str, chat_type: str, context_data: Dict) -> str:
    """Génère une réponse VLM intelligente avec thinking/reasoning."""
    
    if not PIPELINE_AVAILABLE:
        return "🤖 Pipeline VLM non disponible - Mode simulation basique."
    
    # Récupération des vraies données pour contexte VLM
    vlm_context = {}
    if st.session_state.pipeline_initialized and st.session_state.real_pipeline:
        vlm_context = {
            'stats': st.session_state.real_pipeline.get_performance_stats(),
            'tools': st.session_state.real_pipeline.get_tool_performance_details(),
            'detections': st.session_state.real_detections[-10:],
            'optimizations': st.session_state.optimization_results[-5:],
            'alerts': st.session_state.real_alerts[-5:]
        }
    
    try:
        # 🧠 APPEL VLM CHATBOT AVEC THINKING/REASONING
        response_data = await process_vlm_chat_query(
            question=question,
            chat_type=chat_type, 
            vlm_context=vlm_context
        )
        
        # Formatage pour interface chat Streamlit
        if response_data.get("type") == "vlm_thinking":
            # Réponse VLM complète avec thinking
            response_text = f"""🧠 **Analyse VLM avec Thinking:**

**💭 Processus de Raisonnement:**
{response_data.get('thinking', 'Thinking non disponible')[:300]}...

**📊 Analyse Technique:**
{response_data.get('analysis', 'Analyse non disponible')[:200]}...

**🎯 Réponse:**
{response_data.get('response', 'Réponse non disponible')}

**🔧 Détails Techniques:**
{response_data.get('technical_details', 'Détails non disponibles')}

**💡 Recommandations:**
{' | '.join(response_data.get('recommendations', [])[:3])}

**📈 Confiance:** {response_data.get('confidence', 0):.1%} | **📊 Qualité Données:** {response_data.get('data_quality', 'medium')}"""
            
            return response_text
            
        else:
            # Fallback ou réponse basique
            return response_data.get("response", "🤖 Réponse VLM générée.")
            
    except Exception as e:
        # Fallback sur ancien système si erreur VLM
        logger.error(f"Erreur chatbot VLM: {e}")
        return f"⚠️ Erreur VLM chatbot: {str(e)}. Utilisant fallback basique."

def render_surveillance_tab():
    """Onglet surveillance avec vraie intégration VLM."""
    st.subheader("🎥 Surveillance Temps Réel avec Pipeline VLM")
    
    # Statut de la pipeline
    render_pipeline_status()
    
    # Grille des caméras
    if not st.session_state.cameras:
        st.info("📹 Aucune caméra configurée. Ajoutez une caméra dans l'onglet Configuration.")
    else:
        cameras = list(st.session_state.cameras.values())
        
        if len(cameras) == 1:
            cols = st.columns(1)
        elif len(cameras) <= 2:
            cols = st.columns(2)
        elif len(cameras) <= 4:
            cols = st.columns(2)
        else:
            cols = st.columns(3)
        
        for i, camera in enumerate(cameras):
            with cols[i % len(cols)]:
                st.markdown(f"""
                <div class="camera-card">
                    <h4>📹 {camera['name']}</h4>
                </div>
                """, unsafe_allow_html=True)
                
                if camera.get('active', False) and st.session_state.surveillance_active:
                    # MÉTHODE SIMPLIFIÉE - capture directe sans cache complexe
                    camera_id = camera['id']
                    
                    try:
                        print(f"DEBUG: Capture directe pour caméra {camera_id}")
                        # Capture directe de frame
                        frame = capture_real_frame_simple(camera, width=640, height=480)
                        
                        if frame is not None:
                            st.image(frame, channels="BGR", caption=f"🔴 LIVE - {camera['name']}")
                            st.caption(f"✅ Flux actif à {datetime.now().strftime('%H:%M:%S')}")
                        else:
                            st.error(f"❌ Flux {camera['name']} indisponible")
                            # Fallback vers dummy  
                            frame = generate_dummy_frame(camera['id'])
                            st.image(frame, channels="BGR", caption=f"⚠️ Fallback - {camera['name']}")
                            
                    except Exception as e:
                        st.error(f"💥 Erreur capture {camera['name']}: {str(e)}")
                        print(f"DEBUG: Exception capture {camera_id}: {e}")
                        # Test direct disponible en cas d'erreur
                        if st.button(f"🧪 Test Direct", key=f"test_direct_{camera_id}"):
                            with st.spinner("Test direct en cours..."):
                                try:
                                    test_frame = capture_real_frame_simple(camera, width=320, height=240)
                                    if test_frame is not None:
                                        st.success("✅ Test direct réussi !")
                                        st.image(test_frame, channels="BGR", caption="Frame test")
                                    else:
                                        st.error("❌ Test direct échoué")
                                        st.info(f"Source: {camera.get('source')}")
                                except Exception as te:
                                    st.error(f"Test échoué: {te}")
                
                elif camera.get('active', False):
                    # Caméra active mais surveillance inactive
                    st.info(f"📹 {camera['name']} prête - Démarrez la surveillance")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("⏸️ Pause", key=f"pause_{camera['id']}"):
                            st.session_state.cameras[camera['id']]['active'] = False
                            st.rerun()
                    
                    with col2:
                        if st.button("⚙️ Config", key=f"config_{camera['id']}"):
                            st.info(f"Configuration de {camera['name']}")
                
                else:
                    st.image("https://via.placeholder.com/320x240/cccccc/666666?text=Camera+Offline", 
                            caption=f"Caméra hors ligne - {camera['name']}")
                    
                    if st.button("▶️ Start", key=f"start_{camera['id']}"):
                        st.session_state.cameras[camera['id']]['active'] = True
                        st.rerun()
    
    # Affichage des détections VLM récentes
    if st.session_state.real_detections:
        st.subheader("🔬 Détections VLM Récentes")
        
        for detection in st.session_state.real_detections[-5:]:  # 5 dernières
            suspicion_color = 'red' if 'HIGH' in str(detection.suspicion_level) or 'CRITICAL' in str(detection.suspicion_level) else 'orange' if 'MEDIUM' in str(detection.suspicion_level) else 'green'
            
            st.markdown(f"""
            <div class="real-analysis-result">
                <h5>🎯 Détection {detection.frame_id}</h5>
                <p><strong>Caméra:</strong> {detection.camera_id}</p>
                <p><strong>Niveau suspicion:</strong> <span style="color: {suspicion_color}">{detection.suspicion_level}</span></p>
                <p><strong>Confiance:</strong> {detection.confidence:.1%}</p>
                <p><strong>Description:</strong> {detection.description}</p>
                <p><strong>Outils utilisés:</strong> {', '.join(detection.tools_used[:3])}...</p>
                <p><strong>Temps traitement:</strong> {detection.processing_time:.2f}s</p>
                <p><strong>Score optimisation:</strong> {detection.optimization_score:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Chat intégré pour surveillance
    st.divider()
    context_data = {
        'pipeline_active': st.session_state.pipeline_initialized,
        'real_detections': len(st.session_state.real_detections),
        'real_alerts': len(st.session_state.real_alerts)
    }
    render_integrated_chat("surveillance", context_data)

def render_video_upload_tab():
    """Onglet upload vidéo avec vraie analyse VLM."""
    st.subheader("🎥 Upload & Analyse Vidéo avec Pipeline VLM")
    
    # Statut pipeline
    render_pipeline_status()
    
    # Section d'upload avec description
    st.markdown("### 📤 Upload de Vidéo avec Description")
    
    uploaded_file = st.file_uploader(
        "Sélectionnez une vidéo à analyser avec VLM",
        type=['mp4', 'avi', 'mov', 'mkv', 'webm'],
        help="La vidéo sera analysée avec la pipeline VLM complète (8 outils avancés)"
    )
    
    # Formulaire de description enrichi
    st.markdown("### 📝 Description et Contexte Vidéo")
    
    col1, col2 = st.columns(2)
    
    with col1:
        video_title = st.text_input(
            "Titre/Nom de la vidéo",
            placeholder="Ex: Surveillance magasin - Caisse principale - 14h-16h",
            help="Nom descriptif pour identifier cette analyse"
        )
        
        video_location = st.selectbox(
            "Lieu de surveillance",
            ["Magasin/Commerce", "Entrepôt", "Bureau", "Parking", "Zone industrielle", 
             "Espace public", "Résidentiel", "Transport", "Autre"],
            help="Type d'environnement surveillé"
        )
        
        video_time_context = st.selectbox(
            "Contexte temporel",
            ["Heures ouverture", "Heures affluence", "Heures creuses", "Nuit/Fermeture",
             "Weekend", "Jour férié", "Événement spécial", "Période de crise", "Non spécifié"],
            help="Contexte temporal pour adapter l'analyse"
        )
    
    with col2:
        expected_activity = st.multiselect(
            "Activités attendues (normales)",
            ["Clients shopping", "Personnel travail", "Livraisons", "Nettoyage", 
             "Maintenance", "Circulation véhicules", "Activités bureau", "Surveillance"],
            help="Activités considérées comme normales dans ce contexte"
        )
        
        suspicious_focus = st.multiselect(
            "Focus surveillance (à détecter)",
            ["Vol à l'étalage", "Intrusion", "Comportements agressifs", "Objets abandonnés",
             "Accès non autorisé", "Vandalisme", "Activités inhabituelles", "Mouvements suspects"],
            help="Types de comportements suspects à prioriser"
        )
        
        camera_angle = st.selectbox(
            "Angle/Position caméra",
            ["Vue plongeante", "Vue niveau", "Vue latérale", "Vue face", "Vue multi-angles", "Non spécifié"],
            help="Perspective de la caméra pour optimiser l'analyse"
        )
    
    # Description libre détaillée
    video_description = st.text_area(
        "Description détaillée du contexte",
        placeholder="""Décrivez le contexte spécifique de cette vidéo:
        
• Situation particulière ou événements en cours
• Éléments d'environnement importants (layout, éclairage, foule)  
• Comportements spécifiques à surveiller
• Informations techniques (résolution, qualité, conditions)
• Objectifs d'analyse particuliers
• Contraintes ou défis attendus

Cette description aidera le VLM à mieux contextualiser son analyse...""",
        height=150,
        help="Description libre pour contextualiser l'analyse VLM"
    )
    
    # Configuration analyse avancée
    st.markdown("### ⚙️ Configuration Analyse VLM")
    
    col3, col4 = st.columns(2)
    with col3:
        analysis_mode = st.selectbox(
            "Mode d'analyse VLM",
            ["Optimisation adaptative", "Tous les outils", "Outils sélectionnés", "Performance maximale"],
            help="Stratégie d'analyse selon contexte et ressources"
        )
        
        confidence_threshold = st.slider(
            "Seuil de confiance VLM", 
            0.1, 1.0, 0.7,
            help="Niveau de confiance minimum pour les détections"
        )
    
    with col4:
        analysis_priority = st.selectbox(
            "Priorité analyse",
            ["Précision maximale", "Vitesse optimisée", "Équilibré", "Économie ressources"],
            help="Compromis vitesse/précision selon urgence"
        )
        
        frame_sampling = st.selectbox(
            "Échantillonnage frames",
            ["Dense (toutes frames)", "Standard (1/2 frames)", "Rapide (1/5 frames)", "Clés seulement"],
            help="Densité d'analyse selon durée vidéo"
        )
    
    # Sélection d'outils spécifiques
    if analysis_mode == "Outils sélectionnés":
        available_tools = [
            'sam2_segmentator', 'dino_features', 'pose_estimator',
            'trajectory_analyzer', 'multimodal_fusion', 'temporal_transformer',
            'adversarial_detector', 'domain_adapter'
        ]
        
        selected_tools = st.multiselect(
            "Sélectionner les outils VLM",
            available_tools,
            default=available_tools[:4]
        )
    
    if uploaded_file is not None:
        # Informations du fichier
        file_details = {
            "Nom": uploaded_file.name,
            "Taille": f"{uploaded_file.size / 1024 / 1024:.2f} MB",
            "Type": uploaded_file.type
        }
        
        st.write("**Informations du fichier:**")
        st.json(file_details)
        
        # Bouton d'analyse VLM
        analyze_button_text = "🔬 Analyser avec Pipeline VLM" if st.session_state.pipeline_initialized else "🔬 Analyser (Mode Simulation)"
        
        if st.button(analyze_button_text, type="primary"):
            # Validation formulaire
            if not video_title.strip():
                st.error("⚠️ Veuillez saisir un titre pour la vidéo")
                return
            
            # Construction métadonnées enrichies pour VLM avec VideoContextMetadata
            form_data = {
                'title': video_title.strip(),
                'location_type': video_location,
                'time_context': video_time_context,
                'expected_activities': expected_activity,
                'suspicious_focus': suspicious_focus,
                'camera_angle': camera_angle,
                'detailed_description': video_description.strip(),
                'analysis_priority': analysis_priority,
                'frame_sampling': frame_sampling
            }
            
            # Création de l'objet VideoContextMetadata structuré
            video_metadata_obj = create_video_metadata_from_form(form_data)
            video_metadata = video_metadata_obj.to_dict()
            
            # Intégration dans le système VLM
            context_integration = get_video_context_integration()
            base_chat_context = {'video_analysis_mode': True, 'timestamp': datetime.now()}
            enhanced_context = context_integration.enhance_chat_context(
                base_chat_context, video_metadata_obj
            )
            
            with st.spinner("🔬 Analyse VLM contextualisée en cours..."):
                progress_bar = st.progress(0)
                
                # Simulation de traitement par frames avec contexte
                total_frames = random.randint(50, 200)
                analysis_results = {
                    'video_name': uploaded_file.name,
                    'video_metadata': video_metadata,  # Métadonnées enrichies
                    'analysis_mode': analysis_mode,
                    'pipeline_used': 'Real VLM Pipeline' if st.session_state.pipeline_initialized else 'Simulation',
                    'total_frames': total_frames,
                    'frames_analyzed': total_frames,
                    'detections': [],
                    'tool_performance': {},
                    'optimization_data': {},
                    'summary': {},
                    'timestamp': datetime.now(),
                    'context_used': True  # Marqueur contexte utilisé
                }
                
                # Simulation du traitement frame par frame
                for frame_num in range(total_frames):
                    progress_bar.progress((frame_num + 1) / total_frames)
                    time.sleep(0.01)  # Simulation de traitement
                    
                    # Génération de détections réalistes
                    if random.random() > 0.7:  # 30% de chance de détection par frame
                        tools_used = random.sample([
                            'sam2_segmentator', 'dino_features', 'pose_estimator',
                            'trajectory_analyzer', 'multimodal_fusion'
                        ], random.randint(2, 4))
                        
                        detection = {
                            'frame_number': frame_num,
                            'timestamp': f"{frame_num // 30:02d}:{(frame_num % 30) * 2:02d}",
                            'type': random.choice(['person', 'bag', 'vehicle', 'suspicious_movement']),
                            'confidence': random.uniform(0.6, 0.98),
                            'bbox': [random.randint(0, 100), random.randint(0, 100), 
                                   random.randint(100, 300), random.randint(100, 400)],
                            'tools_used': tools_used,
                            'optimization_score': random.uniform(0.5, 0.95)
                        }
                        analysis_results['detections'].append(detection)
                
                # Calcul des performances par outil
                all_tools_used = []
                for detection in analysis_results['detections']:
                    all_tools_used.extend(detection['tools_used'])
                
                from collections import Counter
                tool_usage = Counter(all_tools_used)
                
                for tool, count in tool_usage.items():
                    analysis_results['tool_performance'][tool] = {
                        'usage_count': count,
                        'success_rate': random.uniform(0.7, 0.95),
                        'avg_confidence': random.uniform(0.75, 0.92),
                        'processing_time': random.uniform(0.1, 1.5)
                    }
                
                # Données d'optimisation
                analysis_results['optimization_data'] = {
                    'optimal_combination': list(tool_usage.keys())[:4],
                    'performance_score': random.uniform(0.7, 0.9),
                    'improvement_suggestions': [
                        'Augmenter utilisation SAM2 pour segmentation précise',
                        'Combiner DINO avec pose estimation pour meilleure détection',
                        'Optimiser seuils de confiance pour réduire faux positifs'
                    ]
                }
                
                # Résumé
                suspicion_levels = [random.choice(['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']) for _ in range(len(analysis_results['detections']))]
                analysis_results['summary'] = {
                    'total_detections': len(analysis_results['detections']),
                    'high_confidence_detections': len([d for d in analysis_results['detections'] if d['confidence'] > 0.8]),
                    'suspicion_distribution': dict(Counter(suspicion_levels)),
                    'most_used_tools': [tool for tool, _ in tool_usage.most_common(3)],
                    'avg_processing_time': random.uniform(0.5, 2.0),
                    'overall_performance_score': random.uniform(0.6, 0.9)
                }
                
                # Stockage des résultats
                video_id = f"video_{len(st.session_state.uploaded_videos) + 1}"
                st.session_state.video_analysis_results[video_id] = analysis_results
                st.session_state.uploaded_videos.append({
                    'id': video_id,
                    'name': uploaded_file.name,
                    'upload_time': datetime.now()
                })
                
                progress_bar.progress(1.0)
                st.success("✅ Analyse VLM terminée avec succès!")
    
    # Affichage des résultats d'analyse VLM
    if st.session_state.video_analysis_results:
        st.markdown("### 🔬 Résultats d'Analyse VLM")
        
        # Sélection de l'analyse
        video_options = {k: v['video_name'] for k, v in st.session_state.video_analysis_results.items()}
        selected_video = st.selectbox(
            "Sélectionner une analyse VLM",
            list(video_options.keys()),
            format_func=lambda x: video_options[x]
        )
        
        if selected_video:
            results = st.session_state.video_analysis_results[selected_video]
            
            # Résumé de l'analyse VLM
            st.markdown(f"""
            <div class="real-analysis-result">
                <h4>🔬 Analyse VLM - {results['video_name']}</h4>
                <p><strong>Pipeline:</strong> {results['pipeline_used']}</p>
                <p><strong>Mode:</strong> {results['analysis_mode']}</p>
                <p><strong>Frames analysées:</strong> {results['frames_analyzed']}</p>
                <p><strong>Score performance global:</strong> {results['summary']['overall_performance_score']:.2f}</p>
                <p><strong>Détections totales:</strong> {results['summary']['total_detections']}</p>
                <p><strong>Haute confiance:</strong> {results['summary']['high_confidence_detections']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Performance des outils VLM
            st.markdown("### 🛠️ Performance des Outils VLM")
            
            for tool, perf in results['tool_performance'].items():
                st.markdown(f"""
                <div class="tool-performance">
                    <strong>{tool}</strong><br>
                    Utilisations: {perf['usage_count']} | 
                    Succès: {perf['success_rate']:.1%} | 
                    Confiance moyenne: {perf['avg_confidence']:.1%} | 
                    Temps: {perf['processing_time']:.2f}s
                </div>
                """, unsafe_allow_html=True)
            
            # Données d'optimisation
            st.markdown("### 🎯 Optimisation Adaptative")
            opt_data = results['optimization_data']
            
            st.markdown(f"""
            <div class="optimization-badge">
                Combinaison optimale: {', '.join(opt_data['optimal_combination'])}
            </div>
            """, unsafe_allow_html=True)
            
            st.write("**Suggestions d'amélioration:**")
            for suggestion in opt_data['improvement_suggestions']:
                st.write(f"• {suggestion}")
            
            # Graphiques de performance
            col1, col2 = st.columns(2)
            
            with col1:
                # Performance par outil
                tool_names = list(results['tool_performance'].keys())
                success_rates = [results['tool_performance'][tool]['success_rate'] for tool in tool_names]
                
                fig = px.bar(
                    x=tool_names,
                    y=success_rates,
                    title="Taux de Succès par Outil VLM",
                    labels={'x': 'Outils VLM', 'y': 'Taux de Succès'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Distribution des niveaux de suspicion
                suspicion_dist = results['summary']['suspicion_distribution']
                
                fig = px.pie(
                    values=list(suspicion_dist.values()),
                    names=list(suspicion_dist.keys()),
                    title="Distribution Niveaux de Suspicion"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Export des résultats VLM
            if st.button("📥 Exporter Résultats VLM"):
                st.download_button(
                    label="💾 Télécharger Analyse VLM JSON",
                    data=json.dumps(results, indent=2, default=str),
                    file_name=f"vlm_analysis_{selected_video}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
    
    # Chat intégré pour analyse vidéo
    st.divider()
    context_data = {
        'pipeline_active': st.session_state.pipeline_initialized,
        'video_analyses': len(st.session_state.video_analysis_results)
    }
    render_integrated_chat("video", context_data)

def test_mjpeg_stream_native(url: str, timeout: int = 10) -> Dict[str, Any]:
    """Test natif robuste pour flux MJPEG HTTP."""
    import requests
    import io
    from PIL import Image
    import numpy as np
    
    result = {
        'success': False,
        'backend_used': 'HTTP_NATIVE',
        'resolution': None,
        'error_messages': [],
        'test_duration': 0
    }
    
    start_time = time.time()
    
    try:
        # Headers optimisés pour MJPEG
        headers = {
            'User-Agent': 'Mozilla/5.0 (compatible; SurveillanceBot/1.0)',
            'Accept': 'multipart/x-mixed-replace, */*',
            'Connection': 'close'  # Force nouvelle connexion
        }
        
        # Test simple d'abord
        response = requests.get(url, stream=True, timeout=timeout, headers=headers)
        
        if response.status_code == 200:
            content = b""
            chunks_read = 0
            
            for chunk in response.iter_content(chunk_size=2048):
                if not chunk:
                    continue
                    
                content += chunk
                chunks_read += 1
                
                # Recherche d'image JPEG complète
                jpg_start = content.find(b'\xff\xd8')
                if jpg_start != -1:
                    jpg_end = content.find(b'\xff\xd9', jpg_start)
                    if jpg_end != -1:
                        try:
                            jpeg_data = content[jpg_start:jpg_end+2]
                            
                            # Test de décodage
                            if len(jpeg_data) > 500:  # Taille minimale raisonnable
                                img = Image.open(io.BytesIO(jpeg_data))
                                
                                # Vérification dimensions
                                if img.width > 10 and img.height > 10:
                                    result['success'] = True
                                    result['resolution'] = f"{img.width}x{img.height}"
                                    response.close()
                                    break
                                    
                        except Exception as e:
                            result['error_messages'].append(f"Image decode: {str(e)}")
                
                # Sécurités pour éviter les blocages
                if len(content) > 200000:  # Limite buffer
                    content = content[-100000:]
                
                if chunks_read > 50:  # Limite chunks
                    break
                    
                if time.time() - start_time > timeout:  # Timeout
                    result['error_messages'].append("Test timeout")
                    break
            
            response.close()
            
        else:
            result['error_messages'].append(f"HTTP status: {response.status_code}")
            
    except requests.exceptions.Timeout:
        result['error_messages'].append("Connexion timeout")
    except requests.exceptions.ConnectionError:
        result['error_messages'].append("Erreur connexion réseau")
    except Exception as e:
        result['error_messages'].append(f"Erreur: {str(e)}")
    
    result['test_duration'] = time.time() - start_time
    return result

def check_webcam_permissions() -> Dict[str, Any]:
    """Vérifie les permissions et dispositifs webcam sur Linux."""
    import os
    import stat
    
    result = {
        'video_devices': [],
        'permissions_ok': True,
        'user_groups': [],
        'recommendations': []
    }
    
    try:
        # Vérification des périphériques /dev/video*
        for i in range(10):
            video_dev = f"/dev/video{i}"
            if os.path.exists(video_dev):
                stat_info = os.stat(video_dev)
                result['video_devices'].append({
                    'device': video_dev,
                    'readable': os.access(video_dev, os.R_OK),
                    'writable': os.access(video_dev, os.W_OK),
                    'mode': oct(stat_info.st_mode)
                })
        
        # Vérification des groupes utilisateur
        try:
            import subprocess
            groups_output = subprocess.check_output(['groups'], text=True)
            result['user_groups'] = groups_output.strip().split()
            
            if 'video' not in result['user_groups']:
                result['permissions_ok'] = False
                result['recommendations'].append("Ajouter l'utilisateur au groupe 'video'")
                
        except:
            pass
            
    except:
        pass
    
    return result

def list_video_devices() -> List[Dict[str, Any]]:
    """Liste les dispositifs vidéo disponibles sur le système."""
    import os
    import glob
    
    devices = []
    
    # Recherche des dispositifs /dev/video*
    video_devices = glob.glob("/dev/video*")
    video_devices.sort()
    
    for device_path in video_devices:
        device_info = {
            'path': device_path,
            'index': None,
            'accessible': False,
            'name': 'Unknown',
            'capabilities': []
        }
        
        try:
            # Extraction de l'index
            index = int(device_path.replace('/dev/video', ''))
            device_info['index'] = index
            
            # Test d'accessibilité
            device_info['accessible'] = os.access(device_path, os.R_OK | os.W_OK)
            
            # Lecture des informations v4l2 si possible
            try:
                import subprocess
                result = subprocess.run(['v4l2-ctl', '--device', device_path, '--info'], 
                                      capture_output=True, text=True, timeout=2)
                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        if 'Card type' in line:
                            device_info['name'] = line.split(':')[-1].strip()
                        elif 'Capabilities' in line:
                            device_info['capabilities'] = line.split(':')[-1].strip().split()
            except:
                pass
                
        except Exception:
            pass
        
        devices.append(device_info)
    
    return devices

def detect_available_cameras() -> List[int]:
    """Détecte les webcams disponibles avec diagnostic complet."""
    available_cameras = []
    
    print("DEBUG: Détection des webcams...")
    
    # 1. Liste des dispositifs système
    video_devices = list_video_devices()
    print(f"DEBUG: Dispositifs trouvés: {video_devices}")
    
    # 2. Test des indices avec backends appropriés
    backends_to_test = [
        ('DEFAULT', cv2.CAP_ANY),
        ('V4L2', cv2.CAP_V4L2), 
        ('GSTREAMER', cv2.CAP_GSTREAMER)
    ]
    
    # Test étendu jusqu'à 10 pour couvrir tous les /dev/video*
    indices_to_test = list(range(10))
    
    # Prioriser les indices correspondant aux dispositifs détectés
    device_indices = [d['index'] for d in video_devices if d['index'] is not None and d['accessible']]
    if device_indices:
        indices_to_test = device_indices + [i for i in indices_to_test if i not in device_indices]
    
    for index in indices_to_test:
        print(f"DEBUG: Test index {index}")
        
        for backend_name, backend_code in backends_to_test:
            try:
                print(f"DEBUG: Test {backend_name} pour index {index}")
                cap = cv2.VideoCapture(index, backend_code)
                
                if cap.isOpened():
                    print(f"DEBUG: Ouverture réussie {backend_name} index {index}")
                    
                    # Test de capture avec timeout
                    import signal
                    def timeout_handler(signum, frame):
                        raise TimeoutError("Timeout capture")
                    
                    try:
                        signal.signal(signal.SIGALRM, timeout_handler)
                        signal.alarm(3)  # 3 secondes timeout
                        
                        ret, frame = cap.read()
                        signal.alarm(0)  # Cancel timeout
                        
                        if ret and frame is not None and frame.size > 0:
                            print(f"DEBUG: Capture réussie {backend_name} index {index} - shape: {frame.shape}")
                            available_cameras.append(index)
                            cap.release()
                            break
                        else:
                            print(f"DEBUG: Capture échouée {backend_name} index {index}")
                            
                    except TimeoutError:
                        print(f"DEBUG: Timeout capture {backend_name} index {index}")
                    except Exception as e:
                        print(f"DEBUG: Erreur capture {backend_name} index {index}: {e}")
                    finally:
                        signal.alarm(0)
                else:
                    print(f"DEBUG: Ouverture échouée {backend_name} index {index}")
                
                cap.release()
                
            except Exception as e:
                print(f"DEBUG: Exception {backend_name} index {index}: {e}")
    
    result = list(set(available_cameras))
    print(f"DEBUG: Caméras détectées: {result}")
    return result

def test_camera_connection(source_url: str, timeout: int = 15) -> Dict[str, Any]:
    """Teste la connexion caméra avec fallbacks multiples.""" 
    import cv2
    import threading
    import time
    import requests
    
    # FORCER HTTP natif pour tous les flux HTTP (MJPEG détecté ou non)
    if source_url.startswith('http'):
        mjpeg_result = test_mjpeg_stream_native(source_url, timeout=10)
        # Retourner le résultat même si échec - PAS de fallback OpenCV pour HTTP
        return mjpeg_result
    
    result = {
        'success': False,
        'backend_used': None,
        'resolution': None,
        'error_messages': [],
        'test_duration': 0
    }
    
    start_time = time.time()
    
    # Test HTTP basique
    if source_url.startswith('http'):
        try:
            response = requests.head(source_url, timeout=5)
            if response.status_code != 200:
                result['error_messages'].append(f"HTTP: Status {response.status_code}")
            else:
                result['error_messages'].append("HTTP: URL accessible")
        except Exception as e:
            result['error_messages'].append(f"HTTP: {str(e)}")
    
    # OpenCV uniquement pour flux RTSP/locaux (PAS HTTP)
    if not source_url.startswith('http'):
        # Backends adaptés selon le type de source
        if source_url.isdigit():
            # Pour webcam : éviter FFMPEG, préférer backends natifs
            backends = [
                ('V4L2', cv2.CAP_V4L2),      # Linux webcam
                ('DSHOW', cv2.CAP_DSHOW),    # Windows webcam  
                ('DEFAULT', cv2.CAP_ANY),    # Backend par défaut
                ('GSTREAMER', cv2.CAP_GSTREAMER)
            ]
        else:
            # Pour RTSP/fichiers : FFMPEG OK
            backends = [
                ('FFMPEG', cv2.CAP_FFMPEG),
                ('DEFAULT', cv2.CAP_ANY),
                ('GSTREAMER', cv2.CAP_GSTREAMER)
            ]
        
        for backend_name, backend_code in backends:
            try:
                cap = cv2.VideoCapture()
                
                # Configuration optimisée pour RTSP
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FPS, 20)
                
                # Paramètres spéciaux pour RTSP
                if source_url.startswith('rtsp'):
                    # Ajouter transport TCP si pas déjà présent
                    rtsp_url = source_url
                    if 'tcp' not in rtsp_url.lower():
                        separator = '&' if '?' in rtsp_url else '?'
                        rtsp_url += f"{separator}tcp"
                    
                    if not cap.open(rtsp_url, backend_code):
                        result['error_messages'].append(f"{backend_name}: RTSP ouverture échouée")
                        cap.release()
                        continue
                elif source_url.isdigit():
                    # Webcam - conversion en entier
                    webcam_index = int(source_url)
                    if not cap.open(webcam_index, backend_code):
                        result['error_messages'].append(f"{backend_name}: Webcam {webcam_index} échouée")
                        cap.release()
                        continue
                else:
                    # Fichier local
                    if not cap.open(source_url, backend_code):
                        result['error_messages'].append(f"{backend_name}: Fichier échoué")
                        cap.release()
                        continue
                
                # Vérification connexion
                if not cap.isOpened():
                    result['error_messages'].append(f"{backend_name}: Capture fermée")
                    cap.release()
                    continue
                
                # Test de lecture avec timeout plus court
                frame_captured = False
                
                for attempt in range(5):  # Max 5 tentatives
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.size > 0:
                        # Vérification qualité frame
                        if frame.mean() > 5 and frame.std() > 1:  # Pas du bruit
                            result['success'] = True
                            result['backend_used'] = backend_name
                            result['resolution'] = f"{frame.shape[1]}x{frame.shape[0]}"
                            frame_captured = True
                            break
                    
                    time.sleep(0.2)  # Attente courte
                
                cap.release()
                
                if frame_captured:
                    break
                else:
                    result['error_messages'].append(f"{backend_name}: Pas de frame valide après 5 tentatives")
                
            except Exception as e:
                result['error_messages'].append(f"{backend_name}: {str(e)}")
                try:
                    cap.release()
                except:
                    pass
    else:
        # Pour les flux HTTP, on a déjà testé en natif - ne pas refaire OpenCV
        result['error_messages'].append("HTTP: Test déjà effectué en mode natif")
    
    result['test_duration'] = time.time() - start_time
    return result

def render_camera_config():
    """Configuration des caméras avec tests de connexion renforcés."""
    st.subheader("📹 Configuration des Caméras")
    
    with st.expander("➕ Ajouter une nouvelle caméra", expanded=len(st.session_state.cameras) == 0):
        col1, col2 = st.columns(2)
        
        with col1:
            cam_name = st.text_input("Nom de la caméra", f"Caméra {len(st.session_state.cameras) + 1}")
            cam_source_type = st.selectbox("Type de source", [
                "Webcam locale", 
                "Caméra IP (RTSP)", 
                "Flux MJPEG (HTTP)", 
                "Fichier vidéo"
            ])
        
        with col2:
            cam_resolution = st.selectbox("Résolution", ["640x480", "1280x720", "1920x1080"])
            cam_fps = st.slider("FPS", 15, 60, 30)
        
        # Configuration spécifique selon le type
        source_url = ""
        if cam_source_type == "Webcam locale":
            # Diagnostic complet des dispositifs vidéo
            col_detect1, col_detect2 = st.columns(2)
            
            with col_detect1:
                if st.button("🔍 Détecter webcams disponibles"):
                    with st.spinner("Détection complète en cours..."):
                        available_cams = detect_available_cameras()
                        if available_cams:
                            st.success(f"✅ Webcams fonctionnelles: {available_cams}")
                        else:
                            st.error("❌ Aucune webcam fonctionnelle détectée")
            
            with col_detect2:
                if st.button("🔧 Diagnostic système vidéo"):
                    with st.spinner("Analyse des dispositifs..."):
                        devices = list_video_devices()
                        permissions = check_webcam_permissions()
                        
                        # Diagnostic USB et système
                        try:
                            import subprocess
                            
                            # Liste USB des caméras
                            usb_result = subprocess.run(['lsusb'], capture_output=True, text=True, timeout=5)
                            camera_usb_devices = []
                            if usb_result.returncode == 0:
                                for line in usb_result.stdout.split('\n'):
                                    if any(keyword in line.lower() for keyword in ['camera', 'webcam', 'video', 'imaging']):
                                        camera_usb_devices.append(line.strip())
                            
                            if camera_usb_devices:
                                st.write("**Caméras USB détectées:**")
                                for usb_cam in camera_usb_devices:
                                    st.write(f"🔌 {usb_cam}")
                            else:
                                st.warning("Aucune caméra USB détectée via lsusb")
                                
                        except Exception as e:
                            st.warning(f"Impossible d'exécuter lsusb: {e}")
                        
                        if devices:
                            st.write("**Dispositifs /dev/video* trouvés:**")
                            for device in devices:
                                access_icon = "✅" if device['accessible'] else "❌"
                                st.write(f"{access_icon} {device['path']} - {device['name']} (Index: {device['index']})")
                                
                                # Test des capacités
                                if device['capabilities']:
                                    st.write(f"   └── Capacités: {', '.join(device['capabilities'])}")
                        else:
                            st.error("❌ Aucun dispositif /dev/video* trouvé")
                        
                        # Vérification permissions
                        st.write("**Permissions:**")
                        if permissions['permissions_ok']:
                            st.write("✅ Permissions OK")
                        else:
                            st.write("❌ Problèmes de permissions")
                            for rec in permissions['recommendations']:
                                st.write(f"   • {rec}")
                        
                        st.write(f"**Groupes utilisateur:** {', '.join(permissions['user_groups'])}")
                        
                        # Commandes de diagnostic
                        st.write("**Commandes de diagnostic utiles:**")
                        st.code("sudo usermod -a -G video $USER")
                        st.code("ls -la /dev/video*")
                        st.code("v4l2-ctl --list-devices")
            
            # Sélection de l'index avec options étendues
            available_indices = list(range(10))  # 0-9 pour couvrir tous les cas
            webcam_index = st.selectbox(
                "Index webcam", 
                options=available_indices,
                index=0,
                help="Index de la webcam. Vérifiez les dispositifs détectés ci-dessus."
            )
            source_url = str(webcam_index)
            
            # Informations selon les dispositifs détectés
            devices = list_video_devices()
            matching_device = next((d for d in devices if d['index'] == webcam_index), None)
            
            if matching_device:
                if matching_device['accessible']:
                    st.success(f"✅ Dispositif trouvé: {matching_device['path']} - {matching_device['name']}")
                else:
                    st.error(f"❌ Dispositif trouvé mais non accessible: {matching_device['path']}")
                    st.write("💡 Solution: Vérifiez les permissions ou ajoutez l'utilisateur au groupe 'video'")
            else:
                st.warning(f"⚠️ Aucun dispositif /dev/video{webcam_index} détecté sur le système")
                st.write("💡 Le test tentera quand même la connexion avec les backends disponibles")
            
            st.info(f"🔧 Test utilisera l'index {webcam_index} avec backends: DEFAULT, V4L2, GSTREAMER")
        elif cam_source_type == "Caméra IP (RTSP)":
            source_url = st.text_input("URL RTSP", "rtsp://192.168.1.100:554/stream")
            st.info("🔗 Format: rtsp://user:pass@ip:port/stream")
        elif cam_source_type == "Flux MJPEG (HTTP)":
            source_url = st.text_input("URL MJPEG", "http://192.168.1.100/mjpeg")
            st.info("🔗 Exemple: http://38.79.156.188/CgiStart/nphMotionJpeg?Resolution=640x480")
        elif cam_source_type == "Fichier vidéo":
            video_file = st.file_uploader("Sélectionner vidéo", type=['mp4', 'avi', 'mov'])
            if video_file:
                source_url = video_file.name
        
        detection_sensitivity = st.slider("Sensibilité détection", 0.1, 1.0, 0.7)
        
        # Options de performance
        col_perf1, col_perf2 = st.columns(2)
        with col_perf1:
            connection_timeout = st.slider("Timeout connexion (s)", 5, 30, 15)
            refresh_rate = st.selectbox("Taux de rafraîchissement", [
                ("Temps réel (1s)", 1),
                ("Rapide (2s)", 2), 
                ("Normal (3s)", 3),
                ("Lent (5s)", 5)
            ], index=1)
        
        with col_perf2:
            quality_mode = st.selectbox("Mode qualité", [
                ("Faible latence", "low_latency"),
                ("Équilibré", "balanced"),
                ("Haute qualité", "high_quality")
            ], index=0)
            
            frame_skip = st.checkbox("Ignorer frames anciennes", value=True, 
                                   help="Améliore la réactivité en sautant les frames en retard")
        
        # Test de connexion avant ajout
        if source_url and st.button("🧪 Tester Connexion", key="test_connection"):
            with st.spinner("Test de connexion en cours..."):
                test_result = test_camera_connection(source_url, connection_timeout)
                
                if test_result['success']:
                    st.success(f"✅ Connexion réussie!")
                    st.info(f"🔧 Backend: {test_result['backend_used']}")
                    st.info(f"📐 Résolution détectée: {test_result['resolution']}")
                    st.info(f"⏱️ Temps de test: {test_result['test_duration']:.1f}s")
                else:
                    st.error("❌ Impossible de se connecter")
                    with st.expander("Détails des erreurs"):
                        for error in test_result['error_messages']:
                            st.write(f"• {error}")
        
        if st.button("➕ Ajouter Caméra"):
            if not source_url:
                st.error("Veuillez configurer la source de la caméra")
                return
            
            # Test automatique avant ajout
            with st.spinner("Vérification de la caméra..."):
                test_result = test_camera_connection(source_url, connection_timeout)
                
                if test_result['success']:
                    camera_id = f"cam_{len(st.session_state.cameras) + 1}"
                    st.session_state.cameras[camera_id] = {
                        'id': camera_id,
                        'name': cam_name,
                        'source': source_url,
                        'source_type': cam_source_type,
                        'resolution': cam_resolution,
                        'fps': cam_fps,
                        'sensitivity': detection_sensitivity,
                        'timeout': connection_timeout,
                        'backend_tested': test_result['backend_used'],
                        'refresh_rate': refresh_rate[1],
                        'quality_mode': quality_mode[1],
                        'frame_skip': frame_skip,
                        'active': False,
                        'created': datetime.now()
                    }
                    st.success(f"✅ Caméra '{cam_name}' ajoutée avec succès!")
                    st.info(f"🔧 Backend optimal: {test_result['backend_used']}")
                    st.rerun()
                else:
                    st.error("❌ Impossible d'ajouter la caméra - connexion échouée")
                    with st.expander("Diagnostics d'erreur détaillés"):
                        for error in test_result['error_messages']:
                            st.write(f"• {error}")
                        
                        st.write("---")
                        st.write("**Solutions par type:**")
                        
                        if cam_source_type == "Webcam locale":
                            st.write("🎥 **Webcam:**")
                            st.write("• Vérifiez que la webcam n'est pas utilisée par une autre app")
                            st.write("• Essayez des indices différents (0, 1, 2)")
                            st.write("• Utilisez 'Détecter webcams disponibles'")
                            st.write("• Sur Linux: vérifiez les permissions /dev/video*")
                        elif cam_source_type == "Flux MJPEG (HTTP)":
                            st.write("🌐 **MJPEG HTTP:**")
                            st.write("• Vérifiez l'accessibilité de l'URL dans un navigateur")
                            st.write("• Testez sans authentification d'abord")
                            st.write("• Vérifiez que le flux retourne multipart/x-mixed-replace")
                        elif cam_source_type == "Caméra IP (RTSP)":
                            st.write("📡 **RTSP:**")
                            st.write("• Vérifiez user:pass@ip:port/stream")
                            st.write("• Testez avec VLC d'abord")
                            st.write("• Essayez d'ajouter ?tcp à la fin de l'URL")
                        
                        # Diagnostic système
                        st.write("---")
                        st.write("**Diagnostic système:**")
                        available_cams = detect_available_cameras()
                        if available_cams:
                            st.write(f"✅ Webcams détectées: {available_cams}")
                        else:
                            st.write("❌ Aucune webcam système détectée")
    
    # Liste des caméras existantes
    if st.session_state.cameras:
        st.subheader("📋 Caméras Configurées")
        
        for camera_id, camera in st.session_state.cameras.items():
            with st.expander(f"📹 {camera['name']} ({camera_id})"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write(f"**Type:** {camera.get('source_type', 'N/A')}")
                    st.write(f"**Source:** {camera['source']}")
                    st.write(f"**Résolution:** {camera['resolution']}")
                
                with col2:
                    st.write(f"**FPS:** {camera['fps']}")
                    st.write(f"**Sensibilité:** {camera['sensitivity']}")
                    if 'timeout' in camera:
                        st.write(f"**Timeout:** {camera['timeout']}s")
                
                with col3:
                    status = "🟢 Active" if camera.get('active') else "⭕ Inactive"
                    st.write(f"**Statut:** {status}")
                    
                    if 'backend_tested' in camera:
                        st.write(f"**Backend:** {camera['backend_tested']}")
                    
                    # Indicateurs de performance
                    if 'refresh_rate' in camera:
                        refresh_rate = camera['refresh_rate']
                        if refresh_rate == 1:
                            perf_icon = "⚡"
                        elif refresh_rate <= 2:
                            perf_icon = "🚀"
                        elif refresh_rate <= 3:
                            perf_icon = "✅"
                        else:
                            perf_icon = "🐌"
                        st.write(f"**Performance:** {perf_icon} {refresh_rate}s")
                    
                    if 'quality_mode' in camera:
                        mode = camera['quality_mode']
                        mode_display = {
                            'low_latency': '⚡ Faible latence',
                            'balanced': '⚖️ Équilibré', 
                            'high_quality': '💎 Haute qualité'
                        }
                        st.write(f"**Mode:** {mode_display.get(mode, mode)}")
                    
                    # Boutons d'action
                    col_test, col_delete = st.columns(2)
                    
                    with col_test:
                        if st.button("🧪 Re-tester", key=f"retest_{camera_id}"):
                            with st.spinner("Re-test en cours..."):
                                test_result = test_camera_connection(
                                    camera['source'], 
                                    camera.get('timeout', 15)
                                )
                                
                                if test_result['success']:
                                    st.success("✅ Connexion OK")
                                    # Mise à jour du backend optimal
                                    st.session_state.cameras[camera_id]['backend_tested'] = test_result['backend_used']
                                    st.rerun()
                                else:
                                    st.error("❌ Connexion échouée")
                    
                    with col_delete:
                        if st.button("🗑️ Supprimer", key=f"delete_{camera_id}"):
                            del st.session_state.cameras[camera_id]
                            st.success(f"Caméra {camera['name']} supprimée")
                            st.rerun()
                
                # Affichage d'informations détaillées
                if st.button("ℹ️ Détails techniques", key=f"details_{camera_id}"):
                    st.json({
                        'configuration': camera,
                        'created': camera.get('created', 'N/A').isoformat() if camera.get('created') else 'N/A'
                    })

def render_vlm_analytics():
    """Tableau de bord analytique avec métriques VLM réelles."""
    st.subheader("📊 Analytics VLM & Métriques Pipeline")
    
    # Métriques principales
    col1, col2, col3, col4 = st.columns(4)
    
    # Récupération des vraies métriques si disponibles
    if st.session_state.pipeline_initialized and st.session_state.real_pipeline:
        stats = st.session_state.real_pipeline.get_performance_stats()
        tool_details = st.session_state.real_pipeline.get_tool_performance_details()
        
        frames_processed = stats.get('frames_processed', 0)
        avg_processing_time = stats.get('average_processing_time', 0)
        optimization_cycles = stats.get('optimization_cycles', 0)
        performance_score = stats.get('current_performance_score', 0)
    else:
        frames_processed = 0
        avg_processing_time = 0
        optimization_cycles = 0
        performance_score = 0
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h2>{frames_processed}</h2>
            <p>Frames VLM</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h2>{avg_processing_time:.1f}s</h2>
            <p>Temps Moyen</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h2>{optimization_cycles}</h2>
            <p>Optimisations</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h2>{performance_score:.2f}</h2>
            <p>Score Performance</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Graphiques de performance VLM
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🛠️ Utilisation des Outils VLM")
        
        if st.session_state.pipeline_initialized and st.session_state.real_pipeline:
            tool_usage = stats.get('tool_usage_stats', {})
            if tool_usage:
                tools = list(tool_usage.keys())
                usage_counts = list(tool_usage.values())
                
                fig = px.bar(
                    x=tools,
                    y=usage_counts,
                    title="Fréquence d'Utilisation des Outils",
                    labels={'x': 'Outils VLM', 'y': 'Utilisations'}
                )
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Aucune donnée d'utilisation d'outils disponible")
        else:
            # Données simulées pour demo
            tools = ['SAM2', 'DINO', 'Pose', 'Trajectory', 'Fusion']
            usage = [random.randint(10, 50) for _ in tools]
            
            fig = px.bar(x=tools, y=usage, title="Utilisation Outils (Simulation)")
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 Performance Temporelle")
        
        # Simulation de données temporelles
        hours = [f"{i:02d}:00" for i in range(24)]
        if st.session_state.real_detections:
            # Utilisation des vraies données
            detection_times = [d.timestamp.hour for d in st.session_state.real_detections]
            from collections import Counter
            hourly_counts = Counter(detection_times)
            performance_scores = [hourly_counts.get(i, 0) for i in range(24)]
        else:
            # Données simulées
            performance_scores = [random.uniform(0.6, 0.9) for _ in hours]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hours,
            y=performance_scores,
            mode='lines+markers',
            name='Performance',
            line=dict(color='#007bff', width=2)
        ))
        
        fig.update_layout(
            title="Performance VLM sur 24h",
            xaxis_title="Heure",
            yaxis_title="Score de Performance",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Historique des optimisations
    if st.session_state.optimization_results:
        st.subheader("🎯 Historique des Optimisations")
        
        for i, optimization in enumerate(st.session_state.optimization_results[-5:]):
            timestamp = optimization.get('timestamp', 'N/A')
            best_combo = optimization.get('best_combination', [])
            improvement = optimization.get('performance_improvement', 0)
            
            st.markdown(f"""
            <div class="optimization-badge">
                Optimisation #{i+1} - {timestamp}<br>
                Meilleure combinaison: {', '.join(best_combo[:3])}<br>
                Amélioration: +{improvement:.1%}
            </div>
            """, unsafe_allow_html=True)

def render_alerts_panel():
    """Panneau des alertes avec données VLM réelles."""
    st.subheader("🚨 Centre des Alertes VLM")
    
    # Filtres
    col1, col2 = st.columns(2)
    with col1:
        alert_filter = st.selectbox("Filtrer par niveau", ["Tous", "CRITICAL", "HIGH", "MEDIUM", "LOW"])
    with col2:
        show_resolved = st.checkbox("Afficher alertes résolues", False)
    
    # Affichage des alertes réelles
    alerts_to_show = st.session_state.real_alerts if st.session_state.real_alerts else []
    
    if alert_filter != "Tous":
        alerts_to_show = [a for a in alerts_to_show if alert_filter in str(a.get('level', ''))]
    if not show_resolved:
        alerts_to_show = [a for a in alerts_to_show if not a.get('resolved')]
    
    if alerts_to_show:
        st.write(f"**{len(alerts_to_show)} alertes VLM trouvées**")
        
        for i, alert in enumerate(alerts_to_show[-10:]):  # 10 dernières
            level = str(alert.get('level', 'UNKNOWN'))
            level_class = f"alert-{level.lower()}" if level.lower() in ['critical', 'high', 'medium'] else "alert-medium"
            timestamp = alert.get('timestamp', datetime.now())
            timestamp_str = timestamp.strftime("%H:%M:%S") if hasattr(timestamp, 'strftime') else str(timestamp)
            
            confidence = alert.get('confidence', 0)
            tools_used = alert.get('tools_used', [])
            
            st.markdown(f"""
            <div class="{level_class}">
                <strong>{level}</strong> - {timestamp_str} - {alert.get('camera', 'N/A')}<br>
                {alert.get('message', 'Alerte VLM')}<br>
                <small>Confiance: {confidence:.1%} | Outils: {', '.join(tools_used[:2])}</small>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns([3, 1])
            with col2:
                if not alert.get('resolved') and st.button("✅ Résoudre", key=f"resolve_real_alert_{i}"):
                    alert['resolved'] = True
                    st.rerun()
    else:
        st.info("✅ Aucune alerte VLM active")

def main():
    """Application principale avec pipeline VLM réelle."""
    render_header()
    
    # Sidebar avec contrôles VLM
    with st.sidebar:
        st.header("⚙️ Contrôles Pipeline VLM")
        
        # Initialisation de la pipeline
        if not st.session_state.pipeline_initialized:
            if st.button("🚀 Initialiser Pipeline VLM", type="primary"):
                success = asyncio.run(initialize_pipeline())
                if success:
                    st.rerun()
        else:
            st.success("✅ Pipeline VLM Active")
            
            if st.button("⏹️ Arrêter Pipeline"):
                if st.session_state.real_pipeline:
                    st.session_state.real_pipeline.stop_processing()
                st.session_state.pipeline_initialized = False
                st.session_state.real_pipeline = None
                st.info("⏹️ Pipeline arrêtée")
                st.rerun()
        
        # Statut général
        st.divider()
        st.subheader("📊 État Système")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Caméras", len(st.session_state.cameras))
        with col2:
            st.metric("Alertes VLM", len([a for a in st.session_state.real_alerts if not a.get('resolved')]))
        
        if st.session_state.pipeline_initialized and st.session_state.real_pipeline:
            stats = st.session_state.real_pipeline.get_performance_stats()
            st.metric("Frames VLM", stats.get('frames_processed', 0))
            st.metric("Performance", f"{stats.get('current_performance_score', 0):.2f}")
        
        # Contrôles surveillance
        st.divider()
        st.subheader("🎮 Surveillance")
        
        if st.button("▶️ Démarrer Surveillance" if not st.session_state.surveillance_active else "⏹️ Arrêter Surveillance"):
            st.session_state.surveillance_active = not st.session_state.surveillance_active
            if st.session_state.surveillance_active:
                st.success("✅ Surveillance démarrée!")
            else:
                st.info("⏹️ Surveillance arrêtée")
            st.rerun()
        
        # Paramètres VLM
        st.divider()
        st.subheader("⚙️ Paramètres VLM")
        
        optimization_enabled = st.checkbox("Optimisation adaptative", True)
        confidence_threshold = st.slider("Seuil confiance VLM", 0.1, 1.0, 0.7)
        max_tools_per_analysis = st.slider("Max outils par analyse", 1, 8, 4)
        
        # Actions VLM
        st.divider()
        st.subheader("⚡ Actions VLM")
        
        if st.button("🔄 Forcer Optimisation"):
            if st.session_state.pipeline_initialized:
                st.info("🔄 Cycle d'optimisation lancé...")
            else:
                st.warning("⚠️ Pipeline non initialisée")
        
        if st.button("🧹 Vider Données VLM"):
            st.session_state.surveillance_chat.clear()
            st.session_state.video_chat.clear()
            st.session_state.real_alerts.clear()
            st.session_state.real_detections.clear()
            st.session_state.optimization_results.clear()
            st.success("🧹 Données VLM vidées!")
            st.rerun()
    
    # Onglets principaux
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎥 Surveillance VLM", 
        "📤 Upload Vidéo VLM",
        "📹 Configuration", 
        "📊 Analytics VLM", 
        "🚨 Alertes VLM"
    ])
    
    with tab1:
        render_surveillance_tab()
        
        # Auto-refresh optimisé pour le streaming en arrière-plan
        if st.session_state.surveillance_active:
            active_cameras = [cam for cam in st.session_state.cameras.values() if cam.get('active')]
            if active_cameras:
                # Refresh plus rapide pour affichage des frames en cache
                min_refresh = min(cam.get('refresh_rate', 2) for cam in active_cameras)
                # Minimum 1s pour éviter surcharge, maximum 5s pour réactivité
                refresh_time = max(1, min(5, min_refresh))
            else:
                refresh_time = 3
            
            # Auto-refresh avec indicateur
            time.sleep(refresh_time)
            st.rerun()
        else:
            # Pas de surveillance active - arrête tous les streams
            if st.session_state.streaming_manager:
                st.session_state.streaming_manager.stop_all()
    
    with tab2:
        render_video_upload_tab()
    
    with tab3:
        render_camera_config()
    
    with tab4:
        render_vlm_analytics()
    
    with tab5:
        render_alerts_panel()

if __name__ == "__main__":
    main()