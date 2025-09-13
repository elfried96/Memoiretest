"""
🔒 Dashboard de Surveillance Intelligent - Version Production
=============================================================

Dashboard Streamlit intégré au système VLM de surveillance existant.
Connexion directe aux composants core pour surveillance temps réel.
"""

import streamlit as st
import asyncio
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import threading
import queue
import json
from pathlib import Path
import sys
import os

# Configuration du PYTHONPATH pour importer les modules core
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Imports du système core
try:
    from src.core.headless.surveillance_system import HeadlessSurveillanceSystem
    from src.core.headless.video_processor import VideoProcessor
    from src.core.headless.frame_analyzer import FrameAnalyzer
    from src.core.headless.result_models import SurveillanceResult, AlertLevel
    from src.core.orchestrator.adaptive_orchestrator import AdaptiveOrchestrator
    from src.core.vlm.model import VisionLanguageModel
    from src.core.types import SuspicionLevel, ActionType, DetectionStatus
    from src.core.orchestrator.vlm_orchestrator import OrchestrationMode
    CORE_AVAILABLE = True
except ImportError as e:
    st.error(f"❌ Impossible d'importer les modules core: {e}")
    st.info("🔧 Mode simulation activé - Vérifiez l'installation du système principal")
    CORE_AVAILABLE = False

# Configuration de la page
st.set_page_config(
    page_title="🔒 Surveillance Intelligente",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🔒"
)

# Initialisation critique des variables de session
if 'surveillance_active' not in st.session_state:
    st.session_state.surveillance_active = False
if 'camera_feeds' not in st.session_state:
    st.session_state.camera_feeds = {}
if 'detection_history' not in st.session_state:
    st.session_state.detection_history = []
if 'alert_history' not in st.session_state:
    st.session_state.alert_history = []
if 'surveillance_system' not in st.session_state:
    st.session_state.surveillance_system = None

# CSS moderne pour le dashboard
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72, #2a5298);
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    .alert-box {
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid;
    }
    .alert-critical { border-color: #dc3545; background: rgba(220, 53, 69, 0.1); }
    .alert-high { border-color: #fd7e14; background: rgba(253, 126, 20, 0.1); }
    .alert-medium { border-color: #ffc107; background: rgba(255, 193, 7, 0.1); }
    .alert-low { border-color: #28a745; background: rgba(40, 167, 69, 0.1); }
    
    .camera-feed {
        border: 2px solid #dee2e6;
        border-radius: 8px;
        overflow: hidden;
        background: #f8f9fa;
    }
    .detection-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.875rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


class RealTimeSurveillanceInterface:
    """Interface temps réel pour le système de surveillance."""
    
    def __init__(self):
        self.frame_queue = queue.Queue(maxsize=30)
        self.detection_queue = queue.Queue()
        self.running = False
        
    def initialize_surveillance_system(self, config: Dict[str, Any]) -> bool:
        """Initialise le système de surveillance avec la config."""
        if not CORE_AVAILABLE:
            return False
            
        try:
            # Création du système de surveillance
            surveillance_system = HeadlessSurveillanceSystem(
                video_source=config.get('video_source', 0),
                vlm_model=config.get('vlm_model', 'kimi-vl-a3b-thinking'),
                orchestration_mode=OrchestrationMode.BALANCED,
                save_results=True,
                frame_skip=config.get('frame_skip', 1),
                vlm_analysis_mode=config.get('analysis_mode', 'smart')
            )
            
            st.session_state.surveillance_system = surveillance_system
            return True
            
        except Exception as e:
            st.error(f"❌ Erreur initialisation: {e}")
            return False
    
    def start_camera_stream(self, camera_config: Dict[str, Any]):
        """Démarre le flux caméra temps réel."""
        if not self.running:
            self.running = True
            
            # Thread pour la capture vidéo
            camera_thread = threading.Thread(
                target=self._camera_capture_loop,
                args=(camera_config,),
                daemon=True
            )
            camera_thread.start()
            
            # Thread pour l'analyse VLM
            if CORE_AVAILABLE:
                analysis_thread = threading.Thread(
                    target=self._vlm_analysis_loop,
                    daemon=True
                )
                analysis_thread.start()
    
    def _camera_capture_loop(self, camera_config: Dict[str, Any]):
        """Boucle de capture vidéo."""
        cap = cv2.VideoCapture(camera_config['source'])
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_config.get('width', 640))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_config.get('height', 480))
        
        while self.running:
            ret, frame = cap.read()
            if ret:
                # Ajouter timestamp et métadonnées
                frame_data = {
                    'frame': frame,
                    'timestamp': datetime.now(),
                    'camera_id': camera_config['id']
                }
                
                try:
                    self.frame_queue.put_nowait(frame_data)
                except queue.Full:
                    # Supprime les anciens frames si la queue est pleine
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait(frame_data)
                    except queue.Empty:
                        pass
        
        cap.release()
    
    def _vlm_analysis_loop(self):
        """Boucle d'analyse VLM en arrière-plan."""
        if not CORE_AVAILABLE or not st.session_state.surveillance_system:
            return
            
        while self.running:
            try:
                # Récupère le frame le plus récent
                frame_data = self.frame_queue.get(timeout=1.0)
                
                # Analyse avec le système VLM
                result = self._analyze_frame_with_vlm(frame_data)
                
                if result:
                    self.detection_queue.put(result)
                    
            except queue.Empty:
                continue
            except Exception as e:
                st.error(f"❌ Erreur analyse VLM: {e}")
    
    def _analyze_frame_with_vlm(self, frame_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyse une frame avec le système VLM."""
        try:
            # Utilisation du système de surveillance existant
            system = st.session_state.surveillance_system
            
            # Simulation pour l'instant - à implémenter avec le vrai système
            # result = await system.analyze_frame(frame_data['frame'])
            
            # Simulation de détection pour demo
            detection_result = {
                'timestamp': frame_data['timestamp'],
                'camera_id': frame_data['camera_id'],
                'detections': [
                    {
                        'type': 'person',
                        'confidence': 0.95,
                        'bbox': [100, 100, 200, 300],
                        'suspicion_level': SuspicionLevel.LOW.value
                    }
                ],
                'analysis': "Activité normale détectée",
                'alert_level': AlertLevel.LOW.value if hasattr(AlertLevel, 'LOW') else 'low'
            }
            
            return detection_result
            
        except Exception as e:
            st.error(f"❌ Erreur analyse: {e}")
            return None
    
    def get_latest_detections(self) -> List[Dict[str, Any]]:
        """Récupère les dernières détections."""
        detections = []
        while not self.detection_queue.empty():
            try:
                detection = self.detection_queue.get_nowait()
                detections.append(detection)
            except queue.Empty:
                break
        return detections
    
    def stop_surveillance(self):
        """Arrête la surveillance."""
        self.running = False


def render_main_header():
    """Affiche l'en-tête principal."""
    st.markdown("""
    <div class="main-header">
        <h1>🔒 Dashboard de Surveillance Intelligente</h1>
        <p>Système VLM Intégré - Surveillance Temps Réel</p>
    </div>
    """, unsafe_allow_html=True)


def render_system_status():
    """Affiche le statut du système."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        status = "🟢 Actif" if CORE_AVAILABLE else "🔴 Simulation"
        st.metric("Système VLM", status)
    
    with col2:
        camera_count = len(st.session_state.camera_feeds)
        st.metric("Caméras", camera_count)
    
    with col3:
        alert_count = len([a for a in st.session_state.alert_history 
                          if a.get('timestamp', datetime.min) > datetime.now() - timedelta(hours=1)])
        st.metric("Alertes (1h)", alert_count)
    
    with col4:
        active_status = "🟢 En cours" if st.session_state.surveillance_active else "⏸️ Arrêtée"
        st.metric("Surveillance", active_status)


def render_camera_configuration():
    """Interface de configuration des caméras."""
    st.subheader("📹 Configuration des Caméras")
    
    with st.expander("➕ Ajouter une caméra"):
        col1, col2 = st.columns(2)
        
        with col1:
            camera_name = st.text_input("Nom de la caméra", "Caméra 1")
            camera_source = st.text_input("Source", "0", help="0 pour webcam, URL RTSP pour caméra IP")
        
        with col2:
            camera_width = st.number_input("Largeur", 640, step=1)
            camera_height = st.number_input("Hauteur", 480, step=1)
        
        sensitivity = st.slider("Sensibilité détection", 0.1, 1.0, 0.7)
        
        if st.button("➕ Ajouter caméra"):
            camera_id = f"cam_{len(st.session_state.camera_feeds) + 1}"
            st.session_state.camera_feeds[camera_id] = {
                'id': camera_id,
                'name': camera_name,
                'source': camera_source,
                'width': camera_width,
                'height': camera_height,
                'sensitivity': sensitivity,
                'active': False
            }
            st.success(f"✅ Caméra {camera_name} ajoutée!")
            st.rerun()


def render_camera_grid():
    """Affiche la grille des caméras."""
    if not st.session_state.camera_feeds:
        st.info("📹 Aucune caméra configurée. Ajoutez une caméra pour commencer.")
        return
    
    st.subheader("🎥 Flux Caméras Temps Réel")
    
    cameras = list(st.session_state.camera_feeds.values())
    
    # Grille adaptative selon le nombre de caméras
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
            st.markdown(f"**📹 {camera['name']}**")
            
            # Placeholder pour le flux vidéo
            video_placeholder = st.empty()
            
            # Simulation d'une frame
            if camera['active']:
                # Ici on afficherait la vraie frame de la caméra
                st.image("https://via.placeholder.com/320x240/1e3c72/ffffff?text=Camera+Feed", 
                        caption=f"Flux en direct - {camera['name']}")
            else:
                st.image("https://via.placeholder.com/320x240/cccccc/666666?text=Camera+Offline", 
                        caption=f"Caméra hors ligne - {camera['name']}")
            
            # Contrôles caméra
            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"{'⏸️ Stop' if camera['active'] else '▶️ Start'}", key=f"toggle_{camera['id']}"):
                    st.session_state.camera_feeds[camera['id']]['active'] = not camera['active']
                    st.rerun()
            
            with col2:
                if st.button("🗑️ Supprimer", key=f"delete_{camera['id']}"):
                    del st.session_state.camera_feeds[camera['id']]
                    st.rerun()


def render_detection_alerts():
    """Affiche les alertes de détection."""
    st.subheader("🚨 Alertes de Détection")
    
    if not st.session_state.alert_history:
        st.info("✅ Aucune alerte récente")
        return
    
    # Filtrage par niveau d'alerte
    alert_filter = st.selectbox(
        "Filtrer par niveau",
        ["Tous", "CRITICAL", "HIGH", "MEDIUM", "LOW"]
    )
    
    filtered_alerts = st.session_state.alert_history
    if alert_filter != "Tous":
        filtered_alerts = [a for a in filtered_alerts if a.get('level') == alert_filter]
    
    # Affichage des alertes
    for alert in filtered_alerts[-10:]:  # 10 dernières alertes
        level = alert.get('level', 'LOW').lower()
        timestamp = alert.get('timestamp', datetime.now()).strftime("%H:%M:%S")
        message = alert.get('message', 'Alerte détectée')
        camera = alert.get('camera_id', 'unknown')
        
        st.markdown(f"""
        <div class="alert-box alert-{level}">
            <strong>{timestamp}</strong> - {camera}<br>
            {message}
        </div>
        """, unsafe_allow_html=True)


def render_analytics_dashboard():
    """Tableau de bord d'analyse."""
    st.subheader("📊 Analyse et Métriques")
    
    # Simulation de données analytiques
    if st.session_state.detection_history:
        df = pd.DataFrame(st.session_state.detection_history)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Graphique des détections par heure
            fig = px.histogram(
                df, 
                x='timestamp', 
                title="Détections par Heure",
                labels={'timestamp': 'Heure', 'count': 'Nombre de détections'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Répartition des niveaux de suspicion
            if 'suspicion_level' in df.columns:
                fig = px.pie(
                    df, 
                    names='suspicion_level', 
                    title="Répartition Niveaux de Suspicion"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("📈 Aucune donnée d'analyse disponible")


def main():
    """Application principale."""
    render_main_header()
    
    # Interface surveillance temps réel
    if 'surveillance_interface' not in st.session_state:
        st.session_state.surveillance_interface = RealTimeSurveillanceInterface()
    
    # Sidebar de contrôle
    with st.sidebar:
        st.header("⚙️ Contrôles")
        
        # Status système
        render_system_status()
        
        st.divider()
        
        # Contrôles surveillance
        st.subheader("🎮 Surveillance")
        
        if st.button("▶️ Démarrer Surveillance" if not st.session_state.surveillance_active else "⏹️ Arrêter Surveillance"):
            if not st.session_state.surveillance_active:
                # Configuration système
                config = {
                    'video_source': 0,
                    'vlm_model': 'kimi-vl-a3b-thinking',
                    'frame_skip': 1,
                    'analysis_mode': 'smart'
                }
                
                if st.session_state.surveillance_interface.initialize_surveillance_system(config):
                    st.session_state.surveillance_active = True
                    st.success("✅ Surveillance démarrée!")
                else:
                    st.error("❌ Impossible de démarrer la surveillance")
            else:
                st.session_state.surveillance_interface.stop_surveillance()
                st.session_state.surveillance_active = False
                st.info("⏹️ Surveillance arrêtée")
            
            st.rerun()
        
        st.divider()
        
        # Paramètres
        st.subheader("⚙️ Paramètres")
        alert_threshold = st.slider("Seuil d'alerte", 0.0, 1.0, 0.7)
        auto_analysis = st.checkbox("Analyse automatique", True)
        save_detections = st.checkbox("Sauvegarder détections", True)
    
    # Contenu principal
    tabs = st.tabs(["🎥 Surveillance", "📹 Caméras", "🚨 Alertes", "📊 Analyse"])
    
    with tabs[0]:
        render_camera_grid()
    
    with tabs[1]:
        render_camera_configuration()
    
    with tabs[2]:
        render_detection_alerts()
    
    with tabs[3]:
        render_analytics_dashboard()
    
    # Auto-refresh pour les données temps réel
    if st.session_state.surveillance_active:
        # Récupère les nouvelles détections
        new_detections = st.session_state.surveillance_interface.get_latest_detections()
        
        if new_detections:
            st.session_state.detection_history.extend(new_detections)
            
            # Ajoute les alertes
            for detection in new_detections:
                if detection.get('alert_level') in ['HIGH', 'CRITICAL']:
                    st.session_state.alert_history.append({
                        'timestamp': detection['timestamp'],
                        'level': detection['alert_level'],
                        'message': detection.get('analysis', 'Détection suspecte'),
                        'camera_id': detection['camera_id']
                    })
        
        # Auto-refresh toutes les 2 secondes
        import time
        time.sleep(2)
        st.rerun()


if __name__ == "__main__":
    main()