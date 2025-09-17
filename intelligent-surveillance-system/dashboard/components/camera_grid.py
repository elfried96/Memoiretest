"""Composant grille multi-caméras avancé."""

import streamlit as st
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime
import threading
import queue
import time
from dataclasses import dataclass
from pathlib import Path

from config.settings import get_dashboard_config
from services.session_manager import get_session_manager
from utils.audio_alerts import play_detection_alert

@dataclass
class CameraConfig:
    """Configuration d'une caméra."""
    
    camera_id: str
    name: str
    source: str  # URL RTSP, device index, ou fichier
    enabled: bool = True
    resolution: Tuple[int, int] = (640, 480)
    fps: int = 30
    detection_zones: List[Dict] = None
    sensitivity: float = 0.5
    
    def __post_init__(self):
        if self.detection_zones is None:
            self.detection_zones = []

class CameraStream:
    """Gestionnaire de flux caméra optimisé."""
    
    def __init__(self, config: CameraConfig):
        self.config = config
        self.cap = None
        self.frame_queue = queue.Queue(maxsize=5)
        self.is_running = False
        self.thread = None
        self.last_frame = None
        self.stats = {
            'frames_captured': 0,
            'frames_dropped': 0,
            'fps_actual': 0,
            'last_update': datetime.now()
        }
    
    def start(self) -> bool:
        """Démarre le flux caméra."""
        try:
            # Ouverture du flux avec configuration améliorée
            if self.config.source.isdigit():
                # Webcam locale
                self.cap = cv2.VideoCapture(int(self.config.source))
                # Permissions webcam
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
            else:
                # Flux réseau avec timeout et buffer
                self.cap = cv2.VideoCapture()
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.cap.set(cv2.CAP_PROP_TIMEOUT, 5000)  # 5s timeout
                
                # Essai backends multiples pour MJPEG
                if not self.cap.open(self.config.source, cv2.CAP_FFMPEG):
                    if not self.cap.open(self.config.source, cv2.CAP_GSTREAMER):
                        self.cap.open(self.config.source)
            
            if not self.cap.isOpened():
                st.error(f"❌ Impossible d'ouvrir {self.config.source}")
                return False
            
            # Configuration
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.resolution[1])
            self.cap.set(cv2.CAP_PROP_FPS, self.config.fps)
            
            # Thread de capture
            self.is_running = True
            self.thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.thread.start()
            
            return True
            
        except Exception as e:
            st.error(f"Erreur ouverture caméra {self.config.name}: {e}")
            return False
    
    def stop(self):
        """Arrête le flux caméra."""
        self.is_running = False
        
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)
        
        if self.cap:
            self.cap.release()
            self.cap = None
    
    def _capture_loop(self):
        """Boucle de capture des frames."""
        fps_counter = 0
        fps_start_time = time.time()
        
        while self.is_running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.1)
                    continue
                
                # Redimensionnement si nécessaire
                if frame.shape[:2][::-1] != self.config.resolution:
                    frame = cv2.resize(frame, self.config.resolution)
                
                # Mise à jour queue (non-bloquant)
                try:
                    self.frame_queue.put_nowait(frame)
                    self.stats['frames_captured'] += 1
                except queue.Full:
                    # Supprime l'ancienne frame
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait(frame)
                        self.stats['frames_dropped'] += 1
                    except queue.Empty:
                        pass
                
                # Calcul FPS
                fps_counter += 1
                if time.time() - fps_start_time >= 1.0:
                    self.stats['fps_actual'] = fps_counter
                    fps_counter = 0
                    fps_start_time = time.time()
                    self.stats['last_update'] = datetime.now()
                
            except Exception as e:
                st.error(f"Erreur capture {self.config.name}: {e}")
                time.sleep(1)
    
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Récupère la dernière frame disponible."""
        try:
            # Récupère toutes les frames en attente et garde la dernière
            latest = None
            while not self.frame_queue.empty():
                latest = self.frame_queue.get_nowait()
            
            if latest is not None:
                self.last_frame = latest
                return latest
                
            return self.last_frame
            
        except queue.Empty:
            return self.last_frame
    
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du flux."""
        return self.stats.copy()

class MultiCameraGrid:
    """Grille multi-caméras avancée."""
    
    def __init__(self):
        self.config = get_dashboard_config()
        self.session = get_session_manager()
        self.cameras: Dict[str, CameraStream] = {}
        self.detection_callback: Optional[Callable] = None
        
    def add_camera(self, camera_config: CameraConfig) -> bool:
        """Ajoute une caméra à la grille."""
        if camera_config.camera_id in self.cameras:
            return False
        
        camera_stream = CameraStream(camera_config)
        if camera_stream.start():
            self.cameras[camera_config.camera_id] = camera_stream
            
            # Sauvegarde état
            self.session.set_camera_state(camera_config.camera_id, {
                'name': camera_config.name,
                'source': camera_config.source,
                'enabled': camera_config.enabled,
                'resolution': camera_config.resolution,
                'fps': camera_config.fps,
                'sensitivity': camera_config.sensitivity
            })
            
            return True
        
        return False
    
    def remove_camera(self, camera_id: str) -> bool:
        """Supprime une caméra de la grille."""
        if camera_id in self.cameras:
            self.cameras[camera_id].stop()
            del self.cameras[camera_id]
            
            # Supprime de l'état
            state = self.session.get_all_cameras_state()
            if camera_id in state:
                del state[camera_id]
            
            return True
        
        return False
    
    def render_grid(self, grid_size: Tuple[int, int] = None) -> None:
        """Affiche la grille de caméras."""
        
        if not self.cameras:
            st.info("🎥 Aucune caméra configurée")
            return
        
        # Taille de grille par défaut
        if grid_size is None:
            num_cameras = len(self.cameras)
            if num_cameras <= 4:
                grid_size = (2, 2)
            elif num_cameras <= 9:
                grid_size = (3, 3)
            else:
                grid_size = (4, 4)
        
        # Création de la grille
        cameras_list = list(self.cameras.items())
        rows, cols = grid_size
        
        for row in range(rows):
            columns = st.columns(cols)
            
            for col in range(cols):
                camera_idx = row * cols + col
                
                if camera_idx < len(cameras_list):
                    camera_id, camera_stream = cameras_list[camera_idx]
                    self._render_camera_cell(columns[col], camera_id, camera_stream)
                else:
                    # Cellule vide
                    with columns[col]:
                        st.empty()
    
    def _render_camera_cell(self, container, camera_id: str, camera_stream: CameraStream):
        """Affiche une cellule caméra individuelle."""
        
        with container:
            # En-tête caméra
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.subheader(f"📹 {camera_stream.config.name}")
            
            with col2:
                # Statut
                stats = camera_stream.get_stats()
                if camera_stream.is_running:
                    st.success(f"🟢 {stats['fps_actual']} FPS")
                else:
                    st.error("🔴 Arrêtée")
            
            # Frame vidéo
            frame = camera_stream.get_latest_frame()
            
            if frame is not None:
                # Application des zones de détection
                annotated_frame = self._apply_detection_zones(frame, camera_stream.config)
                
                # Affichage
                st.image(annotated_frame, channels="BGR", use_container_width=True)
                
                # Analyse si callback configuré
                if self.detection_callback:
                    self._run_detection(camera_id, frame)
                
            else:
                st.error("❌ Aucune image")
            
            # Contrôles caméra
            self._render_camera_controls(camera_id, camera_stream.config)
    
    def _apply_detection_zones(self, frame: np.ndarray, config: CameraConfig) -> np.ndarray:
        """Applique les zones de détection sur la frame."""
        
        if not config.detection_zones:
            return frame
        
        annotated = frame.copy()
        
        for zone in config.detection_zones:
            # Dessine la zone
            points = np.array(zone.get('points', []), dtype=np.int32)
            if len(points) > 2:
                cv2.polylines(annotated, [points], True, (0, 255, 0), 2)
                
                # Label de zone
                label = zone.get('name', 'Zone')
                cv2.putText(annotated, label, tuple(points[0]), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return annotated
    
    def _render_camera_controls(self, camera_id: str, config: CameraConfig):
        """Affiche les contrôles d'une caméra."""
        
        with st.expander("⚙️ Paramètres", expanded=False):
            
            # Sensibilité
            new_sensitivity = st.slider(
                "Sensibilité détection",
                0.0, 1.0, config.sensitivity,
                key=f"sensitivity_{camera_id}"
            )
            
            if new_sensitivity != config.sensitivity:
                config.sensitivity = new_sensitivity
                self.session.set_camera_state(camera_id, {
                    'sensitivity': new_sensitivity
                })
            
            # Zones de détection
            if st.button("🎯 Définir zones", key=f"zones_{camera_id}"):
                self._show_zone_editor(camera_id)
            
            # Actions
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("⏸️ Pause", key=f"pause_{camera_id}"):
                    # TODO: Implémenter pause
                    pass
            
            with col2:
                if st.button("❌ Supprimer", key=f"remove_{camera_id}"):
                    self.remove_camera(camera_id)
                    st.rerun()
    
    def _show_zone_editor(self, camera_id: str):
        """Affiche l'éditeur de zones de détection."""
        # TODO: Implémenter éditeur de zones interactif
        st.info("🚧 Éditeur de zones en développement")
    
    def _run_detection(self, camera_id: str, frame: np.ndarray):
        """Lance la détection sur une frame."""
        if self.detection_callback:
            try:
                # Exécution en arrière-plan
                detection_result = self.detection_callback(camera_id, frame)
                
                # Traitement des résultats
                if detection_result and detection_result.get('objects'):
                    for obj in detection_result['objects']:
                        confidence = obj.get('confidence', 0)
                        obj_type = obj.get('type', 'unknown')
                        
                        # Alerte si confiance élevée
                        if confidence > 0.8:
                            play_detection_alert(confidence, obj_type)
                            
                            # Sauvegarde alerte
                            self.session.add_alert(
                                "HIGH",
                                f"Détection {obj_type} sur {self.cameras[camera_id].config.name}",
                                f"camera_{camera_id}"
                            )
                            
            except Exception as e:
                st.error(f"Erreur détection caméra {camera_id}: {e}")
    
    def set_detection_callback(self, callback: Callable[[str, np.ndarray], Dict]):
        """Définit le callback de détection."""
        self.detection_callback = callback
    
    def get_camera_stats(self) -> Dict[str, Dict]:
        """Retourne les stats de toutes les caméras."""
        return {
            camera_id: {
                'config': {
                    'name': stream.config.name,
                    'source': stream.config.source,
                    'enabled': stream.config.enabled
                },
                'stats': stream.get_stats(),
                'running': stream.is_running
            }
            for camera_id, stream in self.cameras.items()
        }
    
    def cleanup(self):
        """Nettoie toutes les ressources."""
        for camera_stream in self.cameras.values():
            camera_stream.stop()
        self.cameras.clear()

# Fonctions utilitaires pour l'interface

def render_camera_configuration_panel():
    """Panneau de configuration des caméras."""
    
    st.subheader("➕ Ajouter une caméra")
    
    with st.form("add_camera"):
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Nom de la caméra", placeholder="Caméra Entrée")
            source = st.text_input("Source", placeholder="0 (webcam) ou rtsp://...")
        
        with col2:
            resolution = st.selectbox(
                "Résolution", 
                ["640x480", "1280x720", "1920x1080"],
                index=0
            )
            fps = st.number_input("FPS", min_value=1, max_value=60, value=30)
        
        if st.form_submit_button("🎥 Ajouter caméra"):
            if name and source:
                # Parse résolution
                width, height = map(int, resolution.split('x'))
                
                # Création config
                config = CameraConfig(
                    camera_id=f"cam_{len(st.session_state.get('cameras', []))}",
                    name=name,
                    source=source,
                    resolution=(width, height),
                    fps=fps
                )
                
                # Ajout à la grille
                if 'camera_grid' not in st.session_state:
                    st.session_state.camera_grid = MultiCameraGrid()
                
                if st.session_state.camera_grid.add_camera(config):
                    st.success(f"✅ Caméra {name} ajoutée")
                    st.rerun()
                else:
                    st.error(f"❌ Impossible d'ajouter la caméra {name}")
            else:
                st.error("Veuillez remplir tous les champs")

# Instance globale
@st.cache_resource
def get_camera_grid() -> MultiCameraGrid:
    """Récupère l'instance de grille caméras."""
    return MultiCameraGrid()

def cleanup_camera_resources():
    """Nettoie les ressources caméras lors de la fermeture."""
    if 'camera_grid' in st.session_state:
        st.session_state.camera_grid.cleanup()