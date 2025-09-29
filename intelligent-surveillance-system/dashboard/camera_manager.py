"""
 Gestionnaire de Caméras Temps Réel
====================================

Gestion des flux RTSP, webcam et streaming vidéo pour le dashboard.
Intégration avec OpenCV et le système VLM existant.
"""

import cv2
import numpy as np
import threading
import queue
import time
from typing import Dict, List, Optional, Callable, Any, Tuple
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class CameraConfig:
    """Configuration d'une caméra."""
    id: str
    name: str
    source: str  # URL RTSP ou index webcam
    width: int = 640
    height: int = 480
    fps: int = 30
    enabled: bool = True
    detection_zones: List[Tuple[int, int, int, int]] = None  # (x1, y1, x2, y2)
    sensitivity: float = 0.7


@dataclass
class FrameData:
    """Données d'une frame capturée."""
    camera_id: str
    frame: np.ndarray
    timestamp: datetime
    frame_number: int
    metadata: Dict[str, Any] = None


class CameraStream:
    """Gestionnaire de flux pour une caméra individuelle."""
    
    def __init__(self, config: CameraConfig, frame_callback: Optional[Callable] = None):
        self.config = config
        self.frame_callback = frame_callback
        self.cap = None
        self.thread = None
        self.running = False
        self.frame_queue = queue.Queue(maxsize=10)
        self.last_frame = None
        self.frame_count = 0
        self.connection_attempts = 0
        self.max_connection_attempts = 5
        
    def start(self) -> bool:
        """Démarre le flux caméra."""
        if self.running:
            return True
            
        if self._initialize_capture():
            self.running = True
            self.thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.thread.start()
            logger.info(f" Caméra {self.config.id} démarrée")
            return True
        else:
            logger.error(f" Impossible de démarrer la caméra {self.config.id}")
            return False
    
    def stop(self):
        """Arrête le flux caméra."""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()
            self.cap = None
        logger.info(f" Caméra {self.config.id} arrêtée")
    
    def _initialize_capture(self) -> bool:
        """Initialise la capture vidéo."""
        try:
            # Détermination de la source
            if self.config.source.isdigit():
                # Webcam locale
                source = int(self.config.source)
            else:
                # URL RTSP ou fichier
                source = self.config.source
            
            self.cap = cv2.VideoCapture(source)
            
            # Configuration de la capture
            if not self.cap.isOpened():
                raise Exception("Impossible d'ouvrir la source vidéo")
            
            # Paramètres de la caméra
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.config.fps)
            
            # Test de capture
            ret, test_frame = self.cap.read()
            if not ret:
                raise Exception("Impossible de lire depuis la source")
            
            logger.info(f" Caméra {self.config.id} initialisée: {self.config.width}x{self.config.height}@{self.config.fps}fps")
            return True
            
        except Exception as e:
            logger.error(f" Erreur initialisation caméra {self.config.id}: {e}")
            if self.cap:
                self.cap.release()
                self.cap = None
            return False
    
    def _capture_loop(self):
        """Boucle principale de capture."""
        while self.running and self.cap:
            try:
                ret, frame = self.cap.read()
                
                if not ret:
                    logger.warning(f" Perte de signal caméra {self.config.id}")
                    if self._attempt_reconnection():
                        continue
                    else:
                        break
                
                # Préparation des données de frame
                frame_data = FrameData(
                    camera_id=self.config.id,
                    frame=frame.copy(),
                    timestamp=datetime.now(),
                    frame_number=self.frame_count,
                    metadata={
                        'camera_name': self.config.name,
                        'resolution': (frame.shape[1], frame.shape[0]),
                        'detection_zones': self.config.detection_zones
                    }
                )
                
                self.frame_count += 1
                self.last_frame = frame_data
                
                # Ajout à la queue (non-bloquant)
                try:
                    self.frame_queue.put_nowait(frame_data)
                except queue.Full:
                    # Supprime l'ancienne frame si la queue est pleine
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait(frame_data)
                    except queue.Empty:
                        pass
                
                # Callback pour traitement externe
                if self.frame_callback:
                    try:
                        self.frame_callback(frame_data)
                    except Exception as e:
                        logger.error(f" Erreur callback frame: {e}")
                
                # Contrôle du framerate
                time.sleep(1.0 / self.config.fps)
                
            except Exception as e:
                logger.error(f" Erreur capture caméra {self.config.id}: {e}")
                if self._attempt_reconnection():
                    continue
                else:
                    break
    
    def _attempt_reconnection(self) -> bool:
        """Tente de reconnecter la caméra."""
        self.connection_attempts += 1
        
        if self.connection_attempts > self.max_connection_attempts:
            logger.error(f" Nombre max de tentatives de reconnexion dépassé pour {self.config.id}")
            return False
        
        logger.info(f" Tentative de reconnexion {self.connection_attempts}/{self.max_connection_attempts} pour {self.config.id}")
        
        if self.cap:
            self.cap.release()
        
        time.sleep(2.0 * self.connection_attempts)  # Délai croissant
        
        if self._initialize_capture():
            self.connection_attempts = 0
            return True
        
        return False
    
    def get_latest_frame(self) -> Optional[FrameData]:
        """Récupère la dernière frame disponible."""
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None
    
    def get_current_frame(self) -> Optional[FrameData]:
        """Récupère la frame courante (sans la retirer de la queue)."""
        return self.last_frame
    
    def is_connected(self) -> bool:
        """Vérifie si la caméra est connectée."""
        return self.running and self.cap and self.cap.isOpened()


class MultiCameraManager:
    """Gestionnaire multi-caméras pour le dashboard."""
    
    def __init__(self):
        self.cameras: Dict[str, CameraStream] = {}
        self.frame_callbacks: List[Callable] = []
        self.detection_callbacks: List[Callable] = []
        self.running = False
        
    def add_camera(self, config: CameraConfig) -> bool:
        """Ajoute une nouvelle caméra."""
        if config.id in self.cameras:
            logger.warning(f" Caméra {config.id} existe déjà")
            return False
        
        try:
            # Callback pour traitement des frames
            def frame_handler(frame_data: FrameData):
                for callback in self.frame_callbacks:
                    try:
                        callback(frame_data)
                    except Exception as e:
                        logger.error(f" Erreur callback frame: {e}")
            
            camera_stream = CameraStream(config, frame_handler)
            self.cameras[config.id] = camera_stream
            
            if self.running:
                return camera_stream.start()
            
            logger.info(f" Caméra {config.id} ajoutée")
            return True
            
        except Exception as e:
            logger.error(f" Erreur ajout caméra {config.id}: {e}")
            return False
    
    def remove_camera(self, camera_id: str) -> bool:
        """Supprime une caméra."""
        if camera_id not in self.cameras:
            return False
        
        camera = self.cameras[camera_id]
        camera.stop()
        del self.cameras[camera_id]
        
        logger.info(f"[DELETED] Caméra {camera_id} supprimée")
        return True
    
    def start_all_cameras(self) -> Dict[str, bool]:
        """Démarre toutes les caméras."""
        self.running = True
        results = {}
        
        for camera_id, camera in self.cameras.items():
            if camera.config.enabled:
                results[camera_id] = camera.start()
            else:
                results[camera_id] = False
                logger.info(f" Caméra {camera_id} désactivée")
        
        return results
    
    def stop_all_cameras(self):
        """Arrête toutes les caméras."""
        self.running = False
        
        for camera in self.cameras.values():
            camera.stop()
        
        logger.info(" Toutes les caméras arrêtées")
    
    def start_camera(self, camera_id: str) -> bool:
        """Démarre une caméra spécifique."""
        if camera_id not in self.cameras:
            return False
        
        return self.cameras[camera_id].start()
    
    def stop_camera(self, camera_id: str) -> bool:
        """Arrête une caméra spécifique."""
        if camera_id not in self.cameras:
            return False
        
        self.cameras[camera_id].stop()
        return True
    
    def get_camera_frame(self, camera_id: str) -> Optional[FrameData]:
        """Récupère la dernière frame d'une caméra."""
        if camera_id not in self.cameras:
            return None
        
        return self.cameras[camera_id].get_latest_frame()
    
    def get_current_frames(self) -> Dict[str, FrameData]:
        """Récupère les frames courantes de toutes les caméras."""
        frames = {}
        
        for camera_id, camera in self.cameras.items():
            if camera.is_connected():
                frame = camera.get_current_frame()
                if frame:
                    frames[camera_id] = frame
        
        return frames
    
    def get_camera_status(self) -> Dict[str, Dict[str, Any]]:
        """Récupère le statut de toutes les caméras."""
        status = {}
        
        for camera_id, camera in self.cameras.items():
            status[camera_id] = {
                'name': camera.config.name,
                'connected': camera.is_connected(),
                'enabled': camera.config.enabled,
                'frame_count': camera.frame_count,
                'resolution': f"{camera.config.width}x{camera.config.height}",
                'fps': camera.config.fps,
                'source': camera.config.source
            }
        
        return status
    
    def add_frame_callback(self, callback: Callable[[FrameData], None]):
        """Ajoute un callback pour traitement des frames."""
        self.frame_callbacks.append(callback)
    
    def remove_frame_callback(self, callback: Callable[[FrameData], None]):
        """Supprime un callback de traitement des frames."""
        if callback in self.frame_callbacks:
            self.frame_callbacks.remove(callback)
    
    def update_camera_config(self, camera_id: str, config_updates: Dict[str, Any]) -> bool:
        """Met à jour la configuration d'une caméra."""
        if camera_id not in self.cameras:
            return False
        
        camera = self.cameras[camera_id]
        was_running = camera.is_connected()
        
        if was_running:
            camera.stop()
        
        # Mise à jour de la configuration
        for key, value in config_updates.items():
            if hasattr(camera.config, key):
                setattr(camera.config, key, value)
        
        if was_running and camera.config.enabled:
            return camera.start()
        
        return True


# Exemple d'utilisation pour tests
if __name__ == "__main__":
    # Configuration de test
    logging.basicConfig(level=logging.INFO)
    
    # Création du gestionnaire
    manager = MultiCameraManager()
    
    # Ajout d'une webcam
    webcam_config = CameraConfig(
        id="webcam_1",
        name="Webcam Test",
        source="0",
        width=640,
        height=480,
        fps=30
    )
    
    # Callback pour traiter les frames
    def process_frame(frame_data: FrameData):
        print(f"Frame reçue de {frame_data.camera_id}: {frame_data.timestamp}")
    
    manager.add_frame_callback(process_frame)
    manager.add_camera(webcam_config)
    
    # Démarrage
    results = manager.start_all_cameras()
    print(f"Résultats démarrage: {results}")
    
    # Test pendant 10 secondes
    time.sleep(10)
    
    # Arrêt
    manager.stop_all_cameras()
    print("Test terminé")