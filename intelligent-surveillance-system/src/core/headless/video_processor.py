"""
Gestionnaire de traitement vidéo pour le système headless.
"""

import cv2
import logging
from pathlib import Path
from typing import Generator, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)


class VideoProcessor:
    """Gestionnaire de capture et traitement vidéo optimisé."""

    def __init__(
        self,
        source: str = 0,
        frame_skip: int = 1,
        max_frames: Optional[int] = None,
        target_fps: Optional[int] = None,
        target_fps_processing: Optional[int] = None,
        enable_motion_detection: bool = False
    ):
        """
        Initialise le processeur vidéo.
        
        Args:
            source: Source vidéo (fichier ou webcam)
            frame_skip: (Obsolète) Nombre de frames à ignorer
            max_frames: Nombre maximum de frames à traiter
            target_fps: FPS cible pour la capture
            target_fps_processing: FPS cible pour l'analyse VLM (échantillonnage)
            enable_motion_detection: Active le filtrage par détection de mouvement
        """
        self.source = source
        self.frame_skip = frame_skip
        self.max_frames = max_frames
        self.target_fps = target_fps
        self.target_fps_processing = target_fps_processing
        self.enable_motion_detection = enable_motion_detection
        
        self.cap = None
        self.frame_count = 0
        self.processed_frames = 0
        
        # Pour la détection de mouvement
        self.previous_frame = None
        self.motion_threshold = 100000  # Seuil empirique, ajustable
        
        self._initialize_capture()

    def _initialize_capture(self) -> None:
        """Initialise la capture vidéo avec optimisations."""
        try:
            # Gestion webcam vs fichier
            if isinstance(self.source, str) and self.source.isdigit():
                source = int(self.source)
            else:
                source = self.source
                
            self.cap = cv2.VideoCapture(source)
            
            if not self.cap.isOpened():
                raise RuntimeError(f"Impossible d'ouvrir la source vidéo: {source}")
            
            # Configuration optimisée
            if self.target_fps:
                self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            
            # Buffer plus petit pour réduire la latence
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Info vidéo
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            logger.info(f"📹 Vidéo initialisée: {width}x{height} @ {fps:.1f}fps")
            
            if isinstance(self.source, str) and Path(self.source).is_file():
                total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = total_frames / fps if fps > 0 else 0
                logger.info(f"📊 Durée: {duration:.1f}s ({total_frames} frames)")
                
        except Exception as e:
            logger.error(f"❌ Erreur initialisation vidéo: {e}")
            raise

    def frames_generator(self) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Générateur de frames optimisé avec échantillonnage basé sur le FPS cible.
        
        Yields:
            Tuple[frame_id, frame]: ID et array de la frame
        """
        if not self.cap or not self.cap.isOpened():
            logger.error("Capture vidéo non initialisée ou fermée.")
            return

        source_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if source_fps <= 0:
            logger.warning("FPS source inconnu, analyse de chaque frame.")
            step = 1
        else:
            # Nouveau paramètre pour contrôler le FPS de l'analyse
            target_fps_processing = getattr(self, 'target_fps_processing', source_fps)
            step = int(source_fps / target_fps_processing) if target_fps_processing > 0 else 1
            step = max(1, step) # Assurer qu'on avance d'au moins 1

        logger.info(f"Échantillonnage: analyse de ~1 frame toutes les {step} frames (Source: {source_fps:.1f}fps -> Cible: {target_fps_processing:.1f}fps).")

        try:
            while True:
                if self.max_frames and self.processed_frames >= self.max_frames:
                    logger.info(f"Limite atteinte: {self.max_frames} frames traitées.")
                    break
                
                ret, frame = self.cap.read()
                if not ret:
                    logger.info("Fin de vidéo.")
                    break

                self.frame_count += 1
                self.processed_frames += 1

                if frame is None or frame.size == 0:
                    logger.warning(f"Frame {self.frame_count} invalide, ignorée.")
                    continue

                # NOUVEAU: Détection de mouvement pour ignorer les frames statiques
                if self.enable_motion_detection:
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

                    if self.previous_frame is not None:
                        frame_delta = cv2.absdiff(self.previous_frame, gray_frame)
                        if np.sum(frame_delta) < self.motion_threshold:
                            # Mouvement insuffisant, on saute cette frame mais on avance dans la vidéo
                            if step > 1:
                                for _ in range(step - 1):
                                    if not self.cap.read()[0]: break
                                    self.frame_count += 1
                            self.previous_frame = gray_frame
                            continue # Passe à l'itération suivante de la boucle while

                    self.previous_frame = gray_frame
                
                yield self.processed_frames, frame

                # Sauter les frames suivantes pour atteindre le FPS cible
                if step > 1:
                    for _ in range(step - 1):
                        if not self.cap.read()[0]:
                            break # Fin de la vidéo
                        self.frame_count += 1
                
        except Exception as e:
            logger.error(f"Erreur lecture frame {self.frame_count}: {e}", exc_info=True)
            raise
        finally:
            self.release()

    def save_frame(
        self, 
        frame: np.ndarray, 
        frame_id: int, 
        output_dir: Path,
        prefix: str = "frame"
    ) -> Path:
        """
        Sauvegarde une frame sur disque.
        
        Args:
            frame: Frame à sauvegarder
            frame_id: ID de la frame
            output_dir: Répertoire de sortie
            prefix: Préfixe du nom de fichier
            
        Returns:
            Path vers le fichier sauvé
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = output_dir / f"{prefix}_{frame_id:06d}.jpg"
        
        try:
            success = cv2.imwrite(str(filename), frame)
            if not success:
                raise RuntimeError("Échec de sauvegarde")
            
            logger.debug(f"💾 Frame sauvée: {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde frame {frame_id}: {e}")
            raise

    def get_video_info(self) -> dict:
        """Retourne les informations de la vidéo."""
        if not self.cap:
            return {}
            
        return {
            "fps": self.cap.get(cv2.CAP_PROP_FPS),
            "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "total_frames": int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "processed_frames": self.processed_frames,
            "frame_count": self.frame_count
        }

    def release(self) -> None:
        """Libère les ressources."""
        if self.cap:
            self.cap.release()
            logger.debug("📹 Ressources vidéo libérées")