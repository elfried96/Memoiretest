"""
🤖 Intégration VLM Temps Réel pour Dashboard
===========================================

Connexion directe avec le système VLM existant pour analyse temps réel.
Interface entre le dashboard et les composants core du système.
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
import numpy as np
import cv2
import threading
import queue
import json
import logging

# Configuration du PYTHONPATH
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Imports du système core
try:
    from src.core.vlm.model import VisionLanguageModel
    from src.core.orchestrator.adaptive_orchestrator import AdaptiveVLMOrchestrator
    from src.core.headless.frame_analyzer import FrameAnalyzer
    from src.core.headless.surveillance_system import HeadlessSurveillanceSystem
    from src.core.types import AnalysisRequest, SuspicionLevel, ActionType, DetectionStatus
    from src.core.orchestrator.vlm_orchestrator import OrchestrationMode, OrchestrationConfig
    CORE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Core modules non disponibles: {e}")
    CORE_AVAILABLE = False

from .camera_manager import FrameData, CameraConfig

logger = logging.getLogger(__name__)


class VLMDetectionResult:
    """Résultat d'analyse VLM."""
    
    def __init__(self, frame_data: FrameData, analysis_result: Dict[str, Any]):
        self.camera_id = frame_data.camera_id
        self.timestamp = frame_data.timestamp
        self.frame_number = frame_data.frame_number
        
        # Résultats VLM
        self.detections = analysis_result.get('detections', [])
        self.suspicion_level = analysis_result.get('suspicion_level', SuspicionLevel.LOW.value)
        self.action_type = analysis_result.get('action_type', ActionType.NORMAL_SHOPPING.value)
        self.confidence = analysis_result.get('confidence', 0.0)
        self.description = analysis_result.get('description', '')
        self.alert_required = analysis_result.get('alert_required', False)
        
        # Métadonnées
        self.processing_time = analysis_result.get('processing_time', 0.0)
        self.tools_used = analysis_result.get('tools_used', [])
        self.bbox_annotations = analysis_result.get('bbox_annotations', [])
        
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire."""
        return {
            'camera_id': self.camera_id,
            'timestamp': self.timestamp.isoformat(),
            'frame_number': self.frame_number,
            'detections': self.detections,
            'suspicion_level': self.suspicion_level,
            'action_type': self.action_type,
            'confidence': self.confidence,
            'description': self.description,
            'alert_required': self.alert_required,
            'processing_time': self.processing_time,
            'tools_used': self.tools_used,
            'bbox_annotations': self.bbox_annotations
        }


class RealTimeVLMProcessor:
    """Processeur VLM temps réel pour le dashboard."""
    
    def __init__(self, 
                 vlm_model_name: str = "kimi-vl-a3b-thinking",
                 orchestration_mode: str = "BALANCED",
                 max_concurrent_analysis: int = 3):
        self.vlm_model_name = vlm_model_name
        self.orchestration_mode = orchestration_mode
        self.max_concurrent_analysis = max_concurrent_analysis
        
        # Composants VLM
        self.vlm_model = None
        self.orchestrator = None
        self.frame_analyzer = None
        self.surveillance_system = None
        
        # Queues de traitement
        self.frame_queue = queue.Queue(maxsize=50)
        self.result_queue = queue.Queue()
        
        # Threads de traitement
        self.processing_threads = []
        self.running = False
        
        # Callbacks
        self.detection_callbacks: List[Callable[[VLMDetectionResult], None]] = []
        self.error_callbacks: List[Callable[[Exception], None]] = []
        
        # Statistiques
        self.stats = {
            'frames_processed': 0,
            'frames_queued': 0,
            'average_processing_time': 0.0,
            'detections_count': 0,
            'alerts_count': 0
        }
        
        self.initialized = False
    
    async def initialize(self) -> bool:
        """Initialise les composants VLM."""
        if not CORE_AVAILABLE:
            logger.error("❌ Modules core non disponibles")
            return False
        
        try:
            logger.info("🔄 Initialisation des composants VLM...")
            
            # Initialisation du modèle VLM
            self.vlm_model = VisionLanguageModel(
                model_name=self.vlm_model_name,
                device="auto",
                load_in_4bit=True,
                max_tokens=512
            )
            
            # Initialisation de l'orchestrateur adaptatif
            self.orchestrator = AdaptiveVLMOrchestrator(
                vlm_model_name=self.vlm_model_name
            )
            
            # Initialisation de l'analyseur de frames
            self.frame_analyzer = FrameAnalyzer(
                orchestrator=self.orchestrator,
                save_frames=False
            )
            
            # Configuration de l'orchestration
            orchestration_config = OrchestrationConfig(
                mode=OrchestrationMode[self.orchestration_mode],
                confidence_threshold=0.7,
                max_tools_per_analysis=3,
                timeout_seconds=30.0
            )
            
            await self.orchestrator.initialize()
            
            self.initialized = True
            logger.info("✅ Composants VLM initialisés avec succès")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur initialisation VLM: {e}")
            self._notify_error(e)
            return False
    
    def start_processing(self) -> bool:
        """Démarre le traitement temps réel."""
        if not self.initialized:
            logger.error("❌ VLM non initialisé")
            return False
        
        if self.running:
            logger.warning("⚠️ Traitement déjà en cours")
            return True
        
        self.running = True
        
        # Création des threads de traitement
        for i in range(self.max_concurrent_analysis):
            thread = threading.Thread(
                target=self._processing_worker,
                args=(f"worker_{i}",),
                daemon=True
            )
            thread.start()
            self.processing_threads.append(thread)
        
        logger.info(f"✅ Traitement VLM démarré avec {self.max_concurrent_analysis} workers")
        return True
    
    def stop_processing(self):
        """Arrête le traitement."""
        self.running = False
        
        # Attendre que les threads se terminent
        for thread in self.processing_threads:
            if thread.is_alive():
                thread.join(timeout=2.0)
        
        self.processing_threads.clear()
        logger.info("⏹️ Traitement VLM arrêté")
    
    def process_frame(self, frame_data: FrameData) -> bool:
        """Ajoute une frame à traiter."""
        if not self.running:
            return False
        
        try:
            self.frame_queue.put_nowait(frame_data)
            self.stats['frames_queued'] += 1
            return True
        except queue.Full:
            logger.warning("⚠️ Queue de traitement pleine, frame ignorée")
            return False
    
    def _processing_worker(self, worker_name: str):
        """Worker thread pour traitement VLM."""
        logger.info(f"🔧 Worker {worker_name} démarré")
        
        while self.running:
            try:
                # Récupération d'une frame à traiter
                frame_data = self.frame_queue.get(timeout=1.0)
                
                # Traitement avec VLM
                start_time = datetime.now()
                result = self._analyze_frame_with_vlm(frame_data)
                processing_time = (datetime.now() - start_time).total_seconds()
                
                if result:
                    result['processing_time'] = processing_time
                    
                    # Création du résultat de détection
                    detection_result = VLMDetectionResult(frame_data, result)
                    
                    # Ajout à la queue des résultats
                    try:
                        self.result_queue.put_nowait(detection_result)
                    except queue.Full:
                        # Supprime un ancien résultat si la queue est pleine
                        try:
                            self.result_queue.get_nowait()
                            self.result_queue.put_nowait(detection_result)
                        except queue.Empty:
                            pass
                    
                    # Notification des callbacks
                    self._notify_detection(detection_result)
                    
                    # Mise à jour des statistiques
                    self._update_stats(detection_result, processing_time)
                
                self.frame_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"❌ Erreur processing worker {worker_name}: {e}")
                self._notify_error(e)
        
        logger.info(f"⏹️ Worker {worker_name} arrêté")
    
    def _analyze_frame_with_vlm(self, frame_data: FrameData) -> Optional[Dict[str, Any]]:
        """Analyse une frame avec le système VLM."""
        try:
            if not self.frame_analyzer:
                # Mode simulation si VLM non disponible
                return self._simulate_analysis(frame_data)
            
            # Préparation de la requête d'analyse
            analysis_request = AnalysisRequest(
                image=frame_data.frame,
                timestamp=frame_data.timestamp,
                camera_id=frame_data.camera_id,
                frame_number=frame_data.frame_number,
                detection_zones=frame_data.metadata.get('detection_zones', [])
            )
            
            # Analyse avec l'orchestrateur
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                analysis_result = loop.run_until_complete(
                    self.orchestrator.analyze_with_adaptive_tools(analysis_request)
                )
            finally:
                loop.close()
            
            if analysis_result:
                return {
                    'detections': analysis_result.detections,
                    'suspicion_level': analysis_result.suspicion_level.value,
                    'action_type': analysis_result.action_type.value,
                    'confidence': analysis_result.confidence,
                    'description': analysis_result.description,
                    'alert_required': analysis_result.suspicion_level in [SuspicionLevel.HIGH, SuspicionLevel.CRITICAL],
                    'tools_used': analysis_result.tools_used,
                    'bbox_annotations': analysis_result.bbox_annotations
                }
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Erreur analyse VLM: {e}")
            # Fallback sur la simulation
            return self._simulate_analysis(frame_data)
    
    def _simulate_analysis(self, frame_data: FrameData) -> Dict[str, Any]:
        """Simulation d'analyse pour tests/demo."""
        import random
        
        # Simulation de détections
        detections = []
        if random.random() > 0.3:  # 70% de chance de détecter quelque chose
            detections.append({
                'type': random.choice(['person', 'bag', 'suspicious_movement']),
                'confidence': random.uniform(0.6, 0.95),
                'bbox': [
                    random.randint(0, frame_data.frame.shape[1]//2),
                    random.randint(0, frame_data.frame.shape[0]//2),
                    random.randint(frame_data.frame.shape[1]//2, frame_data.frame.shape[1]),
                    random.randint(frame_data.frame.shape[0]//2, frame_data.frame.shape[0])
                ]
            })
        
        # Niveau de suspicion basé sur les détections
        if not detections:
            suspicion = SuspicionLevel.LOW
        elif any(d['type'] == 'suspicious_movement' for d in detections):
            suspicion = random.choice([SuspicionLevel.MEDIUM, SuspicionLevel.HIGH])
        else:
            suspicion = SuspicionLevel.LOW
        
        return {
            'detections': detections,
            'suspicion_level': suspicion.value,
            'action_type': ActionType.NORMAL_SHOPPING.value,
            'confidence': random.uniform(0.7, 0.9),
            'description': f"Analyse simulée - {len(detections)} détection(s)",
            'alert_required': suspicion in [SuspicionLevel.HIGH, SuspicionLevel.CRITICAL],
            'tools_used': ['yolo_detector', 'tracker'],
            'bbox_annotations': detections
        }
    
    def get_latest_results(self, max_results: int = 10) -> List[VLMDetectionResult]:
        """Récupère les derniers résultats."""
        results = []
        
        for _ in range(min(max_results, self.result_queue.qsize())):
            try:
                result = self.result_queue.get_nowait()
                results.append(result)
            except queue.Empty:
                break
        
        return results
    
    def add_detection_callback(self, callback: Callable[[VLMDetectionResult], None]):
        """Ajoute un callback pour les nouvelles détections."""
        self.detection_callbacks.append(callback)
    
    def add_error_callback(self, callback: Callable[[Exception], None]):
        """Ajoute un callback pour les erreurs."""
        self.error_callbacks.append(callback)
    
    def _notify_detection(self, result: VLMDetectionResult):
        """Notifie les callbacks de détection."""
        for callback in self.detection_callbacks:
            try:
                callback(result)
            except Exception as e:
                logger.error(f"❌ Erreur callback détection: {e}")
    
    def _notify_error(self, error: Exception):
        """Notifie les callbacks d'erreur."""
        for callback in self.error_callbacks:
            try:
                callback(error)
            except Exception as e:
                logger.error(f"❌ Erreur callback erreur: {e}")
    
    def _update_stats(self, result: VLMDetectionResult, processing_time: float):
        """Met à jour les statistiques."""
        self.stats['frames_processed'] += 1
        
        # Mise à jour du temps de traitement moyen
        current_avg = self.stats['average_processing_time']
        frame_count = self.stats['frames_processed']
        new_avg = (current_avg * (frame_count - 1) + processing_time) / frame_count
        self.stats['average_processing_time'] = new_avg
        
        # Comptage des détections et alertes
        if result.detections:
            self.stats['detections_count'] += len(result.detections)
        
        if result.alert_required:
            self.stats['alerts_count'] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Récupère les statistiques."""
        stats = self.stats.copy()
        stats['queue_size'] = self.frame_queue.qsize()
        stats['results_pending'] = self.result_queue.qsize()
        stats['is_running'] = self.running
        stats['is_initialized'] = self.initialized
        return stats


# Exemple d'utilisation
if __name__ == "__main__":
    import time
    from camera_manager import CameraConfig, MultiCameraManager
    
    logging.basicConfig(level=logging.INFO)
    
    # Configuration
    async def test_vlm_integration():
        # Initialisation du processeur VLM
        vlm_processor = RealTimeVLMProcessor()
        
        if await vlm_processor.initialize():
            print("✅ VLM initialisé")
            
            # Démarrage du traitement
            vlm_processor.start_processing()
            
            # Callback pour les détections
            def on_detection(result: VLMDetectionResult):
                print(f"🔍 Détection: {result.description} (confiance: {result.confidence:.2f})")
                if result.alert_required:
                    print(f"🚨 ALERTE: {result.suspicion_level}")
            
            vlm_processor.add_detection_callback(on_detection)
            
            # Simulation de frames
            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            from camera_manager import FrameData
            for i in range(5):
                frame_data = FrameData(
                    camera_id="test_cam",
                    frame=dummy_frame,
                    timestamp=datetime.now(),
                    frame_number=i
                )
                
                vlm_processor.process_frame(frame_data)
                time.sleep(1)
            
            # Attendre le traitement
            time.sleep(5)
            
            # Statistiques
            stats = vlm_processor.get_stats()
            print(f"📊 Statistiques: {stats}")
            
            # Arrêt
            vlm_processor.stop_processing()
        
        else:
            print("❌ Impossible d'initialiser le VLM")
    
    # Lancement du test
    asyncio.run(test_vlm_integration())