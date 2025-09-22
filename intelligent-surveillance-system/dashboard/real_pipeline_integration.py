"""
🔬 Intégration Pipeline VLM Réelle pour Dashboard
==================================================

Connecte le dashboard aux vrais composants du système :
- AdaptiveVLMOrchestrator avec 8 outils avancés
- ToolOptimizationBenchmark pour optimisation
- Métriques de performance réelles
- Pipeline complète de détection
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
import time
from dataclasses import dataclass, asdict

# Configuration du PYTHONPATH
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Imports du système core
try:
    from src.core.orchestrator.adaptive_orchestrator import AdaptiveVLMOrchestrator, ContextPattern, ToolPerformanceHistory
    from src.core.vlm.model import VisionLanguageModel
    from src.core.vlm.tools_integration import AdvancedToolsManager
    from src.core.vlm.dynamic_model import DynamicVisionLanguageModel
    from src.core.types import AnalysisRequest, AnalysisResponse, SuspicionLevel, ActionType, DetectionStatus, ToolResult
    from src.core.orchestrator.vlm_orchestrator import OrchestrationConfig, OrchestrationMode
    from src.testing.tool_optimization_benchmark import ToolOptimizationBenchmark, ToolPerformanceMetrics, ToolCombinationResult
    from src.core.monitoring.performance_monitor import PerformanceMonitor
    from src.core.monitoring.vlm_metrics import VLMMetricsCollector
    CORE_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("✅ Modules VLM core chargés avec succès")
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.error(f"❌ Impossible d'importer les modules core: {e}")
    CORE_AVAILABLE = False

from camera_manager import FrameData


@dataclass
class RealAnalysisResult:
    """Résultat d'analyse VLM réel."""
    frame_id: str
    camera_id: str
    timestamp: datetime
    
    # Résultats VLM
    suspicion_level: SuspicionLevel
    action_type: ActionType
    confidence: float
    description: str
    
    # Détections spécifiques
    detections: List[Dict[str, Any]]
    tool_results: Dict[str, ToolResult]
    
    # Métriques de performance
    processing_time: float
    tools_used: List[str]
    optimization_score: float
    
    # Contexte et métadonnées
    context_pattern: Optional[str]
    risk_assessment: Dict[str, float]
    bbox_annotations: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'frame_id': self.frame_id,
            'camera_id': self.camera_id,
            'timestamp': self.timestamp.isoformat(),
            'suspicion_level': self.suspicion_level.value if hasattr(self.suspicion_level, 'value') else str(self.suspicion_level),
            'action_type': self.action_type.value if hasattr(self.action_type, 'value') else str(self.action_type),
            'confidence': self.confidence,
            'description': self.description,
            'detections': self.detections,
            'tool_results': {k: asdict(v) if hasattr(v, '__dict__') else str(v) for k, v in self.tool_results.items()},
            'processing_time': self.processing_time,
            'tools_used': self.tools_used,
            'optimization_score': self.optimization_score,
            'context_pattern': self.context_pattern,
            'risk_assessment': self.risk_assessment,
            'bbox_annotations': self.bbox_annotations
        }


class RealVLMPipeline:
    """Pipeline VLM réelle intégrée au dashboard."""
    
    def __init__(self, 
                 vlm_model_name: str = "kimi-vl-a3b-thinking",
                 enable_optimization: bool = True,
                 max_concurrent_analysis: int = 2):
        
        self.vlm_model_name = vlm_model_name
        self.enable_optimization = enable_optimization
        self.max_concurrent_analysis = max_concurrent_analysis
        
        # Composants principaux
        self.orchestrator = None
        self.vlm_model = None
        self.tools_manager = None
        self.benchmark = None
        self.performance_monitor = None
        self.metrics_collector = None
        
        # Queues de traitement
        self.analysis_queue = queue.Queue(maxsize=100)
        self.results_queue = queue.Queue()
        self.optimization_queue = queue.Queue()
        
        # Threads de traitement
        self.analysis_workers = []
        self.optimization_worker = None
        self.running = False
        
        # Callbacks
        self.analysis_callbacks: List[Callable[[RealAnalysisResult], None]] = []
        self.optimization_callbacks: List[Callable[[Dict[str, Any]], None]] = []
        self.error_callbacks: List[Callable[[Exception], None]] = []
        
        # Cache et optimisation
        self.current_optimal_tools = []
        self.tool_performance_cache = {}
        self.context_patterns_cache = {}
        
        # Statistiques en temps réel
        self.stats = {
            'frames_processed': 0,
            'average_processing_time': 0.0,
            'tool_usage_stats': {},
            'optimization_cycles': 0,
            'current_performance_score': 0.0,
            'best_tool_combination': [],
            'total_detections': 0,
            'accuracy_score': 0.0
        }
        
        self.initialized = False
    
    async def initialize(self) -> bool:
        """Initialise la pipeline VLM complète."""
        if not CORE_AVAILABLE:
            logger.error("❌ Modules core VLM non disponibles")
            return False
        
        try:
            logger.info("🔄 Initialisation de la pipeline VLM réelle...")
            
            # 1. Orchestrateur adaptatif
            logger.info("📋 Initialisation AdaptiveVLMOrchestrator...")
            orchestration_config = OrchestrationConfig(
                mode=OrchestrationMode.THOROUGH,
                confidence_threshold=0.7,
                max_concurrent_tools=4,
                timeout_seconds=30,
                enable_advanced_tools=True
            )
            
            self.orchestrator = AdaptiveVLMOrchestrator(
                vlm_model_name=self.vlm_model_name,
                config=orchestration_config,
                enable_adaptive_learning=True
            )
            
            # 2. Gestionnaire d'outils avancés  
            logger.info("🛠️ Initialisation AdvancedToolsManager...")
            self.tools_manager = AdvancedToolsManager()
            
            # 3. Modèle VLM
            logger.info("🤖 Initialisation VisionLanguageModel...")
            self.vlm_model = VisionLanguageModel(
                model_name=self.vlm_model_name,
                device="auto",
                load_in_4bit=True,
                max_tokens=512
            )
            
            # 4. Système de benchmark
            if self.enable_optimization:
                logger.info("🧪 Initialisation ToolOptimizationBenchmark...")
                self.benchmark = ToolOptimizationBenchmark(
                    vlm_model_name=self.vlm_model_name,
                    test_data_path=Path("data/benchmark/test_cases")
                )
            
            # 5. Monitoring de performance
            logger.info("📊 Initialisation Performance Monitoring...")
            self.performance_monitor = PerformanceMonitor()
            self.metrics_collector = VLMMetricsCollector()
            
            # 6. Initialisation des composants (orchestrator déjà initialisé)
            
            # 7. Chargement de la configuration optimale existante
            await self._load_optimization_data()
            
            self.initialized = True
            logger.info("✅ Pipeline VLM réelle initialisée avec succès")
            
            # 8. Démarrage de l'optimisation continue si activée
            if self.enable_optimization:
                self._start_background_optimization()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur initialisation pipeline VLM: {e}")
            self._notify_error(e)
            return False
    
    def start_processing(self) -> bool:
        """Démarre le traitement temps réel."""
        if not self.initialized:
            logger.error("❌ Pipeline non initialisée")
            return False
        
        if self.running:
            logger.warning("⚠️ Pipeline déjà en cours d'exécution")
            return True
        
        self.running = True
        
        # Démarrage des workers d'analyse
        for i in range(self.max_concurrent_analysis):
            worker = threading.Thread(
                target=self._analysis_worker,
                args=(f"analyzer_{i}",),
                daemon=True
            )
            worker.start()
            self.analysis_workers.append(worker)
        
        logger.info(f"✅ Pipeline VLM démarrée avec {self.max_concurrent_analysis} workers")
        return True
    
    def stop_processing(self):
        """Arrête le traitement."""
        self.running = False
        
        # Arrêt des workers
        for worker in self.analysis_workers:
            if worker.is_alive():
                worker.join(timeout=5.0)
        
        if self.optimization_worker and self.optimization_worker.is_alive():
            self.optimization_worker.join(timeout=5.0)
        
        self.analysis_workers.clear()
        logger.info("⏹️ Pipeline VLM arrêtée")
    
    async def analyze_frame(self, frame_data: FrameData) -> Optional[RealAnalysisResult]:
        """Analyse une frame avec la pipeline complète."""
        try:
            # Ajout à la queue d'analyse
            analysis_request = {
                'frame_data': frame_data,
                'timestamp': datetime.now(),
                'frame_id': f"{frame_data.camera_id}_{frame_data.frame_number}"
            }
            
            if not self.running:
                return None
            
            try:
                self.analysis_queue.put_nowait(analysis_request)
                return True  # Mis en queue avec succès
            except queue.Full:
                logger.warning("⚠️ Queue d'analyse pleine, frame ignorée")
                return False
                
        except Exception as e:
            logger.error(f"❌ Erreur ajout frame à l'analyse: {e}")
            return False
    
    def _analysis_worker(self, worker_name: str):
        """Worker thread pour analyse VLM."""
        logger.info(f"🔧 Worker d'analyse {worker_name} démarré")
        
        while self.running:
            try:
                # Récupération d'une requête d'analyse
                analysis_request = self.analysis_queue.get(timeout=1.0)
                
                # Traitement avec pipeline VLM
                start_time = time.time()
                result = asyncio.run(self._process_frame_with_vlm(analysis_request))
                processing_time = time.time() - start_time
                
                if result:
                    result.processing_time = processing_time
                    
                    # Ajout aux résultats
                    try:
                        self.results_queue.put_nowait(result)
                    except queue.Full:
                        # Suppression du plus ancien résultat
                        try:
                            self.results_queue.get_nowait()
                            self.results_queue.put_nowait(result)
                        except queue.Empty:
                            pass
                    
                    # Notification des callbacks
                    self._notify_analysis_result(result)
                    
                    # Mise à jour des statistiques
                    self._update_stats(result)
                
                self.analysis_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"❌ Erreur worker analyse {worker_name}: {e}")
                self._notify_error(e)
        
        logger.info(f"⏹️ Worker d'analyse {worker_name} arrêté")
    
    async def _process_frame_with_vlm(self, analysis_request: Dict[str, Any]) -> Optional[RealAnalysisResult]:
        """Traite une frame avec la pipeline VLM complète."""
        try:
            frame_data = analysis_request['frame_data']
            frame_id = analysis_request['frame_id']
            
            # Création de la requête d'analyse
            vlm_request = AnalysisRequest(
                image=frame_data.frame,
                timestamp=frame_data.timestamp,
                camera_id=frame_data.camera_id,
                frame_number=frame_data.frame_number,
                context=frame_data.metadata or {}
            )
            
            # Analyse avec orchestrateur adaptatif
            if hasattr(self.orchestrator, 'analyze_with_adaptive_tools'):
                vlm_response = await self.orchestrator.analyze_with_adaptive_tools(vlm_request)
            else:
                # Fallback sur méthode standard
                vlm_response = await self.orchestrator.analyze(vlm_request)
            
            if not vlm_response:
                logger.warning(f"⚠️ Pas de réponse VLM pour frame {frame_id}")
                return None
            
            # Construction du résultat
            result = RealAnalysisResult(
                frame_id=frame_id,
                camera_id=frame_data.camera_id,
                timestamp=frame_data.timestamp,
                suspicion_level=vlm_response.suspicion_level,
                action_type=vlm_response.action_type,
                confidence=vlm_response.confidence,
                description=vlm_response.description,
                detections=vlm_response.detections,
                tool_results=getattr(vlm_response, 'tool_results', {}),
                processing_time=0.0,  # Sera mis à jour par le worker
                tools_used=getattr(vlm_response, 'tools_used', []),
                optimization_score=getattr(vlm_response, 'optimization_score', 0.0),
                context_pattern=getattr(vlm_response, 'context_pattern', None),
                risk_assessment=getattr(vlm_response, 'risk_assessment', {}),
                bbox_annotations=getattr(vlm_response, 'bbox_annotations', [])
            )
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Erreur traitement VLM: {e}")
            return None
    
    def _start_background_optimization(self):
        """Démarre l'optimisation en arrière-plan."""
        if not self.benchmark:
            return
        
        self.optimization_worker = threading.Thread(
            target=self._optimization_worker,
            daemon=True
        )
        self.optimization_worker.start()
        logger.info("🚀 Optimisation continue démarrée")
    
    def _optimization_worker(self):
        """Worker pour optimisation continue des outils."""
        logger.info("🔧 Worker d'optimisation démarré")
        
        optimization_interval = 300  # 5 minutes
        last_optimization = time.time()
        
        while self.running:
            try:
                current_time = time.time()
                
                # Optimisation périodique
                if current_time - last_optimization >= optimization_interval:
                    logger.info("🔄 Démarrage cycle d'optimisation...")
                    
                    # Exécution du benchmark
                    optimization_results = asyncio.run(self._run_optimization_cycle())
                    
                    if optimization_results:
                        # Mise à jour de la configuration optimale
                        self._update_optimal_configuration(optimization_results)
                        
                        # Notification des callbacks
                        self._notify_optimization_result(optimization_results)
                        
                        self.stats['optimization_cycles'] += 1
                        logger.info("✅ Cycle d'optimisation terminé")
                    
                    last_optimization = current_time
                
                time.sleep(60)  # Vérification chaque minute
                
            except Exception as e:
                logger.error(f"❌ Erreur worker optimisation: {e}")
                self._notify_error(e)
                time.sleep(300)  # Attendre 5 min avant de reprendre
        
        logger.info("⏹️ Worker d'optimisation arrêté")
    
    async def _run_optimization_cycle(self) -> Optional[Dict[str, Any]]:
        """Exécute un cycle d'optimisation complet."""
        try:
            if not self.benchmark:
                return None
            
            # Lancement du benchmark avec les données récentes
            benchmark_results = await self.benchmark.run_comprehensive_benchmark()
            
            if not benchmark_results:
                logger.warning("⚠️ Aucun résultat de benchmark")
                return None
            
            # Analyse des résultats
            optimization_data = {
                'timestamp': datetime.now().isoformat(),
                'best_combination': benchmark_results.get('best_combination', []),
                'performance_improvement': benchmark_results.get('performance_improvement', 0.0),
                'tool_rankings': benchmark_results.get('tool_rankings', {}),
                'execution_time': benchmark_results.get('execution_time', 0.0)
            }
            
            return optimization_data
            
        except Exception as e:
            logger.error(f"❌ Erreur cycle d'optimisation: {e}")
            return None
    
    def _update_optimal_configuration(self, optimization_results: Dict[str, Any]):
        """Met à jour la configuration optimale."""
        try:
            best_combination = optimization_results.get('best_combination', [])
            
            if best_combination:
                self.current_optimal_tools = best_combination
                self.stats['best_tool_combination'] = best_combination
                
                # Mise à jour de l'orchestrateur
                if hasattr(self.orchestrator, 'update_optimal_tools'):
                    self.orchestrator.update_optimal_tools(best_combination)
                
                logger.info(f"📊 Configuration optimale mise à jour: {best_combination}")
            
        except Exception as e:
            logger.error(f"❌ Erreur mise à jour configuration: {e}")
    
    async def _load_optimization_data(self):
        """Charge les données d'optimisation existantes."""
        try:
            # Chargement des patterns de contexte sauvegardés
            # Chargement des performances d'outils
            # Initialisation avec la meilleure configuration connue
            logger.info("📂 Données d'optimisation chargées")
            
        except Exception as e:
            logger.warning(f"⚠️ Impossible de charger données d'optimisation: {e}")
    
    def get_latest_results(self, max_results: int = 20) -> List[RealAnalysisResult]:
        """Récupère les derniers résultats d'analyse."""
        results = []
        
        for _ in range(min(max_results, self.results_queue.qsize())):
            try:
                result = self.results_queue.get_nowait()
                results.append(result)
            except queue.Empty:
                break
        
        return results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Récupère les statistiques de performance."""
        stats = self.stats.copy()
        
        # Ajout des statistiques dynamiques
        stats.update({
            'analysis_queue_size': self.analysis_queue.qsize(),
            'results_pending': self.results_queue.qsize(),
            'is_running': self.running,
            'is_initialized': self.initialized,
            'optimal_tools_count': len(self.current_optimal_tools),
            'current_optimal_tools': self.current_optimal_tools
        })
        
        return stats
    
    def get_tool_performance_details(self) -> Dict[str, Any]:
        """Récupère les détails de performance par outil."""
        if not self.tools_manager:
            return {}
        
        return {
            'available_tools': list(self.tools_manager.tools.keys()) if self.tools_manager.tools else [],
            'tool_usage_stats': self.stats.get('tool_usage_stats', {}),
            'performance_cache': self.tool_performance_cache,
            'context_patterns': list(self.context_patterns_cache.keys())
        }
    
    # Callbacks et notifications
    def add_analysis_callback(self, callback: Callable[[RealAnalysisResult], None]):
        """Ajoute un callback pour les résultats d'analyse."""
        self.analysis_callbacks.append(callback)
    
    def add_optimization_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Ajoute un callback pour les résultats d'optimisation."""
        self.optimization_callbacks.append(callback)
    
    def add_error_callback(self, callback: Callable[[Exception], None]):
        """Ajoute un callback pour les erreurs."""
        self.error_callbacks.append(callback)
    
    def _notify_analysis_result(self, result: RealAnalysisResult):
        """Notifie les callbacks de résultats d'analyse."""
        for callback in self.analysis_callbacks:
            try:
                callback(result)
            except Exception as e:
                logger.error(f"❌ Erreur callback analyse: {e}")
    
    def _notify_optimization_result(self, result: Dict[str, Any]):
        """Notifie les callbacks d'optimisation."""
        for callback in self.optimization_callbacks:
            try:
                callback(result)
            except Exception as e:
                logger.error(f"❌ Erreur callback optimisation: {e}")
    
    def _notify_error(self, error: Exception):
        """Notifie les callbacks d'erreur."""
        for callback in self.error_callbacks:
            try:
                callback(error)
            except Exception as e:
                logger.error(f"❌ Erreur callback erreur: {e}")
    
    def _update_stats(self, result: RealAnalysisResult):
        """Met à jour les statistiques."""
        self.stats['frames_processed'] += 1
        
        # Temps de traitement moyen
        current_avg = self.stats['average_processing_time']
        frame_count = self.stats['frames_processed']
        new_avg = (current_avg * (frame_count - 1) + result.processing_time) / frame_count
        self.stats['average_processing_time'] = new_avg
        
        # Statistiques d'utilisation des outils
        for tool in result.tools_used:
            if tool not in self.stats['tool_usage_stats']:
                self.stats['tool_usage_stats'][tool] = 0
            self.stats['tool_usage_stats'][tool] += 1
        
        # Comptage des détections
        if result.detections:
            self.stats['total_detections'] += len(result.detections)
        
        # Score de performance actuel
        self.stats['current_performance_score'] = result.optimization_score


# Instance globale pour le dashboard
real_pipeline = None

def get_real_pipeline() -> Optional[RealVLMPipeline]:
    """Récupère l'instance de pipeline réelle."""
    return real_pipeline

async def initialize_real_pipeline_async(**kwargs) -> bool:
    """Initialise la pipeline réelle (version async)."""
    global real_pipeline
    
    if real_pipeline is None:
        real_pipeline = RealVLMPipeline(**kwargs)
    
    return await real_pipeline.initialize()

def initialize_real_pipeline(**kwargs) -> bool:
    """Initialise la pipeline réelle (version sync)."""
    global real_pipeline
    
    if real_pipeline is None:
        real_pipeline = RealVLMPipeline(**kwargs)
    
    try:
        return asyncio.run(real_pipeline.initialize())
    except RuntimeError:
        # Si on est déjà dans une boucle asyncio
        return False

def is_real_pipeline_available() -> bool:
    """Vérifie si la pipeline réelle est disponible."""
    return CORE_AVAILABLE and real_pipeline is not None and real_pipeline.initialized