"""Utilitaires de mesure et optimisation des performances."""

import time
import psutil
import threading
from typing import Dict, List, Optional, Any, Callable, Tuple
from functools import wraps
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import asyncio
import numpy as np
from loguru import logger

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


@dataclass
class PerformanceMetrics:
    """Métriques de performance."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    gpu_percent: Optional[float] = None
    gpu_memory_percent: Optional[float] = None
    gpu_memory_used_mb: Optional[float] = None
    disk_io_read_mb: float = 0.0
    disk_io_write_mb: float = 0.0
    network_bytes_sent: int = 0
    network_bytes_recv: int = 0


@dataclass
class TimingMetrics:
    """Métriques de timing pour les fonctions."""
    function_name: str
    call_count: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    avg_time: float = 0.0
    last_call_time: float = 0.0
    
    def add_measurement(self, execution_time: float) -> None:
        """Ajouter une mesure d'exécution."""
        self.call_count += 1
        self.total_time += execution_time
        self.min_time = min(self.min_time, execution_time)
        self.max_time = max(self.max_time, execution_time)
        self.avg_time = self.total_time / self.call_count
        self.last_call_time = execution_time


class PerformanceMonitor:
    """Moniteur de performance système."""
    
    def __init__(self, collection_interval: float = 1.0):
        self.collection_interval = collection_interval
        self.is_monitoring = False
        self.metrics_history: List[PerformanceMetrics] = []
        self.max_history_size = 3600  # 1 heure à 1Hz
        
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Références pour calculs différentiels
        self._last_disk_io = None
        self._last_network_io = None
    
    def start_monitoring(self) -> None:
        """Démarrage du monitoring continu."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self._stop_event.clear()
        
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True
        )
        self._monitor_thread.start()
        
        logger.info("Monitoring de performance démarré")
    
    def stop_monitoring(self) -> None:
        """Arrêt du monitoring."""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        self._stop_event.set()
        
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
        
        logger.info("Monitoring de performance arrêté")
    
    def _monitor_loop(self) -> None:
        """Boucle principale de monitoring."""
        while not self._stop_event.wait(self.collection_interval):
            try:
                metrics = self.collect_current_metrics()
                self.metrics_history.append(metrics)
                
                # Limitation de l'historique
                if len(self.metrics_history) > self.max_history_size:
                    self.metrics_history = self.metrics_history[-self.max_history_size:]
                
            except Exception as e:
                logger.error(f"Erreur monitoring performance: {e}")
    
    def collect_current_metrics(self) -> PerformanceMetrics:
        """Collecte des métriques actuelles."""
        # Métriques CPU et mémoire
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        
        # Métriques disque
        disk_io = psutil.disk_io_counters()
        if self._last_disk_io is not None and disk_io is not None:
            disk_read_mb = (disk_io.read_bytes - self._last_disk_io.read_bytes) / (1024 * 1024)
            disk_write_mb = (disk_io.write_bytes - self._last_disk_io.write_bytes) / (1024 * 1024)
        else:
            disk_read_mb = disk_write_mb = 0.0
        
        if disk_io:
            self._last_disk_io = disk_io
        
        # Métriques réseau
        network_io = psutil.net_io_counters()
        if self._last_network_io is not None and network_io is not None:
            net_sent = network_io.bytes_sent - self._last_network_io.bytes_sent
            net_recv = network_io.bytes_recv - self._last_network_io.bytes_recv
        else:
            net_sent = net_recv = 0
        
        if network_io:
            self._last_network_io = network_io
        
        # Métriques GPU
        gpu_percent = None
        gpu_memory_percent = None
        gpu_memory_used_mb = None
        
        if GPU_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Premier GPU
                    gpu_percent = gpu.load * 100
                    gpu_memory_percent = gpu.memoryUtil * 100
                    gpu_memory_used_mb = gpu.memoryUsed
            except Exception:
                pass  # GPU non disponible ou erreur
        
        return PerformanceMetrics(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_mb=memory.used / (1024 * 1024),
            gpu_percent=gpu_percent,
            gpu_memory_percent=gpu_memory_percent,
            gpu_memory_used_mb=gpu_memory_used_mb,
            disk_io_read_mb=disk_read_mb,
            disk_io_write_mb=disk_write_mb,
            network_bytes_sent=net_sent,
            network_bytes_recv=net_recv
        )
    
    def get_current_metrics(self) -> PerformanceMetrics:
        """Récupération des métriques actuelles."""
        return self.collect_current_metrics()
    
    def get_metrics_history(self, duration_minutes: int = 10) -> List[PerformanceMetrics]:
        """Récupération de l'historique des métriques."""
        cutoff_time = datetime.now() - timedelta(minutes=duration_minutes)
        return [
            metrics for metrics in self.metrics_history
            if metrics.timestamp >= cutoff_time
        ]
    
    def get_average_metrics(self, duration_minutes: int = 5) -> Optional[Dict[str, float]]:
        """Calcul des métriques moyennes sur une période."""
        history = self.get_metrics_history(duration_minutes)
        
        if not history:
            return None
        
        return {
            "avg_cpu_percent": np.mean([m.cpu_percent for m in history]),
            "avg_memory_percent": np.mean([m.memory_percent for m in history]),
            "avg_memory_used_mb": np.mean([m.memory_used_mb for m in history]),
            "avg_gpu_percent": np.mean([m.gpu_percent for m in history if m.gpu_percent is not None]),
            "max_cpu_percent": np.max([m.cpu_percent for m in history]),
            "max_memory_percent": np.max([m.memory_percent for m in history]),
            "total_disk_read_mb": sum([m.disk_io_read_mb for m in history]),
            "total_disk_write_mb": sum([m.disk_io_write_mb for m in history])
        }
    
    def is_system_overloaded(self, thresholds: Optional[Dict[str, float]] = None) -> bool:
        """Vérification si le système est surchargé."""
        if not thresholds:
            thresholds = {
                "cpu_percent": 85.0,
                "memory_percent": 90.0,
                "gpu_percent": 95.0
            }
        
        current_metrics = self.collect_current_metrics()
        
        if current_metrics.cpu_percent > thresholds["cpu_percent"]:
            return True
        
        if current_metrics.memory_percent > thresholds["memory_percent"]:
            return True
        
        if (current_metrics.gpu_percent is not None and 
            current_metrics.gpu_percent > thresholds["gpu_percent"]):
            return True
        
        return False
    
    def __del__(self):
        """Nettoyage lors de la destruction."""
        self.stop_monitoring()


class FunctionTimer:
    """Timer pour mesurer les performances des fonctions."""
    
    def __init__(self):
        self.timing_metrics: Dict[str, TimingMetrics] = {}
        self._lock = threading.Lock()
    
    def add_timing(self, function_name: str, execution_time: float) -> None:
        """Ajouter une mesure de timing."""
        with self._lock:
            if function_name not in self.timing_metrics:
                self.timing_metrics[function_name] = TimingMetrics(function_name)
            
            self.timing_metrics[function_name].add_measurement(execution_time)
    
    def get_function_stats(self, function_name: str) -> Optional[TimingMetrics]:
        """Récupération des stats d'une fonction."""
        return self.timing_metrics.get(function_name)
    
    def get_all_stats(self) -> Dict[str, TimingMetrics]:
        """Récupération de toutes les stats."""
        return self.timing_metrics.copy()
    
    def get_slowest_functions(self, top_n: int = 5) -> List[TimingMetrics]:
        """Récupération des fonctions les plus lentes."""
        return sorted(
            self.timing_metrics.values(),
            key=lambda x: x.avg_time,
            reverse=True
        )[:top_n]
    
    def reset_stats(self) -> None:
        """Réinitialisation des statistiques."""
        with self._lock:
            self.timing_metrics.clear()


# Instance globale
_performance_monitor = PerformanceMonitor()
_function_timer = FunctionTimer()


def measure_time(func: Optional[Callable] = None, *, name: Optional[str] = None):
    """
    Décorateur pour mesurer le temps d'exécution des fonctions.
    
    Usage:
    @measure_time
    def my_function():
        pass
    
    @measure_time(name="custom_name")
    def my_function():
        pass
    """
    
    def decorator(f: Callable) -> Callable:
        function_name = name or f.__name__
        
        if asyncio.iscoroutinefunction(f):
            @wraps(f)
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = await f(*args, **kwargs)
                    return result
                finally:
                    execution_time = time.time() - start_time
                    _function_timer.add_timing(function_name, execution_time)
            
            return async_wrapper
        else:
            @wraps(f)
            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = f(*args, **kwargs)
                    return result
                finally:
                    execution_time = time.time() - start_time
                    _function_timer.add_timing(function_name, execution_time)
            
            return sync_wrapper
    
    if func is None:
        return decorator
    else:
        return decorator(func)


class PerformanceOptimizer:
    """Optimiseur de performances adaptatif."""
    
    def __init__(self, monitor: Optional[PerformanceMonitor] = None):
        self.monitor = monitor or _performance_monitor
        
        # Seuils d'optimisation
        self.optimization_thresholds = {
            "cpu_high": 80.0,
            "memory_high": 85.0,
            "gpu_high": 90.0,
            "latency_high": 2.0  # secondes
        }
        
        # Actions d'optimisation
        self.optimization_actions = {
            "reduce_quality": False,
            "reduce_fps": False,
            "reduce_concurrent_streams": False,
            "enable_caching": True,
            "use_quantization": False
        }
    
    def should_optimize(self) -> Tuple[bool, List[str]]:
        """Vérification si une optimisation est nécessaire."""
        current_metrics = self.monitor.collect_current_metrics()
        optimization_needed = []
        
        if current_metrics.cpu_percent > self.optimization_thresholds["cpu_high"]:
            optimization_needed.append("cpu_overload")
        
        if current_metrics.memory_percent > self.optimization_thresholds["memory_high"]:
            optimization_needed.append("memory_overload")
        
        if (current_metrics.gpu_percent is not None and 
            current_metrics.gpu_percent > self.optimization_thresholds["gpu_high"]):
            optimization_needed.append("gpu_overload")
        
        # Vérification de la latence des fonctions
        slowest_functions = _function_timer.get_slowest_functions(3)
        for func_metrics in slowest_functions:
            if func_metrics.avg_time > self.optimization_thresholds["latency_high"]:
                optimization_needed.append(f"slow_function_{func_metrics.function_name}")
        
        return len(optimization_needed) > 0, optimization_needed
    
    def suggest_optimizations(self, issues: List[str]) -> Dict[str, Any]:
        """Suggestions d'optimisation basées sur les problèmes détectés."""
        suggestions = {}
        
        if "cpu_overload" in issues:
            suggestions["cpu"] = [
                "Réduire le nombre de flux simultanés",
                "Utiliser la quantization des modèles",
                "Optimiser la fréquence d'inférence",
                "Activer le cache intelligent"
            ]
        
        if "memory_overload" in issues:
            suggestions["memory"] = [
                "Réduire la résolution des images",
                "Limiter l'historique des tracks",
                "Utiliser la quantization 4-bit",
                "Nettoyer les caches plus fréquemment"
            ]
        
        if "gpu_overload" in issues:
            suggestions["gpu"] = [
                "Utiliser des modèles plus légers",
                "Réduire la taille des batches",
                "Optimiser les pipelines CUDA",
                "Utiliser la précision mixte"
            ]
        
        # Suggestions pour fonctions lentes
        slow_functions = [issue for issue in issues if issue.startswith("slow_function_")]
        if slow_functions:
            suggestions["latency"] = [
                "Optimiser les fonctions lentes identifiées",
                "Ajouter de la mise en cache",
                "Paralléliser les traitements",
                "Utiliser des algorithmes plus rapides"
            ]
        
        return suggestions
    
    def auto_optimize(self, issues: List[str]) -> Dict[str, Any]:
        """Optimisation automatique basée sur les problèmes détectés."""
        actions_taken = {}
        
        if "cpu_overload" in issues:
            if not self.optimization_actions["enable_caching"]:
                self.optimization_actions["enable_caching"] = True
                actions_taken["caching"] = "enabled"
            
            if not self.optimization_actions["reduce_fps"]:
                self.optimization_actions["reduce_fps"] = True
                actions_taken["fps_reduction"] = "enabled"
        
        if "memory_overload" in issues:
            if not self.optimization_actions["reduce_quality"]:
                self.optimization_actions["reduce_quality"] = True
                actions_taken["quality_reduction"] = "enabled"
        
        if "gpu_overload" in issues:
            if not self.optimization_actions["use_quantization"]:
                self.optimization_actions["use_quantization"] = True
                actions_taken["quantization"] = "enabled"
        
        return actions_taken
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """État actuel des optimisations."""
        return {
            "thresholds": self.optimization_thresholds.copy(),
            "actions": self.optimization_actions.copy(),
            "current_metrics": self.monitor.collect_current_metrics()
        }


# API publique pour les performances
def start_performance_monitoring() -> None:
    """Démarrer le monitoring de performance global."""
    _performance_monitor.start_monitoring()


def stop_performance_monitoring() -> None:
    """Arrêter le monitoring de performance global."""
    _performance_monitor.stop_monitoring()


def get_current_performance() -> PerformanceMetrics:
    """Récupérer les métriques de performance actuelles."""
    return _performance_monitor.get_current_metrics()


def get_function_performance(function_name: str) -> Optional[TimingMetrics]:
    """Récupérer les performances d'une fonction spécifique."""
    return _function_timer.get_function_stats(function_name)


def get_performance_summary() -> Dict[str, Any]:
    """Résumé complet des performances."""
    return {
        "system_metrics": _performance_monitor.get_average_metrics(5),
        "slowest_functions": [
            {
                "name": metrics.function_name,
                "avg_time": metrics.avg_time,
                "call_count": metrics.call_count,
                "max_time": metrics.max_time
            }
            for metrics in _function_timer.get_slowest_functions(5)
        ],
        "is_overloaded": _performance_monitor.is_system_overloaded()
    }


def reset_performance_stats() -> None:
    """Réinitialiser toutes les statistiques de performance."""
    _function_timer.reset_stats()
    logger.info("Statistiques de performance réinitialisées")