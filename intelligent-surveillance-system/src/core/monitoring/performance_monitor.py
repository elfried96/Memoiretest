"""
Moniteur de performance centralisé pour le système de surveillance.
"""

import time
import logging
import asyncio
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import threading
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Métrique de performance individuelle."""
    name: str
    value: float
    timestamp: float
    unit: str = ""
    category: str = "general"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class PerformanceSnapshot:
    """Instantané des performances à un moment donné."""
    timestamp: float
    metrics: Dict[str, PerformanceMetric]
    system_info: Dict[str, Any] = field(default_factory=dict)
    
    def get_metric(self, name: str) -> Optional[PerformanceMetric]:
        """Récupère une métrique par nom."""
        return self.metrics.get(name)
    
    def get_metrics_by_category(self, category: str) -> Dict[str, PerformanceMetric]:
        """Récupère toutes les métriques d'une catégorie."""
        return {
            name: metric for name, metric in self.metrics.items()
            if metric.category == category
        }


class MetricsCollector(ABC):
    """Interface pour les collecteurs de métriques."""
    
    @abstractmethod
    async def collect_metrics(self) -> Dict[str, PerformanceMetric]:
        """Collecte les métriques et retourne un dictionnaire."""
        pass
    
    @abstractmethod
    def get_collector_name(self) -> str:
        """Retourne le nom du collecteur."""
        pass


class PerformanceMonitor:
    """
    Moniteur de performance centralisé pour le système de surveillance.
    
    Ce moniteur collecte, agrège et analyse les métriques de performance
    de tous les composants du système en temps réel.
    
    Features:
    - Collection automatique de métriques multi-sources
    - Historique des performances avec fenêtres glissantes
    - Détection d'anomalies et alertes
    - Export de métriques pour monitoring externe
    - Threading pour collection non-bloquante
    
    Example:
        >>> monitor = PerformanceMonitor(interval=10.0)
        >>> monitor.add_collector(VLMMetricsCollector())
        >>> monitor.start()
        >>> # ... système en fonctionnement ...
        >>> snapshot = monitor.get_current_snapshot()
        >>> monitor.stop()
    """
    
    def __init__(
        self,
        collection_interval: float = 5.0,
        history_size: int = 1000,
        enable_alerts: bool = True
    ):
        """
        Initialise le moniteur de performance.
        
        Args:
            collection_interval: Intervalle de collection en secondes
            history_size: Taille de l'historique à conserver
            enable_alerts: Active les alertes de performance
        """
        self.collection_interval = collection_interval
        self.history_size = history_size
        self.enable_alerts = enable_alerts
        
        # Collecteurs de métriques
        self.collectors: List[MetricsCollector] = []
        
        # Historique des performances
        self.metrics_history: deque = deque(maxlen=history_size)
        self.current_snapshot: Optional[PerformanceSnapshot] = None
        
        # Statistiques agrégées
        self.aggregated_stats: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        
        # Contrôle du threading
        self._running = False
        self._collection_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        
        # Callbacks pour alertes
        self.alert_callbacks: List[Callable[[str, Dict[str, Any]], None]] = []
        
        # Seuils d'alerte par défaut
        self.alert_thresholds = {
            "cpu_usage": {"warning": 80.0, "critical": 95.0},
            "memory_usage": {"warning": 80.0, "critical": 95.0},
            "gpu_memory_usage": {"warning": 85.0, "critical": 98.0},
            "processing_time": {"warning": 2.0, "critical": 5.0},
            "frame_drop_rate": {"warning": 10.0, "critical": 25.0}
        }
        
        logger.info(f"✅ Moniteur de performance initialisé (intervalle: {collection_interval}s)")
    
    def add_collector(self, collector: MetricsCollector) -> None:
        """
        Ajoute un collecteur de métriques.
        
        Args:
            collector: Instance de collecteur de métriques
        """
        with self._lock:
            self.collectors.append(collector)
            logger.info(f"📊 Collecteur ajouté: {collector.get_collector_name()}")
    
    def remove_collector(self, collector_name: str) -> bool:
        """
        Supprime un collecteur par nom.
        
        Args:
            collector_name: Nom du collecteur à supprimer
            
        Returns:
            True si le collecteur a été trouvé et supprimé
        """
        with self._lock:
            for i, collector in enumerate(self.collectors):
                if collector.get_collector_name() == collector_name:
                    del self.collectors[i]
                    logger.info(f"🗑️ Collecteur supprimé: {collector_name}")
                    return True
            return False
    
    def start(self) -> None:
        """Démarre la collection de métriques en arrière-plan."""
        if self._running:
            logger.warning("⚠️ Moniteur déjà en cours d'exécution")
            return
        
        self._running = True
        self._collection_thread = threading.Thread(
            target=self._collection_loop,
            daemon=True,
            name="PerformanceMonitor"
        )
        self._collection_thread.start()
        
        logger.info("🚀 Moniteur de performance démarré")
    
    def stop(self) -> None:
        """Arrête la collection de métriques."""
        if not self._running:
            return
        
        self._running = False
        if self._collection_thread:
            self._collection_thread.join(timeout=self.collection_interval + 1.0)
        
        logger.info("🛑 Moniteur de performance arrêté")
    
    def _collection_loop(self) -> None:
        """Boucle principale de collection de métriques."""
        while self._running:
            try:
                # Collection asynchrone des métriques
                snapshot = asyncio.run(self._collect_all_metrics())
                
                if snapshot:
                    with self._lock:
                        # Mise à jour de l'historique
                        self.metrics_history.append(snapshot)
                        self.current_snapshot = snapshot
                        
                        # Mise à jour des statistiques agrégées
                        self._update_aggregated_stats(snapshot)
                        
                        # Vérification des alertes
                        if self.enable_alerts:
                            self._check_alerts(snapshot)
                
                # Attente avant la prochaine collection
                time.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"❌ Erreur collection métriques: {e}")
                time.sleep(1.0)  # Pause courte avant de réessayer
    
    async def _collect_all_metrics(self) -> Optional[PerformanceSnapshot]:
        """Collecte toutes les métriques de tous les collecteurs."""
        timestamp = time.time()
        all_metrics = {}
        
        # Collection depuis tous les collecteurs
        for collector in self.collectors:
            try:
                metrics = await collector.collect_metrics()
                all_metrics.update(metrics)
            except Exception as e:
                logger.warning(f"⚠️ Erreur collecteur {collector.get_collector_name()}: {e}")
        
        if not all_metrics:
            return None
        
        return PerformanceSnapshot(
            timestamp=timestamp,
            metrics=all_metrics,
            system_info=self._get_system_info()
        )
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Récupère les informations système de base."""
        return {
            "timestamp": time.time(),
            "active_collectors": len(self.collectors),
            "history_size": len(self.metrics_history)
        }
    
    def _update_aggregated_stats(self, snapshot: PerformanceSnapshot) -> None:
        """Met à jour les statistiques agrégées."""
        for name, metric in snapshot.metrics.items():
            stats = self.aggregated_stats[name]
            
            # Mise à jour des statistiques
            stats["count"] += 1
            stats["sum"] += metric.value
            stats["average"] = stats["sum"] / stats["count"]
            
            # Min/Max
            if "min" not in stats or metric.value < stats["min"]:
                stats["min"] = metric.value
            if "max" not in stats or metric.value > stats["max"]:
                stats["max"] = metric.value
            
            # Dernière valeur
            stats["last"] = metric.value
            stats["last_timestamp"] = metric.timestamp
    
    def _check_alerts(self, snapshot: PerformanceSnapshot) -> None:
        """Vérifie les seuils d'alerte pour les métriques."""
        for name, metric in snapshot.metrics.items():
            if name in self.alert_thresholds:
                thresholds = self.alert_thresholds[name]
                
                alert_level = None
                if metric.value >= thresholds.get("critical", float('inf')):
                    alert_level = "critical"
                elif metric.value >= thresholds.get("warning", float('inf')):
                    alert_level = "warning"
                
                if alert_level:
                    alert_data = {
                        "metric": name,
                        "value": metric.value,
                        "threshold": thresholds[alert_level],
                        "level": alert_level,
                        "timestamp": metric.timestamp,
                        "unit": metric.unit
                    }
                    
                    self._trigger_alert(alert_level, alert_data)
    
    def _trigger_alert(self, level: str, alert_data: Dict[str, Any]) -> None:
        """Déclenche une alerte."""
        message = (
            f"🚨 Alerte {level.upper()}: "
            f"{alert_data['metric']} = {alert_data['value']}{alert_data['unit']} "
            f"(seuil: {alert_data['threshold']})"
        )
        
        logger.warning(message)
        
        # Appel des callbacks d'alerte
        for callback in self.alert_callbacks:
            try:
                callback(level, alert_data)
            except Exception as e:
                logger.error(f"❌ Erreur callback alerte: {e}")
    
    def get_current_snapshot(self) -> Optional[PerformanceSnapshot]:
        """Retourne l'instantané de performances le plus récent."""
        with self._lock:
            return self.current_snapshot
    
    def get_metrics_history(self, limit: Optional[int] = None) -> List[PerformanceSnapshot]:
        """
        Retourne l'historique des métriques.
        
        Args:
            limit: Nombre maximum d'entrées à retourner
            
        Returns:
            Liste des instantanés de performance
        """
        with self._lock:
            history = list(self.metrics_history)
            if limit:
                return history[-limit:]
            return history
    
    def get_aggregated_stats(self) -> Dict[str, Dict[str, float]]:
        """Retourne les statistiques agrégées."""
        with self._lock:
            return dict(self.aggregated_stats)
    
    def get_metric_trend(
        self, 
        metric_name: str, 
        duration: float = 300.0
    ) -> List[float]:
        """
        Retourne la tendance d'une métrique sur une durée donnée.
        
        Args:
            metric_name: Nom de la métrique
            duration: Durée en secondes
            
        Returns:
            Liste des valeurs de la métrique
        """
        with self._lock:
            cutoff_time = time.time() - duration
            values = []
            
            for snapshot in reversed(self.metrics_history):
                if snapshot.timestamp < cutoff_time:
                    break
                
                metric = snapshot.get_metric(metric_name)
                if metric:
                    values.insert(0, metric.value)
            
            return values
    
    def add_alert_callback(self, callback: Callable[[str, Dict[str, Any]], None]) -> None:
        """Ajoute un callback pour les alertes."""
        self.alert_callbacks.append(callback)
    
    def set_alert_threshold(
        self, 
        metric_name: str, 
        warning: Optional[float] = None,
        critical: Optional[float] = None
    ) -> None:
        """
        Configure les seuils d'alerte pour une métrique.
        
        Args:
            metric_name: Nom de la métrique
            warning: Seuil d'avertissement
            critical: Seuil critique
        """
        if metric_name not in self.alert_thresholds:
            self.alert_thresholds[metric_name] = {}
        
        if warning is not None:
            self.alert_thresholds[metric_name]["warning"] = warning
        if critical is not None:
            self.alert_thresholds[metric_name]["critical"] = critical
        
        logger.info(f"📐 Seuils configurés pour {metric_name}: {self.alert_thresholds[metric_name]}")
    
    def export_metrics(
        self, 
        format: str = "json", 
        duration: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Exporte les métriques dans un format spécifique.
        
        Args:
            format: Format d'export ('json', 'prometheus')
            duration: Durée à exporter (None = tout l'historique)
            
        Returns:
            Données exportées dans le format demandé
        """
        with self._lock:
            if duration:
                cutoff_time = time.time() - duration
                snapshots = [
                    s for s in self.metrics_history 
                    if s.timestamp >= cutoff_time
                ]
            else:
                snapshots = list(self.metrics_history)
            
            if format == "json":
                return {
                    "snapshots": [
                        {
                            "timestamp": s.timestamp,
                            "metrics": {
                                name: {
                                    "value": m.value,
                                    "unit": m.unit,
                                    "category": m.category
                                }
                                for name, m in s.metrics.items()
                            }
                        }
                        for s in snapshots
                    ],
                    "aggregated_stats": dict(self.aggregated_stats),
                    "collection_info": {
                        "interval": self.collection_interval,
                        "total_snapshots": len(snapshots),
                        "start_time": snapshots[0].timestamp if snapshots else None,
                        "end_time": snapshots[-1].timestamp if snapshots else None
                    }
                }
            
            elif format == "prometheus":
                # Format Prometheus simple
                lines = []
                if snapshots:
                    latest = snapshots[-1]
                    for name, metric in latest.metrics.items():
                        lines.append(f"surveillance_{name} {metric.value}")
                
                return {"prometheus_text": "\n".join(lines)}
            
            else:
                raise ValueError(f"Format d'export non supporté: {format}")
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()