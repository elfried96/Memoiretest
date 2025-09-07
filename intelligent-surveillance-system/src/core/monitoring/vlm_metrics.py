"""
Collecteur de métriques spécifiques aux modèles VLM.
"""

import time
import logging
from typing import Dict, Optional, List
from collections import deque, defaultdict

from .performance_monitor import MetricsCollector, PerformanceMetric

logger = logging.getLogger(__name__)


class VLMMetricsCollector(MetricsCollector):
    """
    Collecteur de métriques pour les modèles VLM.
    
    Collecte:
    - Temps de traitement par requête
    - Débit (requêtes par seconde)
    - Taux d'erreur et de succès
    - Utilisation des modèles (switches, fallbacks)
    - Métriques de qualité des réponses
    - Cache hit/miss rates
    """
    
    def __init__(self, window_size: int = 100):
        """
        Initialise le collecteur de métriques VLM.
        
        Args:
            window_size: Taille de la fenêtre glissante pour les statistiques
        """
        self.window_size = window_size
        
        # Fenêtres glissantes pour statistiques
        self.processing_times: deque = deque(maxlen=window_size)
        self.request_timestamps: deque = deque(maxlen=window_size)
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.model_usage: Dict[str, int] = defaultdict(int)
        self.success_count = 0
        self.total_requests = 0
        
        # Métriques de cache
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Métriques de qualité
        self.confidence_scores: deque = deque(maxlen=window_size)
        self.tool_call_counts: Dict[str, int] = defaultdict(int)
        
        # Métriques de modèles spécifiques
        self.model_switch_count = 0
        self.fallback_usage_count = 0
        
        logger.info("📊 Collecteur métriques VLM initialisé")
    
    def get_collector_name(self) -> str:
        """Nom du collecteur."""
        return "vlm_metrics"
    
    async def collect_metrics(self) -> Dict[str, PerformanceMetric]:
        """Collecte toutes les métriques VLM."""
        timestamp = time.time()
        metrics = {}
        
        # Métriques de performance
        perf_metrics = self._collect_performance_metrics(timestamp)
        metrics.update(perf_metrics)
        
        # Métriques de fiabilité
        reliability_metrics = self._collect_reliability_metrics(timestamp)
        metrics.update(reliability_metrics)
        
        # Métriques de qualité
        quality_metrics = self._collect_quality_metrics(timestamp)
        metrics.update(quality_metrics)
        
        # Métriques d'utilisation
        usage_metrics = self._collect_usage_metrics(timestamp)
        metrics.update(usage_metrics)
        
        return metrics
    
    def _collect_performance_metrics(self, timestamp: float) -> Dict[str, PerformanceMetric]:
        """Collecte les métriques de performance."""
        metrics = {}
        
        # Temps de traitement moyen
        if self.processing_times:
            avg_processing_time = sum(self.processing_times) / len(self.processing_times)
            metrics["vlm_avg_processing_time"] = PerformanceMetric(
                name="vlm_avg_processing_time",
                value=avg_processing_time,
                timestamp=timestamp,
                unit="s",
                category="vlm_performance"
            )
            
            # Min/Max temps de traitement
            metrics["vlm_min_processing_time"] = PerformanceMetric(
                name="vlm_min_processing_time",
                value=min(self.processing_times),
                timestamp=timestamp,
                unit="s",
                category="vlm_performance"
            )
            
            metrics["vlm_max_processing_time"] = PerformanceMetric(
                name="vlm_max_processing_time",
                value=max(self.processing_times),
                timestamp=timestamp,
                unit="s",
                category="vlm_performance"
            )
        
        # Débit (requêtes par seconde)
        if len(self.request_timestamps) >= 2:
            time_span = self.request_timestamps[-1] - self.request_timestamps[0]
            if time_span > 0:
                throughput = len(self.request_timestamps) / time_span
                metrics["vlm_throughput"] = PerformanceMetric(
                    name="vlm_throughput",
                    value=throughput,
                    timestamp=timestamp,
                    unit="req/s",
                    category="vlm_performance"
                )
        
        return metrics
    
    def _collect_reliability_metrics(self, timestamp: float) -> Dict[str, PerformanceMetric]:
        """Collecte les métriques de fiabilité."""
        metrics = {}
        
        # Taux de succès
        if self.total_requests > 0:
            success_rate = (self.success_count / self.total_requests) * 100
            metrics["vlm_success_rate"] = PerformanceMetric(
                name="vlm_success_rate",
                value=success_rate,
                timestamp=timestamp,
                unit="%",
                category="vlm_reliability"
            )
            
            # Taux d'erreur global
            error_count = self.total_requests - self.success_count
            error_rate = (error_count / self.total_requests) * 100
            metrics["vlm_error_rate"] = PerformanceMetric(
                name="vlm_error_rate",
                value=error_rate,
                timestamp=timestamp,
                unit="%",
                category="vlm_reliability"
            )
        
        # Taux d'erreur par type
        for error_type, count in self.error_counts.items():
            if self.total_requests > 0:
                rate = (count / self.total_requests) * 100
                metrics[f"vlm_error_rate_{error_type}"] = PerformanceMetric(
                    name=f"vlm_error_rate_{error_type}",
                    value=rate,
                    timestamp=timestamp,
                    unit="%",
                    category="vlm_reliability"
                )
        
        # Métriques de cache
        total_cache_requests = self.cache_hits + self.cache_misses
        if total_cache_requests > 0:
            cache_hit_rate = (self.cache_hits / total_cache_requests) * 100
            metrics["vlm_cache_hit_rate"] = PerformanceMetric(
                name="vlm_cache_hit_rate",
                value=cache_hit_rate,
                timestamp=timestamp,
                unit="%",
                category="vlm_reliability"
            )
        
        return metrics
    
    def _collect_quality_metrics(self, timestamp: float) -> Dict[str, PerformanceMetric]:
        """Collecte les métriques de qualité."""
        metrics = {}
        
        # Score de confiance moyen
        if self.confidence_scores:
            avg_confidence = sum(self.confidence_scores) / len(self.confidence_scores)
            metrics["vlm_avg_confidence"] = PerformanceMetric(
                name="vlm_avg_confidence",
                value=avg_confidence,
                timestamp=timestamp,
                unit="",
                category="vlm_quality"
            )
            
            # Confidence minimale/maximale
            metrics["vlm_min_confidence"] = PerformanceMetric(
                name="vlm_min_confidence",
                value=min(self.confidence_scores),
                timestamp=timestamp,
                unit="",
                category="vlm_quality"
            )
            
            metrics["vlm_max_confidence"] = PerformanceMetric(
                name="vlm_max_confidence",
                value=max(self.confidence_scores),
                timestamp=timestamp,
                unit="",
                category="vlm_quality"
            )
        
        # Utilisation des outils (tool calling)
        total_tool_calls = sum(self.tool_call_counts.values())
        if total_tool_calls > 0:
            metrics["vlm_total_tool_calls"] = PerformanceMetric(
                name="vlm_total_tool_calls",
                value=total_tool_calls,
                timestamp=timestamp,
                unit="calls",
                category="vlm_quality"
            )
            
            # Répartition par outil
            for tool_name, count in self.tool_call_counts.items():
                rate = (count / total_tool_calls) * 100
                metrics[f"vlm_tool_usage_{tool_name}"] = PerformanceMetric(
                    name=f"vlm_tool_usage_{tool_name}",
                    value=rate,
                    timestamp=timestamp,
                    unit="%",
                    category="vlm_quality"
                )
        
        return metrics
    
    def _collect_usage_metrics(self, timestamp: float) -> Dict[str, PerformanceMetric]:
        """Collecte les métriques d'utilisation."""
        metrics = {}
        
        # Utilisation par modèle
        total_model_usage = sum(self.model_usage.values())
        if total_model_usage > 0:
            for model_name, count in self.model_usage.items():
                rate = (count / total_model_usage) * 100
                metrics[f"vlm_model_usage_{model_name}"] = PerformanceMetric(
                    name=f"vlm_model_usage_{model_name}",
                    value=rate,
                    timestamp=timestamp,
                    unit="%",
                    category="vlm_usage"
                )
        
        # Statistiques de changement de modèle
        metrics["vlm_model_switches"] = PerformanceMetric(
            name="vlm_model_switches",
            value=self.model_switch_count,
            timestamp=timestamp,
            unit="count",
            category="vlm_usage"
        )
        
        metrics["vlm_fallback_usage"] = PerformanceMetric(
            name="vlm_fallback_usage",
            value=self.fallback_usage_count,
            timestamp=timestamp,
            unit="count",
            category="vlm_usage"
        )
        
        # Requêtes totales
        metrics["vlm_total_requests"] = PerformanceMetric(
            name="vlm_total_requests",
            value=self.total_requests,
            timestamp=timestamp,
            unit="count",
            category="vlm_usage"
        )
        
        return metrics
    
    # Méthodes pour enregistrer les événements
    
    def record_request(
        self,
        processing_time: float,
        success: bool,
        model_name: str,
        confidence_score: Optional[float] = None,
        error_type: Optional[str] = None,
        tools_used: Optional[List[str]] = None
    ) -> None:
        """
        Enregistre une requête VLM.
        
        Args:
            processing_time: Temps de traitement en secondes
            success: Si la requête a réussi
            model_name: Nom du modèle utilisé
            confidence_score: Score de confiance (0-1)
            error_type: Type d'erreur si échec
            tools_used: Liste des outils utilisés
        """
        timestamp = time.time()
        
        # Enregistrement des métriques de base
        self.processing_times.append(processing_time)
        self.request_timestamps.append(timestamp)
        self.total_requests += 1
        self.model_usage[model_name] += 1
        
        if success:
            self.success_count += 1
        elif error_type:
            self.error_counts[error_type] += 1
        
        # Score de confiance
        if confidence_score is not None:
            self.confidence_scores.append(confidence_score)
        
        # Utilisation des outils
        if tools_used:
            for tool in tools_used:
                self.tool_call_counts[tool] += 1
    
    def record_model_switch(self, from_model: str, to_model: str) -> None:
        """Enregistre un changement de modèle."""
        self.model_switch_count += 1
        logger.debug(f"🔄 Switch modèle: {from_model} → {to_model}")
    
    def record_fallback_usage(self, primary_model: str, fallback_model: str) -> None:
        """Enregistre l'utilisation d'un modèle de fallback."""
        self.fallback_usage_count += 1
        logger.debug(f"⚠️ Fallback: {primary_model} → {fallback_model}")
    
    def record_cache_hit(self) -> None:
        """Enregistre un cache hit."""
        self.cache_hits += 1
    
    def record_cache_miss(self) -> None:
        """Enregistre un cache miss."""
        self.cache_misses += 1
    
    def reset_counters(self) -> None:
        """Remet à zéro tous les compteurs."""
        self.processing_times.clear()
        self.request_timestamps.clear()
        self.error_counts.clear()
        self.model_usage.clear()
        self.confidence_scores.clear()
        self.tool_call_counts.clear()
        
        self.success_count = 0
        self.total_requests = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.model_switch_count = 0
        self.fallback_usage_count = 0
        
        logger.info("🔄 Compteurs VLM remis à zéro")
    
    def get_summary_stats(self) -> Dict[str, any]:
        """Retourne un résumé des statistiques."""
        return {
            "total_requests": self.total_requests,
            "success_rate": (self.success_count / self.total_requests * 100) if self.total_requests > 0 else 0,
            "avg_processing_time": sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0,
            "model_switches": self.model_switch_count,
            "fallback_usage": self.fallback_usage_count,
            "cache_hit_rate": (self.cache_hits / (self.cache_hits + self.cache_misses) * 100) if (self.cache_hits + self.cache_misses) > 0 else 0,
            "avg_confidence": sum(self.confidence_scores) / len(self.confidence_scores) if self.confidence_scores else 0,
            "top_models": dict(sorted(self.model_usage.items(), key=lambda x: x[1], reverse=True)[:3]),
            "top_tools": dict(sorted(self.tool_call_counts.items(), key=lambda x: x[1], reverse=True)[:5])
        }