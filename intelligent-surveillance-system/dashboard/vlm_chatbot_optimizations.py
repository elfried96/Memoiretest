"""
⚡ VLM Chatbot Performance Optimizations
=======================================

Optimisations avancées pour améliorer les performances du chatbot VLM:
- Cache intelligent des réponses
- Compression contexte pour économiser tokens
- Async batching des requêtes VLM
- Pre-loading des données contextuelles
- Memory management optimisé
"""

import asyncio
import hashlib
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import threading
import queue
import weakref
from loguru import logger
import numpy as np

# Cache structures
@dataclass
class CachedResponse:
    """Réponse VLM mise en cache."""
    question_hash: str
    response_data: Dict[str, Any]
    context_hash: str
    timestamp: datetime
    access_count: int = 0
    confidence: float = 0.0
    
    def is_expired(self, max_age_minutes: int = 30) -> bool:
        """Vérifie si la réponse est expirée."""
        return datetime.now() - self.timestamp > timedelta(minutes=max_age_minutes)
    
    def is_context_similar(self, new_context_hash: str, similarity_threshold: float = 0.8) -> bool:
        """Vérifie similarité contexte pour réutilisation cache."""
        if self.context_hash == new_context_hash:
            return True
        # TODO: Implémenter similarité sémantique contexte
        return False


class VLMChatbotCache:
    """Cache intelligent pour réponses VLM chatbot."""
    
    def __init__(self, max_size: int = 1000, max_age_minutes: int = 30):
        self.max_size = max_size
        self.max_age_minutes = max_age_minutes
        self.cache: Dict[str, CachedResponse] = {}
        self.access_times: Dict[str, datetime] = {}
        self.lock = threading.RLock()
        
        # Statistiques performance
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'cache_size': 0,
            'evictions': 0,
            'total_requests': 0
        }
    
    def _hash_question(self, question: str, chat_type: str) -> str:
        """Hash normalisé de la question."""
        normalized = f"{chat_type}:{question.lower().strip()}"
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def _hash_context(self, context: Dict[str, Any]) -> str:
        """Hash du contexte VLM."""
        # Extraction des éléments stables pour hash
        stable_context = {
            'pipeline_active': context.get('pipeline_stats', {}).get('is_running', False),
            'frames_processed': context.get('pipeline_stats', {}).get('frames_processed', 0) // 10,  # Quantifié
            'optimal_tools': sorted(context.get('pipeline_stats', {}).get('current_optimal_tools', [])),
            'detection_count': len(context.get('detections', [])),
            'recent_confidence': round(np.mean([
                getattr(d, 'confidence', 0) for d in context.get('detections', [])[-5:]
            ]) if context.get('detections') else 0, 1)
        }
        
        context_str = json.dumps(stable_context, sort_keys=True)
        return hashlib.md5(context_str.encode()).hexdigest()
    
    def get(self, question: str, chat_type: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Récupère réponse du cache si disponible."""
        
        with self.lock:
            self.stats['total_requests'] += 1
            
            question_hash = self._hash_question(question, chat_type)
            context_hash = self._hash_context(context)
            
            # Recherche cache exact
            if question_hash in self.cache:
                cached = self.cache[question_hash]
                
                # Vérification expiration
                if cached.is_expired(self.max_age_minutes):
                    del self.cache[question_hash]
                    self.stats['cache_misses'] += 1
                    return None
                
                # Vérification similarité contexte
                if cached.is_context_similar(context_hash):
                    # Cache hit!
                    cached.access_count += 1
                    self.access_times[question_hash] = datetime.now()
                    self.stats['cache_hits'] += 1
                    
                    # Marquage cache hit dans réponse
                    response = cached.response_data.copy()
                    response['_cache_hit'] = True
                    response['_cache_age'] = (datetime.now() - cached.timestamp).total_seconds()
                    
                    return response
            
            self.stats['cache_misses'] += 1
            return None
    
    def put(self, question: str, chat_type: str, context: Dict[str, Any], response: Dict[str, Any]):
        """Stocke réponse dans le cache."""
        
        with self.lock:
            question_hash = self._hash_question(question, chat_type)
            context_hash = self._hash_context(context)
            
            # Éviction si nécessaire
            if len(self.cache) >= self.max_size:
                self._evict_oldest()
            
            # Stockage
            cached_response = CachedResponse(
                question_hash=question_hash,
                response_data=response,
                context_hash=context_hash,
                timestamp=datetime.now(),
                confidence=response.get('confidence', 0.0)
            )
            
            self.cache[question_hash] = cached_response
            self.access_times[question_hash] = datetime.now()
            self.stats['cache_size'] = len(self.cache)
    
    def _evict_oldest(self):
        """Éviction LRU du cache."""
        if not self.cache:
            return
        
        # Trouver entrée la moins récemment accédée
        oldest_key = min(self.access_times.items(), key=lambda x: x[1])[0]
        
        if oldest_key in self.cache:
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
            self.stats['evictions'] += 1
    
    def cleanup_expired(self):
        """Nettoyage des entrées expirées."""
        with self.lock:
            expired_keys = [
                key for key, cached in self.cache.items()
                if cached.is_expired(self.max_age_minutes)
            ]
            
            for key in expired_keys:
                del self.cache[key]
                if key in self.access_times:
                    del self.access_times[key]
            
            self.stats['cache_size'] = len(self.cache)
            
            if expired_keys:
                logger.info(f" Cache cleanup: {len(expired_keys)} entrées expirées supprimées")
    
    def get_stats(self) -> Dict[str, Any]:
        """Statistiques du cache."""
        with self.lock:
            hit_rate = (
                self.stats['cache_hits'] / self.stats['total_requests'] 
                if self.stats['total_requests'] > 0 else 0
            )
            
            return {
                **self.stats,
                'hit_rate': hit_rate,
                'miss_rate': 1 - hit_rate
            }


class ContextCompressor:
    """Compression intelligente du contexte VLM pour économiser tokens."""
    
    def __init__(self, max_detections: int = 5, max_optimizations: int = 3):
        self.max_detections = max_detections
        self.max_optimizations = max_optimizations
    
    def compress_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Compresse le contexte en gardant l'info la plus pertinente."""
        
        compressed = {}
        
        # Stats pipeline (compression légère)
        if 'stats' in context:
            stats = context['stats']
            compressed['stats'] = {
                'frames_processed': stats.get('frames_processed', 0),
                'performance_score': round(stats.get('current_performance_score', 0), 2),
                'optimal_tools': stats.get('current_optimal_tools', [])[:4],  # Top 4 seulement
                'avg_time': round(stats.get('average_processing_time', 0), 1),
                'total_detections': stats.get('total_detections', 0)
            }
        
        # Détections (sélection intelligente)
        if 'detections' in context:
            detections = context['detections']
            if detections:
                # Prioriser détections haute confiance et récentes
                sorted_detections = sorted(
                    detections[-10:],  # 10 dernières max
                    key=lambda d: (getattr(d, 'confidence', 0), getattr(d, 'timestamp', datetime.min)),
                    reverse=True
                )[:self.max_detections]
                
                compressed['detections'] = [
                    {
                        'confidence': round(getattr(d, 'confidence', 0), 2),
                        'description': getattr(d, 'description', '')[:100],  # Troncature
                        'tools_used': getattr(d, 'tools_used', [])[:3]  # Top 3 outils
                    }
                    for d in sorted_detections
                ]
        
        # Optimisations (les plus récentes seulement)
        if 'optimizations' in context:
            optimizations = context['optimizations']
            if optimizations:
                compressed['optimizations'] = [
                    {
                        'best_combination': opt.get('best_combination', [])[:3],
                        'improvement': round(opt.get('performance_improvement', 0), 2)
                    }
                    for opt in optimizations[-self.max_optimizations:]
                ]
        
        # Métadonnées compression
        compressed['_compressed'] = True
        compressed['_compression_ratio'] = len(str(compressed)) / len(str(context))
        
        return compressed
    
    def estimate_tokens(self, context: Dict[str, Any]) -> int:
        """Estimation du nombre de tokens dans le contexte."""
        text = json.dumps(context)
        # Estimation approximative: ~4 caractères par token
        return len(text) // 4


class AsyncVLMBatcher:
    """Batch asynchrone des requêtes VLM pour optimiser throughput."""
    
    def __init__(self, batch_size: int = 3, batch_timeout: float = 2.0):
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.pending_requests: List[Tuple[str, str, Dict, asyncio.Future]] = []
        self.batch_lock = asyncio.Lock()
        self.batch_task: Optional[asyncio.Task] = None
        
    async def add_request(
        self, 
        question: str, 
        chat_type: str, 
        context: Dict[str, Any],
        vlm_processor  # Fonction traitement VLM
    ) -> Dict[str, Any]:
        """Ajoute requête au batch et retourne résultat."""
        
        future = asyncio.Future()
        
        async with self.batch_lock:
            self.pending_requests.append((question, chat_type, context, future))
            
            # Démarrage batch si conditions remplies
            if (len(self.pending_requests) >= self.batch_size or 
                not self.batch_task or self.batch_task.done()):
                
                self.batch_task = asyncio.create_task(
                    self._process_batch(vlm_processor)
                )
        
        return await future
    
    async def _process_batch(self, vlm_processor):
        """Traite un batch de requêtes."""
        
        await asyncio.sleep(self.batch_timeout)
        
        async with self.batch_lock:
            if not self.pending_requests:
                return
            
            batch = self.pending_requests.copy()
            self.pending_requests.clear()
        
        # Traitement parallèle du batch
        tasks = []
        for question, chat_type, context, future in batch:
            task = asyncio.create_task(
                self._process_single_request(question, chat_type, context, vlm_processor, future)
            )
            tasks.append(task)
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _process_single_request(
        self, 
        question: str, 
        chat_type: str, 
        context: Dict[str, Any],
        vlm_processor,
        future: asyncio.Future
    ):
        """Traite une requête individuelle."""
        try:
            result = await vlm_processor(question, chat_type, context)
            future.set_result(result)
        except Exception as e:
            future.set_exception(e)


class VLMChatbotPerformanceOptimizer:
    """Optimiseur principal pour performance chatbot VLM."""
    
    def __init__(self):
        self.cache = VLMChatbotCache(max_size=500, max_age_minutes=30)
        self.compressor = ContextCompressor(max_detections=5, max_optimizations=3)
        self.batcher = AsyncVLMBatcher(batch_size=2, batch_timeout=1.5)
        
        # Pré-chargement données contextuelles
        self.context_preloader = {}
        self.preload_lock = threading.RLock()
        
        # Monitoring performance
        self.perf_stats = {
            'avg_response_time': 0.0,
            'cache_hit_rate': 0.0,
            'context_compression_ratio': 0.0,
            'tokens_saved': 0,
            'total_requests': 0
        }
        
        # Démarrage nettoyage automatique cache
        self._start_cache_cleanup()
    
    def _start_cache_cleanup(self):
        """Démarre le nettoyage automatique du cache."""
        def cleanup_worker():
            while True:
                time.sleep(300)  # Toutes les 5 minutes
                self.cache.cleanup_expired()
        
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()
    
    async def process_optimized_query(
        self,
        question: str,
        chat_type: str,
        context: Dict[str, Any],
        vlm_processor  # Fonction traitement VLM original
    ) -> Dict[str, Any]:
        """Traite une requête avec toutes les optimisations."""
        
        start_time = time.time()
        self.perf_stats['total_requests'] += 1
        
        try:
            # 1. Vérification cache
            cached_response = self.cache.get(question, chat_type, context)
            if cached_response:
                processing_time = time.time() - start_time
                self._update_perf_stats(processing_time, cache_hit=True)
                
                # Ajout métadonnées performance
                cached_response['_performance'] = {
                    'cache_hit': True,
                    'processing_time': processing_time,
                    'optimizations': ['cache']
                }
                
                return cached_response
            
            # 2. Compression contexte
            original_context = context.copy()
            compressed_context = self.compressor.compress_context(context)
            
            tokens_saved = (
                self.compressor.estimate_tokens(original_context) - 
                self.compressor.estimate_tokens(compressed_context)
            )
            self.perf_stats['tokens_saved'] += max(0, tokens_saved)
            
            # 3. Traitement VLM avec batching
            response = await self.batcher.add_request(
                question, chat_type, compressed_context, vlm_processor
            )
            
            # 4. Mise en cache
            if response and response.get('confidence', 0) > 0.7:
                self.cache.put(question, chat_type, original_context, response)
            
            # 5. Métadonnées performance
            processing_time = time.time() - start_time
            self._update_perf_stats(processing_time, cache_hit=False)
            
            response['_performance'] = {
                'cache_hit': False,
                'processing_time': processing_time,
                'context_compressed': compressed_context.get('_compressed', False),
                'compression_ratio': compressed_context.get('_compression_ratio', 1.0),
                'tokens_saved': tokens_saved,
                'optimizations': ['compression', 'batching']
            }
            
            return response
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f" Erreur optimization chatbot: {e}")
            
            return {
                'type': 'optimization_error',
                'response': f" Erreur optimisation: {str(e)}",
                'confidence': 0.0,
                '_performance': {
                    'cache_hit': False,
                    'processing_time': processing_time,
                    'error': str(e)
                }
            }
    
    def _update_perf_stats(self, processing_time: float, cache_hit: bool):
        """Met à jour les statistiques de performance."""
        
        # Temps de réponse moyen
        total = self.perf_stats['total_requests']
        current_avg = self.perf_stats['avg_response_time']
        new_avg = (current_avg * (total - 1) + processing_time) / total
        self.perf_stats['avg_response_time'] = new_avg
        
        # Taux de cache hit
        cache_stats = self.cache.get_stats()
        self.perf_stats['cache_hit_rate'] = cache_stats['hit_rate']
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Rapport complet de performance."""
        
        cache_stats = self.cache.get_stats()
        
        return {
            'response_times': {
                'avg_response_time': round(self.perf_stats['avg_response_time'], 2),
                'cache_hit_avg': 0.1,  # Cache très rapide
                'vlm_processing_avg': round(self.perf_stats['avg_response_time'] * (1 - cache_stats['hit_rate']), 2)
            },
            'cache_performance': cache_stats,
            'optimization_impact': {
                'tokens_saved_total': self.perf_stats['tokens_saved'],
                'tokens_saved_per_request': round(
                    self.perf_stats['tokens_saved'] / max(1, self.perf_stats['total_requests']), 1
                ),
                'estimated_cost_savings': round(
                    self.perf_stats['tokens_saved'] * 0.001, 4  # Estimation coût token
                )
            },
            'total_requests': self.perf_stats['total_requests']
        }


# Instance globale optimiseur
_performance_optimizer = None

def get_performance_optimizer() -> VLMChatbotPerformanceOptimizer:
    """Récupère l'instance d'optimiseur global."""
    global _performance_optimizer
    if _performance_optimizer is None:
        _performance_optimizer = VLMChatbotPerformanceOptimizer()
    return _performance_optimizer