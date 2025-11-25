"""
🧠 Orchestrateur Adaptatif avec Optimisation Automatique des Outils
===================================================================

Cet orchestrateur utilise les résultats du benchmark pour adapter dynamiquement
la sélection d'outils en fonction des performances observées et du contexte.

Features:
- Sélection adaptative basée sur l'historique de performance
- Apprentissage continu des meilleures combinaisons
- Optimisation en temps réel selon le contexte
- Gestion intelligente du tool calling
"""

import asyncio
import time
import json
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict, deque
import numpy as np

from loguru import logger

from ..types import AnalysisRequest, AnalysisResponse, SuspicionLevel, ActionType
from ..vlm.dynamic_model import DynamicVisionLanguageModel
from .vlm_orchestrator import ModernVLMOrchestrator, OrchestrationConfig, OrchestrationMode
from ...testing.tool_optimization_benchmark import ToolOptimizationBenchmark


@dataclass
class ContextPattern:
    """Pattern de contexte pour optimisation adaptative."""
    scenario_type: str  # "normal", "suspicious", "crowded", etc.
    time_period: str   # "morning", "afternoon", "evening", "night"
    person_count: int  # Nombre de personnes détectées
    location_type: str # "entrance", "electronics", "checkout", etc.
    optimal_tools: List[str]  # Outils optimaux pour ce pattern
    performance_score: float  # Score de performance historique
    usage_count: int = 0
    last_updated: datetime = None


@dataclass
class ToolPerformanceHistory:
    """Historique de performance d'un outil."""
    tool_name: str
    success_rate: float = 0.0
    avg_response_time: float = 0.0
    context_effectiveness: Dict[str, float] = None  # Efficacité par contexte
    recent_scores: deque = None  # Scores récents (sliding window)
    last_performance_update: datetime = None
    
    def __post_init__(self):
        if self.context_effectiveness is None:
            self.context_effectiveness = {}
        if self.recent_scores is None:
            self.recent_scores = deque(maxlen=50)  # 50 derniers scores


class AdaptiveVLMOrchestrator(ModernVLMOrchestrator):
    """
    Orchestrateur VLM adaptatif avec optimisation automatique.
    
    Extends ModernVLMOrchestrator with:
    - Apprentissage continu des performances
    - Sélection adaptative des outils
    - Optimisation basée sur le contexte
    - Gestion intelligente des coûts computationnels
    """
    
    def __init__(
        self,
        vlm_model_name: str = "qwen2.5-vl-7b-instruct",
        config: OrchestrationConfig = None,
        enable_adaptive_learning: bool = True,
        optimization_data_path: Optional[Path] = None
    ):
        super().__init__(vlm_model_name, config)
        
        self.enable_adaptive_learning = enable_adaptive_learning
        self.optimization_data_path = optimization_data_path or Path("data/orchestration/adaptive")
        
        # Gestion adaptative
        self.context_patterns: Dict[str, ContextPattern] = {}
        self.tool_performance_history: Dict[str, ToolPerformanceHistory] = {}
        self.current_optimal_tools: List[str] = []
        
        # Historique d'exécution pour apprentissage
        self.execution_history = deque(maxlen=1000)
        self.performance_window = deque(maxlen=100)  # Fenêtre glissante
        
        # Configuration adaptative
        self.min_performance_threshold = 0.6
        self.learning_rate = 0.1
        self.context_similarity_threshold = 0.8
        self.reoptimization_interval = timedelta(hours=6)  # Re-optimisation toutes les 6h
        self.last_optimization = datetime.now() - timedelta(days=1)  # Forcer première optim
        
        # Cache de sélection d'outils
        self._tool_selection_cache: Dict[str, Tuple[List[str], datetime]] = {}
        self._cache_ttl = timedelta(minutes=30)
        
        self._initialize_adaptive_system()
        
        logger.info("AdaptiveVLMOrchestrator initialisé avec apprentissage adaptatif")
    
    def _initialize_adaptive_system(self) -> None:
        """Initialisation du système adaptatif."""
        
        # Chargement des données d'optimisation existantes
        self._load_optimization_data()
        
        # Initialisation des outils par défaut si pas de données
        if not self.current_optimal_tools:
            self.current_optimal_tools = [
                "dino_features",
                "pose_estimator",
                "multimodal_fusion",
                "adversarial_detector"
            ]
            logger.info("Utilisation des outils par défaut en attendant l'optimisation")
        
        # Initialisation de l'historique des outils
        all_available_tools = [
            "sam2_segmentator", "dino_features", "pose_estimator",
            "trajectory_analyzer", "multimodal_fusion", "temporal_transformer",
            "adversarial_detector", "domain_adapter"
        ]
        
        for tool in all_available_tools:
            if tool not in self.tool_performance_history:
                self.tool_performance_history[tool] = ToolPerformanceHistory(tool_name=tool)
    
    def _load_optimization_data(self) -> None:
        """Chargement des données d'optimisation sauvegardées."""
        
        try:
            # Chargement des patterns de contexte
            patterns_file = self.optimization_data_path / "context_patterns.json"
            if patterns_file.exists():
                with open(patterns_file, 'r') as f:
                    patterns_data = json.load(f)
                
                for pattern_id, data in patterns_data.items():
                    self.context_patterns[pattern_id] = ContextPattern(
                        scenario_type=data["scenario_type"],
                        time_period=data["time_period"],
                        person_count=data["person_count"],
                        location_type=data["location_type"],
                        optimal_tools=data["optimal_tools"],
                        performance_score=data["performance_score"],
                        usage_count=data["usage_count"],
                        last_updated=datetime.fromisoformat(data["last_updated"])
                    )
            
            # Chargement des outils optimaux actuels
            optimal_tools_file = self.optimization_data_path / "current_optimal_tools.json"
            if optimal_tools_file.exists():
                with open(optimal_tools_file, 'r') as f:
                    data = json.load(f)
                    self.current_optimal_tools = data.get("tools", [])
                    
            logger.info(f"✅ Données d'optimisation chargées: {len(self.context_patterns)} patterns")
            
        except Exception as e:
            logger.warning(f"Erreur chargement données d'optimisation: {e}")
    
    def _save_optimization_data(self) -> None:
        """Sauvegarde des données d'optimisation."""
        
        try:
            self.optimization_data_path.mkdir(parents=True, exist_ok=True)
            
            # Sauvegarde des patterns de contexte
            patterns_data = {}
            for pattern_id, pattern in self.context_patterns.items():
                pattern_dict = asdict(pattern)
                if pattern_dict["last_updated"]:
                    pattern_dict["last_updated"] = pattern.last_updated.isoformat()
                patterns_data[pattern_id] = pattern_dict
            
            with open(self.optimization_data_path / "context_patterns.json", 'w') as f:
                json.dump(patterns_data, f, indent=2, default=str)
            
            # Sauvegarde des outils optimaux actuels
            with open(self.optimization_data_path / "current_optimal_tools.json", 'w') as f:
                json.dump({
                    "tools": self.current_optimal_tools,
                    "last_updated": datetime.now().isoformat(),
                    "performance_score": np.mean(list(self.performance_window)) if self.performance_window else 0.0
                }, f, indent=2)
                
            logger.debug("Données d'optimisation sauvegardées")
            
        except Exception as e:
            logger.error(f"Erreur sauvegarde données d'optimisation: {e}")
    
    async def analyze(self, request: AnalysisRequest) -> AnalysisResponse:
        """Méthode d'analyse principale attendue par la pipeline."""
        
        # Conversion de AnalysisRequest vers format compatible
        frame_data = request.frame_data
        context = request.context
        detections = []  # Pas de détections dans AnalysisRequest standard
        
        return await self.analyze_surveillance_frame(frame_data, detections, context)
    
    async def analyze_surveillance_frame(
        self,
        frame_data: str,
        detections: List = None,
        context: Dict[str, Any] = None
    ) -> AnalysisResponse:
        """Analyse adaptative avec sélection optimisée des outils."""
        
        start_time = time.time()
        
        # NOUVEAU: Étape 1 - Analyse préliminaire rapide
        is_scene_interesting = await self.vlm_model_instance.is_scene_interesting(frame_data)
        if not is_scene_interesting:
            return self._create_default_response("Scène statique, aucune analyse requise.")

        # 1. Analyse du contexte pour sélection adaptative
        context_signature = self._analyze_context(context or {}, detections or [])
        
        # 2. Sélection adaptative des outils
        selected_tools = await self._adaptive_tool_selection(context_signature)

        # NOUVEAU: Si aucun outil n'est sélectionné, ne rien faire
        if not selected_tools:
            return self._create_default_response("Aucun outil pertinent pour ce contexte.")
        
        # 3. Override temporaire de la sélection d'outils
        original_select_tools = self._select_tools_for_mode
        self._select_tools_for_mode = lambda: selected_tools
        
        try:
            # 4. Exécution de l'analyse avec les outils sélectionnés
            analysis_result = await super().analyze_surveillance_frame(
                frame_data, detections, context
            )
            
            # 5. Apprentissage post-analyse
            if self.enable_adaptive_learning:
                processing_time = time.time() - start_time
                await self._learn_from_execution(
                    context_signature, selected_tools, analysis_result, processing_time
                )
            
            return analysis_result
            
        finally:
            # Restauration de la méthode originale
            self._select_tools_for_mode = original_select_tools
    
    def _create_default_response(self, reason: str) -> AnalysisResponse:
        """Crée une réponse par défaut pour les scènes non analysées."""
        return AnalysisResponse(
            suspicion_level=SuspicionLevel.LOW,
            action_type=ActionType.NORMAL,
            confidence=0.95,
            description=reason,
            detections=[],
            tools_used=[],
            tool_results={},
            optimization_score=0.0
        )
    
    def _analyze_context(self, context: Dict[str, Any], detections: List) -> str:
        """Analyse du contexte pour génération de signature."""
        
        # Extraction des caractéristiques du contexte
        person_count = len([d for d in detections if hasattr(d, 'class_name') and d.class_name == "person"])
        current_hour = datetime.now().hour
        
        # Période de la journée
        if 6 <= current_hour < 12:
            time_period = "morning"
        elif 12 <= current_hour < 18:
            time_period = "afternoon"
        elif 18 <= current_hour < 22:
            time_period = "evening"
        else:
            time_period = "night"
        
        # Type de scénario basé sur les détections
        if person_count == 0:
            scenario_type = "empty"
        elif person_count == 1:
            scenario_type = "single_person"
        elif person_count <= 3:
            scenario_type = "normal"
        else:
            scenario_type = "crowded"
        
        # Location type depuis le contexte
        location_type = context.get("location", "general")
        
        # Génération de la signature de contexte
        context_signature = f"{scenario_type}_{time_period}_{person_count}_{location_type}"
        
        return context_signature
    
    async def _adaptive_tool_selection(self, context_signature: str) -> List[str]:
        """Sélection adaptative des outils basée sur le contexte."""
        
        # 1. Vérification du cache
        if context_signature in self._tool_selection_cache:
            cached_tools, cache_time = self._tool_selection_cache[context_signature]
            if datetime.now() - cache_time < self._cache_ttl:
                logger.debug(f"Outils depuis cache: {cached_tools}")
                return cached_tools
        
        # 2. Recherche de pattern de contexte similaire
        matching_pattern = self._find_matching_context_pattern(context_signature)
        
        if matching_pattern and matching_pattern.performance_score > self.min_performance_threshold:
            selected_tools = matching_pattern.optimal_tools.copy()
            logger.debug(f"Outils depuis pattern: {selected_tools}")
        else:
            # 3. Sélection basée sur les performances historiques
            selected_tools = self._select_tools_by_performance(context_signature)
            logger.debug(f"Outils par performance: {selected_tools}")
        
        # 4. Validation et ajustement
        selected_tools = self._validate_and_adjust_tool_selection(selected_tools)
        
        # 5. Mise en cache
        self._tool_selection_cache[context_signature] = (selected_tools, datetime.now())
        
        return selected_tools
    
    def _find_matching_context_pattern(self, context_signature: str) -> Optional[ContextPattern]:
        """Recherche d'un pattern de contexte similaire."""
        
        best_match = None
        best_similarity = 0.0
        
        for pattern_id, pattern in self.context_patterns.items():
            similarity = self._calculate_context_similarity(context_signature, pattern)
            
            if similarity > best_similarity and similarity >= self.context_similarity_threshold:
                best_similarity = similarity
                best_match = pattern
        
        return best_match
    
    def _calculate_context_similarity(self, signature: str, pattern: ContextPattern) -> float:
        """Calcul de similarité entre contexte et pattern."""
        
        # Extraction des composants de la signature
        signature_parts = signature.split('_')
        
        if len(signature_parts) < 4:
            return 0.0
        
        scenario_type, time_period, person_count_str, location_type = signature_parts[:4]
        person_count = int(person_count_str) if person_count_str.isdigit() else 0
        
        # Calcul de similarité par composant
        scenario_match = 1.0 if pattern.scenario_type == scenario_type else 0.0
        time_match = 1.0 if pattern.time_period == time_period else 0.5
        location_match = 1.0 if pattern.location_type == location_type else 0.3
        
        # Similarité du nombre de personnes (plus tolérant)
        person_diff = abs(pattern.person_count - person_count)
        person_match = max(0.0, 1.0 - (person_diff * 0.2))
        
        # Score de similarité pondéré
        similarity = (
            0.4 * scenario_match +
            0.2 * time_match +
            0.2 * person_match +
            0.2 * location_match
        )
        
        return similarity
    
    def _select_tools_by_performance(self, context_signature: str) -> List[str]:
        """Sélection d'outils basée sur les performances historiques."""
        
        # Score des outils selon le contexte et performance globale
        tool_scores = {}
        
        for tool_name, history in self.tool_performance_history.items():
            # Score de performance globale
            base_score = history.success_rate * 0.6 + (1.0 - min(1.0, history.avg_response_time / 3.0)) * 0.4
            
            # Bonus/malus selon le contexte si données disponibles
            context_bonus = 0.0
            if context_signature in history.context_effectiveness:
                context_bonus = history.context_effectiveness[context_signature] * 0.3
            
            # Score des performances récentes
            recent_score = 0.0
            if history.recent_scores:
                recent_score = np.mean(list(history.recent_scores)) * 0.2
            
            tool_scores[tool_name] = base_score + context_bonus + recent_score
        
        # Sélection des meilleurs outils
        if not tool_scores:
            return self.current_optimal_tools[:4]  # Fallback
        
        # Tri par score et sélection adaptative
        sorted_tools = sorted(tool_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Sélection intelligente selon le mode d'orchestration
        if self.config.mode == OrchestrationMode.FAST:
            selected = [tool for tool, score in sorted_tools[:3] if score > 0.5]
        elif self.config.mode == OrchestrationMode.BALANCED:
            selected = [tool for tool, score in sorted_tools[:5] if score > 0.4]
        else:  # THOROUGH
            selected = [tool for tool, score in sorted_tools[:6] if score > 0.3]
        
        return selected if selected else [tool for tool, _ in sorted_tools[:3]]
    
    def _validate_and_adjust_tool_selection(self, selected_tools: List[str]) -> List[str]:
        """Validation et ajustement de la sélection d'outils."""
        
        # Vérification des contraintes
        if len(selected_tools) > self.config.max_concurrent_tools:
            # Réduction selon les scores de performance
            tool_scores = {
                tool: self.tool_performance_history[tool].success_rate 
                for tool in selected_tools 
                if tool in self.tool_performance_history
            }
            
            sorted_by_score = sorted(tool_scores.items(), key=lambda x: x[1], reverse=True)
            selected_tools = [tool for tool, _ in sorted_by_score[:self.config.max_concurrent_tools]]
        
        # Ajout d'outils essentiels manquants
        essential_tools = ["dino_features", "multimodal_fusion"]  # Toujours utiles
        for tool in essential_tools:
            if tool not in selected_tools and len(selected_tools) < self.config.max_concurrent_tools:
                selected_tools.append(tool)
        
        # Vérification minimum d'outils
        if len(selected_tools) < 2:
            selected_tools.extend(self.current_optimal_tools[:2])
            selected_tools = list(set(selected_tools))  # Déduplication
        
        return selected_tools[:self.config.max_concurrent_tools]
    
    async def _learn_from_execution(
        self,
        context_signature: str,
        tools_used: List[str],
        result: AnalysisResponse,
        processing_time: float
    ) -> None:
        """Apprentissage à partir de l'exécution."""
        
        # Calcul du score de performance
        performance_score = self._calculate_performance_score(result, processing_time)
        
        # Mise à jour de l'historique des outils
        for tool in tools_used:
            if tool in self.tool_performance_history:
                history = self.tool_performance_history[tool]
                
                # Mise à jour du taux de succès
                success = 1.0 if result.confidence > 0.6 else 0.0
                history.success_rate = (
                    history.success_rate * 0.9 + success * 0.1
                )
                
                # Mise à jour du temps de réponse moyen
                history.avg_response_time = (
                    history.avg_response_time * 0.9 + processing_time * 0.1
                )
                
                # Ajout du score récent
                history.recent_scores.append(performance_score)
                
                # Mise à jour de l'efficacité contextuelle
                if context_signature not in history.context_effectiveness:
                    history.context_effectiveness[context_signature] = performance_score
                else:
                    current = history.context_effectiveness[context_signature]
                    history.context_effectiveness[context_signature] = (
                        current * 0.8 + performance_score * 0.2
                    )
                
                history.last_performance_update = datetime.now()
        
        # Mise à jour ou création du pattern de contexte
        await self._update_context_pattern(context_signature, tools_used, performance_score)
        
        # Ajout à l'historique global
        self.execution_history.append({
            "timestamp": datetime.now(),
            "context": context_signature,
            "tools": tools_used,
            "performance": performance_score,
            "processing_time": processing_time
        })
        
        self.performance_window.append(performance_score)
        
        # Sauvegarde périodique
        if len(self.execution_history) % 50 == 0:
            self._save_optimization_data()
    
    def _calculate_performance_score(
        self, 
        result: AnalysisResponse, 
        processing_time: float
    ) -> float:
        """Calcul du score de performance d'une exécution."""
        
        # Composants du score
        confidence_score = result.confidence
        
        # Score de temps de réponse (pénalité pour lenteur)
        time_score = max(0.0, 1.0 - (processing_time - 1.0) / 10.0)  # Optimal à 1s
        
        # Score de qualité de la réponse (basé sur les outils utilisés)
        tools_quality_score = len(result.tools_used) * 0.1  # Bonus outils utilisés
        
        # Score global
        performance_score = (
            confidence_score * 0.6 +
            time_score * 0.3 +
            tools_quality_score * 0.1
        )
        
        return min(1.0, max(0.0, performance_score))
    
    async def _update_context_pattern(
        self,
        context_signature: str,
        tools_used: List[str],
        performance_score: float
    ) -> None:
        """Mise à jour du pattern de contexte."""
        
        if context_signature in self.context_patterns:
            pattern = self.context_patterns[context_signature]
            
            # Mise à jour de la performance moyenne
            alpha = self.learning_rate
            pattern.performance_score = (
                pattern.performance_score * (1 - alpha) + performance_score * alpha
            )
            
            # Mise à jour des outils optimaux si performance meilleure
            if performance_score > pattern.performance_score:
                pattern.optimal_tools = tools_used.copy()
            
            pattern.usage_count += 1
            pattern.last_updated = datetime.now()
            
        else:
            # Création d'un nouveau pattern
            signature_parts = context_signature.split('_')
            if len(signature_parts) >= 4:
                scenario_type, time_period, person_count_str, location_type = signature_parts[:4]
                person_count = int(person_count_str) if person_count_str.isdigit() else 0
                
                self.context_patterns[context_signature] = ContextPattern(
                    scenario_type=scenario_type,
                    time_period=time_period,
                    person_count=person_count,
                    location_type=location_type,
                    optimal_tools=tools_used.copy(),
                    performance_score=performance_score,
                    usage_count=1,
                    last_updated=datetime.now()
                )
    
    def _should_reoptimize(self) -> bool:
        """Détermine si une re-optimisation est nécessaire."""
        
        # Vérification de l'intervalle de temps
        if datetime.now() - self.last_optimization < self.reoptimization_interval:
            return False
        
        # Vérification de performance dégradée
        if len(self.performance_window) >= 20:
            recent_avg = np.mean(list(self.performance_window)[-10:])
            older_avg = np.mean(list(self.performance_window)[-20:-10])
            
            if recent_avg < older_avg * 0.9:  # Dégradation de 10%+
                logger.info("Dégradation de performance détectée - Re-optimisation programmée")
                return True
        
        # Re-optimisation périodique
        return datetime.now() - self.last_optimization > self.reoptimization_interval
    
    async def _background_reoptimization(self) -> None:
        """Re-optimisation en arrière-plan."""
        
        try:
            logger.info("🔄 Démarrage de la re-optimisation automatique")
            
            # Lancement du benchmark
            benchmark = ToolOptimizationBenchmark(vlm_model_name=self.vlm_model_name)
            new_optimal_tools = await benchmark.auto_optimize_tools()
            
            # Mise à jour des outils optimaux si amélioration
            if new_optimal_tools:
                self.current_optimal_tools = new_optimal_tools
                self.last_optimization = datetime.now()
                
                logger.info(f"✅ Re-optimisation terminée - Nouveaux outils: {new_optimal_tools}")
                
                # Sauvegarde des nouveaux outils
                self._save_optimization_data()
            
        except Exception as e:
            logger.error(f"Erreur re-optimisation: {e}")
    
    def get_adaptive_status(self) -> Dict[str, Any]:
        """État du système adaptatif."""
        
        # Statistiques d'apprentissage
        total_patterns = len(self.context_patterns)
        avg_pattern_performance = np.mean([p.performance_score for p in self.context_patterns.values()]) if self.context_patterns else 0.0
        
        # Performance récente
        recent_performance = np.mean(list(self.performance_window)) if self.performance_window else 0.0
        
        # Top outils par performance
        top_tools = sorted(
            [(tool, history.success_rate) for tool, history in self.tool_performance_history.items()],
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        return {
            "adaptive_learning_enabled": self.enable_adaptive_learning,
            "current_optimal_tools": self.current_optimal_tools,
            "learning_stats": {
                "context_patterns_learned": total_patterns,
                "avg_pattern_performance": avg_pattern_performance,
                "executions_in_history": len(self.execution_history),
                "recent_performance": recent_performance
            },
            "top_performing_tools": [{"tool": tool, "success_rate": rate} for tool, rate in top_tools],
            "last_optimization": self.last_optimization.isoformat(),
            "next_optimization_due": (self.last_optimization + self.reoptimization_interval).isoformat(),
            "cache_stats": {
                "cached_contexts": len(self._tool_selection_cache),
                "cache_hit_potential": f"{len(self._tool_selection_cache) * 100 / max(len(self.execution_history), 1):.1f}%"
            }
        }
    
    async def force_reoptimization(self) -> List[str]:
        """Force une re-optimisation immédiate."""
        
        logger.info("🔄 Re-optimisation forcée")
        await self._background_reoptimization()
        return self.current_optimal_tools
    
    def reset_learning_data(self) -> None:
        """Réinitialisation des données d'apprentissage."""
        
        self.context_patterns.clear()
        self.tool_performance_history.clear()
        self.execution_history.clear()
        self.performance_window.clear()
        self._tool_selection_cache.clear()
        
        self._initialize_adaptive_system()
        self._save_optimization_data()
        
        logger.info("🧹 Données d'apprentissage réinitialisées")


# Factory function pour faciliter l'intégration
def create_adaptive_orchestrator(
    vlm_model_name: str = "kimi-vl-a3b-thinking",
    config: OrchestrationConfig = None,
    enable_learning: bool = True
) -> AdaptiveVLMOrchestrator:
    """Factory pour créer un orchestrateur adaptatif."""
    
    return AdaptiveVLMOrchestrator(
        vlm_model_name=vlm_model_name,
        config=config or OrchestrationConfig(),
        enable_adaptive_learning=enable_learning
    )