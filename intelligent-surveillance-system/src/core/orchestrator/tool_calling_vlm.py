"""
🔧 VLM avec Tool Calling Optimisé
=================================

Interface VLM spécialisée pour le tool calling intelligent avec optimisation automatique.
Ce module implémente la logique de tool calling que le VLM utilise pour orchestrer les outils.

Features:
- Tool calling natif intégré au VLM
- Sélection intelligente basée sur le contexte
- Optimisation automatique des appels d'outils
- Gestion des erreurs et fallbacks
- Cache intelligent des résultats
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import numpy as np

from loguru import logger

from ..types import AnalysisRequest, AnalysisResponse, SuspicionLevel, ActionType
from ..vlm.dynamic_model import DynamicVisionLanguageModel


@dataclass
class ToolCall:
    """Représentation d'un appel d'outil."""
    tool_name: str
    parameters: Dict[str, Any]
    reasoning: str  # Raison de l'appel de cet outil
    priority: int = 1  # 1=haute, 2=moyenne, 3=basse
    expected_output_type: str = "dict"
    timeout_seconds: float = 5.0
    
    
@dataclass
class ToolCallResult:
    """Résultat d'un appel d'outil."""
    tool_call: ToolCall
    success: bool
    result: Any = None
    error_message: str = ""
    execution_time: float = 0.0
    confidence: float = 0.0


@dataclass  
class ToolCallPlan:
    """Plan d'exécution des appels d'outils."""
    tool_calls: List[ToolCall]
    execution_strategy: str  # "sequential", "parallel", "conditional"
    estimated_total_time: float
    confidence_threshold: float = 0.6
    max_retries: int = 2


class ToolCallingVLM(DynamicVisionLanguageModel):
    """
    VLM avec capacités de tool calling optimisé.
    
    Extends DynamicVisionLanguageModel avec:
    - Tool calling intelligent
    - Optimisation basée sur l'historique
    - Gestion des dépendances entre outils
    - Cache des résultats d'outils
    """
    
    def __init__(
        self,
        default_model: str = "kimi-vl-a3b-thinking",
        enable_fallback: bool = True,
        tool_cache_enabled: bool = True
    ):
        super().__init__(default_model, enable_fallback)
        
        self.tool_cache_enabled = tool_cache_enabled
        self.tool_registry: Dict[str, Callable] = {}
        self.tool_call_history: List[ToolCallResult] = []
        self.tool_performance_cache: Dict[str, Dict] = {}
        
        # Cache des résultats d'outils (clé: hash des paramètres, valeur: résultat)
        self.tool_results_cache: Dict[str, Tuple[Any, datetime]] = {}
        self.cache_ttl = timedelta(minutes=15)
        
        # Configuration du tool calling
        self.max_parallel_tools = 4
        self.tool_call_timeout = 30.0
        self.confidence_threshold = 0.6
        
        # Initialisation des outils par défaut
        self._register_default_tools()
        
        logger.info("ToolCallingVLM initialisé avec capacités de tool calling optimisé")
    
    def _register_default_tools(self) -> None:
        """Enregistrement des outils par défaut."""
        
        # Outils de base pour surveillance
        self.register_tool("object_detection", self._tool_object_detection)
        self.register_tool("pose_analysis", self._tool_pose_analysis) 
        self.register_tool("behavior_assessment", self._tool_behavior_assessment)
        self.register_tool("context_analysis", self._tool_context_analysis)
        self.register_tool("risk_evaluation", self._tool_risk_evaluation)
        self.register_tool("temporal_analysis", self._tool_temporal_analysis)
        self.register_tool("spatial_analysis", self._tool_spatial_analysis)
        self.register_tool("confidence_validation", self._tool_confidence_validation)
    
    def register_tool(self, tool_name: str, tool_function: Callable) -> None:
        """Enregistrement d'un outil."""
        self.tool_registry[tool_name] = tool_function
        
        # Initialisation des métriques de performance
        if tool_name not in self.tool_performance_cache:
            self.tool_performance_cache[tool_name] = {
                "usage_count": 0,
                "success_rate": 1.0,
                "avg_execution_time": 1.0,
                "confidence_correlation": 0.5,
                "last_used": datetime.now()
            }
        
        logger.info(f"Outil enregistré: {tool_name}")
    
    async def analyze_with_tool_calling(
        self,
        request: AnalysisRequest,
        optimize_tool_selection: bool = True
    ) -> AnalysisResponse:
        """
        Analyse avec tool calling intelligent.
        
        Args:
            request: Requête d'analyse
            optimize_tool_selection: Active l'optimisation de sélection d'outils
            
        Returns:
            Réponse d'analyse enrichie par les outils
        """
        
        start_time = time.time()
        
        try:
            # 1. Analyse initiale pour déterminer les outils nécessaires
            initial_analysis = await self._initial_context_analysis(request)
            
            # 2. Génération du plan de tool calling
            tool_plan = await self._generate_tool_call_plan(
                request, initial_analysis, optimize_tool_selection
            )
            
            # 3. Exécution des appels d'outils
            tool_results = await self._execute_tool_call_plan(tool_plan)
            
            # 4. Synthèse finale avec intégration des résultats d'outils
            final_analysis = await self._synthesize_with_tool_results(
                request, initial_analysis, tool_results
            )
            
            # 5. Mise à jour des métriques de performance
            self._update_tool_performance_metrics(tool_results)
            
            processing_time = time.time() - start_time
            
            logger.info(
                f"Analyse avec tool calling terminée: {len(tool_results)} outils utilisés "
                f"en {processing_time:.2f}s - Confiance: {final_analysis.confidence:.2f}"
            )
            
            return final_analysis
            
        except Exception as e:
            logger.error(f"Erreur tool calling: {e}")
            
            # Fallback vers analyse simple
            return await self.analyze_frame(request)
    
    async def _initial_context_analysis(self, request: AnalysisRequest) -> Dict[str, Any]:
        """Analyse initiale du contexte pour déterminer les outils nécessaires."""
        
        context = request.context or {}
        
        # Analyse des éléments présents
        elements_detected = {
            "persons": context.get("person_count", 0),
            "objects": context.get("detections_count", 0),
            "location_type": context.get("location", "unknown"),
            "time_period": self._determine_time_period(),
            "complexity_level": self._assess_complexity(context)
        }
        
        # Détermination du niveau de risque initial
        risk_indicators = []
        
        if elements_detected["persons"] > 3:
            risk_indicators.append("crowded_scene")
        
        if elements_detected["location_type"] in ["electronics", "jewelry", "cash_register"]:
            risk_indicators.append("high_value_area")
        
        if elements_detected["time_period"] in ["night", "late_evening"]:
            risk_indicators.append("off_hours")
        
        # Score de priorité pour sélection d'outils
        priority_score = len(risk_indicators) + elements_detected["complexity_level"]
        
        return {
            "elements_detected": elements_detected,
            "risk_indicators": risk_indicators,
            "priority_score": priority_score,
            "recommended_analysis_depth": "thorough" if priority_score >= 3 else "standard"
        }
    
    def _determine_time_period(self) -> str:
        """Détermination de la période temporelle."""
        hour = datetime.now().hour
        
        if 6 <= hour < 9:
            return "morning"
        elif 9 <= hour < 12:
            return "late_morning"
        elif 12 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 20:
            return "evening"
        elif 20 <= hour < 23:
            return "late_evening"
        else:
            return "night"
    
    def _assess_complexity(self, context: Dict[str, Any]) -> int:
        """Évaluation de la complexité de la scène (1-5)."""
        
        complexity = 1
        
        # Nombre d'éléments détectés
        detections = context.get("detections_count", 0)
        if detections > 5:
            complexity += 1
        if detections > 10:
            complexity += 1
        
        # Diversité des classes d'objets
        classes = context.get("detection_classes", [])
        if len(set(classes)) > 3:
            complexity += 1
        
        # Présence de personnes multiples
        if context.get("person_count", 0) > 1:
            complexity += 1
        
        return min(5, complexity)
    
    async def _generate_tool_call_plan(
        self,
        request: AnalysisRequest,
        initial_analysis: Dict[str, Any],
        optimize_selection: bool
    ) -> ToolCallPlan:
        """Génération du plan d'appels d'outils."""
        
        available_tools = list(self.tool_registry.keys())
        selected_tools = []
        
        if optimize_selection:
            selected_tools = self._optimize_tool_selection(
                initial_analysis, available_tools
            )
        else:
            # Sélection basée sur les règles
            selected_tools = self._rule_based_tool_selection(initial_analysis)
        
        # Création des appels d'outils
        tool_calls = []
        
        for tool_name in selected_tools:
            tool_call = ToolCall(
                tool_name=tool_name,
                parameters=self._generate_tool_parameters(tool_name, request, initial_analysis),
                reasoning=self._generate_tool_reasoning(tool_name, initial_analysis),
                priority=self._calculate_tool_priority(tool_name, initial_analysis),
                timeout_seconds=self._get_tool_timeout(tool_name)
            )
            tool_calls.append(tool_call)
        
        # Tri par priorité
        tool_calls.sort(key=lambda tc: tc.priority)
        
        # Stratégie d'exécution
        execution_strategy = self._determine_execution_strategy(tool_calls, initial_analysis)
        
        # Estimation du temps total
        if execution_strategy == "parallel":
            estimated_time = max(tc.timeout_seconds for tc in tool_calls)
        else:
            estimated_time = sum(tc.timeout_seconds for tc in tool_calls)
        
        return ToolCallPlan(
            tool_calls=tool_calls,
            execution_strategy=execution_strategy,
            estimated_total_time=estimated_time,
            confidence_threshold=self.confidence_threshold
        )
    
    def _optimize_tool_selection(
        self,
        initial_analysis: Dict[str, Any],
        available_tools: List[str]
    ) -> List[str]:
        """Sélection optimisée des outils basée sur l'historique de performance."""
        
        priority_score = initial_analysis["priority_score"]
        recommended_depth = initial_analysis["recommended_analysis_depth"]
        
        # Score des outils basé sur performance historique et contexte
        tool_scores = {}
        
        for tool_name in available_tools:
            if tool_name in self.tool_performance_cache:
                metrics = self.tool_performance_cache[tool_name]
                
                # Score de base (performance historique)
                base_score = (
                    metrics["success_rate"] * 0.4 +
                    (1.0 - min(1.0, metrics["avg_execution_time"] / 5.0)) * 0.3 +
                    metrics["confidence_correlation"] * 0.3
                )
                
                # Bonus contextuel
                context_bonus = self._calculate_context_bonus(tool_name, initial_analysis)
                
                # Score de fraîcheur (favorise outils récemment utilisés avec succès)
                freshness_bonus = self._calculate_freshness_bonus(tool_name)
                
                tool_scores[tool_name] = base_score + context_bonus + freshness_bonus
            else:
                # Nouveau outil - score neutre
                tool_scores[tool_name] = 0.5
        
        # Sélection selon le niveau recommandé
        sorted_tools = sorted(tool_scores.items(), key=lambda x: x[1], reverse=True)
        
        if recommended_depth == "thorough":
            selected = [tool for tool, score in sorted_tools[:6] if score > 0.4]
        else:
            selected = [tool for tool, score in sorted_tools[:4] if score > 0.5]
        
        # Assurer les outils essentiels
        essential_tools = ["object_detection", "context_analysis"]
        for tool in essential_tools:
            if tool in available_tools and tool not in selected:
                selected.append(tool)
        
        return selected[:self.max_parallel_tools]
    
    def _rule_based_tool_selection(self, initial_analysis: Dict[str, Any]) -> List[str]:
        """Sélection d'outils basée sur des règles."""
        
        selected_tools = ["object_detection", "context_analysis"]  # Toujours inclus
        
        elements = initial_analysis["elements_detected"]
        risk_indicators = initial_analysis["risk_indicators"]
        
        # Ajout selon les éléments détectés
        if elements["persons"] > 0:
            selected_tools.append("pose_analysis")
            selected_tools.append("behavior_assessment")
        
        # Ajout selon les indicateurs de risque
        if "high_value_area" in risk_indicators:
            selected_tools.append("risk_evaluation")
        
        if "crowded_scene" in risk_indicators:
            selected_tools.append("spatial_analysis")
        
        if "off_hours" in risk_indicators:
            selected_tools.append("temporal_analysis")
        
        # Validation de confiance si situation complexe
        if initial_analysis["priority_score"] >= 3:
            selected_tools.append("confidence_validation")
        
        return list(set(selected_tools))  # Déduplication
    
    def _calculate_context_bonus(self, tool_name: str, initial_analysis: Dict[str, Any]) -> float:
        """Calcul du bonus contextuel pour un outil."""
        
        bonus = 0.0
        
        elements = initial_analysis["elements_detected"]
        risk_indicators = initial_analysis["risk_indicators"]
        
        # Bonus spécifiques par outil
        if tool_name == "pose_analysis" and elements["persons"] > 0:
            bonus += 0.2
        
        if tool_name == "risk_evaluation" and "high_value_area" in risk_indicators:
            bonus += 0.3
        
        if tool_name == "spatial_analysis" and "crowded_scene" in risk_indicators:
            bonus += 0.2
        
        if tool_name == "temporal_analysis" and "off_hours" in risk_indicators:
            bonus += 0.25
        
        return min(0.3, bonus)  # Bonus max de 0.3
    
    def _calculate_freshness_bonus(self, tool_name: str) -> float:
        """Calcul du bonus de fraîcheur."""
        
        if tool_name not in self.tool_performance_cache:
            return 0.0
        
        last_used = self.tool_performance_cache[tool_name]["last_used"]
        time_since_use = datetime.now() - last_used
        
        # Bonus dégressif selon le temps écoulé
        if time_since_use < timedelta(hours=1):
            return 0.1
        elif time_since_use < timedelta(hours=6):
            return 0.05
        else:
            return 0.0
    
    def _generate_tool_parameters(
        self,
        tool_name: str,
        request: AnalysisRequest,
        initial_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Génération des paramètres pour un outil."""
        
        base_params = {
            "frame_data": request.frame_data,
            "context": request.context or {},
            "initial_analysis": initial_analysis
        }
        
        # Paramètres spécifiques par outil
        tool_specific_params = {
            "object_detection": {
                "confidence_threshold": 0.5,
                "filter_classes": ["person", "handbag", "backpack"]
            },
            "pose_analysis": {
                "focus_on_hands": True,
                "detect_suspicious_poses": True
            },
            "behavior_assessment": {
                "analysis_window": 30,
                "behavior_patterns": ["loitering", "erratic_movement", "concealment"]
            },
            "risk_evaluation": {
                "location_context": initial_analysis["elements_detected"]["location_type"],
                "time_context": initial_analysis["elements_detected"]["time_period"]
            },
            "temporal_analysis": {
                "look_back_frames": 10,
                "pattern_detection": True
            },
            "spatial_analysis": {
                "zone_analysis": True,
                "crowd_density": True
            }
        }
        
        if tool_name in tool_specific_params:
            base_params.update(tool_specific_params[tool_name])
        
        return base_params
    
    def _generate_tool_reasoning(self, tool_name: str, initial_analysis: Dict[str, Any]) -> str:
        """Génération du raisonnement pour l'appel d'outil."""
        
        reasoning_templates = {
            "object_detection": "Détection d'objets nécessaire pour identifier les éléments présents",
            "pose_analysis": "Analyse des poses corporelles pour détecter comportements suspects",
            "behavior_assessment": "Évaluation comportementale pour identifier activités anormales",
            "context_analysis": "Analyse contextuelle pour comprendre l'environnement",
            "risk_evaluation": "Évaluation des risques selon le contexte spatio-temporel",
            "temporal_analysis": "Analyse temporelle pour détecter patterns suspects",
            "spatial_analysis": "Analyse spatiale pour évaluer densité et mouvement",
            "confidence_validation": "Validation de confiance pour situation complexe"
        }
        
        base_reasoning = reasoning_templates.get(
            tool_name, 
            f"Analyse spécialisée avec {tool_name}"
        )
        
        # Enrichissement contextuel
        risk_indicators = initial_analysis["risk_indicators"]
        if risk_indicators:
            base_reasoning += f" (Indicateurs: {', '.join(risk_indicators)})"
        
        return base_reasoning
    
    def _calculate_tool_priority(self, tool_name: str, initial_analysis: Dict[str, Any]) -> int:
        """Calcul de la priorité d'un outil (1=haute, 3=basse)."""
        
        # Priorités par défaut
        default_priorities = {
            "object_detection": 1,  # Toujours haute priorité
            "context_analysis": 1,
            "pose_analysis": 2,
            "behavior_assessment": 2,
            "risk_evaluation": 2,
            "temporal_analysis": 3,
            "spatial_analysis": 3,
            "confidence_validation": 3
        }
        
        base_priority = default_priorities.get(tool_name, 2)
        
        # Ajustement selon le contexte
        risk_indicators = initial_analysis["risk_indicators"]
        priority_score = initial_analysis["priority_score"]
        
        # Augmentation de priorité si situation critique
        if priority_score >= 4:
            base_priority = max(1, base_priority - 1)
        
        # Priorités spécifiques selon indicateurs de risque
        if "high_value_area" in risk_indicators and tool_name == "risk_evaluation":
            base_priority = 1
        
        if "crowded_scene" in risk_indicators and tool_name == "spatial_analysis":
            base_priority = 2
        
        return base_priority
    
    def _get_tool_timeout(self, tool_name: str) -> float:
        """Récupération du timeout pour un outil."""
        
        # Timeouts par défaut selon la complexité de l'outil
        default_timeouts = {
            "object_detection": 2.0,
            "context_analysis": 1.5,
            "pose_analysis": 3.0,
            "behavior_assessment": 4.0,
            "risk_evaluation": 2.5,
            "temporal_analysis": 3.5,
            "spatial_analysis": 3.0,
            "confidence_validation": 2.0
        }
        
        return default_timeouts.get(tool_name, 3.0)
    
    def _determine_execution_strategy(
        self,
        tool_calls: List[ToolCall],
        initial_analysis: Dict[str, Any]
    ) -> str:
        """Détermination de la stratégie d'exécution."""
        
        priority_score = initial_analysis["priority_score"]
        
        # Parallèle si situation critique et outils indépendants
        if priority_score >= 4 and len(tool_calls) <= self.max_parallel_tools:
            return "parallel"
        
        # Séquentiel si dépendances ou situation standard
        if self._has_tool_dependencies(tool_calls):
            return "sequential"
        
        return "parallel" if len(tool_calls) <= 3 else "sequential"
    
    def _has_tool_dependencies(self, tool_calls: List[ToolCall]) -> bool:
        """Vérification des dépendances entre outils."""
        
        # Dépendances simples (à étendre selon besoins)
        dependencies = {
            "behavior_assessment": ["object_detection", "pose_analysis"],
            "confidence_validation": ["risk_evaluation"]
        }
        
        tool_names = [tc.tool_name for tc in tool_calls]
        
        for tool_name in tool_names:
            if tool_name in dependencies:
                for dependency in dependencies[tool_name]:
                    if dependency in tool_names:
                        return True
        
        return False
    
    async def _execute_tool_call_plan(self, plan: ToolCallPlan) -> List[ToolCallResult]:
        """Exécution du plan d'appels d'outils."""
        
        results = []
        
        if plan.execution_strategy == "parallel":
            # Exécution parallèle
            tasks = []
            for tool_call in plan.tool_calls:
                task = self._execute_single_tool_call(tool_call)
                tasks.append(task)
            
            parallel_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(parallel_results):
                if isinstance(result, Exception):
                    # Gestion d'erreur
                    error_result = ToolCallResult(
                        tool_call=plan.tool_calls[i],
                        success=False,
                        error_message=str(result)
                    )
                    results.append(error_result)
                else:
                    results.append(result)
        
        else:
            # Exécution séquentielle
            for tool_call in plan.tool_calls:
                result = await self._execute_single_tool_call(tool_call)
                results.append(result)
                
                # Arrêt si outil critique échoue
                if not result.success and tool_call.priority == 1:
                    logger.warning(f"Outil critique {tool_call.tool_name} échoué - Arrêt du plan")
                    break
        
        return results
    
    async def _execute_single_tool_call(self, tool_call: ToolCall) -> ToolCallResult:
        """Exécution d'un appel d'outil unique."""
        
        start_time = time.time()
        
        try:
            # Vérification du cache
            if self.tool_cache_enabled:
                cache_key = self._generate_cache_key(tool_call)
                cached_result = self._get_cached_result(cache_key)
                
                if cached_result is not None:
                    logger.debug(f"Résultat depuis cache pour {tool_call.tool_name}")
                    return ToolCallResult(
                        tool_call=tool_call,
                        success=True,
                        result=cached_result,
                        execution_time=0.001,  # Cache hit
                        confidence=0.9  # Confiance élevée pour cache
                    )
            
            # Exécution de l'outil
            if tool_call.tool_name not in self.tool_registry:
                raise ValueError(f"Outil non enregistré: {tool_call.tool_name}")
            
            tool_function = self.tool_registry[tool_call.tool_name]
            
            # Exécution avec timeout
            result = await asyncio.wait_for(
                tool_function(**tool_call.parameters),
                timeout=tool_call.timeout_seconds
            )
            
            execution_time = time.time() - start_time
            
            # Évaluation de la confiance du résultat
            confidence = self._evaluate_result_confidence(tool_call, result, execution_time)
            
            # Mise en cache si réussite
            if self.tool_cache_enabled and confidence > 0.7:
                cache_key = self._generate_cache_key(tool_call)
                self._cache_result(cache_key, result)
            
            return ToolCallResult(
                tool_call=tool_call,
                success=True,
                result=result,
                execution_time=execution_time,
                confidence=confidence
            )
        
        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            logger.warning(f"Timeout pour {tool_call.tool_name} après {tool_call.timeout_seconds}s")
            
            return ToolCallResult(
                tool_call=tool_call,
                success=False,
                error_message=f"Timeout après {tool_call.timeout_seconds}s",
                execution_time=execution_time
            )
        
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Erreur exécution {tool_call.tool_name}: {e}")
            
            return ToolCallResult(
                tool_call=tool_call,
                success=False,
                error_message=str(e),
                execution_time=execution_time
            )
    
    def _generate_cache_key(self, tool_call: ToolCall) -> str:
        """Génération d'une clé de cache pour un appel d'outil."""
        
        # Sérialisation des paramètres (exclut frame_data pour éviter clés trop longues)
        cache_params = tool_call.parameters.copy()
        if "frame_data" in cache_params:
            # Hash du frame_data au lieu de la valeur complète
            frame_hash = hash(cache_params["frame_data"][:100]) if cache_params["frame_data"] else 0
            cache_params["frame_data_hash"] = frame_hash
            del cache_params["frame_data"]
        
        params_str = json.dumps(cache_params, sort_keys=True, default=str)
        cache_key = f"{tool_call.tool_name}:{hash(params_str)}"
        
        return cache_key
    
    def _get_cached_result(self, cache_key: str) -> Optional[Any]:
        """Récupération d'un résultat depuis le cache."""
        
        if cache_key in self.tool_results_cache:
            result, timestamp = self.tool_results_cache[cache_key]
            
            # Vérification de l'expiration
            if datetime.now() - timestamp < self.cache_ttl:
                return result
            else:
                # Nettoyage du cache expiré
                del self.tool_results_cache[cache_key]
        
        return None
    
    def _cache_result(self, cache_key: str, result: Any) -> None:
        """Mise en cache d'un résultat."""
        
        self.tool_results_cache[cache_key] = (result, datetime.now())
        
        # Nettoyage périodique du cache
        if len(self.tool_results_cache) % 100 == 0:
            self._cleanup_expired_cache()
    
    def _cleanup_expired_cache(self) -> None:
        """Nettoyage du cache expiré."""
        
        current_time = datetime.now()
        expired_keys = []
        
        for key, (_, timestamp) in self.tool_results_cache.items():
            if current_time - timestamp > self.cache_ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.tool_results_cache[key]
        
        if expired_keys:
            logger.debug(f"Cache nettoyé: {len(expired_keys)} entrées expirées supprimées")
    
    def _evaluate_result_confidence(
        self,
        tool_call: ToolCall,
        result: Any,
        execution_time: float
    ) -> float:
        """Évaluation de la confiance d'un résultat d'outil."""
        
        confidence = 0.5  # Base
        
        # Bonus si exécution rapide (dans les temps)
        if execution_time < tool_call.timeout_seconds * 0.5:
            confidence += 0.2
        
        # Bonus selon le type de résultat
        if isinstance(result, dict):
            if "confidence" in result and isinstance(result["confidence"], (int, float)):
                confidence = max(confidence, result["confidence"])
            
            if "error" not in result:
                confidence += 0.1
        
        # Bonus selon l'historique de l'outil
        if tool_call.tool_name in self.tool_performance_cache:
            historical_success = self.tool_performance_cache[tool_call.tool_name]["success_rate"]
            confidence = confidence * 0.7 + historical_success * 0.3
        
        return min(1.0, max(0.0, confidence))
    
    async def _synthesize_with_tool_results(
        self,
        request: AnalysisRequest,
        initial_analysis: Dict[str, Any],
        tool_results: List[ToolCallResult]
    ) -> AnalysisResponse:
        """Synthèse finale avec intégration des résultats d'outils."""
        
        # Extraction des résultats réussis
        successful_results = [tr for tr in tool_results if tr.success]
        failed_results = [tr for tr in tool_results if not tr.success]
        
        # Agrégation des informations
        aggregated_info = {
            "tools_used": [tr.tool_call.tool_name for tr in successful_results],
            "tools_failed": [tr.tool_call.tool_name for tr in failed_results],
            "total_execution_time": sum(tr.execution_time for tr in tool_results),
            "avg_confidence": np.mean([tr.confidence for tr in successful_results]) if successful_results else 0.0
        }
        
        # Analyse du niveau de suspicion basée sur les outils
        suspicion_level = self._determine_suspicion_from_tools(successful_results, initial_analysis)
        
        # Détermination du type d'action
        action_type = self._determine_action_from_analysis(suspicion_level, successful_results)
        
        # Calcul de la confiance globale
        global_confidence = self._calculate_global_confidence(
            successful_results, failed_results, initial_analysis
        )
        
        # Génération de la description
        description = self._generate_analysis_description(
            suspicion_level, successful_results, aggregated_info
        )
        
        # Génération des recommandations
        recommendations = self._generate_recommendations(
            suspicion_level, action_type, successful_results, failed_results
        )
        
        return AnalysisResponse(
            suspicion_level=suspicion_level,
            action_type=action_type,
            confidence=global_confidence,
            description=description,
            tools_used=aggregated_info["tools_used"],
            recommendations=recommendations
        )
    
    def _determine_suspicion_from_tools(
        self,
        successful_results: List[ToolCallResult],
        initial_analysis: Dict[str, Any]
    ) -> SuspicionLevel:
        """Détermination du niveau de suspicion basé sur les résultats d'outils."""
        
        suspicion_indicators = []
        
        for result in successful_results:
            tool_name = result.tool_call.tool_name
            tool_result = result.result
            
            if isinstance(tool_result, dict):
                # Analyse des indicateurs spécifiques par outil
                if tool_name == "behavior_assessment":
                    if tool_result.get("suspicious_behavior", False):
                        suspicion_indicators.append("suspicious_behavior")
                
                if tool_name == "risk_evaluation":
                    risk_score = tool_result.get("risk_score", 0)
                    if risk_score > 0.7:
                        suspicion_indicators.append("high_risk_context")
                
                if tool_name == "pose_analysis":
                    if tool_result.get("concealment_detected", False):
                        suspicion_indicators.append("concealment_behavior")
                
                if tool_name == "temporal_analysis":
                    if tool_result.get("unusual_timing", False):
                        suspicion_indicators.append("temporal_anomaly")
        
        # Détermination du niveau selon les indicateurs
        indicator_count = len(suspicion_indicators)
        priority_score = initial_analysis["priority_score"]
        
        if indicator_count >= 3 or priority_score >= 4:
            return SuspicionLevel.CRITICAL
        elif indicator_count >= 2 or priority_score >= 3:
            return SuspicionLevel.HIGH
        elif indicator_count >= 1 or priority_score >= 2:
            return SuspicionLevel.MEDIUM
        else:
            return SuspicionLevel.LOW
    
    def _determine_action_from_analysis(
        self,
        suspicion_level: SuspicionLevel,
        successful_results: List[ToolCallResult]
    ) -> ActionType:
        """Détermination du type d'action recommandé."""
        
        # Analyse des résultats pour indices spécifiques
        theft_indicators = 0
        suspicious_indicators = 0
        
        for result in successful_results:
            if isinstance(result.result, dict):
                tool_result = result.result
                
                # Indicateurs de vol
                if tool_result.get("concealment_detected", False):
                    theft_indicators += 1
                
                if tool_result.get("suspicious_behavior", False):
                    suspicious_indicators += 1
                
                if tool_result.get("high_risk_context", False):
                    theft_indicators += 1
        
        # Détermination de l'action
        if suspicion_level == SuspicionLevel.CRITICAL or theft_indicators >= 2:
            return ActionType.THEFT_ATTEMPT
        elif suspicion_level == SuspicionLevel.HIGH or suspicious_indicators >= 2:
            return ActionType.SUSPICIOUS_ACTIVITY
        elif suspicion_level == SuspicionLevel.MEDIUM:
            return ActionType.LOITERING
        else:
            return ActionType.NORMAL_SHOPPING
    
    def _calculate_global_confidence(
        self,
        successful_results: List[ToolCallResult],
        failed_results: List[ToolCallResult],
        initial_analysis: Dict[str, Any]
    ) -> float:
        """Calcul de la confiance globale de l'analyse."""
        
        if not successful_results:
            return 0.1  # Très faible confiance si aucun outil réussi
        
        # Confiance moyenne des outils réussis
        tools_confidence = np.mean([tr.confidence for tr in successful_results])
        
        # Pénalité pour outils échoués
        failure_penalty = len(failed_results) * 0.1
        
        # Bonus pour cohérence (si plusieurs outils donnent résultats similaires)
        coherence_bonus = self._calculate_coherence_bonus(successful_results)
        
        # Ajustement selon complexité initiale
        complexity_factor = 1.0 - (initial_analysis["priority_score"] * 0.05)
        
        global_confidence = (
            tools_confidence * complexity_factor + 
            coherence_bonus - 
            failure_penalty
        )
        
        return min(1.0, max(0.1, global_confidence))
    
    def _calculate_coherence_bonus(self, successful_results: List[ToolCallResult]) -> float:
        """Calcul du bonus de cohérence entre outils."""
        
        if len(successful_results) < 2:
            return 0.0
        
        # Analyse de la cohérence des résultats
        coherent_indicators = 0
        total_comparisons = 0
        
        for i, result1 in enumerate(successful_results):
            for result2 in successful_results[i+1:]:
                total_comparisons += 1
                
                # Vérification de cohérence simple
                if (isinstance(result1.result, dict) and isinstance(result2.result, dict)):
                    r1, r2 = result1.result, result2.result
                    
                    # Cohérence des niveaux de confiance
                    if abs(r1.get("confidence", 0.5) - r2.get("confidence", 0.5)) < 0.3:
                        coherent_indicators += 1
        
        if total_comparisons > 0:
            coherence_ratio = coherent_indicators / total_comparisons
            return coherence_ratio * 0.2  # Bonus max de 0.2
        
        return 0.0
    
    def _generate_analysis_description(
        self,
        suspicion_level: SuspicionLevel,
        successful_results: List[ToolCallResult],
        aggregated_info: Dict[str, Any]
    ) -> str:
        """Génération de la description de l'analyse."""
        
        tools_summary = f"Analyse effectuée avec {len(successful_results)} outils"
        
        if aggregated_info["tools_failed"]:
            tools_summary += f" ({len(aggregated_info['tools_failed'])} échecs)"
        
        suspicion_desc = {
            SuspicionLevel.LOW: "Situation normale détectée",
            SuspicionLevel.MEDIUM: "Activité potentiellement suspecte observée",
            SuspicionLevel.HIGH: "Comportement suspect détecté",
            SuspicionLevel.CRITICAL: "Situation critique identifiée"
        }
        
        base_description = suspicion_desc.get(suspicion_level, "Analyse effectuée")
        
        # Ajout d'informations spécifiques des outils
        specific_info = []
        for result in successful_results:
            if isinstance(result.result, dict) and "summary" in result.result:
                specific_info.append(result.result["summary"])
        
        full_description = f"{base_description}. {tools_summary}."
        
        if specific_info:
            full_description += f" Détails: {'; '.join(specific_info[:2])}."
        
        return full_description
    
    def _generate_recommendations(
        self,
        suspicion_level: SuspicionLevel,
        action_type: ActionType,
        successful_results: List[ToolCallResult],
        failed_results: List[ToolCallResult]
    ) -> List[str]:
        """Génération des recommandations."""
        
        recommendations = []
        
        # Recommandations selon le niveau de suspicion
        if suspicion_level == SuspicionLevel.CRITICAL:
            recommendations.extend([
                "Alerte immédiate du personnel de sécurité",
                "Démarrage de l'enregistrement vidéo",
                "Surveillance rapprochée de la zone"
            ])
        elif suspicion_level == SuspicionLevel.HIGH:
            recommendations.extend([
                "Surveillance accrue de la zone",
                "Notification du superviseur",
                "Conservation des images"
            ])
        elif suspicion_level == SuspicionLevel.MEDIUM:
            recommendations.extend([
                "Observation continue",
                "Vérification contextuelle"
            ])
        
        # Recommandations selon les échecs d'outils
        if failed_results:
            recommendations.append("Vérification du système d'analyse recommandée")
        
        # Recommandations spécifiques des outils
        for result in successful_results:
            if isinstance(result.result, dict) and "recommendations" in result.result:
                tool_recs = result.result["recommendations"]
                if isinstance(tool_recs, list):
                    recommendations.extend(tool_recs[:2])  # Max 2 par outil
        
        return list(set(recommendations))  # Déduplication
    
    def _update_tool_performance_metrics(self, tool_results: List[ToolCallResult]) -> None:
        """Mise à jour des métriques de performance des outils."""
        
        for result in tool_results:
            tool_name = result.tool_call.tool_name
            
            if tool_name in self.tool_performance_cache:
                metrics = self.tool_performance_cache[tool_name]
                
                # Mise à jour du compte d'utilisation
                metrics["usage_count"] += 1
                
                # Mise à jour du taux de succès (moyenne mobile)
                alpha = 0.1
                success = 1.0 if result.success else 0.0
                metrics["success_rate"] = (
                    metrics["success_rate"] * (1 - alpha) + success * alpha
                )
                
                # Mise à jour du temps d'exécution moyen
                metrics["avg_execution_time"] = (
                    metrics["avg_execution_time"] * (1 - alpha) + 
                    result.execution_time * alpha
                )
                
                # Mise à jour de la corrélation de confiance
                if result.success and result.confidence > 0:
                    metrics["confidence_correlation"] = (
                        metrics["confidence_correlation"] * (1 - alpha) + 
                        result.confidence * alpha
                    )
                
                metrics["last_used"] = datetime.now()
                
                # Ajout à l'historique
                self.tool_call_history.append(result)
                
                # Limitation de l'historique
                if len(self.tool_call_history) > 1000:
                    self.tool_call_history = self.tool_call_history[-500:]
    
    def get_tool_calling_stats(self) -> Dict[str, Any]:
        """Statistiques du système de tool calling."""
        
        total_calls = len(self.tool_call_history)
        successful_calls = len([r for r in self.tool_call_history if r.success])
        
        # Statistiques par outil
        tool_stats = {}
        for tool_name, metrics in self.tool_performance_cache.items():
            tool_stats[tool_name] = {
                "usage_count": metrics["usage_count"],
                "success_rate": f"{metrics['success_rate']:.1%}",
                "avg_execution_time": f"{metrics['avg_execution_time']:.2f}s",
                "confidence_correlation": f"{metrics['confidence_correlation']:.2f}"
            }
        
        return {
            "total_tool_calls": total_calls,
            "success_rate": f"{successful_calls/max(total_calls, 1):.1%}",
            "cache_entries": len(self.tool_results_cache),
            "registered_tools": len(self.tool_registry),
            "tool_performance": tool_stats
        }
    
    # Implémentations des outils par défaut (stubs - à implémenter selon besoins)
    
    async def _tool_object_detection(self, **kwargs) -> Dict[str, Any]:
        """Outil de détection d'objets."""
        await asyncio.sleep(0.1)  # Simulation
        return {
            "objects_detected": ["person", "handbag"],
            "confidence": 0.85,
            "summary": "2 objets détectés"
        }
    
    async def _tool_pose_analysis(self, **kwargs) -> Dict[str, Any]:
        """Outil d'analyse des poses."""
        await asyncio.sleep(0.2)
        return {
            "poses_detected": ["standing", "walking"],
            "concealment_detected": False,
            "confidence": 0.7,
            "summary": "Poses normales détectées"
        }
    
    async def _tool_behavior_assessment(self, **kwargs) -> Dict[str, Any]:
        """Outil d'évaluation comportementale."""
        await asyncio.sleep(0.3)
        return {
            "suspicious_behavior": False,
            "behavior_type": "normal_shopping",
            "confidence": 0.8,
            "summary": "Comportement normal observé"
        }
    
    async def _tool_context_analysis(self, **kwargs) -> Dict[str, Any]:
        """Outil d'analyse contextuelle."""
        await asyncio.sleep(0.1)
        return {
            "context_type": "retail_environment",
            "time_appropriate": True,
            "confidence": 0.9,
            "summary": "Contexte environnemental normal"
        }
    
    async def _tool_risk_evaluation(self, **kwargs) -> Dict[str, Any]:
        """Outil d'évaluation des risques."""
        await asyncio.sleep(0.2)
        return {
            "risk_score": 0.3,
            "risk_factors": [],
            "confidence": 0.75,
            "summary": "Niveau de risque faible"
        }
    
    async def _tool_temporal_analysis(self, **kwargs) -> Dict[str, Any]:
        """Outil d'analyse temporelle."""
        await asyncio.sleep(0.2)
        return {
            "temporal_patterns": [],
            "unusual_timing": False,
            "confidence": 0.6,
            "summary": "Aucune anomalie temporelle"
        }
    
    async def _tool_spatial_analysis(self, **kwargs) -> Dict[str, Any]:
        """Outil d'analyse spatiale."""
        await asyncio.sleep(0.25)
        return {
            "spatial_anomalies": [],
            "crowd_density": "normal",
            "confidence": 0.65,
            "summary": "Distribution spatiale normale"
        }
    
    async def _tool_confidence_validation(self, **kwargs) -> Dict[str, Any]:
        """Outil de validation de confiance."""
        await asyncio.sleep(0.15)
        return {
            "validation_passed": True,
            "confidence_score": 0.8,
            "confidence": 0.85,
            "summary": "Validation de confiance réussie"
        }


# Factory function
def create_tool_calling_vlm(
    model_name: str = "kimi-vl-a3b-thinking",
    enable_caching: bool = True
) -> ToolCallingVLM:
    """Factory pour créer un VLM avec tool calling."""
    
    return ToolCallingVLM(
        default_model=model_name,
        enable_fallback=True,
        tool_cache_enabled=enable_caching
    )