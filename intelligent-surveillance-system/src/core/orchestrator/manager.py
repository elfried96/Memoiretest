"""Gestionnaire d'orchestration des outils de surveillance."""

import asyncio
import time
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import json

from loguru import logger

from ..types import (
    ToolType, 
    ToolResult, 
    AnalysisRequest, 
    AnalysisResponse,
    SuspicionLevel,
    ActionType,
    DetectedObject,
    Frame
)
from ..vlm.model import VisionLanguageModel
from ...detection.yolo.detector import YOLODetector
from ...detection.tracking.tracker import MultiObjectTracker
from ...utils.exceptions import OrchestrationError, ToolError


class ExecutionStrategy(Enum):
    """Stratégies d'exécution des outils."""
    SEQUENTIAL = "sequential"  # Exécution séquentielle
    PARALLEL = "parallel"     # Exécution parallèle
    CONDITIONAL = "conditional"  # Exécution conditionnelle
    PIPELINE = "pipeline"     # Pipeline avec dépendances


@dataclass
class ToolConfig:
    """Configuration d'un outil."""
    tool_type: ToolType
    enabled: bool = True
    timeout_seconds: float = 5.0
    retry_count: int = 2
    required_confidence: float = 0.0
    depends_on: List[ToolType] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionPlan:
    """Plan d'exécution des outils."""
    tools_sequence: List[ToolType]
    strategy: ExecutionStrategy
    estimated_time: float
    priority_level: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class ToolRegistry:
    """Registre des outils disponibles."""
    
    def __init__(self):
        self._tools: Dict[ToolType, Any] = {}
        self._tool_configs: Dict[ToolType, ToolConfig] = {}
        self._tool_functions: Dict[ToolType, Callable] = {}
        
        self._initialize_default_configs()
    
    def _initialize_default_configs(self) -> None:
        """Initialisation des configurations par défaut."""
        
        default_configs = {
            ToolType.OBJECT_DETECTOR: ToolConfig(
                tool_type=ToolType.OBJECT_DETECTOR,
                timeout_seconds=2.0,
                required_confidence=0.25,
                parameters={"filter_classes": ["person", "handbag", "backpack"]}
            ),
            ToolType.TRACKER: ToolConfig(
                tool_type=ToolType.TRACKER,
                timeout_seconds=1.0,
                depends_on=[ToolType.OBJECT_DETECTOR],
                parameters={"tracker_type": "bytetrack"}
            ),
            ToolType.BEHAVIOR_ANALYZER: ToolConfig(
                tool_type=ToolType.BEHAVIOR_ANALYZER,
                timeout_seconds=3.0,
                depends_on=[ToolType.TRACKER],
                required_confidence=0.5,
                parameters={"analysis_window": 30}
            ),
            ToolType.CONTEXT_VALIDATOR: ToolConfig(
                tool_type=ToolType.CONTEXT_VALIDATOR,
                timeout_seconds=2.0,
                parameters={"validation_rules": ["time_consistency", "spatial_consistency"]}
            ),
            ToolType.FALSE_POSITIVE_FILTER: ToolConfig(
                tool_type=ToolType.FALSE_POSITIVE_FILTER,
                timeout_seconds=1.5,
                depends_on=[ToolType.BEHAVIOR_ANALYZER, ToolType.CONTEXT_VALIDATOR],
                parameters={"threshold": 0.03}
            )
        }
        
        self._tool_configs.update(default_configs)
    
    def register_tool(
        self, 
        tool_type: ToolType, 
        tool_instance: Any,
        tool_function: Callable,
        config: Optional[ToolConfig] = None
    ) -> None:
        """Enregistrement d'un outil."""
        
        self._tools[tool_type] = tool_instance
        self._tool_functions[tool_type] = tool_function
        
        if config:
            self._tool_configs[tool_type] = config
        
        logger.info(f"Outil {tool_type.value} enregistré")
    
    def get_tool(self, tool_type: ToolType) -> Optional[Any]:
        """Récupération d'un outil."""
        return self._tools.get(tool_type)
    
    def get_tool_function(self, tool_type: ToolType) -> Optional[Callable]:
        """Récupération de la fonction d'un outil."""
        return self._tool_functions.get(tool_type)
    
    def get_tool_config(self, tool_type: ToolType) -> Optional[ToolConfig]:
        """Récupération de la configuration d'un outil."""
        return self._tool_configs.get(tool_type)
    
    def is_tool_available(self, tool_type: ToolType) -> bool:
        """Vérification de disponibilité d'un outil."""
        config = self._tool_configs.get(tool_type)
        return (
            tool_type in self._tools and 
            tool_type in self._tool_functions and
            config and config.enabled
        )
    
    def get_available_tools(self) -> List[ToolType]:
        """Liste des outils disponibles."""
        return [
            tool_type for tool_type in self._tools.keys()
            if self.is_tool_available(tool_type)
        ]


class ToolOrchestrator:
    """
    Orchestrateur principal des outils de surveillance.
    
    Features:
    - Orchestration intelligente basée sur VLM
    - Exécution parallèle/séquentielle
    - Gestion des dépendances entre outils
    - Validation croisée automatique
    - Métriques de performance
    """
    
    def __init__(
        self,
        vlm_model: VisionLanguageModel,
        tool_registry: Optional[ToolRegistry] = None
    ):
        self.vlm_model = vlm_model
        self.tool_registry = tool_registry or ToolRegistry()
        
        # Historique des exécutions
        self.execution_history = []
        self.performance_metrics = {
            "total_executions": 0,
            "avg_execution_time": 0.0,
            "tool_usage_stats": {},
            "success_rate": 0.0
        }
        
        # Cache des plans d'exécution
        self._execution_plan_cache = {}
        
        logger.info("ToolOrchestrator initialisé")
    
    def register_surveillance_tools(
        self,
        yolo_detector: YOLODetector,
        object_tracker: MultiObjectTracker
    ) -> None:
        """Enregistrement des outils de surveillance principaux."""
        
        # Détecteur YOLO
        self.tool_registry.register_tool(
            ToolType.OBJECT_DETECTOR,
            yolo_detector,
            self._yolo_detection_wrapper
        )
        
        # Tracker d'objets
        self.tool_registry.register_tool(
            ToolType.TRACKER,
            object_tracker,
            self._tracking_wrapper
        )
        
        # Analyseur comportemental (implémentation simple)
        self.tool_registry.register_tool(
            ToolType.BEHAVIOR_ANALYZER,
            None,  # Pas d'instance spécifique
            self._behavior_analysis_wrapper
        )
        
        # Validateur contextuel
        self.tool_registry.register_tool(
            ToolType.CONTEXT_VALIDATOR,
            None,
            self._context_validation_wrapper
        )
        
        # Filtre anti-faux positifs
        self.tool_registry.register_tool(
            ToolType.FALSE_POSITIVE_FILTER,
            None,
            self._false_positive_filter_wrapper
        )
        
        logger.info("Outils de surveillance enregistrés")
    
    async def analyze_and_orchestrate(
        self,
        frame: Frame,
        context: Dict[str, Any],
        previous_analysis: Optional[AnalysisResponse] = None
    ) -> Tuple[AnalysisResponse, Dict[str, ToolResult]]:
        """
        Analyse d'un frame avec orchestration intelligente des outils.
        
        Args:
            frame: Frame à analyser
            context: Contexte additionnel
            previous_analysis: Analyse précédente pour continuité
            
        Returns:
            Tuple (analyse VLM, résultats des outils)
        """
        
        start_time = time.time()
        
        try:
            # Phase 1: Analyse VLM initiale pour déterminer la stratégie
            initial_analysis = await self._initial_vlm_analysis(frame, context)
            
            # Phase 2: Génération du plan d'exécution
            execution_plan = await self._generate_execution_plan(
                initial_analysis, context, previous_analysis
            )
            
            # Phase 3: Exécution des outils selon le plan
            tool_results = await self._execute_tool_plan(
                frame, execution_plan, context
            )
            
            # Phase 4: Analyse VLM finale avec résultats des outils
            final_analysis = await self._final_vlm_analysis(
                frame, context, tool_results, initial_analysis
            )
            
            # Mise à jour des métriques
            execution_time = time.time() - start_time
            self._update_performance_metrics(execution_time, tool_results)
            
            logger.debug(
                f"Orchestration complète en {execution_time:.3f}s - "
                f"Suspicion: {final_analysis.suspicion_level.value} - "
                f"Outils utilisés: {len(tool_results)}"
            )
            
            return final_analysis, tool_results
            
        except Exception as e:
            logger.error(f"Erreur orchestration: {e}")
            
            # Analyse de fallback
            fallback_analysis = AnalysisResponse(
                suspicion_level=SuspicionLevel.LOW,
                action_type=ActionType.NORMAL_SHOPPING,
                confidence=0.0,
                description=f"Erreur orchestration: {str(e)}",
                tools_used=[],
                recommendations=["Analyse manuelle requise"]
            )
            
            return fallback_analysis, {}
    
    async def _initial_vlm_analysis(
        self, 
        frame: Frame, 
        context: Dict[str, Any]
    ) -> AnalysisResponse:
        """Analyse VLM initiale pour stratégie."""
        
        # Préparation de la requête
        import base64
        from io import BytesIO
        from PIL import Image
        
        # Conversion de l'image en base64
        pil_image = Image.fromarray(frame.image)
        buffer = BytesIO()
        pil_image.save(buffer, format="JPEG", quality=85)
        image_b64 = base64.b64encode(buffer.getvalue()).decode()
        
        # Contexte enrichi
        enriched_context = {
            **context,
            "frame_info": {
                "timestamp": frame.timestamp.isoformat(),
                "stream_id": frame.stream_id,
                "frame_id": frame.frame_id,
                "resolution": f"{frame.width}x{frame.height}"
            },
            "available_tools": [tool.value for tool in self.tool_registry.get_available_tools()]
        }
        
        request = AnalysisRequest(
            frame_data=image_b64,
            context=enriched_context,
            tools_available=enriched_context["available_tools"]
        )
        
        return await self.vlm_model.analyze_frame(request)
    
    async def _generate_execution_plan(
        self,
        initial_analysis: AnalysisResponse,
        context: Dict[str, Any],
        previous_analysis: Optional[AnalysisResponse]
    ) -> ExecutionPlan:
        """Génération du plan d'exécution basé sur l'analyse."""
        
        # Cache key pour éviter de regénérer des plans similaires
        cache_key = f"{initial_analysis.suspicion_level.value}_{initial_analysis.action_type.value}"
        
        if cache_key in self._execution_plan_cache:
            cached_plan = self._execution_plan_cache[cache_key]
            logger.debug(f"Plan d'exécution récupéré du cache: {cache_key}")
            return cached_plan
        
        # Détermination des outils à utiliser selon le niveau de suspicion
        tools_sequence = [ToolType.OBJECT_DETECTOR]  # Toujours commencer par la détection
        
        if initial_analysis.suspicion_level in [SuspicionLevel.MEDIUM, SuspicionLevel.HIGH, SuspicionLevel.CRITICAL]:
            tools_sequence.extend([
                ToolType.TRACKER,
                ToolType.BEHAVIOR_ANALYZER
            ])
        
        if initial_analysis.suspicion_level in [SuspicionLevel.HIGH, SuspicionLevel.CRITICAL]:
            tools_sequence.extend([
                ToolType.CONTEXT_VALIDATOR,
                ToolType.FALSE_POSITIVE_FILTER
            ])
        
        # Stratégie d'exécution selon l'urgence
        if initial_analysis.suspicion_level == SuspicionLevel.CRITICAL:
            strategy = ExecutionStrategy.PARALLEL
            priority = 1
        elif initial_analysis.suspicion_level == SuspicionLevel.HIGH:
            strategy = ExecutionStrategy.PIPELINE
            priority = 2
        else:
            strategy = ExecutionStrategy.SEQUENTIAL
            priority = 3
        
        # Estimation du temps d'exécution
        estimated_time = sum(
            self.tool_registry.get_tool_config(tool).timeout_seconds
            for tool in tools_sequence
            if self.tool_registry.get_tool_config(tool)
        )
        
        if strategy == ExecutionStrategy.PARALLEL:
            estimated_time = estimated_time * 0.6  # Optimisation parallèle
        
        plan = ExecutionPlan(
            tools_sequence=tools_sequence,
            strategy=strategy,
            estimated_time=estimated_time,
            priority_level=priority,
            metadata={
                "suspicion_level": initial_analysis.suspicion_level.value,
                "action_type": initial_analysis.action_type.value,
                "confidence": initial_analysis.confidence
            }
        )
        
        # Mise en cache
        self._execution_plan_cache[cache_key] = plan
        
        return plan
    
    async def _execute_tool_plan(
        self,
        frame: Frame,
        plan: ExecutionPlan,
        context: Dict[str, Any]
    ) -> Dict[str, ToolResult]:
        """Exécution du plan d'outils."""
        
        tool_results = {}
        
        if plan.strategy == ExecutionStrategy.PARALLEL:
            # Exécution parallèle des outils indépendants
            tasks = []
            for tool_type in plan.tools_sequence:
                if self._can_execute_tool_parallel(tool_type, tool_results):
                    task = self._execute_single_tool(tool_type, frame, context, tool_results)
                    tasks.append((tool_type, task))
            
            # Attendre tous les résultats
            results = await asyncio.gather(
                *[task for _, task in tasks], 
                return_exceptions=True
            )
            
            for (tool_type, _), result in zip(tasks, results):
                if isinstance(result, Exception):
                    logger.error(f"Erreur outil {tool_type.value}: {result}")
                    tool_results[tool_type.value] = ToolResult(
                        tool_type=tool_type,
                        success=False,
                        error_message=str(result),
                        execution_time_ms=0
                    )
                else:
                    tool_results[tool_type.value] = result
        
        else:
            # Exécution séquentielle ou pipeline
            for tool_type in plan.tools_sequence:
                if self._check_tool_dependencies(tool_type, tool_results):
                    result = await self._execute_single_tool(
                        tool_type, frame, context, tool_results
                    )
                    tool_results[tool_type.value] = result
                else:
                    logger.warning(f"Dépendances non satisfaites pour {tool_type.value}")
        
        return tool_results
    
    def _can_execute_tool_parallel(
        self, 
        tool_type: ToolType, 
        completed_results: Dict[str, ToolResult]
    ) -> bool:
        """Vérifie si un outil peut être exécuté en parallèle."""
        config = self.tool_registry.get_tool_config(tool_type)
        if not config:
            return False
        
        # Vérifier les dépendances
        for dependency in config.depends_on:
            if dependency.value not in completed_results:
                return False
        
        return True
    
    def _check_tool_dependencies(
        self, 
        tool_type: ToolType, 
        completed_results: Dict[str, ToolResult]
    ) -> bool:
        """Vérification des dépendances d'un outil."""
        config = self.tool_registry.get_tool_config(tool_type)
        if not config:
            return True
        
        for dependency in config.depends_on:
            if dependency.value not in completed_results:
                return False
            if not completed_results[dependency.value].success:
                logger.warning(f"Dépendance {dependency.value} échouée pour {tool_type.value}")
                return False
        
        return True
    
    async def _execute_single_tool(
        self,
        tool_type: ToolType,
        frame: Frame,
        context: Dict[str, Any],
        previous_results: Dict[str, ToolResult]
    ) -> ToolResult:
        """Exécution d'un outil unique."""
        
        start_time = time.time()
        
        try:
            tool_function = self.tool_registry.get_tool_function(tool_type)
            config = self.tool_registry.get_tool_config(tool_type)
            
            if not tool_function or not config:
                raise ToolError(f"Outil {tool_type.value} non configuré")
            
            # Préparation des paramètres
            tool_params = {
                "frame": frame,
                "context": context,
                "previous_results": previous_results,
                **config.parameters
            }
            
            # Exécution avec timeout
            result_data = await asyncio.wait_for(
                tool_function(**tool_params),
                timeout=config.timeout_seconds
            )
            
            execution_time = (time.time() - start_time) * 1000
            
            return ToolResult(
                tool_type=tool_type,
                success=True,
                data=result_data,
                execution_time_ms=execution_time
            )
            
        except asyncio.TimeoutError:
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"Timeout outil {tool_type.value} après {config.timeout_seconds}s")
            
            return ToolResult(
                tool_type=tool_type,
                success=False,
                execution_time_ms=execution_time,
                error_message=f"Timeout après {config.timeout_seconds}s"
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"Erreur outil {tool_type.value}: {e}")
            
            return ToolResult(
                tool_type=tool_type,
                success=False,
                execution_time_ms=execution_time,
                error_message=str(e)
            )
    
    async def _final_vlm_analysis(
        self,
        frame: Frame,
        context: Dict[str, Any],
        tool_results: Dict[str, ToolResult],
        initial_analysis: AnalysisResponse
    ) -> AnalysisResponse:
        """Analyse VLM finale avec résultats des outils."""
        
        # Préparation de la requête enrichie avec les résultats d'outils
        import base64
        from io import BytesIO
        from PIL import Image
        
        pil_image = Image.fromarray(frame.image)
        buffer = BytesIO()
        pil_image.save(buffer, format="JPEG", quality=85)
        image_b64 = base64.b64encode(buffer.getvalue()).decode()
        
        # Contexte enrichi avec résultats d'outils
        enriched_context = {
            **context,
            "initial_analysis": {
                "suspicion_level": initial_analysis.suspicion_level.value,
                "action_type": initial_analysis.action_type.value,
                "confidence": initial_analysis.confidence
            },
            "tool_results": {
                tool_name: {
                    "success": result.success,
                    "data": result.data,
                    "confidence": result.confidence,
                    "execution_time_ms": result.execution_time_ms
                }
                for tool_name, result in tool_results.items()
                if result.success
            }
        }
        
        request = AnalysisRequest(
            frame_data=image_b64,
            context=enriched_context,
            tools_available=list(tool_results.keys())
        )
        
        return await self.vlm_model.analyze_frame(request, tool_results)
    
    # Wrappers pour les outils
    
    async def _yolo_detection_wrapper(self, **kwargs) -> Dict[str, Any]:
        """Wrapper pour le détecteur YOLO."""
        frame = kwargs["frame"]
        filter_classes = kwargs.get("filter_classes")
        
        detector = self.tool_registry.get_tool(ToolType.OBJECT_DETECTOR)
        detections = detector.detect(frame, filter_classes=filter_classes)
        
        return {
            "detections_count": len(detections),
            "detections": [det.to_dict() for det in detections],
            "classes_found": list(set(det.class_name for det in detections))
        }
    
    async def _tracking_wrapper(self, **kwargs) -> Dict[str, Any]:
        """Wrapper pour le tracker."""
        frame = kwargs["frame"]
        previous_results = kwargs["previous_results"]
        
        # Récupération des détections précédentes
        detector_result = previous_results.get("object_detector")
        if not detector_result or not detector_result.success:
            return {"error": "Aucune détection disponible pour le suivi"}
        
        # Reconstruction des objets détectés
        detections = []
        for det_dict in detector_result.data.get("detections", []):
            bbox = BoundingBox(
                x=det_dict["bbox"]["x"],
                y=det_dict["bbox"]["y"],
                width=det_dict["bbox"]["width"],
                height=det_dict["bbox"]["height"],
                confidence=det_dict["bbox"]["confidence"]
            )
            detection = DetectedObject(
                class_id=det_dict["class_id"],
                class_name=det_dict["class_name"],
                bbox=bbox,
                confidence=det_dict["confidence"]
            )
            detections.append(detection)
        
        tracker = self.tool_registry.get_tool(ToolType.TRACKER)
        tracked_objects = tracker.update(detections)
        
        return {
            "tracked_objects_count": len(tracked_objects),
            "active_tracks": [
                {
                    "track_id": track_id,
                    "class_name": state.class_name,
                    "confidence": state.confidence,
                    "age": state.age,
                    "hits": state.hits
                }
                for track_id, state in tracked_objects.items()
            ]
        }
    
    async def _behavior_analysis_wrapper(self, **kwargs) -> Dict[str, Any]:
        """Wrapper pour l'analyse comportementale."""
        previous_results = kwargs["previous_results"]
        
        tracker_result = previous_results.get("tracker")
        if not tracker_result or not tracker_result.success:
            return {"error": "Aucune donnée de suivi disponible"}
        
        # Analyse comportementale simple basée sur les tracks
        active_tracks = tracker_result.data.get("active_tracks", [])
        
        suspicious_behaviors = []
        for track in active_tracks:
            # Règles simples de détection de comportement suspect
            if track["age"] > 100 and track["hits"] < 20:  # Présence longue, peu de détections
                suspicious_behaviors.append({
                    "track_id": track["track_id"],
                    "behavior": "Mouvement erratique",
                    "confidence": 0.6
                })
            
            if track["confidence"] < 0.4 and track["age"] > 50:  # Confiance faible persistante
                suspicious_behaviors.append({
                    "track_id": track["track_id"],
                    "behavior": "Objet partiellement occulté",
                    "confidence": 0.5
                })
        
        return {
            "analyzed_tracks": len(active_tracks),
            "suspicious_behaviors": suspicious_behaviors,
            "suspicion_score": len(suspicious_behaviors) / max(len(active_tracks), 1)
        }
    
    async def _context_validation_wrapper(self, **kwargs) -> Dict[str, Any]:
        """Wrapper pour la validation contextuelle."""
        context = kwargs["context"]
        frame = kwargs["frame"]
        
        # Validation contextuelle simple
        validations = {}
        
        # Validation temporelle
        current_hour = datetime.now().hour
        if 22 <= current_hour or current_hour <= 6:  # Nuit
            validations["time_context"] = {
                "period": "night",
                "suspicion_modifier": 1.2,  # Plus suspect la nuit
                "confidence": 0.8
            }
        else:
            validations["time_context"] = {
                "period": "day",
                "suspicion_modifier": 1.0,
                "confidence": 0.9
            }
        
        # Validation spatiale (à développer avec zones ROI)
        validations["spatial_context"] = {
            "zone": context.get("location", "unknown"),
            "high_risk_zone": context.get("location") in ["electronics", "jewelry", "alcohol"],
            "confidence": 0.7
        }
        
        return {
            "validations": validations,
            "overall_context_score": np.mean([v["confidence"] for v in validations.values()])
        }
    
    async def _false_positive_filter_wrapper(self, **kwargs) -> Dict[str, Any]:
        """Wrapper pour le filtre anti-faux positifs."""
        previous_results = kwargs["previous_results"]
        threshold = kwargs.get("threshold", 0.03)
        
        # Collecte des scores de confiance des différents outils
        confidence_scores = []
        
        # Score du détecteur
        detector_result = previous_results.get("object_detector")
        if detector_result and detector_result.success:
            detections = detector_result.data.get("detections", [])
            if detections:
                avg_detection_confidence = np.mean([d["confidence"] for d in detections])
                confidence_scores.append(avg_detection_confidence)
        
        # Score du tracker
        tracker_result = previous_results.get("tracker")
        if tracker_result and tracker_result.success:
            tracks = tracker_result.data.get("active_tracks", [])
            if tracks:
                avg_track_confidence = np.mean([t["confidence"] for t in tracks])
                confidence_scores.append(avg_track_confidence)
        
        # Score comportemental
        behavior_result = previous_results.get("behavior_analyzer")
        if behavior_result and behavior_result.success:
            suspicion_score = behavior_result.data.get("suspicion_score", 0)
            confidence_scores.append(suspicion_score)
        
        # Score contextuel
        context_result = previous_results.get("context_validator")
        if context_result and context_result.success:
            context_score = context_result.data.get("overall_context_score", 0)
            confidence_scores.append(context_score)
        
        # Calcul du score final
        if confidence_scores:
            final_score = np.mean(confidence_scores)
            is_false_positive = final_score < threshold
        else:
            final_score = 0.0
            is_false_positive = True
        
        return {
            "final_confidence_score": final_score,
            "is_false_positive": is_false_positive,
            "threshold_used": threshold,
            "individual_scores": confidence_scores,
            "filter_recommendation": "reject" if is_false_positive else "accept"
        }
    
    def _update_performance_metrics(
        self, 
        execution_time: float, 
        tool_results: Dict[str, ToolResult]
    ) -> None:
        """Mise à jour des métriques de performance."""
        
        self.performance_metrics["total_executions"] += 1
        
        # Temps d'exécution moyen
        if self.performance_metrics["avg_execution_time"] == 0:
            self.performance_metrics["avg_execution_time"] = execution_time
        else:
            alpha = 0.1
            self.performance_metrics["avg_execution_time"] = (
                alpha * execution_time + 
                (1 - alpha) * self.performance_metrics["avg_execution_time"]
            )
        
        # Statistiques d'usage des outils
        for tool_name, result in tool_results.items():
            if tool_name not in self.performance_metrics["tool_usage_stats"]:
                self.performance_metrics["tool_usage_stats"][tool_name] = {
                    "usage_count": 0,
                    "success_count": 0,
                    "avg_execution_time": 0.0
                }
            
            stats = self.performance_metrics["tool_usage_stats"][tool_name]
            stats["usage_count"] += 1
            
            if result.success:
                stats["success_count"] += 1
            
            # Temps d'exécution moyen de l'outil
            if stats["avg_execution_time"] == 0:
                stats["avg_execution_time"] = result.execution_time_ms
            else:
                alpha = 0.1
                stats["avg_execution_time"] = (
                    alpha * result.execution_time_ms + 
                    (1 - alpha) * stats["avg_execution_time"]
                )
        
        # Taux de succès global
        successful_tools = sum(1 for result in tool_results.values() if result.success)
        total_tools = len(tool_results)
        
        if total_tools > 0:
            current_success_rate = successful_tools / total_tools
            
            if self.performance_metrics["success_rate"] == 0:
                self.performance_metrics["success_rate"] = current_success_rate
            else:
                alpha = 0.1
                self.performance_metrics["success_rate"] = (
                    alpha * current_success_rate + 
                    (1 - alpha) * self.performance_metrics["success_rate"]
                )
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Récupération des métriques de performance."""
        return self.performance_metrics.copy()
    
    def reset_metrics(self) -> None:
        """Réinitialisation des métriques."""
        self.performance_metrics = {
            "total_executions": 0,
            "avg_execution_time": 0.0,
            "tool_usage_stats": {},
            "success_rate": 0.0
        }
        self._execution_plan_cache.clear()
        logger.info("Métriques d'orchestration réinitialisées")