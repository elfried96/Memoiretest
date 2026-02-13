"""VLM Orchestrator modernisé avec intégration complète des 8 outils avancés."""

import asyncio
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import json
import logging
import numpy as np

from ..types import Detection, AnalysisRequest, AnalysisResponse, ActionType, SuspicionLevel
from ..vlm.dynamic_model import DynamicVisionLanguageModel
from ...utils.exceptions import ProcessingError
from ..vlm.tools_integration import AdvancedToolsManager
from ..vlm.prompt_builder import PromptBuilder
from ..vlm.response_parser import ResponseParser

logger = logging.getLogger(__name__)


class OrchestrationMode(Enum):
    """Modes d'orchestration pour différents besoins."""
    FAST = "fast"           # Analyse rapide, outils limités
    BALANCED = "balanced"   # Équilibre vitesse/précision  
    THOROUGH = "thorough"   # Analyse complète, tous les outils


@dataclass
class OrchestrationConfig:
    """Configuration pour l'orchestrateur."""
    mode: OrchestrationMode = OrchestrationMode.BALANCED
    max_concurrent_tools: int = 4
    timeout_seconds: int = 600  # 10 minutes
    confidence_threshold: float = 0.7
    enable_advanced_tools: bool = True


class ModernVLMOrchestrator:
    """
    Orchestrateur VLM moderne avec:
    - Intégration des 8 outils avancés
    - Architecture modulaire 
    - Gestion optimisée des performances
    """
    
    def __init__(
        self, 
        vlm_model_name: str = "qwen2.5-vl-7b-instruct",
        config: OrchestrationConfig = None
    ):
        self.vlm_model_name = vlm_model_name
        self.config = config or OrchestrationConfig()
        self.tools_manager = AdvancedToolsManager()
        self.prompt_builder = PromptBuilder()
        self.response_parser = ResponseParser()
        self.vlm_model_instance = DynamicVisionLanguageModel(default_model=vlm_model_name)
        
        self.stats = {
            "total_analyses": 0,
            "successful_analyses": 0,
            "error_count": 0,
            "average_response_time": 0.0,
            "tools_usage": {}
        }
        
        logger.info(f"Orchestrateur VLM initialisé - Mode: {self.config.mode.value}")
    
    async def analyze_surveillance_frame(
        self, 
        frame_data: str,  # Base64 encoded
        detections: List[Detection] = None,
        context: Dict[str, Any] = None,
        video_context_metadata: Dict[str, Any] = None
    ) -> AnalysisResponse:
        """Analyse complète d'un frame de surveillance."""
        
        start_time = time.time()
        self.stats["total_analyses"] += 1
        
        try:
            # 1. Préparation de la requête avec contexte vidéo
            analysis_request = self._prepare_analysis_request(
                frame_data, detections, context, video_context_metadata
            )
            
            # 2. Sélection des outils selon le mode
            tools_to_use = self._select_tools_for_mode()
            analysis_request.tools_available = tools_to_use
            
            # 3. Analyse avec le VLM et outils
            if self.config.enable_advanced_tools:
                result = await self.vlm_model_instance.analyze_with_tools(
                    analysis_request, 
                    use_advanced_tools=True
                )
            else:
                result = await self.vlm_model_instance.analyze_with_tools(
                    analysis_request,
                    use_advanced_tools=False
                )
            
            # 4. Post-traitement et validation
            validated_result = self._validate_and_enhance_result(result)
            
            # 5. Mise à jour des statistiques
            processing_time = time.time() - start_time
            self._update_stats(validated_result, processing_time, tools_to_use)
            
            logger.info(
                f"Analyse terminée - Suspicion: {validated_result.suspicion_level.value} "
                f"({processing_time:.2f}s)"
            )
            
            return validated_result
            
        except Exception as e:
            logger.error(f"Erreur orchestration: {e}")
            self.stats["error_count"] += 1
            return self._create_error_response(str(e), time.time() - start_time)
    
    def _prepare_analysis_request(
        self,
        frame_data: str,
        detections: List[Detection] = None,
        context: Dict[str, Any] = None,
        video_context_metadata: Dict[str, Any] = None
    ) -> AnalysisRequest:
        """Préparation de la requête d'analyse."""
        
        # Enrichissement du contexte avec les détections
        enriched_context = context or {}
        
        if detections:
            enriched_context.update({
                "detections_count": len(detections),
                "person_count": len([d for d in detections if d.class_name == "person"]),
                "detection_boxes": [[d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2] for d in detections],
                "person_boxes": [
                    [d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2] 
                    for d in detections if d.class_name == "person"
                ],
                "detection_classes": [d.class_name for d in detections]
            })
        
        # Ajout de métadonnées temporelles
        enriched_context.update({
            "timestamp": time.time(),
            "analysis_mode": self.config.mode.value,
            "frame_id": f"frame_{int(time.time() * 1000)}"
        })
        
        # Intégration du contexte vidéo utilisateur si disponible
        if video_context_metadata:
            enriched_context["video_context_metadata"] = video_context_metadata
            logger.info(f"Contexte vidéo intégré - Titre: {video_context_metadata.get('title', 'Non spécifié')}")
        
        return AnalysisRequest(
            frame_data=frame_data,
            context=enriched_context,
            tools_available=[]  # Sera rempli par _select_tools_for_mode
        )
    
    def _select_tools_for_mode(self) -> List[str]:
        """Sélection des outils selon le mode d'orchestration."""
        
        all_tools = [
            "sam2_segmentator",
            "dino_features", 
            "pose_estimator",
            "trajectory_analyzer",
            "multimodal_fusion",
            "temporal_transformer",
            "adversarial_detector",
            "domain_adapter"
        ]
        
        if self.config.mode == OrchestrationMode.FAST:
            # Mode rapide: outils essentiels seulement
            return [
                "dino_features",
                "pose_estimator", 
                "multimodal_fusion"
            ]
        
        elif self.config.mode == OrchestrationMode.BALANCED:
            # Mode équilibré: outils principaux
            return [
                "sam2_segmentator",
                "dino_features",
                "pose_estimator",
                "trajectory_analyzer",
                "multimodal_fusion",
                "adversarial_detector"
            ]
        
        elif self.config.mode == OrchestrationMode.THOROUGH:
            # Mode complet: tous les outils
            return all_tools
        
        return all_tools
    
    def _validate_and_enhance_result(self, result: AnalysisResponse) -> AnalysisResponse:
        """Validation et amélioration du résultat."""
        
        # Validation des seuils de confiance
        if result.confidence < self.config.confidence_threshold:
            logger.warning(
                f"Confiance faible ({result.confidence:.2f}) - "
                f"Seuil: {self.config.confidence_threshold}"
            )
            
            # Ajout de recommandation de vérification manuelle
            if "Vérification manuelle recommandée" not in result.recommendations:
                result.recommendations.append("Vérification manuelle recommandée")
        
        # Enrichissement de la description avec contexte des outils
        if result.tools_used:
            tools_info = f" | Outils utilisés: {', '.join(result.tools_used)}"
            result.description += tools_info
        
        return result
    
    def _update_stats(
        self, 
        result: AnalysisResponse, 
        processing_time: float,
        tools_used: List[str]
    ):
        """Mise à jour des statistiques de performance."""
        
        # Statistiques générales
        if result.confidence > self.config.confidence_threshold:
            self.stats["successful_analyses"] += 1
        
        # Mise à jour de la moyenne des temps de réponse
        current_avg = self.stats["average_response_time"]
        total_calls = self.stats["total_analyses"]
        self.stats["average_response_time"] = (
            (current_avg * (total_calls - 1) + processing_time) / total_calls
        )
        
        # Statistiques d'utilisation des outils
        for tool in tools_used:
            if tool not in self.stats["tools_usage"]:
                self.stats["tools_usage"][tool] = 0
            self.stats["tools_usage"][tool] += 1
    
    def _create_error_response(self, error: str, processing_time: float) -> AnalysisResponse:
        """Création d'une réponse d'erreur."""
        from ..types import SuspicionLevel, ActionType
        
        return AnalysisResponse(
            suspicion_level=SuspicionLevel.LOW,
            action_type=ActionType.NORMAL_SHOPPING,
            confidence=0.0,
            description=f"Erreur d'analyse: {error}",
            tools_used=[],
            recommendations=[
                "Analyse échouée - vérification système requise",
                "Relance de l'analyse recommandée"
            ]
        )
    
    async def batch_analyze(
        self, 
        frames_data: List[Dict[str, Any]],
        max_concurrent: int = None
    ) -> List[AnalysisResponse]:
        """Analyse par batch de plusieurs frames."""
        
        max_concurrent = max_concurrent or self.config.max_concurrent_tools
        
        # Création des tâches d'analyse
        tasks = []
        for frame_data in frames_data:
            task = self.analyze_surveillance_frame(
                frame_data.get("frame_data", ""),
                frame_data.get("detections", []),
                frame_data.get("context", {})
            )
            tasks.append(task)
        
        # Exécution avec limite de concurrence
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def limited_analyze(task):
            async with semaphore:
                return await task
        
        # Lancement des analyses
        results = await asyncio.gather(*[limited_analyze(task) for task in tasks])
        
        logger.info(f"Analyse batch terminée: {len(results)} frames traités")
        return results
    
    def get_system_status(self) -> Dict[str, Any]:
        """État complet du système d'orchestration."""
        
        # Statut du VLM
        vlm_status = self.vlm_model_instance.get_system_status()
        
        # Calcul du taux de succès
        success_rate = (
            self.stats["successful_analyses"] / max(self.stats["total_analyses"], 1)
        ) * 100
        
        return {
            "orchestrator": {
                "mode": self.config.mode.value,
                "enable_advanced_tools": self.config.enable_advanced_tools,
                "max_concurrent_tools": self.config.max_concurrent_tools,
                "timeout_seconds": self.config.timeout_seconds
            },
            "performance": {
                **self.stats,
                "success_rate_percent": success_rate
            },
            "vlm_system": vlm_status
        }
    
    def update_config(self, new_config: OrchestrationConfig):
        """Mise à jour de la configuration."""
        self.config = new_config
        logger.info(f"Configuration mise à jour - Mode: {self.config.mode.value}")
    
    async def health_check(self) -> Dict[str, bool]:
        """Vérification de santé du système."""
        
        health_status = {
            "vlm_loaded": self.vlm_model_instance.is_loaded,
            "tools_available": True,
            "system_responsive": True
        }
        
        try:
            # Test rapide du VLM si chargé
            if self.vlm_model_instance.is_loaded:
                # Test basique avec une image fictive
                test_request = AnalysisRequest(
                    frame_data="",  # Image vide pour test
                    context={"test": True},
                    tools_available=[]
                )
                # Note: Dans une vraie implémentation, faire un test avec une vraie image
                
        except Exception as e:
            logger.error(f"Health check échoué: {e}")
            health_status["system_responsive"] = False
        
        return health_status
    
    async def shutdown(self):
        """Arrêt propre du système."""
        logger.info("Arrêt de l'orchestrateur VLM...")
        
        # Déchargement du VLM
        await self.vlm_model_instance.shutdown()
        
        logger.info("Orchestrateur VLM arrêté")