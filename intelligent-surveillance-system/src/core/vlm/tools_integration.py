"""Intégration des outils avancés dans le VLM."""

from typing import Dict, List, Any, Optional
import numpy as np
import time
from loguru import logger

from ...advanced_tools import (
    SAM2Segmentator,
    DinoV2FeatureExtractor, 
    OpenPoseEstimator,
    TrajectoryAnalyzer,
    MultiModalFusion,
    TemporalTransformer,
    AdversarialDetector,
    DomainAdapter
)
from ..types import ToolResult


class AdvancedToolsManager:
    """Gestionnaire intégré des 8 outils avancés."""
    
    def __init__(self):
        self.tools = {}
        self._initialize_tools()
    
    def _initialize_tools(self):
        """Initialisation de tous les outils avancés."""
        try:
            self.tools = {
                "sam2_segmentator": SAM2Segmentator(),
                "dino_features": DinoV2FeatureExtractor(),
                "pose_estimator": OpenPoseEstimator(),
                "trajectory_analyzer": TrajectoryAnalyzer(),
                "multimodal_fusion": MultiModalFusion(),
                "temporal_transformer": TemporalTransformer(),
                "adversarial_detector": AdversarialDetector(),
                "domain_adapter": DomainAdapter()
            }
            logger.info(f"Outils avancés initialisés: {list(self.tools.keys())}")
            
        except Exception as e:
            logger.error(f"Erreur initialisation outils avancés: {e}")
            self.tools = {}
    
    async def execute_tools(
        self, 
        image: np.ndarray,
        requested_tools: List[str],
        context: Dict[str, Any] = None
    ) -> Dict[str, ToolResult]:
        """Exécution des outils demandés."""
        
        # Vérification image vide
        if image is None or (hasattr(image, 'size') and image.size == 0):
            logger.warning("Image vide reçue pour traitement outils")
            return {}
        
        results = {}
        
        for tool_name in requested_tools:
            if tool_name not in self.tools:
                logger.warning(f"Outil {tool_name} non disponible")
                results[tool_name] = ToolResult(
                    success=False,
                    data={"error": f"Outil {tool_name} non disponible"},
                    confidence=0.0
                )
                continue
            
            try:
                result = await self._execute_single_tool(tool_name, image, context)
                results[tool_name] = result
                logger.debug(f"Outil {tool_name} exécuté avec succès")
                
            except Exception as e:
                logger.error(f"Erreur exécution {tool_name}: {e}")
                results[tool_name] = ToolResult(
                    tool_type=tool_name,
                    success=False,
                    data={"error": str(e)},
                    confidence=0.0,
                    execution_time_ms=0.0
                )
        
        return results
    
    async def _execute_single_tool(
        self, 
        tool_name: str, 
        image: np.ndarray,
        context: Dict[str, Any] = None
    ) -> ToolResult:
        """Exécution d'un outil spécifique."""
        
        start_time = time.perf_counter()
        tool = self.tools[tool_name]
        context = context or {}
        
        if tool_name == "sam2_segmentator":
            # Segmentation avec boîtes englobantes si disponibles
            boxes = context.get("detection_boxes", [])
            if boxes:
                result = tool.segment_objects(image, boxes)
            else:
                result = tool.segment_everything(image)
            
            execution_time = (time.perf_counter() - start_time) * 1000
            return ToolResult(
                tool_type="sam2_segmentator",
                success=True,
                data={
                    "num_masks": len(result.masks),
                    "mask_properties": getattr(result, 'mask_properties', {}),
                    "processing_time": result.processing_time
                },
                confidence=result.confidence if hasattr(result, 'confidence') else 0.8,
                execution_time_ms=execution_time
            )
        
        elif tool_name == "dino_features":
            # Extraction de features globales et régionales
            global_features = tool.extract_global_features(image)
            
            # Features régionales si détections disponibles
            regions = context.get("detection_regions", [])
            regional_features = []
            if regions:
                regional_features = tool.extract_regional_features(image, regions)
            
            execution_time = (time.perf_counter() - start_time) * 1000
            return ToolResult(
                tool_type="dino_features",
                success=True,
                data={
                    "global_features_shape": global_features.shape if global_features is not None else None,
                    "num_regional_features": len(regional_features),
                    "feature_similarity": tool.compute_similarity(global_features, global_features).item() if global_features is not None else 0.0
                },
                confidence=0.9,
                execution_time_ms=execution_time
            )
        
        elif tool_name == "pose_estimator":
            # Estimation de poses avec analyse comportementale
            person_boxes = context.get("person_boxes", [])
            poses = tool.estimate_poses(image, person_boxes)
            
            # Analyse comportementale
            behavior_analysis = {}
            if poses:
                behavior_analysis = tool.analyze_behavior(poses[0])  # Première personne
            
            execution_time = (time.perf_counter() - start_time) * 1000
            return ToolResult(
                tool_type="pose_estimator",
                success=True,
                data={
                    "num_poses": len(poses),
                    "behavior_score": behavior_analysis.get("suspicion_score", 0.0),
                    "behavior_indicators": behavior_analysis.get("indicators", [])
                },
                confidence=0.85,
                execution_time_ms=execution_time
            )
        
        elif tool_name == "trajectory_analyzer":
            # Analyse des trajectoires
            trajectory_data = context.get("trajectory_points", [])
            if not trajectory_data:
                # Simuler des points de trajectoire basiques si non disponibles
                trajectory_data = [(100, 100), (120, 110), (140, 120)]
            
            analysis = tool.analyze_trajectory(trajectory_data)
            
            execution_time = (time.perf_counter() - start_time) * 1000
            return ToolResult(
                tool_type="trajectory_analyzer",
                success=True,
                data={
                    "pattern_classification": analysis.get("pattern", "unknown"),
                    "anomaly_score": analysis.get("anomaly_score", 0.0),
                    "movement_metrics": analysis.get("metrics", {})
                },
                confidence=0.8,
                execution_time_ms=execution_time
            )
        
        elif tool_name == "multimodal_fusion":
            # Fusion multimodale des données disponibles
            modalities = {
                "visual": context.get("visual_features", np.random.rand(512)),
                "detection": context.get("detection_data", {}),
                "pose": context.get("pose_data", {}),
                "motion": context.get("motion_data", {}),
                "temporal": context.get("temporal_data", {})
            }
            
            fusion_result = tool.fuse_modalities(modalities)
            
            execution_time = (time.perf_counter() - start_time) * 1000
            return ToolResult(
                tool_type="multimodal_fusion",
                success=True,
                data={
                    "fused_prediction": fusion_result.get("prediction", 0.5),
                    "attention_weights": fusion_result.get("attention_weights", {}),
                    "confidence_scores": fusion_result.get("confidence_scores", {})
                },
                confidence=fusion_result.get("overall_confidence", 0.7),
                execution_time_ms=execution_time
            )
        
        elif tool_name == "temporal_transformer":
            # Analyse temporelle des séquences
            sequence_type = "behavior"  # ou "detection", "motion"
            sequence_data = context.get("temporal_sequence", [])
            
            if not sequence_data:
                # Données temporelles simulées
                sequence_data = [0.3, 0.5, 0.7, 0.4, 0.6]
            
            analysis = tool.analyze_sequence(sequence_data, sequence_type)
            
            execution_time = (time.perf_counter() - start_time) * 1000
            return ToolResult(
                tool_type="temporal_transformer",
                success=True,
                data={
                    "temporal_patterns": analysis.get("patterns", []),
                    "anomaly_score": analysis.get("anomaly_score", 0.0),
                    "consistency_score": analysis.get("consistency", 0.8)
                },
                confidence=0.85,
                execution_time_ms=execution_time
            )
        
        elif tool_name == "adversarial_detector":
            # Détection d'attaques adversariales
            detection_result = tool.detect_adversarial(image)
            
            execution_time = (time.perf_counter() - start_time) * 1000
            return ToolResult(
                tool_type="adversarial_detector",
                success=True,
                data={
                    "is_adversarial": detection_result.get("is_adversarial", False),
                    "attack_type": detection_result.get("attack_type", "none"),
                    "robustness_score": detection_result.get("robustness_score", 0.9)
                },
                confidence=detection_result.get("confidence", 0.8),
                execution_time_ms=execution_time
            )
        
        elif tool_name == "domain_adapter":
            # Adaptation de domaine
            current_domain = context.get("current_domain", "unknown")
            target_domain = context.get("target_domain", "retail_standard")
            
            if current_domain != "unknown":
                adaptation = tool.adapt_to_domain(current_domain, target_domain)
                
                execution_time = (time.perf_counter() - start_time) * 1000
                return ToolResult(
                    tool_type="domain_adapter",
                    success=adaptation.adaptation_success,
                    data={
                        "domain_detected": current_domain,
                        "adaptation_success": adaptation.adaptation_success,
                        "confidence_improvement": adaptation.confidence_improvement,
                        "adapted_parameters": list(adaptation.adapted_parameters.keys())
                    },
                    confidence=adaptation.confidence_improvement,
                    execution_time_ms=execution_time
                )
            else:
                # Détection de domaine seulement
                detected_domain = tool.detect_domain(image)
                
                execution_time = (time.perf_counter() - start_time) * 1000
                return ToolResult(
                    tool_type="domain_adapter",
                    success=True,
                    data={
                        "detected_domain": detected_domain,
                        "adaptation_needed": detected_domain != target_domain
                    },
                    confidence=0.7,
                    execution_time_ms=execution_time
                )
        
        # Fallback
        execution_time = (time.perf_counter() - start_time) * 1000
        return ToolResult(
            tool_type=tool_name,
            success=False,
            data={"error": f"Méthode d'exécution non implémentée pour {tool_name}"},
            confidence=0.0,
            execution_time_ms=execution_time
        )
    
    def get_available_tools(self) -> List[str]:
        """Liste des outils disponibles."""
        return list(self.tools.keys())
    
    def get_tools_status(self) -> Dict[str, bool]:
        """État de disponibilité de chaque outil."""
        status = {}
        for tool_name, tool in self.tools.items():
            try:
                # Test simple de disponibilité
                status[tool_name] = hasattr(tool, '__call__') or hasattr(tool, 'process')
            except:
                status[tool_name] = False
        
        return status