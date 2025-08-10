"""Système de validation croisée multi-niveaux pour réduction des faux positifs."""

import time
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from loguru import logger

from ..core.types import (
    SurveillanceEvent,
    DetectionStatus,
    SuspicionLevel,
    ActionType,
    ToolResult,
    AnalysisResponse
)
from ..utils.exceptions import ValidationError


class ValidationLevel(Enum):
    """Niveaux de validation."""
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    CRITICAL = "critical"


class ValidationRule(Enum):
    """Types de règles de validation."""
    CONFIDENCE_THRESHOLD = "confidence_threshold"
    TEMPORAL_CONSISTENCY = "temporal_consistency"
    SPATIAL_CONSISTENCY = "spatial_consistency"
    BEHAVIORAL_PATTERN = "behavioral_pattern"
    MULTI_TOOL_CONSENSUS = "multi_tool_consensus"
    HISTORICAL_CONTEXT = "historical_context"
    ENVIRONMENTAL_CONTEXT = "environmental_context"


@dataclass
class ValidationCriteria:
    """Critères de validation."""
    rule_type: ValidationRule
    threshold: float
    weight: float = 1.0
    enabled: bool = True
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class ValidationResult:
    """Résultat d'une validation."""
    rule_type: ValidationRule
    passed: bool
    confidence: float
    score: float
    details: Dict[str, Any] = field(default_factory=dict)
    execution_time_ms: float = 0.0


class FalsePositivePredictor:
    """Prédicteur de faux positifs basé sur l'apprentissage."""
    
    def __init__(self):
        # Historique des validations pour apprentissage
        self.validation_history = []
        
        # Poids adaptatifs pour les règles
        self.adaptive_weights = {
            ValidationRule.CONFIDENCE_THRESHOLD: 1.0,
            ValidationRule.TEMPORAL_CONSISTENCY: 0.8,
            ValidationRule.SPATIAL_CONSISTENCY: 0.9,
            ValidationRule.BEHAVIORAL_PATTERN: 1.2,
            ValidationRule.MULTI_TOOL_CONSENSUS: 1.5,
            ValidationRule.HISTORICAL_CONTEXT: 0.7,
            ValidationRule.ENVIRONMENTAL_CONTEXT: 0.6
        }
        
        # Patterns de faux positifs appris
        self.false_positive_patterns = {
            "low_confidence_high_movement": {"weight": 0.3, "threshold": 0.4},
            "single_frame_detection": {"weight": 0.5, "threshold": 0.6},
            "inconsistent_tracking": {"weight": 0.4, "threshold": 0.5},
            "environmental_artifacts": {"weight": 0.2, "threshold": 0.3}
        }
    
    def predict_false_positive_probability(
        self, 
        event: SurveillanceEvent,
        tool_results: Dict[str, ToolResult],
        validation_results: List[ValidationResult]
    ) -> Tuple[float, List[str]]:
        """Prédiction de la probabilité de faux positif."""
        
        probability_scores = []
        reasons = []
        
        # Score basé sur la confiance générale
        if event.confidence < 0.5:
            probability_scores.append(0.6)
            reasons.append("Confiance de détection faible")
        
        # Score basé sur les résultats de validation
        failed_validations = [r for r in validation_results if not r.passed]
        if len(failed_validations) > len(validation_results) * 0.3:  # Plus de 30% échouent
            validation_failure_score = len(failed_validations) / len(validation_results)
            probability_scores.append(validation_failure_score)
            reasons.append(f"{len(failed_validations)} validations échouées")
        
        # Score basé sur la cohérence des outils
        successful_tools = [r for r in tool_results.values() if r.success]
        if len(successful_tools) < len(tool_results) * 0.7:  # Moins de 70% réussissent
            tool_failure_score = 1 - (len(successful_tools) / len(tool_results))
            probability_scores.append(tool_failure_score * 0.7)
            reasons.append("Outils peu cohérents")
        
        # Patterns spécifiques de faux positifs
        for pattern_name, pattern_config in self.false_positive_patterns.items():
            if self._matches_pattern(pattern_name, event, tool_results):
                probability_scores.append(pattern_config["weight"])
                reasons.append(f"Pattern détecté: {pattern_name}")
        
        # Score final
        if probability_scores:
            final_probability = np.mean(probability_scores)
        else:
            final_probability = 0.1  # Probabilité de base faible
        
        # Limitation à [0, 1]
        final_probability = np.clip(final_probability, 0.0, 1.0)
        
        return final_probability, reasons
    
    def _matches_pattern(
        self, 
        pattern_name: str, 
        event: SurveillanceEvent,
        tool_results: Dict[str, ToolResult]
    ) -> bool:
        """Vérification si un événement correspond à un pattern."""
        
        if pattern_name == "low_confidence_high_movement":
            # Confiance faible avec beaucoup de mouvement détecté
            tracker_result = tool_results.get("tracker")
            if tracker_result and tracker_result.success:
                active_tracks = tracker_result.data.get("active_tracks", [])
                if active_tracks and event.confidence < 0.4:
                    avg_track_age = np.mean([t.get("age", 0) for t in active_tracks])
                    return avg_track_age > 50  # Beaucoup de mouvement
        
        elif pattern_name == "single_frame_detection":
            # Détection sur un seul frame sans continuité
            tracker_result = tool_results.get("tracker")
            if tracker_result and tracker_result.success:
                active_tracks = tracker_result.data.get("active_tracks", [])
                single_frame_tracks = [t for t in active_tracks if t.get("hits", 0) == 1]
                return len(single_frame_tracks) > 0
        
        elif pattern_name == "inconsistent_tracking":
            # Tracking inconsistant
            tracker_result = tool_results.get("tracker")
            if tracker_result and tracker_result.success:
                active_tracks = tracker_result.data.get("active_tracks", [])
                for track in active_tracks:
                    hits = track.get("hits", 0)
                    age = track.get("age", 0)
                    if age > 20 and hits < age * 0.3:  # Moins de 30% de hits
                        return True
        
        elif pattern_name == "environmental_artifacts":
            # Artefacts environnementaux (éclairage, ombres, etc.)
            context_result = tool_results.get("context_validator")
            if context_result and context_result.success:
                validations = context_result.data.get("validations", {})
                time_context = validations.get("time_context", {})
                if time_context.get("period") == "night" and event.confidence < 0.6:
                    return True  # Détections nocturnes peu fiables
        
        return False
    
    def update_weights_from_feedback(
        self, 
        validation_results: List[ValidationResult],
        actual_false_positive: bool
    ) -> None:
        """Mise à jour des poids basée sur le feedback."""
        
        learning_rate = 0.05
        
        for result in validation_results:
            current_weight = self.adaptive_weights.get(result.rule_type, 1.0)
            
            if actual_false_positive:
                # Si c'était un faux positif, diminuer le poids des règles qui ont passé
                if result.passed:
                    new_weight = current_weight * (1 - learning_rate)
                else:
                    new_weight = current_weight * (1 + learning_rate)
            else:
                # Si c'était un vrai positif, augmenter le poids des règles qui ont passé
                if result.passed:
                    new_weight = current_weight * (1 + learning_rate)
                else:
                    new_weight = current_weight * (1 - learning_rate)
            
            # Limitation des poids
            new_weight = np.clip(new_weight, 0.1, 3.0)
            self.adaptive_weights[result.rule_type] = new_weight
        
        logger.info(f"Poids adaptatifs mis à jour (FP: {actual_false_positive})")


class CrossValidator:
    """
    Système de validation croisée multi-niveaux.
    
    Features:
    - Validation par règles configurables
    - Prédiction adaptive des faux positifs
    - Apprentissage continu
    - Métriques de performance détaillées
    """
    
    def __init__(
        self,
        target_false_positive_rate: float = 0.03,
        validation_timeout: float = 2.0
    ):
        self.target_false_positive_rate = target_false_positive_rate
        self.validation_timeout = validation_timeout
        
        # Prédicteur de faux positifs
        self.fp_predictor = FalsePositivePredictor()
        
        # Critères de validation par niveau
        self.validation_criteria = self._initialize_validation_criteria()
        
        # Métriques
        self.stats = {
            "total_validations": 0,
            "false_positives_detected": 0,
            "false_positive_rate": 0.0,
            "avg_validation_time": 0.0,
            "rule_performance": {}
        }
        
        # Cache des validations récentes
        self.recent_validations = {}
        self.cache_ttl = 30  # 30 secondes
        
        logger.info(f"CrossValidator initialisé (objectif FP: {target_false_positive_rate*100:.1f}%)")
    
    def _initialize_validation_criteria(self) -> Dict[ValidationLevel, List[ValidationCriteria]]:
        """Initialisation des critères de validation par niveau."""
        
        return {
            ValidationLevel.BASIC: [
                ValidationCriteria(
                    rule_type=ValidationRule.CONFIDENCE_THRESHOLD,
                    threshold=0.3,
                    weight=1.0,
                    parameters={"min_confidence": 0.3}
                ),
                ValidationCriteria(
                    rule_type=ValidationRule.MULTI_TOOL_CONSENSUS,
                    threshold=0.6,
                    weight=1.5,
                    parameters={"min_tools_agreement": 2}
                )
            ],
            
            ValidationLevel.INTERMEDIATE: [
                ValidationCriteria(
                    rule_type=ValidationRule.CONFIDENCE_THRESHOLD,
                    threshold=0.4,
                    weight=1.0
                ),
                ValidationCriteria(
                    rule_type=ValidationRule.TEMPORAL_CONSISTENCY,
                    threshold=0.7,
                    weight=0.8,
                    parameters={"time_window_seconds": 10, "min_consistency": 0.7}
                ),
                ValidationCriteria(
                    rule_type=ValidationRule.MULTI_TOOL_CONSENSUS,
                    threshold=0.7,
                    weight=1.5
                ),
                ValidationCriteria(
                    rule_type=ValidationRule.BEHAVIORAL_PATTERN,
                    threshold=0.5,
                    weight=1.2,
                    parameters={"pattern_analysis": True}
                )
            ],
            
            ValidationLevel.ADVANCED: [
                ValidationCriteria(
                    rule_type=ValidationRule.CONFIDENCE_THRESHOLD,
                    threshold=0.5,
                    weight=1.0
                ),
                ValidationCriteria(
                    rule_type=ValidationRule.TEMPORAL_CONSISTENCY,
                    threshold=0.8,
                    weight=0.8
                ),
                ValidationCriteria(
                    rule_type=ValidationRule.SPATIAL_CONSISTENCY,
                    threshold=0.7,
                    weight=0.9,
                    parameters={"spatial_analysis": True}
                ),
                ValidationCriteria(
                    rule_type=ValidationRule.MULTI_TOOL_CONSENSUS,
                    threshold=0.8,
                    weight=1.5
                ),
                ValidationCriteria(
                    rule_type=ValidationRule.BEHAVIORAL_PATTERN,
                    threshold=0.6,
                    weight=1.2
                ),
                ValidationCriteria(
                    rule_type=ValidationRule.HISTORICAL_CONTEXT,
                    threshold=0.6,
                    weight=0.7,
                    parameters={"history_window_hours": 24}
                )
            ],
            
            ValidationLevel.CRITICAL: [
                ValidationCriteria(
                    rule_type=ValidationRule.CONFIDENCE_THRESHOLD,
                    threshold=0.7,
                    weight=1.0
                ),
                ValidationCriteria(
                    rule_type=ValidationRule.TEMPORAL_CONSISTENCY,
                    threshold=0.9,
                    weight=0.8
                ),
                ValidationCriteria(
                    rule_type=ValidationRule.SPATIAL_CONSISTENCY,
                    threshold=0.8,
                    weight=0.9
                ),
                ValidationCriteria(
                    rule_type=ValidationRule.MULTI_TOOL_CONSENSUS,
                    threshold=0.9,
                    weight=1.5
                ),
                ValidationCriteria(
                    rule_type=ValidationRule.BEHAVIORAL_PATTERN,
                    threshold=0.8,
                    weight=1.2
                ),
                ValidationCriteria(
                    rule_type=ValidationRule.HISTORICAL_CONTEXT,
                    threshold=0.7,
                    weight=0.7
                ),
                ValidationCriteria(
                    rule_type=ValidationRule.ENVIRONMENTAL_CONTEXT,
                    threshold=0.8,
                    weight=0.6,
                    parameters={"environmental_analysis": True}
                )
            ]
        }
    
    async def validate_detection(
        self,
        event: SurveillanceEvent,
        tool_results: Dict[str, ToolResult],
        analysis_result: AnalysisResponse,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, float, List[str]]:
        """
        Validation croisée d'une détection.
        
        Args:
            event: Événement de surveillance à valider
            tool_results: Résultats des outils utilisés
            analysis_result: Résultat de l'analyse VLM
            context: Contexte additionnel
            
        Returns:
            Tuple (est_valide, confiance_finale, raisons)
        """
        
        start_time = time.time()
        
        try:
            # Détermination du niveau de validation selon la suspicion
            validation_level = self._determine_validation_level(analysis_result.suspicion_level)
            
            # Vérification du cache
            cache_key = self._get_cache_key(event, tool_results)
            cached_result = self._get_cached_validation(cache_key)
            if cached_result:
                return cached_result
            
            # Exécution des validations
            validation_results = await self._execute_validations(
                validation_level, event, tool_results, analysis_result, context
            )
            
            # Calcul du score de validation
            validation_score = self._calculate_validation_score(validation_results)
            
            # Prédiction de faux positif
            fp_probability, fp_reasons = self.fp_predictor.predict_false_positive_probability(
                event, tool_results, validation_results
            )
            
            # Décision finale
            is_valid = self._make_final_decision(
                validation_score, fp_probability, validation_level
            )
            
            # Calcul de la confiance finale
            final_confidence = self._calculate_final_confidence(
                validation_score, fp_probability, analysis_result.confidence
            )
            
            # Génération des raisons
            reasons = self._generate_validation_reasons(
                validation_results, fp_reasons, is_valid
            )
            
            # Mise en cache
            result = (is_valid, final_confidence, reasons)
            self._cache_validation(cache_key, result)
            
            # Mise à jour des statistiques
            execution_time = time.time() - start_time
            self._update_stats(execution_time, validation_results, fp_probability)
            
            logger.debug(
                f"Validation {validation_level.value}: "
                f"{'✓' if is_valid else '✗'} "
                f"(confiance: {final_confidence:.3f}, FP: {fp_probability:.3f})"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Erreur validation croisée: {e}")
            # En cas d'erreur, rejeter par sécurité
            return False, 0.0, [f"Erreur de validation: {str(e)}"]
    
    def _determine_validation_level(self, suspicion_level: SuspicionLevel) -> ValidationLevel:
        """Détermination du niveau de validation selon la suspicion."""
        
        mapping = {
            SuspicionLevel.LOW: ValidationLevel.BASIC,
            SuspicionLevel.MEDIUM: ValidationLevel.INTERMEDIATE,
            SuspicionLevel.HIGH: ValidationLevel.ADVANCED,
            SuspicionLevel.CRITICAL: ValidationLevel.CRITICAL
        }
        
        return mapping.get(suspicion_level, ValidationLevel.BASIC)
    
    async def _execute_validations(
        self,
        validation_level: ValidationLevel,
        event: SurveillanceEvent,
        tool_results: Dict[str, ToolResult],
        analysis_result: AnalysisResponse,
        context: Optional[Dict[str, Any]]
    ) -> List[ValidationResult]:
        """Exécution des validations selon le niveau."""
        
        criteria_list = self.validation_criteria.get(validation_level, [])
        validation_results = []
        
        # Exécution des validations en parallèle
        tasks = []
        for criteria in criteria_list:
            if criteria.enabled:
                task = self._execute_single_validation(
                    criteria, event, tool_results, analysis_result, context
                )
                tasks.append(task)
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, Exception):
                    logger.warning(f"Erreur validation: {result}")
                else:
                    validation_results.append(result)
        
        return validation_results
    
    async def _execute_single_validation(
        self,
        criteria: ValidationCriteria,
        event: SurveillanceEvent,
        tool_results: Dict[str, ToolResult],
        analysis_result: AnalysisResponse,
        context: Optional[Dict[str, Any]]
    ) -> ValidationResult:
        """Exécution d'une validation unique."""
        
        start_time = time.time()
        
        try:
            if criteria.rule_type == ValidationRule.CONFIDENCE_THRESHOLD:
                result = await self._validate_confidence_threshold(
                    criteria, event, analysis_result
                )
            
            elif criteria.rule_type == ValidationRule.TEMPORAL_CONSISTENCY:
                result = await self._validate_temporal_consistency(
                    criteria, event, tool_results, context
                )
            
            elif criteria.rule_type == ValidationRule.SPATIAL_CONSISTENCY:
                result = await self._validate_spatial_consistency(
                    criteria, event, tool_results, context
                )
            
            elif criteria.rule_type == ValidationRule.BEHAVIORAL_PATTERN:
                result = await self._validate_behavioral_pattern(
                    criteria, event, tool_results
                )
            
            elif criteria.rule_type == ValidationRule.MULTI_TOOL_CONSENSUS:
                result = await self._validate_multi_tool_consensus(
                    criteria, tool_results
                )
            
            elif criteria.rule_type == ValidationRule.HISTORICAL_CONTEXT:
                result = await self._validate_historical_context(
                    criteria, event, context
                )
            
            elif criteria.rule_type == ValidationRule.ENVIRONMENTAL_CONTEXT:
                result = await self._validate_environmental_context(
                    criteria, event, context
                )
            
            else:
                result = ValidationResult(
                    rule_type=criteria.rule_type,
                    passed=True,
                    confidence=0.5,
                    score=0.5,
                    details={"error": f"Règle non implémentée: {criteria.rule_type}"}
                )
            
            result.execution_time_ms = (time.time() - start_time) * 1000
            return result
            
        except Exception as e:
            logger.error(f"Erreur validation {criteria.rule_type.value}: {e}")
            return ValidationResult(
                rule_type=criteria.rule_type,
                passed=False,
                confidence=0.0,
                score=0.0,
                details={"error": str(e)},
                execution_time_ms=(time.time() - start_time) * 1000
            )
    
    # Implémentation des règles de validation spécifiques
    
    async def _validate_confidence_threshold(
        self, 
        criteria: ValidationCriteria,
        event: SurveillanceEvent,
        analysis_result: AnalysisResponse
    ) -> ValidationResult:
        """Validation du seuil de confiance."""
        
        min_confidence = criteria.parameters.get("min_confidence", criteria.threshold)
        
        # Score basé sur les confidences disponibles
        confidences = [event.confidence, analysis_result.confidence]
        avg_confidence = np.mean(confidences)
        
        passed = avg_confidence >= min_confidence
        score = min(avg_confidence / min_confidence, 1.0) if min_confidence > 0 else 1.0
        
        return ValidationResult(
            rule_type=criteria.rule_type,
            passed=passed,
            confidence=avg_confidence,
            score=score,
            details={
                "avg_confidence": avg_confidence,
                "min_confidence": min_confidence,
                "confidences": confidences
            }
        )
    
    async def _validate_temporal_consistency(
        self,
        criteria: ValidationCriteria,
        event: SurveillanceEvent,
        tool_results: Dict[str, ToolResult],
        context: Optional[Dict[str, Any]]
    ) -> ValidationResult:
        """Validation de la cohérence temporelle."""
        
        time_window = criteria.parameters.get("time_window_seconds", 10)
        min_consistency = criteria.parameters.get("min_consistency", 0.7)
        
        # Analyse des tracks pour cohérence temporelle
        tracker_result = tool_results.get("tracker")
        if not tracker_result or not tracker_result.success:
            return ValidationResult(
                rule_type=criteria.rule_type,
                passed=True,  # Pas de données = pas d'incohérence
                confidence=0.5,
                score=0.5,
                details={"error": "Pas de données de tracking"}
            )
        
        active_tracks = tracker_result.data.get("active_tracks", [])
        
        # Calcul de cohérence basé sur les hits vs age
        consistency_scores = []
        for track in active_tracks:
            hits = track.get("hits", 0)
            age = track.get("age", 1)
            consistency = hits / age if age > 0 else 0
            consistency_scores.append(consistency)
        
        if consistency_scores:
            avg_consistency = np.mean(consistency_scores)
            passed = avg_consistency >= min_consistency
            score = min(avg_consistency / min_consistency, 1.0)
        else:
            avg_consistency = 1.0
            passed = True
            score = 1.0
        
        return ValidationResult(
            rule_type=criteria.rule_type,
            passed=passed,
            confidence=avg_consistency,
            score=score,
            details={
                "avg_consistency": avg_consistency,
                "min_consistency": min_consistency,
                "track_count": len(active_tracks)
            }
        )
    
    async def _validate_spatial_consistency(
        self,
        criteria: ValidationCriteria,
        event: SurveillanceEvent,
        tool_results: Dict[str, ToolResult],
        context: Optional[Dict[str, Any]]
    ) -> ValidationResult:
        """Validation de la cohérence spatiale."""
        
        # Analyse simple de cohérence spatiale
        detector_result = tool_results.get("object_detector")
        if not detector_result or not detector_result.success:
            return ValidationResult(
                rule_type=criteria.rule_type,
                passed=True,
                confidence=0.5,
                score=0.5,
                details={"error": "Pas de données de détection"}
            )
        
        detections = detector_result.data.get("detections", [])
        
        # Vérification de la distribution spatiale des détections
        if len(detections) < 2:
            spatial_score = 1.0
        else:
            # Calcul de la dispersion spatiale
            centers = []
            for det in detections:
                bbox = det["bbox"]
                center_x = bbox["x"] + bbox["width"] // 2
                center_y = bbox["y"] + bbox["height"] // 2
                centers.append((center_x, center_y))
            
            # Variance de la position
            centers_array = np.array(centers)
            variance = np.var(centers_array, axis=0)
            spatial_dispersion = np.mean(variance)
            
            # Score inversé (moins de dispersion = plus de cohérence)
            spatial_score = 1.0 / (1.0 + spatial_dispersion / 10000)  # Normalisation
        
        passed = spatial_score >= criteria.threshold
        
        return ValidationResult(
            rule_type=criteria.rule_type,
            passed=passed,
            confidence=spatial_score,
            score=spatial_score,
            details={
                "spatial_score": spatial_score,
                "detection_count": len(detections)
            }
        )
    
    async def _validate_behavioral_pattern(
        self,
        criteria: ValidationCriteria,
        event: SurveillanceEvent,
        tool_results: Dict[str, ToolResult]
    ) -> ValidationResult:
        """Validation des patterns comportementaux."""
        
        behavior_result = tool_results.get("behavior_analyzer")
        if not behavior_result or not behavior_result.success:
            return ValidationResult(
                rule_type=criteria.rule_type,
                passed=True,
                confidence=0.5,
                score=0.5,
                details={"error": "Pas d'analyse comportementale"}
            )
        
        suspicious_behaviors = behavior_result.data.get("suspicious_behaviors", [])
        suspicion_score = behavior_result.data.get("suspicion_score", 0.0)
        
        # Validation basée sur la cohérence des comportements suspects
        if event.action_type in [ActionType.POTENTIAL_THEFT, ActionType.CONFIRMED_THEFT]:
            # Pour les vols potentiels, on attend des comportements suspects
            expected_suspicious = True
            behavior_score = suspicion_score if suspicious_behaviors else 0.2
        else:
            # Pour les actions normales, on n'attend pas de comportements suspects
            expected_suspicious = False
            behavior_score = 1.0 - suspicion_score
        
        passed = behavior_score >= criteria.threshold
        
        return ValidationResult(
            rule_type=criteria.rule_type,
            passed=passed,
            confidence=behavior_score,
            score=behavior_score,
            details={
                "suspicious_behaviors_count": len(suspicious_behaviors),
                "suspicion_score": suspicion_score,
                "expected_suspicious": expected_suspicious
            }
        )
    
    async def _validate_multi_tool_consensus(
        self,
        criteria: ValidationCriteria,
        tool_results: Dict[str, ToolResult]
    ) -> ValidationResult:
        """Validation du consensus entre outils."""
        
        min_tools_agreement = criteria.parameters.get("min_tools_agreement", 2)
        
        # Compte des outils ayant réussi
        successful_tools = [r for r in tool_results.values() if r.success]
        total_tools = len(tool_results)
        
        if total_tools == 0:
            consensus_score = 0.0
        else:
            consensus_score = len(successful_tools) / total_tools
        
        # Analyse de la cohérence des confidences
        confidences = []
        for tool_result in successful_tools:
            if tool_result.confidence is not None:
                confidences.append(tool_result.confidence)
        
        if confidences:
            confidence_variance = np.var(confidences)
            confidence_consistency = 1.0 / (1.0 + confidence_variance)
        else:
            confidence_consistency = 1.0
        
        # Score final combinant consensus et cohérence
        final_score = consensus_score * confidence_consistency
        passed = (
            len(successful_tools) >= min_tools_agreement and 
            final_score >= criteria.threshold
        )
        
        return ValidationResult(
            rule_type=criteria.rule_type,
            passed=passed,
            confidence=final_score,
            score=final_score,
            details={
                "successful_tools": len(successful_tools),
                "total_tools": total_tools,
                "consensus_score": consensus_score,
                "confidence_consistency": confidence_consistency,
                "min_tools_agreement": min_tools_agreement
            }
        )
    
    async def _validate_historical_context(
        self,
        criteria: ValidationCriteria,
        event: SurveillanceEvent,
        context: Optional[Dict[str, Any]]
    ) -> ValidationResult:
        """Validation du contexte historique."""
        
        # Implémentation simple basée sur l'heure et la localisation
        current_hour = datetime.now().hour
        location = context.get("location", "unknown") if context else "unknown"
        
        # Facteurs de risque historiques
        risk_factors = []
        
        # Analyse temporelle
        if 22 <= current_hour or current_hour <= 6:  # Nuit
            risk_factors.append({"factor": "night_time", "weight": 0.3})
        
        if 12 <= current_hour <= 14:  # Heure de pointe
            risk_factors.append({"factor": "peak_hours", "weight": -0.1})  # Moins suspect
        
        # Analyse de localisation
        high_risk_locations = ["electronics", "jewelry", "alcohol", "pharmacy"]
        if location in high_risk_locations:
            risk_factors.append({"factor": "high_risk_location", "weight": 0.2})
        
        # Calcul du score historique
        if risk_factors:
            total_weight = sum(rf["weight"] for rf in risk_factors)
            historical_score = 0.5 + total_weight  # Score de base 0.5
        else:
            historical_score = 0.7  # Score par défaut
        
        historical_score = np.clip(historical_score, 0.0, 1.0)
        passed = historical_score >= criteria.threshold
        
        return ValidationResult(
            rule_type=criteria.rule_type,
            passed=passed,
            confidence=historical_score,
            score=historical_score,
            details={
                "current_hour": current_hour,
                "location": location,
                "risk_factors": risk_factors,
                "historical_score": historical_score
            }
        )
    
    async def _validate_environmental_context(
        self,
        criteria: ValidationCriteria,
        event: SurveillanceEvent,
        context: Optional[Dict[str, Any]]
    ) -> ValidationResult:
        """Validation du contexte environnemental."""
        
        # Validation basée sur les conditions environnementales
        environmental_score = 0.8  # Score par défaut
        
        factors = []
        
        # Analyse de la qualité de l'image (si disponible)
        if context:
            image_quality = context.get("image_quality", "good")
            if image_quality == "poor":
                factors.append({"factor": "poor_image_quality", "impact": -0.2})
                environmental_score -= 0.2
            elif image_quality == "excellent":
                factors.append({"factor": "excellent_image_quality", "impact": 0.1})
                environmental_score += 0.1
            
            # Conditions d'éclairage
            lighting = context.get("lighting", "normal")
            if lighting == "low":
                factors.append({"factor": "low_lighting", "impact": -0.15})
                environmental_score -= 0.15
            elif lighting == "bright":
                factors.append({"factor": "bright_lighting", "impact": 0.05})
                environmental_score += 0.05
            
            # Densité de la foule
            crowd_density = context.get("crowd_density", "normal")
            if crowd_density == "high":
                factors.append({"factor": "high_crowd_density", "impact": -0.1})
                environmental_score -= 0.1
            elif crowd_density == "low":
                factors.append({"factor": "low_crowd_density", "impact": 0.05})
                environmental_score += 0.05
        
        environmental_score = np.clip(environmental_score, 0.0, 1.0)
        passed = environmental_score >= criteria.threshold
        
        return ValidationResult(
            rule_type=criteria.rule_type,
            passed=passed,
            confidence=environmental_score,
            score=environmental_score,
            details={
                "environmental_score": environmental_score,
                "factors": factors
            }
        )
    
    def _calculate_validation_score(self, validation_results: List[ValidationResult]) -> float:
        """Calcul du score de validation pondéré."""
        
        if not validation_results:
            return 0.0
        
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for result in validation_results:
            # Récupération du poids adaptatif
            adaptive_weight = self.fp_predictor.adaptive_weights.get(result.rule_type, 1.0)
            
            # Score pondéré
            if result.passed:
                weighted_score = result.score * adaptive_weight
            else:
                weighted_score = result.score * adaptive_weight * 0.5  # Pénalité pour échec
            
            total_weighted_score += weighted_score
            total_weight += adaptive_weight
        
        return total_weighted_score / total_weight if total_weight > 0 else 0.0
    
    def _make_final_decision(
        self,
        validation_score: float,
        fp_probability: float,
        validation_level: ValidationLevel
    ) -> bool:
        """Prise de décision finale."""
        
        # Seuils de décision par niveau
        decision_thresholds = {
            ValidationLevel.BASIC: 0.4,
            ValidationLevel.INTERMEDIATE: 0.6,
            ValidationLevel.ADVANCED: 0.7,
            ValidationLevel.CRITICAL: 0.8
        }
        
        threshold = decision_thresholds.get(validation_level, 0.6)
        
        # Décision basée sur le score de validation et la probabilité de FP
        adjusted_score = validation_score * (1.0 - fp_probability)
        
        return adjusted_score >= threshold
    
    def _calculate_final_confidence(
        self,
        validation_score: float,
        fp_probability: float,
        original_confidence: float
    ) -> float:
        """Calcul de la confiance finale."""
        
        # Combinaison des scores avec pondération
        final_confidence = (
            0.4 * original_confidence +
            0.4 * validation_score +
            0.2 * (1.0 - fp_probability)
        )
        
        return np.clip(final_confidence, 0.0, 1.0)
    
    def _generate_validation_reasons(
        self,
        validation_results: List[ValidationResult],
        fp_reasons: List[str],
        is_valid: bool
    ) -> List[str]:
        """Génération des raisons de validation."""
        
        reasons = []
        
        # Raisons basées sur les validations
        passed_validations = [r for r in validation_results if r.passed]
        failed_validations = [r for r in validation_results if not r.passed]
        
        if is_valid:
            reasons.append(f"{len(passed_validations)}/{len(validation_results)} validations réussies")
            
            if passed_validations:
                top_validations = sorted(passed_validations, key=lambda x: x.score, reverse=True)[:2]
                for val in top_validations:
                    reasons.append(f"✓ {val.rule_type.value} (score: {val.score:.3f})")
        else:
            reasons.append(f"Validation échouée ({len(failed_validations)}/{len(validation_results)} échecs)")
            
            if failed_validations:
                top_failures = sorted(failed_validations, key=lambda x: x.score)[:2]
                for val in top_failures:
                    reasons.append(f"✗ {val.rule_type.value} (score: {val.score:.3f})")
        
        # Ajout des raisons de faux positifs
        if fp_reasons:
            reasons.extend(fp_reasons)
        
        return reasons
    
    def _get_cache_key(
        self, 
        event: SurveillanceEvent, 
        tool_results: Dict[str, ToolResult]
    ) -> str:
        """Génération d'une clé de cache."""
        
        # Clé basée sur l'événement et les résultats d'outils
        tools_signature = "_".join(sorted([
            f"{name}:{result.success}:{result.confidence or 0:.2f}"
            for name, result in tool_results.items()
        ]))
        
        return f"{event.stream_id}_{event.frame_id}_{event.action_type.value}_{tools_signature}"
    
    def _get_cached_validation(self, cache_key: str) -> Optional[Tuple[bool, float, List[str]]]:
        """Récupération d'une validation mise en cache."""
        
        if cache_key in self.recent_validations:
            cached_data, timestamp = self.recent_validations[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                return cached_data
            else:
                del self.recent_validations[cache_key]
        
        return None
    
    def _cache_validation(
        self, 
        cache_key: str, 
        result: Tuple[bool, float, List[str]]
    ) -> None:
        """Mise en cache d'une validation."""
        
        self.recent_validations[cache_key] = (result, time.time())
        
        # Nettoyage du cache si trop volumineux
        if len(self.recent_validations) > 100:
            # Supprimer les entrées les plus anciennes
            sorted_keys = sorted(
                self.recent_validations.keys(),
                key=lambda k: self.recent_validations[k][1]
            )
            for key in sorted_keys[:50]:
                del self.recent_validations[key]
    
    def _update_stats(
        self,
        execution_time: float,
        validation_results: List[ValidationResult],
        fp_probability: float
    ) -> None:
        """Mise à jour des statistiques."""
        
        self.stats["total_validations"] += 1
        
        # Temps d'exécution moyen
        if self.stats["avg_validation_time"] == 0:
            self.stats["avg_validation_time"] = execution_time
        else:
            alpha = 0.1
            self.stats["avg_validation_time"] = (
                alpha * execution_time + 
                (1 - alpha) * self.stats["avg_validation_time"]
            )
        
        # Statistiques de faux positifs
        if fp_probability > 0.5:  # Considéré comme faux positif
            self.stats["false_positives_detected"] += 1
        
        self.stats["false_positive_rate"] = (
            self.stats["false_positives_detected"] / self.stats["total_validations"]
        )
        
        # Performance par règle
        for result in validation_results:
            rule_name = result.rule_type.value
            if rule_name not in self.stats["rule_performance"]:
                self.stats["rule_performance"][rule_name] = {
                    "total": 0,
                    "passed": 0,
                    "avg_score": 0.0,
                    "avg_execution_time": 0.0
                }
            
            rule_stats = self.stats["rule_performance"][rule_name]
            rule_stats["total"] += 1
            
            if result.passed:
                rule_stats["passed"] += 1
            
            # Score moyen
            if rule_stats["avg_score"] == 0:
                rule_stats["avg_score"] = result.score
            else:
                alpha = 0.1
                rule_stats["avg_score"] = (
                    alpha * result.score + 
                    (1 - alpha) * rule_stats["avg_score"]
                )
            
            # Temps d'exécution moyen
            if rule_stats["avg_execution_time"] == 0:
                rule_stats["avg_execution_time"] = result.execution_time_ms
            else:
                alpha = 0.1
                rule_stats["avg_execution_time"] = (
                    alpha * result.execution_time_ms + 
                    (1 - alpha) * rule_stats["avg_execution_time"]
                )
    
    def provide_feedback(
        self,
        event_id: str,
        was_false_positive: bool,
        validation_results: Optional[List[ValidationResult]] = None
    ) -> None:
        """Feedback pour l'apprentissage adaptatif."""
        
        if validation_results:
            self.fp_predictor.update_weights_from_feedback(
                validation_results, was_false_positive
            )
        
        # Mise à jour des patterns si c'était un faux positif
        if was_false_positive:
            self.stats["false_positives_detected"] += 1
            self.stats["false_positive_rate"] = (
                self.stats["false_positives_detected"] / self.stats["total_validations"]
            )
        
        logger.info(f"Feedback reçu pour {event_id}: FP={was_false_positive}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Récupération des statistiques."""
        stats = self.stats.copy()
        stats["adaptive_weights"] = self.fp_predictor.adaptive_weights.copy()
        stats["cache_size"] = len(self.recent_validations)
        return stats
    
    def adjust_target_fp_rate(self, new_target: float) -> None:
        """Ajustement de l'objectif de taux de faux positifs."""
        
        self.target_false_positive_rate = new_target
        
        # Ajustement des seuils de validation si nécessaire
        adjustment_factor = new_target / 0.03  # Par rapport au défaut
        
        for level, criteria_list in self.validation_criteria.items():
            for criteria in criteria_list:
                criteria.threshold *= adjustment_factor
                criteria.threshold = np.clip(criteria.threshold, 0.1, 0.95)
        
        logger.info(f"Objectif FP ajusté à {new_target*100:.1f}%")
    
    def reset_stats(self) -> None:
        """Réinitialisation des statistiques."""
        
        self.stats = {
            "total_validations": 0,
            "false_positives_detected": 0,
            "false_positive_rate": 0.0,
            "avg_validation_time": 0.0,
            "rule_performance": {}
        }
        
        self.recent_validations.clear()
        logger.info("Statistiques de validation réinitialisées")