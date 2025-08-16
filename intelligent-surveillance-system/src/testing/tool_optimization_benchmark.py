"""
🧪 Système de Benchmarking et Optimisation des Outils VLM
========================================================

Ce module teste automatiquement différentes combinaisons d'outils pour déterminer
quels outils permettent au VLM d'obtenir les meilleurs résultats de détection.

Méthodes de test :
1. A/B Testing avec combinaisons d'outils
2. Analyse de performance par outil
3. Optimisation basée sur les métriques de qualité
4. Sélection automatique des outils optimaux
"""

import asyncio
import time
import json
import numpy as np
from typing import Dict, List, Tuple, Set, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import itertools
from collections import defaultdict
import statistics

from loguru import logger
from sklearn.metrics import precision_score, recall_score, f1_score

from ..core.types import AnalysisRequest, AnalysisResponse, SuspicionLevel, ActionType
from ..core.vlm.dynamic_model import DynamicVisionLanguageModel
from ..core.orchestrator.vlm_orchestrator import ModernVLMOrchestrator, OrchestrationConfig, OrchestrationMode


@dataclass
class ToolPerformanceMetrics:
    """Métriques de performance d'un outil."""
    tool_name: str
    usage_count: int = 0
    success_rate: float = 0.0
    avg_execution_time: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    contribution_score: float = 0.0  # Score de contribution aux bonnes détections
    cost_benefit_ratio: float = 0.0  # Rapport coût/bénéfice


@dataclass
class ToolCombinationResult:
    """Résultat d'un test de combinaison d'outils."""
    tools_combination: List[str]
    test_count: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    avg_response_time: float
    false_positive_rate: float
    false_negative_rate: float
    total_cost: float  # Coût computationnel
    quality_score: float  # Score qualité global


@dataclass
class BenchmarkTestCase:
    """Cas de test pour le benchmark."""
    test_id: str
    frame_data: str  # Base64 image
    ground_truth: Dict[str, Any]  # Vérité terrain
    scenario_type: str  # "theft", "normal", "suspicious", etc.
    difficulty_level: int  # 1-5
    expected_suspicion: SuspicionLevel
    expected_action: ActionType


class ToolOptimizationBenchmark:
    """
    Système de benchmark pour optimiser la sélection d'outils VLM.
    
    Features:
    - Test automatique de combinaisons d'outils
    - Métriques de performance détaillées
    - Optimisation basée sur coût/bénéfice
    - Sélection automatique des outils optimaux
    """
    
    def __init__(
        self,
        vlm_model_name: str = "kimi-vl-a3b-thinking",
        test_data_path: Optional[Path] = None
    ):
        self.vlm_model_name = vlm_model_name
        self.test_data_path = test_data_path or Path("data/benchmark/tool_optimization")
        
        # Outils disponibles avec leur coût computationnel relatif
        self.available_tools = {
            "sam2_segmentator": {"cost": 3.0, "category": "vision"},
            "dino_features": {"cost": 2.5, "category": "features"},
            "pose_estimator": {"cost": 2.0, "category": "human"},
            "trajectory_analyzer": {"cost": 1.5, "category": "temporal"},
            "multimodal_fusion": {"cost": 2.0, "category": "fusion"},
            "temporal_transformer": {"cost": 3.5, "category": "temporal"},
            "adversarial_detector": {"cost": 2.5, "category": "security"},
            "domain_adapter": {"cost": 1.0, "category": "adaptation"}
        }
        
        # Résultats des tests
        self.benchmark_results: List[ToolCombinationResult] = []
        self.tool_metrics: Dict[str, ToolPerformanceMetrics] = {}
        self.test_cases: List[BenchmarkTestCase] = []
        
        # Configuration
        self.max_combinations_to_test = 50  # Limite pour éviter explosion combinatoire
        self.min_test_cases_per_combination = 10
        
        logger.info("ToolOptimizationBenchmark initialisé")
    
    async def load_test_cases(self) -> None:
        """Chargement des cas de test."""
        
        # Création des cas de test synthétiques si pas de données
        if not self.test_data_path.exists() or len(list(self.test_data_path.glob("*.json"))) == 0:
            logger.warning("Aucun cas de test trouvé, génération de cas synthétiques")
            await self._generate_synthetic_test_cases()
        else:
            await self._load_test_cases_from_files()
        
        logger.info(f"✅ {len(self.test_cases)} cas de test chargés")
    
    async def _generate_synthetic_test_cases(self) -> None:
        """Génération de cas de test synthétiques."""
        
        scenarios = [
            {
                "type": "normal_shopping",
                "suspicion": SuspicionLevel.LOW,
                "action": ActionType.NORMAL_SHOPPING,
                "count": 15
            },
            {
                "type": "suspicious_behavior",
                "suspicion": SuspicionLevel.MEDIUM,
                "action": ActionType.SUSPICIOUS_ACTIVITY,
                "count": 12
            },
            {
                "type": "theft_attempt",
                "suspicion": SuspicionLevel.HIGH,
                "action": ActionType.THEFT_ATTEMPT,
                "count": 8
            },
            {
                "type": "critical_situation",
                "suspicion": SuspicionLevel.CRITICAL,
                "action": ActionType.THEFT_ATTEMPT,
                "count": 5
            }
        ]
        
        test_id = 0
        for scenario in scenarios:
            for i in range(scenario["count"]):
                self.test_cases.append(BenchmarkTestCase(
                    test_id=f"synthetic_{test_id:03d}",
                    frame_data="",  # Image synthétique vide pour test
                    ground_truth={
                        "has_person": scenario["type"] != "empty_scene",
                        "suspicious_activity": scenario["suspicion"] in [SuspicionLevel.MEDIUM, SuspicionLevel.HIGH, SuspicionLevel.CRITICAL],
                        "theft_detected": scenario["action"] == ActionType.THEFT_ATTEMPT,
                        "confidence": 0.8 + (i * 0.02)  # Variation de confiance
                    },
                    scenario_type=scenario["type"],
                    difficulty_level=min(5, int(scenario["suspicion"].value) + 1),
                    expected_suspicion=scenario["suspicion"],
                    expected_action=scenario["action"]
                ))
                test_id += 1
    
    async def _load_test_cases_from_files(self) -> None:
        """Chargement des cas de test depuis fichiers."""
        
        for test_file in self.test_data_path.glob("*.json"):
            try:
                with open(test_file, 'r') as f:
                    test_data = json.load(f)
                
                test_case = BenchmarkTestCase(
                    test_id=test_data["test_id"],
                    frame_data=test_data["frame_data"],
                    ground_truth=test_data["ground_truth"],
                    scenario_type=test_data["scenario_type"],
                    difficulty_level=test_data["difficulty_level"],
                    expected_suspicion=SuspicionLevel(test_data["expected_suspicion"]),
                    expected_action=ActionType(test_data["expected_action"])
                )
                
                self.test_cases.append(test_case)
                
            except Exception as e:
                logger.error(f"Erreur chargement {test_file}: {e}")
    
    def generate_tool_combinations(self) -> List[List[str]]:
        """Génération intelligente des combinaisons d'outils à tester."""
        
        all_tools = list(self.available_tools.keys())
        combinations = []
        
        # 1. Combinaisons par catégorie
        categories = defaultdict(list)
        for tool, info in self.available_tools.items():
            categories[info["category"]].append(tool)
        
        # Une combinaison par catégorie
        for category, tools in categories.items():
            if tools:
                combinations.append(tools[:2])  # Max 2 outils par catégorie
        
        # 2. Combinaisons par taille (1-4 outils)
        for size in range(1, 5):
            # Sélection basée sur coût/bénéfice potentiel
            sorted_tools = sorted(
                all_tools, 
                key=lambda t: self.available_tools[t]["cost"]
            )
            combinations.append(sorted_tools[:size])
        
        # 3. Combinaisons équilibrées
        low_cost = [t for t, info in self.available_tools.items() if info["cost"] <= 2.0]
        medium_cost = [t for t, info in self.available_tools.items() if 2.0 < info["cost"] <= 2.5]
        high_cost = [t for t, info in self.available_tools.items() if info["cost"] > 2.5]
        
        combinations.extend([
            low_cost[:3],  # Combinaison économique
            medium_cost[:2] + low_cost[:1],  # Combinaison équilibrée
            high_cost[:1] + medium_cost[:1] + low_cost[:1],  # Combinaison premium
        ])
        
        # 4. Combinaisons spécialisées
        vision_tools = [t for t, info in self.available_tools.items() if info["category"] == "vision"]
        security_tools = [t for t, info in self.available_tools.items() if info["category"] == "security"]
        
        if vision_tools:
            combinations.append(vision_tools)
        if security_tools:
            combinations.append(security_tools)
        
        # Déduplication et limitation
        unique_combinations = []
        for combo in combinations:
            if combo and combo not in unique_combinations:
                unique_combinations.append(combo)
        
        # Limitation du nombre de combinaisons
        if len(unique_combinations) > self.max_combinations_to_test:
            # Priorisation par diversité et coût
            scored_combinations = []
            for combo in unique_combinations:
                diversity_score = len(set(self.available_tools[t]["category"] for t in combo))
                cost_score = sum(self.available_tools[t]["cost"] for t in combo)
                total_score = diversity_score / (cost_score + 1)  # Favorise diversité/coût bas
                scored_combinations.append((total_score, combo))
            
            scored_combinations.sort(key=lambda x: x[0], reverse=True)
            unique_combinations = [combo for _, combo in scored_combinations[:self.max_combinations_to_test]]
        
        logger.info(f"🧪 {len(unique_combinations)} combinaisons d'outils générées pour test")
        return unique_combinations
    
    async def test_tool_combination(
        self, 
        tools_combination: List[str],
        test_cases: List[BenchmarkTestCase]
    ) -> ToolCombinationResult:
        """Test d'une combinaison spécifique d'outils."""
        
        logger.info(f"🧪 Test combinaison: {', '.join(tools_combination)}")
        
        # Configuration de l'orchestrateur avec la combinaison
        config = OrchestrationConfig(
            mode=OrchestrationMode.BALANCED,
            enable_advanced_tools=True,
            max_concurrent_tools=len(tools_combination),
            confidence_threshold=0.7
        )
        
        orchestrator = ModernVLMOrchestrator(
            vlm_model_name=self.vlm_model_name,
            config=config
        )
        
        # Override de la sélection d'outils pour forcer la combinaison
        def force_tool_selection():
            return tools_combination
        
        orchestrator._select_tools_for_mode = force_tool_selection
        
        # Variables de résultats
        results = []
        response_times = []
        start_time = time.time()
        
        # Test sur les cas de test
        for test_case in test_cases:
            try:
                case_start = time.time()
                
                # Analyse avec la combinaison d'outils
                analysis = await orchestrator.analyze_surveillance_frame(
                    frame_data=test_case.frame_data or "test_frame",  # Image test
                    detections=[],
                    context={"test_case_id": test_case.test_id}
                )
                
                case_time = time.time() - case_start
                response_times.append(case_time)
                
                # Évaluation des résultats
                result = self._evaluate_analysis_result(analysis, test_case)
                results.append(result)
                
            except Exception as e:
                logger.error(f"Erreur test {test_case.test_id}: {e}")
                results.append({
                    "accuracy": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "false_positive": True
                })
                response_times.append(10.0)  # Timeout simulé
        
        # Calcul des métriques globales
        if results:
            accuracy = np.mean([r["accuracy"] for r in results])
            precision = np.mean([r["precision"] for r in results])
            recall = np.mean([r["recall"] for r in results])
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            false_positive_rate = np.mean([r.get("false_positive", False) for r in results])
            false_negative_rate = np.mean([r.get("false_negative", False) for r in results])
        else:
            accuracy = precision = recall = f1 = 0.0
            false_positive_rate = false_negative_rate = 1.0
        
        avg_response_time = np.mean(response_times) if response_times else 10.0
        total_cost = sum(self.available_tools[tool]["cost"] for tool in tools_combination)
        
        # Score qualité global (pondération des métriques)
        quality_score = (
            0.3 * accuracy + 
            0.25 * precision + 
            0.25 * recall + 
            0.2 * f1
        ) - (0.1 * false_positive_rate)  # Pénalité faux positifs
        
        # Ajustement selon coût/bénéfice
        cost_penalty = min(0.2, total_cost / 20.0)  # Pénalité coût
        quality_score = max(0.0, quality_score - cost_penalty)
        
        result = ToolCombinationResult(
            tools_combination=tools_combination,
            test_count=len(test_cases),
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            avg_response_time=avg_response_time,
            false_positive_rate=false_positive_rate,
            false_negative_rate=false_negative_rate,
            total_cost=total_cost,
            quality_score=quality_score
        )
        
        logger.info(
            f"✅ Combinaison testée: {len(tools_combination)} outils - "
            f"Quality Score: {quality_score:.3f} - "
            f"F1: {f1:.3f} - "
            f"Temps: {avg_response_time:.2f}s"
        )
        
        return result
    
    def _evaluate_analysis_result(
        self, 
        analysis: AnalysisResponse, 
        test_case: BenchmarkTestCase
    ) -> Dict[str, float]:
        """Évaluation d'un résultat d'analyse."""
        
        # Comparaison niveau de suspicion
        expected_level = test_case.expected_suspicion.value
        actual_level = analysis.suspicion_level.value
        
        suspicion_levels = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        expected_idx = suspicion_levels.index(expected_level) if expected_level in suspicion_levels else 0
        actual_idx = suspicion_levels.index(actual_level) if actual_level in suspicion_levels else 0
        
        # Accuracy basée sur proximité des niveaux de suspicion
        level_diff = abs(expected_idx - actual_idx)
        accuracy = max(0.0, 1.0 - (level_diff * 0.25))
        
        # Precision et Recall
        true_positive = (
            expected_idx >= 2 and actual_idx >= 2  # HIGH/CRITICAL détecté correctement
        )
        false_positive = (
            expected_idx < 2 and actual_idx >= 2  # Fausse alarme
        )
        false_negative = (
            expected_idx >= 2 and actual_idx < 2  # Raté une situation critique
        )
        
        precision = 1.0 if true_positive and not false_positive else (0.0 if false_positive else 0.5)
        recall = 1.0 if true_positive else (0.0 if false_negative else 0.5)
        
        # Bonus confiance
        confidence_bonus = min(0.2, analysis.confidence * 0.2)
        accuracy += confidence_bonus
        precision += confidence_bonus
        recall += confidence_bonus
        
        # Normalisation
        accuracy = min(1.0, accuracy)
        precision = min(1.0, precision)
        recall = min(1.0, recall)
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "confidence": analysis.confidence
        }
    
    async def run_full_benchmark(self) -> Dict[str, Any]:
        """Exécution complète du benchmark."""
        
        logger.info("🚀 Démarrage du benchmark d'optimisation des outils")
        
        # 1. Chargement des cas de test
        await self.load_test_cases()
        
        # 2. Génération des combinaisons
        combinations = self.generate_tool_combinations()
        
        # 3. Test de toutes les combinaisons
        start_time = time.time()
        
        for i, combination in enumerate(combinations):
            logger.info(f"🧪 Test {i+1}/{len(combinations)}: {combination}")
            
            # Sélection des cas de test pour cette combinaison
            test_subset = self.test_cases[:self.min_test_cases_per_combination]
            
            # Test de la combinaison
            result = await self.test_tool_combination(combination, test_subset)
            self.benchmark_results.append(result)
            
            # Mise à jour des métriques individuelles des outils
            self._update_tool_metrics(combination, result)
        
        total_time = time.time() - start_time
        
        # 4. Analyse des résultats
        analysis = self._analyze_benchmark_results()
        
        logger.info(f"✅ Benchmark terminé en {total_time:.1f}s")
        logger.info(f"📊 {len(self.benchmark_results)} combinaisons testées")
        
        return analysis
    
    def _update_tool_metrics(
        self, 
        tools_combination: List[str], 
        result: ToolCombinationResult
    ) -> None:
        """Mise à jour des métriques individuelles des outils."""
        
        for tool in tools_combination:
            if tool not in self.tool_metrics:
                self.tool_metrics[tool] = ToolPerformanceMetrics(tool_name=tool)
            
            metrics = self.tool_metrics[tool]
            metrics.usage_count += 1
            
            # Mise à jour des moyennes pondérées
            alpha = 1.0 / metrics.usage_count
            metrics.precision = (1 - alpha) * metrics.precision + alpha * result.precision
            metrics.recall = (1 - alpha) * metrics.recall + alpha * result.recall
            metrics.f1_score = (1 - alpha) * metrics.f1_score + alpha * result.f1_score
            metrics.avg_execution_time = (1 - alpha) * metrics.avg_execution_time + alpha * result.avg_response_time
            
            # Contribution score (qualité pondérée par usage)
            metrics.contribution_score = metrics.f1_score * np.log(metrics.usage_count + 1)
            
            # Rapport coût/bénéfice
            tool_cost = self.available_tools[tool]["cost"]
            metrics.cost_benefit_ratio = metrics.f1_score / tool_cost if tool_cost > 0 else 0.0
    
    def _analyze_benchmark_results(self) -> Dict[str, Any]:
        """Analyse des résultats du benchmark."""
        
        if not self.benchmark_results:
            return {"error": "Aucun résultat de benchmark"}
        
        # Tri par score qualité
        sorted_results = sorted(
            self.benchmark_results, 
            key=lambda r: r.quality_score, 
            reverse=True
        )
        
        best_combination = sorted_results[0]
        worst_combination = sorted_results[-1]
        
        # Statistiques globales
        quality_scores = [r.quality_score for r in self.benchmark_results]
        response_times = [r.avg_response_time for r in self.benchmark_results]
        costs = [r.total_cost for r in self.benchmark_results]
        
        # Top outils individuels
        top_tools = sorted(
            self.tool_metrics.values(),
            key=lambda m: m.cost_benefit_ratio,
            reverse=True
        )[:5]
        
        # Recommandations
        recommendations = self._generate_recommendations(sorted_results)
        
        analysis = {
            "benchmark_summary": {
                "total_combinations_tested": len(self.benchmark_results),
                "total_test_cases": len(self.test_cases),
                "avg_quality_score": np.mean(quality_scores),
                "std_quality_score": np.std(quality_scores),
                "avg_response_time": np.mean(response_times),
                "avg_cost": np.mean(costs)
            },
            "best_combination": {
                "tools": best_combination.tools_combination,
                "quality_score": best_combination.quality_score,
                "precision": best_combination.precision,
                "recall": best_combination.recall,
                "f1_score": best_combination.f1_score,
                "response_time": best_combination.avg_response_time,
                "cost": best_combination.total_cost
            },
            "worst_combination": {
                "tools": worst_combination.tools_combination,
                "quality_score": worst_combination.quality_score,
                "cost": worst_combination.total_cost
            },
            "top_individual_tools": [
                {
                    "tool": tool.tool_name,
                    "cost_benefit_ratio": tool.cost_benefit_ratio,
                    "f1_score": tool.f1_score,
                    "usage_count": tool.usage_count,
                    "avg_execution_time": tool.avg_execution_time
                }
                for tool in top_tools
            ],
            "recommendations": recommendations,
            "full_results": [asdict(result) for result in sorted_results[:10]]  # Top 10
        }
        
        return analysis
    
    def _generate_recommendations(
        self, 
        sorted_results: List[ToolCombinationResult]
    ) -> Dict[str, Any]:
        """Génération de recommandations d'optimisation."""
        
        top_3 = sorted_results[:3]
        
        # Analyse des patterns dans le top 3
        all_top_tools = set()
        for result in top_3:
            all_top_tools.update(result.tools_combination)
        
        # Outils les plus fréquents dans le top
        tool_frequency = defaultdict(int)
        for result in top_3:
            for tool in result.tools_combination:
                tool_frequency[tool] += 1
        
        most_valuable_tools = sorted(
            tool_frequency.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:4]
        
        # Recommandations par mode
        fast_mode_recommendation = self._recommend_for_mode("fast", sorted_results)
        balanced_mode_recommendation = self._recommend_for_mode("balanced", sorted_results)
        thorough_mode_recommendation = self._recommend_for_mode("thorough", sorted_results)
        
        return {
            "optimal_tool_selection": [tool for tool, _ in most_valuable_tools],
            "tools_to_remove": self._identify_underperforming_tools(),
            "mode_specific": {
                "fast": fast_mode_recommendation,
                "balanced": balanced_mode_recommendation,
                "thorough": thorough_mode_recommendation
            },
            "cost_optimization": self._recommend_cost_optimization(sorted_results),
            "summary": f"Utilisez {len(most_valuable_tools)} outils principaux pour {sorted_results[0].quality_score:.1%} de qualité optimale"
        }
    
    def _recommend_for_mode(
        self, 
        mode: str, 
        sorted_results: List[ToolCombinationResult]
    ) -> List[str]:
        """Recommandation d'outils pour un mode spécifique."""
        
        if mode == "fast":
            # Mode rapide: privilégier rapidité et coût faible
            candidates = [r for r in sorted_results if r.avg_response_time <= 2.0 and r.total_cost <= 6.0]
            return candidates[0].tools_combination if candidates else sorted_results[0].tools_combination[:2]
        
        elif mode == "balanced":
            # Mode équilibré: meilleur rapport qualité/coût
            candidates = sorted(sorted_results, key=lambda r: r.quality_score / (r.total_cost + 1), reverse=True)
            return candidates[0].tools_combination
        
        elif mode == "thorough":
            # Mode approfondi: meilleure qualité possible
            return sorted_results[0].tools_combination
        
        return []
    
    def _identify_underperforming_tools(self) -> List[str]:
        """Identification des outils sous-performants."""
        
        underperforming = []
        
        for tool, metrics in self.tool_metrics.items():
            # Critères de sous-performance
            if (
                metrics.cost_benefit_ratio < 0.2 or  # Très mauvais rapport coût/bénéfice
                metrics.f1_score < 0.3 or  # F1 score très faible
                (metrics.usage_count >= 3 and metrics.contribution_score < 1.0)  # Contribution faible
            ):
                underperforming.append(tool)
        
        return underperforming
    
    def _recommend_cost_optimization(
        self, 
        sorted_results: List[ToolCombinationResult]
    ) -> Dict[str, Any]:
        """Recommandations d'optimisation des coûts."""
        
        # Recherche du point optimal coût/qualité
        pareto_frontier = []
        
        for result in sorted_results:
            is_pareto_optimal = True
            for other in sorted_results:
                if (other.quality_score >= result.quality_score and 
                    other.total_cost <= result.total_cost and
                    (other.quality_score > result.quality_score or other.total_cost < result.total_cost)):
                    is_pareto_optimal = False
                    break
            
            if is_pareto_optimal:
                pareto_frontier.append(result)
        
        # Recommandation basée sur frontière de Pareto
        pareto_frontier.sort(key=lambda r: r.quality_score / (r.total_cost + 1), reverse=True)
        
        return {
            "pareto_optimal_combinations": len(pareto_frontier),
            "recommended_combination": pareto_frontier[0].tools_combination if pareto_frontier else [],
            "cost_savings_potential": f"{((sorted_results[0].total_cost - pareto_frontier[0].total_cost) / sorted_results[0].total_cost * 100):.1f}%" if pareto_frontier else "0%",
            "quality_vs_best": f"{(pareto_frontier[0].quality_score / sorted_results[0].quality_score * 100):.1f}%" if pareto_frontier else "100%"
        }
    
    def save_benchmark_results(self, output_path: Path) -> None:
        """Sauvegarde des résultats du benchmark."""
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Analyse complète
        analysis = self._analyze_benchmark_results()
        
        # Sauvegarde JSON
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        logger.info(f"✅ Résultats sauvegardés: {output_path}")
    
    async def auto_optimize_tools(self) -> List[str]:
        """Optimisation automatique et retour des outils optimaux."""
        
        analysis = await self.run_full_benchmark()
        
        if "best_combination" in analysis:
            optimal_tools = analysis["best_combination"]["tools"]
            
            logger.info("🎯 Outils optimaux sélectionnés automatiquement:")
            for tool in optimal_tools:
                metrics = self.tool_metrics.get(tool, {})
                logger.info(f"  • {tool}: F1={getattr(metrics, 'f1_score', 0):.3f}, Coût/Bénéfice={getattr(metrics, 'cost_benefit_ratio', 0):.3f}")
            
            return optimal_tools
        
        return list(self.available_tools.keys())  # Fallback


# Fonctions utilitaires pour intégration

async def optimize_tools_for_system() -> List[str]:
    """Fonction principale d'optimisation des outils."""
    
    benchmark = ToolOptimizationBenchmark()
    optimal_tools = await benchmark.auto_optimize_tools()
    
    # Sauvegarde des résultats
    results_path = Path("data/benchmark/tool_optimization_results.json")
    benchmark.save_benchmark_results(results_path)
    
    return optimal_tools


async def quick_tool_evaluation(tools_to_test: List[str]) -> Dict[str, float]:
    """Évaluation rapide d'outils spécifiques."""
    
    benchmark = ToolOptimizationBenchmark()
    await benchmark.load_test_cases()
    
    # Test des outils spécifiés
    result = await benchmark.test_tool_combination(
        tools_to_test, 
        benchmark.test_cases[:5]  # Test rapide sur 5 cas
    )
    
    return {
        "quality_score": result.quality_score,
        "f1_score": result.f1_score,
        "response_time": result.avg_response_time,
        "total_cost": result.total_cost
    }


if __name__ == "__main__":
    import asyncio
    
    async def main():
        # Exemple d'utilisation
        optimal_tools = await optimize_tools_for_system()
        print(f"🎯 Outils optimaux: {optimal_tools}")
    
    asyncio.run(main())