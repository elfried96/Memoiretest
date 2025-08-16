"""
🎯 Démonstration du Système d'Optimisation des Outils
====================================================

Ce script démontre l'utilisation complète du système d'optimisation des outils :
1. Benchmark automatique des combinaisons d'outils
2. Sélection des outils optimaux
3. Orchestration adaptative avec apprentissage
4. Analyse des performances

Usage:
    python examples/tool_optimization_demo.py --mode benchmark
    python examples/tool_optimization_demo.py --mode adaptive
    python examples/tool_optimization_demo.py --mode compare
"""

import asyncio
import argparse
import time
import json
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.live import Live

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.testing.tool_optimization_benchmark import (
    ToolOptimizationBenchmark, 
    optimize_tools_for_system,
    quick_tool_evaluation
)
from src.core.orchestrator.adaptive_orchestrator import (
    AdaptiveVLMOrchestrator,
    create_adaptive_orchestrator
)
from src.core.orchestrator.vlm_orchestrator import (
    ModernVLMOrchestrator,
    OrchestrationConfig,
    OrchestrationMode
)
from src.core.types import AnalysisRequest

console = Console()


class ToolOptimizationDemo:
    """Classe de démonstration pour l'optimisation des outils."""
    
    def __init__(self):
        self.benchmark_results = {}
        self.adaptive_results = {}
        self.demo_data_path = Path("data/demo")
        self.demo_data_path.mkdir(parents=True, exist_ok=True)
    
    async def run_benchmark_demo(self) -> Dict[str, Any]:
        """Démonstration du benchmark d'optimisation."""
        
        console.print(Panel.fit(
            "[bold blue]🧪 Démonstration du Benchmark d'Optimisation des Outils[/bold blue]\n"
            "[dim]Test automatique de toutes les combinaisons d'outils pour identifier les plus performantes[/dim]",
            border_style="blue"
        ))
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            # Initialisation du benchmark
            task = progress.add_task("[cyan]Initialisation du benchmark...", total=None)
            benchmark = ToolOptimizationBenchmark(vlm_model_name="kimi-vl-a3b-thinking")
            
            # Chargement des cas de test
            progress.update(task, description="[cyan]Chargement des cas de test...")
            await benchmark.load_test_cases()
            
            progress.update(task, description="[green]✅ Cas de test chargés")
            await asyncio.sleep(1)
            
            # Génération des combinaisons
            progress.update(task, description="[cyan]Génération des combinaisons d'outils...")
            combinations = benchmark.generate_tool_combinations()
            
            progress.update(task, description="[green]✅ Combinaisons générées")
            console.print(f"📊 {len(combinations)} combinaisons à tester")
            console.print(f"📝 {len(benchmark.test_cases)} cas de test")
            
            # Exécution du benchmark complet
            progress.update(task, description="[yellow]Exécution du benchmark complet...")
        
        # Benchmark avec affichage en temps réel
        start_time = time.time()
        results = await self._run_benchmark_with_live_updates(benchmark)
        benchmark_time = time.time() - start_time
        
        # Affichage des résultats
        self._display_benchmark_results(results, benchmark_time)
        
        # Sauvegarde des résultats
        results_file = self.demo_data_path / "benchmark_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        console.print(f"💾 Résultats sauvegardés: {results_file}")
        
        self.benchmark_results = results
        return results
    
    async def _run_benchmark_with_live_updates(
        self, 
        benchmark: ToolOptimizationBenchmark
    ) -> Dict[str, Any]:
        """Exécution du benchmark avec mises à jour en temps réel."""
        
        combinations = benchmark.generate_tool_combinations()
        
        # Table pour affichage en temps réel
        table = Table(title="🧪 Progression du Benchmark")
        table.add_column("Combinaison", style="cyan", no_wrap=True)
        table.add_column("Outils", style="yellow")
        table.add_column("Quality Score", style="green")
        table.add_column("F1 Score", style="blue")
        table.add_column("Temps (s)", style="magenta")
        table.add_column("Statut", style="bold")
        
        with Live(table, refresh_per_second=2, console=console) as live:
            
            for i, combination in enumerate(combinations):
                # Mise à jour de la table
                table.add_row(
                    f"{i+1}/{len(combinations)}",
                    f"{len(combination)} outils",
                    "En cours...",
                    "En cours...",
                    "En cours...",
                    "🔄 Test..."
                )
                
                # Test de la combinaison
                start = time.time()
                try:
                    result = await benchmark.test_tool_combination(
                        combination, 
                        benchmark.test_cases[:10]  # Test rapide pour démo
                    )
                    
                    test_time = time.time() - start
                    
                    # Mise à jour de la dernière ligne
                    table._rows[-1] = (
                        f"{i+1}/{len(combinations)}",
                        ", ".join(combination[:2]) + ("..." if len(combination) > 2 else ""),
                        f"{result.quality_score:.3f}",
                        f"{result.f1_score:.3f}",
                        f"{test_time:.2f}",
                        "✅ OK"
                    )
                    
                    benchmark.benchmark_results.append(result)
                    
                except Exception as e:
                    # Mise à jour en cas d'erreur
                    table._rows[-1] = (
                        f"{i+1}/{len(combinations)}",
                        ", ".join(combination[:2]) + ("..." if len(combination) > 2 else ""),
                        "Erreur",
                        "Erreur",
                        f"{time.time() - start:.2f}",
                        f"❌ {str(e)[:20]}..."
                    )
                
                # Petite pause pour la démo
                await asyncio.sleep(0.5)
        
        # Analyse finale
        return benchmark._analyze_benchmark_results()
    
    def _display_benchmark_results(self, results: Dict[str, Any], benchmark_time: float) -> None:
        """Affichage des résultats du benchmark."""
        
        console.print("\n")
        console.print(Panel.fit(
            "[bold green]📊 Résultats du Benchmark d'Optimisation[/bold green]",
            border_style="green"
        ))
        
        # Statistiques générales
        summary = results.get("benchmark_summary", {})
        console.print(f"⏱️  Temps total: {benchmark_time:.1f}s")
        console.print(f"🧪 Combinaisons testées: {summary.get('total_combinations_tested', 0)}")
        console.print(f"📝 Cas de test: {summary.get('total_test_cases', 0)}")
        console.print(f"📊 Score qualité moyen: {summary.get('avg_quality_score', 0):.3f}")
        
        # Meilleure combinaison
        if "best_combination" in results:
            best = results["best_combination"]
            
            console.print("\n🏆 **Meilleure Combinaison d'Outils:**")
            
            tools_table = Table(title="Outils Optimaux")
            tools_table.add_column("Outil", style="cyan")
            tools_table.add_column("Métrique", style="yellow")
            tools_table.add_column("Valeur", style="green")
            
            for tool in best["tools"]:
                tools_table.add_row(tool, "Inclus", "✅")
            
            tools_table.add_row("Quality Score", "Global", f"{best['quality_score']:.3f}")
            tools_table.add_row("Precision", "Global", f"{best['precision']:.3f}")
            tools_table.add_row("Recall", "Global", f"{best['recall']:.3f}")
            tools_table.add_row("F1 Score", "Global", f"{best['f1_score']:.3f}")
            tools_table.add_row("Temps Réponse", "Moyenne", f"{best['response_time']:.2f}s")
            tools_table.add_row("Coût Computationnel", "Total", f"{best['cost']:.1f}")
            
            console.print(tools_table)
        
        # Top outils individuels
        if "top_individual_tools" in results:
            console.print("\n🎯 **Top Outils Individuels (Coût/Bénéfice):**")
            
            individual_table = Table()
            individual_table.add_column("Rang", style="bold")
            individual_table.add_column("Outil", style="cyan")
            individual_table.add_column("Ratio C/B", style="green")
            individual_table.add_column("F1 Score", style="blue")
            individual_table.add_column("Utilisations", style="yellow")
            
            for i, tool_data in enumerate(results["top_individual_tools"][:5], 1):
                individual_table.add_row(
                    f"{i}",
                    tool_data["tool"],
                    f"{tool_data['cost_benefit_ratio']:.3f}",
                    f"{tool_data['f1_score']:.3f}",
                    f"{tool_data['usage_count']}"
                )
            
            console.print(individual_table)
        
        # Recommandations
        if "recommendations" in results:
            recommendations = results["recommendations"]
            
            console.print("\n💡 **Recommandations d'Optimisation:**")
            
            console.print(f"🎯 **Sélection optimale:** {', '.join(recommendations.get('optimal_tool_selection', []))}")
            
            if recommendations.get("tools_to_remove"):
                console.print(f"🗑️  **Outils à supprimer:** {', '.join(recommendations['tools_to_remove'])}")
            
            # Recommandations par mode
            mode_recs = recommendations.get("mode_specific", {})
            for mode, tools in mode_recs.items():
                if tools:
                    console.print(f"⚙️  **Mode {mode.upper()}:** {', '.join(tools)}")
            
            console.print(f"💰 **Économies potentielles:** {recommendations.get('cost_optimization', {}).get('cost_savings_potential', '0%')}")
    
    async def run_adaptive_demo(self) -> Dict[str, Any]:
        """Démonstration de l'orchestration adaptative."""
        
        console.print(Panel.fit(
            "[bold green]🧠 Démonstration de l'Orchestration Adaptative[/bold green]\n"
            "[dim]Apprentissage automatique et optimisation continue des outils[/dim]",
            border_style="green"
        ))
        
        # Création de l'orchestrateur adaptatif
        console.print("🚀 Initialisation de l'orchestrateur adaptatif...")
        
        config = OrchestrationConfig(
            mode=OrchestrationMode.BALANCED,
            enable_advanced_tools=True,
            max_concurrent_tools=4,
            confidence_threshold=0.7
        )
        
        adaptive_orchestrator = create_adaptive_orchestrator(
            vlm_model_name="kimi-vl-a3b-thinking",
            config=config,
            enable_learning=True
        )
        
        # Simulation de plusieurs analyses pour démontrer l'apprentissage
        console.print("📚 Simulation d'analyses pour démonstrer l'apprentissage...")
        
        # Scénarios de test variés
        test_scenarios = [
            {
                "context": {"location": "entrance", "time": "morning"},
                "detections": [{"class_name": "person"}],
                "expected": "Situation normale d'entrée matinale"
            },
            {
                "context": {"location": "electronics", "time": "evening"},
                "detections": [{"class_name": "person"}, {"class_name": "backpack"}],
                "expected": "Zone sensible avec sac à dos"
            },
            {
                "context": {"location": "checkout", "time": "afternoon"},
                "detections": [],
                "expected": "Zone de caisse sans activité"
            },
            {
                "context": {"location": "entrance", "time": "night"},
                "detections": [{"class_name": "person"}],
                "expected": "Activité nocturne suspecte"
            }
        ]
        
        learning_results = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            task = progress.add_task("[cyan]Exécution des scénarios d'apprentissage...", total=len(test_scenarios))
            
            for i, scenario in enumerate(test_scenarios):
                progress.update(task, description=f"[cyan]Scénario {i+1}: {scenario['expected']}")
                
                try:
                    # Simulation d'analyse (données factices pour démo)
                    start_time = time.time()
                    
                    # Note: Dans un vrai environnement, utiliser de vraies images
                    analysis = await adaptive_orchestrator.analyze_surveillance_frame(
                        frame_data="demo_frame_data",  # Image factice
                        detections=scenario["detections"],
                        context=scenario["context"]
                    )
                    
                    processing_time = time.time() - start_time
                    
                    learning_results.append({
                        "scenario": scenario["expected"],
                        "tools_used": analysis.tools_used,
                        "confidence": analysis.confidence,
                        "suspicion_level": analysis.suspicion_level.value,
                        "processing_time": processing_time
                    })
                    
                    progress.advance(task)
                    await asyncio.sleep(0.5)  # Pause pour la démo
                    
                except Exception as e:
                    logger.error(f"Erreur scénario {i+1}: {e}")
                    progress.advance(task)
        
        # Affichage des résultats d'apprentissage
        self._display_adaptive_results(learning_results, adaptive_orchestrator)
        
        # Statut du système adaptatif
        adaptive_status = adaptive_orchestrator.get_adaptive_status()
        
        # Sauvegarde des résultats
        results = {
            "learning_results": learning_results,
            "adaptive_status": adaptive_status,
            "timestamp": time.time()
        }
        
        results_file = self.demo_data_path / "adaptive_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        console.print(f"💾 Résultats adaptatifs sauvegardés: {results_file}")
        
        self.adaptive_results = results
        return results
    
    def _display_adaptive_results(
        self, 
        learning_results: List[Dict], 
        orchestrator: AdaptiveVLMOrchestrator
    ) -> None:
        """Affichage des résultats de l'apprentissage adaptatif."""
        
        console.print("\n")
        console.print(Panel.fit(
            "[bold green]🧠 Résultats de l'Apprentissage Adaptatif[/bold green]",
            border_style="green"
        ))
        
        # Table des résultats d'apprentissage
        learning_table = Table(title="📚 Résultats d'Apprentissage par Scénario")
        learning_table.add_column("Scénario", style="cyan")
        learning_table.add_column("Outils Utilisés", style="yellow")
        learning_table.add_column("Confiance", style="green")
        learning_table.add_column("Suspicion", style="red")
        learning_table.add_column("Temps (s)", style="blue")
        
        for result in learning_results:
            tools_display = ", ".join(result["tools_used"][:2])
            if len(result["tools_used"]) > 2:
                tools_display += f" (+{len(result['tools_used'])-2})"
            
            learning_table.add_row(
                result["scenario"][:30] + "..." if len(result["scenario"]) > 30 else result["scenario"],
                tools_display,
                f"{result['confidence']:.2f}",
                result["suspicion_level"],
                f"{result['processing_time']:.2f}"
            )
        
        console.print(learning_table)
        
        # Statut du système adaptatif
        status = orchestrator.get_adaptive_status()
        
        console.print("\n🎯 **Statut du Système Adaptatif:**")
        
        status_table = Table()
        status_table.add_column("Métrique", style="cyan")
        status_table.add_column("Valeur", style="green")
        
        status_table.add_row("Apprentissage Activé", "✅" if status["adaptive_learning_enabled"] else "❌")
        status_table.add_row("Outils Optimaux Actuels", f"{len(status['current_optimal_tools'])} outils")
        status_table.add_row("Patterns Contextuels Appris", f"{status['learning_stats']['context_patterns_learned']}")
        status_table.add_row("Performance Moyenne", f"{status['learning_stats']['avg_pattern_performance']:.3f}")
        status_table.add_row("Exécutions en Historique", f"{status['learning_stats']['executions_in_history']}")
        
        console.print(status_table)
        
        # Top outils performants
        if status["top_performing_tools"]:
            console.print("\n🏆 **Top Outils Performants:**")
            for i, tool_info in enumerate(status["top_performing_tools"][:3], 1):
                console.print(f"  {i}. {tool_info['tool']}: {tool_info['success_rate']:.1%} de succès")
    
    async def run_comparison_demo(self) -> Dict[str, Any]:
        """Démonstration comparative entre orchestrateur standard et adaptatif."""
        
        console.print(Panel.fit(
            "[bold yellow]⚖️ Comparaison Standard vs Adaptatif[/bold yellow]\n"
            "[dim]Comparaison des performances entre orchestration fixe et adaptative[/dim]",
            border_style="yellow"
        ))
        
        # Configuration commune
        config = OrchestrationConfig(
            mode=OrchestrationMode.BALANCED,
            enable_advanced_tools=True,
            max_concurrent_tools=4,
            confidence_threshold=0.7
        )
        
        # Création des orchestrateurs
        standard_orchestrator = ModernVLMOrchestrator("kimi-vl-a3b-thinking", config)
        adaptive_orchestrator = create_adaptive_orchestrator("kimi-vl-a3b-thinking", config)
        
        # Scénarios de test
        test_scenarios = [
            {"context": {"location": "entrance"}, "detections": [{"class_name": "person"}]},
            {"context": {"location": "electronics"}, "detections": [{"class_name": "person"}, {"class_name": "handbag"}]},
            {"context": {"location": "checkout"}, "detections": []},
        ]
        
        comparison_results = {
            "standard": [],
            "adaptive": [],
            "summary": {}
        }
        
        console.print("🔄 Test des deux orchestrateurs...")
        
        for i, scenario in enumerate(test_scenarios):
            console.print(f"\n🧪 Test {i+1}: {scenario['context'].get('location', 'unknown')}")
            
            # Test orchestrateur standard
            start_time = time.time()
            try:
                standard_result = await standard_orchestrator.analyze_surveillance_frame(
                    frame_data="test_frame",
                    detections=scenario["detections"],
                    context=scenario["context"]
                )
                standard_time = time.time() - start_time
                
                comparison_results["standard"].append({
                    "scenario": i+1,
                    "tools_used": standard_result.tools_used,
                    "confidence": standard_result.confidence,
                    "processing_time": standard_time
                })
                
                console.print(f"  📊 Standard: {len(standard_result.tools_used)} outils, {standard_time:.2f}s")
                
            except Exception as e:
                console.print(f"  ❌ Standard: Erreur - {e}")
            
            # Test orchestrateur adaptatif
            start_time = time.time()
            try:
                adaptive_result = await adaptive_orchestrator.analyze_surveillance_frame(
                    frame_data="test_frame",
                    detections=scenario["detections"],
                    context=scenario["context"]
                )
                adaptive_time = time.time() - start_time
                
                comparison_results["adaptive"].append({
                    "scenario": i+1,
                    "tools_used": adaptive_result.tools_used,
                    "confidence": adaptive_result.confidence,
                    "processing_time": adaptive_time
                })
                
                console.print(f"  🧠 Adaptatif: {len(adaptive_result.tools_used)} outils, {adaptive_time:.2f}s")
                
            except Exception as e:
                console.print(f"  ❌ Adaptatif: Erreur - {e}")
            
            await asyncio.sleep(0.5)
        
        # Analyse comparative
        self._display_comparison_results(comparison_results)
        
        # Sauvegarde
        results_file = self.demo_data_path / "comparison_results.json"
        with open(results_file, 'w') as f:
            json.dump(comparison_results, f, indent=2, default=str)
        
        return comparison_results
    
    def _display_comparison_results(self, results: Dict[str, Any]) -> None:
        """Affichage des résultats comparatifs."""
        
        console.print("\n")
        console.print(Panel.fit(
            "[bold yellow]⚖️ Résultats de la Comparaison[/bold yellow]",
            border_style="yellow"
        ))
        
        # Table comparative
        comparison_table = Table(title="📊 Comparaison Standard vs Adaptatif")
        comparison_table.add_column("Scénario", style="cyan")
        comparison_table.add_column("Standard (Outils)", style="blue")
        comparison_table.add_column("Standard (Temps)", style="blue")
        comparison_table.add_column("Adaptatif (Outils)", style="green")
        comparison_table.add_column("Adaptatif (Temps)", style="green")
        comparison_table.add_column("Amélioration", style="bold")
        
        for i in range(min(len(results["standard"]), len(results["adaptive"]))):
            std_result = results["standard"][i]
            adp_result = results["adaptive"][i]
            
            time_improvement = ((std_result["processing_time"] - adp_result["processing_time"]) / std_result["processing_time"]) * 100
            improvement_indicator = "🚀" if time_improvement > 0 else "🐌" if time_improvement < -10 else "➖"
            
            comparison_table.add_row(
                f"Test {i+1}",
                f"{len(std_result['tools_used'])}",
                f"{std_result['processing_time']:.2f}s",
                f"{len(adp_result['tools_used'])}",
                f"{adp_result['processing_time']:.2f}s",
                f"{improvement_indicator} {time_improvement:+.1f}%"
            )
        
        console.print(comparison_table)
        
        # Statistiques globales
        if results["standard"] and results["adaptive"]:
            std_avg_time = np.mean([r["processing_time"] for r in results["standard"]])
            adp_avg_time = np.mean([r["processing_time"] for r in results["adaptive"]])
            
            std_avg_tools = np.mean([len(r["tools_used"]) for r in results["standard"]])
            adp_avg_tools = np.mean([len(r["tools_used"]) for r in results["adaptive"]])
            
            time_savings = ((std_avg_time - adp_avg_time) / std_avg_time) * 100
            
            console.print(f"\n📈 **Statistiques Globales:**")
            console.print(f"⏱️  Temps moyen - Standard: {std_avg_time:.2f}s, Adaptatif: {adp_avg_time:.2f}s")
            console.print(f"🔧 Outils moyens - Standard: {std_avg_tools:.1f}, Adaptatif: {adp_avg_tools:.1f}")
            console.print(f"💡 Gain de temps: {time_savings:+.1f}%")
    
    async def run_full_demo(self) -> Dict[str, Any]:
        """Démonstration complète du système d'optimisation."""
        
        console.print(Panel.fit(
            "[bold magenta]🎯 Démonstration Complète du Système d'Optimisation des Outils[/bold magenta]\n"
            "[dim]Benchmark → Optimisation → Apprentissage Adaptatif → Comparaison[/dim]",
            border_style="magenta"
        ))
        
        full_results = {}
        
        # 1. Benchmark d'optimisation
        console.print("\n" + "="*60)
        console.print("🧪 ÉTAPE 1: Benchmark d'Optimisation")
        console.print("="*60)
        
        benchmark_results = await self.run_benchmark_demo()
        full_results["benchmark"] = benchmark_results
        
        # 2. Apprentissage adaptatif
        console.print("\n" + "="*60)
        console.print("🧠 ÉTAPE 2: Apprentissage Adaptatif")
        console.print("="*60)
        
        adaptive_results = await self.run_adaptive_demo()
        full_results["adaptive"] = adaptive_results
        
        # 3. Comparaison
        console.print("\n" + "="*60)
        console.print("⚖️ ÉTAPE 3: Comparaison Standard vs Adaptatif")
        console.print("="*60)
        
        comparison_results = await self.run_comparison_demo()
        full_results["comparison"] = comparison_results
        
        # Résumé final
        console.print("\n" + "="*60)
        console.print("📋 RÉSUMÉ FINAL")
        console.print("="*60)
        
        self._display_final_summary(full_results)
        
        # Sauvegarde complète
        full_results_file = self.demo_data_path / "full_demo_results.json"
        with open(full_results_file, 'w') as f:
            json.dump(full_results, f, indent=2, default=str)
        
        console.print(f"\n💾 Résultats complets sauvegardés: {full_results_file}")
        
        return full_results
    
    def _display_final_summary(self, results: Dict[str, Any]) -> None:
        """Affichage du résumé final."""
        
        summary_table = Table(title="📋 Résumé Final de l'Optimisation")
        summary_table.add_column("Aspect", style="cyan")
        summary_table.add_column("Résultat", style="green")
        summary_table.add_column("Impact", style="yellow")
        
        # Benchmark
        if "benchmark" in results and "best_combination" in results["benchmark"]:
            best = results["benchmark"]["best_combination"]
            summary_table.add_row(
                "Outils Optimaux Identifiés",
                f"{len(best['tools'])} outils sélectionnés",
                f"Quality Score: {best['quality_score']:.3f}"
            )
        
        # Adaptatif
        if "adaptive" in results and "adaptive_status" in results["adaptive"]:
            status = results["adaptive"]["adaptive_status"]
            patterns_learned = status["learning_stats"]["context_patterns_learned"]
            summary_table.add_row(
                "Apprentissage Adaptatif",
                f"{patterns_learned} patterns contextuels appris",
                "Optimisation continue activée"
            )
        
        # Comparaison
        if "comparison" in results:
            comp = results["comparison"]
            if comp["standard"] and comp["adaptive"]:
                std_avg = np.mean([r["processing_time"] for r in comp["standard"]])
                adp_avg = np.mean([r["processing_time"] for r in comp["adaptive"]])
                improvement = ((std_avg - adp_avg) / std_avg) * 100
                
                summary_table.add_row(
                    "Performance Comparative",
                    f"Temps réduit de {improvement:.1f}%",
                    "🚀 Système adaptatif plus rapide" if improvement > 0 else "➖ Performance similaire"
                )
        
        console.print(summary_table)
        
        # Recommandations finales
        console.print("\n💡 **Recommandations Finales:**")
        
        if "benchmark" in results and "recommendations" in results["benchmark"]:
            recs = results["benchmark"]["recommendations"]
            console.print(f"🎯 Utilisez ces outils optimaux: {', '.join(recs.get('optimal_tool_selection', [])[:4])}")
            
            if recs.get("tools_to_remove"):
                console.print(f"🗑️  Supprimez ces outils sous-performants: {', '.join(recs['tools_to_remove'])}")
        
        console.print("🧠 Activez l'apprentissage adaptatif pour une optimisation continue")
        console.print("📊 Relancez le benchmark périodiquement pour maintenir les performances")


async def main():
    """Point d'entrée principal."""
    
    parser = argparse.ArgumentParser(
        description="Démonstration du système d'optimisation des outils VLM"
    )
    
    parser.add_argument(
        "--mode", 
        choices=["benchmark", "adaptive", "compare", "full"],
        default="full",
        help="Mode de démonstration (défaut: full)"
    )
    
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Sauvegarder les résultats détaillés"
    )
    
    args = parser.parse_args()
    
    # Initialisation de la démo
    demo = ToolOptimizationDemo()
    
    console.print(f"🚀 Démarrage de la démonstration en mode: {args.mode}")
    
    start_time = time.time()
    
    try:
        if args.mode == "benchmark":
            results = await demo.run_benchmark_demo()
        elif args.mode == "adaptive":
            results = await demo.run_adaptive_demo()
        elif args.mode == "compare":
            results = await demo.run_comparison_demo()
        elif args.mode == "full":
            results = await demo.run_full_demo()
        else:
            console.print(f"❌ Mode inconnu: {args.mode}")
            return
        
        total_time = time.time() - start_time
        
        console.print(f"\n✅ Démonstration terminée en {total_time:.1f}s")
        
        if args.save_results:
            results_file = Path(f"demo_results_{args.mode}_{int(time.time())}.json")
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            console.print(f"💾 Résultats détaillés sauvegardés: {results_file}")
    
    except KeyboardInterrupt:
        console.print("\n🛑 Démonstration interrompue par l'utilisateur")
    except Exception as e:
        console.print(f"\n❌ Erreur durant la démonstration: {e}")
        logger.exception("Erreur démonstration")


if __name__ == "__main__":
    asyncio.run(main())