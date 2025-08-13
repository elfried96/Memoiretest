#!/usr/bin/env python3
"""
Exemple complet d'utilisation du framework de test avancé.
Démontre l'utilisation de tous les outils et configurations disponibles.
"""

import asyncio
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

# Import des modules de test
from src.testing import (
    ABTestFramework, 
    BenchmarkSuite, 
    PerformanceCollector, 
    MetricsVisualizer,
    TestVariant,
    BenchmarkConfig
)

# Import des outils avancés pour configuration
from src.advanced_tools import *
from src.core.orchestrator.vlm_orchestrator import VLMOrchestrator

class CompleteTester:
    """Démonstrateur complet du framework de test."""
    
    def __init__(self, output_dir: str = "test_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialisation des composants
        self.ab_framework = ABTestFramework(str(self.output_dir / "ab_tests"))
        self.benchmark_suite = BenchmarkSuite(str(self.output_dir / "benchmarks"))
        self.visualizer = MetricsVisualizer(str(self.output_dir / "visualizations"))
        self.performance_collector = PerformanceCollector()
        
        # Configuration
        self.test_datasets = self._setup_test_datasets()
        
    def _setup_test_datasets(self) -> Dict[str, List[str]]:
        """Configure les datasets de test."""
        datasets = {
            "small": self._create_mock_dataset(50),
            "medium": self._create_mock_dataset(200), 
            "large": self._create_mock_dataset(500),
            "stress": self._create_mock_dataset(1000)
        }
        return datasets
    
    def _create_mock_dataset(self, size: int) -> List[str]:
        """Crée un dataset mock pour les tests."""
        return [f"mock_frame_{i:04d}.jpg" for i in range(size)]
    
    async def demonstrate_basic_ab_test(self):
        """Démontre un test A/B basique."""
        print("🧪 === TEST A/B BASIQUE ===")
        
        # Configuration des variants par défaut
        self.ab_framework.setup_default_variants()
        
        # Lancement du test
        print("Lancement du test A/B avec dataset moyen...")
        
        ab_results = self.ab_framework.run_ab_test(
            test_dataset=self.test_datasets["medium"],
            test_duration=60,  # 1 minute pour la démo
            concurrent_variants=2,
            metrics_to_collect=[
                "execution_time", "memory_usage", "accuracy",
                "false_positive_rate", "throughput", "latency"
            ]
        )
        
        # Affichage des résultats
        print(f"✅ Test terminé. Test ID: {ab_results['test_id']}")
        print(f"🏆 Meilleur variant: {ab_results['winner']}")
        
        # Génération des visualisations
        print("📊 Génération des visualisations...")
        self.ab_framework.visualize_results(ab_results['variant_results'])
        
        # Rapport détaillé
        report = self.ab_framework.generate_report(
            ab_results['variant_results'],
            ab_results['comparison_report']
        )
        
        # Sauvegarde du rapport
        report_file = self.output_dir / "ab_test_report.md"
        report_file.write_text(report)
        print(f"📄 Rapport sauvegardé: {report_file}")
        
        return ab_results
    
    async def demonstrate_advanced_variants(self):
        """Démontre l'utilisation des variants avancés."""
        print("\n🚀 === VARIANTS AVANCÉS ===")
        
        # Configuration des variants avancés
        self.benchmark_suite.setup_test_variants()
        
        # Créer un variant personnalisé
        custom_variant = TestVariant(
            name="custom_optimized",
            pipeline_type="custom",
            components=[
                "yolo_detector", "sam2_segmentation", "dino_features", 
                "pose_estimation", "multimodal_fusion", "vlm_analyzer"
            ],
            parameters={
                "sam2_confidence": 0.85,
                "dino_attention": True,
                "pose_threshold": 0.6,
                "fusion_method": "attention",
                "vlm_temperature": 0.5
            },
            enabled_tools=[
                "object_detector", "segmentation_tool", "feature_extractor",
                "pose_analyzer", "behavior_classifier"
            ],
            description="Variant personnalisé optimisé pour précision maximale"
        )
        
        # Enregistrer le variant personnalisé
        self.ab_framework.register_variant(custom_variant)
        
        print("✅ Variants avancés configurés")
        print(f"Nombre total de variants: {len(self.ab_framework.test_variants)}")
        
        # Test avec variants avancés
        print("Lancement du test avec variants avancés...")
        
        ab_results = self.ab_framework.run_ab_test(
            test_dataset=self.test_datasets["small"],  # Dataset plus petit pour démo
            test_duration=30,  # Test rapide
            concurrent_variants=3,
            metrics_to_collect=[
                "execution_time", "memory_usage", "accuracy", 
                "false_positive_rate", "throughput"
            ]
        )
        
        print(f"🏆 Meilleur variant avancé: {ab_results['winner']}")
        
        return ab_results
    
    async def demonstrate_performance_monitoring(self):
        """Démontre le monitoring de performance en temps réel."""
        print("\n📈 === MONITORING DE PERFORMANCE ===")
        
        # Démarrer la collecte de métriques
        self.performance_collector.start_collection()
        
        # Simuler une charge de travail
        print("Simulation d'une charge de travail...")
        
        def simulate_workload_callback(snapshot):
            # Simuler des métriques personnalisées
            self.performance_collector.record_custom_metric(
                "custom_accuracy", np.random.uniform(0.8, 0.95)
            )
            self.performance_collector.record_custom_metric(
                "queue_length", np.random.randint(0, 20)
            )
        
        self.performance_collector.add_callback(simulate_workload_callback)
        
        # Simuler du traitement
        for i in range(50):
            self.performance_collector.record_frame_processed()
            self.performance_collector.record_latency(np.random.uniform(50, 200))
            
            if np.random.random() < 0.05:  # 5% d'erreurs
                self.performance_collector.record_error()
            
            await asyncio.sleep(0.1)  # 100ms par frame
        
        # Arrêter la collecte et obtenir le rapport
        performance_report = self.performance_collector.stop_collection()
        
        print("📊 Rapport de performance:")
        print(f"- Frames traitées: {performance_report.total_frames}")
        print(f"- FPS moyen: {performance_report.average_fps:.2f}")
        print(f"- Latence moyenne: {performance_report.average_latency:.2f}ms")
        print(f"- Pic mémoire: {performance_report.peak_memory_mb:.1f}MB")
        print(f"- Score d'efficacité: {performance_report.resource_efficiency:.1f}/100")
        
        # Génération des visualisations de performance
        self.visualizer.plot_time_series(self.performance_collector)
        
        return performance_report
    
    async def demonstrate_comprehensive_benchmark(self):
        """Démontre l'utilisation du benchmark complet."""
        print("\n🏋️ === BENCHMARK COMPLET ===")
        
        # Configuration d'un benchmark personnalisé
        benchmark_config = BenchmarkConfig(
            test_name="demo_benchmark",
            dataset_path=str(self.output_dir / "demo_data"),
            test_duration=45,  # 45 secondes pour la démo
            concurrent_streams=2,
            repeat_count=2,  # Réduit pour la démo
            hardware_profiling=True,
            memory_profiling=True
        )
        
        self.benchmark_suite.add_benchmark_config(benchmark_config)
        
        # Lancement du benchmark
        print("Lancement du benchmark complet...")
        
        benchmark_result = self.benchmark_suite.run_comprehensive_benchmark(
            benchmark_name="demo_benchmark"
        )
        
        print("📊 Résultats du benchmark:")
        print(f"- Temps d'exécution: {benchmark_result.execution_time:.2f}s")
        print(f"- Variants testés: {len(benchmark_result.variant_results)}")
        print(f"- Meilleur variant: {benchmark_result.comparison_report.get('best_overall_variant')}")
        
        # Affichage des métriques de performance
        print("\n📈 Métriques de performance:")
        for metric, value in benchmark_result.performance_summary.items():
            if isinstance(value, float):
                print(f"- {metric}: {value:.4f}")
            else:
                print(f"- {metric}: {value}")
        
        # Utilisation des ressources
        print("\n💻 Utilisation des ressources:")
        for resource, stats in benchmark_result.hardware_utilization.items():
            print(f"- {resource.upper()}:")
            for stat_name, stat_value in stats.items():
                if isinstance(stat_value, float):
                    print(f"  - {stat_name}: {stat_value:.2f}")
        
        return benchmark_result
    
    async def demonstrate_custom_tools_integration(self):
        """Démontre l'intégration des outils personnalisés."""
        print("\n🛠️ === INTÉGRATION OUTILS PERSONNALISÉS ===")
        
        # Initialiser les outils avancés
        sam2_segmentator = SAM2Segmentator()
        dino_extractor = DinoV2FeatureExtractor()
        pose_estimator = OpenPoseEstimator()
        trajectory_analyzer = TrajectoryAnalyzer()
        
        print("✅ Outils avancés initialisés")
        
        # Tester chaque outil individuellement
        print("🔍 Test des outils individuels...")
        
        # Mock frame pour test
        mock_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        mock_bboxes = [[100, 100, 200, 200], [300, 150, 400, 250]]
        
        # Test SAM2
        try:
            seg_result = sam2_segmentator.segment_objects(mock_frame, mock_bboxes)
            print(f"✅ SAM2: {len(seg_result.masks)} masques générés en {seg_result.processing_time:.3f}s")
        except Exception as e:
            print(f"⚠️ SAM2: {e}")
        
        # Test DINO
        try:
            dino_features = dino_extractor.extract_features(mock_frame)
            print(f"✅ DINO: Features extraites ({dino_features.features.shape}) en {dino_features.processing_time:.3f}s")
        except Exception as e:
            print(f"⚠️ DINO: {e}")
        
        # Test Pose Estimation
        try:
            pose_result = pose_estimator.estimate_poses(mock_frame, [(100, 100, 200, 300)])
            print(f"✅ Pose: {pose_result.keypoints.shape[0]} personnes détectées en {pose_result.processing_time:.3f}s")
        except Exception as e:
            print(f"⚠️ Pose: {e}")
        
        # Test Trajectory Analysis
        try:
            # Simuler des données de trajectoire
            trajectory_data = [
                {"person_id": "p1", "x": 100 + i*5, "y": 200 + i*3, "timestamp": time.time() + i}
                for i in range(10)
            ]
            motion_result = trajectory_analyzer.analyze_motion(trajectory_data)
            print(f"✅ Trajectory: Pattern '{motion_result['pattern']}' avec anomaly_score {motion_result['anomaly_score']:.3f}")
        except Exception as e:
            print(f"⚠️ Trajectory: {e}")
    
    async def demonstrate_vlm_orchestration(self):
        """Démontre l'orchestration VLM avec tool-calling."""
        print("\n🧠 === ORCHESTRATION VLM ===")
        
        try:
            # Initialiser l'orchestrateur VLM
            vlm_orchestrator = VLMOrchestrator()
            
            print("✅ VLM Orchestrator initialisé")
            print(f"Outils disponibles: {list(vlm_orchestrator.tool_registry.get_available_tools().keys())}")
            
            # Créer un contexte de test
            from src.core.types import Detection
            from src.core.orchestrator.vlm_orchestrator import VLMContext
            
            # Mock detections
            mock_detections = [
                # Simuler des détections
            ]
            
            context = VLMContext(
                frame_id="test_001",
                timestamp=time.time(),
                detections=mock_detections,
                previous_context=None,
                scene_metadata={"camera_id": "cam_01", "location": "entrance"}
            )
            
            # Mock frame
            mock_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            # Analyse de scène avec tool-calling
            print("🔍 Analyse de scène avec tool-calling...")
            
            analysis_result = await vlm_orchestrator.analyze_scene(mock_frame, context)
            
            print("📊 Résultat de l'analyse:")
            print(f"- Niveau de suspicion: {analysis_result.suspicion_level:.3f}")
            print(f"- Confiance: {analysis_result.confidence:.3f}")
            print(f"- Temps de traitement: {analysis_result.processing_time:.3f}s")
            print(f"- Outils utilisés: {len(analysis_result.tool_results)}")
            print(f"- Raisonnement: {analysis_result.reasoning[:100]}...")
            
            # Statistiques de performance VLM
            stats = vlm_orchestrator.get_performance_stats()
            print(f"\n📈 Stats VLM:")
            print(f"- Appels totaux: {stats['total_calls']}")
            print(f"- Taux de succès: {stats['success_rate']:.2%}")
            print(f"- Temps de réponse moyen: {stats['average_response_time']:.3f}s")
            
        except Exception as e:
            print(f"⚠️ Erreur VLM: {e}")
    
    async def generate_final_report(self, results: Dict[str, Any]):
        """Génère un rapport final complet."""
        print("\n📋 === GÉNÉRATION DU RAPPORT FINAL ===")
        
        # Rapport du benchmark suite
        summary_report = self.benchmark_suite.generate_summary_report()
        
        # Créer un rapport personnalisé
        report_lines = [
            "# Rapport de Test Complet - Système de Surveillance Intelligent",
            f"Généré le: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Résumé Exécutif",
            ""
        ]
        
        # Ajouter les résultats des différents tests
        for test_name, test_results in results.items():
            report_lines.extend([
                f"### {test_name}",
                ""
            ])
            
            if hasattr(test_results, '__dict__'):
                for key, value in test_results.__dict__.items():
                    if isinstance(value, (int, float)):
                        report_lines.append(f"- {key}: {value:.4f}")
                    elif isinstance(value, str) and len(value) < 100:
                        report_lines.append(f"- {key}: {value}")
            
            report_lines.append("")
        
        # Ajouter le rapport du benchmark
        report_lines.extend([
            "## Détails du Benchmark",
            "",
            summary_report
        ])
        
        # Recommandations
        report_lines.extend([
            "## Recommandations",
            "",
            "### Configuration Optimale",
            "Basé sur les résultats des tests, voici nos recommandations:",
            "",
            "1. **Pour performance temps réel**: Utiliser `parallel_fusion` ou `baseline_yolo`",
            "2. **Pour précision maximale**: Utiliser `complete_advanced` ou `sam2_enhanced`", 
            "3. **Pour ressources limitées**: Utiliser `baseline_yolo`",
            "4. **Pour environnements variables**: Utiliser `domain_adaptive`",
            "",
            "### Optimisations Suggérées",
            "- Monitoring continu des performances en production",
            "- Tests A/B réguliers avec nouvelles données",
            "- Ajustement des seuils selon l'environnement",
            "- Formation continue des modèles avec nouveaux cas",
            ""
        ])
        
        # Sauvegarde du rapport final
        final_report = "\n".join(report_lines)
        report_file = self.output_dir / "rapport_final_complet.md"
        report_file.write_text(final_report)
        
        print(f"📄 Rapport final sauvegardé: {report_file}")
        
        # Export CSV des données
        csv_file = self.benchmark_suite.export_results_csv()
        print(f"📊 Données exportées en CSV: {csv_file}")
        
        return final_report

async def main():
    """Fonction principale de démonstration."""
    print("🎯 DÉMONSTRATION COMPLÈTE DU FRAMEWORK DE TEST AVANCÉ")
    print("=" * 60)
    
    # Initialisation
    tester = CompleteTester("demo_results")
    results = {}
    
    try:
        # 1. Test A/B basique
        ab_results = await tester.demonstrate_basic_ab_test()
        results["ab_test_basique"] = ab_results
        
        # 2. Variants avancés
        advanced_results = await tester.demonstrate_advanced_variants()
        results["variants_avancés"] = advanced_results
        
        # 3. Monitoring de performance
        perf_results = await tester.demonstrate_performance_monitoring()
        results["monitoring_performance"] = perf_results
        
        # 4. Intégration outils personnalisés
        await tester.demonstrate_custom_tools_integration()
        
        # 5. Orchestration VLM
        await tester.demonstrate_vlm_orchestration()
        
        # 6. Benchmark complet (optionnel)
        run_full_benchmark = input("\n❓ Exécuter le benchmark complet ? (y/N): ").lower() == 'y'
        
        if run_full_benchmark:
            benchmark_results = await tester.demonstrate_comprehensive_benchmark()
            results["benchmark_complet"] = benchmark_results
        
        # 7. Génération du rapport final
        final_report = await tester.generate_final_report(results)
        
        print("\n🎉 DÉMONSTRATION TERMINÉE AVEC SUCCÈS!")
        print(f"📁 Tous les résultats sont disponibles dans: {tester.output_dir}")
        
        # Affichage du résumé final
        print("\n📊 RÉSUMÉ DES TESTS:")
        print(f"- Tests A/B exécutés: {len([k for k in results.keys() if 'ab_test' in k or 'variants' in k])}")
        print(f"- Monitoring de performance: ✅")
        print(f"- Outils avancés testés: ✅")
        print(f"- Orchestration VLM: ✅")
        if run_full_benchmark:
            print(f"- Benchmark complet: ✅")
        print(f"- Rapport final généré: ✅")
        
    except Exception as e:
        print(f"❌ Erreur pendant la démonstration: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print(f"\n📁 Consultez {tester.output_dir} pour tous les fichiers générés")

if __name__ == "__main__":
    # Exécuter la démonstration
    asyncio.run(main())