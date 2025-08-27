#!/usr/bin/env python3
"""
🧪 Lanceur de Tests Unitaires - CPU Seulement
==============================================

Exécute tous les tests unitaires sans nécessiter de GPU.
Stratégie complète : Mocks + Logique Métier + Interfaces + Performance.
"""

import sys
import time
import importlib
from pathlib import Path
from typing import Dict, List, Any

# Configuration du path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "tests" / "unit"))
sys.path.insert(0, str(PROJECT_ROOT))

class UnitTestRunner:
    """Lanceur de tests unitaires complet"""
    
    def __init__(self):
        self.results = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'errors': [],
            'execution_time': 0,
            'test_suites': {}
        }
        
    def run_test_module(self, module_name: str, description: str) -> Dict[str, Any]:
        """Exécute un module de test"""
        print(f"\n{'='*50}")
        print(f"🧪 {description}")
        print(f"{'='*50}")
        
        suite_results = {
            'total': 0,
            'passed': 0,
            'failed': 0,
            'errors': [],
            'execution_time': 0
        }
        
        start_time = time.time()
        
        try:
            # Import dynamique du module
            module = importlib.import_module(module_name)
            
            # Exécuter si le module a une fonction main
            if hasattr(module, '__main__') and callable(getattr(module, 'main', None)):
                print(f"Exécution de {module_name} via main()...")
                module.main()
            else:
                # Exécution directe du module
                print(f"Exécution directe de {module_name}...")
                exec(f"import {module_name}")
        
        except Exception as e:
            print(f"❌ Erreur dans {module_name}: {e}")
            suite_results['errors'].append(str(e))
            suite_results['failed'] += 1
        
        suite_results['execution_time'] = time.time() - start_time
        
        return suite_results
    
    def run_strategy_demonstration(self):
        """Démonstration de la stratégie de tests"""
        print("🎯 DÉMONSTRATION STRATÉGIE DE TESTS")
        print("=" * 50)
        
        try:
            # Import et exécution du module de stratégie
            from test_strategy_explained import MockDinoV2, MockSAM2, MockKimiVL
            from test_strategy_explained import calculate_suspicion_score, select_optimal_tools
            from test_strategy_explained import validate_detection_format, CPUPerformanceTester
            
            print("\n1. 🎭 DÉMONSTRATION MOCKS:")
            
            # Mock DINO v2
            mock_dino = MockDinoV2()
            dino_result = mock_dino.extract_features(b"fake_image")
            print(f"   ✅ Mock DINO v2: {dino_result['confidence']:.2f} confidence, {dino_result['processing_time']*1000:.1f}ms")
            
            # Mock SAM2
            mock_sam = MockSAM2()
            sam_result = mock_sam.segment_everything(b"fake_image")
            print(f"   ✅ Mock SAM2: {sam_result['total_segments']} segments, {sam_result['processing_time']*1000:.1f}ms")
            
            # Mock Kimi-VL
            mock_vlm = MockKimiVL()
            vlm_result = mock_vlm.analyze_scene(b"fake_image", "person detected")
            print(f"   ✅ Mock Kimi-VL: {vlm_result['suspicion_level']:.2f} suspicion, {vlm_result['processing_time']*1000:.1f}ms")
            
            print("\n2. ✅ DÉMONSTRATION LOGIQUE MÉTIER:")
            
            # Test suspicion
            detections = [{"class_name": "person", "confidence": 0.8}, {"class_name": "bottle", "confidence": 0.7}]
            context = {"time_in_zone": 45}
            suspicion = calculate_suspicion_score(detections, context)
            print(f"   ✅ Calcul suspicion: {suspicion:.2f} (personne + temps + objets)")
            
            # Test sélection outils
            tool_context = {"persons_detected": 2, "suspicion_level": 0.8, "mode": "thorough"}
            tools = select_optimal_tools(tool_context, {})
            print(f"   ✅ Sélection outils: {len(tools)} outils ({', '.join(tools[:3])}...)")
            
            print("\n3. 📋 DÉMONSTRATION VALIDATION:")
            
            # Test validation
            valid_detection = {"bbox": [100, 100, 200, 200], "confidence": 0.8, "class_name": "person"}
            invalid_detection = {"bbox": [100, 100], "confidence": 1.5, "class_name": "person"}
            
            valid_result = validate_detection_format(valid_detection)
            invalid_result = validate_detection_format(invalid_detection)
            print(f"   ✅ Validation format: Valide={valid_result}, Invalide={invalid_result}")
            
            print("\n4. ⚡ DÉMONSTRATION PERFORMANCE:")
            
            # Test performance
            tester = CPUPerformanceTester()
            tracking_perf = tester.benchmark_tracking_algorithm(100)
            suspicion_perf = tester.benchmark_suspicion_calculation(1000)
            
            print(f"   ✅ Performance tracking: {tracking_perf['fps']:.0f} FPS")
            print(f"   ✅ Performance suspicion: {suspicion_perf['calculations_per_second']:.0f} calc/sec")
            
            print(f"\n🎉 Stratégie démontrée avec succès - Aucun GPU requis!")
            
            return True
            
        except Exception as e:
            print(f"❌ Erreur démonstration: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_specific_component_tests(self):
        """Tests de composants spécifiques"""
        print("\n🔧 TESTS DE COMPOSANTS SPÉCIFIQUES")
        print("=" * 50)
        
        components_results = {}
        
        # Tests YOLO Logic
        try:
            print("\n🎯 Tests YOLO - Logique Pure:")
            from test_yolo_logic_only import TestYOLOLogicPure, TestYOLOConfigurationLogic
            
            yolo_logic = TestYOLOLogicPure()
            yolo_logic.test_detection_parsing()
            yolo_logic.test_confidence_filtering()
            yolo_logic.test_class_filtering_supermarket()
            yolo_logic.test_bbox_area_calculation()
            yolo_logic.test_detection_sorting()
            yolo_logic.test_nms_logic_simulation()
            yolo_logic.test_detection_statistics()
            
            yolo_config = TestYOLOConfigurationLogic()
            yolo_config.test_threshold_optimization()
            yolo_config.test_class_priority_logic()
            
            print("   ✅ Tests YOLO: 9 tests passés")
            components_results['yolo'] = {'passed': 9, 'failed': 0}
            
        except Exception as e:
            print(f"   ❌ Tests YOLO échoués: {e}")
            components_results['yolo'] = {'passed': 0, 'failed': 9}
        
        # Tests Tracking Logic  
        try:
            print("\n🎯 Tests Tracking - Logique Pure:")
            from test_tracking_logic import TestTrackingMath, TestTrackPrediction, TestSimpleTracker
            
            tracking_math = TestTrackingMath()
            tracking_math.test_bbox_center_calculation()
            tracking_math.test_bbox_area_calculation()
            tracking_math.test_euclidean_distance()
            tracking_math.test_iou_calculation()
            
            tracking_pred = TestTrackPrediction()
            tracking_pred.test_static_prediction()
            tracking_pred.test_velocity_prediction()
            
            tracker_tests = TestSimpleTracker()
            tracker_tests.test_tracker_initialization()
            tracker_tests.test_first_frame_detection()
            tracker_tests.test_track_association_same_position()
            tracker_tests.test_track_association_moved()
            tracker_tests.test_new_detection_different_position()
            tracker_tests.test_track_disappearance_and_reappearance()
            tracker_tests.test_multiple_objects_tracking()
            tracker_tests.test_tracker_performance_metrics()
            
            print("   ✅ Tests Tracking: 13 tests passés")
            components_results['tracking'] = {'passed': 13, 'failed': 0}
            
        except Exception as e:
            print(f"   ❌ Tests Tracking échoués: {e}")
            import traceback
            traceback.print_exc()
            components_results['tracking'] = {'passed': 0, 'failed': 13}
        
        # Tests Orchestration Logic
        try:
            print("\n🎭 Tests Orchestration - Logique Pure:")
            from test_orchestration_logic import (TestOrchestrationLogic, TestContextAnalysis, 
                                                 TestToolSelection, TestSuspicionCalculation,
                                                 TestProcessingTimeEstimation, TestPerformanceOptimization)
            
            # Config tests
            config_tests = TestOrchestrationLogic()
            config_tests.test_config_initialization()
            config_tests.test_config_custom()
            
            # Context tests
            context_tests = TestContextAnalysis()
            context_tests.setup_method()
            context_tests.test_normal_scenario_analysis()
            context_tests.test_suspicious_scenario_analysis()
            context_tests.test_crowded_scenario_analysis()
            
            # Tool selection tests
            tool_tests = TestToolSelection()
            tool_tests.setup_method()
            tool_tests.test_fast_mode_selection()
            tool_tests.test_balanced_mode_selection()
            tool_tests.test_thorough_mode_selection()
            tool_tests.test_adaptive_tool_selection()
            tool_tests.test_concurrent_tools_limit()
            
            # Suspicion tests
            suspicion_tests = TestSuspicionCalculation()
            suspicion_tests.setup_method()
            suspicion_tests.test_normal_scenario_suspicion()
            suspicion_tests.test_suspicious_scenario_suspicion()
            suspicion_tests.test_suspicion_with_tool_results()
            
            # Time estimation tests
            time_tests = TestProcessingTimeEstimation()
            time_tests.setup_method()
            time_tests.test_fast_mode_time_estimation()
            time_tests.test_thorough_mode_time_estimation()
            time_tests.test_tool_count_impact_on_time()
            
            # Performance optimization
            perf_tests = TestPerformanceOptimization()
            perf_tests.test_performance_tracking_update()
            perf_tests.test_tool_exclusion_based_on_performance()
            
            print("   ✅ Tests Orchestration: 18 tests passés")
            components_results['orchestration'] = {'passed': 18, 'failed': 0}
            
        except Exception as e:
            print(f"   ❌ Tests Orchestration échoués: {e}")
            components_results['orchestration'] = {'passed': 0, 'failed': 18}
        
        return components_results
    
    def generate_final_report(self, strategy_success: bool, components_results: Dict):
        """Génère le rapport final"""
        print("\n" + "="*60)
        print("📊 RAPPORT FINAL - TESTS UNITAIRES CPU")
        print("="*60)
        
        total_passed = sum(r['passed'] for r in components_results.values())
        total_failed = sum(r['failed'] for r in components_results.values())
        total_tests = total_passed + total_failed
        
        print(f"\n📈 RÉSULTATS GLOBAUX:")
        print(f"   Total tests exécutés: {total_tests}")
        print(f"   Tests réussis: {total_passed}")
        print(f"   Tests échoués: {total_failed}")
        print(f"   Taux de succès: {(total_passed/total_tests*100):.1f}%")
        
        print(f"\n🎯 RÉSULTATS PAR COMPOSANT:")
        for component, results in components_results.items():
            status = "✅" if results['failed'] == 0 else "❌"
            print(f"   {status} {component.upper()}: {results['passed']} passés, {results['failed']} échoués")
        
        print(f"\n🎭 STRATÉGIE DE TESTS:")
        strategy_status = "✅" if strategy_success else "❌"
        print(f"   {strategy_status} Démonstration stratégie complète")
        print(f"   ✅ Mocks/Simulations: GPU-intensive tools simulés")
        print(f"   ✅ Logique métier: Algorithmes testés sans ML")
        print(f"   ✅ Validation interfaces: Formats de données vérifiés")
        print(f"   ✅ Performance CPU: Vitesse mesurée sans GPU")
        
        print(f"\n🏆 AVANTAGES DE CETTE APPROCHE:")
        print(f"   • Tests rapides (pas d'attente GPU)")
        print(f"   • Exécution sur n'importe quelle machine")
        print(f"   • Validation de l'architecture métier")
        print(f"   • Détection précoce des bugs logiques")
        print(f"   • CI/CD friendly (pas de dépendances GPU)")
        
        if total_failed == 0 and strategy_success:
            print(f"\n🎉 SUCCÈS COMPLET!")
            print(f"   Votre architecture de surveillance est validée")
            print(f"   Tous les composants métier fonctionnent correctement")
            print(f"   Prêt pour intégration avec les vrais modèles GPU")
        else:
            print(f"\n⚠️ AMÉLIORATIONS NÉCESSAIRES:")
            if total_failed > 0:
                print(f"   • Corriger les {total_failed} tests échoués")
            if not strategy_success:
                print(f"   • Vérifier la stratégie de tests")
    
    def run_all_tests(self):
        """Exécute tous les tests"""
        overall_start_time = time.time()
        
        print("🧪 TESTS UNITAIRES - SURVEILLANCE INTELLIGENTE")
        print("CPU Only - Aucun GPU Requis")
        print("=" * 60)
        
        # 1. Démonstration stratégie
        strategy_success = self.run_strategy_demonstration()
        
        # 2. Tests composants spécifiques
        components_results = self.run_specific_component_tests()
        
        # 3. Rapport final
        overall_time = time.time() - overall_start_time
        print(f"\n⏱️ Temps total d'exécution: {overall_time:.2f} secondes")
        
        self.generate_final_report(strategy_success, components_results)
        
        return strategy_success and all(r['failed'] == 0 for r in components_results.values())

def main():
    """Point d'entrée principal"""
    print("🚀 Lancement des Tests Unitaires CPU...")
    
    runner = UnitTestRunner()
    
    try:
        success = runner.run_all_tests()
        
        if success:
            print(f"\n✨ Tous les tests ont réussi!")
            return 0
        else:
            print(f"\n💥 Certains tests ont échoué")
            return 1
            
    except KeyboardInterrupt:
        print(f"\n⚠️ Tests interrompus par l'utilisateur")
        return 130
    except Exception as e:
        print(f"\n💥 Erreur inattendue: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)