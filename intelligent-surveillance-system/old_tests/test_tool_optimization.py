#!/usr/bin/env python3
"""
🧪 Script de Test Simple pour l'Optimisation des Outils
======================================================

Test rapide pour vérifier que le système d'optimisation fonctionne.
"""

import asyncio
import sys
import os
from pathlib import Path

# Ajout du chemin pour les imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from src.testing.tool_optimization_benchmark import (
        ToolOptimizationBenchmark,
        quick_tool_evaluation
    )
    from src.core.orchestrator.adaptive_orchestrator import create_adaptive_orchestrator
    from src.core.orchestrator.tool_calling_vlm import create_tool_calling_vlm
    print("✅ Tous les modules importés avec succès")
except ImportError as e:
    print(f"❌ Erreur d'import: {e}")
    print("Vérifiez que tous les fichiers sont présents")
    sys.exit(1)


async def test_basic_functionality():
    """Test de fonctionnalité de base."""
    
    print("\n🧪 Test 1: Initialisation du Benchmark")
    try:
        benchmark = ToolOptimizationBenchmark()
        print("✅ Benchmark initialisé")
    except Exception as e:
        print(f"❌ Erreur benchmark: {e}")
        return False
    
    print("\n🧪 Test 2: Génération des cas de test")
    try:
        await benchmark.load_test_cases()
        print(f"✅ {len(benchmark.test_cases)} cas de test générés")
    except Exception as e:
        print(f"❌ Erreur cas de test: {e}")
        return False
    
    print("\n🧪 Test 3: Génération des combinaisons d'outils")
    try:
        combinations = benchmark.generate_tool_combinations()
        print(f"✅ {len(combinations)} combinaisons générées")
        print(f"📝 Exemples: {combinations[:3]}")
    except Exception as e:
        print(f"❌ Erreur combinaisons: {e}")
        return False
    
    print("\n🧪 Test 4: Orchestrateur adaptatif")
    try:
        orchestrator = create_adaptive_orchestrator()
        status = orchestrator.get_adaptive_status()
        print("✅ Orchestrateur adaptatif créé")
        print(f"📊 Outils optimaux: {status['current_optimal_tools']}")
    except Exception as e:
        print(f"❌ Erreur orchestrateur: {e}")
        return False
    
    print("\n🧪 Test 5: VLM avec Tool Calling")
    try:
        vlm = create_tool_calling_vlm()
        stats = vlm.get_tool_calling_stats()
        print("✅ VLM avec tool calling créé")
        print(f"🔧 Outils enregistrés: {stats['registered_tools']}")
    except Exception as e:
        print(f"❌ Erreur VLM tool calling: {e}")
        return False
    
    return True


async def test_quick_evaluation():
    """Test d'évaluation rapide d'outils."""
    
    print("\n🚀 Test d'Évaluation Rapide")
    
    try:
        # Test avec une petite combinaison d'outils
        test_tools = ["dino_features", "pose_estimator", "multimodal_fusion"]
        
        print(f"🔧 Test des outils: {test_tools}")
        
        # Note: Ce test utilisera des données synthétiques
        result = await quick_tool_evaluation(test_tools)
        
        print("✅ Évaluation rapide terminée:")
        print(f"   📊 Quality Score: {result['quality_score']:.3f}")
        print(f"   🎯 F1 Score: {result['f1_score']:.3f}")
        print(f"   ⏱️  Temps: {result['response_time']:.2f}s")
        print(f"   💰 Coût: {result['total_cost']:.1f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur évaluation rapide: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_adaptive_learning():
    """Test d'apprentissage adaptatif simple."""
    
    print("\n🧠 Test d'Apprentissage Adaptatif")
    
    try:
        orchestrator = create_adaptive_orchestrator(enable_learning=True)
        
        # Simulation de quelques analyses
        test_scenarios = [
            {
                "context": {"location": "entrance", "time": "morning"},
                "detections": [{"class_name": "person"}]
            },
            {
                "context": {"location": "electronics", "time": "evening"}, 
                "detections": [{"class_name": "person"}, {"class_name": "backpack"}]
            }
        ]
        
        print("📚 Simulation d'apprentissage...")
        
        for i, scenario in enumerate(test_scenarios):
            print(f"   🔄 Scénario {i+1}: {scenario['context']['location']}")
            
            # Note: Utilise des données factices pour test
            analysis = await orchestrator.analyze_surveillance_frame(
                frame_data="test_frame_data",
                detections=scenario["detections"],
                context=scenario["context"]
            )
            
            print(f"   ✅ Outils utilisés: {len(analysis.tools_used)}")
        
        # Statut après apprentissage
        status = orchestrator.get_adaptive_status()
        print(f"\n📈 Apprentissage effectué:")
        print(f"   🎯 Patterns appris: {status['learning_stats']['context_patterns_learned']}")
        print(f"   📊 Exécutions: {status['learning_stats']['executions_in_history']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur apprentissage adaptatif: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Test principal."""
    
    print("🎯 TESTS DU SYSTÈME D'OPTIMISATION DES OUTILS")
    print("=" * 60)
    
    # Vérification de l'environnement
    print(f"🐍 Python: {sys.version.split()[0]}")
    print(f"📁 Répertoire: {Path.cwd()}")
    
    # Tests de base
    basic_ok = await test_basic_functionality()
    
    if not basic_ok:
        print("\n❌ Tests de base échoués - Arrêt")
        return False
    
    print("\n" + "="*60)
    
    # Test d'évaluation rapide
    eval_ok = await test_quick_evaluation()
    
    # Test d'apprentissage adaptatif
    learning_ok = await test_adaptive_learning()
    
    # Résumé final
    print("\n" + "="*60)
    print("📋 RÉSUMÉ DES TESTS")
    print("="*60)
    
    results = {
        "Tests de base": "✅" if basic_ok else "❌",
        "Évaluation rapide": "✅" if eval_ok else "❌", 
        "Apprentissage adaptatif": "✅" if learning_ok else "❌"
    }
    
    for test_name, status in results.items():
        print(f"{status} {test_name}")
    
    all_passed = all([basic_ok, eval_ok, learning_ok])
    
    if all_passed:
        print("\n🎉 TOUS LES TESTS PASSÉS !")
        print("🚀 Le système d'optimisation est prêt à être utilisé")
        print("\nCommandes disponibles:")
        print("  python examples/tool_optimization_demo.py --mode benchmark")
        print("  python examples/tool_optimization_demo.py --mode adaptive") 
        print("  python examples/tool_optimization_demo.py --mode full")
    else:
        print("\n⚠️  Certains tests ont échoué")
        print("📝 Vérifiez les erreurs ci-dessus avant d'utiliser le système complet")
    
    return all_passed


if __name__ == "__main__":
    asyncio.run(main())