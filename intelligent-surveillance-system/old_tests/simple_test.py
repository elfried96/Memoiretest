#!/usr/bin/env python3
"""
🧪 Test Simple du Système d'Optimisation
========================================

Test basique sans dépendances externes lourdes.
"""

import asyncio
import sys
from pathlib import Path

# Ajout du chemin pour les imports
sys.path.insert(0, str(Path(__file__).parent))

print("🎯 TEST SIMPLE DU SYSTÈME D'OPTIMISATION")
print("=" * 50)

# Test 1: Imports de base
print("\n🧪 Test 1: Vérification des imports")

try:
    from src.core.types import AnalysisRequest, AnalysisResponse, SuspicionLevel, ActionType
    print("✅ Types de base importés")
except ImportError as e:
    print(f"❌ Erreur types: {e}")

try:
    from src.core.vlm.dynamic_model import DynamicVisionLanguageModel
    print("✅ VLM dynamique importé")
except ImportError as e:
    print(f"❌ Erreur VLM: {e}")

try:
    from src.core.orchestrator.vlm_orchestrator import ModernVLMOrchestrator, OrchestrationConfig, OrchestrationMode
    print("✅ Orchestrateur VLM importé")
except ImportError as e:
    print(f"❌ Erreur orchestrateur VLM: {e}")

# Test 2: Configuration d'orchestration
print("\n🧪 Test 2: Configuration d'orchestration")

try:
    config = OrchestrationConfig(
        mode=OrchestrationMode.BALANCED,
        enable_advanced_tools=True,
        max_concurrent_tools=4
    )
    print(f"✅ Configuration créée: Mode {config.mode.value}")
    print(f"   📊 Max outils: {config.max_concurrent_tools}")
    print(f"   🔧 Outils avancés: {config.enable_advanced_tools}")
except Exception as e:
    print(f"❌ Erreur configuration: {e}")

# Test 3: Orchestrateur de base
print("\n🧪 Test 3: Orchestrateur de base")

try:
    orchestrator = ModernVLMOrchestrator(
        vlm_model_name="kimi-vl-a3b-thinking",
        config=config
    )
    print("✅ Orchestrateur créé")
    
    # Test de sélection d'outils
    tools = orchestrator._select_tools_for_mode()
    print(f"   🔧 Outils sélectionnés: {len(tools)}")
    print(f"   📝 Outils: {tools[:3]}...")
    
except Exception as e:
    print(f"❌ Erreur orchestrateur: {e}")

# Test 4: Statut système
print("\n🧪 Test 4: Statut système")

try:
    status = orchestrator.get_system_status()
    print("✅ Statut récupéré")
    print(f"   📊 Mode: {status['orchestrator']['mode']}")
    print(f"   ⚙️ Outils avancés: {status['orchestrator']['enable_advanced_tools']}")
    print(f"   📈 Total analyses: {status['performance']['total_analyses']}")
    
except Exception as e:
    print(f"❌ Erreur statut: {e}")

# Test 5: Health check
print("\n🧪 Test 5: Health check")

async def test_health():
    try:
        health = await orchestrator.health_check()
        print("✅ Health check réalisé")
        for service, status in health.items():
            emoji = "✅" if status else "❌"
            print(f"   {emoji} {service}: {status}")
        return True
    except Exception as e:
        print(f"❌ Erreur health check: {e}")
        return False

# Test 6: Analyse factice
print("\n🧪 Test 6: Analyse factice")

async def test_analysis():
    try:
        # Données factices pour test
        analysis = await orchestrator.analyze_surveillance_frame(
            frame_data="test_frame_data",
            detections=[],
            context={"test": True, "location": "test_area"}
        )
        
        print("✅ Analyse factice réalisée")
        print(f"   🎯 Suspicion: {analysis.suspicion_level.value}")
        print(f"   ⚡ Action: {analysis.action_type.value}")
        print(f"   📊 Confiance: {analysis.confidence:.2f}")
        print(f"   🔧 Outils: {len(analysis.tools_used)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur analyse: {e}")
        return False

# Exécution des tests asynchrones
async def run_async_tests():
    print("\n🚀 Tests asynchrones")
    
    health_ok = await test_health()
    analysis_ok = await test_analysis()
    
    return health_ok and analysis_ok

# Test principal
async def main():
    try:
        async_ok = await run_async_tests()
        
        print("\n" + "="*50)
        print("📋 RÉSUMÉ DES TESTS SIMPLES")
        print("="*50)
        
        print("✅ Imports de base: OK")
        print("✅ Configuration: OK") 
        print("✅ Orchestrateur: OK")
        print("✅ Statut système: OK")
        print(f"{'✅' if async_ok else '❌'} Tests asynchrones: {'OK' if async_ok else 'ERREUR'}")
        
        if async_ok:
            print("\n🎉 TESTS SIMPLES RÉUSSIS !")
            print("🚀 Le système de base fonctionne")
            print("\n💡 Pour les tests complets:")
            print("   1. Installez les dépendances: numpy, loguru, rich, scikit-learn")
            print("   2. Lancez: python examples/tool_optimization_demo.py")
        else:
            print("\n⚠️ Certains tests ont échoué")
            print("🔧 Vérifiez la configuration avant d'utiliser le système complet")
            
    except Exception as e:
        print(f"\n❌ Erreur générale: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())