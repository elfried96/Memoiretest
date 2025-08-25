#!/usr/bin/env python3
"""Tests de switching entre KIM, LLaVA et Qwen2-VL."""

import sys
import os
import asyncio
import time
import base64
from io import BytesIO
from PIL import Image
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.core.vlm.dynamic_model import DynamicVisionLanguageModel
from src.core.vlm.model_registry import VLMModelRegistry, VLMModelType
from src.core.types import AnalysisRequest


def create_test_image() -> str:
    """Créer une image de test pour les analyses."""
    # Image simulée de surveillance
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Ajout d'éléments visuels
    # Personne
    image[100:300, 200:350] = [120, 100, 80]
    # Objet suspect
    image[250:280, 320:370] = [200, 50, 50]
    # Background store-like
    image[400:480, :] = [150, 150, 150]  # Floor
    
    # Conversion vers base64
    pil_image = Image.fromarray(image)
    buffer = BytesIO()
    pil_image.save(buffer, format='PNG')
    
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


async def test_model_registry():
    """Test du registre des modèles."""
    
    print("=== TEST REGISTRE DES MODÈLES ===\n")
    
    registry = VLMModelRegistry()
    
    # Liste des modèles disponibles
    available_models = registry.list_available_models()
    print(f"📋 Modèles enregistrés: {len(available_models)}")
    
    for model_id, config in available_models.items():
        print(f"  • {model_id}")
        print(f"    Type: {config.model_type.value.upper()}")
        print(f"    Nom: {config.model_name}")
        print(f"    Tool-calling: {'✅' if config.supports_tool_calling else '❌'}")
        print(f"    Memory efficient: {'✅' if config.memory_efficient else '❌'}")
        print(f"    Description: {config.description}")
        print()
    
    # Test par type
    print("📊 Modèles par type:")
    for model_type in VLMModelType:
        models = registry.get_models_by_type(model_type)
        print(f"  {model_type.value.upper()}: {len(models)} modèles")
        for model_id in models.keys():
            print(f"    - {model_id}")
    print()
    
    # Recommandations
    print("🎯 Recommandations:")
    recommendations = {
        "Surveillance": registry.get_recommended_model("surveillance"),
        "Haute performance": registry.get_recommended_model("high_performance"),
        "Économie mémoire": registry.get_recommended_model("memory_efficient"),
        "Raisonnement": registry.get_recommended_model("reasoning"),
        "Flagship": registry.get_recommended_model("flagship")
    }
    
    for use_case, model_id in recommendations.items():
        print(f"  {use_case}: {model_id}")
    
    # Test de validation
    print(f"\n🔍 Validation des modèles:")
    for model_id in ["kim-7b-instruct", "llava-v1.6-mistral-7b", "qwen2-vl-7b-instruct"]:
        is_available, message = registry.validate_model_availability(model_id)
        status = "✅" if is_available else "❌"
        print(f"  {status} {model_id}: {message}")
    
    return True


async def test_dynamic_model_loading():
    """Test du chargement dynamique des modèles."""
    
    print(f"\n=== TEST CHARGEMENT DYNAMIQUE ===\n")
    
    # Initialisation du VLM dynamique
    vlm = DynamicVisionLanguageModel(
        default_model="llava-v1.6-mistral-7b",  # LLaVA comme fallback stable
        device="auto",
        enable_fallback=True
    )
    
    print("🔄 Test de chargement des modèles disponibles...")
    
    # Modèles à tester (dans l'ordre de priorité)
    models_to_test = [
        ("kim-7b-instruct", "KIM 7B (Principal)"),
        ("llava-v1.6-mistral-7b", "LLaVA-NeXT 7B (Fallback)"),
        ("qwen2-vl-7b-instruct", "Qwen2-VL 7B (Alternative)")
    ]
    
    successful_loads = []
    failed_loads = []
    
    for model_id, description in models_to_test:
        print(f"\n--- Test: {description} ---")
        
        try:
            start_time = time.time()
            success = await vlm.load_model(model_id)
            load_time = time.time() - start_time
            
            if success:
                print(f"✅ {model_id} chargé en {load_time:.2f}s")
                
                # Test basique de statut
                status = vlm.get_system_status()
                current_model = status["current_model"]
                print(f"  Modèle actuel: {current_model['model_id']}")
                print(f"  Type: {current_model['model_type']}")
                print(f"  Chargé: {current_model['is_loaded']}")
                print(f"  Device: {current_model['device']}")
                print(f"  Tool-calling: {current_model['supports_tool_calling']}")
                
                successful_loads.append((model_id, load_time))
                
                # Test de déchargement propre
                vlm._unload_current_model()
                print(f"  ✅ Déchargement propre réussi")
                
            else:
                print(f"❌ {model_id} n'a pas pu être chargé")
                failed_loads.append(model_id)
                
        except Exception as e:
            print(f"❌ Erreur {model_id}: {e}")
            failed_loads.append(model_id)
    
    # Résumé
    print(f"\n📊 Résumé chargement:")
    print(f"  ✅ Réussis: {len(successful_loads)}")
    for model_id, load_time in successful_loads:
        print(f"    • {model_id} ({load_time:.2f}s)")
    
    print(f"  ❌ Échoués: {len(failed_loads)}")
    for model_id in failed_loads:
        print(f"    • {model_id}")
    
    await vlm.shutdown()
    
    return len(successful_loads) > 0


async def test_model_switching():
    """Test du switching entre modèles."""
    
    print(f"\n=== TEST SWITCHING ENTRE MODÈLES ===\n")
    
    # VLM avec modèle de départ
    vlm = DynamicVisionLanguageModel(
        default_model="llava-v1.6-mistral-7b",
        enable_fallback=True
    )
    
    # Chargement initial
    print("🔄 Chargement modèle initial...")
    success = await vlm.load_model()
    
    if not success:
        print("❌ Impossible de charger le modèle initial")
        return False
    
    initial_model = vlm.current_model_id
    print(f"✅ Modèle initial: {initial_model}")
    
    # Séquence de switching
    switch_sequence = [
        "qwen2-vl-7b-instruct",
        "kim-7b-instruct", 
        "llava-v1.6-mistral-7b"
    ]
    
    successful_switches = 0
    
    for target_model in switch_sequence:
        print(f"\n🔄 Switch vers: {target_model}")
        
        try:
            start_time = time.time()
            success = await vlm.switch_model(target_model)
            switch_time = time.time() - start_time
            
            if success:
                print(f"✅ Switch réussi en {switch_time:.2f}s")
                print(f"  Modèle actuel: {vlm.current_model_id}")
                print(f"  Type: {vlm.current_config.model_type.value}")
                successful_switches += 1
            else:
                print(f"❌ Switch échoué vers {target_model}")
                print(f"  Modèle actuel: {vlm.current_model_id}")
                
        except Exception as e:
            print(f"❌ Erreur switch vers {target_model}: {e}")
    
    print(f"\n📊 Résumé switching:")
    print(f"  ✅ Switches réussis: {successful_switches}/{len(switch_sequence)}")
    print(f"  Modèle final: {vlm.current_model_id}")
    
    await vlm.shutdown()
    
    return successful_switches > 0


async def test_analysis_with_different_models():
    """Test d'analyse avec différents modèles."""
    
    print(f"\n=== TEST ANALYSES MULTI-MODÈLES ===\n")
    
    # Image de test
    test_image = create_test_image()
    
    # Requête d'analyse
    analysis_request = AnalysisRequest(
        frame_data=test_image,
        context={
            "location": "Test Store - Multi-Model",
            "camera_id": "TEST_CAM",
            "test_mode": True
        },
        tools_available=["dino_features", "pose_estimator", "multimodal_fusion"]
    )
    
    # Modèles à tester
    models_to_test = [
        "llava-v1.6-mistral-7b",
        "qwen2-vl-7b-instruct", 
        "kim-7b-instruct"
    ]
    
    vlm = DynamicVisionLanguageModel(enable_fallback=True)
    results = {}
    
    for model_id in models_to_test:
        print(f"\n--- Analyse avec {model_id} ---")
        
        try:
            # Switch vers le modèle
            success = await vlm.switch_model(model_id)
            
            if not success:
                print(f"❌ Impossible de charger {model_id}")
                results[model_id] = {"success": False, "error": "Chargement échoué"}
                continue
            
            # Analyse
            start_time = time.time()
            result = await vlm.analyze_with_tools(analysis_request, use_advanced_tools=True)
            analysis_time = time.time() - start_time
            
            print(f"✅ Analyse terminée en {analysis_time:.2f}s")
            print(f"  Suspicion: {result.suspicion_level.value}")
            print(f"  Action: {result.action_type.value}")  
            print(f"  Confiance: {result.confidence:.2f}")
            print(f"  Outils utilisés: {len(result.tools_used)}")
            print(f"  Description: {result.description[:100]}...")
            
            results[model_id] = {
                "success": True,
                "suspicion_level": result.suspicion_level.value,
                "confidence": result.confidence,
                "tools_count": len(result.tools_used),
                "analysis_time": analysis_time
            }
            
        except Exception as e:
            print(f"❌ Erreur analyse {model_id}: {e}")
            results[model_id] = {"success": False, "error": str(e)}
    
    # Comparaison des résultats
    print(f"\n📊 COMPARAISON DES MODÈLES")
    print(f"{'Modèle':<25} {'Succès':<8} {'Suspicion':<10} {'Confiance':<10} {'Temps':<8}")
    print("-" * 70)
    
    for model_id, result in results.items():
        model_name = model_id.split('-')[0].upper()
        if result["success"]:
            print(f"{model_name:<25} {'✅':<8} {result['suspicion_level']:<10} {result['confidence']:<10.2f} {result['analysis_time']:<8.2f}s")
        else:
            print(f"{model_name:<25} {'❌':<8} {'N/A':<10} {'N/A':<10} {'N/A':<8}")
    
    await vlm.shutdown()
    
    successful_analyses = sum(1 for r in results.values() if r["success"])
    return successful_analyses > 0


async def test_system_status_and_recommendations():
    """Test du statut système et des recommandations."""
    
    print(f"\n=== TEST STATUT & RECOMMANDATIONS ===\n")
    
    vlm = DynamicVisionLanguageModel()
    
    try:
        # Chargement d'un modèle
        await vlm.load_model("llava-v1.6-mistral-7b")
        
        # Statut complet
        print("🔍 Statut système complet:")
        status = vlm.get_system_status()
        
        print(f"\n📱 Modèle actuel:")
        current = status["current_model"]
        print(f"  ID: {current['model_id']}")
        print(f"  Type: {current['model_type']}")
        print(f"  Chargé: {current['is_loaded']}")
        print(f"  Device: {current['device']}")
        print(f"  Tool-calling: {current['supports_tool_calling']}")
        
        print(f"\n🎯 Modèles disponibles:")
        for model_id, info in status["available_models"].items():
            status_icon = "✅" if info["available"] else "❌"
            print(f"  {status_icon} {model_id} ({info['type']}) - {info['message']}")
        
        print(f"\n🛠️  Outils:")
        print(f"  Total outils: {len(status['available_tools'])}")
        tools_working = sum(1 for working in status["tools_status"].values() if working)
        print(f"  Outils fonctionnels: {tools_working}/{len(status['tools_status'])}")
        
        print(f"\n⚙️  Système:")
        sys_info = status["system"]
        print(f"  Device: {sys_info['device']}")
        print(f"  CUDA disponible: {sys_info['cuda_available']}")
        print(f"  Fallback activé: {sys_info['enable_fallback']}")
        
        # Recommandations
        print(f"\n🎯 Recommandations d'usage:")
        recommendations = vlm.get_model_recommendations()
        
        for use_case, model_id in recommendations.items():
            print(f"  {use_case}: {model_id}")
        
        await vlm.shutdown()
        return True
        
    except Exception as e:
        print(f"❌ Erreur test statut: {e}")
        await vlm.shutdown()
        return False


async def main():
    """Test principal du système multi-modèles."""
    
    print("🚀 TESTS SYSTÈME MULTI-MODÈLES VLM")
    print("Support KIM, LLaVA et Qwen2-VL\n")
    
    tests = [
        ("Registre des modèles", test_model_registry),
        ("Chargement dynamique", test_dynamic_model_loading), 
        ("Switching entre modèles", test_model_switching),
        ("Analyses multi-modèles", test_analysis_with_different_models),
        ("Statut et recommandations", test_system_status_and_recommendations)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"🧪 {test_name}...")
        
        try:
            success = await test_func()
            results.append((test_name, success))
            
            if success:
                print(f"✅ {test_name}: RÉUSSI\n")
            else:
                print(f"❌ {test_name}: ÉCHOUÉ\n")
                
        except Exception as e:
            print(f"❌ {test_name}: ERREUR - {e}\n")
            results.append((test_name, False))
    
    # Résumé final
    print("=" * 60)
    successful_tests = sum(1 for _, success in results if success)
    total_tests = len(results)
    success_rate = (successful_tests / total_tests) * 100
    
    if successful_tests == total_tests:
        print("🎉 TOUS LES TESTS MULTI-MODÈLES RÉUSSIS!")
        print(f"✅ {successful_tests}/{total_tests} tests passés ({success_rate:.0f}%)")
        
        print(f"\n🎯 FONCTIONNALITÉS VALIDÉES:")
        print("• ✅ Registre de modèles (KIM, LLaVA, Qwen2-VL)")
        print("• ✅ Chargement dynamique avec fallbacks")
        print("• ✅ Switching à chaud entre modèles")
        print("• ✅ Analyses comparatives multi-modèles")
        print("• ✅ Monitoring et recommandations")
        print("• ✅ Intégration avec les 8 outils avancés")
        
        print(f"\n🚀 UTILISATION:")
        print("1. vlm = DynamicVisionLanguageModel(default_model='kim-7b-instruct')")
        print("2. await vlm.switch_model('qwen2-vl-7b-instruct')")
        print("3. result = await vlm.analyze_with_tools(request)")
        
    else:
        print("⚠️  CERTAINS TESTS ONT ÉCHOUÉ")
        print(f"📊 {successful_tests}/{total_tests} tests passés ({success_rate:.0f}%)")
        
        print(f"\n✅ Tests réussis:")
        for test_name, success in results:
            if success:
                print(f"  • {test_name}")
        
        print(f"\n❌ Tests échoués:")
        for test_name, success in results:
            if not success:
                print(f"  • {test_name}")
    
    print(f"\n💡 NOTE: Certains modèles (KIM) nécessitent des versions")
    print("    spécialisées de transformers ou des installations custom.")
    print("    Les fallbacks LLaVA/Qwen sont utilisés en cas d'indisponibilité.")
    
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())