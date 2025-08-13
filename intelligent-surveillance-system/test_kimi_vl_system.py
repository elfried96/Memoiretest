#!/usr/bin/env python3
"""
Test complet du système Kimi-VL intégré avec switching multi-VLM.
Script de test pour valider l'intégration Kimi-VL-A3B-Thinking.
"""

import asyncio
import base64
import time
from pathlib import Path
from PIL import Image
from io import BytesIO
import json

from src.core.vlm.dynamic_model import DynamicVisionLanguageModel
from src.core.vlm.model_registry import VLMModelRegistry
from src.core.types import AnalysisRequest
from src.core.orchestrator.vlm_orchestrator import (
    ModernVLMOrchestrator, 
    OrchestrationConfig, 
    OrchestrationMode
)


def create_test_image() -> str:
    """Crée une image de test simple encodée en base64."""
    # Création d'une image de test simple
    image = Image.new('RGB', (640, 480), color='lightblue')
    
    # Ajout de formes simples pour test
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(image)
    
    # Rectangle (simule une personne)
    draw.rectangle([200, 150, 300, 400], outline='red', width=3)
    draw.text((210, 130), "Person", fill='red')
    
    # Cercle (simule un objet)
    draw.ellipse([400, 200, 500, 300], outline='blue', width=2)
    draw.text((420, 180), "Object", fill='blue')
    
    # Ligne (simule un mouvement)
    draw.line([150, 100, 450, 350], fill='green', width=2)
    draw.text((300, 220), "Movement", fill='green')
    
    # Conversion en base64
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return image_data


async def test_model_registry():
    """Test du registre des modèles."""
    print("🔍 Test du Model Registry...")
    
    registry = VLMModelRegistry()
    
    # Liste des modèles disponibles
    models = registry.list_available_models()
    print(f"📦 Modèles enregistrés: {len(models)}")
    
    for model_id, config in models.items():
        print(f"  • {model_id} ({config.model_type.value}): {config.description}")
    
    # Test de validation des modèles
    print("\n🔎 Validation des modèles:")
    for model_id in ["kimi-vl-a3b-thinking", "kimi-vl-a3b-instruct", "llava-v1.6-mistral-7b"]:
        is_available, message = registry.validate_model_availability(model_id)
        status = "✅" if is_available else "❌"
        print(f"  {status} {model_id}: {message}")
    
    # Recommandations
    print("\n💡 Recommandations:")
    recommendations = registry.get_model_recommendations()
    for use_case, model in recommendations.items():
        print(f"  • {use_case}: {model}")
    
    return registry


async def test_dynamic_vlm():
    """Test du VLM dynamique avec Kimi-VL."""
    print("\n🧠 Test du VLM Dynamique...")
    
    # Initialisation avec Kimi-VL-A3B-Thinking comme défaut
    vlm = DynamicVisionLanguageModel(
        default_model="kimi-vl-a3b-thinking",
        enable_fallback=True
    )
    
    # Tentative de chargement
    print("⏳ Chargement Kimi-VL-A3B-Thinking...")
    success = await vlm.load_model()
    
    if success:
        print("✅ Kimi-VL chargé avec succès!")
        
        # Test d'analyse
        test_image = create_test_image()
        request = AnalysisRequest(
            frame_data=test_image,
            context={
                "location": "Test Store", 
                "camera": "CAM_TEST",
                "test_mode": True
            },
            tools_available=["dino_features", "pose_estimator", "multimodal_fusion"]
        )
        
        print("🔍 Test d'analyse avec Kimi-VL...")
        start_time = time.time()
        result = await vlm.analyze_with_tools(request, use_advanced_tools=True)
        processing_time = time.time() - start_time
        
        print(f"📊 Résultat ({processing_time:.2f}s):")
        print(f"  • Suspicion: {result.suspicion_level.value}")
        print(f"  • Action: {result.action_type.value}")
        print(f"  • Confiance: {result.confidence:.2f}")
        print(f"  • Description: {result.description}")
        print(f"  • Outils utilisés: {result.tools_used}")
        
    else:
        print("❌ Échec chargement Kimi-VL, test des fallbacks...")
        
        # Test fallback vers LLaVA
        print("⏳ Test fallback LLaVA...")
        success_llava = await vlm.switch_model("llava-v1.6-mistral-7b")
        
        if success_llava:
            print("✅ Fallback LLaVA réussi!")
        else:
            print("❌ Tous les modèles indisponibles")
    
    # Statut système
    print("\n📈 Statut du système:")
    status = vlm.get_system_status()
    print(f"  • Modèle actuel: {status['current_model']['model_id']}")
    print(f"  • Type: {status['current_model']['model_type']}")
    print(f"  • Device: {status['current_model']['device']}")
    print(f"  • CUDA disponible: {status['system']['cuda_available']}")
    
    return vlm


async def test_model_switching():
    """Test du switching entre modèles."""
    print("\n🔄 Test du Model Switching...")
    
    vlm = DynamicVisionLanguageModel(enable_fallback=True)
    
    # Test de switching entre modèles
    models_to_test = [
        "kimi-vl-a3b-thinking",   # Modèle principal
        "kimi-vl-a3b-instruct",   # Variant instruct
        "llava-v1.6-mistral-7b",   # Fallback stable
        "qwen2-vl-7b-instruct"     # Alternative
    ]
    
    test_image = create_test_image()
    request = AnalysisRequest(
        frame_data=test_image,
        context={"location": "Switching Test"},
        tools_available=["dino_features"]
    )
    
    for model_id in models_to_test:
        print(f"\n📱 Test switching vers {model_id}...")
        
        success = await vlm.switch_model(model_id)
        
        if success:
            print(f"✅ Switch réussi vers {model_id}")
            
            # Test rapide d'analyse
            try:
                start_time = time.time()
                result = await vlm.analyze_with_tools(request, use_advanced_tools=False)
                processing_time = time.time() - start_time
                
                print(f"  📊 Analyse ({processing_time:.2f}s): {result.suspicion_level.value}")
                
            except Exception as e:
                print(f"  ❌ Erreur analyse: {e}")
        else:
            print(f"❌ Switch échoué vers {model_id}")
    
    return vlm


async def test_orchestrator():
    """Test de l'orchestrateur avec Kimi-VL."""
    print("\n🎮 Test de l'Orchestrateur...")
    
    # Configuration pour test avec Kimi-VL
    config = OrchestrationConfig(
        mode=OrchestrationMode.BALANCED,
        enable_advanced_tools=True,
        max_concurrent_tools=4,
        confidence_threshold=0.6
    )
    
    # Initialisation avec Kimi-VL par défaut
    orchestrator = ModernVLMOrchestrator(
        vlm_model_name="kimi-vl-a3b-thinking",
        config=config
    )
    
    # Test d'analyse complète
    test_image = create_test_image()
    
    print("🔍 Analyse orchestrée...")
    try:
        start_time = time.time()
        result = await orchestrator.analyze_surveillance_frame(
            frame_data=test_image,
            detections=[],  # Pas de détections YOLO pour ce test
            context={
                "location": "Orchestrator Test Zone",
                "camera": "CAM_ORCH_01",
                "security_level": "high"
            }
        )
        processing_time = time.time() - start_time
        
        print(f"📊 Résultat orchestré ({processing_time:.2f}s):")
        print(f"  • Suspicion: {result.suspicion_level.value}")
        print(f"  • Action: {result.action_type.value}")
        print(f"  • Confiance: {result.confidence:.2f}")
        print(f"  • Description: {result.description}")
        print(f"  • Outils: {result.tools_used}")
        print(f"  • Recommandations: {result.recommendations}")
        
    except Exception as e:
        print(f"❌ Erreur orchestration: {e}")
    
    # Statut complet
    print("\n📈 Statut orchestrateur:")
    status = orchestrator.get_system_status()
    
    print(f"  • Mode: {status['orchestrator']['mode']}")
    print(f"  • Outils avancés: {status['orchestrator']['enable_advanced_tools']}")
    print(f"  • Analyses totales: {status['performance']['total_analyses']}")
    print(f"  • Taux succès: {status['performance']['success_rate_percent']:.1f}%")
    
    # Health check
    health = await orchestrator.health_check()
    print(f"  • Health check: {health}")
    
    return orchestrator


async def test_batch_processing():
    """Test du traitement par batch."""
    print("\n📦 Test Batch Processing...")
    
    vlm = DynamicVisionLanguageModel(
        default_model="kimi-vl-a3b-thinking",
        enable_fallback=True
    )
    
    success = await vlm.load_model()
    if not success:
        print("❌ Modèle non chargé, skip batch test")
        return
    
    # Création de plusieurs images de test
    test_images = []
    for i in range(3):
        image_data = create_test_image()
        test_images.append(AnalysisRequest(
            frame_data=image_data,
            context={
                "batch_id": f"batch_frame_{i}",
                "location": f"Zone_{i+1}"
            },
            tools_available=["dino_features", "pose_estimator"]
        ))
    
    print(f"🔄 Traitement batch de {len(test_images)} images...")
    start_time = time.time()
    
    # Traitement séquentiel pour comparaison
    results = []
    for request in test_images:
        result = await vlm.analyze_with_tools(request, use_advanced_tools=True)
        results.append(result)
    
    processing_time = time.time() - start_time
    
    print(f"📊 Batch terminé ({processing_time:.2f}s):")
    for i, result in enumerate(results):
        print(f"  Frame {i+1}: {result.suspicion_level.value} (conf: {result.confidence:.2f})")
    
    average_time = processing_time / len(results)
    print(f"  • Temps moyen/frame: {average_time:.2f}s")


async def main():
    """Fonction principale de test."""
    print("🚀 Test Complet du Système Kimi-VL Multi-VLM")
    print("=" * 60)
    
    try:
        # Tests individuels
        await test_model_registry()
        await test_dynamic_vlm()
        await test_model_switching()
        await test_orchestrator()
        await test_batch_processing()
        
        print("\n" + "=" * 60)
        print("✅ Tests terminés avec succès!")
        
        print("\n💡 Recommandations d'utilisation:")
        print("1. 🎯 Utilisez kimi-vl-a3b-thinking pour surveillance principale")
        print("2. 🔄 LLaVA comme fallback stable") 
        print("3. 🧠 Qwen2-VL pour analyses complexes")
        print("4. ⚙️ Mode BALANCED pour production")
        print("5. 🛡️ Activez les fallbacks automatiques")
        
    except Exception as e:
        print(f"\n❌ Erreur durant les tests: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())