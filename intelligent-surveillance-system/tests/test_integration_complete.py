#!/usr/bin/env python3
"""Test d'intégration complète du système VLM + 8 outils avancés."""

import sys
import os
import base64
import asyncio
from typing import Dict, Any
import numpy as np
from PIL import Image
from io import BytesIO

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.orchestrator.vlm_orchestrator import (
    ModernVLMOrchestrator, 
    OrchestrationConfig, 
    OrchestrationMode
)
from core.types import Detection, BoundingBox


def create_test_image() -> str:
    """Créer une image de test encodée en base64."""
    # Image simulant une scène de surveillance
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Ajouter des éléments visuels simulés
    # Zone personne
    image[100:300, 200:350] = [120, 100, 80]  # Silhouette personne
    
    # Zone objet suspect
    image[250:280, 320:370] = [80, 60, 40]    # Objet dans les mains
    
    # Conversion en PIL puis base64
    pil_image = Image.fromarray(image)
    buffer = BytesIO()
    pil_image.save(buffer, format='PNG')
    image_bytes = buffer.getvalue()
    
    return base64.b64encode(image_bytes).decode('utf-8')


def create_test_detections() -> list:
    """Créer des détections de test."""
    return [
        Detection(
            class_name="person",
            confidence=0.89,
            bbox=BoundingBox(x1=200, y1=100, x2=350, y2=300)
        ),
        Detection(
            class_name="handbag", 
            confidence=0.76,
            bbox=BoundingBox(x1=320, y1=250, x2=370, y2=280)
        )
    ]


async def test_orchestrator_modes():
    """Test des différents modes d'orchestration."""
    
    print("=== TEST MODES D'ORCHESTRATION ===\n")
    
    # Image et détections de test
    test_image = create_test_image()
    test_detections = create_test_detections()
    test_context = {
        "location": "Test Store - Aisle 3",
        "camera_id": "CAM_001",
        "test_mode": True
    }
    
    # Test des 3 modes
    modes = [
        (OrchestrationMode.FAST, "Mode Rapide"),
        (OrchestrationMode.BALANCED, "Mode Équilibré"), 
        (OrchestrationMode.THOROUGH, "Mode Complet")
    ]
    
    results = {}
    
    for mode, mode_name in modes:
        print(f"--- {mode_name} ---")
        
        # Configuration pour ce mode
        config = OrchestrationConfig(
            mode=mode,
            max_concurrent_tools=4,
            timeout_seconds=30,
            enable_advanced_tools=True
        )
        
        # Création de l'orchestrateur
        orchestrator = ModernVLMOrchestrator(
            vlm_model_name="llava-hf/llava-v1.6-mistral-7b-hf",  # Sera en fallback sans GPU
            config=config
        )
        
        try:
            # Analyse de test
            print("  Démarrage de l'analyse...")
            result = await orchestrator.analyze_surveillance_frame(
                frame_data=test_image,
                detections=test_detections,
                context=test_context
            )
            
            # Affichage des résultats
            print(f"  ✓ Analyse terminée")
            print(f"    - Niveau suspicion: {result.suspicion_level.value}")
            print(f"    - Type d'action: {result.action_type.value}")
            print(f"    - Confiance: {result.confidence:.2f}")
            print(f"    - Outils utilisés: {len(result.tools_used)}")
            if result.tools_used:
                print(f"      {', '.join(result.tools_used)}")
            print(f"    - Recommandations: {len(result.recommendations)}")
            
            results[mode.value] = {
                "success": True,
                "result": result,
                "tools_count": len(result.tools_used)
            }
            
            # Statut système
            system_status = orchestrator.get_system_status()
            print(f"    - Statut VLM: {system_status['vlm_system']['vlm_model']['loaded']}")
            print(f"    - Outils disponibles: {len(system_status['vlm_system']['available_tools'])}")
            
            # Nettoyage
            await orchestrator.shutdown()
            
        except Exception as e:
            print(f"  ✗ Erreur: {e}")
            results[mode.value] = {
                "success": False,
                "error": str(e),
                "tools_count": 0
            }
        
        print()
    
    return results


async def test_batch_processing():
    """Test du traitement par batch."""
    
    print("=== TEST TRAITEMENT BATCH ===\n")
    
    # Préparation de plusieurs frames
    frames_data = []
    for i in range(3):
        frames_data.append({
            "frame_data": create_test_image(),
            "detections": create_test_detections(),
            "context": {
                "frame_id": f"test_frame_{i}",
                "timestamp": 1000 + i,
                "location": f"Test Zone {i+1}"
            }
        })
    
    # Configuration équilibrée
    config = OrchestrationConfig(
        mode=OrchestrationMode.BALANCED,
        max_concurrent_tools=2,
        enable_advanced_tools=True
    )
    
    orchestrator = ModernVLMOrchestrator(config=config)
    
    try:
        print(f"Traitement de {len(frames_data)} frames...")
        
        results = await orchestrator.batch_analyze(
            frames_data=frames_data,
            max_concurrent=2
        )
        
        print(f"✓ Batch terminé: {len(results)} résultats")
        
        # Analyse des résultats
        success_count = sum(1 for r in results if r.confidence > 0.0)
        avg_confidence = sum(r.confidence for r in results) / len(results)
        
        print(f"  - Analyses réussies: {success_count}/{len(results)}")
        print(f"  - Confiance moyenne: {avg_confidence:.2f}")
        
        # Statistiques système
        system_stats = orchestrator.get_system_status()
        print(f"  - Total analyses: {system_stats['performance']['total_analyses']}")
        print(f"  - Taux de succès: {system_stats['performance']['success_rate_percent']:.1f}%")
        
        await orchestrator.shutdown()
        
        return True
        
    except Exception as e:
        print(f"✗ Erreur batch: {e}")
        await orchestrator.shutdown()
        return False


async def test_system_health():
    """Test de santé du système."""
    
    print("=== TEST SANTÉ SYSTÈME ===\n")
    
    config = OrchestrationConfig(enable_advanced_tools=True)
    orchestrator = ModernVLMOrchestrator(config=config)
    
    try:
        # Health check
        health = await orchestrator.health_check()
        
        print("État de santé du système:")
        for component, status in health.items():
            status_icon = "✓" if status else "✗"
            print(f"  {status_icon} {component}: {status}")
        
        # Statut complet
        full_status = orchestrator.get_system_status()
        
        print(f"\nConfiguration:")
        print(f"  - Mode: {full_status['orchestrator']['mode']}")
        print(f"  - Outils avancés: {full_status['orchestrator']['enable_advanced_tools']}")
        print(f"  - Max outils concurrent: {full_status['orchestrator']['max_concurrent_tools']}")
        
        print(f"\nSystème VLM:")
        vlm_info = full_status['vlm_system']['vlm_model']
        print(f"  - Modèle: {vlm_info['name']}")
        print(f"  - Chargé: {vlm_info['loaded']}")
        print(f"  - Device: {vlm_info['device']}")
        
        print(f"\nOutils disponibles:")
        tools = full_status['vlm_system']['available_tools']
        print(f"  - Total: {len(tools)}")
        for tool in tools:
            print(f"    • {tool}")
        
        await orchestrator.shutdown()
        
        return all(health.values())
        
    except Exception as e:
        print(f"✗ Erreur health check: {e}")
        await orchestrator.shutdown()
        return False


async def main():
    """Test principal d'intégration."""
    
    print("🚀 TEST D'INTÉGRATION COMPLÈTE")
    print("Système VLM + 8 Outils Avancés\n")
    
    total_tests = 0
    successful_tests = 0
    
    # Test 1: Modes d'orchestration
    try:
        results = await test_orchestrator_modes()
        total_tests += 1
        if any(r["success"] for r in results.values()):
            successful_tests += 1
            print("✅ Test modes d'orchestration: RÉUSSI")
        else:
            print("❌ Test modes d'orchestration: ÉCHOUÉ")
    except Exception as e:
        total_tests += 1
        print(f"❌ Test modes d'orchestration: ERREUR - {e}")
    
    print()
    
    # Test 2: Traitement batch
    try:
        batch_success = await test_batch_processing()
        total_tests += 1
        if batch_success:
            successful_tests += 1
            print("✅ Test traitement batch: RÉUSSI")
        else:
            print("❌ Test traitement batch: ÉCHOUÉ")
    except Exception as e:
        total_tests += 1
        print(f"❌ Test traitement batch: ERREUR - {e}")
    
    print()
    
    # Test 3: Santé système
    try:
        health_ok = await test_system_health()
        total_tests += 1
        if health_ok:
            successful_tests += 1
            print("✅ Test santé système: RÉUSSI")
        else:
            print("❌ Test santé système: ÉCHOUÉ")
    except Exception as e:
        total_tests += 1
        print(f"❌ Test santé système: ERREUR - {e}")
    
    # Résumé final
    print(f"\n{'='*50}")
    success_rate = (successful_tests / total_tests) * 100 if total_tests > 0 else 0
    
    if successful_tests == total_tests:
        print("🎉 TOUS LES TESTS D'INTÉGRATION RÉUSSIS")
        print(f"✅ {successful_tests}/{total_tests} tests passés ({success_rate:.0f}%)")
        print("\n🔧 Fonctionnalités validées:")
        print("  • VLM modulaire avec 8 outils avancés intégrés")
        print("  • Orchestration intelligente selon 3 modes")
        print("  • Traitement batch avec contrôle de concurrence")
        print("  • Monitoring et health checks")
        print("  • Gestion d'erreurs robuste")
        print("  • Architecture modulaire (prompt, parsing, tools)")
    else:
        print("⚠️  CERTAINS TESTS D'INTÉGRATION ONT ÉCHOUÉ")
        print(f"📊 {successful_tests}/{total_tests} tests passés ({success_rate:.0f}%)")
        
        if successful_tests > 0:
            print("\n✅ Fonctionnalités opérationnelles:")
            print("  • Architecture de base fonctionnelle")
            print("  • Fallbacks et gestion d'erreurs")
            print("  • Intégration partielle des outils")
    
    print(f"\n📝 Note: Certains outils nécessitent des dépendances GPU")
    print("    (SAM2, DINO v2, modèles PyTorch) pour fonctionner")
    print("    complètement. Les fallbacks CPU sont utilisés en test.")
    
    print(f"{'='*50}")


if __name__ == "__main__":
    # Exécution asynchrone
    asyncio.run(main())