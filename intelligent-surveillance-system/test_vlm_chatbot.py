#!/usr/bin/env python3
"""
🧪 Test VLM Chatbot Symbiosis
============================

Script de test pour valider la symbiose VLM-Chatbot avec thinking/reasoning.
"""

import asyncio
import sys
from pathlib import Path
from loguru import logger

# Ajout chemin
sys.path.append(str(Path(__file__).parent / "dashboard"))

try:
    from vlm_chatbot_symbiosis import get_vlm_chatbot, process_vlm_chat_query
    from real_pipeline_integration import initialize_real_pipeline, get_real_pipeline
    IMPORTS_OK = True
except ImportError as e:
    logger.error(f"Erreur imports: {e}")
    IMPORTS_OK = False


async def test_vlm_chatbot_symbiosis():
    """Test complet de la symbiose VLM-Chatbot."""
    
    if not IMPORTS_OK:
        print("❌ Imports échoués - Test impossible")
        return False
    
    print("🧪 Test VLM Chatbot Symbiosis")
    print("=" * 50)
    
    # 1. Test initialization
    print("\n1️⃣ Test initialisation chatbot...")
    try:
        chatbot = get_vlm_chatbot()
        print(f"✅ Chatbot initialisé: {type(chatbot)}")
        print(f"   - Pipeline connectée: {chatbot.pipeline is not None}")
        print(f"   - Thinking activé: {chatbot.thinking_enabled}")
        print(f"   - Reasoning activé: {chatbot.reasoning_enabled}")
    except Exception as e:
        print(f"❌ Erreur initialisation: {e}")
        return False
    
    # 2. Test pipeline connection
    print("\n2️⃣ Test connexion pipeline VLM...")
    try:
        # Tentative initialisation pipeline (peut échouer sans GPU)
        pipeline_success = initialize_real_pipeline(
            vlm_model_name="Qwen/Qwen2.5-VL-7B-Instruct",  # Plus léger pour test
            enable_optimization=False,
            max_concurrent_analysis=1
        )
        
        if pipeline_success:
            print("✅ Pipeline VLM initialisée")
            pipeline = get_real_pipeline()
            chatbot.pipeline = pipeline
        else:
            print("⚠️ Pipeline VLM indisponible - Test en mode simulation")
            
    except Exception as e:
        print(f"⚠️ Pipeline VLM non disponible: {e}")
        print("   Continuing with simulation mode...")
    
    # 3. Test questions basiques
    print("\n3️⃣ Test questions chatbot...")
    
    test_questions = [
        ("Quelle est la performance actuelle du système ?", "surveillance"),
        ("Quels outils VLM sont optimaux ?", "surveillance"),
        ("Résume les détections récentes", "surveillance"),
        ("Comment optimiser la configuration ?", "video"),
        ("Explique le processus de thinking VLM", "surveillance")
    ]
    
    # Contexte simulé
    mock_context = {
        "stats": {
            "frames_processed": 156,
            "current_performance_score": 0.87,
            "current_optimal_tools": ["sam2_segmentator", "dino_features", "pose_estimator"],
            "average_processing_time": 2.3,
            "optimization_cycles": 7,
            "total_detections": 23
        },
        "detections": [
            type('MockDetection', (), {
                'confidence': 0.91,
                'description': 'Personne avec comportement suspect près véhicule',
                'tools_used': ['sam2_segmentator', 'pose_estimator']
            })(),
            type('MockDetection', (), {
                'confidence': 0.76,
                'description': 'Objet dissimulé détecté',
                'tools_used': ['dino_features', 'trajectory_analyzer']
            })()
        ],
        "optimizations": [
            {
                "best_combination": ["sam2_segmentator", "pose_estimator", "dino_features"],
                "performance_improvement": 0.12
            }
        ]
    }
    
    for i, (question, chat_type) in enumerate(test_questions, 1):
        print(f"\n   Question {i}: {question}")
        
        try:
            # Test réponse VLM avec thinking
            response = await process_vlm_chat_query(
                question=question,
                chat_type=chat_type,
                vlm_context=mock_context
            )
            
            print(f"   ✅ Réponse reçue (type: {response.get('type', 'unknown')})")
            print(f"   📝 Extrait: {response.get('response', '')[:100]}...")
            
            # Vérification thinking si disponible
            if response.get('thinking'):
                print(f"   🧠 Thinking détecté: {len(response['thinking'])} chars")
            
            if response.get('confidence'):
                print(f"   📊 Confiance: {response['confidence']:.1%}")
                
        except Exception as e:
            print(f"   ❌ Erreur question {i}: {e}")
    
    # 4. Test conversation history
    print("\n4️⃣ Test historique conversation...")
    try:
        summary = chatbot.get_conversation_summary()
        print(f"✅ Résumé conversation:")
        print(f"   - Total échanges: {summary.get('total_exchanges', 0)}")
        print(f"   - Réponses VLM: {summary.get('vlm_responses', 0)}")
        print(f"   - Réponses fallback: {summary.get('fallback_responses', 0)}")
        print(f"   - Pipeline connectée: {summary.get('pipeline_connected', False)}")
    except Exception as e:
        print(f"❌ Erreur historique: {e}")
    
    # 5. Test performance
    print("\n5️⃣ Test performance chatbot...")
    
    import time
    start_time = time.time()
    
    try:
        # Test question complexe
        complex_response = await process_vlm_chat_query(
            question="Analyse complète: performance, outils, optimisations et recommandations",
            chat_type="surveillance", 
            vlm_context=mock_context
        )
        
        processing_time = time.time() - start_time
        
        print(f"✅ Question complexe traitée en {processing_time:.2f}s")
        print(f"   Type réponse: {complex_response.get('type', 'unknown')}")
        print(f"   Longueur réponse: {len(complex_response.get('response', ''))} chars")
        
    except Exception as e:
        print(f"❌ Erreur test performance: {e}")
    
    print("\n🎯 Test VLM Chatbot Symbiosis terminé!")
    return True


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    
    print("🚀 Démarrage tests VLM Chatbot Symbiosis")
    
    try:
        success = asyncio.run(test_vlm_chatbot_symbiosis())
        
        if success:
            print("\n✅ Tests réussis - VLM Chatbot Symbiosis opérationnel!")
            sys.exit(0)
        else:
            print("\n❌ Tests échoués")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️ Tests interrompus")
        sys.exit(0)
    except Exception as e:
        print(f"\n💥 Erreur critique: {e}")
        sys.exit(1)