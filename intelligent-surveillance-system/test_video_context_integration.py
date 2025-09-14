#!/usr/bin/env python3
"""
🧪 Test d'intégration du contexte vidéo - Vérification complète
==============================================================

Script de test pour vérifier que l'intégration des descriptions vidéo
fonctionne correctement avec le système VLM.
"""

import sys
import os
import asyncio
from datetime import datetime
from pathlib import Path

# Ajout du chemin pour les imports
sys.path.append(str(Path(__file__).parent / "dashboard"))
sys.path.append(str(Path(__file__).parent / "src"))

# Test des imports
print("🔍 Test des imports...")
try:
    from dashboard.video_context_integration import (
        VideoContextMetadata,
        VideoContextPromptBuilder,
        create_video_metadata_from_form,
        get_video_context_integration
    )
    print("✅ Import video_context_integration: OK")
except ImportError as e:
    print(f"❌ Erreur import video_context_integration: {e}")
    sys.exit(1)

try:
    from src.core.vlm.prompt_builder import PromptBuilder
    print("✅ Import PromptBuilder: OK")
except ImportError as e:
    print(f"❌ Erreur import PromptBuilder: {e}")

async def test_video_context_integration():
    """Test complet de l'intégration contexte vidéo."""
    print("\n🧪 Test d'intégration contexte vidéo")
    print("=" * 50)
    
    # 1. Test création métadonnées depuis formulaire
    print("\n1️⃣ Test création VideoContextMetadata depuis formulaire...")
    
    form_data = {
        'title': 'Surveillance Magasin Centre-Ville',
        'location_type': 'Magasin/Commerce',
        'time_context': 'Heures affluence',
        'expected_activities': ['shopping', 'browsing', 'queuing', 'payment'],
        'suspicious_focus': ['Vol à l\'étalage', 'Comportements agressifs'],
        'camera_angle': 'Vue plongeante caisse',
        'detailed_description': 'Zone caisse principale avec forte affluence client. Surveillance renforcée pour détection vol et comportements suspects.',
        'analysis_priority': 'Élevé',
        'frame_sampling': 'Dense'
    }
    
    video_metadata = create_video_metadata_from_form(form_data)
    print(f"✅ VideoContextMetadata créé: {video_metadata.title}")
    print(f"   - Type: {video_metadata.location_type}")
    print(f"   - Focus: {', '.join(video_metadata.suspicious_focus)}")
    
    # 2. Test construction prompt avec contexte
    print("\n2️⃣ Test construction prompt avec contexte vidéo...")
    
    try:
        prompt_builder = VideoContextPromptBuilder()
        
        base_prompt = "Analyse cette image de surveillance."
        enhanced_prompt = prompt_builder.build_context_enhanced_prompt(
            base_prompt, 
            video_metadata
        )
        
        print("✅ Prompt enrichi avec contexte vidéo")
        print(f"   - Longueur prompt de base: {len(base_prompt)} chars")
        print(f"   - Longueur prompt enrichi: {len(enhanced_prompt)} chars")
        print(f"   - Enrichissement: +{len(enhanced_prompt) - len(base_prompt)} chars")
        
        # Vérifications du contenu
        assert video_metadata.title in enhanced_prompt, "Titre manquant dans prompt"
        assert video_metadata.location_type in enhanced_prompt, "Type lieu manquant"
        assert all(focus in enhanced_prompt for focus in video_metadata.suspicious_focus), "Focus surveillance manquant"
        
        print("✅ Tous les éléments contexte présents dans le prompt")
        
    except Exception as e:
        print(f"❌ Erreur construction prompt: {e}")
        return False
    
    # 3. Test intégration chat contexte
    print("\n3️⃣ Test intégration chat contextualisé...")
    
    try:
        context_integration = get_video_context_integration()
        
        base_context = {
            'video_analysis_mode': True,
            'timestamp': datetime.now()
        }
        
        enhanced_context = context_integration.enhance_chat_context(
            base_context,
            video_metadata
        )
        
        print("✅ Contexte chat enrichi")
        print(f"   - Contexte de base: {len(base_context)} clés")
        print(f"   - Contexte enrichi: {len(enhanced_context)} clés")
        
        # Vérifications
        assert 'video_context' in enhanced_context, "Section video_context manquante"
        assert 'user_intent_context' in enhanced_context, "Section user_intent_context manquante"
        
        video_ctx = enhanced_context['video_context']
        assert video_ctx['context_enhanced'] == True, "Flag context_enhanced manquant"
        assert 'metadata' in video_ctx, "Métadonnées manquantes"
        assert 'analysis_objectives' in video_ctx, "Objectifs analyse manquants"
        
        print("✅ Structure contexte chat validée")
        
    except Exception as e:
        print(f"❌ Erreur intégration chat: {e}")
        return False
    
    # 4. Test génération questions contextuelles
    print("\n4️⃣ Test génération questions contextuelles...")
    
    try:
        questions = prompt_builder.generate_contextual_questions(video_metadata)
        
        print(f"✅ {len(questions)} questions générées:")
        for i, question in enumerate(questions, 1):
            print(f"   {i}. {question}")
        
        # Vérifications
        assert len(questions) > 0, "Aucune question générée"
        assert any('Magasin' in q for q in questions), "Questions spécifiques au magasin manquantes"
        assert any('vol' in q.lower() for q in questions), "Questions vol manquantes"
        
        print("✅ Questions contextuelles validées")
        
    except Exception as e:
        print(f"❌ Erreur génération questions: {e}")
        return False
    
    # 5. Test avec PromptBuilder core
    print("\n5️⃣ Test intégration avec PromptBuilder core...")
    
    try:
        core_prompt_builder = PromptBuilder()
        
        context = {
            'location': 'Magasin Centre-Ville',
            'timestamp': datetime.now().isoformat(),
            'previous_detections': []
        }
        
        available_tools = ['sam2_segmentator', 'dino_features', 'pose_estimator']
        
        # Test avec contexte vidéo
        prompt_with_context = core_prompt_builder.build_surveillance_prompt(
            context,
            available_tools,
            None,
            video_metadata.to_dict()
        )
        
        # Test sans contexte vidéo
        prompt_without_context = core_prompt_builder.build_surveillance_prompt(
            context,
            available_tools,
            None,
            None
        )
        
        print("✅ Integration PromptBuilder core")
        print(f"   - Prompt sans contexte: {len(prompt_without_context)} chars")
        print(f"   - Prompt avec contexte: {len(prompt_with_context)} chars")
        print(f"   - Enrichissement: +{len(prompt_with_context) - len(prompt_without_context)} chars")
        
        # Vérifications
        assert len(prompt_with_context) > len(prompt_without_context), "Prompt pas enrichi"
        assert "CONTEXTE VIDÉO SPÉCIFIQUE" in prompt_with_context, "Section contexte manquante"
        assert video_metadata.title in prompt_with_context, "Titre vidéo manquant"
        
        print("✅ Prompt core enrichi avec succès")
        
    except Exception as e:
        print(f"❌ Erreur intégration PromptBuilder core: {e}")
        return False
    
    print("\n🎉 TOUS LES TESTS PASSÉS AVEC SUCCÈS! 🎉")
    print("=" * 50)
    print("✅ L'intégration du contexte vidéo fonctionne correctement:")
    print("   - Création métadonnées depuis formulaire ✅")
    print("   - Construction prompt enrichi ✅")
    print("   - Intégration chat contextualisé ✅") 
    print("   - Génération questions adaptées ✅")
    print("   - Integration PromptBuilder core ✅")
    
    return True

def test_example_scenarios():
    """Test de scénarios d'usage réels."""
    print("\n🎯 Test scénarios d'usage réels")
    print("=" * 40)
    
    scenarios = [
        {
            'name': 'Surveillance Supermarché',
            'data': {
                'title': 'Supermarché Rayon Électronique',
                'location_type': 'Magasin/Commerce',
                'time_context': 'Heures ouverture',
                'expected_activities': ['browsing', 'product_comparison', 'staff_assistance'],
                'suspicious_focus': ['Vol à l\'étalage', 'Manipulation étiquettes'],
                'detailed_description': 'Rayon électronique avec produits haute valeur. Focus sur comportements de dissimulation.'
            }
        },
        {
            'name': 'Bureau Sécurisé',
            'data': {
                'title': 'Bureau Direction - Accès Restreint',
                'location_type': 'Bureau',
                'time_context': 'Heures creuses',
                'expected_activities': ['working', 'meetings'],
                'suspicious_focus': ['Intrusion', 'Accès non autorisé'],
                'detailed_description': 'Zone à accès restreint. Toute présence non autorisée est suspecte.'
            }
        }
    ]
    
    for scenario in scenarios:
        print(f"\n📋 Scénario: {scenario['name']}")
        try:
            metadata = create_video_metadata_from_form(scenario['data'])
            
            builder = VideoContextPromptBuilder()
            enhanced_prompt = builder.build_context_enhanced_prompt(
                "Analyse cette surveillance.", metadata
            )
            
            questions = builder.generate_contextual_questions(metadata)
            
            print(f"✅ Métadonnées: {metadata.location_type} - {metadata.time_context}")
            print(f"✅ Prompt enrichi: +{len(enhanced_prompt) - 30} chars")
            print(f"✅ Questions générées: {len(questions)}")
            
        except Exception as e:
            print(f"❌ Erreur scénario {scenario['name']}: {e}")

if __name__ == "__main__":
    print("🚀 Lancement des tests d'intégration contexte vidéo")
    print("=" * 60)
    
    # Test principal
    success = asyncio.run(test_video_context_integration())
    
    if success:
        # Tests scénarios
        test_example_scenarios()
        
        print("\n🎊 INTÉGRATION CONTEXTE VIDÉO FONCTIONNELLE! 🎊")
        print("\n📝 Utilisation recommandée:")
        print("1. Remplir le formulaire contexte dans dashboard")
        print("2. Le système enrichira automatiquement les prompts VLM")
        print("3. Le chatbot utilisera le contexte pour répondre")
        print("4. Les analyses seront plus précises et contextualisées")
        
    else:
        print("\n💥 ÉCHEC DES TESTS - Vérifier la configuration")
        sys.exit(1)