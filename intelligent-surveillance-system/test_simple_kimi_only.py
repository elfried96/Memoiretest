#!/usr/bin/env python3
"""
🎯 Test Simple - Kimi-VL Uniquement
===================================

Test minimal pour valider que Kimi-VL fonctionne.
Sans outils avancés, sans fallback, juste le VLM de base.
"""

import asyncio
import sys
from pathlib import Path
from PIL import Image
import numpy as np
import os

# Configuration pour éviter les problèmes d'espace
os.environ['TRANSFORMERS_CACHE'] = '/dev/shm/transformers_cache'
os.environ['HF_HOME'] = '/dev/shm/huggingface'

# Path
sys.path.append(str(Path(__file__).parent / "src"))

from src.core.vlm.dynamic_model import DynamicVisionLanguageModel
from src.core.types import AnalysisRequest


async def test_kimi_simple():
    """Test simple de Kimi-VL sans fioritures."""
    
    print("🚀 Test Simple Kimi-VL")
    print("=" * 40)
    
    # Créer image de test simple
    print("📸 Création image test...")
    img_array = np.ones((224, 224, 3), dtype=np.uint8) * [100, 150, 200]
    # Ajouter quelques formes
    img_array[50:100, 50:100] = [255, 100, 100]  # Rectangle rouge
    img_array[150:200, 150:200] = [100, 255, 100]  # Rectangle vert
    
    test_image = Image.fromarray(img_array)
    
    print("🧠 Initialisation VLM...")
    
    # Configuration minimale
    vlm = DynamicVisionLanguageModel(
        default_model="kimi-vl-a3b-thinking",
        enable_fallback=False  # Pas de fallback
    )
    
    print("⏳ Chargement Kimi-VL (peut prendre 10+ minutes)...")
    
    try:
        # Charger SEULEMENT Kimi-VL
        success = await vlm.switch_model("kimi-vl-a3b-thinking")
        
        if not success:
            print("❌ Échec du chargement")
            return
        
        print("✅ Kimi-VL chargé !")
        print(f"📊 Modèle: {vlm.current_model_id}")
        
        # Test d'analyse simple
        print("🔍 Test analyse simple...")
        
        prompt = "Décrivez cette image en une phrase simple."
        
        # Requête minimale (SANS outils avancés)
        request = AnalysisRequest(
            image=test_image,
            prompt=prompt,
            enable_advanced_tools=False,  # IMPORTANT: Désactiver outils
            max_tokens=100
        )
        
        print("⏱️  Analyse en cours...")
        response = await vlm.analyze_image(request)
        
        print("\n✅ RÉSULTAT:")
        print(f"📝 Description: {response.description}")
        print(f"🎯 Suspicion: {response.suspicion_level}")
        print(f"📊 Confiance: {response.confidence_score:.2f}")
        print(f"🔧 Action: {response.recommended_action}")
        
        print("\n🎉 Test réussi ! Kimi-VL fonctionne.")
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("⚠️  ATTENTION: Ce test peut prendre 15+ minutes")
    print("💾 Assurez-vous d'avoir 35GB+ libres dans /dev/shm")
    
    input("Appuyez sur Entrée pour continuer ou Ctrl+C pour annuler...")
    
    asyncio.run(test_kimi_simple())