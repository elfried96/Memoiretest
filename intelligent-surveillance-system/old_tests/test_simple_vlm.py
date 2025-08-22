#!/usr/bin/env python3
"""Test simple du système VLM corrigé."""

import os
import sys
import asyncio
from pathlib import Path

# Configuration de l'environnement
os.environ['TRANSFORMERS_CACHE'] = '/home/elfried-kinzoun/.cache/transformers'
os.environ['HF_HOME'] = '/home/elfried-kinzoun/.cache/huggingface'

# Ajout du path
sys.path.append(str(Path(__file__).parent / "src"))

from src.core.vlm.dynamic_model import DynamicVisionLanguageModel
from src.core.vlm.model_registry import VLMModelRegistry


async def test_model_loading():
    """Test de chargement des modèles."""
    print("🚀 Test Simple du Système VLM Corrigé")
    print("=" * 50)
    
    # Test du registre
    registry = VLMModelRegistry()
    print(f"📦 Modèles disponibles: {len(registry.list_available_models())}")
    
    for model_id, config in registry.list_available_models().items():
        available, msg = registry.validate_model_availability(model_id)
        status = "✅" if available else "❌"
        print(f"  {status} {model_id}: {msg}")
    
    print("\n🧠 Test du VLM Dynamique...")
    
    # Test avec Qwen2-VL (plus stable)
    vlm = DynamicVisionLanguageModel(
        default_model="qwen2-vl-7b-instruct",  # Commencer par Qwen
        enable_fallback=False  # Pas de fallback pour simplifier
    )
    
    print("⏳ Chargement Qwen2-VL-7B...")
    try:
        success = await vlm.switch_model("qwen2-vl-7b-instruct")
        if success:
            print("✅ Qwen2-VL chargé avec succès!")
            print(f"📈 Modèle actuel: {vlm.current_model_id}")
            print(f"🔧 Device: {vlm.device}")
            print(f"💾 CUDA disponible: {vlm._cuda_available}")
        else:
            print("❌ Échec chargement Qwen2-VL")
            
            print("\n⏳ Test fallback Kimi-VL...")
            success = await vlm.switch_model("kimi-vl-a3b-thinking")
            if success:
                print("✅ Kimi-VL chargé en fallback!")
            else:
                print("❌ Tous les modèles ont échoué")
    
    except Exception as e:
        print(f"❌ Erreur: {e}")
    
    print("\n📈 Statut final:")
    print(f"  • Modèle actuel: {vlm.current_model_id}")
    print(f"  • Chargé: {vlm.is_loaded}")
    print(f"  • Device: {vlm.device}")


if __name__ == "__main__":
    asyncio.run(test_model_loading())