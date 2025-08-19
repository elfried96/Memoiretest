#!/usr/bin/env python3
"""Test immédiat pour vérifier que le système de base fonctionne."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

def test_imports():
    """Tester que tous les imports fonctionnent."""
    print("🔍 Test des imports...")
    
    try:
        from src.core.vlm.model_registry import VLMModelRegistry
        print("✅ VLMModelRegistry importé")
        
        from src.core.vlm.dynamic_model import DynamicVisionLanguageModel  
        print("✅ DynamicVisionLanguageModel importé")
        
        from src.core.types import AnalysisRequest
        print("✅ AnalysisRequest importé")
        
        # Test du registre
        registry = VLMModelRegistry()
        models = registry.list_available_models()
        print(f"✅ {len(models)} modèles enregistrés")
        
        for model_id in models.keys():
            available, msg = registry.validate_model_availability(model_id)
            status = "✅" if available else "❌"
            print(f"  {status} {model_id}: {msg}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Test Immédiat - Validation Système")
    print("=" * 40)
    
    if test_imports():
        print("\n✅ Système fonctionnel de base !")
        print("📝 Vous pouvez procéder aux tests VLM")
    else:
        print("\n❌ Problème de configuration de base")
        print("🔧 Vérifiez les imports et dépendances")