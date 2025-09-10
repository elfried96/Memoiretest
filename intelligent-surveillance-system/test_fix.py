#!/usr/bin/env python3
"""Test rapide pour vérifier les corrections."""

import sys
sys.path.append('/home/elfried-kinzoun/PycharmProjects/intelligent-surveillance-system')

def test_temporal_transformer():
    """Test de la méthode analyze_sequence."""
    try:
        from src.advanced_tools.temporal_transformer import TemporalTransformer
        
        transformer = TemporalTransformer()
        
        # Tester la méthode analyze_sequence
        result = transformer.analyze_sequence([0.3, 0.5, 0.7, 0.4, 0.6], "behavior")
        print("✅ TemporalTransformer.analyze_sequence fonctionne!")
        print(f"Résultat: {result}")
        
        return True
    except Exception as e:
        print(f"❌ Erreur TemporalTransformer: {e}")
        return False

def test_pydantic_serialization():
    """Test de la sérialisation Pydantic."""
    try:
        from src.core.types import AnalysisResponse, SuspicionLevel, ActionType
        from datetime import datetime
        
        # Créer un objet AnalysisResponse
        response = AnalysisResponse(
            suspicion_level=SuspicionLevel.LOW,
            action_type=ActionType.NORMAL_SHOPPING,
            confidence=0.8,
            description="Test",
            reasoning="Test reasoning",
            tools_used=["test"],
            recommendations=["test rec"],
            timestamp=datetime.now()
        )
        
        # Tester model_dump()
        result = response.model_dump()
        print("✅ Pydantic model_dump() fonctionne!")
        print(f"Keys: {list(result.keys())}")
        
        return True
    except Exception as e:
        print(f"❌ Erreur Pydantic: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Test des corrections...")
    print()
    
    test1 = test_temporal_transformer()
    print()
    test2 = test_pydantic_serialization()
    print()
    
    if test1 and test2:
        print("🎉 Tous les tests passent! Les corrections sont fonctionnelles.")
    else:
        print("⚠️ Certains tests échouent. Vérifiez les erreurs ci-dessus.")