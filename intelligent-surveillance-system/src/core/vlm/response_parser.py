"""Module de parsing des réponses VLM."""

import json
import re
from typing import Dict, Any
from loguru import logger

from ..types import AnalysisResponse, SuspicionLevel, ActionType


class ResponseParser:
    """Parseur spécialisé pour les réponses VLM de surveillance."""
    
    def parse_vlm_response(self, response_text: str) -> AnalysisResponse:
        """Parse la réponse du VLM en AnalysisResponse structuré."""
        
        try:
            # Tentative de parsing JSON principal
            parsed_json = self._extract_json(response_text)
            if parsed_json:
                return self._json_to_analysis_response(parsed_json)
            
            # Fallback: analyse heuristique
            logger.warning(f"Pas de JSON valide, fallback heuristique. Réponse: {response_text[:200]}")
            return self._heuristic_analysis(response_text)
            
        except Exception as e:
            logger.error(f"Erreur parsing réponse VLM: {e}")
            return self._error_fallback(response_text, str(e))
    
    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extraction JSON optimisée pour Qwen2.5-VL et Kimi-VL 2025."""
        
        # Nettoyer la réponse des tokens de thinking spéciaux (Kimi-VL)
        text = re.sub(r'◁think▷.*?◁/think▷', '', text, flags=re.DOTALL)
        text = text.replace('assistant◁think▷', '').replace('◁/think▷', '')
        
        # Méthode 1: JSON dans blocs markdown
        for pattern in [r'```json\s*(\{.*?\})\s*```', r'```\s*(\{.*?\})\s*```']:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    pass
        
        # Méthode 2: Recherche du premier objet JSON complet
        json_start = text.find('{')
        if json_start != -1:
            # Recherche de l'accolade fermante correspondante
            brace_count = 0
            json_end = json_start
            
            for i, char in enumerate(text[json_start:], json_start):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_end = i + 1
                        break
            
            if brace_count == 0:
                try:
                    json_text = text[json_start:json_end]
                    return json.loads(json_text)
                except json.JSONDecodeError:
                    pass
        
        return None
    
    def _json_to_analysis_response(self, parsed_json: Dict[str, Any]) -> AnalysisResponse:
        """Conversion du JSON parsé en AnalysisResponse."""
        
        # Validation et conversion des enums
        suspicion_level = self._parse_suspicion_level(parsed_json.get("suspicion_level"))
        action_type = self._parse_action_type(parsed_json.get("action_type"))
        
        # Extraction des autres champs
        confidence = float(parsed_json.get("confidence", 0.0))
        confidence = max(0.0, min(1.0, confidence))  # Clamp entre 0 et 1
        
        description = parsed_json.get("description", "")
        reasoning = parsed_json.get("reasoning", "")
        
        tools_to_use = parsed_json.get("tools_to_use", [])
        if not isinstance(tools_to_use, list):
            tools_to_use = []
        
        recommendations = parsed_json.get("recommendations", [])
        if not isinstance(recommendations, list):
            recommendations = []
        
        return AnalysisResponse(
            suspicion_level=suspicion_level,
            action_type=action_type,
            confidence=confidence,
            description=description,
            reasoning=reasoning,  # ✅ AJOUTÉ - Processus de raisonnement
            tools_used=tools_to_use,
            recommendations=recommendations
        )
    
    def _parse_suspicion_level(self, level_str: str) -> SuspicionLevel:
        """Parse le niveau de suspicion."""
        if not level_str:
            return SuspicionLevel.LOW
        
        level_str = level_str.upper().strip()
        
        try:
            return SuspicionLevel(level_str)
        except ValueError:
            # Mapping alternatifs
            mapping = {
                "FAIBLE": SuspicionLevel.LOW,
                "BAS": SuspicionLevel.LOW,
                "MOYEN": SuspicionLevel.MEDIUM,
                "ÉLEVÉ": SuspicionLevel.HIGH,
                "HAUT": SuspicionLevel.HIGH,
                "CRITIQUE": SuspicionLevel.CRITICAL,
                "URGENT": SuspicionLevel.CRITICAL
            }
            return mapping.get(level_str, SuspicionLevel.LOW)
    
    def _parse_action_type(self, action_str: str) -> ActionType:
        """Parse le type d'action."""
        if not action_str:
            return ActionType.NORMAL_SHOPPING
        
        action_str = action_str.lower().strip()
        
        try:
            return ActionType(action_str)
        except ValueError:
            # Mapping avec mots-clés français
            if any(word in action_str for word in ["normal", "courses", "shopping"]):
                return ActionType.NORMAL_SHOPPING
            elif any(word in action_str for word in ["suspect", "bizarre", "étrange"]):
                return ActionType.SUSPICIOUS_MOVEMENT
            elif any(word in action_str for word in ["caché", "dissimul", "conceal"]):
                return ActionType.ITEM_CONCEALMENT
            elif any(word in action_str for word in ["vol", "potentiel", "theft"]):
                return ActionType.POTENTIAL_THEFT
            elif any(word in action_str for word in ["confirmé", "certain", "confirmed"]):
                return ActionType.CONFIRMED_THEFT
            else:
                return ActionType.NORMAL_SHOPPING
    
    def _heuristic_analysis(self, text: str) -> AnalysisResponse:
        """Analyse heuristique de fallback."""
        
        text_lower = text.lower()
        
        # Détection de mots-clés pour déterminer suspicion
        high_suspicion_keywords = ["vol", "theft", "steal", "suspicious", "danger", "critique"]
        medium_suspicion_keywords = ["caché", "conceal", "hidden", "suspect", "bizarre"]
        
        if any(word in text_lower for word in high_suspicion_keywords):
            suspicion = SuspicionLevel.HIGH
            action = ActionType.POTENTIAL_THEFT
            confidence = 0.6
        elif any(word in text_lower for word in medium_suspicion_keywords):
            suspicion = SuspicionLevel.MEDIUM
            action = ActionType.SUSPICIOUS_MOVEMENT
            confidence = 0.4
        else:
            suspicion = SuspicionLevel.LOW
            action = ActionType.NORMAL_SHOPPING
            confidence = 0.2
        
        # Extraction de recommandations basiques
        recommendations = []
        if "surveill" in text_lower:
            recommendations.append("Surveillance renforcée recommandée")
        if "agent" in text_lower:
            recommendations.append("Intervention agent de sécurité")
        if not recommendations:
            recommendations.append("Analyse manuelle recommandée")
        
        return AnalysisResponse(
            suspicion_level=suspicion,
            action_type=action,
            confidence=confidence,
            description=f"Analyse heuristique: {text[:300]}...",
            reasoning=f"Analyse heuristique du texte: {text[:100]}...",  # ✅ AJOUTÉ
            tools_used=[],
            recommendations=recommendations
        )
    
    def _error_fallback(self, response_text: str, error: str) -> AnalysisResponse:
        """Réponse de fallback en cas d'erreur."""
        
        return AnalysisResponse(
            suspicion_level=SuspicionLevel.LOW,
            action_type=ActionType.NORMAL_SHOPPING,
            confidence=0.0,
            description=f"Erreur parsing VLM: {error}. Réponse brute: {response_text[:200]}...",
            reasoning=f"Erreur de parsing: {error}",  # ✅ AJOUTÉ
            tools_used=[],
            recommendations=["Vérification manuelle urgente", "Relance analyse VLM"]
        )