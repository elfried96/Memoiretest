"""Module de construction des prompts pour VLM."""

import json
from typing import Dict, List, Any
from datetime import datetime


class PromptBuilder:
    """Constructeur de prompts spécialisé pour la surveillance."""
    
    def __init__(self):
        self._tools_info = {
            # Outils existants
            "object_detector": "Détection d'objets YOLO - identifie personnes, produits",
            "tracker": "Suivi DeepSORT - analyse trajectoires et mouvements", 
            "behavior_analyzer": "Analyse comportementale - détecte gestes suspects",
            
            # Nouveaux outils avancés intégrés
            "sam2_segmentator": "Segmentation SAM2 - masques précis des objets",
            "dino_features": "Extraction features DINO v2 - représentations visuelles robustes",
            "pose_estimator": "Estimation poses OpenPose - analyse postures et gestes",
            "trajectory_analyzer": "Analyse trajectoires avancée - patterns de mouvement",
            "multimodal_fusion": "Fusion multimodale - agrégation intelligente des données",
            "temporal_transformer": "Analyse temporelle - patterns comportementaux dans le temps",
            "adversarial_detector": "Détection d'attaques - protection contre manipulations",
            "domain_adapter": "Adaptation domaine - optimisation pour différents environnements"
        }
    
    def build_surveillance_prompt(
        self, 
        context: Dict[str, Any],
        available_tools: List[str],
        tools_results: Dict[str, Any] = None
    ) -> str:
        """Construction du prompt principal de surveillance."""
        
        base_prompt = """Tu es un système VLM de surveillance intelligente spécialisé dans la prévention du vol.

CONTEXTE SURVEILLANCE:
- Zone: {location}
- Horodatage: {timestamp}
- Historique: {previous_detections}

OUTILS DISPONIBLES:
{tools_description}

{tools_results_section}

MISSION:
Analyse cette image et détermine:
1. Niveau de suspicion (LOW/MEDIUM/HIGH/CRITICAL)
2. Type d'action observée
3. Outils recommandés pour validation
4. Actions à entreprendre

IMPORTANT:
- Privilégie précision sur vitesse
- Évite absolument les faux positifs
- Utilise validation croisée si suspicion élevée
- Intègre les résultats des outils avancés

FORMAT RÉPONSE JSON:
{json_format}"""

        # Construction des sections
        tools_description = self._build_tools_description(available_tools)
        tools_results_section = self._build_tools_results_section(tools_results)
        json_format = self._get_json_format()
        
        return base_prompt.format(
            location=context.get("location", "Zone inconnue"),
            timestamp=context.get("timestamp", datetime.now().isoformat()),
            previous_detections=json.dumps(context.get("previous_detections", []), indent=2),
            tools_description=tools_description,
            tools_results_section=tools_results_section,
            json_format=json_format
        )
    
    def _build_tools_description(self, available_tools: List[str]) -> str:
        """Construction de la description des outils disponibles."""
        descriptions = []
        for tool in available_tools:
            if tool in self._tools_info:
                descriptions.append(f"- {tool}: {self._tools_info[tool]}")
        return "\n".join(descriptions)
    
    def _build_tools_results_section(self, tools_results: Dict[str, Any] = None) -> str:
        """Construction de la section résultats d'outils."""
        if not tools_results:
            return ""
        
        section = "\nRÉSULTATS DES OUTILS EXÉCUTÉS:\n"
        for tool_name, result in tools_results.items():
            status = "✓" if result.get("success", False) else "✗"
            confidence = result.get("confidence", 0.0)
            section += f"{status} {tool_name} (confiance: {confidence:.2f}): {json.dumps(result.get('data', {}), indent=2)}\n"
        
        return section
    
    def _get_json_format(self) -> str:
        """Format JSON attendu pour la réponse."""
        return """{
    "suspicion_level": "LOW|MEDIUM|HIGH|CRITICAL",
    "action_type": "normal_shopping|suspicious_movement|item_concealment|potential_theft|confirmed_theft",
    "confidence": 0.85,
    "description": "Description détaillée de l'observation",
    "tools_to_use": ["tool1", "tool2"],
    "reasoning": "Explication du raisonnement",
    "recommendations": ["action1", "action2"]
}"""