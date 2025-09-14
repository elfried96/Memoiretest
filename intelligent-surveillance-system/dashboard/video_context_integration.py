"""
🎥 Video Context Integration - Métadonnées pour VLM
==================================================

Intégration des descriptions et métadonnées vidéo dans le contexte VLM
pour une analyse plus précise et contextualisée.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import json
from dataclasses import dataclass
from loguru import logger


@dataclass
class VideoContextMetadata:
    """Métadonnées contextuelles pour analyse vidéo VLM."""
    title: str
    location_type: str
    time_context: str
    expected_activities: List[str]
    suspicious_focus: List[str]
    camera_angle: str
    detailed_description: str
    analysis_priority: str
    frame_sampling: str
    user_timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'title': self.title,
            'location_type': self.location_type,
            'time_context': self.time_context,
            'expected_activities': self.expected_activities,
            'suspicious_focus': self.suspicious_focus,
            'camera_angle': self.camera_angle,
            'detailed_description': self.detailed_description,
            'analysis_priority': self.analysis_priority,
            'frame_sampling': self.frame_sampling,
            'user_timestamp': self.user_timestamp
        }


class VideoContextPromptBuilder:
    """Constructeur de prompts enrichis avec contexte vidéo."""
    
    def __init__(self):
        # Mapping des types de lieux vers caractéristiques surveillance
        self.location_characteristics = {
            "Magasin/Commerce": {
                "typical_behaviors": ["shopping", "browsing", "queuing", "payment"],
                "risk_patterns": ["shoplifting", "price_switching", "distraction_theft"],
                "key_zones": ["aisles", "checkout", "entrance", "changing_rooms"],
                "peak_times": ["lunch_hours", "evenings", "weekends"]
            },
            "Entrepôt": {
                "typical_behaviors": ["loading", "unloading", "inventory", "forklift_operation"],
                "risk_patterns": ["theft", "unauthorized_access", "safety_violations"],
                "key_zones": ["loading_docks", "storage_areas", "office_areas"],
                "peak_times": ["shift_changes", "delivery_hours"]
            },
            "Bureau": {
                "typical_behaviors": ["working", "meetings", "coffee_breaks", "cleaning"],
                "risk_patterns": ["unauthorized_access", "data_theft", "vandalism"],
                "key_zones": ["entrances", "server_rooms", "executive_areas"],
                "peak_times": ["office_hours", "after_hours"]
            },
            "Parking": {
                "typical_behaviors": ["parking", "walking", "loading_vehicles"],
                "risk_patterns": ["car_theft", "break_ins", "vandalism", "loitering"],
                "key_zones": ["entrances", "payment_areas", "isolated_corners"],
                "peak_times": ["rush_hours", "late_evenings"]
            }
        }
        
        # Mapping contextes temporels vers comportements attendus
        self.temporal_contexts = {
            "Heures ouverture": {
                "activity_level": "normal",
                "staff_presence": "full",
                "customer_flow": "regular",
                "alert_sensitivity": "standard"
            },
            "Heures affluence": {
                "activity_level": "high", 
                "staff_presence": "full",
                "customer_flow": "heavy",
                "alert_sensitivity": "reduced_for_crowding"
            },
            "Heures creuses": {
                "activity_level": "low",
                "staff_presence": "minimal", 
                "customer_flow": "sparse",
                "alert_sensitivity": "increased"
            },
            "Nuit/Fermeture": {
                "activity_level": "minimal",
                "staff_presence": "security_only",
                "customer_flow": "none",
                "alert_sensitivity": "maximum"
            }
        }
    
    def build_context_enhanced_prompt(
        self,
        base_prompt: str,
        video_metadata: VideoContextMetadata,
        frame_context: Dict[str, Any] = None
    ) -> str:
        """Construit un prompt enrichi avec le contexte vidéo."""
        
        location_info = self.location_characteristics.get(
            video_metadata.location_type, 
            {"typical_behaviors": [], "risk_patterns": [], "key_zones": [], "peak_times": []}
        )
        
        temporal_info = self.temporal_contexts.get(
            video_metadata.time_context,
            {"activity_level": "unknown", "staff_presence": "unknown", "customer_flow": "unknown", "alert_sensitivity": "standard"}
        )
        
        context_enhancement = f"""

🎥 CONTEXTE VIDÉO SPÉCIFIQUE - INFORMATIONS UTILISATEUR:
=====================================================

📋 IDENTIFICATION:
- Titre: "{video_metadata.title}"
- Type environnement: {video_metadata.location_type}
- Contexte temporel: {video_metadata.time_context}
- Angle caméra: {video_metadata.camera_angle}

📊 CARACTÉRISTIQUES ENVIRONNEMENT {video_metadata.location_type.upper()}:
- Comportements typiques: {', '.join(location_info['typical_behaviors'])}
- Patterns risque connus: {', '.join(location_info['risk_patterns'])}
- Zones critiques: {', '.join(location_info['key_zones'])}
- Pics d'activité: {', '.join(location_info['peak_times'])}

⏰ CONTEXTE TEMPOREL "{video_metadata.time_context}":
- Niveau activité attendu: {temporal_info['activity_level']}
- Présence personnel: {temporal_info['staff_presence']}
- Flux clients: {temporal_info['customer_flow']}
- Sensibilité alertes: {temporal_info['alert_sensitivity']}

✅ ACTIVITÉS NORMALES ATTENDUES:
{self._format_list_for_prompt(video_metadata.expected_activities)}

🚨 FOCUS SURVEILLANCE PRIORITAIRE:
{self._format_list_for_prompt(video_metadata.suspicious_focus)}

📝 DESCRIPTION DÉTAILLÉE UTILISATEUR:
"{video_metadata.detailed_description}"

🎯 PRIORITÉ ANALYSE: {video_metadata.analysis_priority}
📊 ÉCHANTILLONNAGE: {video_metadata.frame_sampling}

INSTRUCTIONS CONTEXTUALISÉES:
=============================

🔍 ADAPTATION SELON ENVIRONNEMENT:
- Calibre tes seuils de suspicion selon le type "{video_metadata.location_type}"
- Prends en compte le contexte "{video_metadata.time_context}" pour évaluer normalité
- Perspective caméra "{video_metadata.camera_angle}" influence interprétation spatiale

⚖️ ÉVALUATION COMPORTEMENTS:
- NORMAUX dans ce contexte: {', '.join(video_metadata.expected_activities)}
- SUSPECTS à prioriser: {', '.join(video_metadata.suspicious_focus)}
- Distinguer activité normale vs comportement inhabituel selon contexte temporel

🎯 OBJECTIFS SPÉCIFIQUES:
- Focus principal: détection patterns listés en "Focus surveillance"
- Ignorer ou minimiser activités normales listées sauf si vraiment suspectes  
- Adapter confiance selon qualité description utilisateur
- Corréler avec description détaillée fournie

⚠️ CONTRAINTES CONTEXTUELLES:
- Environnement "{video_metadata.location_type}" a des patterns comportementaux spécifiques
- Période "{video_metadata.time_context}" influence normalité des activités
- Description utilisateur doit primer sur assumptions générales
- Perspective "{video_metadata.camera_angle}" affecte visibilité détails

CALIBRAGE SUSPICION CONTEXTUEL:
- LOW (0.0-0.3): Activité listée comme normale ET cohérente avec environnement/temps
- MEDIUM (0.3-0.6): Activité non listée mais cohérente avec contexte général
- HIGH (0.6-0.8): Comportement incohérent avec contexte OU focus surveillance détecté
- CRITICAL (0.8-1.0): Focus surveillance confirmé ET description utilisateur validée"""

        return base_prompt + context_enhancement
    
    def _format_list_for_prompt(self, items: List[str]) -> str:
        """Formate une liste pour inclusion dans le prompt."""
        if not items:
            return "Aucun élément spécifié"
        return f"[{', '.join(items)}]"
    
    def extract_analysis_objectives(self, video_metadata: VideoContextMetadata) -> Dict[str, Any]:
        """Extrait les objectifs d'analyse à partir des métadonnées."""
        
        return {
            "primary_focus": video_metadata.suspicious_focus,
            "normal_baseline": video_metadata.expected_activities,
            "environment_adaptation": self.location_characteristics.get(
                video_metadata.location_type, {}
            ),
            "temporal_adaptation": self.temporal_contexts.get(
                video_metadata.time_context, {}
            ),
            "user_priority": video_metadata.analysis_priority,
            "detailed_context": video_metadata.detailed_description
        }
    
    def generate_contextual_questions(self, video_metadata: VideoContextMetadata) -> List[str]:
        """Génère des questions contextualisées pour le chat VLM."""
        
        questions = []
        
        # Questions basées sur le lieu
        if video_metadata.location_type == "Magasin/Commerce":
            questions.extend([
                f"Analyse les comportements de shopping dans '{video_metadata.title}'",
                "Détecte-t-on des tentatives de vol à l'étalage ?",
                "Comment les clients interagissent avec les produits ?",
                "Y a-t-il des comportements d'évitement du personnel ?"
            ])
        elif video_metadata.location_type == "Bureau":
            questions.extend([
                f"Analyse les activités professionnelles dans '{video_metadata.title}'",
                "Détecte-t-on des accès non autorisés ?",
                "Comment se déroulent les interactions professionnelles ?",
                "Y a-t-il des comportements inhabituels pour un environnement bureau ?"
            ])
        
        # Questions basées sur le focus surveillance
        for focus in video_metadata.suspicious_focus:
            if focus == "Vol à l'étalage":
                questions.append("Identifie les signes de vol à l'étalage dans cette séquence")
            elif focus == "Comportements agressifs":
                questions.append("Analyse les interactions pour détecter agressivité ou tensions")
            elif focus == "Intrusion":
                questions.append("Vérifie les accès et détecte d'éventuelles intrusions")
        
        # Questions basées sur la description utilisateur
        if video_metadata.detailed_description:
            questions.append(f"Analyse cette vidéo selon le contexte: '{video_metadata.detailed_description[:100]}...'")
        
        return questions[:8]  # Limite à 8 questions max


class VideoContextChatIntegration:
    """Intégration du contexte vidéo dans le chat VLM."""
    
    def __init__(self):
        self.prompt_builder = VideoContextPromptBuilder()
    
    def enhance_chat_context(
        self,
        base_chat_context: Dict[str, Any],
        video_metadata: Optional[VideoContextMetadata] = None
    ) -> Dict[str, Any]:
        """Enrichit le contexte chat avec les métadonnées vidéo."""
        
        enhanced_context = base_chat_context.copy()
        
        if video_metadata:
            enhanced_context['video_context'] = {
                'metadata': video_metadata.to_dict(),
                'analysis_objectives': self.prompt_builder.extract_analysis_objectives(video_metadata),
                'contextual_questions': self.prompt_builder.generate_contextual_questions(video_metadata),
                'context_enhanced': True
            }
            
            # Ajout de métadonnées pour le chatbot
            enhanced_context['user_intent_context'] = {
                'primary_focus': video_metadata.suspicious_focus,
                'environment': video_metadata.location_type,
                'temporal_context': video_metadata.time_context,
                'user_description': video_metadata.detailed_description
            }
        
        return enhanced_context
    
    def generate_context_aware_response_enhancement(
        self, 
        question: str,
        video_metadata: VideoContextMetadata
    ) -> str:
        """Génère une amélioration de réponse basée sur le contexte vidéo."""
        
        enhancement = f"""
CONTEXTE VIDÉO POUR CETTE QUESTION:
- Analyse de: "{video_metadata.title}"
- Environnement: {video_metadata.location_type} 
- Focus: {', '.join(video_metadata.suspicious_focus)}
- Contexte utilisateur: {video_metadata.detailed_description[:200]}...

Adapte ta réponse selon ce contexte spécifique."""
        
        return enhancement


# Fonctions utilitaires globales
def create_video_metadata_from_form(form_data: Dict[str, Any]) -> VideoContextMetadata:
    """Crée un objet VideoContextMetadata à partir des données formulaire."""
    
    return VideoContextMetadata(
        title=form_data.get('title', ''),
        location_type=form_data.get('location_type', 'Autre'),
        time_context=form_data.get('time_context', 'Non spécifié'),
        expected_activities=form_data.get('expected_activities', []),
        suspicious_focus=form_data.get('suspicious_focus', []),
        camera_angle=form_data.get('camera_angle', 'Non spécifié'),
        detailed_description=form_data.get('detailed_description', ''),
        analysis_priority=form_data.get('analysis_priority', 'Équilibré'),
        frame_sampling=form_data.get('frame_sampling', 'Standard'),
        user_timestamp=datetime.now().isoformat()
    )


def integrate_video_context_in_vlm_analysis(
    base_analysis_request,
    video_metadata: VideoContextMetadata
):
    """Intègre le contexte vidéo dans une requête d'analyse VLM."""
    
    if hasattr(base_analysis_request, 'context'):
        if not base_analysis_request.context:
            base_analysis_request.context = {}
        
        base_analysis_request.context.update({
            'video_metadata': video_metadata.to_dict(),
            'context_enhanced': True,
            'user_provided_context': True
        })
    
    return base_analysis_request


# Instance globale
_video_context_integration = None

def get_video_context_integration() -> VideoContextChatIntegration:
    """Récupère l'instance globale d'intégration contexte vidéo."""
    global _video_context_integration
    if _video_context_integration is None:
        _video_context_integration = VideoContextChatIntegration()
    return _video_context_integration