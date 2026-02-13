"""Service GPT intelligent pour le chat de surveillance VLM."""

import os
import openai
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import asyncio
from pathlib import Path

# Configuration de l'API OpenAI
def load_openai_key():
    """Charge la clé API OpenAI depuis le fichier .env."""
    env_file = Path(__file__).parent.parent / ".env"
    
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                if line.startswith('OPENAI_API_KEY='):
                    api_key = line.split('=', 1)[1].strip()
                    return api_key
    
    # Fallback vers variable d'environnement
    return os.getenv('OPENAI_API_KEY')

# Initialisation
OPENAI_API_KEY = load_openai_key()
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

class GPTSurveillanceChat:
    """Service de chat intelligent basé sur GPT pour l'analyse de surveillance."""
    
    def __init__(self):
        self.api_available = bool(OPENAI_API_KEY)
        self.client = None
        
        if self.api_available:
            try:
                from openai import AsyncOpenAI
                self.client = AsyncOpenAI(api_key=OPENAI_API_KEY)
            except ImportError:
                # Fallback vers ancienne version
                try:
                    import openai
                    openai.api_key = OPENAI_API_KEY
                    self.client = openai
                    self.use_legacy_api = True
                except ImportError:
                    self.api_available = False
        
        self.stats = {
            'queries_processed': 0,
            'successful_queries': 0,
            'error_count': 0,
            'average_response_time': 0.0
        }
    
    async def analyze_surveillance_question(
        self, 
        question: str, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyse intelligente d'une question de surveillance avec GPT.
        
        Args:
            question: Question de l'utilisateur
            context: Contexte complet (analyses vidéo, descriptions, alertes)
            
        Returns:
            Réponse structurée avec contenu et métadonnées
        """
        
        if not self.api_available:
            return self._fallback_response(question, context)
        
        start_time = datetime.now()
        
        try:
            # Construction du prompt contextualisé
            system_prompt = self._build_surveillance_system_prompt()
            user_prompt = self._build_contextual_user_prompt(question, context)
            
            # Appel GPT avec contexte enrichi
            if hasattr(self, 'use_legacy_api') and self.use_legacy_api:
                response_content = await self._call_legacy_gpt(system_prompt, user_prompt)
            else:
                response_content = await self._call_modern_gpt(system_prompt, user_prompt)
            
            # Calcul du temps de traitement
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Mise à jour des stats
            self.stats['queries_processed'] += 1
            self.stats['successful_queries'] += 1
            self.stats['average_response_time'] = (
                (self.stats['average_response_time'] * (self.stats['successful_queries'] - 1) + processing_time)
                / self.stats['successful_queries']
            )
            
            return {
                'content': response_content,
                'metadata': {
                    'provider': 'gpt-4',
                    'analysis_time': processing_time,
                    'confidence': 0.92,  # GPT-4 est très fiable pour l'analyse contextuelle
                    'tools_used': ['gpt-4', 'context_analysis', 'video_analysis'],
                    'context_items': len(context),
                    'api_available': True
                }
            }
            
        except Exception as e:
            self.stats['error_count'] += 1
            print(f"Erreur GPT Chat: {e}")
            return self._fallback_response(question, context)
    
    def _build_surveillance_system_prompt(self) -> str:
        """Construit le prompt système pour GPT spécialisé en surveillance."""
        
        return """Tu es un expert en surveillance retail et sécurité avec 15 ans d'expérience dans l'analyse comportementale et la prévention du vol.

RÔLE ET EXPERTISE:
- Expert en surveillance de magasins/supermarchés
- Spécialiste en détection de comportements suspects et vol à l'étalage
- Analyste en sécurité retail avec compréhension des techniques de vol modernes
- Conseiller en sécurité pour les décisions d'intervention

CAPACITÉS SPÉCIALISÉES:
- Analyse contextuelle des comportements suspects
- Évaluation des niveaux de risque et recommandations d'actions
- Interprétation des données de surveillance (vidéo, détections, timelines)
- Raisonnement basé sur le contexte fourni par l'opérateur

INSTRUCTIONS CRITIQUES:
1. **PRIORITÉ AU CONTEXTE UTILISATEUR**: Si l'opérateur mentionne "sortie sans payer", "vol", "vient de sortir sans passer à la caisse", etc., c'est une INFORMATION FACTUELLE qui prévaut sur l'analyse automatique.

2. **RAISONNEMENT CONTEXTUEL**: Analyse toujours en tenant compte de:
   - La description détaillée fournie par l'opérateur
   - Les résultats d'analyse vidéo automatique  
   - Le niveau de suspicion détecté
   - La timeline des événements
   - Les objets et comportements détectés

3. **RÉPONSES PROFESSIONNELLES**: 
   - Fournir des analyses claires et actionnables
   - Proposer des recommandations spécifiques
   - Évaluer les risques de façon équilibrée
   - Éviter les faux positifs mais ne pas ignorer les signaux d'alarme

4. **STYLE DE COMMUNICATION**:
   - Professionnel mais accessible
   - Structuré avec sections claires
   - Basé sur les faits observés
   - Recommandations pratiques et proportionnées

Tu dois analyser chaque situation avec l'expertise d'un professionnel de la sécurité retail et fournir des conseils pertinents pour l'aide à la décision."""

    def _build_contextual_user_prompt(self, question: str, context: Dict[str, Any]) -> str:
        """Construit le prompt utilisateur avec tout le contexte."""
        
        prompt = f"**QUESTION DE L'OPÉRATEUR**: {question}\n\n"
        
        # Contexte vidéo si disponible
        video_analyses = context.get('video_analyses', {})
        if video_analyses:
            prompt += "**CONTEXTE VIDÉO ANALYSÉ**:\n"
            
            for video_name, analysis in video_analyses.items():
                prompt += f"\n📹 **{video_name}**:\n"
                
                # Description utilisateur si disponible
                if 'options_used' in analysis and analysis['options_used'].get('description_detaillee'):
                    user_description = analysis['options_used']['description_detaillee']
                    prompt += f"🔴 **DESCRIPTION OPÉRATEUR**: {user_description}\n"
                
                # Résultats analyse automatique
                prompt += f"📊 **ANALYSE AUTOMATIQUE**:\n"
                prompt += f"   • Niveau de suspicion: {analysis.get('suspicion_level', 'Non déterminé')}\n"
                prompt += f"   • Confiance: {analysis.get('confidence', 0):.0%}\n"
                
                # Timeline des événements
                timeline = analysis.get('timeline', [])
                if timeline:
                    prompt += f"⏱️ **CHRONOLOGIE**:\n"
                    for event in timeline[:3]:  # Limiter à 3 événements
                        time_marker = event.get('time', 'N/A')
                        event_desc = event.get('event', 'Événement non spécifié')
                        prompt += f"   • {time_marker}: {event_desc}\n"
                
                # Objets détectés
                detected_objects = analysis.get('detected_objects', [])
                if detected_objects:
                    prompt += f"🔍 **DÉTECTIONS**:\n"
                    for obj in detected_objects[:3]:
                        obj_type = obj.get('type', 'objet')
                        obj_count = obj.get('count', 1)
                        obj_confidence = obj.get('confidence', 0.0)
                        prompt += f"   • {obj_count}x {obj_type} (confiance: {obj_confidence:.0%})\n"
                
                prompt += "\n"
        
        # État du système
        cameras_state = context.get('cameras_state', {})
        active_cameras = sum(1 for cam in cameras_state.values() if cam.get('enabled', False))
        prompt += f"📷 **ÉTAT SURVEILLANCE**: {active_cameras}/{len(cameras_state)} caméras actives\n"
        
        # Alertes actives
        active_alerts = context.get('active_alerts', [])
        if active_alerts:
            prompt += f"🚨 **ALERTES ACTIVES**: {len(active_alerts)} alerte(s)\n"
            for alert in active_alerts[:2]:
                level = alert.get('level', 'INFO')
                message = alert.get('message', 'Alerte non spécifiée')
                prompt += f"   • [{level}] {message}\n"
        else:
            prompt += "✅ **ALERTES**: Aucune alerte active\n"
        
        prompt += "\n**ANALYSE DEMANDÉE**: "
        prompt += "En tant qu'expert en surveillance retail, analyse cette situation et répond à la question de l'opérateur avec ton expertise professionnelle. "
        prompt += "Prends en compte TOUS les éléments contextuels, particulièrement la description de l'opérateur qui a la priorité absolue."
        
        return prompt
    
    async def _call_modern_gpt(self, system_prompt: str, user_prompt: str) -> str:
        """Appel API GPT moderne (OpenAI >= 1.0)."""
        
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=800,
                temperature=0.3,  # Faible pour des réponses plus cohérentes
                top_p=0.9
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Erreur appel GPT moderne: {e}")
            # Fallback vers GPT-3.5 si GPT-4 échoue
            try:
                response = await self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=600,
                    temperature=0.3
                )
                return response.choices[0].message.content.strip()
            except Exception as e2:
                raise e2
    
    async def _call_legacy_gpt(self, system_prompt: str, user_prompt: str) -> str:
        """Appel API GPT legacy (OpenAI < 1.0)."""
        
        try:
            response = await openai.ChatCompletion.acreate(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=800,
                temperature=0.3,
                top_p=0.9
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception:
            # Fallback vers GPT-3.5
            response = await openai.ChatCompletion.acreate(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=600,
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
    
    def _fallback_response(self, question: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Réponse de secours si GPT n'est pas disponible."""
        
        video_analyses = context.get('video_analyses', {})
        fallback_content = "🤖 **ANALYSE DE SURVEILLANCE** (Mode Fallback)\n\n"
        
        if video_analyses:
            latest_analysis = list(video_analyses.values())[-1]
            suspicion = latest_analysis.get('suspicion_level', 'LOW')
            
            fallback_content += f"**Votre question**: {question}\n\n"
            fallback_content += f"**Analyse disponible**:\n"
            fallback_content += f"• Niveau de suspicion détecté: {suspicion}\n"
            fallback_content += f"• Confiance: {latest_analysis.get('confidence', 0):.0%}\n\n"
            
            if suspicion in ['HIGH', 'CRITICAL']:
                fallback_content += "⚠️ **RECOMMANDATION**: Niveau de suspicion élevé détecté - vérification manuelle recommandée.\n"
            elif suspicion == 'MEDIUM':
                fallback_content += "👁️ **RECOMMANDATION**: Surveillance continue - situation à surveiller.\n"
            else:
                fallback_content += "✅ **RECOMMANDATION**: Comportement normal détecté - surveillance de routine.\n"
        else:
            fallback_content += f"**Votre question**: {question}\n\n"
            fallback_content += "Aucune analyse vidéo récente disponible. Pour une analyse approfondie, veuillez uploader une vidéo.\n"
        
        fallback_content += "\n💡 **Note**: Service GPT temporairement indisponible - analyse basique utilisée."
        
        return {
            'content': fallback_content,
            'metadata': {
                'provider': 'fallback',
                'analysis_time': 0.1,
                'confidence': 0.6,
                'tools_used': ['fallback_analysis'],
                'context_items': len(context),
                'api_available': False
            }
        }
    
    def get_service_status(self) -> Dict[str, Any]:
        """État du service GPT."""
        
        return {
            'api_available': self.api_available,
            'api_key_configured': bool(OPENAI_API_KEY),
            'stats': self.stats.copy(),
            'models_available': ['gpt-4', 'gpt-3.5-turbo'] if self.api_available else [],
            'service_ready': self.api_available and bool(OPENAI_API_KEY)
        }

# Instance globale
_gpt_chat_service: Optional[GPTSurveillanceChat] = None

def get_gpt_chat_service() -> GPTSurveillanceChat:
    """Récupère le service GPT (singleton)."""
    global _gpt_chat_service
    
    if _gpt_chat_service is None:
        _gpt_chat_service = GPTSurveillanceChat()
    
    return _gpt_chat_service