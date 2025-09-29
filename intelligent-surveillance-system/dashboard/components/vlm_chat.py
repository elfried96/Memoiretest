"""Interface de chat avancée avec VLM."""

import streamlit as st
import asyncio
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
import json
import uuid
import base64
from pathlib import Path

from services.session_manager import get_session_manager
from config.settings import get_dashboard_config

class VLMChatInterface:
    """Interface de chat avancée avec le VLM."""
    
    def __init__(self):
        self.session = get_session_manager()
        self.config = get_dashboard_config()
        self.vlm_callback: Optional[Callable] = None
        self.context_data = {}
        
        # Questions prédéfinies
        self.predefined_questions = {
            "Analyse générale": [
                "Que voyez-vous dans cette vidéo ?",
                "Y a-t-il des comportements suspects ?",
                "Quel est le niveau de risque global ?",
                "Résumez les événements principaux"
            ],
            "Détection spécifique": [
                "Combien de personnes sont visibles ?",
                "Y a-t-il des objets abandonnés ?",
                "Détectez-vous des gestes suspects ?",
                "Y a-t-il des mouvements anormaux ?"
            ],
            "Sécurité": [
                "Évaluez le niveau de menace",
                "Y a-t-il violation de zones interdites ?",
                "Détectez-vous des intrusions ?",
                "Analysez les comportements à risque"
            ],
            "Contextuel": [
                "Que s'est-il passé à ce moment précis ?",
                "Expliquez cette séquence d'événements",
                "Pourquoi cette alerte a-t-elle été déclenchée ?",
                "Analysez cette zone spécifique"
            ]
        }
    
    def render_chat_interface(self):
        """Affiche l'interface de chat complète."""
        
        st.subheader("Chat avec l'IA de Surveillance")
        
        # Onglets pour organiser l'interface
        tab1, tab2, tab3 = st.tabs(["Conversation", "Questions", "Contexte"])
        
        with tab1:
            self._render_conversation_tab()
        
        with tab2:
            self._render_questions_tab()
        
        with tab3:
            self._render_context_tab()
    
    def _render_conversation_tab(self):
        """Onglet conversation principale."""
        
        # Historique des messages
        messages = self.session.get_chat_history()
        
        if not messages:
            st.info("👋 Bonjour ! Je suis votre assistant IA de surveillance. Posez-moi des questions sur les vidéos analysées.")
        
        # Container pour les messages avec hauteur fixe
        chat_container = st.container()
        
        with chat_container:
            for message in messages:
                self._render_message(message)
        
        # Zone de saisie
        self._render_input_area()
    
    def _render_message(self, message: Dict[str, Any]):
        """Affiche un message individuel."""
        
        role = message.get('role', 'user')
        content = message.get('content', '')
        timestamp = message.get('timestamp', '')
        metadata = message.get('metadata', {})
        
        # Formatage timestamp
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            time_str = dt.strftime("%H:%M:%S")
        except:
            time_str = ""
        
        if role == 'user':
            # Message utilisateur
            with st.chat_message("user"):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(content)
                with col2:
                    if time_str:
                        st.caption(time_str)
        
        else:
            # Message assistant
            with st.chat_message("assistant"):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(content)
                    
                    # Métadonnées si disponibles
                    if metadata:
                        with st.expander("Détails de l'analyse", expanded=False):
                            if 'confidence' in metadata:
                                st.progress(metadata['confidence'], "Confiance")
                            
                            if 'analysis_time' in metadata:
                                st.metric("Temps d'analyse", f"{metadata['analysis_time']:.2f}s")
                            
                            if 'tools_used' in metadata:
                                st.write("Outils utilisés:", ", ".join(metadata['tools_used']))
                
                with col2:
                    if time_str:
                        st.caption(time_str)
                    
                    # Boutons d'actions
                    self._render_message_actions(message)
    
    def _render_message_actions(self, message: Dict[str, Any]):
        """Affiche les actions pour un message."""
        
        message_id = message.get('id', '')
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("👍", key=f"like_{message_id}", help="Utile"):
                self._rate_message(message_id, "like")
        
        with col2:
            if st.button("👎", key=f"dislike_{message_id}", help="Pas utile"):
                self._rate_message(message_id, "dislike")
    
    def _render_input_area(self):
        """Zone de saisie des messages."""
        
        # Chat input principal
        user_input = st.chat_input("Posez votre question sur la surveillance...")
        
        # Boutons raccourcis
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("Analyser maintenant", use_container_width=True):
                self._quick_analyze()
        
        with col2:
            if st.button("Quelles alertes ?", use_container_width=True):
                user_input = "Quelles sont les alertes actives et leur niveau de priorité ?"
        
        with col3:
            if st.button("État système", use_container_width=True):
                user_input = "Quel est l'état actuel du système de surveillance ?"
        
        with col4:
            if st.button("Effacer chat", use_container_width=True, type="secondary"):
                self._clear_chat()
        
        # Traitement de l'input
        if user_input:
            self._process_user_input(user_input)
    
    def _render_questions_tab(self):
        """Onglet questions prédéfinies."""
        
        st.write("Sélectionnez une question prédéfinie ou parcourez par catégorie :")
        
        # Sélection par catégorie
        selected_category = st.selectbox(
            "Catégorie",
            list(self.predefined_questions.keys())
        )
        
        # Questions de la catégorie
        if selected_category:
            questions = self.predefined_questions[selected_category]
            
            for i, question in enumerate(questions):
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    st.write(f"**{i+1}.** {question}")
                
                with col2:
                    if st.button("▶", key=f"ask_{selected_category}_{i}"):
                        self._process_user_input(question)
                        st.rerun()
        
        # Section questions personnalisées
        with st.expander("Ajouter une question personnalisée"):
            new_category = st.text_input("Nouvelle catégorie (optionnel)")
            new_question = st.text_input("Question personnalisée")
            
            if st.button("Sauvegarder"):
                if new_question:
                    category = new_category or "Personnalisées"
                    if category not in self.predefined_questions:
                        self.predefined_questions[category] = []
                    
                    self.predefined_questions[category].append(new_question)
                    st.success("Question ajoutée !")
                    st.rerun()
    
    def _render_context_tab(self):
        """Onglet contexte et données."""
        
        st.write("**Contexte actuel pour l'IA :**")
        
        # Analyses vidéo récentes
        video_analyses = self.session.get_all_video_analyses()
        if video_analyses:
            st.write("**Analyses vidéo disponibles:**")
            for video_id, analysis in video_analyses.items():
                with st.expander(f"Vidéo {video_id[:8]}..."):
                    st.json(analysis)
        
        # État des caméras
        cameras_state = self.session.get_all_cameras_state()
        if cameras_state:
            st.write("**État des caméras:**")
            for cam_id, state in cameras_state.items():
                st.write(f"- **{state.get('name', cam_id)}**: "
                        f"{'Actif' if state.get('enabled', False) else 'Inactif'}")
        
        # Alertes récentes
        alerts = self.session.get_active_alerts()
        if alerts:
            st.write("**Alertes actives:**")
            for alert in alerts[-5:]:  # 5 dernières
                level_emoji = {
                    'LOW': '',
                    'MEDIUM': '', 
                    'HIGH': '',
                    'CRITICAL': ''
                }.get(alert.get('level', 'LOW'), '')
                
                st.write(f"{level_emoji} {alert.get('message', 'N/A')}")
        
        # Contexte personnalisé
        with st.expander("Contexte personnalisé"):
            custom_context = st.text_area(
                "Informations additionnelles pour l'IA",
                value=self.context_data.get('custom', ''),
                placeholder="Ex: Nous surveillons un magasin ouvert de 9h à 21h..."
            )
            
            if st.button("Sauvegarder contexte"):
                self.context_data['custom'] = custom_context
                st.success("Contexte sauvegardé !")
    
    def _process_user_input(self, user_input: str):
        """Traite l'input utilisateur."""
        
        if not user_input.strip():
            return
        
        # Ajout du message utilisateur
        self.session.add_chat_message("user", user_input)
        
        # Préparation du contexte
        context = self._build_context()
        
        # Appel au VLM
        with st.spinner("🤔 L'IA analyse votre question..."):
            try:
                response = self._get_vlm_response(user_input, context)
                
                # Ajout de la réponse
                self.session.add_chat_message(
                    "assistant",
                    response.get('content', 'Désolé, je ne peux pas répondre pour le moment.'),
                    response.get('metadata', {})
                )
                
            except Exception as e:
                error_msg = f"Erreur lors de l'analyse: {str(e)}"
                self.session.add_chat_message("assistant", error_msg)
                st.error(error_msg)
        
        st.rerun()
    
    def _build_context(self) -> Dict[str, Any]:
        """Construit le contexte pour le VLM."""
        
        context = {
            'timestamp': datetime.now().isoformat(),
            'session_id': self.session.get_session_id()
        }
        
        # Analyses vidéo
        video_analyses = self.session.get_all_video_analyses()
        if video_analyses:
            context['video_analyses'] = video_analyses
        
        # État des caméras
        cameras_state = self.session.get_all_cameras_state()
        if cameras_state:
            context['cameras_state'] = cameras_state
        
        # Alertes
        alerts = self.session.get_active_alerts()
        if alerts:
            context['active_alerts'] = alerts
        
        # Contexte personnalisé
        if 'custom' in self.context_data:
            context['custom_context'] = self.context_data['custom']
        
        return context
    
    def _get_vlm_response(self, question: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Obtient une réponse du VLM."""
        
        if self.vlm_callback:
            # Appel au VLM via callback
            start_time = datetime.now()
            
            response = self.vlm_callback(question, context)
            
            analysis_time = (datetime.now() - start_time).total_seconds()
            
            # Enrichissement avec métadonnées
            if isinstance(response, str):
                response = {'content': response}
            
            response['metadata'] = {
                'analysis_time': analysis_time,
                'context_size': len(str(context)),
                **response.get('metadata', {})
            }
            
            return response
        
        else:
            # Réponse simulée en mode démo
            return {
                'content': f" Réponse simulée à: {question}\n\nEn mode démo, le VLM n'est pas connecté. Voici ce que je vois dans le contexte:\n- {len(context.get('video_analyses', {}))} analyses vidéo disponibles\n- {len(context.get('cameras_state', {}))} caméras configurées\n- {len(context.get('active_alerts', []))} alertes actives",
                'metadata': {
                    'analysis_time': 0.5,
                    'demo_mode': True,
                    'confidence': 0.8
                }
            }
    
    def _quick_analyze(self):
        """Lance une analyse rapide."""
        self._process_user_input("Analysez la situation actuelle et donnez-moi un résumé des événements importants.")
    
    def _clear_chat(self):
        """Efface l'historique du chat."""
        self.session.clear_chat_history()
        st.success("Historique effacé")
        st.rerun()
    
    def _rate_message(self, message_id: str, rating: str):
        """Note un message."""
        # TODO: Implémenter système de notation
        st.success(f"Merci pour votre retour ! ({rating})")
    
    def set_vlm_callback(self, callback: Callable[[str, Dict], Dict]):
        """Définit le callback VLM."""
        self.vlm_callback = callback
    
    def add_system_message(self, content: str, level: str = "info"):
        """Ajoute un message système."""
        
        level_prefixes = {
            'info': '',
            'warning': '',
            'error': '',
            'success': ''
        }
        
        prefix = level_prefixes.get(level, '')
        formatted_content = f"{prefix} **Système**: {content}"
        
        self.session.add_chat_message(
            "assistant",
            formatted_content,
            {'system_message': True, 'level': level}
        )
    
    def export_chat_history(self, format: str = 'json') -> str:
        """Exporte l'historique du chat."""
        
        history = self.session.get_chat_history()
        
        if format.lower() == 'json':
            return json.dumps(history, indent=2, ensure_ascii=False)
        
        elif format.lower() == 'txt':
            lines = []
            for msg in history:
                timestamp = msg.get('timestamp', '')
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')
                
                lines.append(f"[{timestamp}] {role.upper()}: {content}")
                lines.append("")
            
            return "\n".join(lines)
        
        return json.dumps(history, indent=2, ensure_ascii=False)

# Instance globale
def get_vlm_chat() -> VLMChatInterface:
    """Récupère l'instance de chat VLM."""
    if 'vlm_chat' not in st.session_state:
        st.session_state.vlm_chat = VLMChatInterface()
    
    return st.session_state.vlm_chat

# Fonctions utilitaires

def render_chat_sidebar():
    """Affiche un chat compact dans la sidebar."""
    
    chat = get_vlm_chat()
    
    with st.sidebar:
        st.subheader("Chat Rapide")
        
        # Derniers messages
        messages = chat.session.get_chat_history()
        if messages:
            last_msg = messages[-1]
            role = last_msg.get('role', 'user')
            content = last_msg.get('content', '')
            
            if role == 'assistant':
                st.success(content[:100] + "..." if len(content) > 100 else content)
            else:
                st.info(content[:100] + "..." if len(content) > 100 else content)
        
        # Input rapide
        quick_input = st.text_input("Question rapide", key="sidebar_chat_input")
        
        if st.button("➤ Envoyer", use_container_width=True):
            if quick_input:
                chat._process_user_input(quick_input)
                st.rerun()

def render_quick_questions_bar():
    """Barre de questions rapides."""
    
    st.write("**Questions rapides:**")
    
    col1, col2, col3, col4 = st.columns(4)
    
    chat = get_vlm_chat()
    
    with col1:
        if st.button("Que se passe-t-il ?", use_container_width=True):
            chat._process_user_input("Analysez la situation actuelle et résumez les événements.")
    
    with col2:
        if st.button("Niveau de risque ?", use_container_width=True):
            chat._process_user_input("Quel est le niveau de risque actuel ?")
    
    with col3:
        if st.button("Combien de personnes ?", use_container_width=True):
            chat._process_user_input("Combien de personnes sont visibles sur les caméras ?")
    
    with col4:
        if st.button("Alertes actives ?", use_container_width=True):
            chat._process_user_input("Quelles sont les alertes actives et leur priorité ?")