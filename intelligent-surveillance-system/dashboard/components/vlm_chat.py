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
from services.vlm_integration import get_vlm_service
from services.video_memory_system import get_video_memory_system

class VLMChatInterface:
    """Interface de chat avancée avec le VLM."""
    
    def __init__(self):
        self.session = get_session_manager()
        self.config = get_dashboard_config()
        self.vlm_callback: Optional[Callable] = None
        self.context_data = {}
        
        # Service VLM intégré du projet
        self.vlm_service = get_vlm_service()
        
        # ✅ NOUVEAU: Système mémoire vidéo avancé
        self.video_memory_system = get_video_memory_system()
        
        # ✅ NOUVEAU: Contexte vidéo actuel
        self.current_video_context = None
        
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
                                confidence = metadata['confidence']
                                # Système feu tricolore pour utilisateurs non-techniciens
                                if confidence > 0.8:
                                    st.success("🟢 **Analyse très fiable** - L'IA est sûre de son analyse")
                                elif confidence > 0.6:
                                    st.warning("🟡 **Analyse correcte** - Vérification recommandée")
                                else:
                                    st.error("🔴 **Analyse incertaine** - Vérification manuelle nécessaire")
                                
                                st.progress(confidence, f"Niveau de confiance: {confidence:.0%}")
                            
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
                # Vérification VLM avant analyse
                if not self.get_vlm_status()['initialized']:
                    st.warning("⚠️ VLM non initialisé. Allez dans l'onglet Contexte pour l'initialiser.")
                else:
                    self._quick_analyze()
        
        # ✅ NOUVEAU: Indicateur contexte vidéo actif
        if self.current_video_context:
            st.info(f"🎯 Contexte vidéo actif: {self.current_video_context.get('video_name', 'vidéo')}")
        
        with col2:
            if st.button("Quelles alertes ?", use_container_width=True):
                self._process_user_input("Quelles sont les alertes actives et leur niveau de priorité ?")
                st.rerun()
        
        with col3:
            if st.button("État système", use_container_width=True):
                self._process_user_input("Quel est l'état actuel du système de surveillance ?")
                st.rerun()
        
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
        
        # Status VLM
        st.subheader("🤖 Status du VLM")
        
        vlm_status = self.get_vlm_status()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            status_emoji = "✅" if vlm_status['initialized'] else "❌"
            st.metric("VLM Status", f"{status_emoji} {'Actif' if vlm_status['initialized'] else 'Inactif'}")
        
        with col2:
            available_emoji = "✅" if vlm_status['vlm_available'] else "❌"
            st.metric("Disponibilité", f"{available_emoji} {'Oui' if vlm_status['vlm_available'] else 'Non'}")
        
        with col3:
            model_emoji = "✅" if vlm_status['model_loaded'] else "❌"
            st.metric("Modèle", f"{model_emoji} {'Chargé' if vlm_status['model_loaded'] else 'Non chargé'}")
        
        # Bouton d'initialisation
        if not vlm_status['initialized']:
            if st.button("🚀 Initialiser VLM", use_container_width=True):
                self.initialize_vlm()
                st.rerun()
        
        # Statistiques si disponibles
        if 'stats' in vlm_status and vlm_status['stats']:
            with st.expander("📊 Statistiques VLM"):
                stats = vlm_status['stats']
                st.metric("Requêtes traitées", stats.get('queries_processed', 0))
                st.metric("Vidéos analysées", stats.get('videos_analyzed', 0))
                if stats.get('average_response_time'):
                    st.metric("Temps moyen", f"{stats['average_response_time']:.2f}s")
        
        st.divider()
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
        
        # ✅ NOUVEAU: Contexte vidéo actuel
        if self.current_video_context:
            st.write("**Contexte vidéo actif:**")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Vidéo", self.current_video_context.get('video_name', 'N/A'))
            
            with col2:
                linked_time = self.current_video_context.get('linked_timestamp', '')
                if linked_time:
                    dt = datetime.fromisoformat(linked_time.replace('Z', '+00:00'))
                    time_str = dt.strftime("%H:%M:%S")
                    st.metric("Lié à", time_str)
            
            with col3:
                memory_stats = self.video_memory_system.get_system_stats()
                efficiency = memory_stats.get('memory_efficiency', 0)
                st.metric("Efficacité mémoire", f"{efficiency:.1%}")
            
            # Bouton pour désactiver contexte
            if st.button("🔄 Désactiver contexte vidéo"):
                self.current_video_context = None
                st.success("Contexte vidéo désactivé")
                st.rerun()
        
        # Alertes récentes
        alerts = self.session.get_active_alerts()
        if alerts:
            st.write("**Alertes actives:**")
            for alert in alerts[-5:]:  # 5 dernières
                level_emoji = {
                    'LOW': '🟢',
                    'MEDIUM': '🟡', 
                    'HIGH': '🟠',
                    'CRITICAL': '🔴'
                }.get(alert.get('level', 'LOW'), '🟢')
                
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
        """Obtient une réponse du VLM intégré du projet avec mémoire vidéo."""
        
        # ✅ NOUVEAU: Check si question porte sur vidéo liée
        if (self.current_video_context and 
            self._is_video_related_question(question)):
            
            # Query système mémoire vidéo
            video_id = self.current_video_context['video_id']
            conversation_history = self.session.get_chat_history()
            
            memory_response = self.video_memory_system.query_video_memory(
                video_id, question, conversation_history
            )
            
            if memory_response['confidence'] > 0.8:
                # Réponse haute confiance basée sur mémoire
                return {
                    'content': memory_response['response'],
                    'metadata': {
                        'source': 'video_memory_system',
                        'confidence': memory_response['confidence'],
                        'frames_referenced': memory_response['frames_referenced'],
                        'memory_based': True,
                        **memory_response['metadata']
                    }
                }
        
        # Fallback vers VLM service normal
        try:
            # Appel asynchrone au VLM service
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            response = loop.run_until_complete(
                self.vlm_service.process_chat_query(question, context)
            )
            
            loop.close()
            return response
            
        except Exception as e:
            # Fallback avec analyse locale
            return self._local_analysis_response(question, context)
    
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
        # Sauvegarde de la notation dans la session
        ratings = self.session.get_data('message_ratings', {})
        ratings[message_id] = {
            'rating': rating,
            'timestamp': datetime.now().isoformat()
        }
        self.session.set_data('message_ratings', ratings)
        
        rating_text = "positif" if rating == "like" else "négatif"
        st.success(f"Merci pour votre retour {rating_text} ! Cela nous aide à améliorer l'IA.")
    
    def set_vlm_callback(self, callback: Callable[[str, Dict], Dict]):
        """Définit le callback VLM."""
        self.vlm_callback = callback
    
    def link_video_analysis(self, video_id: str, analysis_result: Dict) -> str:
        """
        Lie une analyse vidéo au chat avec mémoire persistante.
        """
        
        # Stockage dans système mémoire avancé
        memory_id = self.video_memory_system.store_video_analysis(
            video_id, analysis_result
        )
        
        # Mise à jour contexte chat actuel
        self.current_video_context = {
            'video_id': video_id,
            'memory_id': memory_id,
            'video_name': analysis_result.get('video_name', 'vidéo'),
            'analysis_result': analysis_result,
            'linked_timestamp': datetime.now().isoformat()
        }
        
        # Message système avec contexte détaillé
        video_name = analysis_result.get('video_name', 'vidéo')
        duration = analysis_result.get('analysis_time', 'N/A')
        frame_count = len(analysis_result.get('detailed_frames', []))
        
        self.add_system_message(
            f"Mémoire vidéo activée: {video_name} "
            f"({duration}s, {frame_count} frames analysées). "
            f"Je peux maintenant répondre avec des détails précis sur cette vidéo.",
            level='success'
        )
        
        return memory_id
    
    def _is_video_related_question(self, question: str) -> bool:
        """
        Détecte si question porte sur vidéo analysée.
        """
        
        video_keywords = [
            'vidéo', 'video', 'voir', 'vu', 'observé', 'détecté',
            'dans cette', 'exactement', 'précis', 'détail',
            'timeline', 'moment', 'quand', 'combien'
        ]
        
        question_lower = question.lower()
        return any(keyword in question_lower for keyword in video_keywords)
    
    def add_system_message(self, content: str, level: str = "info"):
        """Ajoute un message système."""
        
        level_prefixes = {
            'info': 'ℹ️',
            'warning': '⚠️',
            'error': '❌',
            'success': '✅'
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
    
    def initialize_vlm(self):
        """Initialise le VLM si nécessaire."""
        
        if 'vlm_initialized' not in st.session_state:
            with st.spinner("🤖 Initialisation du VLM..."):
                try:
                    # Initialisation asynchrone
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    success = loop.run_until_complete(
                        self.vlm_service.initialize_vlm()
                    )
                    
                    loop.close()
                    
                    if success:
                        st.session_state.vlm_initialized = True
                        st.success("✅ VLM initialisé avec succès!")
                    else:
                        st.warning("⚠️ VLM en mode simulation")
                        
                except Exception as e:
                    st.error(f"❌ Erreur initialisation VLM: {e}")
        
        return st.session_state.get('vlm_initialized', False)
    
    def get_vlm_status(self) -> Dict[str, Any]:
        """Retourne le statut du VLM avec mémoire vidéo."""
        vlm_status = self.vlm_service.get_system_status()
        
        # ✅ NOUVEAU: Ajout stats mémoire vidéo
        memory_stats = self.video_memory_system.get_system_stats()
        vlm_status['video_memory'] = memory_stats
        
        return vlm_status
    
    def _local_analysis_response(self, question: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Génère une réponse d'analyse locale intelligente (fallback)."""
        
        # Analyse des mots-clés de la question
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['risque', 'danger', 'menace', 'sécurité']):
            response = self._analyze_security_level(context)
        elif any(word in question_lower for word in ['personne', 'gens', 'individu', 'combien']):
            response = self._analyze_people_count(context)
        elif any(word in question_lower for word in ['alerte', 'alarme', 'incident']):
            response = self._analyze_alerts(context)
        elif any(word in question_lower for word in ['état', 'statut', 'système']):
            response = self._analyze_system_status(context)
        else:
            response = self._general_analysis(context)
            
        return {
            'content': f"🔄 **Mode fallback activé**\n\n{response}",
            'metadata': {
                'analysis_time': 0.3,
                'fallback_mode': True,
                'confidence': 0.6
            }
        }
    
    def _analyze_security_level(self, context: Dict[str, Any]) -> str:
        """Analyse le niveau de sécurité."""
        alerts = context.get('active_alerts', [])
        if not alerts:
            return "🟢 **Niveau de sécurité: NORMAL**\n\nAucune alerte active détectée. Le système fonctionne normalement."
        
        critical_count = sum(1 for a in alerts if a.get('level') == 'CRITICAL')
        high_count = sum(1 for a in alerts if a.get('level') == 'HIGH')
        
        if critical_count > 0:
            return f"🔴 **Niveau de sécurité: CRITIQUE**\n\n{critical_count} alerte(s) critique(s) active(s). Intervention immédiate requise."
        elif high_count > 0:
            return f"🟠 **Niveau de sécurité: ÉLEVÉ**\n\n{high_count} alerte(s) haute priorité. Surveillance renforcée recommandée."
        else:
            return f"🟡 **Niveau de sécurité: MOYEN**\n\n{len(alerts)} alerte(s) de niveau faible à moyen."
    
    def _analyze_people_count(self, context: Dict[str, Any]) -> str:
        """Analyse le nombre de personnes."""
        video_analyses = context.get('video_analyses', {})
        if not video_analyses:
            return "👤 **Comptage des personnes**\n\nAucune analyse vidéo récente disponible pour le comptage."
        
        # Simulation d'une analyse (remplacez par votre logique réelle)
        total_people = len(video_analyses) * 2  # Exemple
        return f"👥 **Personnes détectées: {total_people}**\n\nBasé sur {len(video_analyses)} analyse(s) vidéo récente(s)."
    
    def _analyze_alerts(self, context: Dict[str, Any]) -> str:
        """Analyse les alertes."""
        alerts = context.get('active_alerts', [])
        if not alerts:
            return "✅ **Aucune alerte active**\n\nTous les systèmes fonctionnent normalement."
        
        alert_summary = []
        for alert in alerts[-5:]:
            level_emoji = {
                'LOW': '🟢', 'MEDIUM': '🟡', 
                'HIGH': '🟠', 'CRITICAL': '🔴'
            }.get(alert.get('level', 'LOW'), '🟢')
            
            alert_summary.append(f"{level_emoji} **{alert.get('level', 'N/A')}**: {alert.get('message', 'Message non disponible')}")
        
        return f"🚨 **{len(alerts)} alerte(s) active(s)**\n\n" + "\n".join(alert_summary)
    
    def _analyze_system_status(self, context: Dict[str, Any]) -> str:
        """Analyse l'état du système."""
        cameras = context.get('cameras_state', {})
        alerts = context.get('active_alerts', [])
        videos = context.get('video_analyses', {})
        
        active_cams = sum(1 for state in cameras.values() if state.get('enabled', False))
        total_cams = len(cameras)
        
        status_parts = [
            f"📹 **Caméras**: {active_cams}/{total_cams} actives",
            f"📊 **Analyses**: {len(videos)} récentes",
            f"🚨 **Alertes**: {len(alerts)} actives"
        ]
        
        overall_status = "🟢 NORMAL" if len(alerts) == 0 else "🟡 SURVEILLANCE"
        
        return f"🖥️ **État du système: {overall_status}**\n\n" + "\n".join(status_parts)
    
    def _general_analysis(self, context: Dict[str, Any]) -> str:
        """Analyse générale."""
        return f"🔍 **Analyse générale**\n\nSystème opérationnel avec:\n- {len(context.get('cameras_state', {}))} caméra(s) configurée(s)\n- {len(context.get('video_analyses', {}))} analyse(s) vidéo\n- {len(context.get('active_alerts', []))} alerte(s) active(s)\n\nPour une analyse plus précise, posez une question spécifique."

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