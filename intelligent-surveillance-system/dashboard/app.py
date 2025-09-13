"""
🔒 Dashboard de Surveillance Intelligente
========================================

Application Streamlit moderne pour la surveillance avec IA.
Architecture modulaire et optimisée.
"""

import streamlit as st
import sys
from pathlib import Path
import asyncio
from typing import Optional, Dict, Any
import atexit

# Configuration du path pour accéder aux modules du système
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Import des composants dashboard
from config.settings import get_dashboard_config, get_audio_config
from services.session_manager import get_session_manager
from components.camera_grid import get_camera_grid, cleanup_camera_resources
from components.vlm_chat import get_vlm_chat
from utils.audio_alerts import get_audio_system, play_alert

# Imports système de surveillance (si disponible)
try:
    from src.core.vlm.model import VisionLanguageModel
    from src.core.orchestrator.adaptive_orchestrator import AdaptiveOrchestrator
    VLM_AVAILABLE = True
except ImportError:
    VLM_AVAILABLE = False
    st.warning("⚠️ Modules VLM non trouvés - Mode démo activé")

class SurveillanceDashboard:
    """Dashboard principal de surveillance."""
    
    def __init__(self):
        self.config = get_dashboard_config()
        self.audio_config = get_audio_config()
        self.session = get_session_manager()
        
        # Composants principaux
        self.camera_grid = get_camera_grid()
        self.vlm_chat = get_vlm_chat()
        self.audio_system = get_audio_system()
        
        # Système de surveillance (si disponible)
        self.vlm_model: Optional[VisionLanguageModel] = None
        self.orchestrator: Optional[AdaptiveOrchestrator] = None
        
        self._setup_callbacks()
        self._init_vlm_system()
    
    def _setup_callbacks(self):
        """Configure les callbacks entre composants."""
        
        # Callback VLM pour le chat
        self.vlm_chat.set_vlm_callback(self._handle_vlm_query)
        
        # Callback détection pour les caméras
        self.camera_grid.set_detection_callback(self._handle_detection)
    
    def _init_vlm_system(self):
        """Initialise le système VLM si disponible."""
        
        if not VLM_AVAILABLE:
            return
        
        try:
            # Chargement asynchrone du VLM
            if 'vlm_system' not in st.session_state:
                st.session_state.vlm_system = {
                    'loading': False,
                    'loaded': False,
                    'model': None,
                    'orchestrator': None
                }
        
        except Exception as e:
            st.error(f"Erreur initialisation VLM: {e}")
    
    def _handle_vlm_query(self, question: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Traite une requête chat VLM."""
        
        try:
            if not VLM_AVAILABLE or not st.session_state.get('vlm_system', {}).get('loaded', False):
                # Mode démo
                return self._demo_vlm_response(question, context)
            
            # Traitement réel avec VLM
            model = st.session_state.vlm_system.get('model')
            if model:
                # TODO: Implémenter appel VLM réel
                response = f"Réponse VLM à: {question}"
                
                return {
                    'content': response,
                    'metadata': {
                        'confidence': 0.85,
                        'tools_used': ['vlm', 'context_analysis']
                    }
                }
            
        except Exception as e:
            return {
                'content': f"❌ Erreur VLM: {str(e)}",
                'metadata': {'error': True}
            }
        
        return self._demo_vlm_response(question, context)
    
    def _demo_vlm_response(self, question: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Génère une réponse démo intelligente."""
        
        question_lower = question.lower()
        
        # Analyse du contexte
        video_count = len(context.get('video_analyses', {}))
        camera_count = len(context.get('cameras_state', {}))
        alert_count = len(context.get('active_alerts', []))
        
        # Réponses contextuelles
        if any(word in question_lower for word in ['risque', 'danger', 'menace']):
            level = "FAIBLE" if alert_count == 0 else ("MOYEN" if alert_count < 3 else "ÉLEVÉ")
            response = f"🔍 **Évaluation du risque: {level}**\n\n"
            response += f"- {camera_count} caméras actives\n"
            response += f"- {alert_count} alertes en cours\n"
            response += f"- {video_count} analyses récentes\n\n"
            
            if alert_count == 0:
                response += "✅ Aucune activité suspecte détectée actuellement."
            else:
                response += "⚠️ Surveillance renforcée recommandée."
        
        elif any(word in question_lower for word in ['personnes', 'individus', 'gens']):
            response = f"👥 **Analyse des personnes détectées:**\n\n"
            response += f"Basé sur {video_count} analyses récentes:\n"
            response += "- 2-4 personnes généralement visibles\n"
            response += "- Comportement normal observé\n"
            response += "- Aucun regroupement suspect\n\n"
            response += "💡 *Utilisez l'upload vidéo pour une analyse précise.*"
        
        elif any(word in question_lower for word in ['alerte', 'alertes']):
            response = f"🚨 **État des alertes:**\n\n"
            
            if alert_count == 0:
                response += "✅ Aucune alerte active\n"
                response += "🔍 Système de surveillance opérationnel\n"
            else:
                alerts = context.get('active_alerts', [])
                for alert in alerts[-3:]:  # 3 dernières
                    level = alert.get('level', 'INFO')
                    message = alert.get('message', 'N/A')
                    emoji = {'LOW': '🔵', 'MEDIUM': '🟡', 'HIGH': '🟠', 'CRITICAL': '🔴'}.get(level, '⚪')
                    response += f"{emoji} {message}\n"
        
        elif any(word in question_lower for word in ['système', 'état', 'status']):
            response = f"🖥️ **État du système:**\n\n"
            response += f"📹 Caméras: {camera_count} configurées\n"
            response += f"🤖 IA: {'🟢 Opérationnelle' if VLM_AVAILABLE else '🟡 Mode démo'}\n"
            response += f"🔊 Audio: {'🟢 Activé' if self.audio_config.enabled else '🔴 Désactivé'}\n"
            response += f"📊 Analyses: {video_count} disponibles\n\n"
            response += "✅ Tous les systèmes fonctionnels"
        
        else:
            response = f"🤖 **Analyse de votre question:**\n\n"
            response += f"Votre question: *\"{question}\"*\n\n"
            response += f"📊 **Contexte disponible:**\n"
            response += f"- {camera_count} flux caméra actifs\n" 
            response += f"- {video_count} analyses vidéo stockées\n"
            response += f"- {alert_count} alertes en cours\n\n"
            response += "💡 *Posez des questions spécifiques sur les risques, personnes, alertes ou l'état du système.*"
        
        return {
            'content': response,
            'metadata': {
                'demo_mode': True,
                'confidence': 0.8,
                'context_items': video_count + camera_count + alert_count
            }
        }
    
    def _handle_detection(self, camera_id: str, frame) -> Dict[str, Any]:
        """Traite une détection caméra."""
        
        # Simulation détection pour démo
        import random
        
        if random.random() < 0.1:  # 10% chance de détection
            confidence = random.uniform(0.6, 0.95)
            obj_type = random.choice(['personne', 'véhicule', 'objet'])
            
            return {
                'objects': [{
                    'type': obj_type,
                    'confidence': confidence,
                    'bbox': [100, 100, 200, 200]  # Exemple
                }],
                'timestamp': st.session_state.get('current_time', 0)
            }
        
        return {}
    
    def run(self):
        """Lance le dashboard principal."""
        
        # Configuration de la page
        st.set_page_config(
            page_title=self.config.page_title,
            layout=self.config.layout,
            initial_sidebar_state=self.config.initial_sidebar_state,
            page_icon="🔒"
        )
        
        # CSS personnalisé
        self._apply_custom_css()
        
        # Sidebar de contrôle
        self._render_sidebar()
        
        # Contenu principal
        self._render_main_content()
        
        # Nettoyage à la fermeture
        atexit.register(self._cleanup)
    
    def _apply_custom_css(self):
        """Applique le CSS personnalisé."""
        
        st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(90deg, #1f4e79, #2d5aa0);
            padding: 1rem;
            border-radius: 10px;
            color: white;
            margin-bottom: 2rem;
        }
        
        .metric-container {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid #dee2e6;
        }
        
        .alert-high {
            background-color: #ffebee;
            border-left: 4px solid #f44336;
            padding: 1rem;
            border-radius: 4px;
        }
        
        .alert-medium {
            background-color: #fff8e1;
            border-left: 4px solid #ff9800;
            padding: 1rem;
            border-radius: 4px;
        }
        
        .alert-low {
            background-color: #e3f2fd;
            border-left: 4px solid #2196f3;
            padding: 1rem;
            border-radius: 4px;
        }
        
        .camera-cell {
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            padding: 0.5rem;
            margin: 0.25rem;
        }
        
        .camera-active {
            border-color: #4caf50;
        }
        
        .camera-inactive {
            border-color: #f44336;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def _render_sidebar(self):
        """Affiche la sidebar de contrôle."""
        
        with st.sidebar:
            # En-tête
            st.markdown("""
            <div class="main-header">
                <h2>🔒 Surveillance</h2>
                <p>Dashboard IA Avancé</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Contrôles globaux
            st.subheader("⚙️ Contrôles")
            
            # Seuil d'alerte
            alert_threshold = st.slider(
                "Seuil d'alerte global",
                0, 100, 
                self.session.get_user_data('alert_threshold', 70),
                help="Sensibilité des alertes automatiques"
            )
            
            if alert_threshold != self.session.get_user_data('alert_threshold', 70):
                self.session.set_user_data('alert_threshold', alert_threshold)
            
            # Audio
            audio_enabled = st.checkbox(
                "🔊 Sons d'alerte",
                self.audio_config.enabled
            )
            
            if audio_enabled != self.audio_config.enabled:
                self.audio_config.enabled = audio_enabled
            
            # Volume audio
            if audio_enabled:
                volume = st.slider(
                    "Volume",
                    0.0, 1.0,
                    self.audio_config.volume,
                    step=0.1
                )
                self.audio_config.volume = volume
            
            st.divider()
            
            # Métriques système
            self._render_system_metrics()
            
            st.divider()
            
            # Actions rapides
            st.subheader("🚀 Actions")
            
            if st.button("🧪 Test alerte", use_container_width=True):
                play_alert("MEDIUM", "Test du système d'alerte", force=True)
            
            if st.button("📊 Rapport système", use_container_width=True):
                self._generate_system_report()
            
            if st.button("🧹 Nettoyer session", use_container_width=True):
                self._cleanup_session()
                st.rerun()
    
    def _render_system_metrics(self):
        """Affiche les métriques système dans la sidebar."""
        
        st.subheader("📊 État Système")
        
        # Caméras
        camera_stats = self.camera_grid.get_camera_stats()
        active_cameras = sum(1 for stats in camera_stats.values() if stats['running'])
        
        st.metric("Caméras actives", f"{active_cameras}/{len(camera_stats)}")
        
        # Alertes
        alerts = self.session.get_active_alerts()
        critical_alerts = len([a for a in alerts if a.get('level') == 'CRITICAL'])
        
        if critical_alerts > 0:
            st.metric("⚠️ Alertes critiques", critical_alerts, delta=critical_alerts)
        else:
            st.metric("Alertes actives", len(alerts))
        
        # Analyses VLM
        analyses = len(self.session.get_all_video_analyses())
        st.metric("Analyses VLM", analyses)
        
        # Messages chat
        chat_messages = len(self.session.get_chat_history())
        st.metric("Messages chat", chat_messages)
    
    def _render_main_content(self):
        """Affiche le contenu principal."""
        
        # En-tête principal
        st.markdown("""
        <div class="main-header">
            <h1>🔒 Dashboard de Surveillance Intelligente</h1>
            <p>Surveillance automatisée avec IA - Détection comportementale avancée</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Onglets principaux
        tab1, tab2, tab3, tab4 = st.tabs([
            "📹 Surveillance Live", 
            "📱 Upload & Analyse", 
            "💬 Chat IA",
            "📊 Rapports"
        ])
        
        with tab1:
            self._render_live_surveillance()
        
        with tab2:
            self._render_video_analysis()
        
        with tab3:
            self._render_chat_interface()
        
        with tab4:
            self._render_reports()
    
    def _render_live_surveillance(self):
        """Onglet surveillance en temps réel."""
        
        st.subheader("📹 Surveillance Multi-Caméras")
        
        # Panneau de configuration des caméras
        with st.expander("➕ Configuration des Caméras", expanded=False):
            from components.camera_grid import render_camera_configuration_panel
            render_camera_configuration_panel()
        
        # Grille des caméras
        st.subheader("🎥 Flux en Direct")
        
        # Options d'affichage
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            grid_size = st.selectbox(
                "Grille d'affichage",
                ["2x2 (4 caméras)", "3x3 (9 caméras)", "4x4 (16 caméras)"],
                index=0
            )
        
        with col2:
            auto_refresh = st.checkbox("🔄 Actualisation auto", True)
            
        with col3:
            if st.button("🔄 Actualiser"):
                st.rerun()
        
        # Conversion taille grille
        grid_map = {
            "2x2 (4 caméras)": (2, 2),
            "3x3 (9 caméras)": (3, 3),
            "4x4 (16 caméras)": (4, 4)
        }
        
        selected_grid = grid_map[grid_size]
        
        # Affichage grille
        self.camera_grid.render_grid(selected_grid)
        
        # Alertes récentes
        self._render_recent_alerts()
    
    def _render_video_analysis(self):
        """Onglet analyse de vidéos uploadées."""
        
        st.subheader("📱 Analyse de Vidéo")
        
        # Upload de fichier
        uploaded_file = st.file_uploader(
            "Sélectionnez une vidéo à analyser",
            type=['mp4', 'avi', 'mov', 'mkv', 'webm'],
            help="Formats supportés: MP4, AVI, MOV, MKV, WEBM"
        )
        
        if uploaded_file is not None:
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.subheader("🎬 Vidéo uploadée")
                
                # Affichage vidéo
                st.video(uploaded_file)
                
                # Options d'analyse
                with st.expander("⚙️ Options d'analyse", expanded=True):
                    analyze_behavior = st.checkbox("🔍 Analyse comportementale", True)
                    detect_objects = st.checkbox("📦 Détection d'objets", True)
                    track_movement = st.checkbox("👥 Suivi de mouvements", True)
                    
                    sensitivity = st.slider("Sensibilité", 0.0, 1.0, 0.7)
                
                # Bouton d'analyse
                if st.button("🚀 Lancer l'analyse complète", type="primary", use_container_width=True):
                    self._analyze_uploaded_video(uploaded_file, {
                        'behavior': analyze_behavior,
                        'objects': detect_objects,
                        'movement': track_movement,
                        'sensitivity': sensitivity
                    })
            
            with col2:
                st.subheader("📊 Résultats")
                
                # Résultats d'analyse
                video_id = f"video_{hash(uploaded_file.name)}"
                analysis = self.session.get_video_analysis(video_id)
                
                if analysis:
                    # Métriques principales
                    confidence = analysis.get('confidence', 0)
                    suspicion = analysis.get('suspicion_level', 'LOW')
                    
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        st.metric("Confiance", f"{confidence:.1%}")
                    
                    with col_b:
                        suspicion_colors = {
                            'LOW': '🟢',
                            'MEDIUM': '🟡', 
                            'HIGH': '🟠',
                            'CRITICAL': '🔴'
                        }
                        st.metric("Suspicion", f"{suspicion_colors.get(suspicion, '⚪')} {suspicion}")
                    
                    # Détails de l'analyse
                    st.subheader("📋 Détails")
                    st.json(analysis)
                    
                    # Export
                    if st.button("💾 Exporter analyse", use_container_width=True):
                        self._export_analysis(analysis)
                
                else:
                    st.info("🎯 Lancez une analyse pour voir les résultats")
    
    def _render_chat_interface(self):
        """Onglet interface de chat."""
        self.vlm_chat.render_chat_interface()
    
    def _render_reports(self):
        """Onglet rapports et statistiques."""
        
        st.subheader("📊 Rapports et Statistiques")
        
        # Statistiques de session
        session_stats = self.session.get_session_stats()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Messages échangés", session_stats['chat_messages'])
        
        with col2:
            st.metric("Analyses vidéo", session_stats['video_analyses'])
        
        with col3:
            st.metric("Alertes générées", session_stats['active_alerts'])
        
        with col4:
            st.metric("Caméras configurées", session_stats['cameras_configured'])
        
        st.divider()
        
        # Export des données
        st.subheader("💾 Export des données")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📄 Exporter session (JSON)", use_container_width=True):
                data = self.session.export_session_data('json')
                st.download_button(
                    "⬇️ Télécharger JSON",
                    data,
                    file_name=f"session_{self.session.get_session_id()[:8]}.json",
                    mime="application/json"
                )
        
        with col2:
            if st.button("💬 Exporter chat (TXT)", use_container_width=True):
                data = self.vlm_chat.export_chat_history('txt')
                st.download_button(
                    "⬇️ Télécharger TXT",
                    data,
                    file_name=f"chat_{self.session.get_session_id()[:8]}.txt",
                    mime="text/plain"
                )
        
        with col3:
            if st.button("📊 Rapport PDF", use_container_width=True):
                st.info("🚧 Génération PDF en développement")
    
    def _render_recent_alerts(self):
        """Affiche les alertes récentes."""
        
        st.subheader("🚨 Alertes Récentes")
        
        alerts = self.session.get_active_alerts()
        
        if not alerts:
            st.success("✅ Aucune alerte active")
            return
        
        # Affichage des alertes
        for alert in alerts[-5:]:  # 5 dernières
            level = alert.get('level', 'LOW')
            message = alert.get('message', 'N/A')
            timestamp = alert.get('timestamp', '')
            source = alert.get('source', 'system')
            
            # Couleur selon le niveau
            if level == 'CRITICAL':
                st.error(f"🔴 **CRITIQUE** ({source}): {message}")
            elif level == 'HIGH':
                st.error(f"🟠 **ÉLEVÉ** ({source}): {message}")
            elif level == 'MEDIUM':
                st.warning(f"🟡 **MOYEN** ({source}): {message}")
            else:
                st.info(f"🔵 **FAIBLE** ({source}): {message}")
    
    def _analyze_uploaded_video(self, uploaded_file, options: Dict[str, Any]):
        """Analyse une vidéo uploadée."""
        
        with st.spinner("🔍 Analyse en cours..."):
            try:
                # Simulation d'analyse (remplacer par vraie analyse)
                import time
                import random
                
                time.sleep(2)  # Simulation du temps d'analyse
                
                # Génération résultats simulés
                analysis_result = {
                    'video_name': uploaded_file.name,
                    'file_size': uploaded_file.size,
                    'analysis_options': options,
                    'confidence': random.uniform(0.7, 0.95),
                    'suspicion_level': random.choice(['LOW', 'MEDIUM', 'HIGH']),
                    'detected_objects': [
                        {'type': 'person', 'confidence': 0.92, 'count': random.randint(1, 5)},
                        {'type': 'vehicle', 'confidence': 0.78, 'count': random.randint(0, 2)}
                    ],
                    'behaviors': [
                        {'type': 'normal_walking', 'confidence': 0.89},
                        {'type': 'loitering', 'confidence': random.uniform(0.3, 0.7)}
                    ],
                    'timeline': [
                        {'time': '00:05', 'event': 'Personne entre dans le champ'},
                        {'time': '00:12', 'event': 'Mouvement vers la droite'},
                        {'time': '00:18', 'event': 'Arrêt prolongé détecté'}
                    ],
                    'analysis_time': 2.3,
                    'timestamp': st.session_state.get('current_time', time.time())
                }
                
                # Stockage de l'analyse
                video_id = f"video_{hash(uploaded_file.name)}"
                self.session.store_video_analysis(video_id, analysis_result)
                
                # Ajout message système au chat
                suspicion = analysis_result['suspicion_level']
                confidence = analysis_result['confidence']
                
                self.vlm_chat.add_system_message(
                    f"Analyse terminée: {uploaded_file.name} - Suspicion {suspicion} ({confidence:.1%} confiance)",
                    'success' if suspicion == 'LOW' else 'warning'
                )
                
                # Alerte audio si suspicion élevée
                if suspicion in ['HIGH', 'CRITICAL']:
                    play_alert(suspicion, f"Comportement suspect détecté dans {uploaded_file.name}")
                
                st.success("✅ Analyse terminée avec succès!")
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Erreur lors de l'analyse: {str(e)}")
    
    def _export_analysis(self, analysis: Dict[str, Any]):
        """Exporte une analyse."""
        
        import json
        data = json.dumps(analysis, indent=2, ensure_ascii=False)
        
        st.download_button(
            "💾 Télécharger analyse",
            data,
            file_name=f"analyse_{analysis.get('video_name', 'video')}.json",
            mime="application/json"
        )
    
    def _generate_system_report(self):
        """Génère un rapport système."""
        
        st.info("📊 Génération du rapport système...")
        
        # TODO: Implémenter génération rapport complet
        report = {
            'timestamp': st.session_state.get('current_time', 0),
            'system_status': 'operational',
            'cameras': self.camera_grid.get_camera_stats(),
            'session': self.session.get_session_stats(),
            'audio': self.audio_system.get_status()
        }
        
        st.json(report)
    
    def _cleanup_session(self):
        """Nettoie la session."""
        
        self.session.clear_chat_history()
        self.session.clear_alerts()
        
        # Réinitialisation des caches
        self.session.cleanup_expired_data()
        
        st.success("🧹 Session nettoyée")
    
    def _cleanup(self):
        """Nettoyage à la fermeture."""
        cleanup_camera_resources()

def main():
    """Point d'entrée principal."""
    
    try:
        dashboard = SurveillanceDashboard()
        dashboard.run()
        
    except Exception as e:
        st.error(f"❌ Erreur critique: {str(e)}")
        st.error("Veuillez redémarrer l'application")

if __name__ == "__main__":
    main()