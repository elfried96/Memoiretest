"""
🔒 Dashboard Simple - Surveillance Intelligente
==============================================
Version simplifiée pour tests rapides.
"""

import streamlit as st
import json
import time
from datetime import datetime
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import random
# from streamlit_notification_center import notification_center  # Module not available

# Configuration de la page
st.set_page_config(
    page_title="🔒 Surveillance Intelligente",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🔒"
)

# Initialisation critique des variables de session (première ligne)
if 'dark_theme' not in st.session_state:
    st.session_state.dark_theme = False

# CSS personnalisé avec support thème sombre
def get_theme_css(dark_mode=False):
    if dark_mode:
        return """
        <style>
        .stApp {
            background-color: #0e1117;
            color: #fafafa;
        }
        
        .main-header {
            background: linear-gradient(90deg, #1a2332, #2d4a6b);
            padding: 1rem;
            border-radius: 10px;
            color: #fafafa;
            margin-bottom: 2rem;
            text-align: center;
            border: 1px solid #2d4a6b;
        }
        
        .metric-card {
            background: #1e2530;
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid #2d4a6b;
            margin: 0.5rem 0;
            color: #fafafa;
        }
        
        .sidebar .sidebar-content {
            background-color: #1e2530;
        }
        
        .alert-high {
            background-color: #3d1a1a;
            border-left: 4px solid #f44336;
            padding: 1rem;
            border-radius: 4px;
            color: #fafafa;
        }
        
        .alert-medium {
            background-color: #3d2f1a;
            border-left: 4px solid #ff9800;
            padding: 1rem;
            border-radius: 4px;
            color: #fafafa;
        }
        
        .alert-low {
            background-color: #1a2a3d;
            border-left: 4px solid #2196f3;
            padding: 1rem;
            border-radius: 4px;
            color: #fafafa;
        }
        
        .stSelectbox > div > div {
            background-color: #1e2530;
            color: #fafafa;
        }
        
        .stTextInput > div > div > input {
            background-color: #1e2530;
            color: #fafafa;
            border: 1px solid #2d4a6b;
        }
        
        /* Graphiques en mode sombre */
        .js-plotly-plot .plotly {
            background-color: #1e2530 !important;
        }
        </style>
        """
    else:
        return """
        <style>
        .main-header {
            background: linear-gradient(90deg, #1f4e79, #2d5aa0);
            padding: 1rem;
            border-radius: 10px;
            color: white;
            margin-bottom: 2rem;
            text-align: center;
        }
        
        .metric-card {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid #dee2e6;
            margin: 0.5rem 0;
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
        </style>
        """

# Session state initialization
if 'dark_theme' not in st.session_state:
    st.session_state.dark_theme = False

# Appliquer le CSS selon le thème
st.markdown(get_theme_css(st.session_state.dark_theme), unsafe_allow_html=True)
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'active_alerts' not in st.session_state:
    st.session_state.active_alerts = []

if 'video_analyses' not in st.session_state:
    st.session_state.video_analyses = {}

if 'cameras_state' not in st.session_state:
    st.session_state.cameras_state = {}

if 'metrics_history' not in st.session_state:
    st.session_state.metrics_history = {'timestamps': [], 'cpu': [], 'memory': [], 'alerts': []}

def add_chat_message(role: str, content: str):
    """Ajoute un message au chat."""
    message = {
        'role': role,
        'content': content,
        'timestamp': datetime.now().strftime("%H:%M:%S")
    }
    st.session_state.chat_history.append(message)


def add_alert(level: str, message: str):
    """Ajoute une alerte."""
    alert = {
        'level': level,
        'message': message,
        'timestamp': datetime.now().strftime("%H:%M:%S")
    }
    st.session_state.active_alerts.append(alert)
    
    # Garde seulement les 10 dernières alertes
    if len(st.session_state.active_alerts) > 10:
        st.session_state.active_alerts = st.session_state.active_alerts[-10:]

def update_metrics():
    """Met à jour les métriques temps réel."""
    import random as rand
    current_time = datetime.now().strftime("%H:%M:%S")
    
    # Simulation de métriques
    cpu_usage = rand.uniform(20, 80)
    memory_usage = rand.uniform(30, 70)
    alert_count = len(st.session_state.active_alerts)
    
    # Ajouter aux historiques
    st.session_state.metrics_history['timestamps'].append(current_time)
    st.session_state.metrics_history['cpu'].append(cpu_usage)
    st.session_state.metrics_history['memory'].append(memory_usage)
    st.session_state.metrics_history['alerts'].append(alert_count)
    
    # Garder seulement les 20 derniers points
    for key in st.session_state.metrics_history:
        if len(st.session_state.metrics_history[key]) > 20:
            st.session_state.metrics_history[key] = st.session_state.metrics_history[key][-20:]
    
    return cpu_usage, memory_usage, alert_count

def create_metrics_chart():
    """Crée un graphique des métriques temps réel."""
    data = st.session_state.metrics_history
    
    if len(data['timestamps']) < 2:
        return None
    
    fig = go.Figure()
    
    # CPU
    fig.add_trace(go.Scatter(
        x=data['timestamps'],
        y=data['cpu'],
        mode='lines+markers',
        name='CPU %',
        line=dict(color='#FF6B6B', width=2),
        marker=dict(size=4)
    ))
    
    # Memory
    fig.add_trace(go.Scatter(
        x=data['timestamps'],
        y=data['memory'],
        mode='lines+markers',
        name='Mémoire %',
        line=dict(color='#4ECDC4', width=2),
        marker=dict(size=4)
    ))
    
    # Alertes (échelle différente)
    fig.add_trace(go.Scatter(
        x=data['timestamps'],
        y=[x * 10 for x in data['alerts']],  # Multiplier par 10 pour visibilité
        mode='lines+markers',
        name='Alertes (x10)',
        line=dict(color='#FFD93D', width=2),
        marker=dict(size=4)
    ))
    
    fig.update_layout(
        title="📊 Métriques Temps Réel",
        xaxis_title="Temps",
        yaxis_title="Pourcentage",
        height=300,
        showlegend=True,
        plot_bgcolor='rgba(0,0,0,0)' if st.session_state.dark_theme else 'white',
        paper_bgcolor='rgba(0,0,0,0)' if st.session_state.dark_theme else 'white',
        font_color='white' if st.session_state.dark_theme else 'black'
    )
    
    fig.update_xaxes(showgrid=True, gridcolor='gray', gridwidth=0.5)
    fig.update_yaxes(showgrid=True, gridcolor='gray', gridwidth=0.5)
    
    return fig

def simulate_vlm_response(question: str) -> str:
    """Génère une réponse VLM simulée."""
    
    question_lower = question.lower()
    
    if any(word in question_lower for word in ['risque', 'danger', 'sécurité']):
        return "🔍 **Évaluation du risque: FAIBLE**\n\nSystème de surveillance opérationnel. Aucune activité suspecte détectée actuellement."
    
    elif any(word in question_lower for word in ['personnes', 'individus']):
        return "👥 **Analyse des personnes:**\n\n2-3 personnes détectées dans les zones surveillées. Comportements normaux observés."
    
    elif any(word in question_lower for word in ['alerte', 'alertes']):
        alert_count = len(st.session_state.active_alerts)
        if alert_count == 0:
            return "✅ **Aucune alerte active**\n\nTous les systèmes fonctionnent normalement."
        else:
            return f"🚨 **{alert_count} alerte(s) active(s)**\n\nSurveillance en cours."
    
    elif any(word in question_lower for word in ['système', 'état', 'status']):
        return "🖥️ **État du système:**\n\n✅ IA: Opérationnelle (mode démo)\n📹 Caméras: Configurables\n🔊 Audio: Activé\n📊 Dashboard: Fonctionnel"
    
    else:
        return f"🤖 **Réponse à votre question:**\n\n*\"{question}\"*\n\nJe suis votre assistant IA de surveillance en mode démo. Posez-moi des questions sur les risques, personnes, alertes ou l'état du système."

def main():
    """Application principale."""
    
    # En-tête principal
    st.markdown("""
    <div class="main-header">
        <h1>🔒 Dashboard de Surveillance Intelligente</h1>
        <p>Interface moderne avec IA - Mode Démo Fonctionnel</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Contrôles")
        
        # Thème
        col_theme1, col_theme2 = st.columns(2)
        with col_theme1:
            if st.button("☀️ Clair", use_container_width=True, disabled=not st.session_state.dark_theme):
                st.session_state.dark_theme = False
                st.rerun()
        with col_theme2:
            if st.button("🌙 Sombre", use_container_width=True, disabled=st.session_state.dark_theme):
                st.session_state.dark_theme = True
                st.rerun()
        
        st.divider()
        
        # Paramètres
        alert_threshold = st.slider("Seuil d'alerte", 0, 100, 70)
        audio_enabled = st.checkbox("🔊 Sons d'alerte", True)
        
        if audio_enabled:
            volume = st.slider("Volume", 0.0, 1.0, 0.8)
        
        st.divider()
        
        # Métriques temps réel
        st.markdown("### 📊 Métriques Temps Réel")
        
        # Update métriques
        cpu_usage, memory_usage, alert_count = update_metrics()
        
        # Métriques principales
        col1, col2 = st.columns(2)
        with col1:
            import random as rand
            st.metric("🖥️ CPU", f"{cpu_usage:.1f}%", delta=f"{rand.uniform(-5, 5):.1f}%")
            st.metric("💾 RAM", f"{memory_usage:.1f}%", delta=f"{rand.uniform(-3, 3):.1f}%")
        with col2:
            st.metric("🚨 Alertes", alert_count)
            st.metric("📹 Caméras", len(st.session_state.cameras_state))
        
        # Graphique temps réel
        fig = create_metrics_chart()
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # Auto-refresh
        auto_refresh = st.checkbox("🔄 Auto-refresh (5s)", value=True)
        if auto_refresh:
            time.sleep(0.1)  # Petit délai pour éviter le spam
            st.rerun()
        
        st.divider()
        
        # État système
        st.markdown("### 🏷️ État Système")
        
        cameras_count = len(st.session_state.cameras_state)
        analyses_count = len(st.session_state.video_analyses)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Analyses", analyses_count)
        with col2:
            st.metric("Messages", len(st.session_state.chat_history))
        
        st.divider()
        
        # Actions rapides
        st.markdown("### 🚀 Actions")
        
        if st.button("🧪 Test alerte", use_container_width=True):
            add_alert("MEDIUM", "Test du système d'alerte")
            
            # Son d'alerte pour le test rapide
            if audio_enabled and st.session_state.get('audio_activated', False):
                # Son via contexte global activé
                st.markdown(f"""
                <script>
                    if (window.audioContext) {{
                        try {{
                            const osc = window.audioContext.createOscillator();
                            const gain = window.audioContext.createGain();
                            osc.connect(gain);
                            gain.connect(window.audioContext.destination);
                            osc.frequency.value = 800;
                            osc.type = 'sine';
                            gain.gain.setValueAtTime({volume * 0.4}, window.audioContext.currentTime);
                            gain.gain.exponentialRampToValueAtTime(0.01, window.audioContext.currentTime + 0.5);
                            osc.start();
                            osc.stop(window.audioContext.currentTime + 0.5);
                            console.log('🔊 Son test généré');
                        }} catch(e) {{
                            console.log('Erreur son test:', e);
                        }}
                    }}
                </script>
                """, unsafe_allow_html=True)
                st.success("🔊 Alerte test + son généré!")
            else:
                st.success("🔊 Alerte test générée!")
            
            st.rerun()
        
        # Notifications audio avec composant
        if audio_enabled:
            st.markdown("### 🔊 Notifications Audio")
            
            # Centre de notifications (disabled - module not available)
            # notifications = notification_center()
            
            col_audio1, col_audio2 = st.columns(2)
            
            with col_audio1:
                if st.button("🔊 Test Son", use_container_width=True):
                    # notifications.create_notification(
                    #     title="🔔 Test Audio",
                    #     body="Test du système de notification audio",
                    #     sound=True
                    # )
                    st.success("Notification envoyée! (simulation)")
                
                if st.button("🚨 Alerte Test", use_container_width=True):
                    # notifications.create_notification(
                    #     title="🚨 ALERTE",
                    #     body="Test d'alerte de sécurité",
                    #     sound=True,
                    #     priority="high"
                    # )
                    st.success("Alerte envoyée! (simulation)")
            
            with col_audio2:
                if st.button("📢 Info", use_container_width=True):
                    # notifications.create_notification(
                    #     title="📢 Information",
                    #     body="Notification d'information",
                    #     sound=True,
                    #     priority="normal"
                    # )
                    st.success("Info envoyée! (simulation)")
                
                if st.button("⚠️ Attention", use_container_width=True):
                    # notifications.create_notification(
                    #     title="⚠️ ATTENTION",
                    #     body="Notification d'avertissement",
                    #     sound=True,
                    #     priority="medium"
                    # )
                    st.success("Avertissement envoyé!")
            
            st.info("💡 **Notifications avec son intégré** - Autorisez les notifications de votre navigateur pour entendre les sons.")
        
        if st.button("🚨 Simuler Alerte", use_container_width=True):
            import random
            alert_types = [
                ("LOW", "Mouvement détecté zone A"),
                ("MEDIUM", "Activité suspecte détectée"),
                ("HIGH", "Intrusion possible zone sécurisée"),
                ("CRITICAL", "Alerte de sécurité maximale!")
            ]
            level, message = random.choice(alert_types)
            add_alert(level, message)
            st.rerun()
        
        if st.button("🧹 Nettoyer", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.active_alerts = []
            st.success("Session nettoyée!")
            st.rerun()
    
    # Onglets principaux - Chat intégré
    tab1, tab2, tab3 = st.tabs([
        "📹 Surveillance Live + Chat", 
        "📱 Upload & Analyse + Chat", 
        "📊 Rapports"
    ])
    
    with tab1:
        st.subheader("📹 Surveillance Multi-Caméras")
        
        # Configuration caméras
        with st.expander("➕ Ajouter une caméra", expanded=False):
            with st.form("add_camera"):
                col1, col2 = st.columns(2)
                
                with col1:
                    name = st.text_input("Nom", placeholder="Caméra Entrée")
                    source = st.text_input("Source", placeholder="0 ou rtsp://...")
                
                with col2:
                    resolution = st.selectbox("Résolution", ["640x480", "1280x720", "1920x1080"])
                    fps = st.number_input("FPS", 1, 60, 30)
                
                if st.form_submit_button("🎥 Ajouter"):
                    if name and source:
                        camera_id = f"cam_{len(st.session_state.cameras_state)}"
                        st.session_state.cameras_state[camera_id] = {
                            'name': name,
                            'source': source,
                            'resolution': resolution,
                            'fps': fps,
                            'enabled': True
                        }
                        st.success(f"Caméra {name} ajoutée!")
                        st.rerun()
        
        # Grille caméras
        if st.session_state.cameras_state:
            st.subheader("🎥 Flux en Direct")
            
            # Options grille
            grid_option = st.selectbox("Grille", ["2x2", "3x3", "4x4"])
            
            cameras = list(st.session_state.cameras_state.items())
            grid_size = int(grid_option[0])
            
            # Affichage grille
            for row in range(grid_size):
                cols = st.columns(grid_size)
                for col in range(grid_size):
                    camera_idx = row * grid_size + col
                    
                    if camera_idx < len(cameras):
                        camera_id, camera_config = cameras[camera_idx]
                        
                        with cols[col]:
                            st.markdown(f"### 📹 {camera_config['name']}")
                            
                            # Statut simulation
                            if camera_config['enabled']:
                                st.success("🟢 Actif - 30 FPS")
                            else:
                                st.error("🔴 Inactif")
                            
                            # Placeholder pour vidéo
                            st.info("📷 Flux vidéo ici\n(Mode démo)")
                            
                            # Contrôles
                            sensitivity = st.slider(
                                f"Sensibilité {camera_config['name']}", 
                                0.0, 1.0, 0.5,
                                key=f"sens_{camera_id}"
                            )
                            
                            if st.button(f"❌ Supprimer", key=f"del_{camera_id}"):
                                del st.session_state.cameras_state[camera_id]
                                st.rerun()
        else:
            st.info("🎥 Aucune caméra configurée. Ajoutez-en une ci-dessus!")
        
        # Chat intégré pour surveillance
        st.subheader("💬 Questions sur la Surveillance")
        
        col_chat1, col_chat2 = st.columns([2, 1])
        
        with col_chat1:
            # Questions rapides surveillance
            st.write("**Questions rapides:**")
            col_q1, col_q2 = st.columns(2)
            
            with col_q1:
                if st.button("🔍 État des caméras ?", use_container_width=True):
                    add_chat_message("user", "Quel est l'état actuel des caméras ?")
                    response = f"📹 **État des caméras:**\n\n{len(st.session_state.cameras_state)} caméras configurées.\n"
                    for cam_id, cam in st.session_state.cameras_state.items():
                        status = "🟢 Actif" if cam.get('enabled', False) else "🔴 Inactif"
                        response += f"- **{cam.get('name', cam_id)}**: {status}\n"
                    if not st.session_state.cameras_state:
                        response += "Aucune caméra configurée pour le moment."
                    add_chat_message("assistant", response)
                    st.rerun()
                
                if st.button("🚨 Niveau de risque ?", use_container_width=True):
                    add_chat_message("user", "Quel est le niveau de risque actuel ?")
                    alert_count = len(st.session_state.active_alerts)
                    if alert_count == 0:
                        response = "🔒 **Niveau de risque: FAIBLE**\n\nAucune alerte active. Surveillance normale."
                    else:
                        response = f"⚠️ **Niveau de risque: ÉLEVÉ**\n\n{alert_count} alertes actives nécessitent votre attention."
                    add_chat_message("assistant", response)
                    st.rerun()
            
            with col_q2:
                if st.button("👥 Activité détectée ?", use_container_width=True):
                    add_chat_message("user", "Y a-t-il de l'activité suspecte ?")
                    response = "👥 **Analyse d'activité:**\n\nBasé sur la surveillance en cours:\n- Mouvements normaux détectés\n- Aucun comportement suspect identifié\n- Zones surveillées: opérationnelles"
                    add_chat_message("assistant", response)
                    st.rerun()
                
                if st.button("📊 Résumé système", use_container_width=True):
                    add_chat_message("user", "Donnez-moi un résumé du système")
                    response = f"🖥️ **Résumé système:**\n\n📹 Caméras: {len(st.session_state.cameras_state)}\n🚨 Alertes: {len(st.session_state.active_alerts)}\n💬 Messages: {len(st.session_state.chat_history)}\n✅ Statut: Opérationnel"
                    add_chat_message("assistant", response)
                    st.rerun()
            
            # Input libre
            user_input_live = st.text_input("💬 Posez votre question sur la surveillance:", key="chat_live")
            if st.button("➤ Envoyer", key="send_live") and user_input_live:
                add_chat_message("user", user_input_live)
                response = simulate_vlm_response(user_input_live)
                add_chat_message("assistant", response)
                st.rerun()
        
        with col_chat2:
            st.write("**💬 Chat récent:**")
            # Affiche les 3 derniers messages
            recent_messages = st.session_state.chat_history[-6:] if st.session_state.chat_history else []
            
            for msg in recent_messages:
                if msg['role'] == 'user':
                    st.info(f"**Vous:** {msg['content'][:80]}{'...' if len(msg['content']) > 80 else ''}")
                else:
                    st.success(f"**IA:** {msg['content'][:80]}{'...' if len(msg['content']) > 80 else ''}")
            
            if not recent_messages:
                st.write("_Aucun message encore_")
        
        # Alertes récentes
        if st.session_state.active_alerts:
            st.subheader("🚨 Alertes Récentes")
            
            for alert in st.session_state.active_alerts[-3:]:
                level = alert['level']
                message = alert['message']
                timestamp = alert['timestamp']
                
                if level == 'CRITICAL':
                    st.error(f"🔴 **CRITIQUE** ({timestamp}): {message}")
                elif level == 'HIGH':
                    st.error(f"🟠 **ÉLEVÉ** ({timestamp}): {message}")
                elif level == 'MEDIUM':
                    st.warning(f"🟡 **MOYEN** ({timestamp}): {message}")
                else:
                    st.info(f"🔵 **FAIBLE** ({timestamp}): {message}")
    
    with tab2:
        st.subheader("📱 Analyse de Vidéo")
        
        uploaded_file = st.file_uploader(
            "Sélectionnez une vidéo à analyser",
            type=['mp4', 'avi', 'mov', 'mkv']
        )
        
        if uploaded_file:
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.subheader("🎬 Vidéo")
                st.video(uploaded_file)
                
                with st.expander("⚙️ Options d'analyse"):
                    behavior_analysis = st.checkbox("🔍 Analyse comportementale", True)
                    object_detection = st.checkbox("📦 Détection d'objets", True)
                    movement_tracking = st.checkbox("👥 Suivi de mouvements", True)
                    sensitivity = st.slider("Sensibilité", 0.0, 1.0, 0.7)
                
                if st.button("🚀 Analyser", type="primary", use_container_width=True):
                    with st.spinner("🔍 Analyse en cours..."):
                        time.sleep(2)  # Simulation
                        
                        # Résultats simulés
                        analysis = {
                            'video_name': uploaded_file.name,
                            'confidence': np.random.uniform(0.7, 0.95),
                            'suspicion_level': np.random.choice(['LOW', 'MEDIUM', 'HIGH']),
                            'detected_objects': [
                                {'type': 'person', 'confidence': 0.92, 'count': 2},
                                {'type': 'vehicle', 'confidence': 0.78, 'count': 1}
                            ],
                            'timeline': [
                                {'time': '00:05', 'event': 'Personne entre dans le champ'},
                                {'time': '00:12', 'event': 'Mouvement vers la droite'},
                                {'time': '00:18', 'event': 'Véhicule détecté'}
                            ]
                        }
                        
                        video_id = f"video_{hash(uploaded_file.name)}"
                        st.session_state.video_analyses[video_id] = analysis
                        
                        st.success("✅ Analyse terminée!")
                        st.rerun()
            
            with col2:
                st.subheader("📊 Résultats")
                
                video_id = f"video_{hash(uploaded_file.name)}"
                if video_id in st.session_state.video_analyses:
                    analysis = st.session_state.video_analyses[video_id]
                    
                    # Métriques
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Confiance", f"{analysis['confidence']:.1%}")
                    with col_b:
                        suspicion_colors = {
                            'LOW': '🟢', 'MEDIUM': '🟡', 'HIGH': '🟠', 'CRITICAL': '🔴'
                        }
                        color = suspicion_colors.get(analysis['suspicion_level'], '⚪')
                        st.metric("Suspicion", f"{color} {analysis['suspicion_level']}")
                    
                    # Objets détectés
                    st.subheader("🎯 Objets détectés")
                    for obj in analysis['detected_objects']:
                        st.write(f"- **{obj['type']}**: {obj['confidence']:.1%} ({obj['count']})")
                    
                    # Timeline
                    st.subheader("⏱️ Timeline")
                    for event in analysis['timeline']:
                        st.write(f"**{event['time']}**: {event['event']}")
                    
                    # Export
                    if st.button("💾 Exporter", use_container_width=True):
                        json_data = json.dumps(analysis, indent=2)
                        st.download_button(
                            "📄 Télécharger JSON",
                            json_data,
                            f"analyse_{uploaded_file.name}.json",
                            "application/json"
                        )
                else:
                    st.info("🎯 Lancez une analyse pour voir les résultats")
        
        # Chat intégré pour analyse vidéo
        st.subheader("💬 Questions sur l'Analyse")
        
        col_chat_v1, col_chat_v2 = st.columns([2, 1])
        
        with col_chat_v1:
            # Questions spécifiques à l'analyse
            st.write("**Questions sur l'analyse vidéo:**")
            col_qv1, col_qv2 = st.columns(2)
            
            with col_qv1:
                if st.button("🎬 Que contient la vidéo ?", use_container_width=True):
                    add_chat_message("user", "Que contient la dernière vidéo analysée ?")
                    if st.session_state.video_analyses:
                        latest_analysis = list(st.session_state.video_analyses.values())[-1]
                        response = f"🎬 **Analyse de la vidéo:**\n\n"
                        response += f"- **Confiance:** {latest_analysis.get('confidence', 0):.1%}\n"
                        response += f"- **Suspicion:** {latest_analysis.get('suspicion_level', 'N/A')}\n"
                        response += f"- **Objets détectés:** {len(latest_analysis.get('detected_objects', []))}\n"
                        response += f"- **Événements:** {len(latest_analysis.get('timeline', []))}"
                    else:
                        response = "Aucune vidéo analysée pour le moment. Uploadez une vidéo pour commencer l'analyse."
                    add_chat_message("assistant", response)
                    st.rerun()
                
                if st.button("🔍 Comportements suspects ?", use_container_width=True):
                    add_chat_message("user", "Y a-t-il des comportements suspects dans la vidéo ?")
                    if st.session_state.video_analyses:
                        latest_analysis = list(st.session_state.video_analyses.values())[-1]
                        suspicion = latest_analysis.get('suspicion_level', 'LOW')
                        if suspicion in ['HIGH', 'CRITICAL']:
                            response = f"⚠️ **Attention:** Niveau de suspicion {suspicion}\n\nComportements détectés nécessitant une vérification manuelle."
                        else:
                            response = f"✅ **Rassant:** Niveau de suspicion {suspicion}\n\nAucun comportement particulièrement suspect détecté."
                    else:
                        response = "Aucune analyse disponible pour évaluer les comportements."
                    add_chat_message("assistant", response)
                    st.rerun()
            
            with col_qv2:
                if st.button("👥 Combien de personnes ?", use_container_width=True):
                    add_chat_message("user", "Combien de personnes sont détectées ?")
                    if st.session_state.video_analyses:
                        latest_analysis = list(st.session_state.video_analyses.values())[-1]
                        objects = latest_analysis.get('detected_objects', [])
                        people_count = 0
                        for obj in objects:
                            if obj.get('type') == 'person':
                                people_count = obj.get('count', 0)
                                break
                        response = f"👥 **Personnes détectées:** {people_count}\n\nBasé sur l'analyse de la dernière vidéo uploadée."
                    else:
                        response = "Aucune analyse disponible. Uploadez une vidéo pour détecter les personnes."
                    add_chat_message("assistant", response)
                    st.rerun()
                
                if st.button("⏱️ Timeline des événements", use_container_width=True):
                    add_chat_message("user", "Quelle est la timeline des événements ?")
                    if st.session_state.video_analyses:
                        latest_analysis = list(st.session_state.video_analyses.values())[-1]
                        timeline = latest_analysis.get('timeline', [])
                        if timeline:
                            response = "⏱️ **Timeline des événements:**\n\n"
                            for event in timeline:
                                response += f"- **{event.get('time', 'N/A')}**: {event.get('event', 'N/A')}\n"
                        else:
                            response = "Aucun événement spécifique détecté dans la timeline."
                    else:
                        response = "Aucune analyse disponible pour la timeline."
                    add_chat_message("assistant", response)
                    st.rerun()
            
            # Input libre pour analyse
            user_input_video = st.text_input("💬 Questions sur l'analyse vidéo:", key="chat_video")
            if st.button("➤ Envoyer", key="send_video") and user_input_video:
                add_chat_message("user", user_input_video)
                response = simulate_vlm_response(user_input_video)
                add_chat_message("assistant", response)
                st.rerun()
        
        with col_chat_v2:
            st.write("**💬 Chat récent:**")
            # Affiche les derniers messages (mêmes que surveillance)
            recent_messages = st.session_state.chat_history[-6:] if st.session_state.chat_history else []
            
            for msg in recent_messages:
                if msg['role'] == 'user':
                    st.info(f"**Vous:** {msg['content'][:80]}{'...' if len(msg['content']) > 80 else ''}")
                else:
                    st.success(f"**IA:** {msg['content'][:80]}{'...' if len(msg['content']) > 80 else ''}")
            
            if not recent_messages:
                st.write("_Aucun message encore_")
    
    with tab3:
        st.subheader("📊 Rapports et Statistiques")
        
        # Statistiques générales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Messages Chat", len(st.session_state.chat_history))
        with col2:
            st.metric("Analyses Vidéo", len(st.session_state.video_analyses))
        with col3:
            st.metric("Alertes Générées", len(st.session_state.active_alerts))
        with col4:
            st.metric("Caméras Configurées", len(st.session_state.cameras_state))
        
        st.divider()
        
        # Graphiques
        if st.session_state.active_alerts:
            st.subheader("📈 Distribution des Alertes")
            
            alert_levels = [alert['level'] for alert in st.session_state.active_alerts]
            alert_counts = pd.Series(alert_levels).value_counts()
            
            st.bar_chart(alert_counts)
        
        # Export session
        st.subheader("💾 Export des Données")
        
        session_data = {
            'timestamp': datetime.now().isoformat(),
            'chat_history': st.session_state.chat_history,
            'active_alerts': st.session_state.active_alerts,
            'video_analyses': st.session_state.video_analyses,
            'cameras_state': st.session_state.cameras_state
        }
        
        if st.button("📄 Exporter Session Complète", use_container_width=True):
            json_data = json.dumps(session_data, indent=2)
            st.download_button(
                "⬇️ Télécharger Session JSON",
                json_data,
                f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                "application/json"
            )

if __name__ == "__main__":
    main()