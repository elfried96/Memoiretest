"""
🔒 Dashboard de Surveillance Fonctionnel
=======================================

Version fonctionnelle avec tous les composants visibles :
- Grille multi-caméras
- Chat VLM 
- Tableau de bord analytics
- Alertes temps réel
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import random
import json
from typing import Dict, List, Any, Optional

# Configuration de la page
st.set_page_config(
    page_title="🔒 Surveillance Intelligente",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🔒"
)

# Initialisation des variables de session
if 'cameras' not in st.session_state:
    st.session_state.cameras = {}
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'detections' not in st.session_state:
    st.session_state.detections = []
if 'alerts' not in st.session_state:
    st.session_state.alerts = []
if 'surveillance_active' not in st.session_state:
    st.session_state.surveillance_active = False

# CSS pour l'interface
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #1e3c72, #2a5298);
    padding: 1.5rem;
    border-radius: 10px;
    text-align: center;
    color: white;
    margin-bottom: 1rem;
}

.camera-card {
    border: 2px solid #dee2e6;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
    background: #f8f9fa;
}

.alert-critical {
    background: #dc3545;
    color: white;
    padding: 0.5rem;
    border-radius: 5px;
    margin: 0.25rem 0;
}

.alert-high {
    background: #fd7e14;
    color: white;
    padding: 0.5rem;
    border-radius: 5px;
    margin: 0.25rem 0;
}

.alert-medium {
    background: #ffc107;
    color: black;
    padding: 0.5rem;
    border-radius: 5px;
    margin: 0.25rem 0;
}

.chat-user {
    background: #e3f2fd;
    padding: 0.5rem;
    border-radius: 5px;
    margin: 0.25rem 0;
    text-align: right;
}

.chat-ai {
    background: #f3e5f5;
    padding: 0.5rem;
    border-radius: 5px;
    margin: 0.25rem 0;
}

.metric-card {
    background: white;
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 1rem;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

def render_header():
    """Affiche l'en-tête principal."""
    st.markdown("""
    <div class="main-header">
        <h1>🔒 Dashboard de Surveillance Intelligente</h1>
        <p>Système VLM Multi-Caméras Temps Réel</p>
    </div>
    """, unsafe_allow_html=True)

def generate_dummy_frame(camera_id: str, width: int = 320, height: int = 240):
    """Génère une frame simulée pour demo."""
    # Image de base
    img = np.random.randint(50, 200, (height, width, 3), dtype=np.uint8)
    
    # Ajout du nom de la caméra
    cv2.putText(img, f"Camera {camera_id}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Ajout de l'heure
    timestamp = datetime.now().strftime("%H:%M:%S")
    cv2.putText(img, timestamp, (10, height - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Simulation de détection (rectangle aléatoire)
    if random.random() > 0.7:
        x1, y1 = random.randint(10, width//2), random.randint(10, height//2)
        x2, y2 = x1 + 60, y1 + 80
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, "Person", (x1, y1-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    return img

def render_camera_grid():
    """Affiche la grille des caméras."""
    st.subheader("🎥 Flux Caméras Temps Réel")
    
    if not st.session_state.cameras:
        st.info("📹 Aucune caméra configurée. Ajoutez une caméra dans la section Configuration.")
        return
    
    # Organisation en grille
    cameras = list(st.session_state.cameras.values())
    
    if len(cameras) == 1:
        cols = st.columns(1)
    elif len(cameras) <= 2:
        cols = st.columns(2)
    elif len(cameras) <= 4:
        cols = st.columns(2)
    else:
        cols = st.columns(3)
    
    for i, camera in enumerate(cameras):
        with cols[i % len(cols)]:
            st.markdown(f"""
            <div class="camera-card">
                <h4>📹 {camera['name']}</h4>
            </div>
            """, unsafe_allow_html=True)
            
            # Génération de frame simulée
            if camera.get('active', False) and st.session_state.surveillance_active:
                frame = generate_dummy_frame(camera['id'])
                st.image(frame, channels="BGR", caption=f"Live Feed - {camera['name']}")
                
                # Boutons de contrôle
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("⏸️ Pause", key=f"pause_{camera['id']}"):
                        st.session_state.cameras[camera['id']]['active'] = False
                        st.rerun()
                
                with col2:
                    if st.button("⚙️ Config", key=f"config_{camera['id']}"):
                        st.info(f"Configuration de {camera['name']}")
            
            else:
                st.image("https://via.placeholder.com/320x240/cccccc/666666?text=Camera+Offline", 
                        caption=f"Caméra hors ligne - {camera['name']}")
                
                if st.button("▶️ Start", key=f"start_{camera['id']}"):
                    st.session_state.cameras[camera['id']]['active'] = True
                    st.rerun()

def render_camera_config():
    """Configuration des caméras."""
    st.subheader("📹 Configuration des Caméras")
    
    with st.expander("➕ Ajouter une nouvelle caméra", expanded=len(st.session_state.cameras) == 0):
        col1, col2 = st.columns(2)
        
        with col1:
            cam_name = st.text_input("Nom de la caméra", f"Caméra {len(st.session_state.cameras) + 1}")
            cam_source = st.selectbox("Source", ["Webcam (0)", "RTSP URL", "Fichier vidéo"])
        
        with col2:
            cam_resolution = st.selectbox("Résolution", ["640x480", "1280x720", "1920x1080"])
            cam_fps = st.slider("FPS", 15, 60, 30)
        
        if cam_source == "RTSP URL":
            rtsp_url = st.text_input("URL RTSP", "rtsp://192.168.1.100:554/stream")
        elif cam_source == "Fichier vidéo":
            video_file = st.file_uploader("Sélectionner vidéo", type=['mp4', 'avi', 'mov'])
        
        detection_sensitivity = st.slider("Sensibilité détection", 0.1, 1.0, 0.7)
        
        if st.button("➕ Ajouter Caméra"):
            camera_id = f"cam_{len(st.session_state.cameras) + 1}"
            st.session_state.cameras[camera_id] = {
                'id': camera_id,
                'name': cam_name,
                'source': cam_source,
                'resolution': cam_resolution,
                'fps': cam_fps,
                'sensitivity': detection_sensitivity,
                'active': False,
                'created': datetime.now()
            }
            st.success(f"✅ Caméra '{cam_name}' ajoutée avec succès!")
            st.rerun()
    
    # Liste des caméras existantes
    if st.session_state.cameras:
        st.subheader("📋 Caméras Configurées")
        
        for camera_id, camera in st.session_state.cameras.items():
            with st.expander(f"📹 {camera['name']} ({camera_id})"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write(f"**Source:** {camera['source']}")
                    st.write(f"**Résolution:** {camera['resolution']}")
                
                with col2:
                    st.write(f"**FPS:** {camera['fps']}")
                    st.write(f"**Sensibilité:** {camera['sensitivity']}")
                
                with col3:
                    status = "🟢 Active" if camera.get('active') else "⭕ Inactive"
                    st.write(f"**Statut:** {status}")
                    
                    if st.button("🗑️ Supprimer", key=f"delete_{camera_id}"):
                        del st.session_state.cameras[camera_id]
                        st.success(f"Caméra {camera['name']} supprimée")
                        st.rerun()

def render_vlm_chat():
    """Interface de chat VLM."""
    st.subheader("💬 Chat avec l'IA de Surveillance")
    
    # Affichage de l'historique
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history[-10:]:  # 10 derniers messages
            if message['role'] == 'user':
                st.markdown(f"""
                <div class="chat-user">
                    <strong>👤 Vous:</strong> {message['content']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-ai">
                    <strong>🤖 IA:</strong> {message['content']}
                </div>
                """, unsafe_allow_html=True)
    
    # Questions prédéfinies
    st.subheader("❓ Questions Rapides")
    questions = [
        "Que se passe-t-il sur les caméras en ce moment ?",
        "Y a-t-il des activités suspectes détectées ?",
        "Résume les détections des 10 dernières minutes",
        "Quelles sont les zones les plus actives ?",
        "Génère un rapport de surveillance"
    ]
    
    selected_question = st.selectbox("Sélectionner une question", [""] + questions)
    
    if selected_question:
        if st.button("📤 Envoyer Question"):
            # Ajouter la question
            st.session_state.chat_history.append({
                'role': 'user',
                'content': selected_question,
                'timestamp': datetime.now()
            })
            
            # Générer une réponse simulée
            responses = [
                f"🔍 Analyse en cours... Je détecte actuellement {len(st.session_state.cameras)} caméras configurées.",
                f"📊 Basé sur les données récentes, j'observe une activité normale avec {random.randint(0, 5)} détections dans la dernière heure.",
                f"🚨 Aucune activité suspecte majeure détectée. Le système surveille {len([c for c in st.session_state.cameras.values() if c.get('active')])} caméras actives.",
                f"📈 Zone d'activité principale : Entrée principale. Confiance moyenne : {random.randint(85, 95)}%",
                f"📋 Rapport généré : {random.randint(10, 50)} détections totales, {random.randint(0, 3)} alertes de niveau moyen."
            ]
            
            ai_response = random.choice(responses)
            
            st.session_state.chat_history.append({
                'role': 'assistant', 
                'content': ai_response,
                'timestamp': datetime.now()
            })
            
            st.rerun()
    
    # Zone de saisie libre
    st.subheader("✏️ Question Personnalisée")
    user_input = st.text_input("Posez votre question à l'IA...")
    
    if st.button("📤 Envoyer") and user_input:
        # Ajouter la question
        st.session_state.chat_history.append({
            'role': 'user',
            'content': user_input,
            'timestamp': datetime.now()
        })
        
        # Réponse simulée basée sur le contenu
        if "alerte" in user_input.lower():
            response = f"🚨 Actuellement {len(st.session_state.alerts)} alertes actives. Niveau moyen de suspicion."
        elif "caméra" in user_input.lower():
            response = f"📹 {len(st.session_state.cameras)} caméras configurées, {len([c for c in st.session_state.cameras.values() if c.get('active')])} actives."
        elif "rapport" in user_input.lower():
            response = f"📊 Rapport généré : Surveillance active depuis {datetime.now().strftime('%H:%M')}, {random.randint(15, 45)} détections analysées."
        else:
            response = f"🤖 J'ai analysé votre question. Basé sur les données actuelles du système de surveillance, voici ma réponse adaptée à votre demande."
        
        st.session_state.chat_history.append({
            'role': 'assistant',
            'content': response, 
            'timestamp': datetime.now()
        })
        
        st.rerun()

def render_analytics_dashboard():
    """Tableau de bord analytique."""
    st.subheader("📊 Analytics & Métriques")
    
    # Métriques principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_cameras = len(st.session_state.cameras)
        st.markdown(f"""
        <div class="metric-card">
            <h2>{total_cameras}</h2>
            <p>Caméras Total</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        active_cameras = len([c for c in st.session_state.cameras.values() if c.get('active')])
        st.markdown(f"""
        <div class="metric-card">
            <h2>{active_cameras}</h2>
            <p>Caméras Actives</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        total_detections = len(st.session_state.detections)
        st.markdown(f"""
        <div class="metric-card">
            <h2>{total_detections}</h2>
            <p>Détections Total</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        active_alerts = len(st.session_state.alerts)
        st.markdown(f"""
        <div class="metric-card">
            <h2>{active_alerts}</h2>
            <p>Alertes Actives</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Graphiques
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Détections par Heure")
        
        # Génération de données simulées
        hours = [f"{i:02d}:00" for i in range(24)]
        detections = [random.randint(0, 20) for _ in hours]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hours,
            y=detections,
            mode='lines+markers',
            name='Détections',
            line=dict(color='#1f77b4', width=2)
        ))
        
        fig.update_layout(
            title="Activité sur 24h",
            xaxis_title="Heure",
            yaxis_title="Nombre de détections",
            height=300
        )
        
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        st.subheader("🎯 Types de Détections")
        
        detection_types = ['Personne', 'Véhicule', 'Mouvement Suspect', 'Objet Abandonné']
        detection_counts = [random.randint(5, 30) for _ in detection_types]
        
        fig = go.Figure(data=[go.Pie(
            labels=detection_types,
            values=detection_counts,
            hole=0.4
        )])
        
        fig.update_layout(
            title="Répartition des détections",
            height=300
        )
        
        st.plotly_chart(fig, width='stretch')
    
    # Tableau des détections récentes
    st.subheader("📋 Détections Récentes")
    
    if st.session_state.detections:
        df = pd.DataFrame(st.session_state.detections[-20:])  # 20 dernières
        st.dataframe(df, width='stretch')
    else:
        # Données simulées pour demo
        sample_data = []
        for i in range(10):
            sample_data.append({
                'Timestamp': (datetime.now() - timedelta(minutes=i*5)).strftime("%H:%M:%S"),
                'Caméra': f"Caméra {random.randint(1, 3)}",
                'Type': random.choice(['Personne', 'Véhicule', 'Mouvement']),
                'Confiance': f"{random.randint(75, 98)}%",
                'Statut': random.choice(['Normal', 'Attention', 'Alerte'])
            })
        
        df = pd.DataFrame(sample_data)
        st.dataframe(df, width='stretch')

def render_alerts_panel():
    """Panneau des alertes."""
    st.subheader("🚨 Centre des Alertes")
    
    # Génération d'alertes simulées si vide
    if not st.session_state.alerts and st.session_state.surveillance_active:
        if random.random() > 0.8:  # 20% de chance
            alert_types = ['CRITICAL', 'HIGH', 'MEDIUM']
            alert_messages = [
                'Mouvement suspect détecté dans la zone principale',
                'Personne non autorisée dans zone restreinte', 
                'Objet abandonné détecté',
                'Activité inhabituelle détectée'
            ]
            
            new_alert = {
                'level': random.choice(alert_types),
                'message': random.choice(alert_messages),
                'camera': f"Caméra {random.randint(1, 3)}",
                'timestamp': datetime.now(),
                'resolved': False
            }
            
            st.session_state.alerts.append(new_alert)
    
    # Filtres
    col1, col2 = st.columns(2)
    with col1:
        alert_filter = st.selectbox("Filtrer par niveau", ["Tous", "CRITICAL", "HIGH", "MEDIUM", "LOW"])
    with col2:
        show_resolved = st.checkbox("Afficher alertes résolues", False)
    
    # Affichage des alertes
    filtered_alerts = st.session_state.alerts
    if alert_filter != "Tous":
        filtered_alerts = [a for a in filtered_alerts if a['level'] == alert_filter]
    if not show_resolved:
        filtered_alerts = [a for a in filtered_alerts if not a.get('resolved')]
    
    if filtered_alerts:
        for i, alert in enumerate(filtered_alerts[-10:]):  # 10 dernières
            level_class = f"alert-{alert['level'].lower()}"
            timestamp = alert['timestamp'].strftime("%H:%M:%S")
            
            st.markdown(f"""
            <div class="{level_class}">
                <strong>{alert['level']}</strong> - {timestamp} - {alert['camera']}<br>
                {alert['message']}
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns([3, 1])
            with col2:
                if not alert.get('resolved') and st.button("✅ Résoudre", key=f"resolve_{i}"):
                    st.session_state.alerts[i]['resolved'] = True
                    st.rerun()
    else:
        st.info("✅ Aucune alerte active")

def main():
    """Application principale."""
    render_header()
    
    # Sidebar de contrôle
    with st.sidebar:
        st.header("⚙️ Contrôles Système")
        
        # Statut général
        st.subheader("📊 État Système")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Caméras", len(st.session_state.cameras))
        with col2:
            st.metric("Alertes", len([a for a in st.session_state.alerts if not a.get('resolved')]))
        
        # Contrôles principaux
        st.divider()
        st.subheader("🎮 Surveillance")
        
        if st.button("▶️ Démarrer Surveillance" if not st.session_state.surveillance_active else "⏹️ Arrêter Surveillance"):
            st.session_state.surveillance_active = not st.session_state.surveillance_active
            if st.session_state.surveillance_active:
                st.success("✅ Surveillance démarrée!")
            else:
                st.info("⏹️ Surveillance arrêtée")
            st.rerun()
        
        # Paramètres
        st.divider()
        st.subheader("⚙️ Paramètres")
        
        sensitivity = st.slider("Sensibilité générale", 0.1, 1.0, 0.7)
        auto_alerts = st.checkbox("Alertes automatiques", True)
        save_recordings = st.checkbox("Enregistrer vidéos", False)
        
        # Actions rapides
        st.divider()
        st.subheader("⚡ Actions Rapides")
        
        if st.button("🔄 Rafraîchir Tout"):
            st.rerun()
        
        if st.button("🧹 Vider Historique"):
            st.session_state.chat_history.clear()
            st.session_state.alerts.clear()
            st.session_state.detections.clear()
            st.success("Historique vidé!")
            st.rerun()
        
        if st.button("📊 Générer Rapport"):
            st.info("Rapport généré avec succès!")
    
    # Onglets principaux
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎥 Surveillance", 
        "📹 Configuration", 
        "💬 Chat IA", 
        "📊 Analytics", 
        "🚨 Alertes"
    ])
    
    with tab1:
        render_camera_grid()
        
        # Auto-refresh si surveillance active
        if st.session_state.surveillance_active:
            time.sleep(2)
            st.rerun()
    
    with tab2:
        render_camera_config()
    
    with tab3:
        render_vlm_chat()
    
    with tab4:
        render_analytics_dashboard()
    
    with tab5:
        render_alerts_panel()

if __name__ == "__main__":
    main()