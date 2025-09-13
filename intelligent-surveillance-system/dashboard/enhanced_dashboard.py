"""
🔒 Dashboard de Surveillance Intelligent - Version Améliorée
===========================================================

Version avec :
- Chat intégré dans Surveillance et Upload Vidéo
- Onglet Upload Vidéo avec analyse complète
- Chat contextuel adapté à chaque onglet
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
import tempfile
import os

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
if 'surveillance_chat' not in st.session_state:
    st.session_state.surveillance_chat = []
if 'video_chat' not in st.session_state:
    st.session_state.video_chat = []
if 'detections' not in st.session_state:
    st.session_state.detections = []
if 'alerts' not in st.session_state:
    st.session_state.alerts = []
if 'surveillance_active' not in st.session_state:
    st.session_state.surveillance_active = False
if 'uploaded_videos' not in st.session_state:
    st.session_state.uploaded_videos = []
if 'video_analysis_results' not in st.session_state:
    st.session_state.video_analysis_results = {}

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

.chat-container {
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 1rem;
    background: #f8f9fa;
    margin-top: 1rem;
    max-height: 300px;
    overflow-y: auto;
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

.camera-card {
    border: 2px solid #dee2e6;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
    background: #f8f9fa;
}

.analysis-result {
    border: 1px solid #28a745;
    border-radius: 8px;
    padding: 1rem;
    margin: 1rem 0;
    background: #f8fff8;
}

.detection-box {
    border: 1px solid #17a2b8;
    border-radius: 5px;
    padding: 0.5rem;
    margin: 0.25rem 0;
    background: #e8f4f8;
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
        <p>Système VLM Multi-Caméras & Analyse Vidéo</p>
    </div>
    """, unsafe_allow_html=True)

def render_integrated_chat(chat_type: str, context_data: Dict = None):
    """Chat intégré avec contexte spécifique à l'onglet."""
    
    # Sélection de l'historique selon le type
    chat_key = f"{chat_type}_chat"
    if chat_key not in st.session_state:
        st.session_state[chat_key] = []
    
    st.markdown("### 💬 Chat IA Contextualisé")
    
    # Questions prédéfinies selon le contexte
    if chat_type == "surveillance":
        questions = [
            "Que se passe-t-il sur les caméras en ce moment ?",
            "Y a-t-il des activités suspectes détectées ?",
            "Résume l'activité des 10 dernières minutes",
            "Quelles caméras sont les plus actives ?",
            "Génère un rapport de surveillance temps réel"
        ]
        context_info = f"📹 {len([c for c in st.session_state.cameras.values() if c.get('active')])} caméras actives, {len(st.session_state.alerts)} alertes"
    
    elif chat_type == "video":
        questions = [
            "Analyse les résultats de détection de cette vidéo",
            "Quels objets suspects as-tu identifiés ?",
            "Résume les événements chronologiques",
            "Évalue le niveau de risque global",
            "Compare avec les patterns de surveillance"
        ]
        context_info = f"🎥 {len(st.session_state.uploaded_videos)} vidéos analysées"
    
    # Affichage du contexte
    if context_data:
        st.info(f"📊 Contexte: {context_info}")
    
    # Affichage de l'historique dans une zone scrollable
    with st.container():
        st.markdown(f"""
        <div class="chat-container">
        """, unsafe_allow_html=True)
        
        for message in st.session_state[chat_key][-8:]:  # 8 derniers messages
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
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Interface de chat
    col1, col2 = st.columns([2, 1])
    
    with col1:
        user_input = st.text_input(
            f"Posez votre question sur {chat_type}...", 
            key=f"chat_input_{chat_type}",
            placeholder=f"Ex: {questions[0]}"
        )
    
    with col2:
        selected_question = st.selectbox(
            "Questions rapides", 
            [""] + questions,
            key=f"quick_q_{chat_type}"
        )
    
    # Traitement des messages
    message_to_send = None
    if st.button("📤 Envoyer", key=f"send_{chat_type}") and user_input:
        message_to_send = user_input
    elif selected_question:
        if st.button("📤 Envoyer Question", key=f"send_quick_{chat_type}"):
            message_to_send = selected_question
    
    if message_to_send:
        # Ajouter la question
        st.session_state[chat_key].append({
            'role': 'user',
            'content': message_to_send,
            'timestamp': datetime.now()
        })
        
        # Générer réponse contextuelle
        ai_response = generate_contextual_response(message_to_send, chat_type, context_data)
        
        st.session_state[chat_key].append({
            'role': 'assistant',
            'content': ai_response,
            'timestamp': datetime.now()
        })
        
        st.rerun()

def generate_contextual_response(question: str, chat_type: str, context_data: Dict) -> str:
    """Génère une réponse IA adaptée au contexte."""
    
    question_lower = question.lower()
    
    if chat_type == "surveillance":
        if "caméra" in question_lower or "camera" in question_lower:
            active_cameras = len([c for c in st.session_state.cameras.values() if c.get('active')])
            return f"📹 Statut caméras: {active_cameras} caméras actives sur {len(st.session_state.cameras)} configurées. Surveillance {'en cours' if st.session_state.surveillance_active else 'arrêtée'}."
        
        elif "alerte" in question_lower or "suspect" in question_lower:
            recent_alerts = len([a for a in st.session_state.alerts if not a.get('resolved')])
            return f"🚨 Analyse des alertes: {recent_alerts} alertes actives. Niveau de menace actuel: {'ÉLEVÉ' if recent_alerts > 3 else 'NORMAL'}."
        
        elif "rapport" in question_lower:
            return f"📊 Rapport surveillance temps réel:\n• Caméras: {len(st.session_state.cameras)} configurées\n• Détections: {len(st.session_state.detections)} total\n• Alertes: {len(st.session_state.alerts)} générées\n• Statut: {'Surveillance active' if st.session_state.surveillance_active else 'Système en pause'}"
        
        else:
            return f"🔍 Analyse en cours des flux de surveillance. {len(st.session_state.cameras)} caméras configurées. Dernière analyse: {datetime.now().strftime('%H:%M:%S')}."
    
    elif chat_type == "video":
        if "résultat" in question_lower or "détection" in question_lower:
            if st.session_state.video_analysis_results:
                last_analysis = list(st.session_state.video_analysis_results.values())[-1]
                detections = last_analysis.get('detections', [])
                return f"🎯 Analyse vidéo: {len(detections)} détections identifiées. Objets principaux: {', '.join([d['type'] for d in detections[:3]])}. Confiance moyenne: {random.randint(85, 95)}%."
            return "📹 Aucune analyse vidéo récente disponible. Uploadez une vidéo pour commencer l'analyse."
        
        elif "suspect" in question_lower or "risque" in question_lower:
            return f"⚠️ Évaluation des risques: Analyse comportementale en cours. Niveau de suspicion détecté: {random.choice(['FAIBLE', 'MODÉRÉ', 'ÉLEVÉ'])}. Recommandation: Surveillance continue."
        
        elif "chronologie" in question_lower or "événement" in question_lower:
            return f"📅 Timeline des événements:\n• 00:15 - Détection personne\n• 01:22 - Mouvement suspect\n• 02:10 - Objet abandonné\n• 03:45 - Sortie de zone\n\nDurée totale analysée: {random.randint(2, 10)} minutes."
        
        else:
            video_count = len(st.session_state.uploaded_videos)
            return f"🎥 Système d'analyse vidéo prêt. {video_count} vidéos traitées. Outils disponibles: YOLO, SAM2, Pose Detection, Analyse comportementale."
    
    return "🤖 Réponse générée par l'IA de surveillance intelligente."

def generate_dummy_frame(camera_id: str, width: int = 320, height: int = 240):
    """Génère une frame simulée pour demo."""
    img = np.random.randint(50, 200, (height, width, 3), dtype=np.uint8)
    
    cv2.putText(img, f"Camera {camera_id}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    timestamp = datetime.now().strftime("%H:%M:%S")
    cv2.putText(img, timestamp, (10, height - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    if random.random() > 0.7:
        x1, y1 = random.randint(10, width//2), random.randint(10, height//2)
        x2, y2 = x1 + 60, y1 + 80
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, "Person", (x1, y1-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    return img

def render_surveillance_tab():
    """Onglet de surveillance avec chat intégré."""
    st.subheader("🎥 Surveillance Temps Réel")
    
    # Grille des caméras
    if not st.session_state.cameras:
        st.info("📹 Aucune caméra configurée. Ajoutez une caméra dans l'onglet Configuration.")
    else:
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
                
                if camera.get('active', False) and st.session_state.surveillance_active:
                    frame = generate_dummy_frame(camera['id'])
                    st.image(frame, channels="BGR", caption=f"Live Feed - {camera['name']}")
                    
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
    
    # Génération d'alertes simulées
    if st.session_state.surveillance_active and random.random() > 0.9:
        new_alert = {
            'level': random.choice(['HIGH', 'MEDIUM', 'CRITICAL']),
            'message': random.choice([
                'Mouvement suspect détecté',
                'Personne non autorisée',
                'Objet abandonné'
            ]),
            'camera': f"Caméra {random.randint(1, 3)}",
            'timestamp': datetime.now(),
            'resolved': False
        }
        st.session_state.alerts.append(new_alert)
    
    # Chat intégré pour surveillance
    st.divider()
    context_data = {
        'active_cameras': len([c for c in st.session_state.cameras.values() if c.get('active')]),
        'total_cameras': len(st.session_state.cameras),
        'alerts': len(st.session_state.alerts),
        'surveillance_active': st.session_state.surveillance_active
    }
    render_integrated_chat("surveillance", context_data)

def render_video_upload_tab():
    """Onglet d'upload et analyse vidéo avec chat intégré."""
    st.subheader("🎥 Upload & Analyse Vidéo")
    
    # Section d'upload
    st.markdown("### 📤 Upload de Vidéo")
    
    uploaded_file = st.file_uploader(
        "Sélectionnez une vidéo à analyser",
        type=['mp4', 'avi', 'mov', 'mkv', 'webm'],
        help="Formats supportés: MP4, AVI, MOV, MKV, WEBM"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        analysis_mode = st.selectbox(
            "Mode d'analyse",
            ["Détection d'objets", "Analyse comportementale", "Analyse complète", "Détection de vol"]
        )
    
    with col2:
        confidence_threshold = st.slider("Seuil de confiance", 0.1, 1.0, 0.7)
    
    if uploaded_file is not None:
        # Affichage des informations du fichier
        file_details = {
            "Nom": uploaded_file.name,
            "Taille": f"{uploaded_file.size / 1024 / 1024:.2f} MB",
            "Type": uploaded_file.type
        }
        
        st.write("**Informations du fichier:**")
        st.json(file_details)
        
        # Bouton d'analyse
        if st.button("🔍 Analyser la Vidéo", type="primary"):
            with st.spinner("Analyse en cours..."):
                # Simulation d'analyse
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                
                # Génération de résultats simulés
                analysis_results = {
                    'video_name': uploaded_file.name,
                    'duration': f"{random.randint(30, 300)} secondes",
                    'frames_analyzed': random.randint(100, 1000),
                    'detections': [
                        {
                            'type': 'person',
                            'confidence': random.uniform(0.8, 0.98),
                            'timestamp': f"00:{random.randint(10, 59):02d}",
                            'bbox': [random.randint(0, 100), random.randint(0, 100), 
                                   random.randint(100, 300), random.randint(100, 400)]
                        },
                        {
                            'type': 'bag',
                            'confidence': random.uniform(0.7, 0.95),
                            'timestamp': f"01:{random.randint(10, 59):02d}",
                            'bbox': [random.randint(0, 100), random.randint(0, 100), 
                                   random.randint(50, 150), random.randint(50, 150)]
                        },
                        {
                            'type': 'suspicious_movement',
                            'confidence': random.uniform(0.6, 0.9),
                            'timestamp': f"02:{random.randint(10, 59):02d}",
                            'bbox': [random.randint(0, 200), random.randint(0, 200), 
                                   random.randint(200, 400), random.randint(200, 400)]
                        }
                    ],
                    'summary': {
                        'risk_level': random.choice(['FAIBLE', 'MODÉRÉ', 'ÉLEVÉ', 'CRITIQUE']),
                        'person_count': random.randint(1, 5),
                        'suspicious_events': random.randint(0, 3),
                        'objects_detected': random.randint(5, 20)
                    },
                    'timestamp': datetime.now()
                }
                
                # Stockage des résultats
                video_id = f"video_{len(st.session_state.uploaded_videos) + 1}"
                st.session_state.video_analysis_results[video_id] = analysis_results
                st.session_state.uploaded_videos.append({
                    'id': video_id,
                    'name': uploaded_file.name,
                    'upload_time': datetime.now()
                })
                
                st.success("✅ Analyse terminée avec succès!")
    
    # Affichage des résultats d'analyse
    if st.session_state.video_analysis_results:
        st.markdown("### 📊 Résultats d'Analyse")
        
        # Sélection de l'analyse à afficher
        video_options = {k: v['video_name'] for k, v in st.session_state.video_analysis_results.items()}
        selected_video = st.selectbox(
            "Sélectionner une analyse",
            list(video_options.keys()),
            format_func=lambda x: video_options[x]
        )
        
        if selected_video:
            results = st.session_state.video_analysis_results[selected_video]
            
            # Résumé général
            st.markdown(f"""
            <div class="analysis-result">
                <h4>📋 Résumé d'Analyse - {results['video_name']}</h4>
                <p><strong>Durée:</strong> {results['duration']}</p>
                <p><strong>Frames analysées:</strong> {results['frames_analyzed']}</p>
                <p><strong>Niveau de risque:</strong> <span style="color: {'red' if results['summary']['risk_level'] in ['ÉLEVÉ', 'CRITIQUE'] else 'orange' if results['summary']['risk_level'] == 'MODÉRÉ' else 'green'}">{results['summary']['risk_level']}</span></p>
                <p><strong>Personnes détectées:</strong> {results['summary']['person_count']}</p>
                <p><strong>Événements suspects:</strong> {results['summary']['suspicious_events']}</p>
                <p><strong>Objets détectés:</strong> {results['summary']['objects_detected']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Détections détaillées
            st.markdown("### 🎯 Détections Détaillées")
            
            for i, detection in enumerate(results['detections']):
                confidence_color = 'green' if detection['confidence'] > 0.8 else 'orange' if detection['confidence'] > 0.6 else 'red'
                
                st.markdown(f"""
                <div class="detection-box">
                    <strong>Détection {i+1}:</strong> {detection['type']}<br>
                    <strong>Timestamp:</strong> {detection['timestamp']}<br>
                    <strong>Confiance:</strong> <span style="color: {confidence_color}">{detection['confidence']:.2%}</span><br>
                    <strong>Position:</strong> {detection['bbox']}
                </div>
                """, unsafe_allow_html=True)
            
            # Graphique des détections
            col1, col2 = st.columns(2)
            
            with col1:
                # Graphique par type
                detection_types = [d['type'] for d in results['detections']]
                type_counts = {t: detection_types.count(t) for t in set(detection_types)}
                
                fig = px.pie(
                    values=list(type_counts.values()),
                    names=list(type_counts.keys()),
                    title="Répartition des détections"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Graphique de confiance
                confidences = [d['confidence'] for d in results['detections']]
                types = [d['type'] for d in results['detections']]
                
                fig = px.bar(
                    x=types,
                    y=confidences,
                    title="Niveau de confiance par détection",
                    labels={'x': 'Type de détection', 'y': 'Confiance'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Export des résultats
            if st.button("📥 Exporter Résultats JSON"):
                st.download_button(
                    label="💾 Télécharger JSON",
                    data=json.dumps(results, indent=2, default=str),
                    file_name=f"analysis_{selected_video}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
    
    # Chat intégré pour analyse vidéo
    st.divider()
    context_data = {
        'uploaded_videos': len(st.session_state.uploaded_videos),
        'analysis_results': len(st.session_state.video_analysis_results),
        'last_analysis': list(st.session_state.video_analysis_results.values())[-1] if st.session_state.video_analysis_results else None
    }
    render_integrated_chat("video", context_data)

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
        total_videos = len(st.session_state.uploaded_videos)
        st.markdown(f"""
        <div class="metric-card">
            <h2>{total_videos}</h2>
            <p>Vidéos Analysées</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        active_alerts = len([a for a in st.session_state.alerts if not a.get('resolved')])
        st.markdown(f"""
        <div class="metric-card">
            <h2>{active_alerts}</h2>
            <p>Alertes Actives</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Graphiques
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Activité Surveillance")
        
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
            title="Détections sur 24h",
            xaxis_title="Heure",
            yaxis_title="Nombre de détections",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
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
        
        st.plotly_chart(fig, use_container_width=True)

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
                if not alert.get('resolved') and st.button("✅ Résoudre", key=f"resolve_alert_{i}"):
                    alert['resolved'] = True
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
        
        st.metric("Vidéos Analysées", len(st.session_state.uploaded_videos))
        
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
            st.session_state.surveillance_chat.clear()
            st.session_state.video_chat.clear()
            st.session_state.alerts.clear()
            st.session_state.detections.clear()
            st.success("Historique vidé!")
            st.rerun()
        
        if st.button("📊 Générer Rapport"):
            st.info("Rapport généré avec succès!")
    
    # Onglets principaux - NOUVEAU: Upload Vidéo ajouté, Chat supprimé
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎥 Surveillance", 
        "📤 Upload Vidéo",
        "📹 Configuration", 
        "📊 Analytics", 
        "🚨 Alertes"
    ])
    
    with tab1:
        render_surveillance_tab()
        
        # Auto-refresh si surveillance active
        if st.session_state.surveillance_active:
            time.sleep(2)
            st.rerun()
    
    with tab2:
        render_video_upload_tab()
    
    with tab3:
        render_camera_config()
    
    with tab4:
        render_analytics_dashboard()
    
    with tab5:
        render_alerts_panel()

if __name__ == "__main__":
    main()