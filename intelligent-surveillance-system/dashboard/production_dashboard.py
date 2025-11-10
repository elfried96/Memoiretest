"""
 Dashboard de Surveillance Intelligent - VERSION PRODUCTION
============================================================

Version finale avec vraie intégration VLM :
- Pipeline VLM réelle avec 8 outils avancés
- AdaptiveOrchestrator pour sélection intelligente
- ToolOptimizationBenchmark pour optimisation
- Métriques de performance réelles
- Chat contextualisé avec vraies données
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
from typing import Dict, List, Any, Optional, Tuple
import tempfile
import os
import asyncio
import requests
import threading
import queue
import io
from PIL import Image
import signal
from collections import deque, Counter
import logging

# Configuration du logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import contexte vidéo
try:
    from video_context_integration import (
        VideoContextMetadata, 
        create_video_metadata_from_form,
        get_video_context_integration
    )
except ImportError as e:
    logger.warning(f"Contexte vidéo non disponible: {e}")
    VideoContextMetadata = None
    create_video_metadata_from_form = None
    get_video_context_integration = None

# Import système d'alertes audio
try:
    from utils.audio_alerts import (
        AudioAlertSystem, 
        play_alert, 
        play_behavior_alert,
        play_detection_alert
    )
    AUDIO_AVAILABLE = True
    logger.info("Système d'alertes audio chargé")
except ImportError as e:
    logger.warning(f"Système audio non disponible: {e}")
    AUDIO_AVAILABLE = False

# Configuration de la page
st.set_page_config(
    page_title="Surveillance Intelligente - Production",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon=None
)

# Style CSS avec icônes vectorielles propres
st.markdown("""
<style>
/* Status indicators avec couleurs uniquement */
.status-active { color: #28a745; font-weight: 600; }
.status-inactive { color: #dc3545; font-weight: 600; }
.status-warning { color: #ffc107; font-weight: 600; }
.status-info { color: #17a2b8; font-weight: 600; }

/* Indicateurs de statut simples avec points colorés */
.status-dot-active::before {
    content: "●";
    color: #28a745;
    margin-right: 6px;
}
.status-dot-inactive::before {
    content: "●";
    color: #dc3545;
    margin-right: 6px;
}
.status-dot-warning::before {
    content: "●";
    color: #ffc107;
    margin-right: 6px;
}

/* Animation de chargement simple */
.loading-dots::after {
    content: "...";
    animation: dots 1.5s steps(4, end) infinite;
}

@keyframes dots {
    0%, 20% { content: ""; }
    40% { content: "."; }
    60% { content: ".."; }
    80%, 100% { content: "..."; }
}
</style>
""", unsafe_allow_html=True)

# Configuration du logging pour debug VLM
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# Configuration du PYTHONPATH pour les imports
import sys
from pathlib import Path
dashboard_root = Path(__file__).parent
project_root = dashboard_root.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(dashboard_root) not in sys.path:
    sys.path.insert(0, str(dashboard_root))

# Imports de la pipeline réelle
try:
    logger.info(" Chargement des modules VLM...")
    try:
        from .real_pipeline_integration import (
            RealVLMPipeline, 
            RealAnalysisResult,
            initialize_real_pipeline,
            get_real_pipeline,
            is_real_pipeline_available
        )
        from .camera_manager import CameraConfig, MultiCameraManager, FrameData
        from .vlm_chatbot_symbiosis import process_vlm_chat_query, get_vlm_chatbot
        # ✅ NOUVEAU: Imports pour mémoire vidéo
        from .components.vlm_chat import get_vlm_chat
        from .services.video_memory_system import get_video_memory_system
    except ImportError:
        from real_pipeline_integration import (
            RealVLMPipeline, 
            RealAnalysisResult,
            initialize_real_pipeline,
            get_real_pipeline,
            is_real_pipeline_available
        )
        from camera_manager import CameraConfig, MultiCameraManager, FrameData
        from vlm_chatbot_symbiosis import process_vlm_chat_query, get_vlm_chatbot
        # ✅ NOUVEAU: Imports pour mémoire vidéo (fallback)
        from components.vlm_chat import get_vlm_chat
        from services.video_memory_system import get_video_memory_system
    PIPELINE_AVAILABLE = True
    logger.info(" Modules VLM chargés avec succès")
    
    # ✅ NOUVEAU: Initialisation système mémoire vidéo
    try:
        video_memory_system = get_video_memory_system()
        VIDEO_MEMORY_AVAILABLE = True
        logger.info(" Système mémoire vidéo initialisé")
    except Exception as e:
        logger.warning(f"⚠️ Système mémoire vidéo non disponible: {e}")
        VIDEO_MEMORY_AVAILABLE = False
except ImportError as e:
    logger.error(f" Erreur import pipeline VLM: {e}")
    st.error(f" Impossible d'importer la pipeline VLM: {e}")
    PIPELINE_AVAILABLE = False
    VIDEO_MEMORY_AVAILABLE = False

# Initialisation des variables de session
if 'cameras' not in st.session_state:
    st.session_state.cameras = {}
if 'surveillance_chat' not in st.session_state:
    st.session_state.surveillance_chat = []
if 'video_chat' not in st.session_state:
    st.session_state.video_chat = []
if 'real_detections' not in st.session_state:
    st.session_state.real_detections = []
if 'real_alerts' not in st.session_state:
    st.session_state.real_alerts = []
if 'audio_system' not in st.session_state and AUDIO_AVAILABLE:
    st.session_state.audio_system = AudioAlertSystem()
    logger.info("Système audio initialisé")
if 'auto_descriptions' not in st.session_state:
    st.session_state.auto_descriptions = deque(maxlen=20)
if 'alert_thresholds' not in st.session_state:
    st.session_state.alert_thresholds = {
        'confidence_threshold': 0.7,
        'auto_description_threshold': 0.75,
        'audio_enabled': True,
        'auto_alerts_enabled': True
    }
if 'surveillance_active' not in st.session_state:
    st.session_state.surveillance_active = False
if 'uploaded_videos' not in st.session_state:
    st.session_state.uploaded_videos = []
if 'video_analysis_results' not in st.session_state:
    st.session_state.video_analysis_results = {}
if 'pipeline_initialized' not in st.session_state:
    st.session_state.pipeline_initialized = False
if 'camera_manager' not in st.session_state:
    st.session_state.camera_manager = None
if 'real_pipeline' not in st.session_state:
    st.session_state.real_pipeline = None
if 'optimization_results' not in st.session_state:
    st.session_state.optimization_results = []
if 'streaming_manager' not in st.session_state:
    st.session_state.streaming_manager = None
if 'frame_cache' not in st.session_state:
    st.session_state.frame_cache = {}
if 'last_frame_update' not in st.session_state:
    st.session_state.last_frame_update = {}
if 'threaded_captures' not in st.session_state:
    st.session_state.threaded_captures = {}
if 'capture_threads' not in st.session_state:
    st.session_state.capture_threads = {}
if 'fluid_cache' not in st.session_state:
    st.session_state.fluid_cache = {}
if 'last_display_time' not in st.session_state:
    st.session_state.last_display_time = {}
if 'network_monitor' not in st.session_state:
    st.session_state.network_monitor = {}
if 'adaptive_settings' not in st.session_state:
    st.session_state.adaptive_settings = {}

# ========================================
# NOUVELLES FONCTIONS INTÉGRÉES
# ========================================

def generate_auto_description(detection_result, frame_data=None):
    """Génère automatiquement une description de scène lors des détections."""
    if not st.session_state.alert_thresholds['auto_alerts_enabled']:
        return
    
    try:
        if detection_result.confidence > st.session_state.alert_thresholds['auto_description_threshold']:
            # Description basée sur les données de détection
            scene_desc = f"""DESCRIPTION AUTO - {detection_result.timestamp.strftime('%H:%M:%S')}
            
DÉTECTION: {detection_result.description}
CONFIANCE: {detection_result.confidence:.1%}
NIVEAU: {detection_result.suspicion_level}
📍 CAMÉRA: {detection_result.camera_id}
OUTILS: {', '.join(detection_result.tools_used[:3])}

CONTEXTE: {get_scene_context_description(detection_result)}"""
            
            description_entry = {
                'timestamp': detection_result.timestamp,
                'description': scene_desc,
                'detection_trigger': detection_result.description,
                'confidence': detection_result.confidence,
                'suspicion_level': detection_result.suspicion_level,
                'camera_id': detection_result.camera_id
            }
            
            st.session_state.auto_descriptions.appendleft(description_entry)
            logger.info(f"Description auto générée pour détection {detection_result.frame_id}")
            
    except Exception as e:
        logger.error(f"Erreur génération description auto: {e}")

def get_scene_context_description(detection_result):
    """Génère un contexte de scène intelligent."""
    context_parts = []
    
    # Analyse du niveau de suspicion
    if detection_result.suspicion_level == "CRITICAL":
        context_parts.append("SITUATION CRITIQUE - Investigation immédiate requise")
    elif detection_result.suspicion_level == "HIGH":
        context_parts.append("ACTIVITÉ SUSPECTE - Surveillance renforcée recommandée")
    elif detection_result.suspicion_level == "MEDIUM":
        context_parts.append("COMPORTEMENT INHABITUEL - Observation continue")
    else:
        context_parts.append("ACTIVITÉ DÉTECTÉE - Niveau normal")
    
    # Analyse des outils utilisés
    if 'pose_estimator' in detection_result.tools_used:
        context_parts.append("• Analyse comportementale active")
    if 'sam2_segmentator' in detection_result.tools_used:
        context_parts.append("• Segmentation d'objets détectée")
    if 'trajectory_analyzer' in detection_result.tools_used:
        context_parts.append("• Mouvement anormal identifié")
    
    # Recommandations basées sur la confiance
    if detection_result.confidence > 0.9:
        context_parts.append("CONFIANCE TRÈS ÉLEVÉE - Résultat fiable")
    elif detection_result.confidence > 0.7:
        context_parts.append("CONFIANCE ÉLEVÉE - Résultat probable")
    
    return " | ".join(context_parts)

def trigger_integrated_alert(detection_result):
    """Déclenche les alertes intégrées (audio + visuel + description)."""
    suspicion_str = str(detection_result.suspicion_level)
    
    # 1. Alerte audio si activée
    if AUDIO_AVAILABLE and st.session_state.alert_thresholds['audio_enabled']:
        try:
            play_behavior_alert(suspicion_str, detection_result.description)
            logger.info(f"Alerte audio déclenchée: {suspicion_str}")
        except Exception as e:
            logger.error(f"Erreur alerte audio: {e}")
    
    # 2. Description automatique
    generate_auto_description(detection_result)
    
    # 3. Alerte visuelle (déjà existante)
    alert = {
        'level': suspicion_str,
        'message': detection_result.description,
        'camera': detection_result.camera_id,
        'timestamp': detection_result.timestamp,
        'resolved': False,
        'confidence': detection_result.confidence,
        'tools_used': detection_result.tools_used,
        'auto_generated': True  # Marquer comme auto-générée
    }
    st.session_state.real_alerts.append(alert)
    
    # 4. Log pour debug
    logger.info(f"Alerte intégrée déclenchée: {suspicion_str} - {detection_result.description}")

def render_auto_descriptions():
    """Affiche les descriptions automatiques de scènes."""
    st.subheader("Descriptions Automatiques de Scènes")
    
    if st.session_state.auto_descriptions:
        st.write(f"**{len(st.session_state.auto_descriptions)} descriptions générées**")
        
        # Options de filtrage
        col1, col2 = st.columns(2)
        with col1:
            show_count = st.selectbox("Afficher", [5, 10, 15, 20], index=0)
        with col2:
            level_filter = st.selectbox("Niveau", ["Tous", "CRITICAL", "HIGH", "MEDIUM", "LOW"])
        
        # Filtrage
        descriptions_to_show = list(st.session_state.auto_descriptions)
        if level_filter != "Tous":
            descriptions_to_show = [d for d in descriptions_to_show if level_filter in str(d.get('suspicion_level', ''))]
        
        # Affichage
        for i, desc in enumerate(descriptions_to_show[:show_count]):
            with st.expander(
                f"{desc['timestamp'].strftime('%H:%M:%S')} - {desc['detection_trigger'][:50]}...", 
                expanded=(i == 0)  # Premier élément ouvert
            ):
                st.markdown(desc['description'])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Confiance", f"{desc['confidence']:.1%}")
                with col2:
                    st.metric("Niveau", desc['suspicion_level'])
                with col3:
                    st.metric("Caméra", desc['camera_id'])
                    
    else:
        st.info("Aucune description automatique générée")
        st.caption("Les descriptions sont générées automatiquement lors des détections avec confiance > 75%")

def render_detection_timeline():
    """Crée une timeline interactive des détections."""
    st.subheader("Timeline Interactive des Détections")
    
    if st.session_state.real_detections:
        # Préparation des données
        timeline_data = []
        for detection in st.session_state.real_detections[-100:]:  # 100 dernières détections
            timeline_data.append({
                'timestamp': detection.timestamp,
                'confidence': detection.confidence,
                'suspicion': str(detection.suspicion_level),
                'description': detection.description[:40] + "..." if len(detection.description) > 40 else detection.description,
                'camera': detection.camera_id,
                'tools_count': len(detection.tools_used)
            })
        
        if timeline_data:
            df = pd.DataFrame(timeline_data)
            
            # Graphique timeline principal
            fig = px.scatter(
                df,
                x='timestamp',
                y='confidence',
                color='suspicion',
                size='tools_count',
                hover_data=['description', 'camera'],
                title="Timeline des Détections VLM",
                color_discrete_map={
                    'CRITICAL': '#dc3545',
                    'HIGH': '#fd7e14', 
                    'MEDIUM': '#ffc107',
                    'LOW': '#28a745'
                }
            )
            
            fig.update_layout(
                height=400,
                xaxis_title="Temps",
                yaxis_title="Confiance",
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistiques rapides
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total", len(df))
            with col2:
                critical_count = len(df[df['suspicion'] == 'CRITICAL'])
                st.metric("Critiques", critical_count)
            with col3:
                high_conf = len(df[df['confidence'] > 0.8])
                st.metric("Haute confiance", high_conf)
            with col4:
                avg_conf = df['confidence'].mean()
                st.metric("Confiance moy.", f"{avg_conf:.1%}")
                
        else:
            st.info("Données de timeline indisponibles")
    else:
        st.info("Aucune détection disponible pour la timeline")
        st.caption("Les détections apparaîtront ici une fois la surveillance démarrée")

def render_alert_controls():
    """Contrôles pour la configuration des alertes."""
    st.subheader("Configuration Alertes & Descriptions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Seuils de Déclenchement**")
        
        new_conf_threshold = st.slider(
            "Seuil confiance alertes",
            0.1, 1.0,
            st.session_state.alert_thresholds['confidence_threshold'],
            step=0.05,
            help="Confiance minimum pour déclencher une alerte"
        )
        
        new_desc_threshold = st.slider(
            "Seuil descriptions auto",
            0.1, 1.0, 
            st.session_state.alert_thresholds['auto_description_threshold'],
            step=0.05,
            help="Confiance minimum pour générer une description automatique"
        )
        
        # Mise à jour des seuils
        if new_conf_threshold != st.session_state.alert_thresholds['confidence_threshold']:
            st.session_state.alert_thresholds['confidence_threshold'] = new_conf_threshold
            st.success(f" Seuil confiance mis à jour: {new_conf_threshold:.0%}")
            
        if new_desc_threshold != st.session_state.alert_thresholds['auto_description_threshold']:
            st.session_state.alert_thresholds['auto_description_threshold'] = new_desc_threshold
            st.success(f" Seuil descriptions mis à jour: {new_desc_threshold:.0%}")
    
    with col2:
        st.markdown("**Options Audio & Auto**")
        
        new_audio_enabled = st.checkbox(
            "Alertes audio activées",
            st.session_state.alert_thresholds['audio_enabled'],
            help="Active/désactive les sons d'alerte"
        )
        
        new_auto_enabled = st.checkbox(
            "Alertes automatiques",
            st.session_state.alert_thresholds['auto_alerts_enabled'],
            help="Déclenche automatiquement alertes et descriptions"
        )
        
        # Mise à jour des options
        if new_audio_enabled != st.session_state.alert_thresholds['audio_enabled']:
            st.session_state.alert_thresholds['audio_enabled'] = new_audio_enabled
            st.success(f" Audio {'activé' if new_audio_enabled else 'désactivé'}")
            
        if new_auto_enabled != st.session_state.alert_thresholds['auto_alerts_enabled']:
            st.session_state.alert_thresholds['auto_alerts_enabled'] = new_auto_enabled
            st.success(f" Alertes auto {'activées' if new_auto_enabled else 'désactivées'}")
        
        # Test audio si disponible
        if AUDIO_AVAILABLE:
            st.markdown("**Test Audio**")
            col_test1, col_test2 = st.columns(2)
            with col_test1:
                if st.button("Test Medium"):
                    play_alert("MEDIUM", "Test alerte Medium", force=True)
            with col_test2:
                if st.button("Test Critical"):
                    play_alert("CRITICAL", "Test alerte Critical", force=True)

#  INITIALISATION AUTOMATIQUE DE LA VRAIE PIPELINE VLM (Mode optionnel)
# Ajout variable d'environnement pour bypass si problème
import os
AUTO_INIT_VLM = os.getenv('AUTO_INIT_VLM', 'true').lower() == 'true'

if AUTO_INIT_VLM and not st.session_state.pipeline_initialized and PIPELINE_AVAILABLE:
    try:
        with st.spinner(" Initialisation automatique de la pipeline VLM..."):
            if initialize_real_pipeline(
                vlm_model_name="Qwen/Qwen2.5-VL-7B-Instruct",
                enable_optimization=True
            ):
                st.success(" Pipeline VLM réelle initialisée automatiquement!")
            else:
                st.warning(" Échec initialisation pipeline VLM - Mode manuel disponible")
    except Exception as e:
        st.error(f" Erreur initialisation auto: {e}")
        st.info(" Pour bypasser: définir AUTO_INIT_VLM=false")

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

.pipeline-status {
    padding: 1rem;
    border-radius: 8px;
    margin: 1rem 0;
    border-left: 4px solid;
}

.pipeline-active { 
    border-color: #28a745; 
    background: rgba(40, 167, 69, 0.1); 
    color: #28a745;
}

.pipeline-inactive { 
    border-color: #dc3545; 
    background: rgba(220, 53, 69, 0.1); 
    color: #dc3545;
}

.real-analysis-result {
    border: 2px solid #007bff;
    border-radius: 8px;
    padding: 1rem;
    margin: 1rem 0;
    background: #f8f9ff;
}

.tool-performance {
    background: #e8f5e8;
    border: 1px solid #28a745;
    border-radius: 5px;
    padding: 0.5rem;
    margin: 0.25rem 0;
}

.optimization-badge {
    display: inline-block;
    padding: 0.25rem 0.5rem;
    border-radius: 0.25rem;
    font-size: 0.875rem;
    font-weight: 500;
    background: #17a2b8;
    color: white;
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

.metric-card {
    background: white;
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 1rem;
    text-align: center;
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
</style>
""", unsafe_allow_html=True)

def render_header():
    """Affiche l'en-tête principal."""
    st.markdown("""
    <div class="main-header">
        <h1> Dashboard de Surveillance Intelligente - PRODUCTION</h1>
        <p>Pipeline VLM Réelle • 8 Outils Avancés • Optimisation Adaptative</p>
    </div>
    """, unsafe_allow_html=True)

def render_pipeline_status():
    """Affiche le statut de la pipeline VLM."""
    if PIPELINE_AVAILABLE and is_real_pipeline_available():
        pipeline = get_real_pipeline()
        if pipeline and pipeline.running:
            stats = pipeline.get_performance_stats()
            
            # Configuration orchestrateur
            orchestration_info = "THOROUGH (8 outils complets)"
            if hasattr(pipeline, 'orchestrator') and hasattr(pipeline.orchestrator, 'config'):
                mode = getattr(pipeline.orchestrator.config, 'mode', 'THOROUGH')
                mode_names = {
                    'fast': 'FAST (3 outils rapides)',
                    'balanced': 'BALANCED (6 outils principaux)', 
                    'thorough': 'THOROUGH (8 outils complets)'
                }
                orchestration_info = mode_names.get(str(mode).lower(), 'THOROUGH (8 outils complets)')
            
            st.markdown(f"""
            <div class="pipeline-status pipeline-active">
                <h4>🎛️ Pipeline VLM Active - AdaptiveVLMOrchestrator</h4>
                <p><strong>Mode Orchestration:</strong> {orchestration_info}</p>
                <p><strong>Frames traitées:</strong> {stats.get('frames_processed', 0)}</p>
                <p><strong>Temps moyen:</strong> {stats.get('average_processing_time', 0):.2f}s</p>
                <p><strong>Outils optimaux actifs:</strong> {len(stats.get('current_optimal_tools', []))}/8</p>
                <p><strong>Score performance:</strong> {stats.get('current_performance_score', 0):.2f}</p>
                <p><strong>Cycles optimisation:</strong> {stats.get('optimization_cycles', 0)}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Section configuration orchestration avancée
            with st.expander("⚙️ Configuration Orchestration VLM & Architecture", expanded=False):
                st.subheader("🎛️ Architecture d'Orchestration AdaptiveVLMOrchestrator")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("""
                    **🎯 Modes d'Orchestration Disponibles:**
                    
                    🚀 **FAST Mode** (3 outils - ~0.8s/frame):
                    - DINO Features (Analyse comportementale)
                    - Pose Estimator (Détection postures)  
                    - Multimodal Fusion (Fusion données)
                    - ⚡ Vitesse maximale pour surveillance temps réel
                    
                    ⚖️ **BALANCED Mode** (6 outils - ~1.5s/frame):
                    - SAM2 + DINO + Pose + Trajectory
                    - Multimodal Fusion + Adversarial Detector
                    - 🎯 Équilibre optimal vitesse/précision
                    
                    🔬 **THOROUGH Mode** ✅ **ACTUEL** (8 outils - ~2.3s/frame):
                    - TOUS les outils VLM avancés actifs
                    - Analyse complète avec optimisation continue
                    - 🎖️ Précision maximale pour analyse forensique
                    """)
                
                with col2:
                    st.markdown("**📊 Configuration Pipeline Actuelle:**")
                    
                    if hasattr(pipeline, 'orchestrator') and hasattr(pipeline.orchestrator, 'config'):
                        config = pipeline.orchestrator.config
                        mode_val = str(getattr(config, 'mode', 'THOROUGH')).upper()
                        st.write(f"🎛️ **Mode Orchestration**: {mode_val}")
                        st.write(f"⚡ **Outils simultanés**: {getattr(config, 'max_concurrent_tools', 4)}")
                        st.write(f"🎯 **Seuil confiance**: {getattr(config, 'confidence_threshold', 0.7)}")
                        timeout_val = getattr(config, 'timeout_seconds', None)
                        timeout_text = 'Illimité ✅' if timeout_val is None else f'{timeout_val}s'
                        st.write(f"⏱️ **Timeout**: {timeout_text}")
                        advanced = getattr(config, 'enable_advanced_tools', True)
                        st.write(f"🔧 **Outils avancés**: {'✅ TOUS ACTIVÉS' if advanced else '❌ Limités'}")
                    else:
                        st.write("🎛️ **Mode**: THOROUGH (défaut)")
                        st.write("⚡ **Outils simultanés**: 4") 
                        st.write("🎯 **Seuil confiance**: 0.7")
                        st.write("⏱️ **Timeout**: Illimité ✅")
                        st.write("🔧 **Outils avancés**: ✅ TOUS ACTIVÉS")
                    
                    st.markdown("**🔄 Optimisation Continue ToolOptimizationBenchmark:**")
                    st.write(f"🔄 **Cycles exécutés**: {stats.get('optimization_cycles', 0)}")
                    best_tools = stats.get('best_tool_combination', [])
                    if best_tools:
                        st.write(f"🏆 **Meilleure combinaison**: {', '.join(best_tools[:3])}...")
                    else:
                        st.write("🏆 **Meilleure combinaison**: En cours d'apprentissage")
                    
                    current_tools = stats.get('current_optimal_tools', [])
                    if current_tools:
                        st.write(f"⚡ **Outils actifs**: {', '.join(current_tools[:3])}...")
                
                st.subheader("🛠️ Arsenal d'Outils VLM Intégrés")
                
                # Affichage des outils avec statut
                tools_info = [
                    ("🔍 SAM2 Segmentator", "Segmentation précise objets/personnes", "Critique", "sam2_segmentator"),
                    ("🧠 DINO Features", "Analyse comportementale avancée", "Critique", "dino_features"), 
                    ("🏃 Pose Estimator", "Détection postures & mouvements", "Essentiel", "pose_estimator"),
                    ("📍 Trajectory Analyzer", "Analyse trajectoires & patterns", "Important", "trajectory_analyzer"),
                    ("🔗 Multimodal Fusion", "Fusion données visuelles + contexte", "Critique", "multimodal_fusion"),
                    ("⏱️ Temporal Transformer", "Analyse séquences temporelles", "Avancé", "temporal_transformer"),
                    ("🛡️ Adversarial Detector", "Détection attaques adversaires", "Sécurité", "adversarial_detector"),
                    ("🎯 Domain Adapter", "Adaptation domaines spécifiques", "Spécialisé", "domain_adapter")
                ]
                
                tool_usage_stats = stats.get('tool_usage_stats', {})
                
                for tool_name, description, priority, tool_key in tools_info:
                    priority_colors = {
                        "Critique": "🔴", "Essentiel": "🟠", "Important": "🟡", 
                        "Avancé": "🔵", "Sécurité": "🟣", "Spécialisé": "🟢"
                    }
                    
                    usage_count = tool_usage_stats.get(tool_key, 0)
                    active_status = "✅ ACTIF" if tool_key in current_tools else "⏸️ Standby"
                    
                    st.write(f"{tool_name} - {description}")
                    st.write(f"   {priority_colors.get(priority, '⚪')} {priority} | {active_status} | Utilisations: {usage_count}")
                
                st.subheader("📈 Métriques de Performance Orchestration")
                
                perf_col1, perf_col2 = st.columns(2)
                with perf_col1:
                    total_frames = stats.get('frames_processed', 0)
                    success_rate = 0.0
                    if total_frames > 0:
                        total_detections = stats.get('total_detections', 0)
                        success_rate = (total_detections / total_frames) * 100
                    
                    st.metric("🎯 Taux Détection", f"{success_rate:.1f}%")
                    st.metric("⚡ Performance Score", f"{stats.get('current_performance_score', 0):.2f}")
                
                with perf_col2:
                    st.metric("🔄 Optimisations", stats.get('optimization_cycles', 0))
                    st.metric("⏱️ Temps Moyen", f"{stats.get('average_processing_time', 0):.2f}s")
        else:
            st.markdown("""
            <div class="pipeline-status pipeline-inactive">
                <h4>[PAUSED] Pipeline VLM Arrêtée</h4>
                <p>Cliquez sur "Initialiser Pipeline" pour démarrer</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="pipeline-status pipeline-inactive">
            <h4> Pipeline VLM Non Disponible</h4>
            <p>Pipeline VLM en attente d'initialisation - Vérifiez l'installation du système core</p>
        </div>
        """, unsafe_allow_html=True)

async def initialize_pipeline():
    """Initialise la pipeline VLM réelle."""
    if not PIPELINE_AVAILABLE:
        logger.error(" PIPELINE_AVAILABLE = False")
        st.error(" Modules pipeline non disponibles")
        return False
    
    try:
        logger.info(" Début initialisation pipeline VLM")
        with st.spinner(" Initialisation de la pipeline VLM..."):
            # Initialisation de la pipeline  
            logger.info("📞 Appel initialize_real_pipeline...")
            success = initialize_real_pipeline(
                vlm_model_name="Qwen/Qwen2.5-VL-7B-Instruct",
                enable_optimization=True
            )
            logger.info(f" Résultat initialize_real_pipeline: {success}")
            
            if success:
                logger.info(" Pipeline initialisée, récupération instance...")
                st.session_state.real_pipeline = get_real_pipeline()
                logger.info(f" Pipeline récupérée: {st.session_state.real_pipeline is not None}")
                
                if st.session_state.real_pipeline:
                    logger.info(f" Pipeline state: initialized={st.session_state.real_pipeline.initialized}")
                    logger.info(f" Pipeline model: {st.session_state.real_pipeline.vlm_model_name}")
                
                # Callbacks pour intégration dashboard
                def on_analysis_result(result):
                    st.session_state.real_detections.append(result)
                    
                    # Déclenchement intégré d'alertes (audio + visuel + description)
                    if hasattr(result, 'suspicion_level') and st.session_state.alert_thresholds['auto_alerts_enabled']:
                        suspicion_str = str(result.suspicion_level)
                        # Vérification du seuil de confiance
                        if (result.confidence > st.session_state.alert_thresholds['confidence_threshold'] or 
                            'HIGH' in suspicion_str or 'CRITICAL' in suspicion_str):
                            # Utilise la nouvelle fonction intégrée
                            trigger_integrated_alert(result)
                
                def on_optimization_result(result):
                    st.session_state.optimization_results.append(result)
                
                def on_error(error):
                    st.error(f" Erreur pipeline: {error}")
                
                # Enregistrement des callbacks
                st.session_state.real_pipeline.add_analysis_callback(on_analysis_result)
                st.session_state.real_pipeline.add_optimization_callback(on_optimization_result)
                st.session_state.real_pipeline.add_error_callback(on_error)
                
                # Démarrage du traitement
                if st.session_state.real_pipeline.start_processing():
                    st.session_state.pipeline_initialized = True
                    st.success(" Pipeline VLM initialisée et démarrée!")
                    return True
            
            logger.error(" Échec de l'initialisation de la pipeline")
            st.error(" Échec de l'initialisation de la pipeline")
            return False
            
    except Exception as e:
        logger.error(f" Exception initialisation: {e}")
        import traceback
        logger.error(f" Traceback: {traceback.format_exc()}")
        st.error(f" Erreur initialisation: {e}")
        return False

def generate_real_frame_analysis(frame_data: FrameData) -> Optional[RealAnalysisResult]:
    """Analyse une frame avec la vraie pipeline VLM uniquement."""
    if not st.session_state.pipeline_initialized or not st.session_state.real_pipeline:
        st.error(" Pipeline VLM non initialisée - Veuillez initialiser la pipeline d'abord")
        return None
        
    # Utilisation EXCLUSIVE de la vraie pipeline
    try:
        # Utiliser la même méthode async que pour l'initialisation
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            
            def run_analysis():
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    return new_loop.run_until_complete(st.session_state.real_pipeline.analyze_frame(frame_data))
                finally:
                    new_loop.close()
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_analysis)
                return future.result()  # Pas de timeout - attendre le temps nécessaire
        else:
            return asyncio.run(st.session_state.real_pipeline.analyze_frame(frame_data))
            
    except Exception as e:
        st.error(f" Erreur pipeline VLM réelle: {e}")
        logger.error(f" Erreur analyse VLM: {e}")
        return None

# FONCTION SUPPRIMÉE : Plus de simulation - VLM réelle uniquement

class StreamingManager:
    """Gestionnaire de streaming en arrière-plan pour éviter les rechargements."""
    
    def __init__(self):
        self.active_streams = {}  # Dict[camera_id, thread]
        self.frame_cache = {}     # Dict[camera_id, frame]
        self.last_update = {}     # Dict[camera_id, timestamp]
        self.running = False
        
    def start_stream(self, camera_id: str, camera_config: dict):
        """Démarre un stream en arrière-plan pour une caméra."""
        if camera_id in self.active_streams:
            return  # Déjà actif
        
        def stream_worker():
            """Worker thread pour capture continue."""
            attempt = 0
            while self.running and camera_id in self.active_streams:
                try:
                    attempt += 1
                    print(f"DEBUG: Stream {camera_id} tentative {attempt}")
                    
                    frame = capture_real_frame(camera_config, width=640, height=480)
                    if frame is not None:
                        print(f"DEBUG: Frame capturée pour {camera_id} - shape: {frame.shape}")
                        self.frame_cache[camera_id] = frame
                        self.last_update[camera_id] = time.time()
                    else:
                        print(f"DEBUG: Frame NULL pour {camera_id}")
                    
                    # Pause selon la configuration
                    refresh_rate = camera_config.get('refresh_rate', 2)
                    time.sleep(refresh_rate)
                    
                except Exception as e:
                    print(f"ERREUR stream {camera_id}: {e}")
                    import traceback
                    traceback.print_exc()
                    time.sleep(5)  # Retry après erreur
        
        # Démarre le thread
        thread = threading.Thread(target=stream_worker, daemon=True)
        thread.start()
        self.active_streams[camera_id] = thread
        self.running = True
    
    def stop_stream(self, camera_id: str):
        """Arrête le stream d'une caméra."""
        if camera_id in self.active_streams:
            del self.active_streams[camera_id]
            if camera_id in self.frame_cache:
                del self.frame_cache[camera_id]
            if camera_id in self.last_update:
                del self.last_update[camera_id]
    
    def get_latest_frame(self, camera_id: str):
        """Récupère la dernière frame en cache."""
        return self.frame_cache.get(camera_id, None)
    
    def is_frame_fresh(self, camera_id: str, max_age: float = 10.0) -> bool:
        """Vérifie si la frame est récente."""
        if camera_id not in self.last_update:
            return False
        return (time.time() - self.last_update[camera_id]) < max_age
    
    def stop_all(self):
        """Arrête tous les streams."""
        self.running = False
        self.active_streams.clear()
        self.frame_cache.clear()
        self.last_update.clear()

class MJPEGStreamManager:
    """Gestionnaire optimisé pour flux MJPEG avec cache et réduction latence."""
    
    def __init__(self):
        self.sessions = {}  # Cache des sessions par URL
        self.frame_cache = {}  # Cache des dernières frames
        self.last_update = {}  # Timestamp dernière mise à jour
        
    def get_session(self, url: str):
        """Récupère ou crée une session HTTP optimisée."""
        if url not in self.sessions:
            session = requests.Session()
            # Configuration optimisée pour faible latence
            session.headers.update({
                'User-Agent': 'SurveillanceBot/1.0 (low-latency)',
                'Accept': 'multipart/x-mixed-replace, */*',
                'Connection': 'keep-alive',
                'Cache-Control': 'no-cache',
                'Pragma': 'no-cache'
            })
            self.sessions[url] = session
        return self.sessions[url]
    
    def capture_latest_frame(self, url: str, width: int = 640, height: int = 480, max_age_ms: int = 500):
        """Capture avec cache intelligent pour réduire la latence."""
        current_time = time.time() * 1000
        
        # Vérification cache
        if url in self.frame_cache and url in self.last_update:
            age = current_time - self.last_update[url]
            if age < max_age_ms:
                return self.frame_cache[url]  # Frame récente en cache
        
        # Capture nouvelle frame
        frame = self._capture_fresh_frame(url, width, height)
        if frame is not None:
            self.frame_cache[url] = frame
            self.last_update[url] = current_time
            return frame
        
        # Fallback sur cache même ancien
        return self.frame_cache.get(url, None)
    
    def _capture_fresh_frame(self, url: str, width: int, height: int):
        """Capture MJPEG robuste avec validation stricte."""
        try:
            session = self.get_session(url)
            
            # Stream optimisé
            response = session.get(url, stream=True, timeout=3)
            
            if response.status_code == 200:
                content = b""
                frames_tested = 0
                max_frames_to_test = 3
                
                # Lecture par chunks plus grands
                for i, chunk in enumerate(response.iter_content(chunk_size=8192)):
                    if not chunk:
                        break
                        
                    content += chunk
                    
                    # Recherche d'images complètes
                    while frames_tested < max_frames_to_test:
                        jpg_start = content.find(b'\xff\xd8')
                        if jpg_start == -1:
                            break
                            
                        jpg_end = content.find(b'\xff\xd9', jpg_start)
                        if jpg_end == -1:
                            # Image incomplète
                            content = content[jpg_start:]
                            break
                            
                        try:
                            # Extraction
                            jpeg_data = content[jpg_start:jpg_end+2]
                            
                            # Validation stricte
                            if len(jpeg_data) > 5000:  # Taille minimale pour image correcte
                                pil_image = Image.open(io.BytesIO(jpeg_data))
                                
                                # Vérifications de qualité
                                if (pil_image.size[0] >= 320 and pil_image.size[1] >= 240 and
                                    pil_image.mode in ['RGB', 'L']):
                                    
                                    frame = np.array(pil_image)
                                    
                                    # Conversion couleur
                                    if len(frame.shape) == 3 and frame.shape[2] == 3:
                                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                                    elif len(frame.shape) == 2:  # Grayscale
                                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                                    
                                    # Validation du contenu (pas que du bruit)
                                    if frame.std() > 10:  # Vérification variance
                                        # Redimensionnement final
                                        if frame.shape[1] != width or frame.shape[0] != height:
                                            frame = cv2.resize(frame, (width, height))
                                        
                                        return frame
                            
                        except Exception:
                            pass
                        
                        frames_tested += 1
                        # Supprime cette image et continue
                        content = content[jpg_end+2:]
                    
                    # Limite buffer
                    if len(content) > 200000:
                        content = content[-50000:]
                    
                    # Limite chunks
                    if i > 25:
                        break
                        
        except Exception:
            pass
        
        return None

# Instance globale pour réutilisation
_mjpeg_manager = MJPEGStreamManager()

class NetworkQualityMonitor:
    """Moniteur de qualité réseau ULTRA-AVANCÉ pour optimisation automatique."""
    
    def __init__(self):
        self.running = False
        self.thread = None
        
        # Historique des métriques avec limite de mémoire
        self.latency_history = deque(maxlen=50)
        self.speed_history = deque(maxlen=30)
        self.packet_loss_history = deque(maxlen=20)
        
        # Métriques courantes
        self.current_latency = 100
        self.current_speed = 1000  # KB/s
        self.current_quality = "good"
        self.avg_latency = 100
        self.avg_speed = 1000
        self.jitter = 0
        self.packet_loss = 0
        
        # Paramètres adaptatifs optimaux
        self.adaptive_params = {
            "resolution": (640, 480),
            "fps": 25,
            "compression": 80,
            "timeout": 5,
            "buffer_size": 2,
            "retry_count": 3
        }
    
    def start_monitoring(self):
        """Démarre le monitoring réseau continu."""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._monitor_network_quality, daemon=True)
        self.thread.start()
        print("Monitoring réseau adaptatif démarré")
    
    def stop_monitoring(self):
        """Arrête le monitoring."""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=3.0)
    
    def _monitor_network_quality(self):
        """Thread de surveillance ULTRA-AVANCÉE de la qualité réseau."""
        consecutive_errors = 0
        max_consecutive_errors = 3
        test_iteration = 0
        
        while self.running:
            test_iteration += 1
            
            try:
                # === Test latence multi-serveurs avec failover ===
                latency_tests = []
                test_urls = [
                    'https://httpbin.org/json',
                    'https://jsonplaceholder.typicode.com/posts/1',
                    'https://api.github.com'
                ]
                
                for i, url in enumerate(test_urls[:2]):  # Max 2 serveurs
                    try:
                        start_time = time.perf_counter()
                        response = requests.get(url, timeout=4, 
                                              headers={'User-Agent': 'NetworkMonitor/1.0'})
                        if response.status_code == 200:
                            latency = (time.perf_counter() - start_time) * 1000
                            latency_tests.append(latency)
                            if i == 0:  # Premier test réussi, pas besoin d'autres
                                break
                    except Exception as e:
                        print(f" Test latence serveur {i+1} échoué: {e}")
                        continue
                
                # === Calcul latence finale ===
                if latency_tests:
                    # Utilise la médiane pour robustesse contre outliers
                    current_latency = np.median(latency_tests)
                    consecutive_errors = max(0, consecutive_errors - 1)  # Récupération graduelle
                else:
                    current_latency = 3000  # Latence très dégradée
                    consecutive_errors += 1
                
                # === Test vitesse adaptatif avec streaming ===
                speed_kbps = 0
                try:
                    # Taille test adaptée à la qualité actuelle
                    test_sizes = {
                        "excellent": 200000,
                        "good": 100000, 
                        "fair": 50000,
                        "poor": 20000
                    }
                    test_size = test_sizes.get(self.current_quality, 100000)
                    
                    speed_start = time.perf_counter()
                    response = requests.get(f'https://httpbin.org/bytes/{test_size}', 
                                          timeout=10, stream=True,
                                          headers={'User-Agent': 'SpeedTest/1.0'})
                    
                    total_bytes = 0
                    chunk_times = []
                    
                    for chunk in response.iter_content(chunk_size=8192):
                        chunk_start = time.perf_counter()
                        total_bytes += len(chunk)
                        chunk_times.append(time.perf_counter() - chunk_start)
                        
                        # Timeout progressif
                        if time.perf_counter() - speed_start > 8:
                            break
                    
                    download_time = time.perf_counter() - speed_start
                    if download_time > 0:
                        speed_kbps = (total_bytes / download_time) / 1024
                        
                        # Calcul du jitter basé sur la variance des chunks
                        if len(chunk_times) > 5:
                            self.jitter = np.std(chunk_times) * 1000
                            
                except Exception as e:
                    print(f" Test vitesse échoué: {e}")
                    speed_kbps = 30  # Vitesse très dégradée
                    consecutive_errors += 1
                
                # === Mise à jour métriques avec validation ===
                if 5 < current_latency < 10000:  # Latence réaliste
                    self.latency_history.append(current_latency)
                    self.current_latency = current_latency
                
                if 1 < speed_kbps < 500000:  # Vitesse réaliste (1KB/s à 500MB/s)
                    self.speed_history.append(speed_kbps)
                    self.current_speed = speed_kbps
                
                # === Calcul moyennes pondérées (priorité au récent) ===
                if self.latency_history:
                    recent_samples = min(5, len(self.latency_history))
                    recent_latency = np.mean(list(self.latency_history)[-recent_samples:])
                    historical_latency = np.mean(self.latency_history)
                    self.avg_latency = 0.8 * recent_latency + 0.2 * historical_latency
                
                if self.speed_history:
                    recent_samples = min(3, len(self.speed_history))
                    recent_speed = np.mean(list(self.speed_history)[-recent_samples:])
                    historical_speed = np.mean(self.speed_history)
                    self.avg_speed = 0.8 * recent_speed + 0.2 * historical_speed
                
                # === Détection qualité avec HYSTERESIS pour éviter oscillations ===
                previous_quality = self.current_quality
                new_quality = self._calculate_network_quality(consecutive_errors)
                
                # Hysteresis: changement seulement si différence significative
                quality_levels = {"poor": 0, "fair": 1, "good": 2, "excellent": 3}
                current_level = quality_levels[self.current_quality]
                new_level = quality_levels[new_quality]
                
                # Changement seulement si écart >= 1 niveau OU détérioration immédiate
                if abs(new_level - current_level) >= 1 or new_level < current_level:
                    self.current_quality = new_quality
                
                # === Adaptation paramètres si changement qualité ===
                if previous_quality != self.current_quality:
                    self._adapt_parameters_advanced()
                    print(f" Qualité réseau: {previous_quality} -> {self.current_quality}")
                
                # === Logging informatif périodique ===
                if test_iteration % 8 == 0:  # Toutes les 8 itérations
                    stability = "stable" if self.jitter < 50 else "variable" if self.jitter < 150 else "instable"
                    print(f" Réseau: {self.current_quality} ({stability}) | "
                          f"Latence: {self.current_latency:.0f}ms (moy: {self.avg_latency:.0f}ms) | "
                          f"Vitesse: {self.current_speed:.0f}KB/s | "
                          f"Jitter: {self.jitter:.1f}ms | Errors: {consecutive_errors}")
                
                # === Sauvegarde métriques pour dashboard ===
                current_time = time.time()
                stability_score = max(0, 100 - self.jitter * 2)
                
                if 'network_metrics' not in st.session_state:
                    st.session_state.network_metrics = {}
                
                st.session_state.network_metrics.update({
                    'quality': self.current_quality,
                    'latency': self.current_latency,
                    'avg_latency': self.avg_latency,
                    'speed': self.current_speed,
                    'avg_speed': self.avg_speed,
                    'jitter': self.jitter,
                    'stability_score': stability_score,
                    'consecutive_errors': consecutive_errors,
                    'last_update': current_time,
                    'params': self.adaptive_params.copy()
                })
                
            except Exception as e:
                consecutive_errors += 1
                print(f" Erreur monitoring réseau critique: {e}")
                
                # Dégradation progressive selon erreurs
                if consecutive_errors >= max_consecutive_errors:
                    self.current_quality = "poor"
                    self._adapt_parameters_advanced()
                    print(f" Connexion très instable, mode dégradé activé")
            
            # === Fréquence adaptative intelligente ===
            base_intervals = {"excellent": 20, "good": 15, "fair": 10, "poor": 6}
            base_interval = base_intervals.get(self.current_quality, 12)
            
            # Augmente la fréquence si problèmes détectés
            if consecutive_errors > 0:
                interval = max(4, base_interval - consecutive_errors * 2)
            else:
                interval = base_interval
            
            time.sleep(interval)
    
    def _calculate_network_quality(self, consecutive_errors: int) -> str:
        """Calcule la qualité réseau avec algorithme de scoring avancé."""
        
        # === Scores individuels (0-100) ===
        
        # Score latence (logarithmique pour plus de sensibilité)
        if self.current_latency <= 50:
            latency_score = 100
        elif self.current_latency <= 150:
            latency_score = 100 - (self.current_latency - 50) * 0.6  # Dégradation douce
        else:
            latency_score = max(0, 40 - (self.current_latency - 150) * 0.1)
        
        # Score vitesse (adaptatif selon usage)
        min_speed = 100  # KB/s minimum pour "good"
        optimal_speed = 2000  # KB/s pour "excellent"
        
        if self.current_speed >= optimal_speed:
            speed_score = 100
        elif self.current_speed >= min_speed:
            speed_score = 60 + ((self.current_speed - min_speed) / (optimal_speed - min_speed)) * 40
        else:
            speed_score = max(0, (self.current_speed / min_speed) * 60)
        
        # Score stabilité (basé sur jitter)
        if self.jitter <= 20:
            stability_score = 100
        elif self.jitter <= 100:
            stability_score = 100 - (self.jitter - 20) * 1.25
        else:
            stability_score = max(0, 30 - (self.jitter - 100) * 0.3)
        
        # Pénalité erreurs consécutives
        error_penalty = min(50, consecutive_errors * 15)
        
        # === Score composite pondéré ===
        composite_score = (
            latency_score * 0.4 +      # 40% - latence critique pour streaming
            speed_score * 0.35 +       # 35% - vitesse importante
            stability_score * 0.25     # 25% - stabilité pour qualité
        ) - error_penalty
        
        # === Classification avec seuils calibrés ===
        if composite_score >= 85:
            return "excellent"
        elif composite_score >= 65:
            return "good" 
        elif composite_score >= 35:
            return "fair"
        else:
            return "poor"
    
    def _adapt_parameters_advanced(self):
        """Adaptation ULTRA-INTELLIGENTE des paramètres selon qualité réseau."""
        
        if self.current_quality == "excellent":
            self.adaptive_params.update({
                "resolution": (1280, 720),    # HD pour excellent réseau
                "fps": 30,
                "compression": 95,            # Qualité maximale
                "timeout": 3,
                "buffer_size": 1,             # Buffer minimal, réactivité max
                "retry_count": 2,
                "stream_optimization": "quality"
            })
            
        elif self.current_quality == "good":
            self.adaptive_params.update({
                "resolution": (960, 540),     # Résolution intermédiaire
                "fps": 25,
                "compression": 85,
                "timeout": 4,
                "buffer_size": 2,
                "retry_count": 3,
                "stream_optimization": "balanced"
            })
            
        elif self.current_quality == "fair":
            self.adaptive_params.update({
                "resolution": (640, 480),     # VGA pour réseau moyen
                "fps": 20,
                "compression": 75,
                "timeout": 6,
                "buffer_size": 3,
                "retry_count": 4,
                "stream_optimization": "stability"
            })
            
        else:  # poor
            self.adaptive_params.update({
                "resolution": (320, 240),     # Résolution minimale
                "fps": 15,
                "compression": 60,            # Compression agressive
                "timeout": 10,                # Timeout élevé
                "buffer_size": 5,             # Buffer élevé pour absorber variabilité
                "retry_count": 6,
                "stream_optimization": "survival"
            })
        
        print(f" Paramètres adaptés pour qualité '{self.current_quality}': "
              f"{self.adaptive_params['resolution']} @ {self.adaptive_params['fps']}fps")
    
    def get_adaptive_parameters(self) -> dict:
        """Retourne les paramètres optimaux actuels."""
        return self.adaptive_params.copy()
    
    def get_network_status(self) -> dict:
        """Retourne le status réseau complet."""
        return {
            'quality': self.current_quality,
            'latency': self.current_latency,
            'speed': self.current_speed,
            'jitter': self.jitter,
            'adaptive_params': self.adaptive_params.copy(),
            'running': self.running
        }

class ThreadedVideoCapture:
    """Capture vidéo ULTRA optimisée avec double buffering et prédiction."""
    
    def __init__(self, source, backend=cv2.CAP_ANY, buffer_size=1):  # Buffer minimal pour latence zéro
        self.source = source
        self.backend = backend
        self.buffer_size = buffer_size
        self.frame_queue = queue.Queue(maxsize=buffer_size)
        self.cap = None
        self.running = False
        self.thread = None
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.actual_fps = 0
        self.frame_count = 0
        self.drop_count = 0
        self.target_fps = 30  # Remonte à 30 FPS avec optimisations
        
        # NOUVEAU: Double buffering pour fluidité parfaite
        self.current_frame = None
        self.previous_frame = None
        self.frame_lock = threading.Lock()
        
        # NOUVEAU: Système de prédiction pour éviter coupures
        self.last_successful_capture = time.time()
        self.capture_interval_history = deque(maxlen=10)
        self.predicted_next_frame_time = 0
        
    def start(self):
        """Démarre la capture en thread séparé."""
        if self.running:
            return True
            
        try:
            # Initialisation capture avec optimisations
            if isinstance(self.source, str) and self.source.isdigit():
                self.cap = cv2.VideoCapture(int(self.source), self.backend)
            else:
                self.cap = cv2.VideoCapture(self.source, self.backend)
            
            if not self.cap.isOpened():
                return False
            
            # Configurations ULTRA optimisées pour fluidité maximale
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Buffer minimal absolu
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)  # FPS cible optimisé
            
            # Pour webcams : optimisations spécifiques
            if isinstance(self.source, str) and self.source.isdigit():
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y','U','Y','V'))  # Format plus rapide que MJPEG
                # Résolution plus basse pour fluidité
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            else:
                # RTSP optimisations
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H','2','6','4'))
                
            # Désactive l'auto-exposition/focus pour éviter lag
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Manual mode
            
            # Démarre le thread de capture
            self.running = True
            self.thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.thread.start()
            
            return True
            
        except Exception as e:
            print(f"Erreur initialisation ThreadedVideoCapture: {e}")
            return False
    
    def _capture_loop(self):
        """Boucle de capture ULTRA optimisée avec double buffering et prédiction."""
        last_frame_time = time.time()
        target_interval = 1.0 / self.target_fps
        
        while self.running and self.cap and self.cap.isOpened():
            try:
                capture_start = time.time()
                
                ret, frame = self.cap.read()
                
                if ret and frame is not None:
                    self.frame_count += 1
                    current_time = time.time()
                    
                    # Mise à jour système de prédiction
                    if self.last_successful_capture > 0:
                        interval = current_time - self.last_successful_capture
                        self.capture_interval_history.append(interval)
                    self.last_successful_capture = current_time
                    
                    # Calcul FPS
                    self.fps_counter += 1
                    if current_time - self.last_fps_time >= 1.0:
                        self.actual_fps = self.fps_counter
                        self.fps_counter = 0
                        self.last_fps_time = current_time
                    
                    # DOUBLE BUFFERING: Rotation des frames pour continuité
                    with self.frame_lock:
                        self.previous_frame = self.current_frame
                        self.current_frame = frame.copy()  # Copie pour thread safety
                    
                    # Queue management ultra-rapide
                    while not self.frame_queue.empty():
                        try:
                            self.frame_queue.get_nowait()
                            self.drop_count += 1
                        except queue.Empty:
                            break
                    
                    try:
                        self.frame_queue.put_nowait(frame)
                    except queue.Full:
                        self.drop_count += 1
                        
                else:
                    # Pas de frame - utilise prédiction pour éviter coupures
                    if self.capture_interval_history and self.current_frame is not None:
                        avg_interval = sum(self.capture_interval_history) / len(self.capture_interval_history)
                        if current_time - self.last_successful_capture < avg_interval * 3:
                            # Réutilise la dernière frame valide temporairement
                            with self.frame_lock:
                                if self.current_frame is not None:
                                    try:
                                        self.frame_queue.put_nowait(self.current_frame.copy())
                                    except queue.Full:
                                        pass
                    
                    # Pause très courte pour retry immédiat
                    time.sleep(0.01)
                    continue
                
                # Timing ultra-précis avec compensation
                elapsed = time.time() - capture_start
                sleep_time = target_interval - elapsed
                if sleep_time > 0.005:  # Seuil minimum pour éviter overhead
                    time.sleep(sleep_time)
                    
            except Exception as e:
                print(f"Erreur capture thread: {e}")
                # En cas d'erreur, utilise frame précédente si disponible
                if self.previous_frame is not None:
                    with self.frame_lock:
                        try:
                            self.frame_queue.put_nowait(self.previous_frame.copy())
                        except queue.Full:
                            pass
                time.sleep(0.05)  # Pause très courte
    
    def get_latest_frame(self):
        """Récupère la frame la plus récente avec fallback intelligent."""
        frame = None
        try:
            # Récupère frame de la queue
            while not self.frame_queue.empty():
                frame = self.frame_queue.get_nowait()
        except queue.Empty:
            pass
        
        # NOUVEAU: Fallback sur double buffer si queue vide
        if frame is None:
            with self.frame_lock:
                if self.current_frame is not None:
                    frame = self.current_frame.copy()
                elif self.previous_frame is not None:
                    # En dernier recours, utilise la frame précédente
                    frame = self.previous_frame.copy()
        
        return frame
    
    def get_stats(self):
        """Statistiques de performance."""
        return {
            'fps': self.actual_fps,
            'frames_captured': self.frame_count,
            'frames_dropped': self.drop_count,
            'queue_size': self.frame_queue.qsize(),
            'running': self.running
        }
    
    def stop(self):
        """Arrête la capture."""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()
            self.cap = None

class AdaptiveFrameCompressor:
    """Compresseur de frames adaptatif pour optimisation réseau."""
    
    def __init__(self):
        self.quality_history = deque(maxlen=10)
        self.size_history = deque(maxlen=10)
        self.target_size = 50000  # 50KB par défaut
        self.current_quality = 85
        self.compression_stats = {
            'frames_processed': 0,
            'total_compression_time': 0,
            'avg_size_reduction': 0,
            'quality_adjustments': 0
        }
    
    def compress_frame_adaptive(self, frame: np.ndarray, network_quality: str = "good") -> Tuple[np.ndarray, dict]:
        """Compresse une frame selon la qualité réseau."""
        start_time = time.perf_counter()
        
        # Adaptation qualité selon réseau
        quality_map = {
            "excellent": 95,
            "good": 85,
            "fair": 70,
            "poor": 50
        }
        base_quality = quality_map.get(network_quality, 85)
        
        # Ajustement dynamique basé sur historique
        if len(self.size_history) > 3:
            avg_size = np.mean(self.size_history)
            if avg_size > self.target_size * 1.2:  # Trop gros
                self.current_quality = max(30, self.current_quality - 10)
                self.compression_stats['quality_adjustments'] += 1
            elif avg_size < self.target_size * 0.8:  # Trop petit
                self.current_quality = min(95, self.current_quality + 5)
        else:
            self.current_quality = base_quality
        
        # Compression JPEG optimisée
        original_size = frame.nbytes
        
        # Paramètres de compression adaptatifs
        encode_params = [
            cv2.IMWRITE_JPEG_QUALITY, int(self.current_quality),
            cv2.IMWRITE_JPEG_OPTIMIZE, 1,
            cv2.IMWRITE_JPEG_PROGRESSIVE, 1 if network_quality in ["excellent", "good"] else 0
        ]
        
        # Compression avec gestion d'erreur
        try:
            success, compressed_data = cv2.imencode('.jpg', frame, encode_params)
            if success:
                compressed_size = len(compressed_data)
                
                # Décompression pour frame utilisable
                frame_compressed = cv2.imdecode(compressed_data, cv2.IMREAD_COLOR)
                
                # Mise à jour statistiques
                compression_time = time.perf_counter() - start_time
                compression_ratio = original_size / compressed_size if compressed_size > 0 else 1.0
                
                self.size_history.append(compressed_size)
                self.quality_history.append(self.current_quality)
                
                self.compression_stats.update({
                    'frames_processed': self.compression_stats['frames_processed'] + 1,
                    'total_compression_time': self.compression_stats['total_compression_time'] + compression_time,
                    'avg_size_reduction': np.mean([original_size / s for s in self.size_history if s > 0])
                })
                
                # Métriques de retour
                metrics = {
                    'original_size': original_size,
                    'compressed_size': compressed_size,
                    'compression_ratio': compression_ratio,
                    'quality_used': self.current_quality,
                    'processing_time': compression_time,
                    'network_quality': network_quality
                }
                
                return frame_compressed, metrics
            else:
                # Échec compression, retourne frame originale
                return frame, {'error': 'compression_failed', 'original_size': original_size}
                
        except Exception as e:
            print(f" Erreur compression adaptative: {e}")
            return frame, {'error': str(e), 'original_size': original_size}
    
    def get_compression_stats(self) -> dict:
        """Retourne les statistiques de compression."""
        avg_time = (self.compression_stats['total_compression_time'] / 
                   max(1, self.compression_stats['frames_processed']))
        
        return {
            'frames_processed': self.compression_stats['frames_processed'],
            'avg_processing_time_ms': avg_time * 1000,
            'current_quality': self.current_quality,
            'avg_size_reduction': self.compression_stats['avg_size_reduction'],
            'quality_adjustments': self.compression_stats['quality_adjustments'],
            'target_size_kb': self.target_size / 1024
        }
    
    def set_target_size(self, size_kb: int):
        """Définit la taille cible en KB."""
        self.target_size = size_kb * 1024
        print(f" Taille cible de compression: {size_kb}KB")

class ThreadedMJPEGCapture:
    """Capture MJPEG HTTP ULTRA-OPTIMISÉE avec compression adaptative."""
    
    def __init__(self, url, width=640, height=480, buffer_size=3):
        self.url = url
        self.width = width
        self.height = height
        self.buffer_size = buffer_size
        self.frame_queue = queue.Queue(maxsize=buffer_size)
        self.running = False
        self.thread = None
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'Mozilla/5.0'})
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.actual_fps = 0
        self.frame_count = 0
        self.drop_count = 0
        
        # NOUVEAU: Compression adaptative intégrée
        self.compressor = AdaptiveFrameCompressor()
        self.compression_enabled = True
        self.compression_stats = {}
    
    def start(self):
        """Démarre la capture MJPEG."""
        if self.running:
            return True
        
        self.running = True
        self.thread = threading.Thread(target=self._mjpeg_loop, daemon=True)
        self.thread.start()
        return True
    
    def _mjpeg_loop(self):
        """Boucle de capture MJPEG optimisée."""
        while self.running:
            try:
                with self.session.get(self.url, stream=True, timeout=5) as response:
                    if response.status_code == 200:
                        content_buffer = b""
                        
                        for chunk in response.iter_content(chunk_size=8192):
                            if not self.running:
                                break
                                
                            content_buffer += chunk
                            
                            # Recherche et traitement frames JPEG
                            while True:
                                start_marker = content_buffer.find(b'\xff\xd8')
                                if start_marker == -1:
                                    break
                                
                                end_marker = content_buffer.find(b'\xff\xd9', start_marker)
                                if end_marker == -1:
                                    content_buffer = content_buffer[start_marker:]
                                    break
                                
                                jpeg_data = content_buffer[start_marker:end_marker + 2]
                                
                                if len(jpeg_data) > 5000:  # Frame valide
                                    try:
                                        img = Image.open(io.BytesIO(jpeg_data))
                                        if img.width >= 320 and img.height >= 240:
                                            frame = np.array(img)
                                            
                                            # Conversion couleur
                                            if len(frame.shape) == 3 and frame.shape[2] == 3:
                                                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                                            elif len(frame.shape) == 2:
                                                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                                            
                                            # Redimensionnement
                                            if frame.shape[:2] != (self.height, self.width):
                                                frame = cv2.resize(frame, (self.width, self.height))
                                            
                                            # Calcul FPS
                                            current_time = time.time()
                                            self.fps_counter += 1
                                            self.frame_count += 1
                                            
                                            if current_time - self.last_fps_time >= 1.0:
                                                self.actual_fps = self.fps_counter
                                                self.fps_counter = 0
                                                self.last_fps_time = current_time
                                            
                                            # Ajout à la queue
                                            try:
                                                while not self.frame_queue.empty():
                                                    try:
                                                        self.frame_queue.get_nowait()
                                                        self.drop_count += 1
                                                    except queue.Empty:
                                                        break
                                                
                                                self.frame_queue.put_nowait(frame)
                                            except queue.Full:
                                                self.drop_count += 1
                                    
                                    except Exception:
                                        pass
                                
                                content_buffer = content_buffer[end_marker + 2:]
                            
                            # Limite buffer
                            if len(content_buffer) > 100000:
                                content_buffer = content_buffer[-50000:]
                    else:
                        time.sleep(1)
                        
            except Exception as e:
                print(f"Erreur MJPEG thread: {e}")
                time.sleep(2)
    
    def get_latest_frame(self):
        """Récupère la frame la plus récente."""
        frame = None
        try:
            while not self.frame_queue.empty():
                frame = self.frame_queue.get_nowait()
        except queue.Empty:
            pass
        return frame
    
    def get_stats(self):
        """Statistiques de performance."""
        return {
            'fps': self.actual_fps,
            'frames_captured': self.frame_count,
            'frames_dropped': self.drop_count,
            'queue_size': self.frame_queue.qsize(),
            'running': self.running
        }
    
    def stop(self):
        """Arrête la capture."""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)

def get_or_create_threaded_capture(camera_id: str, camera_config: dict):
    """Récupère ou crée une capture threadée ULTRA-OPTIMISÉE pour une caméra."""
    
    # Initialise les systèmes adaptatifs
    initialize_adaptive_systems()
    
    # Nettoie les captures inactives
    to_remove = []
    for cam_id, capture in st.session_state.threaded_captures.items():
        if not capture.running:
            to_remove.append(cam_id)
    
    for cam_id in to_remove:
        del st.session_state.threaded_captures[cam_id]
        print(f" Capture inactive supprimée: {cam_id}")
    
    # Récupère le monitoring réseau pour cette caméra
    network_monitor = get_network_monitor(camera_id)
    adaptive_params = network_monitor.get_adaptive_parameters()
    
    # Création nouvelle capture si nécessaire
    if camera_id not in st.session_state.threaded_captures:
        source = camera_config['source']
        
        # Utilise les nouveaux composants optimisés
        try:
            # Détermine le type de backend optimal
            if source.isdigit():
                backend = cv2.CAP_V4L2  # Optimal pour webcam Linux
            elif source.startswith('rtsp'):
                backend = cv2.CAP_FFMPEG  # Optimal pour RTSP
                # Ajoute TCP si pas présent
                if 'tcp' not in source.lower():
                    separator = '&' if '?' in source else '?'
                    source = f"{source}{separator}tcp"
            else:
                backend = cv2.CAP_ANY  # Fallback générique
            
            # Crée la capture avec paramètres adaptatifs
            capture = ThreadedVideoCapture(source, backend, 
                                         buffer_size=adaptive_params.get('buffer_size', 1))
            
            if capture.start():
                st.session_state.threaded_captures[camera_id] = capture
                print(f" Capture threadée optimisée créée pour {camera_id} "
                      f"(qualité: {network_monitor.current_quality})")
            else:
                print(f" Échec création capture pour {camera_id}")
                return None
                
        except Exception as e:
            print(f" Erreur création capture {camera_id}: {e}")
            return None
    
    return st.session_state.threaded_captures.get(camera_id)

def capture_frame_threaded(camera_config: dict, width: int = 640, height: int = 480):
    """Capture threadée ULTRA-OPTIMISÉE avec compression adaptative."""
    camera_id = camera_config.get('id', camera_config.get('source', 'unknown'))
    
    # Récupère ou crée la capture threadée optimisée
    capture = get_or_create_threaded_capture(camera_id, camera_config)
    
    if capture:
        frame = capture.get_latest_frame()
        
        if frame is not None:
            # Récupère le monitoring réseau et compresseur adaptatif
            network_monitor = get_network_monitor(camera_id)
            compressor = get_adaptive_compressor()
            
            # Redimensionne selon paramètres adaptatifs
            adaptive_params = network_monitor.get_adaptive_parameters()
            target_width, target_height = adaptive_params['resolution']
            
            if frame.shape[:2] != (target_height, target_width):
                frame = cv2.resize(frame, (target_width, target_height), 
                                 interpolation=cv2.INTER_LINEAR)
            
            # Compression adaptative selon qualité réseau
            frame_compressed, compression_metrics = compressor.compress_frame_adaptive(
                frame, network_monitor.current_quality
            )
            
            # Overlays avec informations réseau et performance
            timestamp = datetime.now().strftime("%H:%M:%S")
            stats = capture.get_stats()
            
            # Overlay principal
            cv2.putText(frame_compressed, f"{camera_config.get('name', 'Camera')}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Overlay timing et performance
            cv2.putText(frame_compressed, f"ADAPTIVE | {timestamp}", (10, target_height - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Overlay statistiques réseau
            cv2.putText(frame_compressed, 
                       f"FPS: {stats['fps']:.1f} | Net: {network_monitor.current_quality}", 
                       (10, target_height - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            # Overlay compression (si disponible)
            if 'compression_ratio' in compression_metrics:
                cv2.putText(frame_compressed, 
                           f"Compress: {compression_metrics['compression_ratio']:.1f}x | "
                           f"Q: {compression_metrics['quality_used']}", 
                           (10, target_height - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (128, 255, 128), 1)
            
            return frame_compressed
    
    # Fallback vers méthode simple en cas d'échec
    return capture_real_frame_simple(camera_config, width, height)

def cleanup_threaded_captures():
    """Nettoie toutes les captures threadées actives."""
    try:
        # Arrête toutes les captures threadées
        for camera_id, capture in list(st.session_state.threaded_captures.items()):
            try:
                print(f" Arrêt capture threadée {camera_id}")
                capture.stop()
            except Exception as e:
                print(f" Erreur arrêt capture {camera_id}: {e}")
        
        # Vide le dictionnaire
        st.session_state.threaded_captures.clear()
        print(" Toutes les captures threadées arrêtées")
        
    except Exception as e:
        print(f" Erreur cleanup global: {e}")

def get_threaded_capture_stats():
    """Retourne les statistiques de toutes les captures threadées."""
    stats = {}
    
    for camera_id, capture in st.session_state.threaded_captures.items():
        try:
            stats[camera_id] = capture.get_stats()
        except Exception as e:
            stats[camera_id] = {'error': str(e)}
    
    return stats

def get_fluid_frame(camera_id: str, camera_config: dict, max_age_ms: int = 33):
    """Cache ULTRA-RAPIDE avec double buffering - élimine les coupures."""
    current_time = time.time() * 1000
    
    # Tentative récupération frame fraîche via threading
    capture = st.session_state.threaded_captures.get(camera_id)
    fresh_frame = None
    
    if capture and capture.running:
        # Essaie frame fraîche du double buffer directement
        fresh_frame = capture.get_latest_frame()
        
        if fresh_frame is not None:
            # Mise à jour cache avec frame fraîche
            st.session_state.fluid_cache[camera_id] = {
                'frame': fresh_frame,
                'timestamp': current_time,
                'backup_frame': st.session_state.fluid_cache.get(camera_id, {}).get('frame'),  # Garde backup
            }
            return fresh_frame
    
    # Fallback 1: Cache récent
    if camera_id in st.session_state.fluid_cache:
        cached_data = st.session_state.fluid_cache[camera_id]
        age = current_time - cached_data['timestamp']
        
        if age < max_age_ms:
            return cached_data['frame']
        elif age < max_age_ms * 3 and cached_data.get('backup_frame') is not None:
            # Fallback 2: Frame de backup si cache pas trop ancien
            return cached_data['backup_frame']
    
    # Fallback 3: Capture forcée en cas d'échec total
    frame = capture_frame_threaded(camera_config, 640, 480)
    
    if frame is not None:
        # Sauvegarde avec backup
        old_frame = st.session_state.fluid_cache.get(camera_id, {}).get('frame')
        st.session_state.fluid_cache[camera_id] = {
            'frame': frame,
            'timestamp': current_time,
            'backup_frame': old_frame
        }
    else:
        # Fallback 4: Dernière frame connue si TOUT échoue
        if camera_id in st.session_state.fluid_cache:
            return st.session_state.fluid_cache[camera_id].get('frame') or \
                   st.session_state.fluid_cache[camera_id].get('backup_frame')
    
    return frame

class CameraNetworkQualityMonitor:
    """Moniteur de qualité réseau adaptatif pour optimisation automatique par caméra."""
    
    def __init__(self, camera_id: str):
        self.camera_id = camera_id
        self.latency_history = deque(maxlen=20)
        self.success_rate_history = deque(maxlen=50)
        self.bandwidth_history = deque(maxlen=10)
        self.last_quality_check = 0
        self.current_quality = "good"  # poor, fair, good, excellent
        self.adaptive_params = {
            "resolution": (640, 480),
            "fps": 30,
            "compression": 85,
            "timeout": 5
        }
    
    def measure_network_latency(self, url: str) -> float:
        """Mesure la latence réseau vers la source."""
        import requests
        try:
            start_time = time.time()
            
            if url.startswith('http'):
                # Test ping HTTP simple
                response = requests.head(url, timeout=3)
                latency = (time.time() - start_time) * 1000
                return latency if response.status_code == 200 else 9999
            else:
                # Pour RTSP/webcam, mesure réelle indisponible
                return 0  # Latence indéterminée
                
        except Exception:
            return 9999  # Très mauvaise latence en cas d'erreur
    
    def update_quality_metrics(self, capture_success: bool, capture_time: float, frame_size: int = 0):
        """Met à jour les métriques de qualité réseau."""
        current_time = time.time()
        
        # Historique des succès/échecs
        self.success_rate_history.append(1 if capture_success else 0)
        
        if capture_success:
            # Temps de capture (proxy pour latence)
            self.latency_history.append(capture_time * 1000)
            
            # Estimation bandwidth si taille frame disponible
            if frame_size > 0:
                bandwidth = frame_size / max(capture_time, 0.001)  # bytes/sec
                self.bandwidth_history.append(bandwidth)
        
        # Réévaluation qualité toutes les 5 secondes
        if current_time - self.last_quality_check > 5:
            self._evaluate_network_quality()
            self.last_quality_check = current_time
    
    def _evaluate_network_quality(self):
        """Évalue la qualité réseau et ajuste les paramètres."""
        if not self.success_rate_history:
            return
        
        # Calcul métriques
        success_rate = sum(self.success_rate_history) / len(self.success_rate_history)
        avg_latency = sum(self.latency_history) / len(self.latency_history) if self.latency_history else 100
        
        # Détermination qualité
        if success_rate > 0.95 and avg_latency < 50:
            new_quality = "excellent"
        elif success_rate > 0.85 and avg_latency < 100:
            new_quality = "good"
        elif success_rate > 0.7 and avg_latency < 200:
            new_quality = "fair"
        else:
            new_quality = "poor"
        
        # Ajustement paramètres si qualité change
        if new_quality != self.current_quality:
            self.current_quality = new_quality
            self._adapt_parameters()
            print(f"  Qualité réseau {self.camera_id}: {new_quality} (Succès: {success_rate:.1%}, Latence: {avg_latency:.0f}ms)")
    
    def _adapt_parameters(self):
        """Adapte automatiquement les paramètres selon la qualité."""
        if self.current_quality == "excellent":
            # Qualité parfaite - paramètres maximaux
            self.adaptive_params.update({
                "resolution": (1280, 720),
                "fps": 30,
                "compression": 95,
                "timeout": 3,
                "buffer_size": 1,
                "retry_count": 2
            })
        elif self.current_quality == "good":
            # Bonne qualité - paramètres standards
            self.adaptive_params.update({
                "resolution": (640, 480),
                "fps": 30,
                "compression": 85,
                "timeout": 5,
                "buffer_size": 1,
                "retry_count": 3
            })
        elif self.current_quality == "fair":
            # Qualité moyenne - réduction légère
            self.adaptive_params.update({
                "resolution": (640, 480),
                "fps": 20,
                "compression": 75,
                "timeout": 8,
                "buffer_size": 2,
                "retry_count": 4
            })
        else:  # poor
            # Mauvaise qualité - mode survie
            self.adaptive_params.update({
                "resolution": (320, 240),
                "fps": 15,
                "compression": 60,
                "timeout": 10,
                "buffer_size": 3,
                "retry_count": 5
            })
    
    def get_adaptive_params(self) -> dict:
        """Retourne les paramètres adaptés à la qualité réseau."""
        return self.adaptive_params.copy()
    
    def get_quality_info(self) -> dict:
        """Retourne informations sur la qualité réseau."""
        success_rate = sum(self.success_rate_history) / len(self.success_rate_history) if self.success_rate_history else 0
        avg_latency = sum(self.latency_history) / len(self.latency_history) if self.latency_history else 0
        
        return {
            "quality": self.current_quality,
            "success_rate": success_rate,
            "avg_latency": avg_latency,
            "adaptive_params": self.adaptive_params
        }

def get_network_monitor(camera_id: str) -> NetworkQualityMonitor:
    """Récupère ou crée un moniteur réseau pour une caméra."""
    if camera_id not in st.session_state.network_monitor:
        st.session_state.network_monitor[camera_id] = NetworkQualityMonitor()
        st.session_state.network_monitor[camera_id].start_monitoring()
    return st.session_state.network_monitor[camera_id]

def get_threaded_capture(camera_id: str, source_url: str) -> ThreadedVideoCapture:
    """Récupère ou crée une capture threadée pour une caméra."""
    if camera_id not in st.session_state.threaded_captures:
        capture = ThreadedVideoCapture(source_url)
        if capture.start():
            st.session_state.threaded_captures[camera_id] = capture
            print(f" Capture threadée démarrée pour {camera_id}")
        else:
            print(f" Échec démarrage capture threadée pour {camera_id}")
            return None
    return st.session_state.threaded_captures.get(camera_id)

def cleanup_threaded_captures():
    """Nettoie toutes les captures threadées."""
    for camera_id, capture in st.session_state.threaded_captures.items():
        if capture:
            capture.stop()
    st.session_state.threaded_captures.clear()
    print(" Captures threadées nettoyées")

def get_adaptive_compressor() -> AdaptiveFrameCompressor:
    """Récupère le compresseur adaptatif global."""
    if 'adaptive_compressor' not in st.session_state:
        st.session_state.adaptive_compressor = AdaptiveFrameCompressor()
    return st.session_state.adaptive_compressor

def initialize_adaptive_systems():
    """Initialise tous les systèmes adaptatifs."""
    # Démarre le monitoring global si pas déjà fait
    if 'global_network_monitor' not in st.session_state:
        st.session_state.global_network_monitor = NetworkQualityMonitor()
        st.session_state.global_network_monitor.start_monitoring()
        print("[NETWORK] Système de monitoring adaptatif initialisé")
    
    # Initialise le compresseur adaptatif
    get_adaptive_compressor()
    
    # Nettoyage automatique périodique
    current_time = time.time()
    if not hasattr(st.session_state, 'last_cleanup') or current_time - st.session_state.last_cleanup > 300:  # 5 min
        # Nettoie les captures inactives
        inactive_captures = []
        for camera_id, capture in st.session_state.threaded_captures.items():
            if capture and not capture.running:
                inactive_captures.append(camera_id)
        
        for camera_id in inactive_captures:
            del st.session_state.threaded_captures[camera_id]
            print(f" Capture inactive supprimée: {camera_id}")
        
        st.session_state.last_cleanup = current_time

def capture_mjpeg_frame_simple(url: str, width: int = 640, height: int = 480):
    """Capture MJPEG simple et robuste."""
    try:
        # Headers optimisés pour MJPEG
        headers = {
            'User-Agent': 'Mozilla/5.0 (Linux; SurveillanceBot)',
            'Accept': 'multipart/x-mixed-replace, image/jpeg, */*',
            'Connection': 'keep-alive',
            'Cache-Control': 'no-cache'
        }
        
        # Stream pour capture continue
        with requests.get(url, stream=True, timeout=5, headers=headers) as response:
            if response.status_code == 200:
                content_buffer = b""
                
                # Lecture par chunks
                for chunk in response.iter_content(chunk_size=4096):
                    if not chunk:
                        continue
                        
                    content_buffer += chunk
                    
                    # Recherche image JPEG la plus récente
                    while True:
                        # Trouve début JPEG
                        start_marker = content_buffer.find(b'\xff\xd8')
                        if start_marker == -1:
                            break
                            
                        # Trouve fin JPEG
                        end_marker = content_buffer.find(b'\xff\xd9', start_marker)
                        if end_marker == -1:
                            # Image incomplète, garder le buffer
                            content_buffer = content_buffer[start_marker:]
                            break
                        
                        # Extraction de l'image
                        jpeg_data = content_buffer[start_marker:end_marker + 2]
                        
                        try:
                            # Validation stricte taille et contenu
                            if len(jpeg_data) > 5000:  # Image suffisamment grande
                                try:
                                    img = Image.open(io.BytesIO(jpeg_data))
                                    
                                    # Vérifications multiples
                                    if (img.width >= 320 and img.height >= 240 and 
                                        img.mode in ['RGB', 'L', 'RGBA']):
                                        
                                        frame = np.array(img)
                                        
                                        # Conversion couleur selon le mode
                                        if len(frame.shape) == 3 and frame.shape[2] == 3:
                                            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                                        elif len(frame.shape) == 3 and frame.shape[2] == 4:
                                            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                                        elif len(frame.shape) == 2:
                                            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                                        
                                        # Validation du contenu (pas de bruit)
                                        if frame.std() > 15:  # Variance significative
                                            # Redimensionnement
                                            if frame.shape[1] != width or frame.shape[0] != height:
                                                frame = cv2.resize(frame, (width, height))
                                            
                                            return frame
                                            
                                except Exception:
                                    pass
                        except Exception:
                            pass
                        
                        # Supprime cette image et continue
                        content_buffer = content_buffer[end_marker + 2:]
                    
                    # Limite taille buffer
                    if len(content_buffer) > 100000:
                        content_buffer = content_buffer[-50000:]
                    
                    # Limite lecture pour éviter blocage
                    if len(content_buffer) > 200000:
                        break
                        
    except Exception:
        pass
    
    return None

def capture_mjpeg_frame_native(url: str, width: int = 640, height: int = 480):
    """Capture MJPEG avec double fallback."""
    # Essai 1 : Système optimisé
    frame = _mjpeg_manager.capture_latest_frame(url, width, height, max_age_ms=500)
    if frame is not None:
        return frame
    
    # Essai 2 : Méthode simple
    frame = capture_mjpeg_frame_simple(url, width, height)
    if frame is not None:
        return frame
    
    return None

def capture_real_frame_simple(camera_config: dict, width: int = 640, height: int = 480):
    """Capture simplifiée et directe - pas de cache complexe."""
    source_url = camera_config['source']
    camera_name = camera_config.get('name', 'Camera')
    
    print(f"DEBUG SIMPLE: Capture pour {camera_name} depuis {source_url}")
    
    # 1. FLUX HTTP/MJPEG - méthode directe
    if source_url.startswith('http'):
        print(f"DEBUG SIMPLE: Tentative HTTP native pour {source_url}")
        try:
            frame = capture_mjpeg_frame_simple(source_url, width, height)
            if frame is not None:
                print(f"DEBUG SIMPLE: HTTP réussi - shape: {frame.shape}")
                # Overlay minimal
                timestamp = datetime.now().strftime("%H:%M:%S")
                cv2.putText(frame, f"{camera_name}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"HTTP | {timestamp}", (10, height - 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                return frame
            else:
                print(f"DEBUG SIMPLE: HTTP échoué pour {source_url}")
        except Exception as e:
            print(f"DEBUG SIMPLE: Exception HTTP {source_url}: {e}")
    
    # 2. WEBCAM/RTSP - OpenCV direct
    else:
        print(f"DEBUG SIMPLE: Tentative OpenCV pour {source_url}")
        try:
            # Sélection backend selon type
            if source_url.isdigit():
                # Webcam
                backends = [cv2.CAP_ANY, cv2.CAP_V4L2]
                source_index = int(source_url)
            else:
                # RTSP/fichier
                backends = [cv2.CAP_FFMPEG, cv2.CAP_ANY]
                source_index = source_url
                if source_url.startswith('rtsp') and 'tcp' not in source_url.lower():
                    separator = '&' if '?' in source_url else '?'
                    source_index = f"{source_url}{separator}tcp"
            
            for backend in backends:
                print(f"DEBUG SIMPLE: Test backend {backend} pour {source_index}")
                try:
                    cap = cv2.VideoCapture(source_index, backend)
                    if cap.isOpened():
                        ret, frame = cap.read()
                        if ret and frame is not None and frame.size > 0:
                            print(f"DEBUG SIMPLE: OpenCV réussi - shape: {frame.shape}")
                            
                            # Redimensionnement si nécessaire
                            if frame.shape[:2] != (height, width):
                                frame = cv2.resize(frame, (width, height))
                            
                            # Overlay minimal
                            timestamp = datetime.now().strftime("%H:%M:%S")
                            cv2.putText(frame, f"{camera_name}", (10, 30), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            cv2.putText(frame, f"CV2 | {timestamp}", (10, height - 15), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                            
                            cap.release()
                            return frame
                        else:
                            print(f"DEBUG SIMPLE: Pas de frame valide avec backend {backend}")
                    else:
                        print(f"DEBUG SIMPLE: Ouverture échouée avec backend {backend}")
                    cap.release()
                except Exception as e:
                    print(f"DEBUG SIMPLE: Exception backend {backend}: {e}")
        except Exception as e:
            print(f"DEBUG SIMPLE: Exception OpenCV globale: {e}")
    
    # 3. Frame d'erreur si tout échoue
    print(f"DEBUG SIMPLE: Échec total pour {source_url}")
    error_frame = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(error_frame, f"ECHEC: {camera_name}", (10, height//2 - 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(error_frame, f"Source: {source_url[:30]}...", (10, height//2 + 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    cv2.putText(error_frame, datetime.now().strftime("%H:%M:%S"), (10, height - 15), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
    return error_frame

def capture_real_frame(camera_config: dict, width: int = 640, height: int = 480):
    """Wrapper principal - utilise capture threadée optimisée avec fallback."""
    # Priorité 1: Capture threadée pour fluidité optimale
    return capture_frame_threaded(camera_config, width, height)
    
    # OpenCV pour RTSP/locaux uniquement (PAS HTTP)
    if not source_url.startswith('http'):
        # Sélection du backend selon le type et ce qui a marché au test
        if source_url.isdigit():
            # Webcam : utiliser backend natif, éviter FFMPEG
            if backend_tested in ['V4L2', 'DSHOW', 'DEFAULT']:
                backend_map = {
                    'V4L2': cv2.CAP_V4L2,
                    'DSHOW': cv2.CAP_DSHOW, 
                    'DEFAULT': cv2.CAP_ANY,
                    'GSTREAMER': cv2.CAP_GSTREAMER
                }
            else:
                # Forcer DEFAULT si backend testé pas adapté webcam
                backend_tested = 'DEFAULT'
                backend_map = {'DEFAULT': cv2.CAP_ANY}
        else:
            # RTSP/fichier : FFMPEG OK
            backend_map = {
                'FFMPEG': cv2.CAP_FFMPEG,
                'DEFAULT': cv2.CAP_ANY,
                'GSTREAMER': cv2.CAP_GSTREAMER,
                'V4L2': cv2.CAP_V4L2,
                'DSHOW': cv2.CAP_DSHOW
            }
        
        try:
            cap = cv2.VideoCapture()
            
            # Configuration optimisée selon le type
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # URL finale selon le type
            final_url = source_url
            
            # Optimisations RTSP
            if source_url.startswith('rtsp'):
                # Transport TCP pour RTSP
                if 'tcp' not in source_url.lower():
                    separator = '&' if '?' in source_url else '?'
                    final_url += f"{separator}tcp"
                
                # FPS plus bas pour RTSP
                cap.set(cv2.CAP_PROP_FPS, 15)
            elif source_url.isdigit():
                # Webcam - convertir en entier
                final_url = int(source_url)
                cap.set(cv2.CAP_PROP_FPS, 30)
                # Configuration webcam
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
            else:
                # Fichier local
                cap.set(cv2.CAP_PROP_FPS, 25)
            
            backend_code = backend_map.get(backend_tested, cv2.CAP_FFMPEG)
            
            if cap.open(final_url, backend_code) and cap.isOpened():
                # Configuration résolution
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                
                # Vide buffer pour frame récente (seulement pour RTSP)
                if source_url.startswith('rtsp'):
                    for _ in range(2):
                        cap.read()
                
                # Capture
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    # Redimensionnement si nécessaire
                    if frame.shape[:2] != (height, width):
                        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
                    
                    # Overlay
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    cv2.putText(frame, f"{camera_config['name']}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, timestamp, (10, height - 15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(frame, f"{backend_tested} | {source_url[:4].upper()}", (10, height - 35), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    
                    cap.release()
                    return frame
            
            cap.release()
            
        except Exception as e:
            pass
    
    # Frame d'erreur
    error_frame = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(error_frame, f"ERREUR: {camera_config['name']}", (10, height//2), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(error_frame, "Flux non disponible", (10, height//2 + 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(error_frame, f"Source: {source_url[:50]}...", (10, height//2 + 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (128, 128, 128), 1)
    
    return error_frame

def generate_dummy_frame(camera_id: str, width: int = 320, height: int = 240):
    """Génère une frame d'erreur pour caméra indisponible (fallback)."""
    img = np.zeros((height, width, 3), dtype=np.uint8)  # Frame noire au lieu d'aléatoire
    
    # Informations de base
    cv2.putText(img, f"Camera {camera_id}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    timestamp = datetime.now().strftime("%H:%M:%S")
    cv2.putText(img, timestamp, (10, height - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Création des données de frame pour analyse
    frame_data = FrameData(
        camera_id=camera_id,
        frame=img,
        timestamp=datetime.now(),
        frame_number=1,
        metadata={'resolution': (width, height)}
    )
    
    # Analyse avec pipeline VLM
    if st.session_state.surveillance_active:
        analysis_result = generate_real_frame_analysis(frame_data)
        
        if analysis_result and analysis_result.detections:
            # Ajout des détections visuelles
            for detection in analysis_result.detections:
                bbox = detection.get('bbox', [])
                if len(bbox) >= 4:
                    x1, y1, x2, y2 = bbox[:4]
                    # Assurer que les coordonnées sont dans les limites
                    x1, y1 = max(0, min(x1, width-1)), max(0, min(y1, height-1))
                    x2, y2 = max(x1+1, min(x2, width)), max(y1+1, min(y2, height))
                    
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(img, f"{detection['type']}", (int(x1), int(y1)-5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    return img

def render_integrated_chat(chat_type: str, context_data: Dict = None):
    """Chat intégré avec vraies données VLM."""
    
    chat_key = f"{chat_type}_chat"
    if chat_key not in st.session_state:
        st.session_state[chat_key] = []
    
    st.markdown("###  Chat IA avec Vraies Données VLM")
    
    # Questions prédéfinies selon le contexte
    if chat_type == "surveillance":
        questions = [
            "Analyse les détections VLM en temps réel",
            "Quels outils VLM sont les plus performants ?",
            "Résume l'optimisation adaptative actuelle",
            "Évalue les scores de performance des outils",
            "Recommande des améliorations de configuration"
        ]
        
        # Contexte avec vraies données
        if st.session_state.pipeline_initialized and st.session_state.real_pipeline:
            stats = st.session_state.real_pipeline.get_performance_stats()
            tool_details = st.session_state.real_pipeline.get_tool_performance_details()
            context_info = f" Pipeline active: {stats.get('frames_processed', 0)} frames, {len(stats.get('current_optimal_tools', []))} outils optimaux"
        else:
            context_info = " Pipeline VLM non initialisée - Analyse indisponible"
    
    elif chat_type == "video":
        questions = [
            "Analyse les outils VLM utilisés dans cette vidéo",
            "Compare les performances des différents outils",
            "Explique le processus d'optimisation adaptative",
            "Détaille les scores de confiance par outil",
            "Recommande la meilleure configuration d'outils"
        ]
        context_info = f" {len(st.session_state.uploaded_videos)} vidéos analysées avec pipeline VLM"
    
    # Affichage du contexte
    st.info(f" Contexte: {context_info}")
    
    # Zone de chat
    with st.container():
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        for message in st.session_state[chat_key][-8:]:
            if message['role'] == 'user':
                st.markdown(f"""
                <div class="chat-user">
                    <strong>Vous:</strong> {message['content']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-ai">
                    <strong> IA VLM:</strong> {message['content']}
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Interface de chat
    col1, col2 = st.columns([2, 1])
    
    with col1:
        user_input = st.text_input(
            f"Question sur {chat_type} VLM...", 
            key=f"chat_input_{chat_type}",
            placeholder=f"Ex: {questions[0]}"
        )
    
    with col2:
        selected_question = st.selectbox(
            "Questions VLM", 
            [""] + questions,
            key=f"quick_q_{chat_type}"
        )
    
    # Traitement des messages
    message_to_send = None
    if st.button(" Envoyer", key=f"send_{chat_type}") and user_input:
        message_to_send = user_input
    elif selected_question:
        if st.button(" Envoyer Question", key=f"send_quick_{chat_type}"):
            message_to_send = selected_question
    
    if message_to_send:
        # Ajouter la question
        st.session_state[chat_key].append({
            'role': 'user',
            'content': message_to_send,
            'timestamp': datetime.now()
        })
        
        # Générer réponse avec VLM thinking/reasoning
        with st.spinner(" Analyse VLM avec thinking..."):
            ai_response = asyncio.run(generate_real_vlm_response(message_to_send, chat_type, context_data))
        
        st.session_state[chat_key].append({
            'role': 'assistant',
            'content': ai_response,
            'timestamp': datetime.now()
        })
        
        st.rerun()

async def generate_real_vlm_response(question: str, chat_type: str, context_data: Dict) -> str:
    """Génère une réponse VLM intelligente avec thinking/reasoning."""
    
    if not PIPELINE_AVAILABLE:
        return " Pipeline VLM non disponible - Chat indisponible."
    
    # Récupération des vraies données pour contexte VLM
    vlm_context = {}
    if st.session_state.pipeline_initialized and st.session_state.real_pipeline:
        vlm_context = {
            'stats': st.session_state.real_pipeline.get_performance_stats(),
            'tools': st.session_state.real_pipeline.get_tool_performance_details(),
            'detections': st.session_state.real_detections[-10:],
            'optimizations': st.session_state.optimization_results[-5:],
            'alerts': st.session_state.real_alerts[-5:]
        }
    
    try:
        #  APPEL VLM CHATBOT AVEC THINKING/REASONING
        response_data = await process_vlm_chat_query(
            question=question,
            chat_type=chat_type, 
            vlm_context=vlm_context
        )
        
        # Formatage pour interface chat Streamlit
        if response_data.get("type") == "vlm_thinking":
            # Réponse VLM complète avec thinking
            response_text = f""" **Analyse VLM avec Thinking:**

**Processus de Raisonnement:**
{response_data.get('thinking', 'Thinking non disponible')[:300]}...

**[ANALYSIS] Analyse Technique:**
{response_data.get('analysis', 'Analyse non disponible')[:200]}...

**[RESPONSE] Réponse:**
{response_data.get('response', 'Réponse non disponible')}

**[DETAILS] Détails Techniques:**
{response_data.get('technical_details', 'Détails non disponibles')}

**[RECOMMENDATIONS] Recommandations:**
{' | '.join(response_data.get('recommendations', [])[:3])}

**[CONFIDENCE] Confiance:** {response_data.get('confidence', 0):.1%} | **[QUALITY] Qualité Données:** {response_data.get('data_quality', 'medium')}"""
            
            return response_text
            
        else:
            # Fallback ou réponse basique
            return response_data.get("response", " Réponse VLM générée.")
            
    except Exception as e:
        # Fallback sur ancien système si erreur VLM
        logger.error(f"Erreur chatbot VLM: {e}")
        return f" Erreur VLM chatbot: {str(e)}. Utilisant fallback basique."

def render_surveillance_tab():
    """Onglet surveillance avec vraie intégration VLM."""
    st.subheader(" Surveillance Temps Réel avec Pipeline VLM")
    
    # Statut de la pipeline
    render_pipeline_status()
    
    # Grille des caméras
    if not st.session_state.cameras:
        st.info(" Aucune caméra configurée. Ajoutez une caméra dans l'onglet Configuration.")
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
                    <h4> {camera['name']}</h4>
                </div>
                """, unsafe_allow_html=True)
                
                if camera.get('active', False) and st.session_state.surveillance_active:
                    # MÉTHODE SIMPLIFIÉE - capture directe sans cache complexe
                    camera_id = camera['id']
                    
                    try:
                        print(f"DEBUG: Capture FLUIDE pour caméra {camera_id}")
                        # NOUVEAU: Capture avec cache ultrarapide pour fluidité maximale
                        frame = get_fluid_frame(camera_id, camera, max_age_ms=50)
                        
                        if frame is not None:
                            st.image(frame, channels="BGR", caption=f" LIVE - {camera['name']}")
                            st.caption(f"[ACTIVE] Flux actif à {datetime.now().strftime('%H:%M:%S')}")
                        else:
                            st.error(f" Flux {camera['name']} indisponible")
                            # Fallback vers dummy  
                            frame = generate_dummy_frame(camera['id'])
                            st.image(frame, channels="BGR", caption=f"[FALLBACK] Fallback - {camera['name']}")
                            
                    except Exception as e:
                        st.error(f" Erreur capture {camera['name']}: {str(e)}")
                        print(f"DEBUG: Exception capture {camera_id}: {e}")
                        # Test direct disponible en cas d'erreur
                        if st.button(f" Test Direct", key=f"test_direct_{camera_id}"):
                            with st.spinner("Test direct en cours..."):
                                try:
                                    test_frame = capture_real_frame_simple(camera, width=320, height=240)
                                    if test_frame is not None:
                                        st.success(" Test direct réussi !")
                                        st.image(test_frame, channels="BGR", caption="Frame test")
                                    else:
                                        st.error(" Test direct échoué")
                                        st.info(f"Source: {camera.get('source')}")
                                except Exception as te:
                                    st.error(f"Test échoué: {te}")
                
                elif camera.get('active', False):
                    # Caméra active mais surveillance inactive
                    st.info(f" {camera['name']} prête - Démarrez la surveillance")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(" Pause", key=f"pause_{camera['id']}"):
                            st.session_state.cameras[camera['id']]['active'] = False
                            st.rerun()
                    
                    with col2:
                        if st.button(" Config", key=f"config_{camera['id']}"):
                            st.info(f"Configuration de {camera['name']}")
                
                else:
                    st.image("https://via.placeholder.com/320x240/cccccc/666666?text=Camera+Offline", 
                            caption=f"Caméra hors ligne - {camera['name']}")
                    
                    if st.button(" Start", key=f"start_{camera['id']}"):
                        st.session_state.cameras[camera['id']]['active'] = True
                        st.rerun()
    
    # Affichage des détections VLM récentes
    if st.session_state.real_detections:
        st.subheader(" Détections VLM Récentes")
        
        for detection in st.session_state.real_detections[-5:]:  # 5 dernières
            suspicion_color = 'red' if 'HIGH' in str(detection.suspicion_level) or 'CRITICAL' in str(detection.suspicion_level) else 'orange' if 'MEDIUM' in str(detection.suspicion_level) else 'green'
            
            st.markdown(f"""
            <div class="real-analysis-result">
                <h5>[DETECTION] Détection {detection.frame_id}</h5>
                <p><strong>Caméra:</strong> {detection.camera_id}</p>
                <p><strong>Niveau suspicion:</strong> <span style="color: {suspicion_color}">{detection.suspicion_level}</span></p>
                <p><strong>Confiance:</strong> {detection.confidence:.1%}</p>
                <p><strong>Description:</strong> {detection.description}</p>
                <p><strong>Outils utilisés:</strong> {', '.join(detection.tools_used[:3])}...</p>
                <p><strong>Temps traitement:</strong> {detection.processing_time:.2f}s</p>
                <p><strong>Score optimisation:</strong> {detection.optimization_score:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Chat intégré pour surveillance
    st.divider()
    context_data = {
        'pipeline_active': st.session_state.pipeline_initialized,
        'real_detections': len(st.session_state.real_detections),
        'real_alerts': len(st.session_state.real_alerts)
    }
    render_integrated_chat("surveillance", context_data)

def render_video_upload_tab():
    """Onglet upload vidéo avec vraie analyse VLM."""
    st.subheader(" Upload & Analyse Vidéo avec Pipeline VLM")
    
    # Statut pipeline
    render_pipeline_status()
    
    # Section d'upload avec description
    st.markdown("###  Upload de Vidéo avec Description")
    
    uploaded_file = st.file_uploader(
        "Sélectionnez une vidéo à analyser avec VLM",
        type=['mp4', 'avi', 'mov', 'mkv', 'webm'],
        help="La vidéo sera analysée avec la pipeline VLM complète (8 outils avancés)"
    )
    
    # Formulaire de description enrichi
    st.markdown("###  Description et Contexte Vidéo")
    
    col1, col2 = st.columns(2)
    
    with col1:
        video_title = st.text_input(
            "Titre/Nom de la vidéo",
            placeholder="Ex: Surveillance magasin - Caisse principale - 14h-16h",
            help="Nom descriptif pour identifier cette analyse"
        )
        
        video_location = st.selectbox(
            "Lieu de surveillance",
            ["Magasin/Commerce", "Entrepôt", "Bureau", "Parking", "Zone industrielle", 
             "Espace public", "Résidentiel", "Transport", "Autre"],
            help="Type d'environnement surveillé"
        )
        
        video_time_context = st.selectbox(
            "Contexte temporel",
            ["Heures ouverture", "Heures affluence", "Heures creuses", "Nuit/Fermeture",
             "Weekend", "Jour férié", "Événement spécial", "Période de crise", "Non spécifié"],
            help="Contexte temporal pour adapter l'analyse"
        )
    
    with col2:
        expected_activity = st.multiselect(
            "Activités attendues (normales)",
            ["Clients shopping", "Personnel travail", "Livraisons", "Nettoyage", 
             "Maintenance", "Circulation véhicules", "Activités bureau", "Surveillance"],
            help="Activités considérées comme normales dans ce contexte"
        )
        
        suspicious_focus = st.multiselect(
            "Focus surveillance (à détecter)",
            ["Vol à l'étalage", "Intrusion", "Comportements agressifs", "Objets abandonnés",
             "Accès non autorisé", "Vandalisme", "Activités inhabituelles", "Mouvements suspects"],
            help="Types de comportements suspects à prioriser"
        )
        
        camera_angle = st.selectbox(
            "Angle/Position caméra",
            ["Vue plongeante", "Vue niveau", "Vue latérale", "Vue face", "Vue multi-angles", "Non spécifié"],
            help="Perspective de la caméra pour optimiser l'analyse"
        )
    
    # Description libre détaillée
    video_description = st.text_area(
        "Description détaillée du contexte",
        placeholder="""Décrivez le contexte spécifique de cette vidéo:
        
• Situation particulière ou événements en cours
• Éléments d'environnement importants (layout, éclairage, foule)  
• Comportements spécifiques à surveiller
• Informations techniques (résolution, qualité, conditions)
• Objectifs d'analyse particuliers
• Contraintes ou défis attendus

Cette description aidera le VLM à mieux contextualiser son analyse...""",
        height=150,
        help="Description libre pour contextualiser l'analyse VLM"
    )
    
    # Configuration analyse avancée
    st.markdown("###  Configuration Analyse VLM")
    
    col3, col4 = st.columns(2)
    with col3:
        analysis_mode = st.selectbox(
            "Mode d'analyse VLM",
            ["Optimisation adaptative", "Tous les outils", "Outils sélectionnés", "Performance maximale"],
            help="Stratégie d'analyse selon contexte et ressources"
        )
        
        confidence_threshold = st.slider(
            "Seuil de confiance VLM", 
            0.1, 1.0, 0.7,
            help="Niveau de confiance minimum pour les détections"
        )
    
    with col4:
        analysis_priority = st.selectbox(
            "Priorité analyse",
            ["Précision maximale", "Vitesse optimisée", "Équilibré", "Économie ressources"],
            help="Compromis vitesse/précision selon urgence"
        )
        
        frame_sampling = st.selectbox(
            "Échantillonnage frames",
            ["Dense (toutes frames)", "Standard (1/2 frames)", "Rapide (1/5 frames)", "Clés seulement"],
            help="Densité d'analyse selon durée vidéo"
        )
    
    # Sélection d'outils spécifiques
    if analysis_mode == "Outils sélectionnés":
        available_tools = [
            'sam2_segmentator', 'dino_features', 'pose_estimator',
            'trajectory_analyzer', 'multimodal_fusion', 'temporal_transformer',
            'adversarial_detector', 'domain_adapter'
        ]
        
        selected_tools = st.multiselect(
            "Sélectionner les outils VLM",
            available_tools,
            default=available_tools[:4]
        )
    
    if uploaded_file is not None:
        # Informations du fichier
        file_details = {
            "Nom": uploaded_file.name,
            "Taille": f"{uploaded_file.size / 1024 / 1024:.2f} MB",
            "Type": uploaded_file.type
        }
        
        st.write("**Informations du fichier:**")
        st.json(file_details)
        
        # Bouton d'analyse VLM
        analyze_button_text = " Analyser avec Pipeline VLM" if st.session_state.pipeline_initialized else "🔬 Analyser (Mode Simulation)"
        
        if st.button(analyze_button_text, type="primary"):
            # Validation formulaire
            if not video_title.strip():
                st.error(" Veuillez saisir un titre pour la vidéo")
                return
            
            # Construction métadonnées enrichies pour VLM avec VideoContextMetadata
            form_data = {
                'title': video_title.strip(),
                'location_type': video_location,
                'time_context': video_time_context,
                'expected_activities': expected_activity,
                'suspicious_focus': suspicious_focus,
                'camera_angle': camera_angle,
                'detailed_description': video_description.strip(),
                'analysis_priority': analysis_priority,
                'frame_sampling': frame_sampling
            }
            
            # Création de l'objet VideoContextMetadata structuré
            if create_video_metadata_from_form:
                video_metadata_obj = create_video_metadata_from_form(form_data)
                video_metadata = video_metadata_obj.to_dict()
                
                # Intégration dans le système VLM
                if get_video_context_integration:
                    context_integration = get_video_context_integration()
                    base_chat_context = {'video_analysis_mode': True, 'timestamp': datetime.now()}
                    enhanced_context = context_integration.enhance_chat_context(
                        base_chat_context, video_metadata_obj
                    )
                else:
                    enhanced_context = {'video_analysis_mode': True, 'timestamp': datetime.now()}
            else:
                # Fallback si contexte vidéo non disponible
                video_metadata = form_data
                enhanced_context = {'video_analysis_mode': True, 'timestamp': datetime.now()}
            
            with st.spinner(" Analyse VLM contextualisée en cours..."):
                progress_bar = st.progress(0)
                
                # ✅ VRAIE ANALYSE VLM AVEC PIPELINE RÉELLE
                if st.session_state.pipeline_initialized and st.session_state.real_pipeline:
                    pipeline = st.session_state.real_pipeline
                    
                    # Traitement vidéo avec vraie pipeline VLM
                    video_file_bytes = uploaded_file.read()
                    
                    # Démarrage pipeline si pas encore active
                    if not pipeline.running:
                        pipeline.start_processing()
                    
                    # Extraction frames et analyse VLM frame par frame
                    # Création fichier temporaire pour OpenCV
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                        tmp_file.write(video_file_bytes)
                        tmp_file.flush()
                        
                        cap = cv2.VideoCapture(tmp_file.name)
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        fps = cap.get(cv2.CAP_PROP_FPS)
                    
                    analysis_results = {
                        'video_name': uploaded_file.name,
                        'video_metadata': video_metadata,
                        'analysis_mode': analysis_mode,
                        'pipeline_used': 'Real VLM Pipeline',
                        'total_frames': total_frames,
                        'frames_analyzed': 0,
                        'detections': [],
                        'tool_performance': {},
                        'optimization_data': {},
                        'summary': {},
                        'timestamp': datetime.now(),
                        'context_used': True
                    }
                    
                    # Analyse frame par frame avec vraie pipeline
                    frame_count = 0
                    real_analysis_results = []
                    
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        progress_bar.progress((frame_count + 1) / total_frames)
                        
                        # Création FrameData pour pipeline
                        frame_data = FrameData(
                            frame=frame,
                            camera_id=f"uploaded_video_{uploaded_file.name}",
                            frame_number=frame_count,
                            timestamp=datetime.now(),
                            metadata=video_metadata
                        )
                        
                        # Analyse VLM réelle
                        try:
                            analysis_task = asyncio.create_task(pipeline.analyze_frame(frame_data))
                            # Attendre analyse (pas de timeout - traitement complet)
                            real_result = asyncio.get_event_loop().run_until_complete(analysis_task)
                            
                            if real_result:
                                real_analysis_results.append(real_result)
                        except Exception as e:
                            st.warning(f"Erreur analyse frame {frame_count}: {e}")
                        
                        frame_count += 1
                        
                        # Échantillonnage basé sur frame_sampling pour éviter surcharge
                        if frame_sampling == "Élevé (1/10)":
                            for _ in range(9):
                                ret, _ = cap.read()
                                if not ret:
                                    break
                                frame_count += 9
                        elif frame_sampling == "Moyen (1/20)":
                            for _ in range(19):
                                ret, _ = cap.read()
                                if not ret:
                                    break
                                frame_count += 19
                    
                    cap.release()
                    
                    # Nettoyage fichier temporaire
                    try:
                        os.unlink(tmp_file.name)
                    except Exception as e:
                        st.warning(f"Impossible de supprimer fichier temporaire: {e}")
                    
                    # Récupération résultats pipeline
                    pipeline_results = pipeline.get_latest_results()
                    
                    # Construction résultats finaux depuis vraie pipeline
                    analysis_results['frames_analyzed'] = len(real_analysis_results)
                    
                    for real_result in real_analysis_results:
                        detection = {
                            'frame_number': real_result.frame_id.split('_')[-1],
                            'timestamp': real_result.timestamp.strftime("%H:%M:%S"),
                            'type': real_result.action_type.value if hasattr(real_result.action_type, 'value') else str(real_result.action_type),
                            'confidence': real_result.confidence,
                            'bbox': real_result.bbox_annotations[0] if real_result.bbox_annotations else [],
                            'tools_used': real_result.tools_used,
                            'optimization_score': real_result.optimization_score,
                            'description': real_result.description,
                            'suspicion_level': real_result.suspicion_level.value if hasattr(real_result.suspicion_level, 'value') else str(real_result.suspicion_level)
                        }
                        analysis_results['detections'].append(detection)
                    
                    # Performance par outil depuis vraie pipeline
                    pipeline_stats = pipeline.get_performance_stats()
                    tool_details = pipeline.get_tool_performance_details()
                    
                    analysis_results['tool_performance'] = tool_details.get('tool_usage_stats', {})
                    analysis_results['optimization_data'] = {
                        'optimal_combination': pipeline_stats.get('current_optimal_tools', []),
                        'performance_score': pipeline_stats.get('current_performance_score', 0.0),
                        'improvement_suggestions': []
                    }
                    
                    # Résumé depuis vraies données
                    suspicion_levels = [d['suspicion_level'] for d in analysis_results['detections']]
                    analysis_results['summary'] = {
                        'total_detections': len(analysis_results['detections']),
                        'high_confidence_detections': len([d for d in analysis_results['detections'] if d['confidence'] > 0.8]),
                        'suspicion_distribution': dict(Counter(suspicion_levels)),
                        'most_used_tools': pipeline_stats.get('current_optimal_tools', [])[:3],
                        'avg_processing_time': pipeline_stats.get('average_processing_time', 0.0),
                        'overall_performance_score': pipeline_stats.get('current_performance_score', 0.0)
                    }
                
                else:
                    # Fallback si pipeline non disponible
                    st.error("❌ Pipeline VLM réelle non initialisée. Impossible d'analyser la vidéo.")
                    return
                
                # ✅ NOUVEAU: Stockage avec système mémoire vidéo
                video_id = f"video_{len(st.session_state.uploaded_videos) + 1}_{int(datetime.now().timestamp())}"
                
                # Enrichissement avec données mémoire vidéo
                analysis_results['video_id'] = video_id
                analysis_results['video_name'] = uploaded_file.name
                analysis_results['analysis_time'] = analysis_results['summary']['avg_processing_time']
                analysis_results['frame_count'] = len(analysis_results.get('detections', []))
                
                # Création detailed_frames pour compatibilité mémoire
                detailed_frames = []
                for i, detection in enumerate(analysis_results.get('detections', [])):
                    frame_data = {
                        'frame_index': i,
                        'timestamp': i * (analysis_results['analysis_time'] / max(len(analysis_results['detections']), 1)),
                        'description': detection.get('description', ''),
                        'objects_detected': [{
                            'type': detection.get('object_type', 'objet'),
                            'confidence': detection.get('confidence', 0.7),
                            'count': 1
                        }] if detection.get('object_type') else [],
                        'behaviors': [{
                            'type': detection.get('behavior_type', 'normal'),
                            'confidence': detection.get('confidence', 0.7)
                        }] if detection.get('behavior_type') else [],
                        'confidence': detection.get('confidence', 0.7),
                        'suspicion_level': detection.get('suspicion_level', 'LOW'),
                        'tools_used': detection.get('tools_used', [])
                    }
                    detailed_frames.append(frame_data)
                
                analysis_results['detailed_frames'] = detailed_frames
                
                # Stockage traditionnel
                st.session_state.video_analysis_results[video_id] = analysis_results
                st.session_state.uploaded_videos.append({
                    'id': video_id,
                    'name': uploaded_file.name,
                    'upload_time': datetime.now()
                })
                
                progress_bar.progress(1.0)
                
                # ✅ NOUVEAU: Stockage dans système mémoire avancé
                if VIDEO_MEMORY_AVAILABLE:
                    try:
                        vlm_chat = get_vlm_chat()
                        memory_id = vlm_chat.link_video_analysis(video_id, analysis_results)
                        
                        st.success("✅ Analyse VLM terminée avec succès! Mémoire contextuelle activée.")
                        
                        # ✅ NOUVEAU: Section test conversation immédiat
                        with st.expander("💬 Tester la conversation contextuelle", expanded=True):
                            st.write("**Questions suggérées:**")
                            
                            col_test1, col_test2 = st.columns(2)
                            
                            with col_test1:
                                if st.button("Dit-moi exactement ce que tu as vu", key=f"detail_{video_id}"):
                                    test_response = vlm_chat.video_memory_system.query_video_memory(
                                        video_id, 
                                        "Dit-moi exactement ce que tu as vu dans la vidéo avec plus de détails"
                                    )
                                    st.write("🤖 **Réponse:**")
                                    st.write(test_response['response'])
                            
                            with col_test2:
                                if st.button("Combien de personnes ?", key=f"count_{video_id}"):
                                    test_response = vlm_chat.video_memory_system.query_video_memory(
                                        video_id,
                                        "Combien de personnes as-tu détectées dans la vidéo ?"
                                    )
                                    st.write("🤖 **Réponse:**") 
                                    st.write(test_response['response'])
                            
                            # Input libre pour test
                            custom_question = st.text_input(
                                "Posez votre propre question sur cette vidéo:",
                                placeholder="Ex: À quel moment les personnes ont-elles touché des produits ?",
                                key=f"custom_{video_id}"
                            )
                            
                            if custom_question:
                                test_response = vlm_chat.video_memory_system.query_video_memory(
                                    video_id, custom_question
                                )
                                
                                st.write("🤖 **Réponse:**")
                                st.write(test_response['response'])
                                
                                # Métadonnées debug
                                with st.expander("Debug Info"):
                                    st.json(test_response['metadata'])
                    
                    except Exception as e:
                        st.warning(f"⚠️ Mémoire vidéo non disponible: {e}")
                        st.success("✅ Analyse VLM terminée avec succès!")
                else:
                    st.success("✅ Analyse VLM terminée avec succès!")
    
    # Affichage des résultats d'analyse VLM
    if st.session_state.video_analysis_results:
        st.markdown("###  Résultats d'Analyse VLM")
        
        # Sélection de l'analyse
        video_options = {k: v['video_name'] for k, v in st.session_state.video_analysis_results.items()}
        selected_video = st.selectbox(
            "Sélectionner une analyse VLM",
            list(video_options.keys()),
            format_func=lambda x: video_options[x]
        )
        
        if selected_video:
            results = st.session_state.video_analysis_results[selected_video]
            
            # Résumé de l'analyse VLM
            st.markdown(f"""
            <div class="real-analysis-result">
                <h4> Analyse VLM - {results['video_name']}</h4>
                <p><strong>Pipeline:</strong> {results['pipeline_used']}</p>
                <p><strong>Mode:</strong> {results['analysis_mode']}</p>
                <p><strong>Frames analysées:</strong> {results['frames_analyzed']}</p>
                <p><strong>Score performance global:</strong> {results['summary']['overall_performance_score']:.2f}</p>
                <p><strong>Détections totales:</strong> {results['summary']['total_detections']}</p>
                <p><strong>Haute confiance:</strong> {results['summary']['high_confidence_detections']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Performance des outils VLM
            st.markdown("### [PERFORMANCE] Performance des Outils VLM")
            
            for tool, perf in results['tool_performance'].items():
                st.markdown(f"""
                <div class="tool-performance">
                    <strong>{tool}</strong><br>
                    Utilisations: {perf['usage_count']} | 
                    Succès: {perf['success_rate']:.1%} | 
                    Confiance moyenne: {perf['avg_confidence']:.1%} | 
                    Temps: {perf['processing_time']:.2f}s
                </div>
                """, unsafe_allow_html=True)
            
            # Données d'optimisation
            st.markdown("###  Optimisation Adaptative")
            opt_data = results['optimization_data']
            
            st.markdown(f"""
            <div class="optimization-badge">
                Combinaison optimale: {', '.join(opt_data['optimal_combination'])}
            </div>
            """, unsafe_allow_html=True)
            
            st.write("**Suggestions d'amélioration:**")
            for suggestion in opt_data['improvement_suggestions']:
                st.write(f"• {suggestion}")
            
            # Graphiques de performance
            col1, col2 = st.columns(2)
            
            with col1:
                # Performance par outil
                tool_names = list(results['tool_performance'].keys())
                success_rates = [results['tool_performance'][tool]['success_rate'] for tool in tool_names]
                
                fig = px.bar(
                    x=tool_names,
                    y=success_rates,
                    title="Taux de Succès par Outil VLM",
                    labels={'x': 'Outils VLM', 'y': 'Taux de Succès'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Distribution des niveaux de suspicion
                suspicion_dist = results['summary']['suspicion_distribution']
                
                fig = px.pie(
                    values=list(suspicion_dist.values()),
                    names=list(suspicion_dist.keys()),
                    title="Distribution Niveaux de Suspicion"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Export des résultats VLM
            if st.button(" Exporter Résultats VLM"):
                st.download_button(
                    label=" Télécharger Analyse VLM JSON",
                    data=json.dumps(results, indent=2, default=str),
                    file_name=f"vlm_analysis_{selected_video}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
    
    # Chat intégré pour analyse vidéo
    st.divider()
    context_data = {
        'pipeline_active': st.session_state.pipeline_initialized,
        'video_analyses': len(st.session_state.video_analysis_results)
    }
    render_integrated_chat("video", context_data)

def test_mjpeg_stream_native(url: str, timeout: int = 10) -> Dict[str, Any]:
    """Test natif robuste pour flux MJPEG HTTP."""
    import requests
    import io
    from PIL import Image
    import numpy as np
    
    result = {
        'success': False,
        'backend_used': 'HTTP_NATIVE',
        'resolution': None,
        'error_messages': [],
        'test_duration': 0
    }
    
    start_time = time.time()
    
    try:
        # Headers optimisés pour MJPEG
        headers = {
            'User-Agent': 'Mozilla/5.0 (compatible; SurveillanceBot/1.0)',
            'Accept': 'multipart/x-mixed-replace, */*',
            'Connection': 'close'  # Force nouvelle connexion
        }
        
        # Test simple d'abord
        response = requests.get(url, stream=True, timeout=timeout, headers=headers)
        
        if response.status_code == 200:
            content = b""
            chunks_read = 0
            
            for chunk in response.iter_content(chunk_size=2048):
                if not chunk:
                    continue
                    
                content += chunk
                chunks_read += 1
                
                # Recherche d'image JPEG complète
                jpg_start = content.find(b'\xff\xd8')
                if jpg_start != -1:
                    jpg_end = content.find(b'\xff\xd9', jpg_start)
                    if jpg_end != -1:
                        try:
                            jpeg_data = content[jpg_start:jpg_end+2]
                            
                            # Test de décodage
                            if len(jpeg_data) > 500:  # Taille minimale raisonnable
                                img = Image.open(io.BytesIO(jpeg_data))
                                
                                # Vérification dimensions
                                if img.width > 10 and img.height > 10:
                                    result['success'] = True
                                    result['resolution'] = f"{img.width}x{img.height}"
                                    response.close()
                                    break
                                    
                        except Exception as e:
                            result['error_messages'].append(f"Image decode: {str(e)}")
                
                # Sécurités pour éviter les blocages
                if len(content) > 200000:  # Limite buffer
                    content = content[-100000:]
                
                if chunks_read > 50:  # Limite chunks
                    break
                    
                if time.time() - start_time > timeout:  # Timeout
                    result['error_messages'].append("Test timeout")
                    break
            
            response.close()
            
        else:
            result['error_messages'].append(f"HTTP status: {response.status_code}")
            
    except requests.exceptions.Timeout:
        result['error_messages'].append("Connexion timeout")
    except requests.exceptions.ConnectionError:
        result['error_messages'].append("Erreur connexion réseau")
    except Exception as e:
        result['error_messages'].append(f"Erreur: {str(e)}")
    
    result['test_duration'] = time.time() - start_time
    return result

def check_webcam_permissions() -> Dict[str, Any]:
    """Vérifie les permissions et dispositifs webcam sur Linux."""
    import os
    import stat
    
    result = {
        'video_devices': [],
        'permissions_ok': True,
        'user_groups': [],
        'recommendations': []
    }
    
    try:
        # Vérification des périphériques /dev/video*
        for i in range(10):
            video_dev = f"/dev/video{i}"
            if os.path.exists(video_dev):
                stat_info = os.stat(video_dev)
                result['video_devices'].append({
                    'device': video_dev,
                    'readable': os.access(video_dev, os.R_OK),
                    'writable': os.access(video_dev, os.W_OK),
                    'mode': oct(stat_info.st_mode)
                })
        
        # Vérification des groupes utilisateur
        try:
            import subprocess
            groups_output = subprocess.check_output(['groups'], text=True)
            result['user_groups'] = groups_output.strip().split()
            
            if 'video' not in result['user_groups']:
                result['permissions_ok'] = False
                result['recommendations'].append("Ajouter l'utilisateur au groupe 'video'")
                
        except:
            pass
            
    except:
        pass
    
    return result

def list_video_devices() -> List[Dict[str, Any]]:
    """Liste les dispositifs vidéo disponibles sur le système."""
    import os
    import glob
    
    devices = []
    
    # Recherche des dispositifs /dev/video*
    video_devices = glob.glob("/dev/video*")
    video_devices.sort()
    
    for device_path in video_devices:
        device_info = {
            'path': device_path,
            'index': None,
            'accessible': False,
            'name': 'Unknown',
            'capabilities': []
        }
        
        try:
            # Extraction de l'index
            index = int(device_path.replace('/dev/video', ''))
            device_info['index'] = index
            
            # Test d'accessibilité
            device_info['accessible'] = os.access(device_path, os.R_OK | os.W_OK)
            
            # Lecture des informations v4l2 si possible
            try:
                import subprocess
                result = subprocess.run(['v4l2-ctl', '--device', device_path, '--info'], 
                                      capture_output=True, text=True, timeout=2)
                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        if 'Card type' in line:
                            device_info['name'] = line.split(':')[-1].strip()
                        elif 'Capabilities' in line:
                            device_info['capabilities'] = line.split(':')[-1].strip().split()
            except:
                pass
                
        except Exception:
            pass
        
        devices.append(device_info)
    
    return devices

def detect_available_cameras() -> List[int]:
    """Détecte les webcams disponibles avec diagnostic complet."""
    available_cameras = []
    
    print("DEBUG: Détection des webcams...")
    
    # 1. Liste des dispositifs système
    video_devices = list_video_devices()
    print(f"DEBUG: Dispositifs trouvés: {video_devices}")
    
    # 2. Test des indices avec backends appropriés
    backends_to_test = [
        ('DEFAULT', cv2.CAP_ANY),
        ('V4L2', cv2.CAP_V4L2), 
        ('GSTREAMER', cv2.CAP_GSTREAMER)
    ]
    
    # Test étendu jusqu'à 10 pour couvrir tous les /dev/video*
    indices_to_test = list(range(10))
    
    # Prioriser les indices correspondant aux dispositifs détectés
    device_indices = [d['index'] for d in video_devices if d['index'] is not None and d['accessible']]
    if device_indices:
        indices_to_test = device_indices + [i for i in indices_to_test if i not in device_indices]
    
    for index in indices_to_test:
        print(f"DEBUG: Test index {index}")
        
        for backend_name, backend_code in backends_to_test:
            try:
                print(f"DEBUG: Test {backend_name} pour index {index}")
                cap = cv2.VideoCapture(index, backend_code)
                
                if cap.isOpened():
                    print(f"DEBUG: Ouverture réussie {backend_name} index {index}")
                    
                    # Test de capture avec timeout
                    import signal
                    def timeout_handler(signum, frame):
                        raise TimeoutError("Timeout capture")
                    
                    try:
                        signal.signal(signal.SIGALRM, timeout_handler)
                        signal.alarm(3)  # 3 secondes timeout
                        
                        ret, frame = cap.read()
                        signal.alarm(0)  # Cancel timeout
                        
                        if ret and frame is not None and frame.size > 0:
                            print(f"DEBUG: Capture réussie {backend_name} index {index} - shape: {frame.shape}")
                            available_cameras.append(index)
                            cap.release()
                            break
                        else:
                            print(f"DEBUG: Capture échouée {backend_name} index {index}")
                            
                    except TimeoutError:
                        print(f"DEBUG: Timeout capture {backend_name} index {index}")
                    except Exception as e:
                        print(f"DEBUG: Erreur capture {backend_name} index {index}: {e}")
                    finally:
                        signal.alarm(0)
                else:
                    print(f"DEBUG: Ouverture échouée {backend_name} index {index}")
                
                cap.release()
                
            except Exception as e:
                print(f"DEBUG: Exception {backend_name} index {index}: {e}")
    
    result = list(set(available_cameras))
    print(f"DEBUG: Caméras détectées: {result}")
    return result

def test_camera_connection(source_url: str, timeout: int = 15) -> Dict[str, Any]:
    """Teste la connexion caméra avec fallbacks multiples.""" 
    import cv2
    import threading
    import time
    import requests
    
    # FORCER HTTP natif pour tous les flux HTTP (MJPEG détecté ou non)
    if source_url.startswith('http'):
        mjpeg_result = test_mjpeg_stream_native(source_url, timeout=10)
        # Retourner le résultat même si échec - PAS de fallback OpenCV pour HTTP
        return mjpeg_result
    
    result = {
        'success': False,
        'backend_used': None,
        'resolution': None,
        'error_messages': [],
        'test_duration': 0
    }
    
    start_time = time.time()
    
    # Test HTTP basique
    if source_url.startswith('http'):
        try:
            response = requests.head(source_url, timeout=5)
            if response.status_code != 200:
                result['error_messages'].append(f"HTTP: Status {response.status_code}")
            else:
                result['error_messages'].append("HTTP: URL accessible")
        except Exception as e:
            result['error_messages'].append(f"HTTP: {str(e)}")
    
    # OpenCV uniquement pour flux RTSP/locaux (PAS HTTP)
    if not source_url.startswith('http'):
        # Backends adaptés selon le type de source
        if source_url.isdigit():
            # Pour webcam : éviter FFMPEG, préférer backends natifs
            backends = [
                ('V4L2', cv2.CAP_V4L2),      # Linux webcam
                ('DSHOW', cv2.CAP_DSHOW),    # Windows webcam  
                ('DEFAULT', cv2.CAP_ANY),    # Backend par défaut
                ('GSTREAMER', cv2.CAP_GSTREAMER)
            ]
        else:
            # Pour RTSP/fichiers : FFMPEG OK
            backends = [
                ('FFMPEG', cv2.CAP_FFMPEG),
                ('DEFAULT', cv2.CAP_ANY),
                ('GSTREAMER', cv2.CAP_GSTREAMER)
            ]
        
        for backend_name, backend_code in backends:
            try:
                cap = cv2.VideoCapture()
                
                # Configuration optimisée pour RTSP
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FPS, 20)
                
                # Paramètres spéciaux pour RTSP
                if source_url.startswith('rtsp'):
                    # Ajouter transport TCP si pas déjà présent
                    rtsp_url = source_url
                    if 'tcp' not in rtsp_url.lower():
                        separator = '&' if '?' in rtsp_url else '?'
                        rtsp_url += f"{separator}tcp"
                    
                    if not cap.open(rtsp_url, backend_code):
                        result['error_messages'].append(f"{backend_name}: RTSP ouverture échouée")
                        cap.release()
                        continue
                elif source_url.isdigit():
                    # Webcam - conversion en entier
                    webcam_index = int(source_url)
                    if not cap.open(webcam_index, backend_code):
                        result['error_messages'].append(f"{backend_name}: Webcam {webcam_index} échouée")
                        cap.release()
                        continue
                else:
                    # Fichier local
                    if not cap.open(source_url, backend_code):
                        result['error_messages'].append(f"{backend_name}: Fichier échoué")
                        cap.release()
                        continue
                
                # Vérification connexion
                if not cap.isOpened():
                    result['error_messages'].append(f"{backend_name}: Capture fermée")
                    cap.release()
                    continue
                
                # Test de lecture avec timeout plus court
                frame_captured = False
                
                for attempt in range(5):  # Max 5 tentatives
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.size > 0:
                        # Vérification qualité frame
                        if frame.mean() > 5 and frame.std() > 1:  # Pas du bruit
                            result['success'] = True
                            result['backend_used'] = backend_name
                            result['resolution'] = f"{frame.shape[1]}x{frame.shape[0]}"
                            frame_captured = True
                            break
                    
                    time.sleep(0.2)  # Attente courte
                
                cap.release()
                
                if frame_captured:
                    break
                else:
                    result['error_messages'].append(f"{backend_name}: Pas de frame valide après 5 tentatives")
                
            except Exception as e:
                result['error_messages'].append(f"{backend_name}: {str(e)}")
                try:
                    cap.release()
                except:
                    pass
    else:
        # Pour les flux HTTP, on a déjà testé en natif - ne pas refaire OpenCV
        result['error_messages'].append("HTTP: Test déjà effectué en mode natif")
    
    result['test_duration'] = time.time() - start_time
    return result

def render_camera_config():
    """Configuration des caméras avec tests de connexion renforcés."""
    st.subheader(" Configuration des Caméras")
    
    with st.expander(" Ajouter une nouvelle caméra", expanded=len(st.session_state.cameras) == 0):
        col1, col2 = st.columns(2)
        
        with col1:
            cam_name = st.text_input("Nom de la caméra", f"Caméra {len(st.session_state.cameras) + 1}")
            cam_source_type = st.selectbox("Type de source", [
                "Webcam locale", 
                "Caméra IP (RTSP)", 
                "Flux MJPEG (HTTP)", 
                "Fichier vidéo"
            ])
        
        with col2:
            cam_resolution = st.selectbox("Résolution", ["640x480", "1280x720", "1920x1080"])
            cam_fps = st.slider("FPS", 15, 60, 30)
        
        # Configuration spécifique selon le type
        source_url = ""
        if cam_source_type == "Webcam locale":
            # Diagnostic complet des dispositifs vidéo
            col_detect1, col_detect2 = st.columns(2)
            
            with col_detect1:
                if st.button("[DETECT] Détecter webcams disponibles"):
                    with st.spinner("Détection complète en cours..."):
                        available_cams = detect_available_cameras()
                        if available_cams:
                            st.success(f" Webcams fonctionnelles: {available_cams}")
                        else:
                            st.error(" Aucune webcam fonctionnelle détectée")
            
            with col_detect2:
                if st.button("[DIAGNOSTIC] Diagnostic système vidéo"):
                    with st.spinner("Analyse des dispositifs..."):
                        devices = list_video_devices()
                        permissions = check_webcam_permissions()
                        
                        # Diagnostic USB et système
                        try:
                            import subprocess
                            
                            # Liste USB des caméras
                            usb_result = subprocess.run(['lsusb'], capture_output=True, text=True, timeout=5)
                            camera_usb_devices = []
                            if usb_result.returncode == 0:
                                for line in usb_result.stdout.split('\n'):
                                    if any(keyword in line.lower() for keyword in ['camera', 'webcam', 'video', 'imaging']):
                                        camera_usb_devices.append(line.strip())
                            
                            if camera_usb_devices:
                                st.write("**Caméras USB détectées:**")
                                for usb_cam in camera_usb_devices:
                                    st.write(f"kk {usb_cam}")
                            else:
                                st.warning("Aucune caméra USB détectée via lsusb")
                                
                        except Exception as e:
                            st.warning(f"Impossible d'exécuter lsusb: {e}")
                        
                        if devices:
                            st.write("**Dispositifs /dev/video* trouvés:**")
                            for device in devices:
                                access_icon = "[OK]" if device['accessible'] else "[FAIL]"
                                st.write(f"{access_icon} {device['path']} - {device['name']} (Index: {device['index']})")
                                
                                # Test des capacités
                                if device['capabilities']:
                                    st.write(f"   └── Capacités: {', '.join(device['capabilities'])}")
                        else:
                            st.error(" Aucun dispositif /dev/video* trouvé")
                        
                        # Vérification permissions
                        st.write("**Permissions:**")
                        if permissions['permissions_ok']:
                            st.write("[OK] Permissions OK")
                        else:
                            st.write(" Problèmes de permissions")
                            for rec in permissions['recommendations']:
                                st.write(f"   • {rec}")
                        
                        st.write(f"**Groupes utilisateur:** {', '.join(permissions['user_groups'])}")
                        
                        # Commandes de diagnostic
                        st.write("**Commandes de diagnostic utiles:**")
                        st.code("sudo usermod -a -G video $USER")
                        st.code("ls -la /dev/video*")
                        st.code("v4l2-ctl --list-devices")
            
            # Sélection de l'index avec options étendues
            available_indices = list(range(10))  # 0-9 pour couvrir tous les cas
            webcam_index = st.selectbox(
                "Index webcam", 
                options=available_indices,
                index=0,
                help="Index de la webcam. Vérifiez les dispositifs détectés ci-dessus."
            )
            source_url = str(webcam_index)
            
            # Informations selon les dispositifs détectés
            devices = list_video_devices()
            matching_device = next((d for d in devices if d['index'] == webcam_index), None)
            
            if matching_device:
                if matching_device['accessible']:
                    st.success(f" Dispositif trouvé: {matching_device['path']} - {matching_device['name']}")
                else:
                    st.error(f" Dispositif trouvé mais non accessible: {matching_device['path']}")
                    st.write(" Solution: Vérifiez les permissions ou ajoutez l'utilisateur au groupe 'video'")
            else:
                st.warning(f" Aucun dispositif /dev/video{webcam_index} détecté sur le système")
                st.write(" Le test tentera quand même la connexion avec les backends disponibles")
            
            st.info(f" Test utilisera l'index {webcam_index} avec backends: DEFAULT, V4L2, GSTREAMER")
        elif cam_source_type == "Caméra IP (RTSP)":
            source_url = st.text_input("URL RTSP", "rtsp://192.168.1.100:554/stream")
            st.info(" Format: rtsp://user:pass@ip:port/stream")
        elif cam_source_type == "Flux MJPEG (HTTP)":
            source_url = st.text_input("URL MJPEG", "http://192.168.1.100/mjpeg")
            st.info(" Exemple: http://38.79.156.188/CgiStart/nphMotionJpeg?Resolution=640x480")
        elif cam_source_type == "Fichier vidéo":
            video_file = st.file_uploader("Sélectionner vidéo", type=['mp4', 'avi', 'mov'])
            if video_file:
                source_url = video_file.name
        
        detection_sensitivity = st.slider("Sensibilité détection", 0.1, 1.0, 0.7)
        
        # Options de performance
        col_perf1, col_perf2 = st.columns(2)
        with col_perf1:
            connection_timeout = st.slider("Timeout connexion (s)", 5, 30, 15)
            refresh_rate = st.selectbox("Taux de rafraîchissement", [
                ("Temps réel (1s)", 1),
                ("Rapide (2s)", 2), 
                ("Normal (3s)", 3),
                ("Lent (5s)", 5)
            ], index=1)
        
        with col_perf2:
            quality_mode = st.selectbox("Mode qualité", [
                ("Faible latence", "low_latency"),
                ("Équilibré", "balanced"),
                ("Haute qualité", "high_quality")
            ], index=0)
            
            frame_skip = st.checkbox("Ignorer frames anciennes", value=True, 
                                   help="Améliore la réactivité en sautant les frames en retard")
        
        # Test de connexion avant ajout
        if source_url and st.button(" Tester Connexion", key="test_connection"):
            with st.spinner("Test de connexion en cours..."):
                test_result = test_camera_connection(source_url, connection_timeout)
                
                if test_result['success']:
                    st.success(f" Connexion réussie!")
                    st.info(f" Backend: {test_result['backend_used']}")
                    st.info(f" Résolution détectée: {test_result['resolution']}")
                    st.info(f" Temps de test: {test_result['test_duration']:.1f}s")
                else:
                    st.error(" Impossible de se connecter")
                    with st.expander("Détails des erreurs"):
                        for error in test_result['error_messages']:
                            st.write(f"• {error}")
        
        if st.button(" Ajouter Caméra"):
            if not source_url:
                st.error("Veuillez configurer la source de la caméra")
                return
            
            # Test automatique avant ajout
            with st.spinner("Vérification de la caméra..."):
                test_result = test_camera_connection(source_url, connection_timeout)
                
                if test_result['success']:
                    camera_id = f"cam_{len(st.session_state.cameras) + 1}"
                    st.session_state.cameras[camera_id] = {
                        'id': camera_id,
                        'name': cam_name,
                        'source': source_url,
                        'source_type': cam_source_type,
                        'resolution': cam_resolution,
                        'fps': cam_fps,
                        'sensitivity': detection_sensitivity,
                        'timeout': connection_timeout,
                        'backend_tested': test_result['backend_used'],
                        'refresh_rate': refresh_rate[1],
                        'quality_mode': quality_mode[1],
                        'frame_skip': frame_skip,
                        'active': False,
                        'created': datetime.now()
                    }
                    st.success(f" Caméra '{cam_name}' ajoutée avec succès!")
                    st.info(f" Backend optimal: {test_result['backend_used']}")
                    st.rerun()
                else:
                    st.error(" Impossible d'ajouter la caméra - connexion échouée")
                    with st.expander("Diagnostics d'erreur détaillés"):
                        for error in test_result['error_messages']:
                            st.write(f"• {error}")
                        
                        st.write("---")
                        st.write("**Solutions par type:**")
                        
                        if cam_source_type == "Webcam locale":
                            st.write(" **Webcam:**")
                            st.write("• Vérifiez que la webcam n'est pas utilisée par une autre app")
                            st.write("• Essayez des indices différents (0, 1, 2)")
                            st.write("• Utilisez 'Détecter webcams disponibles'")
                            st.write("• Sur Linux: vérifiez les permissions /dev/video*")
                        elif cam_source_type == "Flux MJPEG (HTTP)":
                            st.write("[NETWORK] **MJPEG HTTP:**")
                            st.write("• Vérifiez l'accessibilité de l'URL dans un navigateur")
                            st.write("• Testez sans authentification d'abord")
                            st.write("• Vérifiez que le flux retourne multipart/x-mixed-replace")
                        elif cam_source_type == "Caméra IP (RTSP)":
                            st.write(" **RTSP:**")
                            st.write("• Vérifiez user:pass@ip:port/stream")
                            st.write("• Testez avec VLC d'abord")
                            st.write("• Essayez d'ajouter ?tcp à la fin de l'URL")
                        
                        # Diagnostic système
                        st.write("---")
                        st.write("**Diagnostic système:**")
                        available_cams = detect_available_cameras()
                        if available_cams:
                            st.write(f" Webcams détectées: {available_cams}")
                        else:
                            st.write(" Aucune webcam système détectée")
    
    # Liste des caméras existantes
    if st.session_state.cameras:
        st.subheader(" Caméras Configurées")
        
        for camera_id, camera in st.session_state.cameras.items():
            with st.expander(f" {camera['name']} ({camera_id})"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write(f"**Type:** {camera.get('source_type', 'N/A')}")
                    st.write(f"**Source:** {camera['source']}")
                    st.write(f"**Résolution:** {camera['resolution']}")
                
                with col2:
                    st.write(f"**FPS:** {camera['fps']}")
                    st.write(f"**Sensibilité:** {camera['sensitivity']}")
                    if 'timeout' in camera:
                        st.write(f"**Timeout:** {camera['timeout']}s")
                
                with col3:
                    status = " Active" if camera.get('active') else "⭕ Inactive"
                    st.write(f"**Statut:** {status}")
                    
                    if 'backend_tested' in camera:
                        st.write(f"**Backend:** {camera['backend_tested']}")
                    
                    # Indicateurs de performance
                    if 'refresh_rate' in camera:
                        refresh_rate = camera['refresh_rate']
                        if refresh_rate == 1:
                            perf_icon = "[FAST]"
                        elif refresh_rate <= 2:
                            perf_icon = ""
                        elif refresh_rate <= 3:
                            perf_icon = "[OK]"
                        else:
                            perf_icon = "🐌"
                        st.write(f"**Performance:** {perf_icon} {refresh_rate}s")
                    
                    if 'quality_mode' in camera:
                        mode = camera['quality_mode']
                        mode_display = {
                            'low_latency': ' Faible latence',
                            'balanced': ' Équilibré',
                            'high_quality': ' Haute qualité'
                        }
                        st.write(f"**Mode:** {mode_display.get(mode, mode)}")
                    
                    # Boutons d'action
                    col_test, col_delete = st.columns(2)
                    
                    with col_test:
                        if st.button(" Re-tester", key=f"retest_{camera_id}"):
                            with st.spinner("Re-test en cours..."):
                                test_result = test_camera_connection(
                                    camera['source'], 
                                    camera.get('timeout', 15)
                                )
                                
                                if test_result['success']:
                                    st.success("[OK] Connexion OK")
                                    # Mise à jour du backend optimal
                                    st.session_state.cameras[camera_id]['backend_tested'] = test_result['backend_used']
                                    st.rerun()
                                else:
                                    st.error(" Connexion échouée")
                    
                    with col_delete:
                        if st.button(" Supprimer", key=f"delete_{camera_id}"):
                            del st.session_state.cameras[camera_id]
                            st.success(f"Caméra {camera['name']} supprimée")
                            st.rerun()
                
                # Affichage d'informations détaillées
                if st.button(" Détails techniques", key=f"details_{camera_id}"):
                    st.json({
                        'configuration': camera,
                        'created': camera.get('created', 'N/A').isoformat() if camera.get('created') else 'N/A'
                    })

def render_vlm_analytics():
    """Tableau de bord analytique avec métriques VLM réelles."""
    st.subheader(" Analytics VLM & Métriques Pipeline")
    
    # Métriques principales
    col1, col2, col3, col4 = st.columns(4)
    
    # Récupération des vraies métriques si disponibles
    if st.session_state.pipeline_initialized and st.session_state.real_pipeline:
        stats = st.session_state.real_pipeline.get_performance_stats()
        tool_details = st.session_state.real_pipeline.get_tool_performance_details()
        
        frames_processed = stats.get('frames_processed', 0)
        avg_processing_time = stats.get('average_processing_time', 0)
        optimization_cycles = stats.get('optimization_cycles', 0)
        performance_score = stats.get('current_performance_score', 0)
    else:
        frames_processed = 0
        avg_processing_time = 0
        optimization_cycles = 0
        performance_score = 0
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h2>{frames_processed}</h2>
            <p>Frames VLM</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h2>{avg_processing_time:.1f}s</h2>
            <p>Temps Moyen</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h2>{optimization_cycles}</h2>
            <p>Optimisations</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h2>{performance_score:.2f}</h2>
            <p>Score Performance</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Graphiques de performance VLM
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(" Utilisation des Outils VLM")
        
        if st.session_state.pipeline_initialized and st.session_state.real_pipeline:
            tool_usage = stats.get('tool_usage_stats', {})
            if tool_usage:
                tools = list(tool_usage.keys())
                usage_counts = list(tool_usage.values())
                
                fig = px.bar(
                    x=tools,
                    y=usage_counts,
                    title="Fréquence d'Utilisation des Outils",
                    labels={'x': 'Outils VLM', 'y': 'Utilisations'}
                )
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Aucune donnée d'utilisation d'outils disponible")
        else:
            st.info("Pipeline VLM non initialisée - Données d'utilisation indisponibles")
    
    with col2:
        st.subheader(" Performance Temporelle")
        
        # Données temporelles réelles uniquement
        hours = [f"{i:02d}:00" for i in range(24)]
        if st.session_state.real_detections:
            # Utilisation des vraies données de détection
            detection_times = [d.timestamp.hour for d in st.session_state.real_detections]
            hourly_counts = Counter(detection_times)
            performance_scores = [hourly_counts.get(i, 0) for i in range(24)]
        else:
            # Pas de données disponibles
            performance_scores = [0 for _ in range(24)]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hours,
            y=performance_scores,
            mode='lines+markers',
            name='Performance',
            line=dict(color='#007bff', width=2)
        ))
        
        fig.update_layout(
            title="Performance VLM sur 24h",
            xaxis_title="Heure",
            yaxis_title="Score de Performance",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Historique des optimisations
    if st.session_state.optimization_results:
        st.subheader(" Historique des Optimisations")
        
        for i, optimization in enumerate(st.session_state.optimization_results[-5:]):
            timestamp = optimization.get('timestamp', 'N/A')
            best_combo = optimization.get('best_combination', [])
            improvement = optimization.get('performance_improvement', 0)
            
            st.markdown(f"""
            <div class="optimization-badge">
                Optimisation #{i+1} - {timestamp}<br>
                Meilleure combinaison: {', '.join(best_combo[:3])}<br>
                Amélioration: +{improvement:.1%}
            </div>
            """, unsafe_allow_html=True)

def render_alerts_panel():
    """Panneau des alertes avec données VLM réelles."""
    st.subheader(" Centre des Alertes VLM")
    
    # Filtres
    col1, col2 = st.columns(2)
    with col1:
        alert_filter = st.selectbox("Filtrer par niveau", ["Tous", "CRITICAL", "HIGH", "MEDIUM", "LOW"])
    with col2:
        show_resolved = st.checkbox("Afficher alertes résolues", False)
    
    # Affichage des alertes réelles
    alerts_to_show = st.session_state.real_alerts if st.session_state.real_alerts else []
    
    if alert_filter != "Tous":
        alerts_to_show = [a for a in alerts_to_show if alert_filter in str(a.get('level', ''))]
    if not show_resolved:
        alerts_to_show = [a for a in alerts_to_show if not a.get('resolved')]
    
    if alerts_to_show:
        st.write(f"**{len(alerts_to_show)} alertes VLM trouvées**")
        
        for i, alert in enumerate(alerts_to_show[-10:]):  # 10 dernières
            level = str(alert.get('level', 'UNKNOWN'))
            level_class = f"alert-{level.lower()}" if level.lower() in ['critical', 'high', 'medium'] else "alert-medium"
            timestamp = alert.get('timestamp', datetime.now())
            timestamp_str = timestamp.strftime("%H:%M:%S") if hasattr(timestamp, 'strftime') else str(timestamp)
            
            confidence = alert.get('confidence', 0)
            tools_used = alert.get('tools_used', [])
            
            st.markdown(f"""
            <div class="{level_class}">
                <strong>{level}</strong> - {timestamp_str} - {alert.get('camera', 'N/A')}<br>
                {alert.get('message', 'Alerte VLM')}<br>
                <small>Confiance: {confidence:.1%} | Outils: {', '.join(tools_used[:2])}</small>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns([3, 1])
            with col2:
                if not alert.get('resolved') and st.button("[RESOLVE] Résoudre", key=f"resolve_real_alert_{i}"):
                    alert['resolved'] = True
                    st.rerun()
    else:
        st.info("[OK] Aucune alerte VLM active")

def main():
    """Application principale avec pipeline VLM réelle."""
    render_header()
    
    # Initialise les systèmes adaptatifs en arrière-plan
    initialize_adaptive_systems()
    
    # Sidebar avec contrôles VLM
    with st.sidebar:
        st.header("[CONTROLS] Contrôles Pipeline VLM")
        
        # Initialisation de la pipeline
        if not st.session_state.pipeline_initialized:
            if st.button(" Initialiser Pipeline VLM", type="primary"):
                success = asyncio.run(initialize_pipeline())
                if success:
                    st.rerun()
        else:
            st.success("[ACTIVE] Pipeline VLM Active")
            
            if st.button("[STOP] Arrêter Pipeline"):
                if st.session_state.real_pipeline:
                    st.session_state.real_pipeline.stop_processing()
                st.session_state.pipeline_initialized = False
                st.session_state.real_pipeline = None
                st.info(" Pipeline arrêtée")
                st.rerun()
        
        # Statut général
        st.divider()
        st.subheader("[STATUS] État Système")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Caméras", len(st.session_state.cameras))
        with col2:
            st.metric("Alertes VLM", len([a for a in st.session_state.real_alerts if not a.get('resolved')]))
        
        if st.session_state.pipeline_initialized and st.session_state.real_pipeline:
            stats = st.session_state.real_pipeline.get_performance_stats()
            st.metric("Frames VLM", stats.get('frames_processed', 0))
            st.metric("Performance", f"{stats.get('current_performance_score', 0):.2f}")
        
        # Statistiques captures threadées
        if st.session_state.threaded_captures:
            st.divider()
            st.subheader(" Performance Captures Threadées")
            
            capture_stats = get_threaded_capture_stats()
            
            total_fps = 0
            total_dropped = 0
            active_captures = 0
            
            for camera_id, stat in capture_stats.items():
                if not stat.get('error') and stat.get('running'):
                    active_captures += 1
                    total_fps += stat.get('fps', 0)
                    total_dropped += stat.get('frames_dropped', 0)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Captures Actives", active_captures)
            with col2:
                st.metric("FPS Total", f"{total_fps:.1f}")
            with col3:
                st.metric("Frames Perdues", total_dropped)
            
            # Détails par caméra
            if st.checkbox("Détails captures"):
                for camera_id, stat in capture_stats.items():
                    with st.expander(f" {camera_id}"):
                        if stat.get('error'):
                            st.error(f"Erreur: {stat['error']}")
                        else:
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("FPS", f"{stat.get('fps', 0):.1f}")
                            with col2:
                                st.metric("Frames OK", stat.get('frames_captured', 0))
                            with col3:
                                st.metric("Dropped", stat.get('frames_dropped', 0))
                            
                            status = " Active" if stat.get('running') else " Arrêtée"
                            st.write(f"Statut: {status}")
                            st.write(f"Queue: {stat.get('queue_size', 0)}")
        else:
            st.info("Aucune capture threadée active")
        
        # Monitoring réseau adaptatif
        st.divider()
        st.subheader("[NETWORK] Monitoring Réseau Adaptatif")
        
        # Affiche les métriques réseau si disponibles
        if 'network_metrics' in st.session_state and st.session_state.network_metrics:
            metrics = st.session_state.network_metrics
            current_time = time.time()
            
            # Vérifie si les métriques sont récentes (moins de 30s)
            if current_time - metrics.get('last_update', 0) < 30:
                # Icône de qualité réseau
                quality_icons = {
                    "excellent": "",
                    "good": "", 
                    "fair": "",
                    "poor": ""
                }
                quality_icon = quality_icons.get(metrics['quality'], "")
                
                st.write(f"**Qualité réseau:** {quality_icon} {metrics['quality'].title()}")
                
                # Métriques principales
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Latence", f"{metrics['latency']:.0f}ms")
                with col2:
                    st.metric("Vitesse", f"{metrics['speed']:.0f}KB/s")
                with col3:
                    st.metric("Stabilité", f"{metrics['stability_score']:.0f}%")
                
                # Paramètres adaptatifs actuels
                if st.checkbox("Paramètres adaptatifs"):
                    params = metrics.get('params', {})
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Résolution:** {params.get('resolution', 'N/A')}")
                        st.write(f"**FPS:** {params.get('fps', 'N/A')}")
                        st.write(f"**Compression:** {params.get('compression', 'N/A')}%")
                    with col2:
                        st.write(f"**Timeout:** {params.get('timeout', 'N/A')}s")
                        st.write(f"**Buffer:** {params.get('buffer_size', 'N/A')}")
                        st.write(f"**Retry:** {params.get('retry_count', 'N/A')}")
                
                # Statistiques compression
                if 'adaptive_compressor' in st.session_state:
                    compressor_stats = st.session_state.adaptive_compressor.get_compression_stats()
                    if compressor_stats['frames_processed'] > 0:
                        if st.checkbox("Stats compression"):
                            st.write(f"**Frames traitées:** {compressor_stats['frames_processed']}")
                            st.write(f"**Temps moyen:** {compressor_stats['avg_processing_time_ms']:.1f}ms")
                            st.write(f"**Qualité actuelle:** {compressor_stats['current_quality']}")
                            st.write(f"**Réduction taille:** {compressor_stats['avg_size_reduction']:.1f}x")
            else:
                st.warning(" Métriques réseau obsolètes")
        else:
            st.info("[NETWORK] Monitoring réseau en cours d'initialisation...")
        
        # Contrôles surveillance
        st.divider()
        st.subheader("[SURVEILLANCE] Surveillance")
        
        if st.button(" Démarrer Surveillance" if not st.session_state.surveillance_active else "[STOP] Arrêter Surveillance"):
            st.session_state.surveillance_active = not st.session_state.surveillance_active
            if st.session_state.surveillance_active:
                st.success("[STARTED] Surveillance démarrée!")
            else:
                st.info(" Surveillance arrêtée")
            st.rerun()
        
        # Paramètres VLM
        st.divider()
        st.subheader("[SETTINGS] Paramètres VLM")
        
        optimization_enabled = st.checkbox("Optimisation adaptative", True)
        confidence_threshold = st.slider("Seuil confiance VLM", 0.1, 1.0, 0.7)
        max_tools_per_analysis = st.slider("Max outils par analyse", 1, 8, 4)
        
        # Section contrôles alertes intégrés
        st.divider()
        st.subheader("Contrôles Alertes Intégrées")
        
        # Contrôles rapides
        col1, col2 = st.columns(2)
        with col1:
            audio_status = "✅" if st.session_state.alert_thresholds['audio_enabled'] and AUDIO_AVAILABLE else "❌"
            st.write(f"**Audio:** {audio_status}")
            
            auto_status = "✅" if st.session_state.alert_thresholds['auto_alerts_enabled'] else "❌"
            st.write(f"**Auto:** {auto_status}")
        
        with col2:
            st.metric("Seuil", f"{st.session_state.alert_thresholds['confidence_threshold']:.0%}")
            st.metric("Descriptions", len(st.session_state.auto_descriptions))
        
        # Test audio rapide
        if AUDIO_AVAILABLE:
            if st.button("Test Audio", key="sidebar_audio_test"):
                play_alert("MEDIUM", "Test sidebar", force=True)
        
        # Actions VLM
        st.divider()
        st.subheader("[ACTIONS] Actions VLM")
        
        if st.button("[OPTIMIZE] Forcer Optimisation"):
            if st.session_state.pipeline_initialized:
                st.info("[RUNNING] Cycle d'optimisation lancé...")
            else:
                st.warning(" Pipeline non initialisée")
        
        if st.button(" Vider Données VLM"):
            st.session_state.surveillance_chat.clear()
            st.session_state.video_chat.clear()
            st.session_state.real_alerts.clear()
            st.session_state.real_detections.clear()
            st.session_state.optimization_results.clear()
            cleanup_threaded_captures()  # Nettoie aussi les captures
            st.success(" Données VLM et captures vidées!")
            st.rerun()
    
    # Onglets principaux
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        " Surveillance VLM", 
        "📤 Upload Vidéo VLM",
        " Configuration", 
        "[ANALYTICS] Analytics VLM", 
        "[ALERTS] Alertes VLM",
        "Timeline & Descriptions",
        "🤖 Chat VLM Contextuel"  # ✅ NOUVEAU: Onglet chat avec mémoire
    ])
    
    with tab1:
        render_surveillance_tab()
        
        # Auto-refresh optimisé pour le streaming en arrière-plan
        if st.session_state.surveillance_active:
            active_cameras = [cam for cam in st.session_state.cameras.values() if cam.get('active')]
            if active_cameras:
                # Refresh plus rapide pour affichage des frames en cache
                min_refresh = min(cam.get('refresh_rate', 2) for cam in active_cameras)
                # Minimum 1s pour éviter surcharge, maximum 5s pour réactivité
                refresh_time = max(1, min(5, min_refresh))
            else:
                refresh_time = 3
            
            # Auto-refresh MAXIMAL - élimination des coupures
            time.sleep(0.1)  # Refresh ultra-rapide - 10 FPS UI
            st.rerun()
        else:
            # Pas de surveillance active - arrête tous les streams
            if st.session_state.streaming_manager:
                st.session_state.streaming_manager.stop_all()
            # Arrête aussi les captures threadées
            cleanup_threaded_captures()
    
    with tab2:
        render_video_upload_tab()
    
    with tab3:
        render_camera_config()
    
    with tab4:
        render_vlm_analytics()
    
    with tab5:
        render_alerts_panel()
    
    with tab6:
        st.header("Timeline & Descriptions Automatiques")
        
        # Configuration des alertes en haut
        with st.expander("Configuration Alertes", expanded=False):
            render_alert_controls()
        
        # Deux colonnes principales
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Timeline interactive
            render_detection_timeline()
            
        with col2:
            # Descriptions automatiques
            render_auto_descriptions()
    
    # ✅ NOUVEAU: Onglet Chat VLM Contextuel
    with tab7:
        st.header("🤖 Chat VLM Contextuel avec Mémoire Vidéo")
        
        # Vérification disponibilité système mémoire
        if VIDEO_MEMORY_AVAILABLE:
            try:
                vlm_chat = get_vlm_chat()
                
                # Indicateur contexte vidéo actuel
                if vlm_chat.current_video_context:
                    video_name = vlm_chat.current_video_context.get('video_name', 'Vidéo')
                    st.success(f"🎯 Contexte vidéo actif: {video_name}")
                    
                    # Métriques contexte
                    col_ctx1, col_ctx2, col_ctx3 = st.columns(3)
                    
                    with col_ctx1:
                        memory_stats = vlm_chat.video_memory_system.get_system_stats()
                        efficiency = memory_stats.get('memory_efficiency', 0)
                        st.metric("Efficacité mémoire", f"{efficiency:.1%}")
                    
                    with col_ctx2:
                        videos_stored = memory_stats.get('videos_stored', 0)
                        st.metric("Vidéos en mémoire", videos_stored)
                    
                    with col_ctx3:
                        if st.button("🔄 Désactiver contexte"):
                            vlm_chat.current_video_context = None
                            st.rerun()
                else:
                    st.info("💡 Uploadez d'abord une vidéo dans l'onglet 'Upload Vidéo VLM' pour activer la mémoire contextuelle")
                    
                    # Liste des vidéos disponibles
                    if st.session_state.video_analysis_results:
                        st.write("**Vidéos analysées disponibles:**")
                        
                        available_videos = []
                        for video_id, analysis in st.session_state.video_analysis_results.items():
                            video_name = analysis.get('video_name', video_id)
                            available_videos.append({
                                'id': video_id,
                                'name': video_name,
                                'analysis': analysis
                            })
                        
                        # Sélection vidéo pour activer contexte
                        if available_videos:
                            selected_video_idx = st.selectbox(
                                "Activer contexte pour une vidéo:",
                                range(len(available_videos)),
                                format_func=lambda x: available_videos[x]['name'] if x < len(available_videos) else "Aucune"
                            )
                            
                            if st.button("🔗 Activer ce contexte vidéo"):
                                selected_video = available_videos[selected_video_idx]
                                memory_id = vlm_chat.link_video_analysis(
                                    selected_video['id'], 
                                    selected_video['analysis']
                                )
                                st.success(f"✅ Contexte activé pour: {selected_video['name']}")
                                st.rerun()
                
                # Interface chat principale
                st.markdown("---")
                vlm_chat.render_chat_interface()
                
            except Exception as e:
                st.error(f"❌ Erreur système chat VLM: {e}")
                st.write("**Mode fallback:** Utilisez l'interface chat classique dans les autres onglets.")
        
        else:
            st.warning("⚠️ Système de mémoire vidéo non disponible")
            st.write("Fonctionnalités disponibles:")
            st.write("- Chat VLM de base")
            st.write("- Analyses vidéo classiques") 
            st.write("- Interface de surveillance standard")
            
            # Fallback vers chat intégré classique
            st.markdown("---")
            st.subheader("Chat VLM Classique")
            render_integrated_chat("video", {})

if __name__ == "__main__":
    main()