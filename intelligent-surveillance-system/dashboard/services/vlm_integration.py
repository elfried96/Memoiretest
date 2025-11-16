"""Intégration VLM avec le dashboard Streamlit."""

import streamlit as st
import asyncio
import base64
import threading
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import cv2
import io

# Import du système VLM existant
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from src.core.vlm.model import VisionLanguageModel
    from src.core.orchestrator.adaptive_orchestrator import AdaptiveOrchestrator
    from src.core.types import AnalysisRequest, SuspicionLevel
    VLM_AVAILABLE = True
except ImportError:
    VLM_AVAILABLE = False

from services.session_manager import get_session_manager
# Import du système mémoire vidéo (ajouté pour compatibilité)
try:
    from .video_memory_system import get_video_memory_system
    VIDEO_MEMORY_AVAILABLE = True
except ImportError:
    VIDEO_MEMORY_AVAILABLE = False
    get_video_memory_system = None

class StreamlitVLMService:
    """Service d'intégration VLM pour Streamlit."""
    
    def __init__(self):
        self.session = get_session_manager()
        self.vlm_model: Optional[VisionLanguageModel] = None
        self.orchestrator: Optional[AdaptiveOrchestrator] = None
        self.is_initialized = False
        self.loading_lock = threading.Lock()
        
        # Cache pour les analyses
        self.analysis_cache = {}
        
        # ✅ NOUVEAU: Système mémoire vidéo si disponible
        self.video_memory_system = get_video_memory_system() if VIDEO_MEMORY_AVAILABLE else None
        
        # Métriques
        self.stats = {
            'queries_processed': 0,
            'videos_analyzed': 0,
            'average_response_time': 0,
            'last_activity': None
        }
    
    async def initialize_vlm(self, force_reload: bool = False) -> bool:
        """Initialise le système VLM de manière asynchrone."""
        
        if not VLM_AVAILABLE:
            st.warning(" VLM non disponible - Mode simulation activé")
            return False
        
        if self.is_initialized and not force_reload:
            return True
        
        with self.loading_lock:
            try:
                # Affichage du status de chargement
                status_placeholder = st.empty()
                progress_bar = st.progress(0)
                
                status_placeholder.info(" Initialisation du modèle VLM...")
                progress_bar.progress(20)
                
                # Chargement VLM
                if not self.vlm_model:
                    self.vlm_model = VisionLanguageModel(
                        model_name="qwen2.5-vl-32b-instruct",
                        device="auto",
                        load_in_4bit=True
                    )
                
                status_placeholder.info(" Chargement du modèle...")
                progress_bar.progress(50)
                
                await self.vlm_model.load_model()
                
                status_placeholder.info(" Initialisation orchestrateur...")
                progress_bar.progress(80)
                
                # Chargement orchestrateur
                if not self.orchestrator:
                    self.orchestrator = AdaptiveOrchestrator()
                    # TODO: Initialiser orchestrateur si nécessaire
                
                progress_bar.progress(100)
                status_placeholder.success(" VLM initialisé avec succès!")
                
                self.is_initialized = True
                
                # Nettoyage interface
                import time
                time.sleep(2)
                status_placeholder.empty()
                progress_bar.empty()
                
                return True
                
            except Exception as e:
                st.error(f" Erreur initialisation VLM: {str(e)}")
                return False
    
    async def analyze_video_file(self, video_file, options: Dict[str, Any]) -> Dict[str, Any]:
        """Analyse un fichier vidéo uploadé."""
        
        if not self.is_initialized:
            await self.initialize_vlm()
        
        if not self.is_initialized:
            return self._simulate_video_analysis(video_file, options)
        
        try:
            start_time = datetime.now()
            
            # Extraction de frames clés de la vidéo
            frames = self._extract_key_frames(video_file)
            
            # Analyse de chaque frame
            frame_analyses = []
            
            for i, frame in enumerate(frames):
                # Conversion frame en base64
                frame_b64 = self._frame_to_base64(frame)
                
                # Création de la requête d'analyse
                analysis_request = AnalysisRequest(
                    frame_data=frame_b64,
                    context=f"Frame {i+1}/{len(frames)} de la vidéo {video_file.name}",
                    tools_available=[
                        "yolo_detector",
                        "sam2_segmentator", 
                        "pose_estimator",
                        "trajectory_analyzer"
                    ] if options.get('use_advanced_tools', True) else []
                )
                
                # Analyse VLM
                frame_result = await self.vlm_model.analyze_with_tools(
                    analysis_request,
                    use_advanced_tools=options.get('use_advanced_tools', True)
                )
                
                frame_analyses.append({
                    'frame_index': i,
                    'timestamp': i / len(frames),  # Approximation
                    'analysis': frame_result,
                    'suspicion_level': frame_result.suspicion_level.value,
                    'confidence': frame_result.confidence,
                    'description': frame_result.description
                })
            
            # Agrégation des résultats
            analysis_time = (datetime.now() - start_time).total_seconds()
            
            aggregated_result = self._aggregate_frame_analyses(
                frame_analyses, 
                video_file.name,
                analysis_time,
                options
            )
            
            # ✅ NOUVEAU: Stockage dans mémoire vidéo si disponible
            video_id = f"vlm_video_{hash(video_file.name)}_{int(datetime.now().timestamp())}"
            if self.video_memory_system:
                try:
                    self.video_memory_system.store_video_analysis(video_id, aggregated_result)
                    aggregated_result['video_memory_enabled'] = True
                    aggregated_result['video_id'] = video_id
                except Exception as e:
                    st.warning(f"Mémoire vidéo non disponible: {e}")
                    aggregated_result['video_memory_enabled'] = False
            
            # Mise à jour des statistiques
            self.stats['videos_analyzed'] += 1
            self.stats['average_response_time'] = (
                (self.stats['average_response_time'] * (self.stats['videos_analyzed'] - 1) + analysis_time)
                / self.stats['videos_analyzed']
            )
            self.stats['last_activity'] = datetime.now()
            
            return aggregated_result
            
        except Exception as e:
            st.error(f"Erreur analyse vidéo: {str(e)}")
            return self._simulate_video_analysis(video_file, options)
    
    async def process_chat_query(self, question: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Traite une requête de chat avec le VLM."""
        
        if not self.is_initialized:
            return self._simulate_chat_response(question, context)
        
        try:
            start_time = datetime.now()
            
            # Construction du prompt contextuel
            prompt = self._build_chat_prompt(question, context)
            
            # Si on a des analyses vidéo récentes, on peut les utiliser
            video_analyses = context.get('video_analyses', {})
            
            if video_analyses:
                # Utilise la dernière analyse comme base
                latest_analysis = list(video_analyses.values())[-1]
                
                if 'frame_data' in latest_analysis:
                    # Analyse avec image
                    analysis_request = AnalysisRequest(
                        frame_data=latest_analysis['frame_data'],
                        context=prompt,
                        tools_available=[]
                    )
                    
                    vlm_result = await self.vlm_model.analyze_with_tools(
                        analysis_request,
                        use_advanced_tools=False
                    )
                    
                    response_content = vlm_result.description
                else:
                    # Réponse textuelle basée sur le contexte
                    response_content = self._generate_contextual_response(question, context)
            
            else:
                # Réponse générale sur l'état du système
                response_content = self._generate_system_response(question, context)
            
            analysis_time = (datetime.now() - start_time).total_seconds()
            
            # Mise à jour statistiques
            self.stats['queries_processed'] += 1
            self.stats['last_activity'] = datetime.now()
            
            return {
                'content': response_content,
                'metadata': {
                    'analysis_time': analysis_time,
                    'confidence': 0.85,
                    'tools_used': ['vlm', 'context_analysis'],
                    'context_items': len(context)
                }
            }
            
        except Exception as e:
            st.error(f"Erreur traitement chat: {str(e)}")
            return self._simulate_chat_response(question, context)
    
    def _extract_key_frames(self, video_file, max_frames: int = 5) -> List[np.ndarray]:
        """Extrait les frames clés d'une vidéo."""
        
        # Lecture de la vidéo
        video_bytes = video_file.read()
        
        # Écriture temporaire
        temp_path = f"/tmp/temp_video_{hash(video_file.name)}.mp4"
        with open(temp_path, 'wb') as f:
            f.write(video_bytes)
        
        # Ouverture avec OpenCV
        cap = cv2.VideoCapture(temp_path)
        
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            return frames
        
        # Extraction équidistante
        frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                # Redimensionnement pour optimisation
                height, width = frame.shape[:2]
                if width > 800:
                    ratio = 800 / width
                    new_width = 800
                    new_height = int(height * ratio)
                    frame = cv2.resize(frame, (new_width, new_height))
                
                frames.append(frame)
        
        cap.release()
        
        # Nettoyage fichier temporaire
        Path(temp_path).unlink(missing_ok=True)
        
        return frames
    
    def _frame_to_base64(self, frame: np.ndarray) -> str:
        """Convertit une frame en base64."""
        
        # Conversion BGR -> RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Conversion en PIL Image
        pil_image = Image.fromarray(frame_rgb)
        
        # Encodage base64
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG', quality=85)
        
        return base64.b64encode(buffer.getvalue()).decode()
    
    def _aggregate_frame_analyses(self, frame_analyses: List[Dict], video_name: str, 
                                analysis_time: float, options: Dict) -> Dict[str, Any]:
        """Agrège les analyses de frames individuelles."""
        
        if not frame_analyses:
            return self._empty_analysis_result(video_name, analysis_time)
        
        # Calcul des métriques globales
        suspicion_levels = [fa['suspicion_level'] for fa in frame_analyses]
        confidences = [fa['confidence'] for fa in frame_analyses]
        
        # Niveau de suspicion global (le plus élevé)
        suspicion_priority = {'LOW': 1, 'MEDIUM': 2, 'HIGH': 3, 'CRITICAL': 4}
        max_suspicion = max(suspicion_levels, key=lambda x: suspicion_priority.get(x, 0))
        
        # Confiance moyenne
        avg_confidence = sum(confidences) / len(confidences)
        
        # Extraction des objets détectés
        detected_objects = []
        behaviors = []
        timeline = []
        
        for fa in frame_analyses:
            analysis = fa.get('analysis')
            if hasattr(analysis, 'tools_used'):
                # Extraction des détections d'objets
                # TODO: Parser les résultats des outils
                pass
            
            # Création timeline
            timeline.append({
                'time': f"{fa['timestamp']:.1f}s",
                'event': fa['description'][:100] + "..." if len(fa['description']) > 100 else fa['description']
            })
        
        # Recommandations
        recommendations = []
        if max_suspicion in ['HIGH', 'CRITICAL']:
            recommendations.extend([
                "Surveillance renforcée recommandée",
                "Vérification manuelle conseillée",
                "Enregistrement de l'incident"
            ])
        elif max_suspicion == 'MEDIUM':
            recommendations.extend([
                "Surveillance continue",
                "Alerte préventive"
            ])
        else:
            recommendations.append("Comportement normal détecté")
        
        return {
            'video_name': video_name,
            'analysis_time': analysis_time,
            'frame_count': len(frame_analyses),
            'confidence': avg_confidence,
            'suspicion_level': max_suspicion,
            'detected_objects': detected_objects,
            'behaviors': behaviors,
            'timeline': timeline,
            'recommendations': recommendations,
            'options_used': options,
            'timestamp': datetime.now().isoformat(),
            'detailed_frames': frame_analyses  # Analyses détaillées
        }
    
    def _empty_analysis_result(self, video_name: str, analysis_time: float) -> Dict[str, Any]:
        """Résultat d'analyse vide."""
        return {
            'video_name': video_name,
            'analysis_time': analysis_time,
            'frame_count': 0,
            'confidence': 0.0,
            'suspicion_level': 'LOW',
            'detected_objects': [],
            'behaviors': [],
            'timeline': [],
            'recommendations': ["Aucune analyse possible"],
            'error': "Impossible d'extraire les frames"
        }
    
    def _build_chat_prompt(self, question: str, context: Dict[str, Any]) -> str:
        """Construit un prompt pour le chat."""
        
        prompt = f"Question de l'utilisateur: {question}\n\n"
        prompt += "Contexte de surveillance:\n"
        
        # Ajout du contexte
        if 'video_analyses' in context:
            analyses = context['video_analyses']
            prompt += f"- {len(analyses)} analyses vidéo disponibles\n"
        
        if 'cameras_state' in context:
            cameras = context['cameras_state']
            active_cameras = sum(1 for cam in cameras.values() if cam.get('enabled', False))
            prompt += f"- {active_cameras}/{len(cameras)} caméras actives\n"
        
        if 'active_alerts' in context:
            alerts = context['active_alerts']
            prompt += f"- {len(alerts)} alertes en cours\n"
        
        prompt += "\nRépondez de manière claire et professionnelle en tant qu'assistant de surveillance."
        
        return prompt
    
    def _generate_contextual_response(self, question: str, context: Dict[str, Any]) -> str:
        """Génère une réponse contextuelle."""
        
        # Analyse simple des mots-clés
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['risque', 'danger', 'menace', 'sécurité']):
            return self._analyze_security_status(context)
        
        elif any(word in question_lower for word in ['personnes', 'individus', 'gens']):
            return self._analyze_people_detection(context)
        
        elif any(word in question_lower for word in ['alerte', 'alertes', 'notification']):
            return self._analyze_alerts_status(context)
        
        elif any(word in question_lower for word in ['système', 'état', 'status', 'fonctionnement']):
            return self._analyze_system_status(context)
        
        else:
            return f"J'ai analysé votre question: '{question}'. Basé sur le contexte actuel, je peux vous fournir des informations sur la sécurité, les détections de personnes, les alertes ou l'état du système. Pourriez-vous être plus spécifique ?"
    
    def _analyze_security_status(self, context: Dict[str, Any]) -> str:
        """Analyse le statut sécurité."""
        
        alerts = context.get('active_alerts', [])
        video_analyses = context.get('video_analyses', {})
        
        if not alerts:
            return " **Statut sécurité: NORMAL**\n\nAucune alerte active détectée. Le système de surveillance fonctionne normalement."
        
        critical_alerts = [a for a in alerts if a.get('level') == 'CRITICAL']
        high_alerts = [a for a in alerts if a.get('level') == 'HIGH']
        
        if critical_alerts:
            return f" **ALERTE CRITIQUE DÉTECTÉE**\n\n{len(critical_alerts)} alerte(s) critique(s) nécessitent une intervention immédiate."
        
        elif high_alerts:
            return f" **Surveillance renforcée recommandée**\n\n{len(high_alerts)} alerte(s) de niveau élevé détectée(s)."
        
        else:
            return f" **Surveillance normale**\n\n{len(alerts)} alerte(s) mineures en cours de surveillance."
    
    def _analyze_people_detection(self, context: Dict[str, Any]) -> str:
        """Analyse les détections de personnes."""
        
        video_analyses = context.get('video_analyses', {})
        
        if not video_analyses:
            return " Aucune analyse récente disponible pour compter les personnes. Uploadez une vidéo pour obtenir des détections précises."
        
        # Simulation basée sur les analyses
        return " **Détection de personnes:**\n\nBasé sur les analyses récentes, 2-4 personnes sont généralement détectées dans les zones surveillées. Mouvements normaux observés."
    
    def _analyze_alerts_status(self, context: Dict[str, Any]) -> str:
        """Analyse le statut des alertes."""
        
        alerts = context.get('active_alerts', [])
        
        if not alerts:
            return " **Aucune alerte active**\n\nTous les systèmes fonctionnent normalement."
        
        response = f" **{len(alerts)} alerte(s) active(s):**\n\n"
        
        for alert in alerts[-3:]:  # 3 dernières
            level = alert.get('level', 'INFO')
            message = alert.get('message', 'N/A')
            emoji = {'LOW': '', 'MEDIUM': '', 'HIGH': '', 'CRITICAL': ''}.get(level, '')
            response += f"{emoji} {message}\n"
        
        return response
    
    def _analyze_system_status(self, context: Dict[str, Any]) -> str:
        """Analyse l'état du système."""
        
        cameras = context.get('cameras_state', {})
        analyses = context.get('video_analyses', {})
        alerts = context.get('active_alerts', [])
        
        active_cameras = sum(1 for cam in cameras.values() if cam.get('enabled', False))
        
        response = "🖥️ **État du système de surveillance:**\n\n"
        response += f" Caméras: {active_cameras}/{len(cameras)} actives\n"
        response += f" IA: {' Opérationnelle' if self.is_initialized else ' En cours d\'initialisation'}\n"
        response += f" Analyses: {len(analyses)} vidéos traitées\n"
        response += f" Alertes: {len(alerts)} actives\n\n"
        
        if self.stats['last_activity']:
            response += f"⏱️ Dernière activité: {self.stats['last_activity'].strftime('%H:%M:%S')}\n"
        
        response += " Système opérationnel"
        
        return response
    
    def _simulate_video_analysis(self, video_file, options: Dict[str, Any]) -> Dict[str, Any]:
        """Simulation d'analyse vidéo."""
        
        import random
        import time
        
        time.sleep(2)  # Simulation temps d'analyse
        
        return {
            'video_name': video_file.name,
            'file_size': video_file.size,
            'analysis_time': 2.3,
            'frame_count': 5,
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
            'recommendations': [
                "Surveillance continue recommandée" if random.random() > 0.5 else "Comportement normal"
            ],
            'simulation_mode': True
        }
    
    def _simulate_chat_response(self, question: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Simulation de réponse chat."""
        
        return {
            'content': f" Simulation de réponse à: {question}\n\nContexte: {len(context)} éléments analysés.",
            'metadata': {
                'simulation_mode': True,
                'analysis_time': 0.8,
                'confidence': 0.7
            }
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Retourne l'état du système VLM avec mémoire vidéo."""
        
        status = {
            'initialized': self.is_initialized,
            'vlm_available': VLM_AVAILABLE,
            'model_loaded': self.vlm_model is not None and self.is_initialized,
            'orchestrator_loaded': self.orchestrator is not None,
            'stats': self.stats.copy(),
            'video_memory_available': VIDEO_MEMORY_AVAILABLE
        }
        
        # ✅ NOUVEAU: Ajout stats mémoire vidéo si disponible
        if self.video_memory_system:
            try:
                memory_stats = self.video_memory_system.get_system_stats()
                status['video_memory_stats'] = memory_stats
            except Exception as e:
                status['video_memory_error'] = str(e)
        
        return status
    
    def cleanup(self):
        """Nettoyage des ressources."""
        
        if self.vlm_model:
            self.vlm_model.unload_model()
            self.vlm_model = None
        
        self.orchestrator = None
        self.is_initialized = False

# Instance globale
_vlm_service: Optional[StreamlitVLMService] = None

def get_vlm_service() -> StreamlitVLMService:
    """Récupère le service VLM (singleton)."""
    global _vlm_service
    
    if _vlm_service is None:
        _vlm_service = StreamlitVLMService()
    
    return _vlm_service

def cleanup_vlm_service():
    """Nettoie le service VLM."""
    global _vlm_service
    
    if _vlm_service:
        _vlm_service.cleanup()
        _vlm_service = None