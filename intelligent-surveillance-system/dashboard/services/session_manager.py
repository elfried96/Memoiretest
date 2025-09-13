"""Gestionnaire de sessions utilisateur optimisé."""

import streamlit as st
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import json
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path

from config.settings import get_security_config

@dataclass
class UserSession:
    """Représente une session utilisateur."""
    
    session_id: str
    user_id: str
    created_at: datetime
    last_activity: datetime
    data: Dict[str, Any]
    is_active: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Conversion en dictionnaire."""
        return {
            **asdict(self),
            'created_at': self.created_at.isoformat(),
            'last_activity': self.last_activity.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserSession':
        """Création depuis dictionnaire."""
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['last_activity'] = datetime.fromisoformat(data['last_activity'])
        return cls(**data)

class SessionManager:
    """Gestionnaire de sessions Streamlit avancé."""
    
    def __init__(self):
        self.config = get_security_config()
        self._init_session()
    
    def _init_session(self):
        """Initialisation de la session Streamlit."""
        
        # ID de session unique
        if 'session_id' not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())
        
        # Données utilisateur
        if 'user_data' not in st.session_state:
            st.session_state.user_data = {}
        
        # Historique chat VLM
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Métriques en temps réel
        if 'metrics_cache' not in st.session_state:
            st.session_state.metrics_cache = {}
        
        # État des caméras
        if 'cameras_state' not in st.session_state:
            st.session_state.cameras_state = {}
        
        # Alertes actives
        if 'active_alerts' not in st.session_state:
            st.session_state.active_alerts = []
        
        # Analyses vidéo
        if 'video_analyses' not in st.session_state:
            st.session_state.video_analyses = {}
        
        # Dernière activité
        st.session_state.last_activity = datetime.now()
    
    def get_session_id(self) -> str:
        """Récupère l'ID de session."""
        return st.session_state.session_id
    
    def set_user_data(self, key: str, value: Any) -> None:
        """Stocke des données utilisateur."""
        st.session_state.user_data[key] = value
        self._update_activity()
    
    def get_user_data(self, key: str, default: Any = None) -> Any:
        """Récupère des données utilisateur."""
        return st.session_state.user_data.get(key, default)
    
    def add_chat_message(self, role: str, content: str, metadata: Optional[Dict] = None) -> None:
        """Ajoute un message au chat."""
        message = {
            'id': str(uuid.uuid4()),
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        st.session_state.chat_history.append(message)
        
        # Limite l'historique
        max_history = 50  # Depuis config
        if len(st.session_state.chat_history) > max_history:
            st.session_state.chat_history = st.session_state.chat_history[-max_history:]
        
        self._update_activity()
    
    def get_chat_history(self) -> List[Dict[str, Any]]:
        """Récupère l'historique du chat."""
        return st.session_state.chat_history
    
    def clear_chat_history(self) -> None:
        """Efface l'historique du chat."""
        st.session_state.chat_history = []
        self._update_activity()
    
    def add_alert(self, level: str, message: str, source: str = "system") -> None:
        """Ajoute une alerte."""
        alert = {
            'id': str(uuid.uuid4()),
            'level': level,
            'message': message,
            'source': source,
            'timestamp': datetime.now().isoformat(),
            'acknowledged': False
        }
        
        st.session_state.active_alerts.append(alert)
        
        # Limite le nombre d'alertes
        max_alerts = 10  # Depuis config
        if len(st.session_state.active_alerts) > max_alerts:
            st.session_state.active_alerts = st.session_state.active_alerts[-max_alerts:]
        
        self._update_activity()
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Récupère les alertes actives."""
        return st.session_state.active_alerts
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acquitte une alerte."""
        for alert in st.session_state.active_alerts:
            if alert['id'] == alert_id:
                alert['acknowledged'] = True
                self._update_activity()
                return True
        return False
    
    def clear_alerts(self) -> None:
        """Efface toutes les alertes."""
        st.session_state.active_alerts = []
        self._update_activity()
    
    def set_camera_state(self, camera_id: str, state: Dict[str, Any]) -> None:
        """Définit l'état d'une caméra."""
        st.session_state.cameras_state[camera_id] = {
            **state,
            'last_update': datetime.now().isoformat()
        }
        self._update_activity()
    
    def get_camera_state(self, camera_id: str) -> Dict[str, Any]:
        """Récupère l'état d'une caméra."""
        return st.session_state.cameras_state.get(camera_id, {})
    
    def get_all_cameras_state(self) -> Dict[str, Dict[str, Any]]:
        """Récupère l'état de toutes les caméras."""
        return st.session_state.cameras_state
    
    def store_video_analysis(self, video_id: str, analysis: Dict[str, Any]) -> None:
        """Stocke une analyse vidéo."""
        st.session_state.video_analyses[video_id] = {
            **analysis,
            'stored_at': datetime.now().isoformat()
        }
        self._update_activity()
    
    def get_video_analysis(self, video_id: str) -> Optional[Dict[str, Any]]:
        """Récupère une analyse vidéo."""
        return st.session_state.video_analyses.get(video_id)
    
    def get_all_video_analyses(self) -> Dict[str, Dict[str, Any]]:
        """Récupère toutes les analyses vidéo."""
        return st.session_state.video_analyses
    
    def cache_metrics(self, key: str, data: Any, ttl_seconds: int = 300) -> None:
        """Cache des métriques avec TTL."""
        st.session_state.metrics_cache[key] = {
            'data': data,
            'cached_at': datetime.now(),
            'ttl': ttl_seconds
        }
    
    def get_cached_metrics(self, key: str) -> Optional[Any]:
        """Récupère des métriques cachées."""
        if key in st.session_state.metrics_cache:
            cached = st.session_state.metrics_cache[key]
            
            # Vérification TTL
            if datetime.now() - cached['cached_at'] < timedelta(seconds=cached['ttl']):
                return cached['data']
            else:
                # Cache expiré
                del st.session_state.metrics_cache[key]
        
        return None
    
    def export_session_data(self, format: str = 'json') -> str:
        """Exporte les données de session."""
        
        export_data = {
            'session_id': self.get_session_id(),
            'exported_at': datetime.now().isoformat(),
            'user_data': st.session_state.user_data,
            'chat_history': st.session_state.chat_history,
            'active_alerts': st.session_state.active_alerts,
            'cameras_state': st.session_state.cameras_state,
            'video_analyses': st.session_state.video_analyses
        }
        
        if format.lower() == 'json':
            return json.dumps(export_data, indent=2, ensure_ascii=False)
        
        # Autres formats à implémenter (CSV, etc.)
        return json.dumps(export_data, indent=2, ensure_ascii=False)
    
    def is_session_expired(self) -> bool:
        """Vérifie si la session a expiré."""
        if not self.config.session_timeout:
            return False
        
        last_activity = st.session_state.get('last_activity', datetime.now())
        return datetime.now() - last_activity > timedelta(seconds=self.config.session_timeout)
    
    def _update_activity(self) -> None:
        """Met à jour la dernière activité."""
        st.session_state.last_activity = datetime.now()
    
    def cleanup_expired_data(self) -> None:
        """Nettoie les données expirées."""
        
        # Cache métriques
        expired_keys = []
        for key, cached in st.session_state.metrics_cache.items():
            if datetime.now() - cached['cached_at'] > timedelta(seconds=cached['ttl']):
                expired_keys.append(key)
        
        for key in expired_keys:
            del st.session_state.metrics_cache[key]
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Statistiques de la session."""
        return {
            'session_id': self.get_session_id(),
            'created_at': st.session_state.get('session_created', datetime.now()).isoformat(),
            'last_activity': st.session_state.last_activity.isoformat(),
            'chat_messages': len(st.session_state.chat_history),
            'active_alerts': len(st.session_state.active_alerts),
            'cameras_configured': len(st.session_state.cameras_state),
            'video_analyses': len(st.session_state.video_analyses),
            'cached_metrics': len(st.session_state.metrics_cache)
        }

# Instance globale
session_manager = SessionManager()

def get_session_manager() -> SessionManager:
    """Récupère le gestionnaire de session."""
    return session_manager