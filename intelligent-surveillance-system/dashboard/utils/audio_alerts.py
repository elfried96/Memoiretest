"""Système d'alertes audio avancé pour Streamlit."""

import streamlit as st
import base64
import json
from pathlib import Path
from typing import Dict, Optional, Callable
import threading
import time
from datetime import datetime, timedelta

from config.settings import get_audio_config

class AudioAlertSystem:
    """Système d'alertes audio optimisé."""
    
    def __init__(self):
        self.config = get_audio_config()
        self.last_alert = {}  # Cooldown par type d'alerte
        self.alert_queue = []
        self._setup_default_sounds()
    
    def _setup_default_sounds(self):
        """Crée les sons par défaut si inexistants."""
        sounds_dir = self.config.sounds_dir
        sounds_dir.mkdir(parents=True, exist_ok=True)
        
        # Génération de sons basiques en base64
        self.default_sounds = {
            "LOW": self._generate_beep_sound(frequency=440, duration=0.2),
            "MEDIUM": self._generate_beep_sound(frequency=660, duration=0.5),
            "HIGH": self._generate_beep_sound(frequency=880, duration=1.0),
            "CRITICAL": self._generate_alarm_sound()
        }
    
    def _generate_beep_sound(self, frequency: int = 440, duration: float = 0.5) -> str:
        """Génère un son basique en base64."""
        import numpy as np
        import wave
        from io import BytesIO
        
        sample_rate = 44100
        frames = int(duration * sample_rate)
        
        # Génération onde sinusoïdale
        t = np.linspace(0, duration, frames)
        wave_data = np.sin(frequency * 2 * np.pi * t)
        
        # Application enveloppe pour éviter les clics
        fade_frames = int(0.01 * sample_rate)  # 10ms fade
        wave_data[:fade_frames] *= np.linspace(0, 1, fade_frames)
        wave_data[-fade_frames:] *= np.linspace(1, 0, fade_frames)
        
        # Conversion en 16-bit
        wave_data = (wave_data * 32767).astype(np.int16)
        
        # Création du fichier WAV en mémoire
        buffer = BytesIO()
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(wave_data.tobytes())
        
        # Encodage base64
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode()
    
    def _generate_alarm_sound(self) -> str:
        """Génère un son d'alarme complexe."""
        import numpy as np
        import wave
        from io import BytesIO
        
        sample_rate = 44100
        duration = 1.5
        frames = int(duration * sample_rate)
        
        t = np.linspace(0, duration, frames)
        
        # Alarme modulée (sirène)
        base_freq = 800
        mod_freq = 5
        wave_data = np.sin(2 * np.pi * base_freq * t) * np.sin(2 * np.pi * mod_freq * t)
        wave_data += 0.3 * np.sin(2 * np.pi * base_freq * 1.5 * t)
        
        # Normalisation et conversion
        wave_data = wave_data / np.max(np.abs(wave_data))
        wave_data = (wave_data * 32767 * 0.8).astype(np.int16)
        
        # Création WAV
        buffer = BytesIO()
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(wave_data.tobytes())
        
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode()
    
    def play_alert(self, level: str, message: str = "", force: bool = False) -> bool:
        """
        Joue une alerte audio.
        
        Args:
            level: Niveau d'alerte (LOW, MEDIUM, HIGH, CRITICAL)
            message: Message d'alerte optionnel
            force: Force la lecture même en cooldown
            
        Returns:
            True si l'alerte a été jouée, False sinon
        """
        
        if not self.config.enabled:
            return False
        
        # Vérification cooldown
        if not force and self._is_in_cooldown(level):
            return False
        
        # Récupération du son
        sound_b64 = self._get_sound_data(level)
        if not sound_b64:
            return False
        
        # Lecture audio via HTML
        self._play_sound_html(sound_b64, level, message)
        
        # Mise à jour cooldown
        self.last_alert[level] = datetime.now()
        
        return True
    
    def _is_in_cooldown(self, level: str) -> bool:
        """Vérifie si l'alerte est en cooldown."""
        if level not in self.last_alert:
            return False
        
        cooldown = 5  # 5 secondes par défaut
        if level == "CRITICAL":
            cooldown = 2
        elif level == "HIGH":
            cooldown = 3
        
        return datetime.now() - self.last_alert[level] < timedelta(seconds=cooldown)
    
    def _get_sound_data(self, level: str) -> Optional[str]:
        """Récupère les données audio en base64."""
        
        # Fichier personnalisé
        sound_file = self.config.sounds_dir / self.config.alert_sounds.get(level, "")
        if sound_file.exists():
            try:
                with open(sound_file, "rb") as f:
                    return base64.b64encode(f.read()).decode()
            except Exception as e:
                st.error(f"Erreur lecture son {sound_file}: {e}")
        
        # Son par défaut
        return self.default_sounds.get(level)
    
    def _play_sound_html(self, sound_b64: str, level: str, message: str):
        """Joue le son via HTML audio."""
        
        # Volume selon le niveau
        volume = self.config.volume
        if level == "CRITICAL":
            volume = min(1.0, volume * 1.2)
        elif level == "LOW":
            volume = volume * 0.7
        
        # HTML pour lecture audio
        audio_html = f"""
        <audio id="alert_audio_{level.lower()}" preload="auto" style="display: none;">
            <source src="data:audio/wav;base64,{sound_b64}" type="audio/wav">
        </audio>
        
        <script>
            (function() {{
                var audio = document.getElementById('alert_audio_{level.lower()}');
                if (audio) {{
                    audio.volume = {volume};
                    audio.play().catch(function(e) {{
                        console.log('Erreur lecture audio:', e);
                    }});
                }}
            }})();
        </script>
        """
        
        # Affichage notification visuelle
        if message:
            if level == "CRITICAL":
                st.error(f" CRITIQUE: {message}")
            elif level == "HIGH":
                st.error(f" ALERTE: {message}")
            elif level == "MEDIUM":
                st.warning(f"🔶 ATTENTION: {message}")
            else:
                st.info(f"ℹ️ {message}")
        
        # Injection HTML
        st.markdown(audio_html, unsafe_allow_html=True)
    
    def play_custom_sound(self, sound_file: str, volume: float = None) -> bool:
        """Joue un son personnalisé."""
        try:
            sound_path = self.config.sounds_dir / sound_file
            if not sound_path.exists():
                return False
            
            with open(sound_path, "rb") as f:
                sound_b64 = base64.b64encode(f.read()).decode()
            
            vol = volume or self.config.volume
            
            audio_html = f"""
            <audio preload="auto" style="display: none;" autoplay>
                <source src="data:audio/wav;base64,{sound_b64}" type="audio/wav">
            </audio>
            
            <script>
                document.querySelector('audio').volume = {vol};
            </script>
            """
            
            st.markdown(audio_html, unsafe_allow_html=True)
            return True
            
        except Exception as e:
            st.error(f"Erreur lecture son personnalisé: {e}")
            return False
    
    def create_sound_test_interface(self):
        """Interface de test des sons."""
        st.subheader("🔊 Test des alertes audio")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Niveaux d'alerte:**")
            if st.button(" LOW", use_container_width=True):
                self.play_alert("LOW", "Test alerte faible", force=True)
            
            if st.button(" MEDIUM", use_container_width=True):
                self.play_alert("MEDIUM", "Test alerte modérée", force=True)
        
        with col2:
            if st.button(" HIGH", use_container_width=True):
                self.play_alert("HIGH", "Test alerte élevée", force=True)
            
            if st.button(" CRITICAL", use_container_width=True):
                self.play_alert("CRITICAL", "Test alerte critique", force=True)
        
        with col3:
            st.write("**Paramètres:**")
            
            # Volume
            new_volume = st.slider(
                "Volume",
                0.0, 1.0,
                self.config.volume,
                step=0.1
            )
            if new_volume != self.config.volume:
                self.config.volume = new_volume
            
            # Activation/Désactivation
            new_enabled = st.checkbox(
                "Sons activés",
                self.config.enabled
            )
            if new_enabled != self.config.enabled:
                self.config.enabled = new_enabled
    
    def get_status(self) -> Dict:
        """Retourne le statut du système audio."""
        return {
            "enabled": self.config.enabled,
            "volume": self.config.volume,
            "sounds_available": list(self.config.alert_sounds.keys()),
            "cooldown_status": {
                level: (datetime.now() - last_time).total_seconds()
                for level, last_time in self.last_alert.items()
            }
        }

# Instance globale
audio_system = AudioAlertSystem()

def get_audio_system() -> AudioAlertSystem:
    """Récupère le système d'alertes audio."""
    return audio_system

# Fonctions utilitaires
def play_alert(level: str, message: str = "", force: bool = False) -> bool:
    """Raccourci pour jouer une alerte."""
    return audio_system.play_alert(level, message, force)

def play_detection_alert(confidence: float, object_type: str = ""):
    """Alerte spécifique pour détection."""
    if confidence > 0.9:
        play_alert("HIGH", f"Détection confirmée: {object_type} ({confidence:.1%})")
    elif confidence > 0.7:
        play_alert("MEDIUM", f"Détection probable: {object_type} ({confidence:.1%})")
    else:
        play_alert("LOW", f"Détection possible: {object_type} ({confidence:.1%})")

def play_behavior_alert(suspicion_level: str, behavior: str = ""):
    """Alerte spécifique pour comportement suspect."""
    level_map = {
        "LOW": "LOW",
        "MEDIUM": "MEDIUM", 
        "HIGH": "HIGH",
        "CRITICAL": "CRITICAL"
    }
    
    mapped_level = level_map.get(suspicion_level.upper(), "LOW")
    message = f"Comportement détecté: {behavior}" if behavior else "Comportement suspect détecté"
    
    play_alert(mapped_level, message)