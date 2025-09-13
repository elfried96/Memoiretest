# 🔒 Dashboard de Surveillance Intelligente

Dashboard Streamlit moderne et modulaire pour système de surveillance avec IA.

## ✨ Fonctionnalités

### 🎥 **Multi-Caméras en Temps Réel**
- Grilles configurables (2x2, 3x3, 4x4)
- Flux RTSP, webcam, fichiers vidéo
- Zones de détection personnalisables
- Contrôles individuels par caméra

### 📱 **Analyse Vidéo Avancée**
- Upload de vidéos (MP4, AVI, MOV, MKV, WEBM)
- Détection comportementale avec VLM
- Analyse multi-outils (YOLO, SAM2, Pose)
- Export des résultats (JSON, CSV)

### 💬 **Chat IA Interactif**
- Conversation avec le VLM
- Questions prédéfinies par catégorie
- Contexte automatique des analyses
- Historique persistant

### 🚨 **Alertes Audio/Visuelles**
- Sons générés dynamiquement
- 4 niveaux d'alerte (LOW, MEDIUM, HIGH, CRITICAL)
- Cooldown intelligent
- Notifications en temps réel

### 📊 **Métriques & Rapports**
- Statistiques temps réel
- Export des données de session
- Rapports détaillés
- Monitoring des performances

## 🚀 Installation Rapide

```bash
# 1. Aller dans le répertoire dashboard
cd intelligent-surveillance-system/dashboard/

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Lancer l'application
streamlit run app.py
```

## 📁 Architecture Modulaire

```
dashboard/
├── app.py                     # 🚀 Point d'entrée principal
├── config/
│   └── settings.py           # ⚙️ Configuration centralisée
├── components/
│   ├── camera_grid.py        # 📹 Grille multi-caméras
│   └── vlm_chat.py          # 💬 Interface chat VLM
├── services/
│   ├── session_manager.py    # 👤 Gestion sessions
│   └── vlm_integration.py    # 🤖 Intégration VLM
├── utils/
│   └── audio_alerts.py       # 🔊 Système alertes audio
└── assets/
    └── alert_sounds/         # 🎵 Fichiers audio
```

## ⚙️ Configuration

### Variables d'Environnement

```bash
# Nombre maximum de caméras
export MAX_CAMERAS=9

# Seuil d'alerte par défaut
export ALERT_THRESHOLD=70

# Audio activé
export AUDIO_ENABLED=true
export AUDIO_VOLUME=0.8

# Authentification (optionnel)
export ENABLE_AUTH=false
export SESSION_TIMEOUT=3600
```

### Configuration Caméras

```python
# Ajout caméra via interface
camera_config = CameraConfig(
    camera_id=\"cam_01\",
    name=\"Caméra Entrée\",
    source=\"rtsp://ip:port/stream\",  # ou \"0\" pour webcam
    resolution=(1280, 720),
    fps=30,
    sensitivity=0.7
)
```

## 🔧 Utilisation

### 1. Surveillance Live
- Configurez vos caméras
- Définissez les zones de détection  
- Ajustez la sensibilité
- Surveillez en temps réel

### 2. Analyse Vidéo
- Uploadez une vidéo
- Sélectionnez les options d'analyse
- Lancez l'analyse VLM
- Consultez les résultats

### 3. Chat avec l'IA
- Posez des questions sur les analyses
- Utilisez les questions prédéfinies
- Consultez l'historique
- Exportez les conversations

### 4. Gestion des Alertes
- Configurez les seuils
- Activez/désactivez les sons
- Consultez l'historique
- Exportez les rapports

## 🧩 Intégration VLM

Le dashboard s'intègre automatiquement avec votre système VLM existant :

```python
# Détection automatique du VLM
from src.core.vlm.model import VisionLanguageModel
from src.core.orchestrator.adaptive_orchestrator import AdaptiveOrchestrator

# Mode dégradé si VLM indisponible
if VLM_AVAILABLE:
    # Utilise le vrai VLM
    response = await vlm_model.analyze_with_tools(request)
else:
    # Mode simulation intelligent
    response = simulate_analysis(request)
```

## 📊 API des Composants

### Session Manager
```python
from services.session_manager import get_session_manager

session = get_session_manager()
session.add_chat_message(\"user\", \"Question\")
session.add_alert(\"HIGH\", \"Comportement suspect\")
session.set_camera_state(\"cam_01\", {\"enabled\": True})
```

### Système Audio
```python
from utils.audio_alerts import play_alert, play_detection_alert

# Alerte simple
play_alert(\"HIGH\", \"Intrusion détectée\")

# Alerte détection
play_detection_alert(0.95, \"personne\")
```

### Chat VLM
```python
from components.vlm_chat import get_vlm_chat

chat = get_vlm_chat()
chat.set_vlm_callback(my_vlm_function)
chat.add_system_message(\"Système initialisé\", \"success\")
```

## 🎨 Personnalisation

### Thème CSS
Modifiez `app.py` pour personnaliser l'apparence :

```python
st.markdown(\"\"\"\n<style>\n.main-header {\n    background: linear-gradient(90deg, #your-colors);\n    # Vos styles...\n}\n</style>\n\"\"\", unsafe_allow_html=True)
```

### Questions Prédéfinies
Ajoutez vos propres catégories dans `vlm_chat.py` :

```python
self.predefined_questions = {
    \"Ma Catégorie\": [
        \"Ma question personnalisée\",
        \"Autre question\"
    ]
}
```

### Alertes Audio
Ajoutez vos sons dans `assets/alert_sounds/` et configurez :

```python
audio_config.alert_sounds = {
    \"CUSTOM\": \"mon_son.wav\"
}
```

## 🔒 Sécurité

- Sessions isolées par utilisateur
- Pas de stockage de données sensibles
- Validation des inputs
- Nettoyage automatique des caches
- Logs d'activité optionnels

## 🚀 Performance

- Cache intelligent des métriques (TTL: 5min)
- Streaming vidéo optimisé
- Traitement asynchrone VLM
- Nettoyage automatique mémoire
- Compression des frames

## 🧪 Mode Développement

```bash
# Activer le mode debug
export STREAMLIT_DEBUG=true

# Logs détaillés  
export LOG_LEVEL=DEBUG

# Rechargement automatique
streamlit run app.py --server.runOnSave=true
```

## 📱 Accès

- **Local**: http://localhost:8501
- **Réseau**: http://votre-ip:8501
- **Production**: Configurez proxy inverse (nginx/apache)

## 🆘 Dépannage

### VLM non trouvé
```
⚠️ Modules VLM non trouvés - Mode démo activé
```
→ Vérifiez que le système principal est installé

### Erreur caméra
```
❌ Impossible d'ouvrir la caméra
```  
→ Vérifiez l'URL RTSP ou l'index de périphérique

### Audio ne fonctionne pas
→ Vérifiez les permissions navigateur pour l'audio

## 📄 Licence

Partie du système de surveillance intelligent - Voir LICENSE principal.

---

**🎯 Dashboard prêt pour la production avec architecture modulaire optimisée !**