# 📚 Guide d'Utilisation Complet - Système de Surveillance Intelligente

## 🎯 Aperçu du Code Principal

### **main.py** - Architecture Complète

Le fichier `main.py` est le **cœur du système** qui orchestre tous les composants :

```python
# Structure principale du main.py :

1. 📦 IMPORTS
   - asyncio, cv2, argparse (système)
   - DynamicVisionLanguageModel (VLM multi-modèles)
   - ModernVLMOrchestrator (coordination 8 outils)
   - YOLODetector, BYTETracker (détection/tracking)

2. 🏗️ CLASSES PRINCIPALES
   - AlertLevel : Enum des niveaux d'alerte
   - SurveillanceFrame : Structure d'un frame enrichi
   - SurveillanceDecisionEngine : Moteur de décision
   - IntelligentSurveillanceSystem : Système complet

3. 🔧 FONCTIONS UTILITAIRES
   - parse_arguments() : Arguments ligne de commande
   - validate_video_source() : Validation sources vidéo
   - main() : Point d'entrée principal
```

## 🚀 Utilisation avec Arguments

### **Webcam (Défaut)**
```bash
python main.py
# ou explicitement
python main.py --video webcam
```

### **Fichier Vidéo Local**
```bash
# Fichiers supportés : .mp4, .avi, .mov, .mkv, .webm
python main.py --video /chemin/vers/video.mp4
python main.py -v ./surveillance_footage.avi
python main.py --video "C:/Videos/ma video avec espaces.mp4"
```

### **Flux Réseau (RTSP/HTTP)**
```bash
# Caméra IP RTSP
python main.py --video rtsp://192.168.1.100:554/stream
python main.py --video rtsp://admin:password@camera_ip/live

# Flux HTTP
python main.py --video http://example.com/livestream.mjpg
```

### **Configuration Avancée**
```bash
# Mode rapide avec LLaVA
python main.py --video video.mp4 --model llava-v1.6-mistral-7b --mode FAST

# Mode complet avec Kimi-VL
python main.py --video webcam --model kimi-vl-a3b-thinking --mode THOROUGH

# Mode headless (sans affichage)
python main.py --video video.mp4 --no-display --max-frames 1000

# Qwen pour analyse complexe
python main.py --video surveillance.mp4 --model qwen2-vl-7b-instruct --mode BALANCED
```

## 🧩 Explication Détaillée des Classes

### **1. AlertLevel (Enum)**
```python
class AlertLevel(Enum):
    NORMAL = "normal"      # Situation normale
    ATTENTION = "attention" # Surveillance renforcée
    ALERTE = "alerte"      # Intervention requise
    CRITIQUE = "critique"   # Action immédiate
```

### **2. SurveillanceFrame (@dataclass)**
```python
@dataclass
class SurveillanceFrame:
    frame_id: int                              # ID unique du frame
    timestamp: float                           # Timestamp Unix
    frame: np.ndarray                          # Image OpenCV (BGR)
    detections: List[Detection]                # Détections YOLO
    tracked_objects: List[Dict]                # Objets suivis avec IDs
    vlm_analysis: Optional[AnalysisResponse]   # Analyse VLM (si effectuée)
    alert_level: AlertLevel                    # Niveau d'alerte calculé
    actions_taken: List[str]                   # Actions automatisées prises
```

### **3. SurveillanceDecisionEngine**

**Rôle** : Convertit les analyses VLM en décisions concrètes et actions automatisées.

```python
def process_analysis(self, analysis: AnalysisResponse, frame_context: Dict) -> Dict:
    """
    LOGIQUE DE DÉCISION :
    
    1. Niveau de suspicion élevé → ALERTE + enregistrement
    2. Confiance faible (< 0.6) → Révision humaine requise
    3. Actions spécifiques par type :
       - SUSPICIOUS_ACTIVITY → Surveillance renforcée + alerte sécurité
       - THEFT_ATTEMPT → Alerte immédiate + contact sécurité + tracking suspect
       - LOITERING → Observation prolongée + intervention staff
    4. Contexte temporel → Hors heures = plus strict
    5. Outils utilisés → Actions spécialisées
    """
```

**Exemples de décisions** :
```python
# Suspicion élevée détectée
{
    "alert_level": AlertLevel.ALERTE,
    "actions": ["start_recording", "alert_security", "track_individual"],
    "notifications": ["Suspicion HIGH détectée"],
    "recording_required": True,
    "human_review": False
}

# Confiance faible
{
    "alert_level": AlertLevel.NORMAL,
    "actions": ["request_human_review"],
    "human_review": True
}
```

### **4. IntelligentSurveillanceSystem** - Classe Principale

#### **__init__()** - Initialisation
```python
def __init__(self, video_source, vlm_model, orchestration_mode):
    # 1. Configuration source vidéo
    self.video_source = video_source
    
    # 2. Composants de détection
    self.yolo_detector = YOLODetector(model_path="yolov11n.pt")
    self.tracker = BYTETracker()
    
    # 3. VLM multi-modèles
    self.vlm = DynamicVisionLanguageModel(
        default_model=vlm_model,
        enable_fallback=True  # Auto-switch vers LLaVA si Kimi-VL indispo
    )
    
    # 4. Orchestrateur 8 outils
    self.orchestrator = ModernVLMOrchestrator(config=OrchestrationConfig())
    
    # 5. Moteur de décision
    self.decision_engine = SurveillanceDecisionEngine()
    
    # 6. Statistiques de performance
    self.processing_stats = {
        "total_frames": 0,
        "detected_objects": 0,
        "vlm_analyses": 0,
        "alerts_triggered": 0,
        "average_fps": 0.0
    }
```

#### **process_frame()** - Cœur du Pipeline
```python
async def process_frame(self, frame: np.ndarray) -> SurveillanceFrame:
    """
    PIPELINE COMPLET D'UN FRAME :
    
    1. 🔍 DÉTECTION YOLO
       - yolo_results = self.yolo_detector.detect(frame)
       - Détecte personnes, objets avec bbox + confiance
    
    2. 🎯 TRACKING MULTI-OBJETS
       - tracked_objects = self.tracker.update(detections)
       - Assigne IDs persistants, suit mouvements
    
    3. 🧠 ANALYSE VLM (CONDITIONNELLE)
       Déclenche si :
       - Personnes détectées OU
       - Tous les 30 frames (analyse périodique)
       
       Pipeline VLM :
       - frame_b64 = encode_frame_to_base64(frame)
       - context = {frame_id, timestamp, location, person_count, ...}
       - vlm_analysis = orchestrator.analyze_surveillance_frame()
    
    4. 🚨 PRISE DE DÉCISIONS
       - decisions = decision_engine.process_analysis(vlm_analysis)
       - alert_level, actions_taken = decisions
    
    5. 📊 RETOUR SURVEILLANCE_FRAME
       - Agrège toutes les données du frame
    """
```

#### **run_surveillance()** - Boucle Principale
```python
async def run_surveillance(self, max_frames=None, display=True):
    """
    BOUCLE INFINIE DE SURVEILLANCE :
    
    1. cap = cv2.VideoCapture(self.video_source)
    2. while True:
         - ret, frame = cap.read()
         - surveillance_frame = await self.process_frame(frame)
         - if display: cv2.imshow() avec overlays
         - ESC pour quitter
         - Logging périodique (60 frames)
    3. Nettoyage : cap.release(), cv2.destroyAllWindows()
    """
```

#### **draw_surveillance_overlay()** - Interface Visuelle
```python
def draw_surveillance_overlay(self, frame, surveillance_frame):
    """
    OVERLAYS AFFICHÉS :
    
    🔍 DÉTECTIONS :
    - Rectangles colorés : vert=personne, rouge=objet
    - Labels avec confiance : "person: 0.85"
    
    📊 INFORMATIONS SYSTÈME :
    - Frame ID actuel
    - Niveau d'alerte (couleur selon gravité)
    - Résultats VLM : suspicion + confiance
    - Action type détecté
    
    📈 STATISTIQUES TEMPS RÉEL :
    - FPS moyen
    - Nombre détections frame actuel
    - Total alertes déclenchées
    """
```

## 🔧 Fonctions Utilitaires

### **parse_arguments()** - Arguments CLI
```python
def parse_arguments():
    """
    ARGUMENTS SUPPORTÉS :
    
    --video/-v     : Source vidéo (webcam/fichier/RTSP)
    --model/-m     : Modèle VLM (kimi-vl-a3b-thinking par défaut)
    --mode         : Mode orchestration (FAST/BALANCED/THOROUGH)
    --max-frames   : Limite nombre frames (None = infini)
    --no-display   : Mode headless sans affichage
    
    EXEMPLES AUTO-GÉNÉRÉS dans --help
    """
```

### **validate_video_source()** - Validation Sources
```python
def validate_video_source(video_arg: str):
    """
    LOGIQUE DE VALIDATION :
    
    1. "webcam" → return 0 (device webcam)
    2. Fichier local existant → return str(path)
       - Extensions recommandées : .mp4, .avi, .mov, .mkv, .webm
       - Warning si extension non standard
    3. URL réseau (rtsp://, http://) → return url
    4. Fichier inexistant → Warning + tentative ouverture
    """
```

## 📊 Workflow Détaillé Frame par Frame

### **Frame N** - Exemple de Traitement
```python
# ENTRÉE : frame OpenCV (640x480, BGR)
frame = np.ndarray(shape=(480, 640, 3))

# 1. DÉTECTION YOLO (10-20ms)
yolo_results = yolo_detector.detect(frame)
detections = [
    Detection(bbox=BoundingBox(100, 50, 200, 300), confidence=0.85, class_name="person"),
    Detection(bbox=BoundingBox(300, 150, 400, 250), confidence=0.72, class_name="backpack")
]

# 2. TRACKING (5ms)
tracked_objects = tracker.update(detections)
# → Assigne track_1, track_2, met à jour trajectoires

# 3. ANALYSE VLM CONDITIONNELLE (2000-5000ms selon mode)
if len(persons) > 0 or frame_count % 30 == 0:
    frame_b64 = encode_frame_to_base64(frame)
    context = {
        "frame_id": 1234,
        "person_count": 1,
        "location": "Store Main Area",
        "time_of_day": "14:30:15"
    }
    
    # Orchestration selon mode :
    # FAST: 3 outils (dino, pose, fusion) → ~1.2s
    # BALANCED: 6 outils + sam2, trajectory, adversarial → ~2.5s  
    # THOROUGH: 8 outils complets → ~4.8s
    
    vlm_analysis = AnalysisResponse(
        suspicion_level=SuspicionLevel.MEDIUM,
        action_type=ActionType.SUSPICIOUS_ACTIVITY,
        confidence=0.78,
        description="Personne near high-value items, extended observation",
        tools_used=["sam2_segmentator", "pose_estimator", "trajectory_analyzer"],
        recommendations=["Surveillance renforcée", "Intervention discrète"]
    )

# 4. DÉCISIONS AUTOMATISÉES (1ms)
decisions = decision_engine.process_analysis(vlm_analysis, context)
{
    "alert_level": AlertLevel.ATTENTION,
    "actions": ["increase_monitoring", "alert_security", "track_individual"],
    "recording_required": False,
    "human_review": False
}

# 5. FRAME DE SURVEILLANCE ENRICHI
surveillance_frame = SurveillanceFrame(
    frame_id=1234,
    timestamp=1704886800.123,
    frame=frame,
    detections=detections,
    tracked_objects=tracked_objects,
    vlm_analysis=vlm_analysis,
    alert_level=AlertLevel.ATTENTION,
    actions_taken=["increase_monitoring", "alert_security"]
)

# 6. AFFICHAGE AVEC OVERLAYS
overlay_frame = draw_surveillance_overlay(frame, surveillance_frame)
cv2.imshow("Surveillance", overlay_frame)
```

## 🎛️ Modes d'Orchestration Détaillés

### **FAST Mode** (⚡ Performance)
```python
tools_used = ["dino_features", "pose_estimator", "multimodal_fusion"]

# Utilisation :
# - Edge computing, embedded systems
# - Surveillance temps réel haute fréquence
# - Ressources limitées
# - Temps de réponse critique (< 1.5s)

# Trade-offs :
# + Très rapide, faible consommation
# - Analyse moins approfondie
# - Peut manquer des nuances comportementales
```

### **BALANCED Mode** (⚖️ Production)
```python
tools_used = [
    "sam2_segmentator",      # Segmentation précise
    "dino_features",         # Features robustes  
    "pose_estimator",        # Analyse posturale
    "trajectory_analyzer",   # Patterns mouvement
    "multimodal_fusion",     # Fusion intelligente
    "adversarial_detector"   # Protection attaques
]

# Utilisation :
# - Production standard (RECOMMANDÉ)
# - Équilibre performance/précision
# - Surveillance commerciale/industrielle
# - Bon compromis toutes situations

# Trade-offs :
# + Très bon équilibre
# + Détection robuste
# - Temps traitement modéré (2-3s)
```

### **THOROUGH Mode** (🔬 Analyse)
```python
tools_used = [
    "sam2_segmentator", "dino_features", "pose_estimator",
    "trajectory_analyzer", "multimodal_fusion", 
    "temporal_transformer",  # Analyse séquentielle
    "adversarial_detector",  # Sécurité avancée
    "domain_adapter"         # Multi-environnements
]

# Utilisation :
# - Analyse forensique post-incident
# - Recherche et développement
# - Environnements haute sécurité
# - Quand précision > vitesse

# Trade-offs :
# + Analyse la plus complète
# + Détection sophistiquée
# - Plus lent (4-5s/frame)
# - Consommation ressources élevée
```

## 🎯 Exemples d'Utilisation Réels

### **Surveillance Magasin**
```bash
# Webcam temps réel, mode équilibré
python main.py --video webcam --mode BALANCED --model kimi-vl-a3b-thinking

# Analyse vidéo sécurité existante
python main.py --video surveillance_20240115.mp4 --mode THOROUGH --model qwen2-vl-7b-instruct
```

### **Surveillance Entrepôt**
```bash
# Caméra IP fixe, mode rapide pour monitoring continu
python main.py --video rtsp://192.168.1.100:554/stream --mode FAST --model llava-v1.6-mistral-7b

# Mode headless pour serveur
python main.py --video rtsp://warehouse_cam/live --mode BALANCED --no-display
```

### **Analyse Forensique**
```bash
# Analyse approfondie incident
python main.py --video incident_footage.mp4 --mode THOROUGH --model kimi-vl-a3b-thinking --max-frames 500

# Export sans affichage
python main.py --video evidence.avi --mode THOROUGH --no-display --model qwen2-vl-7b-instruct
```

## 📈 Performance et Optimisation

### **Temps de Traitement Typiques** (par frame)
```
Mode FAST    : 1.0-1.5s  (3 outils)
Mode BALANCED: 2.0-3.0s  (6 outils) 
Mode THOROUGH: 4.0-6.0s  (8 outils)

Facteurs d'influence :
- Résolution vidéo (plus haute = plus lent)
- Nombre de détections (plus d'objets = plus lent)
- Modèle VLM choisi (Kimi-VL vs LLaVA vs Qwen)
- Hardware GPU (CUDA vs CPU)
```

### **Ressources Système**
```
RAM Minimum   : 8 GB (16 GB recommandé)
GPU           : 6 GB VRAM (RTX 3060+) ou CPU puissant
Stockage      : 5 GB pour modèles + espace vidéos
Réseau        : Stable pour flux RTSP/modèles distants
```

## 🛟 Dépannage et Logs

### **Logs Utiles**
```python
# Niveaux de log configurables
logging.basicConfig(level=logging.DEBUG)  # Plus de détails

# Types de messages :
logger.info("🎯 Système initialisé")        # Informations générales
logger.warning("⚠️ VLM échec, fallback")    # Avertissements
logger.error("❌ Erreur critique")           # Erreurs
logger.debug("🔍 Frame 1234 - Analyse VLM") # Debug détaillé
```

### **Erreurs Communes**
```bash
# Source vidéo indisponible
❌ Impossible d'ouvrir la source vidéo: /path/video.mp4
→ Vérifier chemin, permissions, format supporté

# Modèle VLM indisponible  
⚠️ VLM principal échec, fallback activé
→ Normal, utilisation automatique de LLaVA

# CUDA out of memory
❌ CUDA out of memory
→ Réduire résolution, utiliser mode FAST, ou --no-display
```

Le système est maintenant **complètement configuré** pour traiter vos vidéos avec le chemin de fichier spécifié ! 🚀