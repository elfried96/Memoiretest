# 📖 Guide d'Apprentissage de l'Architecture

## 🎯 **MODULE 1: Point d'Entrée et Configuration**

### **Fichiers Clés**
- `main_headless_refactored.py` - Point d'entrée modernisé
- `config/base_config.py` - Configuration centralisée
- `config/config_manager.py` - Gestionnaire de configuration

### **Ce que vous devez comprendre**
1. **Arguments CLI** : Comment les paramètres sont parsés et appliqués
2. **Configuration hiérarchique** : VideoConfig, VLMConfig, DetectionConfig, etc.
3. **Variables d'environnement** : Override de configuration
4. **Validation** : Comment la configuration est validée

### **Exercices pratiques - MODULE 1**

#### **🔬 Exercice 1.1: Test des configurations CLI**
```bash
# Étape 1: Testez les modèles VLM différents
python main_headless_refactored.py --model kimi-vl-a3b-thinking --video 0 --mode FAST
python main_headless_refactored.py --model qwen2-vl-7b-instruct --video 0 --mode THOROUGH

# Étape 2: Observez les différences
# - Temps de traitement par frame
# - Qualité de l'analyse
# - Consommation GPU/CPU
```

#### **🔬 Exercice 1.2: Variables d'environnement**
```bash
# Étape 1: Test avec variables d'environnement
SURVEILLANCE_VLM_MODE=smart python main_headless_refactored.py
SURVEILLANCE_DEBUG=true python main_headless_refactored.py --max-frames 5

# Étape 2: Créez votre fichier .env personnalisé
echo "SURVEILLANCE_VLM_MODE=balanced" > .env
echo "SURVEILLANCE_MAX_FRAMES=20" >> .env
echo "SURVEILLANCE_SAVE_RESULTS=true" >> .env

# Étape 3: Testez la priorité des configurations
python main_headless_refactored.py --mode THOROUGH  # CLI override env
```

#### **🔬 Exercice 1.3: Analyse de la configuration**
```bash
# Étape 1: Inspectez les configs chargées
python -c "
from config.config_loader import load_config
config = load_config()
print('VideoConfig:', config.video)
print('VLMConfig:', config.vlm)
print('DetectionConfig:', config.detection)
"

# Étape 2: Validez une configuration invalide
python main_headless_refactored.py --model inexistant_model --video /path/invalid
# Observez les messages d'erreur et la validation
```

#### **🎯 Objectifs d'Apprentissage - Module 1:**
- [ ] Comprendre la hiérarchie de configuration (CLI > ENV > defaults)
- [ ] Identifier tous les paramètres configurables
- [ ] Maîtriser la validation et gestion d'erreurs
- [ ] Créer ses propres presets de configuration

---

## 🎯 **MODULE 2: Système Headless (Cœur)**

### **Fichiers Clés**
- `src/core/headless/surveillance_system.py` - Orchestrateur principal
- `src/core/headless/video_processor.py` - Traitement vidéo
- `src/core/headless/frame_analyzer.py` - Analyse de frames
- `src/core/headless/result_models.py` - Modèles de données

### **Ce que vous devez comprendre**

#### **HeadlessSurveillanceSystem**
```python
# Flow principal
async def run_surveillance(self):
    # 1. Traitement des frames (générateur)
    async for result in self._process_frames():
        # 2. Mise à jour des métriques
        self._update_metrics(result)
        # 3. Sauvegarde conditionnelle
        if self.save_results:
            self.results.append(result)
```

#### **VideoProcessor** 
```python
# Capture vidéo optimisée
def frames_generator(self):
    # 1. Gestion webcam vs fichier
    # 2. Frame skipping
    # 3. Limite max_frames
    # 4. Gestion d'erreurs
```

#### **FrameAnalyzer**
```python
async def analyze_frame(self, frame, frame_id, timestamp):
    # 1. Détection YOLO
    detections = await self._detect_objects(frame)
    # 2. Tracking
    tracked = self._track_objects(detections) 
    # 3. Analyse VLM conditionnelle
    vlm_analysis = await self._analyze_with_vlm(frame, tracked)
    # 4. Détermination actions
    actions = self._determine_actions(alert_level, vlm_analysis)
```

### **Exercices pratiques - MODULE 2**

#### **🔬 Exercice 2.1: Traçage du Flow Principal**
```python
# Étape 1: Ajoutez des logs de debug dans surveillance_system.py
import logging
logging.basicConfig(level=logging.DEBUG)

# Dans HeadlessSurveillanceSystem.__init__():
self.logger = logging.getLogger(__name__)

# Dans run_surveillance():
async def run_surveillance(self):
    self.logger.info("🚀 Démarrage surveillance")
    frame_count = 0
    async for result in self._process_frames():
        frame_count += 1
        self.logger.info(f"📸 Frame {frame_count}: {result.alert_level}")
        self._update_metrics(result)
        if self.save_results:
            self.results.append(result)
    self.logger.info(f"✅ Surveillance terminée: {frame_count} frames")
```

```bash
# Étape 2: Lancez avec traces détaillées
python main_headless_refactored.py --debug --max-frames 5 --video 0

# Étape 3: Analysez les logs pour comprendre:
# - Temps entre capture et analyse
# - Déclenchement des alertes
# - Performance de chaque étape
```

#### **🔬 Exercice 2.2: Modification des seuils d'alerte**
```python
# Étape 1: Créez un script de test des seuils
# test_alert_thresholds.py
import asyncio
from src.core.headless.frame_analyzer import FrameAnalyzer
from config.config_loader import load_config

async def test_thresholds():
    config = load_config()
    analyzer = FrameAnalyzer(config)
    
    # Test différents seuils
    test_scenarios = [
        {"confidence_threshold": 0.3, "alert_threshold": 0.5},
        {"confidence_threshold": 0.5, "alert_threshold": 0.7},
        {"confidence_threshold": 0.7, "alert_threshold": 0.9},
    ]
    
    for scenario in test_scenarios:
        print(f"Test avec seuils: {scenario}")
        # Modifiez les seuils dynamiquement
        analyzer.confidence_threshold = scenario["confidence_threshold"]
        analyzer.alert_threshold = scenario["alert_threshold"]
        
        # Testez avec une frame de test
        # ... votre logique de test
```

#### **🔬 Exercice 2.3: VideoProcessor - Analyse détaillée**
```python
# Étape 1: Créez un script d'analyse du processeur vidéo
# analyze_video_processor.py
from src.core.headless.video_processor import VideoProcessor
import cv2
import time

def analyze_frame_generation():
    processor = VideoProcessor(
        video_source=0,  # Webcam
        max_frames=10,
        frame_skip=2
    )
    
    start_time = time.time()
    frame_count = 0
    
    for frame_id, frame, timestamp in processor.frames_generator():
        frame_count += 1
        print(f"Frame {frame_id}: {frame.shape}, timestamp: {timestamp}")
        
        # Analysez la qualité de la frame
        blur_score = cv2.Laplacian(frame, cv2.CV_64F).var()
        print(f"  Blur score: {blur_score:.2f}")
        
    total_time = time.time() - start_time
    fps = frame_count / total_time
    print(f"FPS moyen: {fps:.2f}")
```

#### **🔬 Exercice 2.4: Monitoring temps réel**
```python
# Étape 1: Créez un moniteur de performance simple
# performance_tracker.py
import time
from collections import defaultdict

class SimplePerformanceTracker:
    def __init__(self):
        self.timings = defaultdict(list)
        self.start_times = {}
    
    def start_timer(self, operation):
        self.start_times[operation] = time.time()
    
    def end_timer(self, operation):
        if operation in self.start_times:
            duration = time.time() - self.start_times[operation]
            self.timings[operation].append(duration)
            del self.start_times[operation]
            return duration
    
    def get_stats(self):
        stats = {}
        for op, times in self.timings.items():
            stats[op] = {
                'avg': sum(times) / len(times),
                'min': min(times),
                'max': max(times),
                'count': len(times)
            }
        return stats

# Étape 2: Intégrez dans FrameAnalyzer
tracker = SimplePerformanceTracker()

async def analyze_frame_with_timing(self, frame, frame_id, timestamp):
    tracker.start_timer('total_analysis')
    
    tracker.start_timer('detection')
    detections = await self._detect_objects(frame)
    tracker.end_timer('detection')
    
    tracker.start_timer('tracking') 
    tracked = self._track_objects(detections)
    tracker.end_timer('tracking')
    
    tracker.start_timer('vlm_analysis')
    vlm_analysis = await self._analyze_with_vlm(frame, tracked)
    tracker.end_timer('vlm_analysis')
    
    tracker.end_timer('total_analysis')
    
    # Affichez stats toutes les 10 frames
    if frame_id % 10 == 0:
        print(tracker.get_stats())
```

#### **🎯 Objectifs d'Apprentissage - Module 2:**
- [ ] Comprendre le flow complet d'une frame
- [ ] Maîtriser les paramètres de performance (frame_skip, max_frames)
- [ ] Identifier les goulots d'étranglement
- [ ] Personnaliser la logique d'alerte
- [ ] Créer ses propres métriques de performance

---

## 🎯 **MODULE 3: Modèles VLM (Intelligence)**

### **Fichiers Clés**
- `src/core/vlm/dynamic_model.py` - Modèle VLM principal
- `src/core/vlm/model_registry.py` - Registre des modèles
- `src/core/vlm/prompt_builder.py` - Construction prompts
- `src/core/orchestrator/vlm_orchestrator.py` - Orchestration VLM

### **Ce que vous devez comprendre**

#### **DynamicVisionLanguageModel**
```python
class DynamicVisionLanguageModel:
    # 1. Support multi-modèles (Kimi, Qwen)
    # 2. Fallback automatique
    # 3. Tool calling intégré
    # 4. Gestion d'erreurs robuste
```

#### **VLMModelRegistry**
```python
# Tous les modèles supportés avec leurs configurations
models = {
    "kimi-vl-a3b-thinking": ModelConfig(...),
    "qwen2-vl-7b-instruct": ModelConfig(...),
    ...
}
```

#### **ModernVLMOrchestrator**
```python
# 3 modes d'orchestration
FAST: 1.2s - Outils essentiels
BALANCED: 2.5s - Équilibre performance/précision  
THOROUGH: 4.8s - Tous les outils avancés
```

### **Exercices pratiques - MODULE 3**

#### **🔬 Exercice 3.1: Benchmark des modèles VLM**
```python
# Étape 1: Créez un script de benchmark
# benchmark_vlm_models.py
import asyncio
import time
import json
from src.core.vlm.dynamic_model import DynamicVisionLanguageModel
from src.core.vlm.model_registry import VLMModelRegistry

async def benchmark_models():
    models_to_test = [
        "kimi-vl-a3b-thinking",
        "qwen2-vl-7b-instruct",
        "claude-3-5-haiku-20241022",
        "gpt-4o-mini-2024-07-18"
    ]
    
    results = {}
    test_image_path = "path/to/test_image.jpg"
    test_prompt = "Décrivez cette image en détail."
    
    for model_name in models_to_test:
        print(f"🧪 Test du modèle: {model_name}")
        
        try:
            vlm = DynamicVisionLanguageModel(model_name)
            
            # Mesure du temps de chargement
            start_time = time.time()
            await vlm.initialize()
            load_time = time.time() - start_time
            
            # Mesure du temps d'inférence
            start_time = time.time()
            response = await vlm.analyze_image(test_image_path, test_prompt)
            inference_time = time.time() - start_time
            
            results[model_name] = {
                "load_time": load_time,
                "inference_time": inference_time,
                "response_length": len(response),
                "status": "success"
            }
            
        except Exception as e:
            results[model_name] = {
                "error": str(e),
                "status": "failed"
            }
    
    # Sauvegarde des résultats
    with open("vlm_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results

# Étape 2: Lancez le benchmark
if __name__ == "__main__":
    results = asyncio.run(benchmark_models())
    print("\n📊 Résultats du benchmark:")
    for model, stats in results.items():
        if stats["status"] == "success":
            print(f"{model}:")
            print(f"  ⏱️  Temps chargement: {stats['load_time']:.2f}s")
            print(f"  🚀 Temps inférence: {stats['inference_time']:.2f}s")
            print(f"  📝 Longueur réponse: {stats['response_length']} chars")
        else:
            print(f"{model}: ❌ {stats['error']}")
```

#### **🔬 Exercice 3.2: Test des modes d'orchestration**
```python
# Étape 1: Créez un comparateur de modes
# compare_orchestration_modes.py
import asyncio
from src.core.orchestrator.vlm_orchestrator import ModernVLMOrchestrator
from src.core.types import DetectedObject, BoundingBox

async def compare_modes():
    orchestrator = ModernVLMOrchestrator()
    
    # Données de test simulées
    test_frame = "path/to/test_frame.jpg"
    test_objects = [
        DetectedObject(
            class_name="person",
            confidence=0.85,
            bbox=BoundingBox(x=100, y=100, width=200, height=300),
            track_id=1
        )
    ]
    
    modes = ["FAST", "BALANCED", "THOROUGH"]
    results = {}
    
    for mode in modes:
        print(f"🎯 Test mode: {mode}")
        
        start_time = time.time()
        
        # Changez le mode de l'orchestrateur
        orchestrator.set_mode(mode)
        
        # Analysez avec ce mode
        analysis = await orchestrator.analyze_scene(
            frame=test_frame,
            detected_objects=test_objects,
            context={"alert_level": "medium"}
        )
        
        execution_time = time.time() - start_time
        
        results[mode] = {
            "execution_time": execution_time,
            "tools_used": analysis.get("tools_used", []),
            "analysis_quality": len(analysis.get("description", "")),
            "actions_suggested": len(analysis.get("actions", []))
        }
    
    # Analyse comparative
    print("\n📈 Analyse comparative:")
    for mode, stats in results.items():
        print(f"\n{mode}:")
        print(f"  ⏱️  Temps: {stats['execution_time']:.2f}s")
        print(f"  🔧 Outils: {len(stats['tools_used'])}")
        print(f"  📝 Qualité: {stats['analysis_quality']} chars")
        print(f"  🎯 Actions: {stats['actions_suggested']}")
    
    return results

# Étape 2: Analysez les trade-offs
def analyze_tradeoffs(results):
    print("\n⚖️ Analyse des trade-offs:")
    
    fastest = min(results.items(), key=lambda x: x[1]['execution_time'])
    most_detailed = max(results.items(), key=lambda x: x[1]['analysis_quality'])
    most_tools = max(results.items(), key=lambda x: len(x[1]['tools_used']))
    
    print(f"Plus rapide: {fastest[0]} ({fastest[1]['execution_time']:.2f}s)")
    print(f"Plus détaillé: {most_detailed[0]} ({most_detailed[1]['analysis_quality']} chars)")
    print(f"Plus d'outils: {most_tools[0]} ({len(most_tools[1]['tools_used'])} outils)")
```

#### **🔬 Exercice 3.3: Création de prompts personnalisés**
```python
# Étape 1: Créez un générateur de prompts personnalisés
# custom_prompt_builder.py
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class CustomPromptTemplate:
    name: str
    template: str
    variables: List[str]
    context_requirements: Dict[str, Any]

class CustomPromptBuilder:
    def __init__(self):
        self.templates = {
            "security_focus": CustomPromptTemplate(
                name="Security Focus",
                template="""
Analysez cette image de surveillance avec un focus sécuritaire:

Contexte: {context}
Objets détectés: {objects}
Timestamp: {timestamp}

Questions spécifiques:
1. Y a-t-il des comportements suspects?
2. Les accès sont-ils autorisés?
3. Des objets dangereux sont-ils présents?
4. Le niveau d'alerte recommandé?

Répondez en format JSON avec:
- risk_level: LOW/MEDIUM/HIGH/CRITICAL
- suspicious_activities: []
- recommended_actions: []
- confidence_score: 0.0-1.0
                """,
                variables=["context", "objects", "timestamp"],
                context_requirements={"security_zone": True}
            ),
            
            "customer_behavior": CustomPromptTemplate(
                name="Customer Behavior",
                template="""
Analysez le comportement client dans cette scène commerciale:

Zone: {zone}
Personnes détectées: {people_count}
Objets d'intérêt: {products}

Analysez:
1. Patterns de déplacement
2. Interactions avec produits
3. Temps passé dans la zone
4. Indicateurs d'intérêt d'achat

Format de réponse:
- behavior_type: BROWSING/INTERESTED/PURCHASING/LEAVING
- engagement_score: 0.0-1.0
- product_interactions: []
- recommendations: []
                """,
                variables=["zone", "people_count", "products"],
                context_requirements={"commercial_context": True}
            )
        }
    
    def build_prompt(self, template_name: str, variables: Dict[str, Any]) -> str:
        template = self.templates[template_name]
        return template.template.format(**variables)
    
    def add_custom_template(self, template: CustomPromptTemplate):
        self.templates[template.name] = template

# Étape 2: Testez vos prompts personnalisés
async def test_custom_prompts():
    builder = CustomPromptBuilder()
    vlm = DynamicVisionLanguageModel("claude-3-5-haiku-20241022")
    
    # Test du prompt sécuritaire
    security_prompt = builder.build_prompt("security_focus", {
        "context": "Zone d'entrée principale - 14h30",
        "objects": "2 personnes, 1 sac, porte d'accès",
        "timestamp": "2024-01-15 14:30:22"
    })
    
    response = await vlm.analyze_with_prompt(
        image_path="test_image.jpg",
        prompt=security_prompt
    )
    
    print("🔒 Analyse sécuritaire:")
    print(response)
    
    return response

# Étape 3: Créez votre propre template
def create_your_template():
    # Exercice: Créez un template pour votre use-case spécifique
    your_template = CustomPromptTemplate(
        name="your_custom_analysis",
        template="""
        # Votre template personnalisé ici
        # Variables: {var1}, {var2}
        # Instructions spécifiques à votre cas d'usage
        """,
        variables=["var1", "var2"],
        context_requirements={"your_requirement": True}
    )
    
    builder = CustomPromptBuilder()
    builder.add_custom_template(your_template)
    
    return builder
```

#### **🔬 Exercice 3.4: Optimisation et fallback**
```python
# Étape 1: Testez le système de fallback
# test_fallback_system.py
import asyncio
from src.core.vlm.dynamic_model import DynamicVisionLanguageModel

async def test_fallback_mechanism():
    # Configurez un modèle principal qui pourrait échouer
    primary_model = "modele_inexistant"
    fallback_models = ["claude-3-5-haiku-20241022", "gpt-4o-mini-2024-07-18"]
    
    vlm = DynamicVisionLanguageModel(
        primary_model, 
        fallback_models=fallback_models
    )
    
    try:
        result = await vlm.analyze_image(
            "test_image.jpg",
            "Décrivez cette image"
        )
        print(f"✅ Analyse réussie avec: {vlm.current_model}")
        print(f"Résultat: {result[:100]}...")
        
    except Exception as e:
        print(f"❌ Tous les modèles ont échoué: {e}")

# Étape 2: Optimisez les performances
async def optimize_vlm_performance():
    vlm = DynamicVisionLanguageModel("claude-3-5-haiku-20241022")
    
    # Test avec différentes tailles d'images
    image_sizes = [
        (640, 480),   # Standard
        (1280, 720),  # HD
        (1920, 1080)  # Full HD
    ]
    
    for width, height in image_sizes:
        start_time = time.time()
        # Redimensionnez et analysez l'image
        # ... votre code de redimensionnement
        processing_time = time.time() - start_time
        
        print(f"Taille {width}x{height}: {processing_time:.2f}s")
```

#### **🎯 Objectifs d'Apprentissage - Module 3:**
- [ ] Comparer les performances de différents modèles VLM
- [ ] Maîtriser les 3 modes d'orchestration (FAST/BALANCED/THOROUGH)
- [ ] Créer des prompts personnalisés pour votre use-case
- [ ] Comprendre le système de fallback et robustesse
- [ ] Optimiser les performances selon vos contraintes

---

## 🎯 **MODULE 4: Détection et Tracking**

### **Fichiers Clés**
- `src/detection/yolo_detector.py` - Détection YOLO
- `src/detection/tracking/byte_tracker.py` - Tracking BYTETracker
- `src/core/types.py` - Types DetectedObject, BoundingBox

### **Ce que vous devez comprendre**

#### **YOLODetector**
```python
def detect(self, frame):
    # 1. Préprocessing de l'image
    # 2. Inférence YOLO
    # 3. Post-processing (NMS, filtrage)
    # 4. Conversion vers DetectedObject
```

#### **BYTETracker**
```python
def update(self, detections):
    # 1. Association des détections aux tracks existants
    # 2. Gestion des nouvelles détections
    # 3. Suppression des tracks perdus
    # 4. Assignation d'IDs uniques
```

### **Exercices pratiques - MODULE 4**

#### **🔬 Exercice 4.1: Optimisation YOLO**
```python
# Étape 1: Testez différents seuils de confiance
# test_yolo_thresholds.py
import cv2
import numpy as np
from src.detection.yolo_detector import YOLODetector
import matplotlib.pyplot as plt

def test_confidence_thresholds():
    detector = YOLODetector()
    test_image = cv2.imread("path/to/test_image.jpg")
    
    # Test différents seuils
    confidence_thresholds = [0.3, 0.5, 0.7, 0.9]
    results = {}
    
    for conf_threshold in confidence_thresholds:
        detector.confidence_threshold = conf_threshold
        detections = detector.detect(test_image)
        
        results[conf_threshold] = {
            "num_detections": len(detections),
            "avg_confidence": np.mean([d.confidence for d in detections]) if detections else 0,
            "classes_detected": list(set([d.class_name for d in detections]))
        }
        
        print(f"Seuil {conf_threshold}: {len(detections)} détections")
        for detection in detections[:3]:  # Top 3
            print(f"  - {detection.class_name}: {detection.confidence:.3f}")
    
    return results

# Étape 2: Visualisez l'impact des seuils
def visualize_detection_impact(results):
    thresholds = list(results.keys())
    num_detections = [results[t]["num_detections"] for t in thresholds]
    avg_confidences = [results[t]["avg_confidence"] for t in thresholds]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Nombre de détections vs seuil
    ax1.plot(thresholds, num_detections, 'bo-')
    ax1.set_xlabel('Seuil de confiance')
    ax1.set_ylabel('Nombre de détections')
    ax1.set_title('Impact du seuil sur le nombre de détections')
    ax1.grid(True)
    
    # Confiance moyenne vs seuil
    ax2.plot(thresholds, avg_confidences, 'ro-')
    ax2.set_xlabel('Seuil de confiance')
    ax2.set_ylabel('Confiance moyenne')
    ax2.set_title('Impact du seuil sur la qualité')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('yolo_threshold_analysis.png')
    plt.show()

# Étape 3: Test du NMS (Non-Maximum Suppression)
def test_nms_parameters():
    detector = YOLODetector()
    test_image = cv2.imread("path/to/crowded_scene.jpg")
    
    nms_thresholds = [0.3, 0.5, 0.7]
    
    for nms_threshold in nms_thresholds:
        detector.nms_threshold = nms_threshold
        detections = detector.detect(test_image)
        
        print(f"NMS {nms_threshold}: {len(detections)} détections finales")
        
        # Analysez la densité des boîtes
        if len(detections) > 1:
            overlaps = detector.calculate_overlaps(detections)
            print(f"  Overlap moyen: {np.mean(overlaps):.3f}")
```

#### **🔬 Exercice 4.2: Analyse du tracking BYTETracker**
```python
# Étape 1: Analysez la performance du tracker
# analyze_tracking_performance.py
from src.detection.tracking.byte_tracker import BYTETracker
from src.detection.yolo_detector import YOLODetector
import cv2

class TrackingAnalyzer:
    def __init__(self):
        self.tracker = BYTETracker()
        self.detector = YOLODetector()
        self.track_history = {}
        self.metrics = {
            'new_tracks': 0,
            'lost_tracks': 0,
            'track_switches': 0,
            'avg_track_length': []
        }
    
    def analyze_video_tracking(self, video_path, max_frames=100):
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Détection
            detections = self.detector.detect(frame)
            
            # Tracking
            tracked_objects = self.tracker.update(detections)
            
            # Analyse des métriques
            self._update_metrics(tracked_objects, frame_count)
            
            # Visualisation (optionnelle)
            if frame_count % 10 == 0:
                self._visualize_tracks(frame, tracked_objects, frame_count)
            
            frame_count += 1
        
        cap.release()
        return self._compute_final_metrics()
    
    def _update_metrics(self, tracked_objects, frame_count):
        current_track_ids = set([obj.track_id for obj in tracked_objects])
        
        if hasattr(self, 'previous_track_ids'):
            # Nouveaux tracks
            new_tracks = current_track_ids - self.previous_track_ids
            self.metrics['new_tracks'] += len(new_tracks)
            
            # Tracks perdus
            lost_tracks = self.previous_track_ids - current_track_ids
            self.metrics['lost_tracks'] += len(lost_tracks)
            
            for track_id in lost_tracks:
                if track_id in self.track_history:
                    track_length = len(self.track_history[track_id])
                    self.metrics['avg_track_length'].append(track_length)
        
        # Mise à jour de l'historique
        for obj in tracked_objects:
            if obj.track_id not in self.track_history:
                self.track_history[obj.track_id] = []
            self.track_history[obj.track_id].append({
                'frame': frame_count,
                'bbox': obj.bbox,
                'confidence': obj.confidence
            })
        
        self.previous_track_ids = current_track_ids
    
    def _visualize_tracks(self, frame, tracked_objects, frame_count):
        vis_frame = frame.copy()
        
        for obj in tracked_objects:
            # Dessinez la bounding box
            x, y, w, h = obj.bbox.x, obj.bbox.y, obj.bbox.width, obj.bbox.height
            cv2.rectangle(vis_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Ajoutez l'ID de tracking
            cv2.putText(vis_frame, f"ID:{obj.track_id}", 
                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Dessinez la trajectoire (derniers 10 points)
            if obj.track_id in self.track_history:
                history = self.track_history[obj.track_id][-10:]
                points = [(int(h['bbox'].x + h['bbox'].width/2), 
                          int(h['bbox'].y + h['bbox'].height/2)) for h in history]
                
                for i in range(1, len(points)):
                    cv2.line(vis_frame, points[i-1], points[i], (255, 255, 0), 2)
        
        cv2.imwrite(f"tracking_frame_{frame_count:04d}.jpg", vis_frame)
    
    def _compute_final_metrics(self):
        avg_track_length = np.mean(self.metrics['avg_track_length']) if self.metrics['avg_track_length'] else 0
        
        final_metrics = {
            'total_tracks': len(self.track_history),
            'new_tracks': self.metrics['new_tracks'],
            'lost_tracks': self.metrics['lost_tracks'],
            'avg_track_length': avg_track_length,
            'track_efficiency': 1 - (self.metrics['track_switches'] / max(self.metrics['new_tracks'], 1))
        }
        
        return final_metrics

# Étape 2: Testez différents paramètres de tracking
def test_tracking_parameters():
    analyzer = TrackingAnalyzer()
    
    # Paramètres à tester
    test_configs = [
        {"track_thresh": 0.5, "track_buffer": 30, "match_thresh": 0.8},
        {"track_thresh": 0.6, "track_buffer": 50, "match_thresh": 0.7},
        {"track_thresh": 0.4, "track_buffer": 20, "match_thresh": 0.9}
    ]
    
    results = {}
    
    for i, config in enumerate(test_configs):
        print(f"🧪 Test configuration {i+1}: {config}")
        
        # Configurez le tracker
        analyzer.tracker.track_thresh = config["track_thresh"]
        analyzer.tracker.track_buffer = config["track_buffer"]
        analyzer.tracker.match_thresh = config["match_thresh"]
        
        # Analysez
        metrics = analyzer.analyze_video_tracking("test_video.mp4")
        results[f"config_{i+1}"] = {**config, **metrics}
        
        print(f"  Résultats: {metrics}")
    
    return results

# Étape 3: Optimisation automatique des paramètres
def optimize_tracking_parameters():
    # Implémentation d'une recherche par grille simple
    best_score = 0
    best_params = {}
    
    track_thresh_values = [0.4, 0.5, 0.6, 0.7]
    match_thresh_values = [0.6, 0.7, 0.8, 0.9]
    
    for track_thresh in track_thresh_values:
        for match_thresh in match_thresh_values:
            analyzer = TrackingAnalyzer()
            analyzer.tracker.track_thresh = track_thresh
            analyzer.tracker.match_thresh = match_thresh
            
            metrics = analyzer.analyze_video_tracking("test_video.mp4", max_frames=50)
            
            # Score composite (à ajuster selon vos priorités)
            score = (metrics['avg_track_length'] * 0.4 + 
                    metrics['track_efficiency'] * 0.6 - 
                    metrics['lost_tracks'] * 0.01)
            
            if score > best_score:
                best_score = score
                best_params = {
                    'track_thresh': track_thresh,
                    'match_thresh': match_thresh,
                    'score': score,
                    'metrics': metrics
                }
    
    print(f"🏆 Meilleurs paramètres: {best_params}")
    return best_params
```

#### **🔬 Exercice 4.3: Visualisation avancée**
```python
# Étape 1: Créez un visualiseur interactif
# interactive_detection_viewer.py
import cv2
import numpy as np
from dataclasses import dataclass
from typing import List

@dataclass
class VisualizationConfig:
    show_confidence: bool = True
    show_class_names: bool = True
    show_track_ids: bool = True
    show_trajectories: bool = True
    trajectory_length: int = 10
    bbox_thickness: int = 2
    font_scale: float = 0.6

class InteractiveViewer:
    def __init__(self, config: VisualizationConfig = None):
        self.config = config or VisualizationConfig()
        self.colors = self._generate_colors(100)  # 100 couleurs différentes
        self.track_history = {}
    
    def _generate_colors(self, num_colors):
        """Génère des couleurs distinctes pour le tracking"""
        colors = []
        for i in range(num_colors):
            hue = int(180 * i / num_colors)
            color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
            colors.append((int(color[0]), int(color[1]), int(color[2])))
        return colors
    
    def visualize_detections(self, frame, detections, tracked_objects=None):
        """Visualise les détections et le tracking"""
        vis_frame = frame.copy()
        
        # Dessinez les détections simples
        for detection in detections:
            self._draw_detection(vis_frame, detection, (0, 255, 0))
        
        # Dessinez les objets trackés
        if tracked_objects:
            for obj in tracked_objects:
                color = self.colors[obj.track_id % len(self.colors)]
                self._draw_tracked_object(vis_frame, obj, color)
        
        return vis_frame
    
    def _draw_detection(self, frame, detection, color):
        """Dessine une détection simple"""
        x, y, w, h = detection.bbox.x, detection.bbox.y, detection.bbox.width, detection.bbox.height
        
        # Boîte
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, self.config.bbox_thickness)
        
        # Texte
        if self.config.show_class_names or self.config.show_confidence:
            text = ""
            if self.config.show_class_names:
                text += detection.class_name
            if self.config.show_confidence:
                text += f" {detection.confidence:.2f}"
            
            cv2.putText(frame, text, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, self.config.font_scale, color, 2)
    
    def _draw_tracked_object(self, frame, obj, color):
        """Dessine un objet tracké avec trajectoire"""
        x, y, w, h = obj.bbox.x, obj.bbox.y, obj.bbox.width, obj.bbox.height
        
        # Boîte principale
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, self.config.bbox_thickness)
        
        # ID de tracking
        if self.config.show_track_ids:
            cv2.putText(frame, f"ID:{obj.track_id}", 
                       (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, self.config.font_scale, color, 2)
        
        # Classe et confiance
        if self.config.show_class_names or self.config.show_confidence:
            text = ""
            if self.config.show_class_names:
                text += obj.class_name
            if self.config.show_confidence:
                text += f" {obj.confidence:.2f}"
            
            cv2.putText(frame, text, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, self.config.font_scale, color, 2)
        
        # Trajectoire
        if self.config.show_trajectories and obj.track_id in self.track_history:
            self._draw_trajectory(frame, obj.track_id, color)
        
        # Mise à jour de l'historique
        center = (int(x + w/2), int(y + h/2))
        if obj.track_id not in self.track_history:
            self.track_history[obj.track_id] = []
        
        self.track_history[obj.track_id].append(center)
        
        # Gardez seulement les N derniers points
        if len(self.track_history[obj.track_id]) > self.config.trajectory_length:
            self.track_history[obj.track_id] = self.track_history[obj.track_id][-self.config.trajectory_length:]
    
    def _draw_trajectory(self, frame, track_id, color):
        """Dessine la trajectoire d'un objet"""
        points = self.track_history[track_id]
        
        if len(points) < 2:
            return
        
        # Dessinez des lignes entre les points successifs
        for i in range(1, len(points)):
            # Épaisseur dégradée (plus récent = plus épais)
            thickness = max(1, int(3 * i / len(points)))
            cv2.line(frame, points[i-1], points[i], color, thickness)
        
        # Point actuel plus visible
        cv2.circle(frame, points[-1], 4, color, -1)

# Étape 2: Interface de contrôle en temps réel
def create_interactive_controls():
    """Crée des contrôles trackbar pour ajuster les paramètres en temps réel"""
    cv2.namedWindow('Controls', cv2.WINDOW_NORMAL)
    
    # Créez les trackbars
    cv2.createTrackbar('Confidence', 'Controls', 50, 100, lambda x: None)
    cv2.createTrackbar('NMS Threshold', 'Controls', 50, 100, lambda x: None)
    cv2.createTrackbar('Track Threshold', 'Controls', 50, 100, lambda x: None)
    cv2.createTrackbar('Show Trajectories', 'Controls', 1, 1, lambda x: None)
    cv2.createTrackbar('Trajectory Length', 'Controls', 10, 50, lambda x: None)
    
    return True

def get_control_values():
    """Récupère les valeurs actuelles des contrôles"""
    return {
        'confidence': cv2.getTrackbarPos('Confidence', 'Controls') / 100.0,
        'nms_threshold': cv2.getTrackbarPos('NMS Threshold', 'Controls') / 100.0,
        'track_threshold': cv2.getTrackbarPos('Track Threshold', 'Controls') / 100.0,
        'show_trajectories': bool(cv2.getTrackbarPos('Show Trajectories', 'Controls')),
        'trajectory_length': cv2.getTrackbarPos('Trajectory Length', 'Controls')
    }

# Étape 3: Application complète interactive
def run_interactive_detection():
    """Lance une session de détection/tracking interactive"""
    # Initialisation
    detector = YOLODetector()
    tracker = BYTETracker()
    viewer = InteractiveViewer()
    
    create_interactive_controls()
    
    cap = cv2.VideoCapture(0)  # Webcam
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Récupérez les paramètres des contrôles
        controls = get_control_values()
        
        # Ajustez les paramètres du détecteur
        detector.confidence_threshold = controls['confidence']
        detector.nms_threshold = controls['nms_threshold']
        tracker.track_thresh = controls['track_threshold']
        
        # Ajustez les paramètres du visualiseur
        viewer.config.show_trajectories = controls['show_trajectories']
        viewer.config.trajectory_length = controls['trajectory_length']
        
        # Traitement
        detections = detector.detect(frame)
        tracked_objects = tracker.update(detections)
        
        # Visualisation
        vis_frame = viewer.visualize_detections(frame, detections, tracked_objects)
        
        # Affichez les métriques
        cv2.putText(vis_frame, f"Detections: {len(detections)}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis_frame, f"Tracks: {len(tracked_objects)}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Interactive Detection', vis_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
```

#### **🎯 Objectifs d'Apprentissage - Module 4:**
- [ ] Optimiser les seuils YOLO pour votre environnement
- [ ] Comprendre l'impact des paramètres NMS sur la qualité
- [ ] Maîtriser le tuning des paramètres de tracking BYTETracker
- [ ] Créer des visualisations personnalisées informatives
- [ ] Développer des métriques de performance pour évaluer le tracking

---

## 🎯 **MODULE 5: Outils Avancés IA**

### **Fichiers Clés**
- `src/advanced_tools/sam2_segmentation.py` - Segmentation SAM2
- `src/advanced_tools/pose_estimation.py` - Estimation de pose
- `src/advanced_tools/trajectory_analyzer.py` - Analyse trajectoires
- `src/advanced_tools/multimodal_fusion.py` - Fusion multimodale

### **Ce que vous devez comprendre**

#### **Les 8 Outils**
1. **SAM2Segmentator** - Segmentation précise
2. **DinoV2FeatureExtractor** - Features visuelles
3. **OpenPoseEstimator** - Analyse posturale
4. **TrajectoryAnalyzer** - Patterns mouvement
5. **MultiModalFusion** - Fusion données
6. **TemporalTransformer** - Analyse temporelle
7. **AdversarialDetector** - Protection attaques
8. **DomainAdapter** - Adaptation contexte

#### **Sélection Adaptative**
```python
def select_tools(self, context):
    if context["alert_level"] == "normal":
        return ["basic_segmentation", "pose_estimation"]
    else:
        return ["all_advanced_tools"]  # 6 outils
```

### **Exercices pratiques - MODULE 5**

#### **🔬 Exercice 5.1: Test des outils avancés individuellement**
```python
# Étape 1: Créez un testeur d'outils modulaire
# advanced_tools_tester.py
import asyncio
import time
import numpy as np
from pathlib import Path
from src.advanced_tools.sam2_segmentation import SAM2Segmentator
from src.advanced_tools.pose_estimation import OpenPoseEstimator
from src.advanced_tools.trajectory_analyzer import TrajectoryAnalyzer
from src.advanced_tools.multimodal_fusion import MultiModalFusion

class AdvancedToolsTester:
    def __init__(self):
        self.tools = {
            "sam2": SAM2Segmentator(),
            "pose": OpenPoseEstimator(), 
            "trajectory": TrajectoryAnalyzer(),
            "fusion": MultiModalFusion()
        }
        self.results = {}
    
    async def test_tool_performance(self, tool_name, test_data):
        """Teste la performance d'un outil spécifique"""
        if tool_name not in self.tools:
            raise ValueError(f"Outil {tool_name} non disponible")
        
        tool = self.tools[tool_name]
        
        # Mesure de performance
        start_time = time.time()
        memory_before = self._get_memory_usage()
        
        try:
            if tool_name == "sam2":
                result = await tool.segment_objects(
                    test_data["image"], 
                    test_data["bboxes"]
                )
            elif tool_name == "pose":
                result = await tool.estimate_poses(test_data["image"])
            elif tool_name == "trajectory":
                result = await tool.analyze_trajectories(test_data["tracks"])
            elif tool_name == "fusion":
                result = await tool.fuse_modalities(
                    test_data["visual_features"],
                    test_data["temporal_features"]
                )
            
            execution_time = time.time() - start_time
            memory_after = self._get_memory_usage()
            
            self.results[tool_name] = {
                "success": True,
                "execution_time": execution_time,
                "memory_usage": memory_after - memory_before,
                "result_quality": self._evaluate_result_quality(tool_name, result),
                "result": result
            }
            
        except Exception as e:
            self.results[tool_name] = {
                "success": False,
                "error": str(e),
                "execution_time": time.time() - start_time
            }
        
        return self.results[tool_name]
    
    def _get_memory_usage(self):
        """Mesure l'utilisation mémoire actuelle"""
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # MB
    
    def _evaluate_result_quality(self, tool_name, result):
        """Évalue la qualité du résultat selon l'outil"""
        if tool_name == "sam2":
            # Qualité basée sur la précision de segmentation
            return {
                "num_segments": len(result.get("segments", [])),
                "avg_confidence": np.mean([s.confidence for s in result.get("segments", [])]),
                "detail_level": sum([len(s.mask) for s in result.get("segments", [])])
            }
        elif tool_name == "pose":
            # Qualité basée sur la détection de poses
            return {
                "num_poses": len(result.get("poses", [])),
                "avg_keypoint_confidence": np.mean([p.confidence for p in result.get("poses", [])]),
                "completeness": np.mean([p.completeness for p in result.get("poses", [])])
            }
        # Ajoutez d'autres évaluations selon vos besoins
        
        return {"score": 1.0}  # Score par défaut

# Étape 2: Benchmark complet des outils
async def benchmark_all_tools():
    tester = AdvancedToolsTester()
    
    # Données de test
    test_image = "path/to/test_image.jpg"
    test_data = {
        "sam2": {
            "image": test_image,
            "bboxes": [(100, 100, 200, 200), (300, 150, 100, 150)]
        },
        "pose": {
            "image": test_image
        },
        "trajectory": {
            "tracks": [
                {"id": 1, "points": [(100, 200), (110, 205), (120, 210)]},
                {"id": 2, "points": [(200, 300), (190, 295), (180, 290)]}
            ]
        },
        "fusion": {
            "visual_features": np.random.rand(512),
            "temporal_features": np.random.rand(256)
        }
    }
    
    print("🚀 Démarrage du benchmark des outils avancés\n")
    
    for tool_name, data in test_data.items():
        print(f"🧪 Test de l'outil: {tool_name}")
        result = await tester.test_tool_performance(tool_name, data)
        
        if result["success"]:
            print(f"  ✅ Succès - {result['execution_time']:.2f}s")
            print(f"  💾 Mémoire: {result['memory_usage']:.1f}MB")
            print(f"  📊 Qualité: {result['result_quality']}")
        else:
            print(f"  ❌ Échec: {result['error']}")
        print()
    
    return tester.results

# Étape 3: Comparaison des configurations d'outils
def compare_tool_configurations():
    """Compare différentes configurations d'activation d'outils"""
    configurations = [
        {
            "name": "Minimal",
            "tools": ["pose"],
            "expected_time": 1.0,
            "use_cases": ["Détection de mouvement basique"]
        },
        {
            "name": "Standard", 
            "tools": ["sam2", "pose"],
            "expected_time": 2.5,
            "use_cases": ["Analyse de comportement"]
        },
        {
            "name": "Advanced",
            "tools": ["sam2", "pose", "trajectory", "fusion"],
            "expected_time": 5.0,
            "use_cases": ["Analyse comportementale complète"]
        }
    ]
    
    print("⚖️ Comparaison des configurations:\n")
    
    for config in configurations:
        print(f"📋 {config['name']}:")
        print(f"  🔧 Outils: {', '.join(config['tools'])}")
        print(f"  ⏱️  Temps estimé: {config['expected_time']}s")
        print(f"  🎯 Cas d'usage: {', '.join(config['use_cases'])}")
        print()
```

#### **🔬 Exercice 5.2: Création d'un nouvel outil personnalisé**
```python
# Étape 1: Créez votre propre outil avancé
# custom_advanced_tool.py
from abc import ABC, abstractmethod
import numpy as np
import cv2
from typing import Dict, Any, List
from dataclasses import dataclass

@dataclass
class ToolResult:
    success: bool
    data: Dict[str, Any]
    confidence: float
    processing_time: float
    metadata: Dict[str, Any] = None

class BaseAdvancedTool(ABC):
    """Classe de base pour tous les outils avancés"""
    
    def __init__(self, name: str):
        self.name = name
        self.initialized = False
        self.config = {}
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialise l'outil"""
        pass
    
    @abstractmethod
    async def process(self, **kwargs) -> ToolResult:
        """Traite les données d'entrée"""
        pass
    
    def configure(self, **config):
        """Configure l'outil avec des paramètres personnalisés"""
        self.config.update(config)

# Étape 2: Implémentez votre outil personnalisé - Détecteur d'émotions
class EmotionDetector(BaseAdvancedTool):
    """Détecteur d'émotions basé sur l'expression faciale"""
    
    def __init__(self):
        super().__init__("emotion_detector")
        self.model = None
        self.face_cascade = None
    
    async def initialize(self) -> bool:
        """Initialise le détecteur d'émotions"""
        try:
            # Chargez votre modèle d'émotion (ex: avec transformers)
            # from transformers import pipeline
            # self.model = pipeline("image-classification", model="emotion-model")
            
            # Pour cet exemple, on simule avec OpenCV
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            
            self.initialized = True
            return True
            
        except Exception as e:
            print(f"Erreur initialisation EmotionDetector: {e}")
            return False
    
    async def process(self, image: np.ndarray, faces: List[Dict] = None) -> ToolResult:
        """Détecte les émotions dans les visages"""
        import time
        start_time = time.time()
        
        try:
            if not self.initialized:
                await self.initialize()
            
            if faces is None:
                # Détection automatique des visages
                faces = self._detect_faces(image)
            
            emotions_detected = []
            
            for face in faces:
                x, y, w, h = face['bbox']
                face_roi = image[y:y+h, x:x+w]
                
                # Analyse d'émotion simulée (remplacez par votre modèle réel)
                emotion_scores = self._analyze_emotion(face_roi)
                
                emotions_detected.append({
                    'bbox': (x, y, w, h),
                    'emotions': emotion_scores,
                    'dominant_emotion': max(emotion_scores.items(), key=lambda x: x[1])
                })
            
            processing_time = time.time() - start_time
            
            return ToolResult(
                success=True,
                data={
                    'emotions': emotions_detected,
                    'num_faces': len(faces),
                    'dominant_emotions': [e['dominant_emotion'][0] for e in emotions_detected]
                },
                confidence=np.mean([e['dominant_emotion'][1] for e in emotions_detected]) if emotions_detected else 0.0,
                processing_time=processing_time,
                metadata={'tool_version': '1.0', 'method': 'opencv_simulation'}
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                data={'error': str(e)},
                confidence=0.0,
                processing_time=time.time() - start_time
            )
    
    def _detect_faces(self, image: np.ndarray) -> List[Dict]:
        """Détecte les visages dans l'image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        return [{'bbox': (x, y, w, h)} for (x, y, w, h) in faces]
    
    def _analyze_emotion(self, face_roi: np.ndarray) -> Dict[str, float]:
        """Analyse l'émotion d'un visage (version simulée)"""
        # Dans un vrai cas, vous utiliseriez un modèle pré-entraîné
        # Ici on simule avec des valeurs aléatoires basées sur des caractéristiques visuelles
        
        emotions = ['happy', 'sad', 'angry', 'surprised', 'neutral', 'fear', 'disgust']
        
        # Analyse basique basée sur la luminosité et les contours (très simplifiée)
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray_face)
        edges = cv2.Canny(gray_face, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # Scores simulés (à remplacer par un vrai modèle)
        scores = {}
        base_scores = np.random.rand(len(emotions))
        
        # Ajustements basés sur les caractéristiques visuelles
        if brightness > 120:  # Visage plus lumineux -> plus probable d'être heureux
            base_scores[emotions.index('happy')] += 0.3
        
        if edge_density > 0.1:  # Beaucoup de contours -> plus de tension
            base_scores[emotions.index('angry')] += 0.2
            base_scores[emotions.index('surprised')] += 0.2
        
        # Normalisation
        base_scores = base_scores / np.sum(base_scores)
        
        return {emotion: float(score) for emotion, score in zip(emotions, base_scores)}

# Étape 3: Intégrez votre outil dans le système
class CustomToolsManager:
    """Gestionnaire pour les outils personnalisés"""
    
    def __init__(self):
        self.custom_tools = {}
    
    def register_tool(self, tool: BaseAdvancedTool):
        """Enregistre un nouvel outil personnalisé"""
        self.custom_tools[tool.name] = tool
        print(f"✅ Outil {tool.name} enregistré")
    
    async def initialize_all_tools(self):
        """Initialise tous les outils enregistrés"""
        for name, tool in self.custom_tools.items():
            success = await tool.initialize()
            print(f"{'✅' if success else '❌'} Initialisation {name}: {'Succès' if success else 'Échec'}")
    
    async def process_with_tool(self, tool_name: str, **kwargs) -> ToolResult:
        """Utilise un outil spécifique pour traiter des données"""
        if tool_name not in self.custom_tools:
            raise ValueError(f"Outil {tool_name} non trouvé")
        
        tool = self.custom_tools[tool_name]
        return await tool.process(**kwargs)

# Étape 4: Test de votre outil personnalisé
async def test_custom_emotion_tool():
    # Créez et enregistrez votre outil
    manager = CustomToolsManager()
    emotion_tool = EmotionDetector()
    manager.register_tool(emotion_tool)
    
    # Initialisez
    await manager.initialize_all_tools()
    
    # Testez avec une image
    test_image = cv2.imread("path/to/image_with_faces.jpg")
    
    result = await manager.process_with_tool(
        "emotion_detector",
        image=test_image
    )
    
    print("🎭 Résultats de détection d'émotion:")
    print(f"Succès: {result.success}")
    print(f"Temps de traitement: {result.processing_time:.3f}s")
    print(f"Nombre de visages: {result.data.get('num_faces', 0)}")
    print(f"Émotions dominantes: {result.data.get('dominant_emotions', [])}")
    
    return result
```

#### **🔬 Exercice 5.3: Optimisation sélective des outils**
```python
# Étape 1: Système de sélection adaptative intelligent
# adaptive_tool_selector.py
from enum import Enum
from typing import List, Dict, Set
import numpy as np

class AlertLevel(Enum):
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"

class ContextType(Enum):
    INDOOR = "indoor"
    OUTDOOR = "outdoor"
    CROWDED = "crowded"
    ISOLATED = "isolated"
    DAYLIGHT = "daylight"
    NIGHT = "night"

class AdaptiveToolSelector:
    """Sélectionne intelligemment les outils selon le contexte"""
    
    def __init__(self):
        self.tool_profiles = {
            "sam2_segmentation": {
                "cost": 3.0,  # Temps relatif
                "accuracy": 0.95,
                "best_contexts": [ContextType.OUTDOOR, ContextType.DAYLIGHT],
                "alert_threshold": AlertLevel.MEDIUM
            },
            "pose_estimation": {
                "cost": 1.5,
                "accuracy": 0.85,
                "best_contexts": [ContextType.INDOOR, ContextType.ISOLATED],
                "alert_threshold": AlertLevel.LOW
            },
            "trajectory_analyzer": {
                "cost": 2.0,
                "accuracy": 0.90,
                "best_contexts": [ContextType.CROWDED, ContextType.OUTDOOR],
                "alert_threshold": AlertLevel.MEDIUM
            },
            "multimodal_fusion": {
                "cost": 4.0,
                "accuracy": 0.98,
                "best_contexts": [ContextType.CRITICAL],
                "alert_threshold": AlertLevel.HIGH
            },
            "emotion_detector": {
                "cost": 2.5,
                "accuracy": 0.80,
                "best_contexts": [ContextType.INDOOR, ContextType.DAYLIGHT],
                "alert_threshold": AlertLevel.MEDIUM
            }
        }
        
        self.usage_history = {}
        self.performance_cache = {}
    
    def select_optimal_tools(self, 
                           alert_level: AlertLevel,
                           context: Set[ContextType],
                           time_budget: float = 5.0,
                           min_accuracy: float = 0.8) -> List[str]:
        """Sélectionne les outils optimaux selon les contraintes"""
        
        available_tools = []
        
        for tool_name, profile in self.tool_profiles.items():
            # Filtrage par niveau d'alerte
            if self._meets_alert_threshold(profile["alert_threshold"], alert_level):
                # Filtrage par contexte
                context_score = self._calculate_context_score(profile["best_contexts"], context)
                # Filtrage par précision minimum
                if profile["accuracy"] >= min_accuracy:
                    available_tools.append({
                        "name": tool_name,
                        "cost": profile["cost"],
                        "accuracy": profile["accuracy"],
                        "context_score": context_score,
                        "priority": profile["accuracy"] * context_score / profile["cost"]
                    })
        
        # Tri par priorité décroissante
        available_tools.sort(key=lambda x: x["priority"], reverse=True)
        
        # Sélection dans le budget temps
        selected_tools = []
        total_cost = 0
        
        for tool in available_tools:
            if total_cost + tool["cost"] <= time_budget:
                selected_tools.append(tool["name"])
                total_cost += tool["cost"]
        
        return selected_tools
    
    def _meets_alert_threshold(self, tool_threshold: AlertLevel, current_alert: AlertLevel) -> bool:
        """Vérifie si l'outil est approprié pour le niveau d'alerte"""
        threshold_levels = {
            AlertLevel.LOW: 0,
            AlertLevel.MEDIUM: 1, 
            AlertLevel.HIGH: 2,
            AlertLevel.CRITICAL: 3
        }
        return threshold_levels[current_alert] >= threshold_levels[tool_threshold]
    
    def _calculate_context_score(self, best_contexts: List[ContextType], current_context: Set[ContextType]) -> float:
        """Calcule un score de pertinence contextuelle"""
        if not best_contexts:
            return 1.0
        
        matches = len(set(best_contexts).intersection(current_context))
        return (matches / len(best_contexts)) + 0.5  # Score minimum de 0.5
    
    def learn_from_usage(self, tool_name: str, actual_performance: Dict):
        """Apprend des performances réelles pour améliorer les sélections futures"""
        if tool_name not in self.usage_history:
            self.usage_history[tool_name] = []
        
        self.usage_history[tool_name].append(actual_performance)
        
        # Mise à jour du profil basée sur l'historique
        if len(self.usage_history[tool_name]) > 10:
            recent_performances = self.usage_history[tool_name][-10:]
            avg_accuracy = np.mean([p.get("accuracy", 0.5) for p in recent_performances])
            avg_time = np.mean([p.get("execution_time", 0) for p in recent_performances])
            
            # Ajustement des profils
            self.tool_profiles[tool_name]["accuracy"] = 0.7 * self.tool_profiles[tool_name]["accuracy"] + 0.3 * avg_accuracy
            self.tool_profiles[tool_name]["cost"] = 0.7 * self.tool_profiles[tool_name]["cost"] + 0.3 * avg_time

# Étape 2: Testez la sélection adaptative
def test_adaptive_selection():
    selector = AdaptiveToolSelector()
    
    # Scénarios de test
    test_scenarios = [
        {
            "name": "Surveillance nocturne calme",
            "alert_level": AlertLevel.LOW,
            "context": {ContextType.NIGHT, ContextType.ISOLATED},
            "time_budget": 2.0
        },
        {
            "name": "Foule en journée - alerte moyenne",
            "alert_level": AlertLevel.MEDIUM,
            "context": {ContextType.DAYLIGHT, ContextType.CROWDED, ContextType.OUTDOOR},
            "time_budget": 4.0
        },
        {
            "name": "Incident critique",
            "alert_level": AlertLevel.CRITICAL,
            "context": {ContextType.INDOOR, ContextType.DAYLIGHT},
            "time_budget": 10.0
        }
    ]
    
    print("🧠 Test de sélection adaptative d'outils:\n")
    
    for scenario in test_scenarios:
        print(f"📋 Scénario: {scenario['name']}")
        print(f"   Niveau d'alerte: {scenario['alert_level'].value}")
        print(f"   Contexte: {[c.value for c in scenario['context']]}")
        print(f"   Budget temps: {scenario['time_budget']}s")
        
        selected_tools = selector.select_optimal_tools(
            alert_level=scenario['alert_level'],
            context=scenario['context'],
            time_budget=scenario['time_budget']
        )
        
        total_cost = sum([selector.tool_profiles[tool]["cost"] for tool in selected_tools])
        avg_accuracy = np.mean([selector.tool_profiles[tool]["accuracy"] for tool in selected_tools]) if selected_tools else 0
        
        print(f"   ✅ Outils sélectionnés: {selected_tools}")
        print(f"   ⏱️  Coût total: {total_cost:.1f}s")
        print(f"   🎯 Précision moyenne: {avg_accuracy:.2f}")
        print()

# Étape 3: Monitoring des performances en temps réel
class ToolPerformanceMonitor:
    """Monitore les performances des outils en temps réel"""
    
    def __init__(self):
        self.metrics = {}
        self.alert_thresholds = {
            "max_execution_time": 10.0,
            "min_accuracy": 0.7,
            "max_memory_usage": 1000  # MB
        }
    
    def record_execution(self, tool_name: str, metrics: Dict):
        """Enregistre les métriques d'exécution d'un outil"""
        if tool_name not in self.metrics:
            self.metrics[tool_name] = []
        
        self.metrics[tool_name].append(metrics)
        
        # Vérification des seuils d'alerte
        self._check_performance_alerts(tool_name, metrics)
    
    def _check_performance_alerts(self, tool_name: str, metrics: Dict):
        """Vérifie si des seuils d'alerte sont dépassés"""
        alerts = []
        
        if metrics.get("execution_time", 0) > self.alert_thresholds["max_execution_time"]:
            alerts.append(f"⚠️ {tool_name}: Temps d'exécution élevé ({metrics['execution_time']:.2f}s)")
        
        if metrics.get("accuracy", 1.0) < self.alert_thresholds["min_accuracy"]:
            alerts.append(f"⚠️ {tool_name}: Précision faible ({metrics['accuracy']:.2f})")
        
        if metrics.get("memory_usage", 0) > self.alert_thresholds["max_memory_usage"]:
            alerts.append(f"⚠️ {tool_name}: Usage mémoire élevé ({metrics['memory_usage']:.1f}MB)")
        
        for alert in alerts:
            print(alert)
    
    def get_tool_statistics(self, tool_name: str) -> Dict:
        """Récupère les statistiques d'un outil"""
        if tool_name not in self.metrics:
            return {}
        
        data = self.metrics[tool_name]
        
        return {
            "total_executions": len(data),
            "avg_execution_time": np.mean([d.get("execution_time", 0) for d in data]),
            "avg_accuracy": np.mean([d.get("accuracy", 0) for d in data]),
            "avg_memory_usage": np.mean([d.get("memory_usage", 0) for d in data]),
            "success_rate": np.mean([d.get("success", False) for d in data])
        }
```

#### **🎯 Objectifs d'Apprentissage - Module 5:**
- [ ] Comprendre les 8 outils avancés et leurs spécialités
- [ ] Créer votre propre outil personnalisé
- [ ] Maîtriser la sélection adaptative selon le contexte
- [ ] Optimiser les performances avec monitoring temps réel
- [ ] Intégrer de nouveaux outils dans l'architecture existante

---

## 🎯 **MODULE 6: Monitoring des Performances**

### **Fichiers Clés**
- `src/core/monitoring/performance_monitor.py` - Moniteur principal
- `src/core/monitoring/system_metrics.py` - Métriques système
- `src/core/monitoring/vlm_metrics.py` - Métriques VLM

### **Ce que vous devez comprendre**

#### **PerformanceMonitor**
```python
# Collection temps réel de métriques
monitor.add_collector(SystemMetricsCollector())
monitor.add_collector(VLMMetricsCollector()) 
monitor.start()  # Thread background
```

#### **Types de Métriques**
- **Système** : CPU, GPU, mémoire
- **VLM** : Temps traitement, débit, erreurs
- **Qualité** : Scores confiance, tool calls

### **Exercices pratiques - MODULE 6**

#### **🔬 Exercice 6.1: Surveillance complète des métriques**
```python
# Étape 1: Créez un dashboard de monitoring complet
# comprehensive_monitoring.py
import asyncio
import time
import psutil
import GPUtil
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import json

@dataclass
class SystemMetrics:
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_available_mb: float
    gpu_percent: float
    gpu_memory_percent: float
    disk_io_read: int
    disk_io_write: int

@dataclass
class VLMMetrics:
    timestamp: float
    model_name: str
    inference_time: float
    queue_size: int
    tokens_processed: int
    success_rate: float
    error_count: int

@dataclass
class DetectionMetrics:
    timestamp: float
    fps: float
    detections_count: int
    tracking_accuracy: float
    processing_latency: float
    frame_drops: int

class ComprehensiveMonitor:
    """Moniteur complet de toutes les métriques du système"""
    
    def __init__(self):
        self.running = False
        self.metrics_history = {
            'system': [],
            'vlm': [],
            'detection': []
        }
        self.alerts = []
        self.alert_thresholds = {
            'cpu_max': 80.0,
            'memory_max': 85.0,
            'gpu_max': 90.0,
            'inference_time_max': 5.0,
            'fps_min': 10.0
        }
    
    async def start_monitoring(self, interval: float = 1.0):
        """Démarre le monitoring continu"""
        self.running = True
        print("🔍 Démarrage du monitoring complet...")
        
        while self.running:
            # Collecte des métriques
            system_metrics = self._collect_system_metrics()
            self.metrics_history['system'].append(system_metrics)
            
            # Vérification des alertes
            self._check_system_alerts(system_metrics)
            
            # Nettoyage de l'historique (garde les 1000 derniers points)
            for key in self.metrics_history:
                if len(self.metrics_history[key]) > 1000:
                    self.metrics_history[key] = self.metrics_history[key][-1000:]
            
            await asyncio.sleep(interval)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collecte les métriques système"""
        # CPU et mémoire
        cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        
        # GPU (si disponible)
        gpu_percent = 0
        gpu_memory_percent = 0
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                gpu_percent = gpu.load * 100
                gpu_memory_percent = gpu.memoryUtil * 100
        except:
            pass
        
        # I/O disque
        disk_io = psutil.disk_io_counters()
        
        return SystemMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_available_mb=memory.available / (1024**2),
            gpu_percent=gpu_percent,
            gpu_memory_percent=gpu_memory_percent,
            disk_io_read=disk_io.read_bytes if disk_io else 0,
            disk_io_write=disk_io.write_bytes if disk_io else 0
        )
    
    def _check_system_alerts(self, metrics: SystemMetrics):
        """Vérifie les seuils d'alerte système"""
        alerts = []
        
        if metrics.cpu_percent > self.alert_thresholds['cpu_max']:
            alerts.append(f"🚨 CPU élevé: {metrics.cpu_percent:.1f}%")
        
        if metrics.memory_percent > self.alert_thresholds['memory_max']:
            alerts.append(f"🚨 Mémoire élevée: {metrics.memory_percent:.1f}%")
        
        if metrics.gpu_percent > self.alert_thresholds['gpu_max']:
            alerts.append(f"🚨 GPU élevé: {metrics.gpu_percent:.1f}%")
        
        for alert in alerts:
            print(alert)
            self.alerts.append({
                'timestamp': datetime.now().isoformat(),
                'message': alert,
                'type': 'system'
            })
    
    def record_vlm_metrics(self, metrics: VLMMetrics):
        """Enregistre les métriques VLM"""
        self.metrics_history['vlm'].append(metrics)
        
        # Vérification alertes VLM
        if metrics.inference_time > self.alert_thresholds['inference_time_max']:
            alert = f"🚨 VLM lent: {metrics.inference_time:.2f}s ({metrics.model_name})"
            print(alert)
            self.alerts.append({
                'timestamp': datetime.now().isoformat(),
                'message': alert,
                'type': 'vlm'
            })
    
    def record_detection_metrics(self, metrics: DetectionMetrics):
        """Enregistre les métriques de détection"""
        self.metrics_history['detection'].append(metrics)
        
        # Vérification alertes détection
        if metrics.fps < self.alert_thresholds['fps_min']:
            alert = f"🚨 FPS faible: {metrics.fps:.1f}"
            print(alert)
            self.alerts.append({
                'timestamp': datetime.now().isoformat(),
                'message': alert,
                'type': 'detection'
            })
    
    def get_current_status(self) -> Dict:
        """Retourne l'état actuel du système"""
        latest_system = self.metrics_history['system'][-1] if self.metrics_history['system'] else None
        latest_vlm = self.metrics_history['vlm'][-1] if self.metrics_history['vlm'] else None
        latest_detection = self.metrics_history['detection'][-1] if self.metrics_history['detection'] else None
        
        return {
            'system': asdict(latest_system) if latest_system else None,
            'vlm': asdict(latest_vlm) if latest_vlm else None,
            'detection': asdict(latest_detection) if latest_detection else None,
            'recent_alerts': self.alerts[-5:] if self.alerts else [],
            'total_alerts': len(self.alerts)
        }
    
    def export_metrics(self, filename: str = None):
        """Exporte toutes les métriques vers un fichier JSON"""
        if filename is None:
            filename = f"metrics_export_{int(time.time())}.json"
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'metrics_history': {
                'system': [asdict(m) for m in self.metrics_history['system']],
                'vlm': [asdict(m) for m in self.metrics_history['vlm']],
                'detection': [asdict(m) for m in self.metrics_history['detection']]
            },
            'alerts': self.alerts,
            'thresholds': self.alert_thresholds
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"📊 Métriques exportées vers: {filename}")
        return filename

# Étape 2: Intégration avec le système de surveillance
class SurveillanceWithMonitoring:
    """Système de surveillance avec monitoring intégré"""
    
    def __init__(self):
        self.monitor = ComprehensiveMonitor()
        self.frame_count = 0
        self.start_time = time.time()
    
    async def run_with_monitoring(self):
        """Lance la surveillance avec monitoring"""
        # Démarrage du monitoring en arrière-plan
        monitor_task = asyncio.create_task(self.monitor.start_monitoring())
        
        try:
            # Simulation du système de surveillance
            await self._simulate_surveillance()
        finally:
            # Arrêt du monitoring
            self.monitor.running = False
            await monitor_task
    
    async def _simulate_surveillance(self):
        """Simule le système de surveillance avec métriques"""
        print("🎥 Simulation surveillance avec monitoring...")
        
        for frame_id in range(100):  # 100 frames simulées
            frame_start = time.time()
            
            # Simulation traitement VLM
            vlm_start = time.time()
            await asyncio.sleep(0.1)  # Simule traitement VLM
            vlm_time = time.time() - vlm_start
            
            # Enregistrement métriques VLM
            vlm_metrics = VLMMetrics(
                timestamp=time.time(),
                model_name="claude-3-5-haiku",
                inference_time=vlm_time,
                queue_size=0,
                tokens_processed=150,
                success_rate=0.98,
                error_count=0
            )
            self.monitor.record_vlm_metrics(vlm_metrics)
            
            # Simulation détection
            detection_time = time.time() - frame_start
            current_fps = self.frame_count / (time.time() - self.start_time + 1)
            
            # Enregistrement métriques détection
            detection_metrics = DetectionMetrics(
                timestamp=time.time(),
                fps=current_fps,
                detections_count=3,
                tracking_accuracy=0.92,
                processing_latency=detection_time,
                frame_drops=0
            )
            self.monitor.record_detection_metrics(detection_metrics)
            
            self.frame_count += 1
            
            # Affichage périodique du status
            if frame_id % 20 == 0:
                status = self.monitor.get_current_status()
                print(f"\n📊 Frame {frame_id} - Status:")
                if status['system']:
                    print(f"   CPU: {status['system']['cpu_percent']:.1f}% | "
                          f"GPU: {status['system']['gpu_percent']:.1f}% | "
                          f"RAM: {status['system']['memory_percent']:.1f}%")
                if status['detection']:
                    print(f"   FPS: {status['detection']['fps']:.1f} | "
                          f"Latence: {status['detection']['processing_latency']:.3f}s")
                if status['recent_alerts']:
                    print(f"   🚨 Alertes récentes: {len(status['recent_alerts'])}")
            
            await asyncio.sleep(0.05)  # 20 FPS simulé
        
        # Export final des métriques
        export_file = self.monitor.export_metrics()
        return export_file

# Étape 3: Lancez le monitoring complet
async def run_comprehensive_monitoring():
    surveillance = SurveillanceWithMonitoring()
    export_file = await surveillance.run_with_monitoring()
    
    print(f"\n✅ Monitoring terminé. Données exportées: {export_file}")
    return export_file
```

#### **🔬 Exercice 6.2: Alertes et notifications personnalisées**
```python
# Étape 1: Système d'alertes avancé
# advanced_alerting.py
from enum import Enum
from dataclasses import dataclass
from typing import Callable, Dict, List, Any
import smtplib
from email.mime.text import MimeText
import requests
import logging

class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class Alert:
    id: str
    timestamp: float
    severity: AlertSeverity
    component: str
    message: str
    metrics: Dict[str, Any]
    resolved: bool = False

class AlertHandler:
    """Gestionnaire d'alertes avec notifications multiples"""
    
    def __init__(self):
        self.handlers = {}
        self.alert_history = []
        self.suppression_rules = {}
        
    def add_handler(self, name: str, handler: Callable):
        """Ajoute un gestionnaire d'alerte personnalisé"""
        self.handlers[name] = handler
        print(f"✅ Gestionnaire d'alerte '{name}' ajouté")
    
    async def trigger_alert(self, alert: Alert):
        """Déclenche une alerte et notifie tous les gestionnaires"""
        # Vérification des règles de suppression
        if self._is_suppressed(alert):
            return
        
        self.alert_history.append(alert)
        
        print(f"🚨 [{alert.severity.value.upper()}] {alert.component}: {alert.message}")
        
        # Notification via tous les gestionnaires configurés
        for name, handler in self.handlers.items():
            try:
                await handler(alert)
            except Exception as e:
                logging.error(f"Erreur gestionnaire {name}: {e}")
    
    def _is_suppressed(self, alert: Alert) -> bool:
        """Vérifie si l'alerte doit être supprimée"""
        rule_key = f"{alert.component}_{alert.severity.value}"
        if rule_key in self.suppression_rules:
            # Logique de suppression basée sur la fréquence
            recent_similar = [
                a for a in self.alert_history[-10:]
                if a.component == alert.component and a.severity == alert.severity
            ]
            return len(recent_similar) > self.suppression_rules[rule_key]['max_per_period']
        return False
    
    def set_suppression_rule(self, component: str, severity: AlertSeverity, max_per_period: int):
        """Configure une règle de suppression d'alertes"""
        rule_key = f"{component}_{severity.value}"
        self.suppression_rules[rule_key] = {'max_per_period': max_per_period}

# Étape 2: Gestionnaires de notification personnalisés
class EmailNotifier:
    """Notificateur email"""
    
    def __init__(self, smtp_server: str, smtp_port: int, username: str, password: str):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
    
    async def send_alert(self, alert: Alert):
        """Envoie une alerte par email"""
        subject = f"[{alert.severity.value.upper()}] Alerte Surveillance - {alert.component}"
        
        body = f"""
        Alerte de surveillance détectée:
        
        Composant: {alert.component}
        Sévérité: {alert.severity.value}
        Message: {alert.message}
        Timestamp: {datetime.fromtimestamp(alert.timestamp)}
        
        Métriques:
        {json.dumps(alert.metrics, indent=2)}
        """
        
        msg = MimeText(body)
        msg['Subject'] = subject
        msg['From'] = self.username
        msg['To'] = "admin@surveillance.com"  # Configurez votre destinataire
        
        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)
                print(f"📧 Email envoyé pour alerte {alert.id}")
        except Exception as e:
            print(f"❌ Erreur envoi email: {e}")

class SlackNotifier:
    """Notificateur Slack"""
    
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
    
    async def send_alert(self, alert: Alert):
        """Envoie une alerte vers Slack"""
        color_map = {
            AlertSeverity.INFO: "#36a64f",
            AlertSeverity.WARNING: "#ff9900",
            AlertSeverity.ERROR: "#ff3300",
            AlertSeverity.CRITICAL: "#990000"
        }
        
        payload = {
            "attachments": [
                {
                    "color": color_map[alert.severity],
                    "title": f"Alerte {alert.severity.value.upper()} - {alert.component}",
                    "text": alert.message,
                    "fields": [
                        {"title": "Timestamp", "value": datetime.fromtimestamp(alert.timestamp).strftime("%Y-%m-%d %H:%M:%S"), "short": True},
                        {"title": "Composant", "value": alert.component, "short": True}
                    ]
                }
            ]
        }
        
        try:
            response = requests.post(self.webhook_url, json=payload)
            if response.status_code == 200:
                print(f"💬 Notification Slack envoyée pour alerte {alert.id}")
            else:
                print(f"❌ Erreur Slack: {response.status_code}")
        except Exception as e:
            print(f"❌ Erreur notification Slack: {e}")

class FileLogger:
    """Enregistreur de fichier pour les alertes"""
    
    def __init__(self, filename: str = "alerts.log"):
        self.filename = filename
    
    async def send_alert(self, alert: Alert):
        """Enregistre l'alerte dans un fichier"""
        log_entry = f"{datetime.fromtimestamp(alert.timestamp)} | {alert.severity.value.upper()} | {alert.component} | {alert.message}\n"
        
        try:
            with open(self.filename, "a") as f:
                f.write(log_entry)
            print(f"📝 Alerte enregistrée dans {self.filename}")
        except Exception as e:
            print(f"❌ Erreur enregistrement fichier: {e}")

# Étape 3: Configuration et test du système d'alertes
async def setup_alerting_system():
    """Configure le système d'alertes complet"""
    alert_handler = AlertHandler()
    
    # Configuration des gestionnaires
    # Email (configurez avec vos vrais paramètres)
    # email_notifier = EmailNotifier("smtp.gmail.com", 587, "your-email", "your-password")
    # alert_handler.add_handler("email", email_notifier.send_alert)
    
    # Slack (configurez avec votre webhook)
    # slack_notifier = SlackNotifier("https://hooks.slack.com/your-webhook")
    # alert_handler.add_handler("slack", slack_notifier.send_alert)
    
    # Fichier log (toujours actif)
    file_logger = FileLogger("surveillance_alerts.log")
    alert_handler.add_handler("file", file_logger.send_alert)
    
    # Configuration des règles de suppression
    alert_handler.set_suppression_rule("VLM", AlertSeverity.WARNING, 3)
    alert_handler.set_suppression_rule("System", AlertSeverity.INFO, 5)
    
    return alert_handler

# Étape 4: Test avec alertes simulées
async def test_alerting_system():
    alert_handler = await setup_alerting_system()
    
    # Test différents types d'alertes
    test_alerts = [
        Alert(
            id="test_001",
            timestamp=time.time(),
            severity=AlertSeverity.WARNING,
            component="VLM",
            message="Temps d'inférence élevé détecté",
            metrics={"inference_time": 4.5, "model": "claude-3-5-haiku"}
        ),
        Alert(
            id="test_002",
            timestamp=time.time(),
            severity=AlertSeverity.ERROR,
            component="System",
            message="Utilisation mémoire critique",
            metrics={"memory_percent": 95.2, "available_mb": 512}
        ),
        Alert(
            id="test_003",
            timestamp=time.time(),
            severity=AlertSeverity.CRITICAL,
            component="Detection",
            message="Système de détection arrêté",
            metrics={"fps": 0, "error": "Camera disconnected"}
        )
    ]
    
    print("🧪 Test du système d'alertes...")
    
    for alert in test_alerts:
        await alert_handler.trigger_alert(alert)
        await asyncio.sleep(1)  # Délai entre alertes
    
    print(f"\n📊 {len(alert_handler.alert_history)} alertes traitées")
    return alert_handler
```

#### **🔬 Exercice 6.3: Analyse et optimisation des performances**
```python
# Étape 1: Analyseur de performance automatisé
# performance_analyzer.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple
import json

class PerformanceAnalyzer:
    """Analyseur automatique des performances du système"""
    
    def __init__(self, metrics_file: str):
        self.metrics_file = metrics_file
        self.data = self._load_metrics()
        self.analysis_results = {}
    
    def _load_metrics(self) -> Dict:
        """Charge les métriques depuis le fichier d'export"""
        try:
            with open(self.metrics_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Erreur chargement métriques: {e}")
            return {}
    
    def analyze_system_performance(self) -> Dict:
        """Analyse les performances système"""
        if not self.data.get('metrics_history', {}).get('system'):
            return {"error": "Pas de données système"}
        
        system_data = pd.DataFrame(self.data['metrics_history']['system'])
        
        analysis = {
            'cpu_stats': {
                'mean': system_data['cpu_percent'].mean(),
                'max': system_data['cpu_percent'].max(),
                'std': system_data['cpu_percent'].std(),
                'peaks': len(system_data[system_data['cpu_percent'] > 80])
            },
            'memory_stats': {
                'mean': system_data['memory_percent'].mean(),
                'max': system_data['memory_percent'].max(),
                'std': system_data['memory_percent'].std(),
                'peaks': len(system_data[system_data['memory_percent'] > 80])
            },
            'gpu_stats': {
                'mean': system_data['gpu_percent'].mean(),
                'max': system_data['gpu_percent'].max(),
                'utilization_efficiency': (system_data['gpu_percent'] > 10).sum() / len(system_data)
            },
            'recommendations': []
        }
        
        # Génération de recommandations
        if analysis['cpu_stats']['mean'] > 70:
            analysis['recommendations'].append("⚠️ CPU moyen élevé - Considérez l'optimisation du code")
        
        if analysis['memory_stats']['peaks'] > 10:
            analysis['recommendations'].append("⚠️ Pics mémoire fréquents - Vérifiez les fuites mémoire")
        
        if analysis['gpu_stats']['utilization_efficiency'] < 0.3:
            analysis['recommendations'].append("💡 GPU sous-utilisé - Optimisez l'utilisation GPU")
        
        self.analysis_results['system'] = analysis
        return analysis
    
    def analyze_vlm_performance(self) -> Dict:
        """Analyse les performances des modèles VLM"""
        if not self.data.get('metrics_history', {}).get('vlm'):
            return {"error": "Pas de données VLM"}
        
        vlm_data = pd.DataFrame(self.data['metrics_history']['vlm'])
        
        # Analyse par modèle
        model_analysis = {}
        for model in vlm_data['model_name'].unique():
            model_data = vlm_data[vlm_data['model_name'] == model]
            
            model_analysis[model] = {
                'avg_inference_time': model_data['inference_time'].mean(),
                'p95_inference_time': model_data['inference_time'].quantile(0.95),
                'success_rate': model_data['success_rate'].mean(),
                'throughput': len(model_data) / (model_data['timestamp'].max() - model_data['timestamp'].min())
            }
        
        # Recommandations
        recommendations = []
        best_model = min(model_analysis.items(), key=lambda x: x[1]['avg_inference_time'])
        worst_model = max(model_analysis.items(), key=lambda x: x[1]['avg_inference_time'])
        
        if worst_model[1]['avg_inference_time'] > best_model[1]['avg_inference_time'] * 2:
            recommendations.append(f"🚀 Considérez utiliser {best_model[0]} au lieu de {worst_model[0]}")
        
        analysis = {
            'models': model_analysis,
            'overall_avg_time': vlm_data['inference_time'].mean(),
            'overall_success_rate': vlm_data['success_rate'].mean(),
            'recommendations': recommendations
        }
        
        self.analysis_results['vlm'] = analysis
        return analysis
    
    def analyze_detection_performance(self) -> Dict:
        """Analyse les performances de détection"""
        if not self.data.get('metrics_history', {}).get('detection'):
            return {"error": "Pas de données de détection"}
        
        detection_data = pd.DataFrame(self.data['metrics_history']['detection'])
        
        analysis = {
            'fps_stats': {
                'mean': detection_data['fps'].mean(),
                'min': detection_data['fps'].min(),
                'stable_periods': self._find_stable_periods(detection_data['fps'])
            },
            'latency_stats': {
                'mean': detection_data['processing_latency'].mean(),
                'p95': detection_data['processing_latency'].quantile(0.95),
                'spikes': len(detection_data[detection_data['processing_latency'] > detection_data['processing_latency'].quantile(0.95)])
            },
            'tracking_stats': {
                'mean_accuracy': detection_data['tracking_accuracy'].mean(),
                'consistency': detection_data['tracking_accuracy'].std()
            },
            'recommendations': []
        }
        
        # Recommandations
        if analysis['fps_stats']['mean'] < 15:
            analysis['recommendations'].append("⚠️ FPS faible - Optimisez le pipeline de détection")
        
        if analysis['latency_stats']['spikes'] > 10:
            analysis['recommendations'].append("⚠️ Pics de latence fréquents - Vérifiez la charge système")
        
        self.analysis_results['detection'] = analysis
        return analysis
    
    def _find_stable_periods(self, fps_series: pd.Series, threshold: float = 2.0) -> int:
        """Trouve les périodes de FPS stable"""
        rolling_std = fps_series.rolling(window=10).std()
        return (rolling_std < threshold).sum()
    
    def generate_performance_report(self) -> str:
        """Génère un rapport complet de performance"""
        report = []
        report.append("# 📊 RAPPORT D'ANALYSE DE PERFORMANCE")
        report.append(f"Généré le: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Analyse système
        sys_analysis = self.analyze_system_performance()
        if 'error' not in sys_analysis:
            report.append("## 🖥️ Performances Système")
            report.append(f"CPU moyen: {sys_analysis['cpu_stats']['mean']:.1f}%")
            report.append(f"Mémoire moyenne: {sys_analysis['memory_stats']['mean']:.1f}%")
            report.append(f"GPU moyen: {sys_analysis['gpu_stats']['mean']:.1f}%")
            
            if sys_analysis['recommendations']:
                report.append("\n### Recommandations Système:")
                for rec in sys_analysis['recommendations']:
                    report.append(f"- {rec}")
            report.append("")
        
        # Analyse VLM
        vlm_analysis = self.analyze_vlm_performance()
        if 'error' not in vlm_analysis:
            report.append("## 🧠 Performances VLM")
            report.append(f"Temps inférence moyen: {vlm_analysis['overall_avg_time']:.2f}s")
            report.append(f"Taux de succès: {vlm_analysis['overall_success_rate']*100:.1f}%")
            
            if vlm_analysis['recommendations']:
                report.append("\n### Recommandations VLM:")
                for rec in vlm_analysis['recommendations']:
                    report.append(f"- {rec}")
            report.append("")
        
        # Analyse détection
        det_analysis = self.analyze_detection_performance()
        if 'error' not in det_analysis:
            report.append("## 👁️ Performances Détection")
            report.append(f"FPS moyen: {det_analysis['fps_stats']['mean']:.1f}")
            report.append(f"Latence moyenne: {det_analysis['latency_stats']['mean']:.3f}s")
            report.append(f"Précision tracking: {det_analysis['tracking_stats']['mean_accuracy']*100:.1f}%")
            
            if det_analysis['recommendations']:
                report.append("\n### Recommandations Détection:")
                for rec in det_analysis['recommendations']:
                    report.append(f"- {rec}")
        
        return "\n".join(report)
    
    def create_performance_dashboard(self) -> str:
        """Crée un dashboard visuel des performances"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Dashboard de Performance - Système de Surveillance', fontsize=16)
        
        # Graphique CPU/Mémoire
        if self.data.get('metrics_history', {}).get('system'):
            system_data = pd.DataFrame(self.data['metrics_history']['system'])
            axes[0, 0].plot(system_data['cpu_percent'], label='CPU %', color='red')
            axes[0, 0].plot(system_data['memory_percent'], label='Mémoire %', color='blue')
            axes[0, 0].set_title('Utilisation CPU/Mémoire')
            axes[0, 0].set_ylabel('Pourcentage')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
        
        # Graphique FPS
        if self.data.get('metrics_history', {}).get('detection'):
            detection_data = pd.DataFrame(self.data['metrics_history']['detection'])
            axes[0, 1].plot(detection_data['fps'], color='green')
            axes[0, 1].set_title('FPS de Détection')
            axes[0, 1].set_ylabel('FPS')
            axes[0, 1].grid(True)
        
        # Graphique temps VLM
        if self.data.get('metrics_history', {}).get('vlm'):
            vlm_data = pd.DataFrame(self.data['metrics_history']['vlm'])
            axes[1, 0].plot(vlm_data['inference_time'], color='purple')
            axes[1, 0].set_title('Temps Inférence VLM')
            axes[1, 0].set_ylabel('Secondes')
            axes[1, 0].grid(True)
        
        # Graphique précision tracking
        if self.data.get('metrics_history', {}).get('detection'):
            axes[1, 1].plot(detection_data['tracking_accuracy'], color='orange')
            axes[1, 1].set_title('Précision Tracking')
            axes[1, 1].set_ylabel('Précision')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        dashboard_file = f"performance_dashboard_{int(time.time())}.png"
        plt.savefig(dashboard_file)
        plt.show()
        
        return dashboard_file

# Étape 4: Utilisation complète de l'analyseur
async def run_performance_analysis():
    """Lance une analyse complète des performances"""
    # D'abord, générer des métriques
    print("🔄 Génération des métriques...")
    surveillance = SurveillanceWithMonitoring()
    metrics_file = await surveillance.run_with_monitoring()
    
    # Ensuite, analyser
    print("📊 Analyse des performances...")
    analyzer = PerformanceAnalyzer(metrics_file)
    
    # Génération du rapport
    report = analyzer.generate_performance_report()
    
    # Sauvegarde du rapport
    report_file = f"performance_report_{int(time.time())}.md"
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"📄 Rapport généré: {report_file}")
    print("\n" + report)
    
    # Création du dashboard
    dashboard_file = analyzer.create_performance_dashboard()
    print(f"📈 Dashboard créé: {dashboard_file}")
    
    return report_file, dashboard_file
```

#### **🎯 Objectifs d'Apprentissage - Module 6:**
- [ ] Maîtriser la collecte de métriques système, VLM et détection
- [ ] Créer des systèmes d'alertes personnalisées multi-canaux
- [ ] Analyser automatiquement les performances et identifier les goulots
- [ ] Générer des rapports et dashboards de monitoring
- [ ] Optimiser les performances basées sur l'analyse des métriques

---

## 🎯 **ÉTAPE FINALE: Intégration Complète**

### **Projet Pratique - Mission Finale**

#### **🎯 Mission : Système de surveillance magasin complet**
```python
# master_surveillance_project.py
# MISSION FINALE - Intégration de tous les modules appris

import asyncio
import time
from pathlib import Path
from config.config_loader import load_config
from src.core.headless.surveillance_system import HeadlessSurveillanceSystem
from src.core.vlm.dynamic_model import DynamicVisionLanguageModel
from src.detection.yolo_detector import YOLODetector
from src.advanced_tools.sam2_segmentation import SAM2Segmentator
from src.core.monitoring.performance_monitor import ComprehensiveMonitor

class RetailSurveillanceSystem:
    """Système de surveillance retail complet - Projet final"""
    
    def __init__(self):
        self.config = load_config()
        self.performance_monitor = ComprehensiveMonitor()
        self.surveillance_zones = {
            'entrance': {'alert_threshold': 0.7, 'tools': ['pose', 'sam2']},
            'cashier': {'alert_threshold': 0.9, 'tools': ['pose', 'trajectory', 'emotion']},
            'aisles': {'alert_threshold': 0.5, 'tools': ['pose', 'tracking']},
            'stockroom': {'alert_threshold': 0.8, 'tools': ['sam2', 'trajectory']}
        }
        
    async def deploy_retail_surveillance(self):
        """Déploie le système de surveillance retail"""
        print("🏪 Déploiement du système de surveillance retail...")
        
        # 1. Configuration personnalisée
        await self._configure_for_retail()
        
        # 2. Initialisation des composants
        surveillance_system = await self._initialize_components()
        
        # 3. Démarrage monitoring
        monitor_task = asyncio.create_task(
            self.performance_monitor.start_monitoring()
        )
        
        try:
            # 4. Lancement surveillance par zones
            await self._run_multi_zone_surveillance(surveillance_system)
            
        finally:
            # 5. Export des résultats et rapport final
            await self._generate_final_report()
            self.performance_monitor.running = False
            await monitor_task
    
    async def _configure_for_retail(self):
        """Configuration spécialisée retail"""
        # Personnalisation des seuils pour environnement commercial
        self.config.detection.confidence_threshold = 0.6
        self.config.vlm.mode = "BALANCED"
        
        # Prompts spécialisés retail
        self.retail_prompts = {
            'theft_detection': """
            Analysez cette scène de magasin pour détecter des comportements suspects:
            - Vol à l'étalage potentiel
            - Manipulation suspecte de produits
            - Comportements d'évasion
            - Interactions anormales avec le personnel
            
            Répondez avec niveau de risque et actions recommandées.
            """,
            'customer_flow': """
            Analysez le flux de clients dans cette zone:
            - Densité de personnes
            - Patterns de mouvement
            - Zones de congestion
            - Efficacité de la circulation
            
            Suggérez des optimisations d'agencement.
            """
        }
        
        print("✅ Configuration retail appliquée")
    
    async def _initialize_components(self):
        """Initialise tous les composants du système"""
        # VLM avec modèles multiples pour robustesse
        vlm_primary = DynamicVisionLanguageModel("claude-3-5-haiku-20241022")
        vlm_fallback = DynamicVisionLanguageModel("gpt-4o-mini-2024-07-18")
        
        # Détecteurs spécialisés
        detector = YOLODetector()
        detector.confidence_threshold = 0.6
        
        # Outils avancés selon les zones
        advanced_tools = {
            'sam2': SAM2Segmentator(),
            # Ajoutez d'autres outils selon vos modules précédents
        }
        
        # Système principal
        surveillance_system = HeadlessSurveillanceSystem(
            config=self.config,
            vlm_model=vlm_primary,
            detector=detector,
            advanced_tools=advanced_tools
        )
        
        await surveillance_system.initialize()
        print("✅ Composants initialisés")
        
        return surveillance_system
    
    async def _run_multi_zone_surveillance(self, system):
        """Surveillance multi-zones avec logique adaptative"""
        print("🎥 Démarrage surveillance multi-zones...")
        
        zone_results = {}
        
        for zone_name, zone_config in self.surveillance_zones.items():
            print(f"\n📍 Zone: {zone_name.upper()}")
            
            # Configuration adaptative par zone
            system.configure_for_zone(
                alert_threshold=zone_config['alert_threshold'],
                active_tools=zone_config['tools']
            )
            
            # Simulation surveillance de zone (remplacez par vraie caméra)
            zone_video = f"data/retail_videos/{zone_name}_sample.mp4"
            if Path(zone_video).exists():
                results = await system.process_video(zone_video, max_frames=50)
                zone_results[zone_name] = results
                
                # Analyse des résultats de zone
                self._analyze_zone_results(zone_name, results)
            else:
                print(f"⚠️  Vidéo de zone non trouvée: {zone_video}")
        
        return zone_results
    
    def _analyze_zone_results(self, zone_name, results):
        """Analyse les résultats par zone avec recommandations"""
        if not results:
            return
        
        total_frames = len(results)
        alert_frames = len([r for r in results if r.alert_level in ['HIGH', 'CRITICAL']])
        avg_confidence = sum([r.confidence_score for r in results]) / total_frames
        
        print(f"   📊 Analyse zone {zone_name}:")
        print(f"   - Frames traitées: {total_frames}")
        print(f"   - Alertes détectées: {alert_frames}")
        print(f"   - Confiance moyenne: {avg_confidence:.2f}")
        
        # Recommandations spécifiques par zone
        if zone_name == 'entrance' and alert_frames > total_frames * 0.3:
            print("   💡 Recommandation: Augmentez la surveillance à l'entrée")
        
        elif zone_name == 'cashier' and avg_confidence < 0.7:
            print("   💡 Recommandation: Améliorez l'éclairage aux caisses")
        
        elif zone_name == 'aisles' and alert_frames < total_frames * 0.1:
            print("   💡 Recommandation: Les allées sont sécurisées")
    
    async def _generate_final_report(self):
        """Génère le rapport final complet"""
        print("\n📋 Génération du rapport final...")
        
        # Export des métriques
        metrics_file = self.performance_monitor.export_metrics("retail_metrics_final.json")
        
        # Rapport personnalisé retail
        report = self._create_retail_report()
        
        # Sauvegarde
        report_file = f"RETAIL_SURVEILLANCE_REPORT_{int(time.time())}.md"
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"✅ Rapport final généré: {report_file}")
        print(f"✅ Métriques exportées: {metrics_file}")
        
        return report_file, metrics_file
    
    def _create_retail_report(self):
        """Crée un rapport spécialisé retail"""
        report_sections = [
            "# 🏪 RAPPORT DE SURVEILLANCE RETAIL",
            f"Généré le: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## 📊 Résumé Exécutif",
            "Ce rapport présente l'analyse complète du système de surveillance retail.",
            "",
            "## 🎯 Zones Surveillées",
            "- **Entrée**: Détection de comportements suspects",
            "- **Caisses**: Prévention des vols et analyse des files",  
            "- **Allées**: Monitoring du comportement client",
            "- **Réserve**: Sécurité du stock",
            "",
            "## 📈 Métriques de Performance",
            f"- Système opérationnel: {self.performance_monitor.running}",
            f"- Alertes générées: {len(self.performance_monitor.alerts)}",
            "",
            "## 💡 Recommandations",
            "1. ✅ Le système fonctionne correctement",
            "2. 🔧 Ajustements suggérés par zone (voir détails)",
            "3. 📊 Monitoring continu recommandé",
            "",
            "## 🎯 Prochaines Étapes",
            "1. Déploiement en production",
            "2. Formation équipe sécurité", 
            "3. Optimisation continue basée sur les données",
            "",
            "---",
            "*Rapport généré automatiquement par le système de surveillance intelligent*"
        ]
        
        return "\n".join(report_sections)

# Script principal pour lancer le projet final
async def run_master_project():
    """Lance le projet de maîtrise complet"""
    print("🚀 PROJET FINAL - SYSTÈME DE SURVEILLANCE RETAIL")
    print("=" * 60)
    
    retail_system = RetailSurveillanceSystem()
    
    try:
        await retail_system.deploy_retail_surveillance()
        print("\n🎉 MISSION ACCOMPLIE!")
        print("Vous maîtrisez maintenant l'architecture complète du système de surveillance.")
        
    except Exception as e:
        print(f"❌ Erreur durant la mission: {e}")
        print("💡 Revoyez les modules précédents et réessayez.")

# Point d'entrée
if __name__ == "__main__":
    asyncio.run(run_master_project())
```

#### **🎯 Checklist de Validation de Maîtrise**

Avant de considérer le projet terminé, vérifiez que vous pouvez :

**Configuration (Module 1)**
- [ ] Modifier les paramètres CLI et ENV sans regarder la documentation
- [ ] Créer une nouvelle configuration pour un cas d'usage spécifique
- [ ] Débugger les problèmes de configuration

**Système Headless (Module 2)** 
- [ ] Expliquer le flow complet d'une frame de A à Z
- [ ] Modifier les seuils d'alerte et prédire l'impact
- [ ] Optimiser les performances en ajustant les paramètres

**Modèles VLM (Module 3)**
- [ ] Changer de modèle VLM et ajuster les prompts
- [ ] Créer des prompts personnalisés pour votre domaine
- [ ] Comprendre les trade-offs entre modes FAST/BALANCED/THOROUGH

**Détection et Tracking (Module 4)**
- [ ] Tuner YOLO pour optimiser détection/performance
- [ ] Ajuster les paramètres BYTETracker selon vos besoins
- [ ] Créer des visualisations personnalisées

**Outils Avancés (Module 5)**
- [ ] Activer/désactiver des outils selon le contexte
- [ ] Créer votre propre outil avancé personnalisé
- [ ] Comprendre l'impact performance de chaque outil

**Monitoring (Module 6)**
- [ ] Configurer un monitoring complet temps réel
- [ ] Créer des alertes personnalisées multi-canaux
- [ ] Analyser et optimiser les performances du système

#### **🏆 Niveaux de Maîtrise**

**🥉 Bronze - Utilisateur Avancé**
- Peut utiliser le système avec différentes configurations
- Comprend les paramètres principaux et leur impact
- Capable de diagnostiquer les problèmes courants

**🥈 Argent - Développeur Système** 
- Peut modifier et étendre les composants existants
- Crée des outils personnalisés pour ses besoins
- Optimise les performances selon ses contraintes

**🥇 Or - Architecte Expert**
- Maîtrise l'architecture complète et ses interactions
- Peut concevoir des systèmes dérivés pour d'autres domaines
- Contribue à l'amélioration de l'architecture générale

#### **🎓 Certification de Maîtrise**

Une fois tous les exercices complétés et le projet final réussi :

1. **Portfolio de Preuves** : Conservez tous vos scripts, rapports et dashboards
2. **Projet Personnel** : Adaptez le système à votre propre cas d'usage
3. **Documentation** : Créez votre propre guide pour votre configuration spécifique
4. **Partage** : Contribuez à l'amélioration du système avec vos optimisations

### **Maîtrise Validée Quand Vous Pouvez :**
1. ✅ Modifier n'importe quel paramètre et prédire l'impact
2. ✅ Ajouter un nouveau type de détection
3. ✅ Créer un outil personnalisé
4. ✅ Optimiser les performances pour votre use-case
5. ✅ Débugger et corriger n'importe quelle erreur
6. ✅ Expliquer chaque ligne de code du flow principal

---

## 📊 **Ordre d'Apprentissage Recommandé**

### **Semaine 1 : Fondations**
- Configuration et point d'entrée
- Flow principal du système headless
- Modèles de données

### **Semaine 2 : Cœur Technique** 
- Détection YOLO + Tracking
- Système VLM et orchestration
- Tests et validation

### **Semaine 3 : Avancé**
- Outils IA spécialisés
- Monitoring et performances
- Optimisations personnalisées

### **Semaine 4 : Maîtrise**
- Projet complet de A à Z
- Personnalisations avancées
- Débogage et troubleshooting

## 🎯 **Ressources d'Apprentissage**

### **Documentation à Lire**
- Docstrings de chaque classe principale
- Tests unitaires (exemples d'usage)
- `ARCHITECTURE_VALIDATION_REPORT.md`

### **Commandes Essentielles**
```bash
# Tests complets
pytest tests/ -v

# Tests spécifiques
pytest tests/test_vlm_extended.py -v
pytest tests/test_headless_system.py -v

# Lancement avec debug
python main_headless_refactored.py --debug --max-frames 10

# Monitoring performances
python -c "from src.core.monitoring import *; # Code monitoring"
```

---

**🎓 Objectif Final :** Vous devez pouvoir modifier, étendre et optimiser chaque composant du système selon vos besoins spécifiques de surveillance.