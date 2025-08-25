# 🔄 WORKFLOW COMPLET DU SYSTÈME

## 🎯 Vue d'Ensemble

Voici le parcours complet de votre système de surveillance intelligente, **du début à la fin** :

```
📹 VIDEO → 🔍 YOLO → 🎯 TRACKING → 🧠 KIMI-VL → ⚙️ ORCHESTRATION → 🚨 DÉCISIONS → 📊 ACTIONS
```

## 📖 Explication Détaillée du Workflow

### **🚀 ÉTAPE 1: Démarrage du Système**

Quand vous lancez `python main.py`, voici ce qui se passe :

1. **Initialisation des composants** :
   ```python
   # 1. Détecteur YOLO
   self.yolo_detector = YOLODetector(model_path="yolov11n.pt")
   
   # 2. Tracker multi-objets
   self.tracker = BYTETracker()
   
   # 3. VLM Kimi-VL (principal)
   self.vlm = DynamicVisionLanguageModel(default_model="kimi-vl-a3b-thinking")
   
   # 4. Orchestrateur des 8 outils
   self.orchestrator = ModernVLMOrchestrator(config=config)
   
   # 5. Moteur de décision
   self.decision_engine = SurveillanceDecisionEngine()
   ```

2. **Chargement des modèles** :
   - ✅ YOLO télécharge `yolov11n.pt` (6 MB)
   - ✅ Kimi-VL tente de se connecter à Moonshot AI
   - ✅ Si échec → Fallback automatique vers LLaVA
   - ✅ Initialisation des 8 outils avancés

---

### **📹 ÉTAPE 2: Capture Vidéo**

```python
cap = cv2.VideoCapture(video_source)  # 0 = webcam
ret, frame = cap.read()               # Lecture frame par frame
```

**Gestion des sources** :
- `video_source = 0` → Webcam par défaut
- `video_source = "video.mp4"` → Fichier vidéo
- `video_source = "rtsp://camera_ip"` → Flux IP

---

### **🔍 ÉTAPE 3: Détection YOLO**

Pour chaque frame capturé :

```python
yolo_results = self.yolo_detector.detect(frame)
detections = self.create_detections_list(yolo_results)
```

**Ce qui est détecté** :
- 👤 **Personnes** (classe 0) - **PRIORITÉ MAXIMALE**
- 🚗 Véhicules (bicycles, cars, motorcycles)
- 🎒 Objets (handbags, backpacks, suitcases)
- 📦 Autres objets pertinents

**Format de sortie** :
```python
Detection(
    bbox=BoundingBox(x1=100, y1=50, x2=200, y2=300),
    confidence=0.85,
    class_name="person",
    track_id=None  # Sera ajouté par le tracker
)
```

---

### **🎯 ÉTAPE 4: Tracking Multi-Objets**

```python
tracked_objects = self.tracker.update(detections)
```

**Fonctionnement** :
1. **Association** : Lie les nouvelles détections aux objets existants (IoU + distance)
2. **Identité persistante** : Assigne des IDs uniques (`track_1`, `track_2`...)
3. **Historique** : Maintient les trajectoires de mouvement
4. **Gestion des pertes** : Supprime les objets non vus depuis 30 frames

**Avantages** :
- 🎯 Suivi continu des personnes même si temporairement cachées
- 📈 Analyse des patterns de mouvement 
- 🚨 Détection de comportements suspects (loitering, trajectoires anormales)

---

### **🧠 ÉTAPE 5: Analyse VLM (Conditionnelle)**

L'analyse VLM n'est **PAS lancée à chaque frame** pour optimiser les performances :

```python
should_analyze = (
    len([d for d in detections if d.class_name == "person"]) > 0 or  # Personnes détectées
    self.frame_count % 30 == 0  # Ou analyse périodique (toutes les 30 frames)
)
```

**Si analyse nécessaire** :

1. **Encodage de l'image** :
   ```python
   frame_b64 = self.encode_frame_to_base64(frame)  # Conversion BGR→RGB→Base64
   ```

2. **Construction du contexte** :
   ```python
   context = {
       "frame_id": 12345,
       "timestamp": 1704886800.123,
       "location": "Store Main Area",
       "camera": "CAM_01",
       "person_count": 2,
       "total_objects": 5,
       "time_of_day": "14:30:15"
   }
   ```

3. **Appel de l'orchestrateur** :
   ```python
   vlm_analysis = await self.orchestrator.analyze_surveillance_frame(
       frame_data=frame_b64,
       detections=detections,
       context=context
   )
   ```

---

### **⚙️ ÉTAPE 6: Orchestration Intelligente**

L'orchestrateur sélectionne les outils selon le **mode configuré** :

#### **Mode FAST** (⚡ ~1.2s)
```python
tools_used = ["dino_features", "pose_estimator", "multimodal_fusion"]
```
- 🦕 **DinoV2** : Features visuelles robustes
- 🤸 **OpenPose** : Analyse posturale de base
- 🔗 **Fusion** : Combinaison des données

#### **Mode BALANCED** (⚖️ ~2.5s) - **RECOMMANDÉ**
```python
tools_used = [
    "sam2_segmentator", "dino_features", "pose_estimator",
    "trajectory_analyzer", "multimodal_fusion", "adversarial_detector"
]
```
- ✂️ **SAM2** : Segmentation précise personnes/objets
- 📈 **Trajectoires** : Analyse des patterns de mouvement
- 🛡️ **Adversarial** : Détection d'attaques/manipulations

#### **Mode THOROUGH** (🔬 ~4.8s)
```python
tools_used = [
    "sam2_segmentator", "dino_features", "pose_estimator", "trajectory_analyzer",
    "multimodal_fusion", "temporal_transformer", "adversarial_detector", "domain_adapter"
]
```
- ⏰ **Temporal** : Analyse séquentielle temporelle
- 🌍 **Domain Adapter** : Adaptation multi-environnements

---

### **🤖 ÉTAPE 7: Analyse Kimi-VL**

Le modèle **Kimi-VL-A3B-Thinking** analyse l'image avec le contexte :

1. **Prompt optimisé** :
   ```python
   prompt = f"""
   Analyse cette image de surveillance pour détecter des activités suspectes.
   
   Contexte : {context}
   Détections YOLO : {detections}
   Outils utilisés : {tools_results}
   
   Raisonne étape par étape pour cette analyse de surveillance.
   """
   ```

2. **Génération de la réponse** :
   ```python
   generated_text = self.model.generate(**inputs, max_new_tokens=768, temperature=0.8)
   ```

3. **Parsing intelligent** :
   ```python
   analysis_result = self.response_parser.parse_vlm_response(generated_text)
   # → Extracte niveau de suspicion, actions, confiance, recommandations
   ```

**Format de sortie** :
```python
AnalysisResponse(
    suspicion_level=SuspicionLevel.MEDIUM,
    action_type=ActionType.SUSPICIOUS_ACTIVITY,
    confidence=0.82,
    description="Personne loitering près des produits de valeur",
    tools_used=["sam2_segmentator", "pose_estimator", "trajectory_analyzer"],
    recommendations=["Surveillance renforcée", "Intervention discrète staff"]
)
```

---

### **🚨 ÉTAPE 8: Prise de Décisions**

Le moteur de décision analyse les résultats VLM et **prend des actions automatiques** :

```python
decisions = self.decision_engine.process_analysis(vlm_analysis, frame_context)
```

#### **Logique de décision** :

1. **Niveau de suspicion élevé** :
   ```python
   if analysis.suspicion_level.value in ["HIGH", "CRITICAL"]:
       decisions["alert_level"] = AlertLevel.ALERTE
       decisions["actions"] = ["start_recording", "alert_security"]
   ```

2. **Actions spécifiques par type** :
   ```python
   action_responses = {
       "SUSPICIOUS_ACTIVITY": ["increase_monitoring", "alert_security", "track_individual"],
       "THEFT_ATTEMPT": ["immediate_alert", "contact_security", "track_suspect"],
       "LOITERING": ["extended_observation", "gentle_staff_intervention"]
   }
   ```

3. **Contexte temporel** :
   ```python
   if current_time < "08:00" or current_time > "22:00":
       # Hors heures = plus strict
       decisions["alert_level"] = AlertLevel.ATTENTION
       decisions["actions"].append("after_hours_protocol")
   ```

---

### **📊 ÉTAPE 9: Affichage et Actions**

1. **Overlay temps réel** :
   ```python
   overlay_frame = self.draw_surveillance_overlay(frame, surveillance_frame)
   cv2.imshow("🎯 Surveillance Intelligente", overlay_frame)
   ```

   **Informations affichées** :
   - 🔍 Bounding boxes des détections (vert=personne, rouge=objet)
   - 🎯 IDs de tracking persistants
   - 🧠 Niveau d'alerte (NORMAL/ATTENTION/ALERTE/CRITIQUE)
   - 📊 Résultats VLM (suspicion + confiance)
   - ⚙️ Outils utilisés
   - 📈 Statistiques FPS, détections, alertes

2. **Actions automatisées** :
   ```python
   if decisions["recording_required"]:
       self.start_recording(frame)
   
   if "alert_security" in decisions["actions"]:
       self.send_security_alert(analysis, frame_location)
   
   if decisions["human_review"]:
       self.queue_for_human_review(frame, analysis)
   ```

---

### **🔄 ÉTAPE 10: Boucle Continue**

Le système **boucle indéfiniment** :

```python
while True:
    ret, frame = cap.read()                          # Capture frame
    surveillance_frame = await self.process_frame(frame)  # Pipeline complet
    
    if cv2.waitKey(1) & 0xFF == 27:  # ESC pour quitter
        break
```

**Optimisations** :
- ⚡ VLM seulement si personnes détectées ou période
- 🎯 Tracking maintient la continuité entre frames
- 🧠 Fallbacks automatiques si modèles indisponibles
- 📊 Statistiques en temps réel

---

## 🛠️ Configuration et Utilisation

### **Lancement Standard**
```bash
python main.py
```

### **Configuration Personnalisée**
```python
# Dans main.py, modifiez ces variables :
VIDEO_SOURCE = 0                           # 0=webcam, "video.mp4"=fichier
VLM_MODEL = "kimi-vl-a3b-thinking"        # Modèle principal
ORCHESTRATION_MODE = OrchestrationMode.BALANCED  # FAST/BALANCED/THOROUGH
```

### **Contrôles Temps Réel**
- **ESC** : Quitter le système
- **Fenêtre vidéo** : Voir surveillance en direct avec overlays
- **Terminal** : Logs détaillés du workflow

---

## 📈 Performance Attendue

| Mode | Vitesse | Outils | Usage |
|------|---------|--------|-------|
| **FAST** | ~1.2s/frame | 3 essentiels | Edge, temps réel |
| **BALANCED** | ~2.5s/frame | 6 principaux | **Production standard** |
| **THOROUGH** | ~4.8s/frame | 8 complets | Analyse forensique |

---

## 🎯 Résumé Exécutif

**Votre système est maintenant COMPLET et UNIFORME** avec :

✅ **Pipeline intégré** : Video → YOLO → Tracking → Kimi-VL → Orchestration → Décisions  
✅ **Multi-VLM** : Kimi-VL principal + fallbacks LLaVA/Qwen  
✅ **8 outils avancés** : Tous intégrés et orchestrés intelligemment  
✅ **Prise de décision** : Actions automatisées selon le contexte  
✅ **Interface temps réel** : Affichage surveillance avec overlays  
✅ **Architecture modulaire** : Code organisé et maintenable  

🚀 **Lancez `python main.py` pour voir votre système en action !**