# Documentation des Outils Avancés - Système de Surveillance Intelligente

## Vue d'ensemble

Le système intègre 8 outils avancés d'analyse vidéo pour la surveillance intelligente. Chaque outil utilise des références académiques et des implémentations reconnues.

## 1. SAM2 Segmentator

### Référence
- **Source académique :** Meta's Segment Anything Model 2 (SAM2)
- **Modèle :** `facebook/sam2-hiera-large` via Transformers
- **Paper :** "Segment Anything in Images and Videos" (Meta AI, 2024)

### Implémentation
```python
# Localisation : src/advanced_tools/sam2_segmentation.py
from transformers import Sam2Model, Sam2Processor

# Optimisations GPU
if self.device.type == "cuda":
    self.model = self.model.half()  # FP16 pour économiser VRAM
    
# Utilisation autocast pour performance
with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
    outputs = self.model(**inputs)
```

### Compatibilité GPU
- ✅ **Excellent** - Support CUDA natif
- **VRAM requise :** ~2GB pour le modèle large
- **Optimisations :** FP16, autocast, batch processing
- **Fallback :** Segmentation Canny edge detection + OpenCV

### Utilisation
- Segmentation précise d'objets à partir de bounding boxes
- Extraction de propriétés de masques (aire, périmètre, compacité)
- Analyse de forme pour détection comportementale

---

## 2. DINO v2 Features

### Référence
- **Source académique :** Meta's DINOv2: Learning Robust Visual Features without Supervision
- **Modèle :** `facebookresearch/dinov2` via torch.hub, `dinov2_vitb14`
- **Paper :** "DINOv2: Learning Robust Visual Features without Supervision" (Oquab et al., 2023)

### Implémentation
```python
# Localisation : src/advanced_tools/dino_features.py
# Chargement depuis torch hub
self.model = torch.hub.load('facebookresearch/dinov2', self.model_name)

# Extraction avec attention maps
features = self.model.forward_features(input_tensor)
cls_token = features['x_norm_clstoken']
patch_tokens = features['x_norm_patchtokens']
```

### Compatibilité GPU
- ✅ **Excellent** - Optimisé CUDA
- **VRAM requise :** ~1GB pour ViT-B/14
- **Optimisations :** autocast, feature caching
- **Fallback :** HOG features avec scikit-image

### Utilisation
- Extraction de features visuelles robustes (768 dimensions)
- Calcul de similarité entre régions d'intérêt
- Clustering de features pour analyse comportementale
- Support attention maps pour interprétabilité

---

## 3. Pose Estimator

### Références
- **MediaPipe Pose :** Google's MediaPipe framework
- **MoveNet :** Google's lightweight pose detection
- **Papers :**
  - "MediaPipe: A Framework for Building Perception Pipelines" (Lugaresi et al., 2019)
  - "MoveNet: Ultra fast and accurate pose detection model" (Google, 2021)

### Implémentation
```python
# Localisation : src/advanced_tools/pose_estimation.py

# MediaPipe (33 keypoints)
import mediapipe as mp
self.mp_pose = mp.solutions.pose
self.pose_detector = self.mp_pose.Pose(
    model_complexity=1,
    min_detection_confidence=0.5
)

# MoveNet alternative (17 keypoints)
import tensorflow as tf
self.pose_detector = tf.keras.utils.get_file(
    'movenet_thunder.tflite',
    'https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/tflite/int8/4'
)
```

### Compatibilité GPU
- ⚠️ **MediaPipe :** CPU seulement, mais très rapide
- ✅ **MoveNet :** GPU compatible avec TensorFlow
- **Performance :** MediaPipe ~30ms CPU, MoveNet ~10ms GPU
- **Fallback :** Estimation heuristique basée sur bounding boxes

### Utilisation
- Détection de 33 points clés corporels (MediaPipe) ou 17 (MoveNet)
- Analyse comportementale (accroupissement, mains près de la taille)
- Détection de mouvements suspects et patterns anormaux
- Analyse de séquences temporelles de poses

---

## 4. Trajectory Analyzer

### Références
- **DBSCAN :** "A Density-Based Algorithm for Discovering Clusters" (Ester et al., 1996)
- **Behavioral patterns :** "Abnormal Behavior Detection in Videos" (surveillance literature)
- **Algorithmes :** `sklearn.cluster.DBSCAN`, `scipy.spatial.distance.cdist`

### Implémentation
```python
# Localisation : src/advanced_tools/trajectory_analyzer.py
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist

# Clustering des points d'arrêt
clustering = DBSCAN(eps=30, min_samples=3)
stop_clusters = clustering.fit_predict(stop_points)

# Analyse de patterns comportementaux
pattern_templates = {
    "browsing": {"velocity_variance": "high", "direction_changes": "many"},
    "purposeful_walking": {"velocity_variance": "low", "direction_changes": "few"},
    "suspicious_loitering": {"total_distance": "low", "time_spent": "high"}
}
```

### Compatibilité GPU
- ❌ **CPU seulement** - sklearn, scipy
- **Performance :** Très rapide sur CPU (~5ms par trajectoire)
- **Pas de fallback nécessaire**

### Utilisation
- Analyse de trajectoires individuelles avec historique
- Détection de patterns suspects (loitering, movement evasif)
- Classification comportementale automatique
- Détection d'anomalies dans les déplacements

---

## 5. Multimodal Fusion

### Références
- **Attention mechanism :** "Attention Is All You Need" (Vaswani et al., 2017)
- **Multimodal fusion :** "Multimodal Deep Learning for Robust RGB-D Object Recognition" (Eitel et al., 2015)
- **Architecture :** PyTorch MultiheadAttention

### Implémentation
```python
# Localisation : src/advanced_tools/multimodal_fusion.py

class AttentionFusion(nn.Module):
    def __init__(self, feature_dims: Dict[str, int], hidden_dim: int = 256):
        # Projection vers dimension commune
        self.projections = nn.ModuleDict()
        for modality, dim in feature_dims.items():
            self.projections[modality] = nn.Linear(dim, hidden_dim)
        
        # Mécanisme d'attention
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8)
```

### Compatibilité GPU
- ✅ **Excellent** - PyTorch natif
- **VRAM requise :** ~100MB
- **Fallback :** Statistical fusion avec moyennes pondérées

### Utilisation
- Fusion de 5 modalités : visual, detection, pose, motion, temporal
- Attention weights pour interprétabilité
- Feature extraction automatique de chaque modalité
- Prédiction finale avec scores de confiance

---

## 6. Temporal Transformer

### Références
- **Transformer :** "Attention Is All You Need" (Vaswani et al., 2017)
- **Temporal modeling :** "VideoBERT: A Joint Model for Video and Language" (Sun et al., 2019)
- **Architecture :** PyTorch TransformerEncoder standard

### Implémentation
```python
# Localisation : src/advanced_tools/temporal_transformer.py

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

class TemporalTransformer(nn.Module):
    def __init__(self, feature_dim: int = 512, num_heads: int = 8, num_layers: int = 6):
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
```

### Compatibilité GPU
- ✅ **Excellent** - PyTorch Transformer standard
- **VRAM requise :** ~200MB pour séquences de 50 frames
- **Optimisations :** Attention masking, gradient checkpointing

### Utilisation
- Analyse de cohérence temporelle sur séquences vidéo
- Détection d'anomalies temporelles
- Prédiction de patterns futurs
- Memory bank pour contexte historique long

---

## 7. Adversarial Detector

### Références
- **Isolation Forest :** "Isolation Forest" (Liu et al., 2008)
- **Adversarial detection :** "On the (Statistical) Detection of Adversarial Examples" (Grosse et al., 2017)
- **Elliptic Envelope :** Robust covariance estimation

### Implémentation
```python
# Localisation : src/advanced_tools/adversarial_detector.py
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope

class StatisticalDetector:
    def __init__(self):
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.elliptic_envelope = EllipticEnvelope(contamination=0.1, random_state=42)
        
    def _extract_statistical_features(self, image):
        # Analyse spectrale, gradients, statistiques de couleur
        features = [
            np.mean(image), np.std(image),  # Statistiques de base
            self._compute_edge_density(image),  # Densité des contours
            self._analyze_frequency_domain(image),  # Analyse fréquentielle
            self._compute_local_gradients(image)  # Gradients locaux
        ]
```

### Compatibilité GPU
- ⚠️ **Mixte** - sklearn (CPU) + PyTorch (GPU)
- **Performance :** Détection en ~20ms
- **Fallback :** Détection statistique simple

### Utilisation
- Détection d'attaques adverses sur images
- Analyse de perturbations artificielles
- Validation de robustesse du système
- Alertes de sécurité automatiques

---

## 8. Domain Adapter

### Références
- **Domain adaptation :** "Domain-Adversarial Training of Neural Networks" (Ganin & Lempitsky, 2015)
- **Feature alignment :** "Deep Transfer Learning: A Survey" (Tan et al., 2018)
- **Unsupervised DA :** "Return of Frustratingly Easy Domain Adaptation" (Sun et al., 2016)

### Implémentation
```python
# Localisation : src/advanced_tools/domain_adapter.py

class FeatureAlignmentNetwork(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 256):
        self.feature_extractor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Gradient Reversal Layer pour adversarial training
        self.domain_classifier = nn.Sequential(
            GradientReversalLayer(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

class DomainAdapter:
    def __init__(self):
        # Adaptation pour différents domaines
        self.domain_types = [
            DomainType.LIGHTING_CONDITIONS,
            DomainType.CAMERA_ANGLES,
            DomainType.CROWD_DENSITY,
            DomainType.ENVIRONMENT_TYPE,
            DomainType.TIME_OF_DAY,
            DomainType.WEATHER_CONDITIONS
        ]
```

### Compatibilité GPU
- ✅ **Excellent** - PyTorch avec composants sklearn
- **VRAM requise :** ~50MB pour réseau léger
- **Optimisations :** Batch adaptation, feature caching

### Utilisation
- Adaptation automatique aux conditions d'éclairage
- Calibration pour différents angles de caméra
- Adaptation à la densité de foule
- Robustesse aux conditions météorologiques

---

## Résumé de Compatibilité GPU

| Outil | GPU Compatible | VRAM Requise | Performance GPU | Fallback |
|-------|---------------|--------------|-----------------|----------|
| SAM2 Segmentator | ✅ Excellent | ~2GB | ~50ms | Canny + OpenCV |
| DINO v2 Features | ✅ Excellent | ~1GB | ~30ms | HOG features |
| Pose Estimator | ⚠️ MediaPipe CPU only | N/A | ~30ms CPU | Heuristique |
| Trajectory Analyzer | ❌ CPU only | N/A | ~5ms CPU | Aucun |
| Multimodal Fusion | ✅ Excellent | ~100MB | ~10ms | Statistical |
| Temporal Transformer | ✅ Excellent | ~200MB | ~15ms | Simple smoothing |
| Adversarial Detector | ⚠️ Mixte | ~50MB | ~20ms | Stats simples |
| Domain Adapter | ✅ Excellent | ~50MB | ~5ms | PCA + Scaling |

## Recommandations pour Tests GPU

### Configuration Recommandée
- **GPU :** NVIDIA RTX 3060 (12GB) minimum
- **VRAM totale nécessaire :** ~4GB pour tous les outils + VLM
- **CUDA :** Version 11.8 ou supérieure

### Stratégie de Test
1. **Test séquentiel** de chaque outil individuellement
2. **Monitoring VRAM** avec `nvidia-smi`
3. **Batch size réduit** pour éviter OOM
4. **Validation des fallbacks** en cas d'erreur GPU

### Points d'Attention
- **SAM2 + DINO v2 :** Peuvent saturer VRAM ensemble
- **MediaPipe :** CPU bottleneck possible
- **Temporal sequences :** Limiter à 30-50 frames max
- **Concurrent loading :** Charger les modèles séquentiellement

### Script de Test Recommandé
```python
# Test individual des outils avec monitoring GPU
for tool_name, tool_instance in tools.items():
    print(f"Testing {tool_name}...")
    gpu_memory_before = torch.cuda.memory_allocated()
    
    # Test de l'outil
    result = tool_instance.process(test_data)
    
    gpu_memory_after = torch.cuda.memory_allocated()
    memory_usage = (gpu_memory_after - gpu_memory_before) / (1024**3)  # GB
    
    print(f"{tool_name}: {memory_usage:.2f}GB VRAM, Success: {result.success}")
```