"""DINO v2 Feature Extraction for robust visual representations."""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import logging
from dataclasses import dataclass
import time
from PIL import Image
import torchvision.transforms as transforms
import cv2  # Ajout de l'import manquant
from enum import Enum

logger = logging.getLogger(__name__)

# Un seul modèle : VITB14_REG (86M paramètres, 768 dimensions, avec registers)

@dataclass
class DinoFeatures:
    """DINO v2 extracted features."""
    features: np.ndarray
    patch_features: Optional[np.ndarray]
    attention_maps: Optional[np.ndarray]
    processing_time: float

class DinoV2FeatureExtractor:
    """DINO v2 feature extractor for robust visual representations."""
    
    def __init__(self, device: str = "auto", lazy_loading: bool = False):
        self.device = self._select_device(device)
        # Un seul modèle fixe : VITB14_REG
        self.model_name = "dinov2_vitb14_reg" 
        self.feature_dim = 768  # Dimension fixe pour VITB14
        self.model = None
        self.transform = None
        self.lazy_loading = lazy_loading
        self._model_loaded = False
        
        # Chargement immédiat ou lazy selon paramètre
        # LAZY = False : Modèle chargé MAINTENANT (dans __init__)  
        # LAZY = True  : Modèle chargé PLUS TARD (au premier usage)
        if not lazy_loading:
            self._load_model()  # Chargement immédiat
    
    def _select_device(self, device: str) -> torch.device:
        """Select optimal device."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    def _ensure_model_loaded(self):
        """Ensure model is loaded (lazy loading)."""
        if not self._model_loaded:
            self._load_model()
    
    def _load_model(self):
        """Load DINO v2 model selon documentation officielle."""
        if self._model_loaded:
            return
            
        try:
            logger.info(f"Loading DINOv2 model {self.model_name} (modèle physique)...")
            
            # Chargement du modèle physique (téléchargé localement)
            # torch.hub télécharge et sauvegarde automatiquement dans ~/.cache/torch/hub/
            self.model = torch.hub.load('facebookresearch/dinov2', self.model_name, trust_repo=True)
            self.model = self.model.to(self.device).eval()
            
            # Le modèle est maintenant stocké physiquement sur le disque
            # Localisation: ~/.cache/torch/hub/facebookresearch_dinov2_main/
            
            # Setup transforms selon les standards DINOv2
            # Note: DINOv2 utilise résolution 224x224 par défaut
            self.transform = transforms.Compose([
                transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            # Dimension fixe pour VITB14_REG
            self.feature_dim = 768
            self._model_loaded = True
            
            logger.info(f"DINOv2 model {self.model_name} loaded on {self.device} (feature_dim: {self.feature_dim})")
            
        except Exception as e:
            logger.error(f"Could not load DINOv2: {e}, using fallback")
            self.model = None
            self._model_loaded = False
    
    # Plus besoin de cette méthode - dimension fixe à 768 pour VITB14_REG
    
    @torch.inference_mode()
    def extract_features(self, frame: np.ndarray, regions: Optional[List[Tuple[int, int, int, int]]] = None,
                        extract_attention: bool = False) -> DinoFeatures:
        """Extract DINOv2 features from frame or regions."""
        start_time = time.perf_counter()
        
        # Ensure model is loaded
        self._ensure_model_loaded()
        
        if self.model is None:
            return self._fallback_features(frame, start_time)
        
        try:
            if regions is None:
                # Extract features from entire frame
                return self._extract_global_features(frame, start_time, extract_attention)
            else:
                # Extract features from specified regions
                return self._extract_regional_features(frame, regions, start_time, extract_attention)
                
        except Exception as e:
            logger.error(f"DINOv2 feature extraction failed: {e}")
            return self._fallback_features(frame, start_time)
    
    def _extract_global_features(self, frame: np.ndarray, start_time: float,
                               extract_attention: bool = False) -> DinoFeatures:
        """Extract features from entire frame."""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if len(frame.shape) == 3 else frame
        pil_image = Image.fromarray(frame_rgb)
        
        # Transform
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        # Extract features selon API DINOv2 officielle
        with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
            # API DINOv2: Direct forward pass pour CLS token
            cls_features = self.model(input_tensor)  # Shape: [1, feature_dim]
            
            if extract_attention:
                # Pour attention et patch tokens, utiliser les hooks internes
                patch_features, attention_maps = self._extract_advanced_features(input_tensor)
                
                return DinoFeatures(
                    features=cls_features.cpu().numpy().squeeze(),
                    patch_features=patch_features,
                    attention_maps=attention_maps,
                    processing_time=time.perf_counter() - start_time
                )
            else:
                # Mode standard: seulement CLS token
                return DinoFeatures(
                    features=cls_features.cpu().numpy().squeeze(),
                    patch_features=None,
                    attention_maps=None,
                    processing_time=time.perf_counter() - start_time
                )
    
    def _extract_regional_features(self, frame: np.ndarray, regions: List[Tuple[int, int, int, int]],
                                 start_time: float, extract_attention: bool = False) -> DinoFeatures:
        """Extract features from multiple regions."""
        regional_features = []
        
        for x1, y1, x2, y2 in regions:
            # Extract region
            roi = frame[y1:y2, x1:x2]
            
            if roi.size == 0:
                continue
                
            # Convert and transform
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB) if len(roi.shape) == 3 else roi
            pil_roi = Image.fromarray(roi_rgb)
            input_tensor = self.transform(pil_roi).unsqueeze(0).to(self.device)
            
            # Extract features
            with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
                features = self.model(input_tensor)
                regional_features.append(features.cpu().numpy().squeeze())
        
        # Stack features
        stacked_features = np.stack(regional_features) if regional_features else np.array([])
        
        return DinoFeatures(
            features=stacked_features,
            patch_features=None,
            attention_maps=None,
            processing_time=time.perf_counter() - start_time
        )
    
    def _extract_advanced_features(self, input_tensor: torch.Tensor) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Extract patch features and attention maps via hooks."""
        try:
            # Variables pour capturer les features intermédiaires
            patch_features_captured = []
            attention_maps_captured = []
            
            def patch_hook(module, input, output):
                """Hook pour capturer les patch embeddings."""
                if hasattr(output, 'shape') and len(output.shape) == 3:
                    # Format: [batch, num_patches, embed_dim]
                    patch_features_captured.append(output.detach())
            
            def attention_hook(module, input, output):
                """Hook pour capturer les attention maps.""" 
                if isinstance(output, tuple) and len(output) > 1:
                    attn_weights = output[1]  # Attention weights
                    if attn_weights is not None:
                        attention_maps_captured.append(attn_weights.detach())
            
            # Enregistrer les hooks sur les couches appropriées
            hooks = []
            
            # Hook sur les blocs transformer pour patch features
            for name, module in self.model.named_modules():
                if 'blocks' in name and hasattr(module, 'norm1'):
                    hooks.append(module.register_forward_hook(patch_hook))
                elif 'attn' in name and hasattr(module, 'qkv'):
                    hooks.append(module.register_forward_hook(attention_hook))
            
            # Forward pass pour déclencher les hooks
            _ = self.model(input_tensor)
            
            # Nettoyer les hooks
            for hook in hooks:
                hook.remove()
            
            # Traitement des patch features capturées
            patch_features = None
            if patch_features_captured:
                # Prendre la dernière couche
                last_patches = patch_features_captured[-1]
                patch_features = last_patches.cpu().numpy().squeeze()
            
            # Traitement des attention maps capturées
            attention_maps = None
            if attention_maps_captured:
                # Moyenner sur les têtes d'attention
                last_attention = attention_maps_captured[-1]
                attention_maps = last_attention.mean(dim=1).cpu().numpy().squeeze()
            
            return patch_features, attention_maps
            
        except Exception as e:
            logger.warning(f"Could not extract advanced features: {e}")
            return None, None
    
    def _get_attention_maps(self, input_tensor: torch.Tensor) -> np.ndarray:
        """Extract attention maps from DINO v2."""
        # This is a simplified version - actual implementation depends on DINO v2 internals
        try:
            # Register hook to capture attention
            attention_maps = []
            
            def hook(module, input, output):
                if hasattr(output, 'shape') and len(output.shape) == 4:
                    attention_maps.append(output.detach())
            
            # Register hooks on attention layers
            hooks = []
            for name, module in self.model.named_modules():
                if 'attn' in name and hasattr(module, 'register_forward_hook'):
                    hooks.append(module.register_forward_hook(hook))
            
            # Forward pass
            _ = self.model.forward_features(input_tensor)
            
            # Remove hooks
            for hook in hooks:
                hook.remove()
            
            if attention_maps:
                return attention_maps[-1].cpu().numpy()  # Return last attention map
            else:
                return np.array([])
                
        except Exception as e:
            logger.warning(f"Could not extract attention maps: {e}")
            return np.array([])
    
    def _fallback_features(self, frame: np.ndarray, start_time: float) -> DinoFeatures:
        """Fallback feature extraction using traditional CV."""
        # Simple HOG features as fallback
        from skimage.feature import hog
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        resized = cv2.resize(gray, (224, 224))
        
        features = hog(
            resized,
            orientations=9,
            pixels_per_cell=(16, 16),
            cells_per_block=(2, 2),
            feature_vector=True
        )
        
        return DinoFeatures(
            features=features,
            patch_features=None,
            attention_maps=None,
            processing_time=time.perf_counter() - start_time
        )
    
    def compute_similarity(self, features1: np.ndarray, features2: np.ndarray,
                         metric: str = "cosine") -> float:
        """Compute similarity between feature vectors."""
        if metric == "cosine":
            # Cosine similarity
            dot_product = np.dot(features1, features2)
            norm1 = np.linalg.norm(features1)
            norm2 = np.linalg.norm(features2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
        
        elif metric == "euclidean":
            # Euclidean distance (inverted to similarity)
            distance = np.linalg.norm(features1 - features2)
            return 1.0 / (1.0 + distance)
        
        else:
            raise ValueError(f"Unknown similarity metric: {metric}")
    
    def cluster_features(self, features_list: List[np.ndarray], n_clusters: int = 3) -> Dict[str, Any]:
        """Cluster extracted features."""
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        
        if not features_list:
            return {"labels": [], "centers": [], "inertia": 0}
        
        # Stack and normalize features
        features_matrix = np.vstack(features_list)
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(features_matrix)
        
        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(features_normalized)
        
        return {
            "labels": labels.tolist(),
            "centers": kmeans.cluster_centers_.tolist(),
            "inertia": float(kmeans.inertia_)
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded DINOv2 model."""
        return {
            "model_name": self.model_name,
            "device": str(self.device),
            "feature_dimension": self.feature_dim,
            "model_loaded": self._model_loaded,
            "lazy_loading": self.lazy_loading,
            "model_available": self.model is not None,
            "parameters": "86M"  # Fixe pour VITB14_REG
        }