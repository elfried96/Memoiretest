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

logger = logging.getLogger(__name__)

@dataclass
class DinoFeatures:
    """DINO v2 extracted features."""
    features: np.ndarray
    patch_features: Optional[np.ndarray]
    attention_maps: Optional[np.ndarray]
    processing_time: float

class DinoV2FeatureExtractor:
    """DINO v2 feature extractor for robust visual representations."""
    
    def __init__(self, model_name: str = "dinov2_vitb14", device: str = "auto"):
        self.device = self._select_device(device)
        self.model_name = model_name
        self.model = None
        self.transform = None
        self._load_model()
    
    def _select_device(self, device: str) -> torch.device:
        """Select optimal device."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    def _load_model(self):
        """Load DINO v2 model."""
        try:
            # Try loading from torch hub
            self.model = torch.hub.load('facebookresearch/dinov2', self.model_name)
            self.model = self.model.to(self.device).eval()
            
            # Setup transforms
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            logger.info(f"DINO v2 model {self.model_name} loaded on {self.device}")
            
        except Exception as e:
            logger.warning(f"Could not load DINO v2: {e}, using fallback")
            self.model = None
    
    @torch.inference_mode()
    def extract_features(self, frame: np.ndarray, regions: Optional[List[Tuple[int, int, int, int]]] = None,
                        extract_attention: bool = False) -> DinoFeatures:
        """Extract DINO v2 features from frame or regions."""
        start_time = time.perf_counter()
        
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
            logger.error(f"DINO feature extraction failed: {e}")
            return self._fallback_features(frame, start_time)
    
    def _extract_global_features(self, frame: np.ndarray, start_time: float,
                               extract_attention: bool = False) -> DinoFeatures:
        """Extract features from entire frame."""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if len(frame.shape) == 3 else frame
        pil_image = Image.fromarray(frame_rgb)
        
        # Transform
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
            if extract_attention:
                # Get features with attention
                features = self.model.forward_features(input_tensor)
                cls_token = features['x_norm_clstoken']
                patch_tokens = features['x_norm_patchtokens']
                
                # Get attention maps
                attention_maps = self._get_attention_maps(input_tensor)
                
                return DinoFeatures(
                    features=cls_token.cpu().numpy().squeeze(),
                    patch_features=patch_tokens.cpu().numpy().squeeze(),
                    attention_maps=attention_maps,
                    processing_time=time.perf_counter() - start_time
                )
            else:
                # Just CLS token
                features = self.model(input_tensor)
                return DinoFeatures(
                    features=features.cpu().numpy().squeeze(),
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