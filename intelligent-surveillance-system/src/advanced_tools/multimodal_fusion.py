"""Multimodal fusion for combining different analysis modalities."""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging
import time

logger = logging.getLogger(__name__)

@dataclass
class FusionInput:
    """Input data for multimodal fusion."""
    visual_features: Optional[np.ndarray] = None
    detection_features: Optional[np.ndarray] = None
    pose_features: Optional[np.ndarray] = None
    motion_features: Optional[np.ndarray] = None
    temporal_features: Optional[np.ndarray] = None
    contextual_data: Optional[Dict[str, Any]] = None

@dataclass
class FusionResult:
    """Result from multimodal fusion."""
    fused_features: np.ndarray
    attention_weights: Dict[str, float]
    confidence_scores: Dict[str, float]
    final_prediction: float
    processing_time: float

class AttentionFusion(nn.Module):
    """Attention-based fusion mechanism."""
    
    def __init__(self, feature_dims: Dict[str, int], hidden_dim: int = 256):
        super().__init__()
        self.feature_dims = feature_dims
        self.hidden_dim = hidden_dim
        
        # Feature projection layers
        self.projections = nn.ModuleDict()
        for modality, dim in feature_dims.items():
            self.projections[modality] = nn.Linear(dim, hidden_dim)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        
        # Output layers
        self.fusion_layer = nn.Linear(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 1)
        
        self.dropout = nn.Dropout(0.1)
        self.activation = nn.ReLU()
        
    def forward(self, features_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Forward pass with attention weights."""
        
        # Project features to common dimension
        projected_features = []
        modality_names = []
        
        for modality, features in features_dict.items():
            if modality in self.projections and features is not None:
                proj_features = self.projections[modality](features)
                projected_features.append(proj_features.unsqueeze(1))  # Add sequence dimension
                modality_names.append(modality)
        
        if not projected_features:
            # Return zero tensor if no features
            batch_size = 1
            return torch.zeros(batch_size, self.hidden_dim), {}
        
        # Stack features
        stacked_features = torch.cat(projected_features, dim=1)  # [batch, seq_len, hidden]
        
        # Apply attention
        attended_features, attention_weights = self.attention(
            stacked_features, stacked_features, stacked_features
        )
        
        # Pool attended features
        pooled_features = torch.mean(attended_features, dim=1)  # [batch, hidden]
        
        # Final fusion
        fused = self.activation(self.fusion_layer(pooled_features))
        fused = self.dropout(fused)
        
        # Get prediction
        prediction = torch.sigmoid(self.classifier(fused))
        
        # Extract attention weights
        attention_dict = {}
        if len(modality_names) > 0:
            attention_weights_np = attention_weights.squeeze(0).mean(0).detach().cpu().numpy()
            for i, modality in enumerate(modality_names):
                if i < len(attention_weights_np):
                    attention_dict[modality] = float(attention_weights_np[i])
        
        return fused, prediction, attention_dict

class MultiModalFusion:
    """Advanced multimodal fusion system."""
    
    def __init__(self, device: str = "auto"):
        self.device = self._select_device(device)
        
        # Feature dimensions for different modalities
        self.feature_dims = {
            "visual": 768,      # DINO v2 features
            "detection": 256,   # Detection features
            "pose": 132,        # Pose keypoints (33 * 4)
            "motion": 64,       # Motion features
            "temporal": 128     # Temporal features
        }
        
        # Fusion models
        self.attention_fusion = None
        self.statistical_fusion = StatisticalFusion()
        
        # Feature extractors
        self.feature_extractors = {}
        
        self._initialize_fusion_model()
    
    def _select_device(self, device: str) -> torch.device:
        """Select optimal device."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    def _initialize_fusion_model(self):
        """Initialize the attention-based fusion model."""
        try:
            self.attention_fusion = AttentionFusion(
                feature_dims=self.feature_dims,
                hidden_dim=256
            ).to(self.device)
            
            logger.info("Attention fusion model initialized")
        except Exception as e:
            logger.error(f"Failed to initialize fusion model: {e}")
            self.attention_fusion = None
    
    def extract_detection_features(self, detections: List[Dict]) -> np.ndarray:
        """Extract features from detection results."""
        if not detections:
            return np.zeros(self.feature_dims["detection"])
        
        features = []
        
        for detection in detections:
            # Confidence score
            confidence = detection.get("confidence", 0.0)
            
            # Bounding box features
            bbox = detection.get("bbox", [0, 0, 0, 0])
            box_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            box_aspect_ratio = (bbox[2] - bbox[0]) / max((bbox[3] - bbox[1]), 1)
            
            # Class information (one-hot encoding for common classes)
            class_name = detection.get("class", "unknown")
            class_features = self._encode_class(class_name)
            
            # Spatial features
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            
            det_features = [
                confidence,
                box_area / 10000,  # Normalize area
                box_aspect_ratio,
                center_x / 1920,   # Normalize to [0,1] assuming 1920px width
                center_y / 1080,   # Normalize to [0,1] assuming 1080px height
            ]
            
            det_features.extend(class_features)
            features.append(det_features)
        
        # Aggregate features (mean, max, count)
        if features:
            features_array = np.array(features)
            aggregated = np.concatenate([
                np.mean(features_array, axis=0),
                np.max(features_array, axis=0),
                [len(features)]  # Detection count
            ])
        else:
            aggregated = np.zeros(self.feature_dims["detection"])
        
        # Pad or truncate to expected dimension
        if len(aggregated) < self.feature_dims["detection"]:
            aggregated = np.pad(aggregated, (0, self.feature_dims["detection"] - len(aggregated)))
        else:
            aggregated = aggregated[:self.feature_dims["detection"]]
        
        return aggregated
    
    def _encode_class(self, class_name: str) -> List[float]:
        """One-hot encode common surveillance classes."""
        classes = ["person", "handbag", "backpack", "suitcase", "bottle", "cell phone"]
        encoding = [0.0] * len(classes)
        
        if class_name in classes:
            encoding[classes.index(class_name)] = 1.0
        
        return encoding
    
    def extract_pose_features(self, pose_data: Dict) -> np.ndarray:
        """Extract features from pose estimation."""
        if not pose_data or "keypoints" not in pose_data:
            return np.zeros(self.feature_dims["pose"])
        
        keypoints = pose_data["keypoints"]
        if len(keypoints) == 0:
            return np.zeros(self.feature_dims["pose"])
        
        # Take first person's keypoints
        person_keypoints = keypoints[0] if isinstance(keypoints[0], list) else keypoints
        
        # Flatten keypoints (x, y, confidence for each keypoint)
        flattened = np.array(person_keypoints).flatten()
        
        # Pad or truncate to expected dimension
        if len(flattened) < self.feature_dims["pose"]:
            flattened = np.pad(flattened, (0, self.feature_dims["pose"] - len(flattened)))
        else:
            flattened = flattened[:self.feature_dims["pose"]]
        
        return flattened
    
    def extract_motion_features(self, motion_data: Dict) -> np.ndarray:
        """Extract features from motion analysis."""
        if not motion_data:
            return np.zeros(self.feature_dims["motion"])
        
        features = []
        
        # Movement speed
        speed = motion_data.get("average_velocity", 0.0)
        features.append(speed)
        
        # Direction changes
        direction_changes = motion_data.get("direction_changes", 0)
        features.append(direction_changes / 10.0)  # Normalize
        
        # Anomaly score
        anomaly_score = motion_data.get("anomaly_score", 0.0)
        features.append(anomaly_score)
        
        # Total distance
        total_distance = motion_data.get("total_distance", 0.0)
        features.append(total_distance / 1000.0)  # Normalize
        
        # Pattern classification (one-hot)
        pattern = motion_data.get("pattern", "unknown")
        pattern_features = self._encode_motion_pattern(pattern)
        features.extend(pattern_features)
        
        # Pad to expected dimension
        while len(features) < self.feature_dims["motion"]:
            features.append(0.0)
        
        return np.array(features[:self.feature_dims["motion"]])
    
    def _encode_motion_pattern(self, pattern: str) -> List[float]:
        """Encode motion pattern as one-hot vector."""
        patterns = [
            "browsing", "purposeful_walking", "suspicious_loitering", 
            "evasive_movement", "return_pattern", "normal_movement"
        ]
        
        encoding = [0.0] * len(patterns)
        if pattern in patterns:
            encoding[patterns.index(pattern)] = 1.0
        
        return encoding
    
    def extract_temporal_features(self, temporal_data: Optional[Dict]) -> np.ndarray:
        """Extract temporal consistency features."""
        if not temporal_data:
            return np.zeros(self.feature_dims["temporal"])
        
        features = []
        
        # Time-based features
        current_time = time.time()
        
        # Hour of day (normalized)
        hour_of_day = (current_time % (24 * 3600)) / (24 * 3600)
        features.append(hour_of_day)
        
        # Day of week (one-hot)
        day_of_week = int((current_time // (24 * 3600)) % 7)
        day_encoding = [0.0] * 7
        day_encoding[day_of_week] = 1.0
        features.extend(day_encoding)
        
        # Historical consistency scores
        consistency_score = temporal_data.get("consistency_score", 0.5)
        features.append(consistency_score)
        
        # Sequence length
        sequence_length = temporal_data.get("sequence_length", 0)
        features.append(sequence_length / 100.0)  # Normalize
        
        # Pad to expected dimension
        while len(features) < self.feature_dims["temporal"]:
            features.append(0.0)
        
        return np.array(features[:self.feature_dims["temporal"]])
    
    def fuse_features(self, fusion_input: FusionInput, fusion_method: str = "attention") -> FusionResult:
        """Main fusion function."""
        start_time = time.time()
        
        try:
            if fusion_method == "attention" and self.attention_fusion is not None:
                result = self._attention_based_fusion(fusion_input)
            else:
                result = self._statistical_fusion(fusion_input)
            
            result.processing_time = time.time() - start_time
            return result
            
        except Exception as e:
            logger.error(f"Fusion failed: {e}")
            return FusionResult(
                fused_features=np.zeros(256),
                attention_weights={},
                confidence_scores={},
                final_prediction=0.0,
                processing_time=time.time() - start_time
            )
    
    def _attention_based_fusion(self, fusion_input: FusionInput) -> FusionResult:
        """Attention-based fusion using neural network."""
        
        # Prepare tensors
        features_dict = {}
        confidence_scores = {}
        
        if fusion_input.visual_features is not None:
            # Handle dimension mismatch by padding or truncating
            visual_features = fusion_input.visual_features
            expected_dim = self.feature_dims["visual"]
            
            if len(visual_features) < expected_dim:
                # Pad with zeros if too small
                visual_features = np.pad(visual_features, (0, expected_dim - len(visual_features)))
            elif len(visual_features) > expected_dim:
                # Truncate if too large
                visual_features = visual_features[:expected_dim]
            
            features_dict["visual"] = torch.FloatTensor(visual_features).unsqueeze(0).to(self.device)
            confidence_scores["visual"] = np.mean(visual_features)
        
        if fusion_input.detection_features is not None:
            features_dict["detection"] = torch.FloatTensor(fusion_input.detection_features).unsqueeze(0).to(self.device)
            confidence_scores["detection"] = np.max(fusion_input.detection_features)
        
        if fusion_input.pose_features is not None:
            features_dict["pose"] = torch.FloatTensor(fusion_input.pose_features).unsqueeze(0).to(self.device)
            confidence_scores["pose"] = np.mean(fusion_input.pose_features[fusion_input.pose_features > 0])
        
        if fusion_input.motion_features is not None:
            features_dict["motion"] = torch.FloatTensor(fusion_input.motion_features).unsqueeze(0).to(self.device)
            confidence_scores["motion"] = 1.0 - fusion_input.motion_features[2]  # 1 - anomaly_score
        
        if fusion_input.temporal_features is not None:
            features_dict["temporal"] = torch.FloatTensor(fusion_input.temporal_features).unsqueeze(0).to(self.device)
            confidence_scores["temporal"] = fusion_input.temporal_features[8]  # consistency_score
        
        # Forward pass
        with torch.no_grad():
            fused_features, prediction, attention_weights = self.attention_fusion(features_dict)
        
        return FusionResult(
            fused_features=fused_features.cpu().numpy().squeeze(),
            attention_weights=attention_weights,
            confidence_scores=confidence_scores,
            final_prediction=float(prediction.cpu().numpy().squeeze()),
            processing_time=0.0  # Will be set by caller
        )
    
    def _statistical_fusion(self, fusion_input: FusionInput) -> FusionResult:
        """Statistical fusion fallback."""
        return self.statistical_fusion.fuse(fusion_input)

class StatisticalFusion:
    """Statistical-based fusion as fallback."""
    
    def __init__(self):
        # Weights for different modalities
        self.modality_weights = {
            "visual": 0.3,
            "detection": 0.25,
            "pose": 0.2,
            "motion": 0.15,
            "temporal": 0.1
        }
    
    def fuse(self, fusion_input: FusionInput) -> FusionResult:
        """Statistical fusion implementation."""
        
        features_list = []
        weights = []
        confidence_scores = {}
        
        if fusion_input.visual_features is not None:
            features_list.append(fusion_input.visual_features)
            weights.append(self.modality_weights["visual"])
            confidence_scores["visual"] = np.mean(np.abs(fusion_input.visual_features))
        
        if fusion_input.detection_features is not None:
            features_list.append(fusion_input.detection_features)
            weights.append(self.modality_weights["detection"])
            confidence_scores["detection"] = np.max(fusion_input.detection_features)
        
        if fusion_input.pose_features is not None:
            features_list.append(fusion_input.pose_features)
            weights.append(self.modality_weights["pose"])
            confidence_scores["pose"] = np.mean(fusion_input.pose_features[fusion_input.pose_features > 0])
        
        if fusion_input.motion_features is not None:
            features_list.append(fusion_input.motion_features)
            weights.append(self.modality_weights["motion"])
            confidence_scores["motion"] = 1.0 - fusion_input.motion_features[2]  # 1 - anomaly_score
        
        if fusion_input.temporal_features is not None:
            features_list.append(fusion_input.temporal_features)
            weights.append(self.modality_weights["temporal"])
            confidence_scores["temporal"] = fusion_input.temporal_features[8] if len(fusion_input.temporal_features) > 8 else 0.5
        
        if not features_list:
            return FusionResult(
                fused_features=np.zeros(256),
                attention_weights={},
                confidence_scores={},
                final_prediction=0.0,
                processing_time=0.0
            )
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        # Weighted average of features
        max_len = max(len(f) for f in features_list)
        
        # Pad features to same length
        padded_features = []
        for features in features_list:
            if len(features) < max_len:
                padded = np.pad(features, (0, max_len - len(features)))
            else:
                padded = features[:max_len]
            padded_features.append(padded)
        
        # Weighted fusion
        fused_features = np.zeros(max_len)
        for features, weight in zip(padded_features, weights):
            fused_features += features * weight
        
        # Calculate final prediction (simple weighted average of confidence scores)
        if confidence_scores:
            final_prediction = np.mean(list(confidence_scores.values()))
        else:
            final_prediction = 0.0
        
        # Create attention weights (uniform for statistical fusion)
        modality_names = []
        if fusion_input.visual_features is not None:
            modality_names.append("visual")
        if fusion_input.detection_features is not None:
            modality_names.append("detection")
        if fusion_input.pose_features is not None:
            modality_names.append("pose")
        if fusion_input.motion_features is not None:
            modality_names.append("motion")
        if fusion_input.temporal_features is not None:
            modality_names.append("temporal")
        
        attention_weights = {}
        for i, name in enumerate(modality_names):
            if i < len(weights):
                attention_weights[name] = float(weights[i])
        
        return FusionResult(
            fused_features=fused_features[:256],  # Limit to 256 dimensions
            attention_weights=attention_weights,
            confidence_scores=confidence_scores,
            final_prediction=final_prediction,
            processing_time=0.0
        )