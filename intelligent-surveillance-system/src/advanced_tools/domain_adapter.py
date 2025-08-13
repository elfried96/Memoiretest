"""Domain adaptation for robust performance across different environments."""

import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
import time
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

class DomainType(Enum):
    """Different domain types for adaptation."""
    LIGHTING_CONDITIONS = "lighting"
    CAMERA_ANGLES = "camera_angles"
    CROWD_DENSITY = "crowd_density"
    ENVIRONMENT_TYPE = "environment"
    TIME_OF_DAY = "time_of_day"
    WEATHER_CONDITIONS = "weather"

@dataclass
class DomainCharacteristics:
    """Characteristics of a specific domain."""
    domain_id: str
    domain_type: DomainType
    features: np.ndarray
    metadata: Dict[str, Any]
    sample_count: int
    adaptation_score: float

@dataclass
class AdaptationResult:
    """Result from domain adaptation."""
    source_domain: str
    target_domain: str
    adaptation_success: bool
    confidence_improvement: float
    feature_alignment_score: float
    adapted_parameters: Dict[str, Any]
    processing_time: float

class FeatureAlignmentNetwork(nn.Module):
    """Neural network for feature alignment between domains."""
    
    def __init__(self, feature_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # Domain classifier (adversarial component)
        self.domain_classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Feature extractor (domain-invariant features)
        self.feature_extractor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, feature_dim)
        )
        
        # Gradient reversal layer
        self.gradient_reversal_layer = GradientReversalLayer()
    
    def forward(self, x, alpha=1.0):
        """Forward pass with optional gradient reversal."""
        # Extract domain-invariant features
        adapted_features = self.feature_extractor(x)
        
        # Domain classification with gradient reversal
        reversed_features = self.gradient_reversal_layer(adapted_features, alpha)
        domain_pred = self.domain_classifier(reversed_features)
        
        return adapted_features, domain_pred

class GradientReversalLayer(nn.Module):
    """Gradient reversal layer for adversarial training."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x, alpha=1.0):
        return GradientReversalFunction.apply(x, alpha)

class GradientReversalFunction(torch.autograd.Function):
    """Gradient reversal function."""
    
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None

class DomainAdapter:
    """Domain adaptation system for surveillance environments."""
    
    def __init__(self, device: str = "auto"):
        self.device = self._select_device(device)
        
        # Domain registry
        self.known_domains = {}  # domain_id -> DomainCharacteristics
        
        # Adaptation models
        self.feature_alignment_net = None
        self.statistical_adapters = {}
        
        # Domain detection
        self.domain_classifier = None
        
        # Adaptation strategies
        self.adaptation_strategies = {
            DomainType.LIGHTING_CONDITIONS: self._adapt_lighting,
            DomainType.CAMERA_ANGLES: self._adapt_camera_angle,
            DomainType.CROWD_DENSITY: self._adapt_crowd_density,
            DomainType.ENVIRONMENT_TYPE: self._adapt_environment,
            DomainType.TIME_OF_DAY: self._adapt_time_of_day,
            DomainType.WEATHER_CONDITIONS: self._adapt_weather
        }
        
        # Feature extractors
        self.feature_extractors = {
            'image_statistics': self._extract_image_statistics,
            'detection_patterns': self._extract_detection_patterns,
            'temporal_patterns': self._extract_temporal_patterns
        }
    
    def _select_device(self, device: str) -> torch.device:
        """Select optimal device."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    def register_domain(self, domain_id: str, domain_type: DomainType, 
                       sample_images: List[np.ndarray], metadata: Dict[str, Any] = None) -> None:
        """Register a new domain with sample data."""
        
        # Extract domain features
        domain_features = self._extract_domain_features(sample_images)
        
        # Create domain characteristics
        characteristics = DomainCharacteristics(
            domain_id=domain_id,
            domain_type=domain_type,
            features=domain_features,
            metadata=metadata or {},
            sample_count=len(sample_images),
            adaptation_score=1.0  # Initial score
        )
        
        self.known_domains[domain_id] = characteristics
        
        # Update domain classifier if needed
        self._update_domain_classifier()
        
        logger.info(f"Registered domain {domain_id} of type {domain_type.value} with {len(sample_images)} samples")
    
    def _extract_domain_features(self, images: List[np.ndarray]) -> np.ndarray:
        """Extract domain-specific features from image samples."""
        
        all_features = []
        
        for image in images:
            features = []
            
            # Image statistics features
            img_stats = self.feature_extractors['image_statistics'](image)
            features.extend(img_stats)
            
            # Add more feature types as needed
            all_features.append(features)
        
        # Aggregate features (mean across samples)
        domain_features = np.mean(all_features, axis=0)
        return domain_features
    
    def _extract_image_statistics(self, image: np.ndarray) -> List[float]:
        """Extract image statistics for domain characterization."""
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        features = []
        
        # Brightness statistics
        features.extend([
            float(np.mean(gray)),
            float(np.std(gray)),
            float(np.median(gray)),
            float(np.percentile(gray, 25)),
            float(np.percentile(gray, 75))
        ])
        
        # Contrast measures
        features.append(float(np.std(gray) / (np.mean(gray) + 1e-8)))  # Coefficient of variation
        
        # Edge statistics
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        features.append(float(edge_density))
        
        # Texture measures
        # Local Binary Pattern-like measure
        texture = self._compute_texture_measure(gray)
        features.append(texture)
        
        # Color distribution (if color image)
        if len(image.shape) == 3:
            for channel in range(3):
                channel_data = image[:, :, channel]
                features.extend([
                    float(np.mean(channel_data)),
                    float(np.std(channel_data))
                ])
        else:
            # Pad with zeros for grayscale
            features.extend([0.0] * 6)
        
        # Frequency domain features
        fft = np.fft.fft2(gray)
        fft_magnitude = np.abs(fft)
        
        # Low, mid, high frequency energy
        h, w = fft_magnitude.shape
        low_freq = np.sum(fft_magnitude[:h//4, :w//4])
        mid_freq = np.sum(fft_magnitude[h//4:3*h//4, w//4:3*w//4])
        high_freq = np.sum(fft_magnitude[3*h//4:, 3*w//4:])
        total_energy = low_freq + mid_freq + high_freq
        
        features.extend([
            float(low_freq / (total_energy + 1e-8)),
            float(mid_freq / (total_energy + 1e-8)),
            float(high_freq / (total_energy + 1e-8))
        ])
        
        return features
    
    def _compute_texture_measure(self, gray: np.ndarray) -> float:
        """Compute texture measure for domain characterization."""
        # Simple texture measure using local variance
        kernel = np.ones((5, 5), np.float32) / 25
        mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        sqr_mean = cv2.filter2D((gray.astype(np.float32))**2, -1, kernel)
        variance = sqr_mean - mean**2
        return float(np.mean(variance))
    
    def _extract_detection_patterns(self, detections: List[Dict[str, Any]]) -> List[float]:
        """Extract detection pattern features."""
        if not detections:
            return [0.0] * 10  # Return default features
        
        features = []
        
        # Detection count
        features.append(float(len(detections)))
        
        # Confidence statistics
        confidences = [d.get('confidence', 0.0) for d in detections]
        features.extend([
            float(np.mean(confidences)),
            float(np.std(confidences)),
            float(np.max(confidences)),
            float(np.min(confidences))
        ])
        
        # Class distribution
        classes = [d.get('class', 'unknown') for d in detections]
        common_classes = ['person', 'handbag', 'backpack', 'suitcase']
        
        for cls in common_classes:
            features.append(float(classes.count(cls)))
        
        return features
    
    def _extract_temporal_patterns(self, temporal_data: List[Dict[str, Any]]) -> List[float]:
        """Extract temporal pattern features."""
        if not temporal_data:
            return [0.0] * 5
        
        features = []
        
        # Activity level over time
        activity_levels = [d.get('activity_level', 0.0) for d in temporal_data]
        features.extend([
            float(np.mean(activity_levels)),
            float(np.std(activity_levels)),
            float(np.max(activity_levels))
        ])
        
        # Trend analysis
        if len(activity_levels) > 3:
            trend = np.polyfit(range(len(activity_levels)), activity_levels, 1)[0]
            features.append(float(trend))
        else:
            features.append(0.0)
        
        # Periodicity measure (simplified)
        if len(activity_levels) > 10:
            fft = np.fft.fft(activity_levels)
            dominant_freq = np.argmax(np.abs(fft[1:len(fft)//2])) + 1
            features.append(float(dominant_freq))
        else:
            features.append(0.0)
        
        return features
    
    def detect_domain(self, image: np.ndarray, detections: List[Dict[str, Any]] = None) -> Optional[str]:
        """Detect which domain the current input belongs to."""
        
        if not self.known_domains:
            return None
        
        # Extract features from current input
        current_features = self._extract_image_statistics(image)
        
        if detections:
            detection_features = self._extract_detection_patterns(detections)
            current_features.extend(detection_features)
        
        current_features = np.array(current_features)
        
        # Find closest domain
        best_domain = None
        min_distance = float('inf')
        
        for domain_id, characteristics in self.known_domains.items():
            # Ensure same feature dimension
            domain_features = characteristics.features
            min_len = min(len(current_features), len(domain_features))
            
            current_truncated = current_features[:min_len]
            domain_truncated = domain_features[:min_len]
            
            # Calculate distance
            distance = np.linalg.norm(current_truncated - domain_truncated)
            
            if distance < min_distance:
                min_distance = distance
                best_domain = domain_id
        
        return best_domain
    
    def adapt_to_domain(self, source_domain: str, target_domain: str, 
                       adaptation_data: Dict[str, Any] = None) -> AdaptationResult:
        """Adapt from source domain to target domain."""
        start_time = time.time()
        
        if source_domain not in self.known_domains or target_domain not in self.known_domains:
            return AdaptationResult(
                source_domain=source_domain,
                target_domain=target_domain,
                adaptation_success=False,
                confidence_improvement=0.0,
                feature_alignment_score=0.0,
                adapted_parameters={},
                processing_time=time.time() - start_time
            )
        
        source_char = self.known_domains[source_domain]
        target_char = self.known_domains[target_domain]
        
        # Choose adaptation strategy based on domain type
        adaptation_strategy = self.adaptation_strategies.get(
            target_char.domain_type, self._generic_adaptation
        )
        
        # Perform adaptation
        adaptation_result = adaptation_strategy(source_char, target_char, adaptation_data)
        
        processing_time = time.time() - start_time
        
        return AdaptationResult(
            source_domain=source_domain,
            target_domain=target_domain,
            adaptation_success=adaptation_result['success'],
            confidence_improvement=adaptation_result['confidence_improvement'],
            feature_alignment_score=adaptation_result['alignment_score'],
            adapted_parameters=adaptation_result['parameters'],
            processing_time=processing_time
        )
    
    def _adapt_lighting(self, source: DomainCharacteristics, target: DomainCharacteristics, 
                       data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Adapt to different lighting conditions."""
        
        # Extract brightness characteristics
        source_brightness = source.features[0] if len(source.features) > 0 else 128.0
        target_brightness = target.features[0] if len(target.features) > 0 else 128.0
        
        # Calculate brightness adjustment
        brightness_diff = target_brightness - source_brightness
        brightness_scale = target_brightness / (source_brightness + 1e-8)
        
        # Contrast adaptation
        source_contrast = source.features[5] if len(source.features) > 5 else 1.0
        target_contrast = target.features[5] if len(target.features) > 5 else 1.0
        contrast_scale = target_contrast / (source_contrast + 1e-8)
        
        # Adaptation parameters
        parameters = {
            'brightness_adjustment': float(brightness_diff),
            'brightness_scale': float(brightness_scale),
            'contrast_scale': float(contrast_scale),
            'gamma_correction': float(np.log(target_brightness + 1) / np.log(source_brightness + 1))
        }
        
        # Calculate alignment score
        alignment_score = 1.0 / (1.0 + abs(brightness_diff) / 100.0)
        
        return {
            'success': True,
            'confidence_improvement': 0.15 * alignment_score,
            'alignment_score': alignment_score,
            'parameters': parameters
        }
    
    def _adapt_camera_angle(self, source: DomainCharacteristics, target: DomainCharacteristics,
                          data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Adapt to different camera angles."""
        
        # Extract edge density (proxy for viewing angle)
        source_edges = source.features[6] if len(source.features) > 6 else 0.3
        target_edges = target.features[6] if len(target.features) > 6 else 0.3
        
        # Calculate perspective adjustments
        edge_ratio = target_edges / (source_edges + 1e-8)
        
        parameters = {
            'edge_density_ratio': float(edge_ratio),
            'perspective_adjustment': float(np.log(edge_ratio + 1e-8)),
            'detection_threshold_adjustment': float(0.05 * (1 - edge_ratio)) if edge_ratio < 1 else 0.0
        }
        
        alignment_score = 1.0 / (1.0 + abs(1 - edge_ratio))
        
        return {
            'success': True,
            'confidence_improvement': 0.10 * alignment_score,
            'alignment_score': alignment_score,
            'parameters': parameters
        }
    
    def _adapt_crowd_density(self, source: DomainCharacteristics, target: DomainCharacteristics,
                           data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Adapt to different crowd density levels."""
        
        # Use detection count as proxy for crowd density
        source_density = source.metadata.get('avg_detections', 5.0)
        target_density = target.metadata.get('avg_detections', 5.0)
        
        density_ratio = target_density / (source_density + 1e-8)
        
        parameters = {
            'density_ratio': float(density_ratio),
            'nms_threshold_adjustment': float(0.1 * (density_ratio - 1)) if density_ratio > 1 else 0.0,
            'confidence_threshold_adjustment': float(-0.05 * (density_ratio - 1)) if density_ratio > 1 else 0.0,
            'tracking_distance_threshold': float(50.0 / (density_ratio + 1e-8))
        }
        
        alignment_score = 1.0 / (1.0 + abs(np.log(density_ratio + 1e-8)))
        
        return {
            'success': True,
            'confidence_improvement': 0.12 * alignment_score,
            'alignment_score': alignment_score,
            'parameters': parameters
        }
    
    def _adapt_environment(self, source: DomainCharacteristics, target: DomainCharacteristics,
                         data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Adapt to different environment types."""
        
        # Extract texture measures
        source_texture = source.features[7] if len(source.features) > 7 else 50.0
        target_texture = target.features[7] if len(target.features) > 7 else 50.0
        
        texture_ratio = target_texture / (source_texture + 1e-8)
        
        parameters = {
            'texture_ratio': float(texture_ratio),
            'feature_sensitivity': float(1.0 / (texture_ratio + 1e-8)),
            'background_subtraction_threshold': float(20.0 * texture_ratio),
            'edge_detection_threshold': float(100.0 / (texture_ratio + 1e-8))
        }
        
        alignment_score = 1.0 / (1.0 + abs(1 - texture_ratio))
        
        return {
            'success': True,
            'confidence_improvement': 0.08 * alignment_score,
            'alignment_score': alignment_score,
            'parameters': parameters
        }
    
    def _adapt_time_of_day(self, source: DomainCharacteristics, target: DomainCharacteristics,
                         data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Adapt to different times of day."""
        
        # Similar to lighting adaptation but with temporal considerations
        return self._adapt_lighting(source, target, data)
    
    def _adapt_weather(self, source: DomainCharacteristics, target: DomainCharacteristics,
                      data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Adapt to different weather conditions."""
        
        # Extract frequency domain features (weather affects image clarity)
        source_high_freq = source.features[-1] if len(source.features) > 0 else 0.3
        target_high_freq = target.features[-1] if len(target.features) > 0 else 0.3
        
        clarity_ratio = target_high_freq / (source_high_freq + 1e-8)
        
        parameters = {
            'clarity_ratio': float(clarity_ratio),
            'denoising_strength': float(1.0 - clarity_ratio) if clarity_ratio < 1 else 0.0,
            'sharpening_strength': float(clarity_ratio - 1.0) if clarity_ratio > 1 else 0.0,
            'detection_confidence_boost': float(0.1 * (1 - clarity_ratio)) if clarity_ratio < 1 else 0.0
        }
        
        alignment_score = 1.0 / (1.0 + abs(1 - clarity_ratio))
        
        return {
            'success': True,
            'confidence_improvement': 0.20 * alignment_score,
            'alignment_score': alignment_score,
            'parameters': parameters
        }
    
    def _generic_adaptation(self, source: DomainCharacteristics, target: DomainCharacteristics,
                          data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generic adaptation strategy."""
        
        # Calculate feature distance
        min_len = min(len(source.features), len(target.features))
        source_features = source.features[:min_len]
        target_features = target.features[:min_len]
        
        feature_distance = np.linalg.norm(source_features - target_features)
        alignment_score = 1.0 / (1.0 + feature_distance / 100.0)
        
        parameters = {
            'feature_distance': float(feature_distance),
            'adaptation_weight': float(alignment_score),
            'threshold_adjustment': float(0.05 * (1 - alignment_score))
        }
        
        return {
            'success': alignment_score > 0.5,
            'confidence_improvement': 0.05 * alignment_score,
            'alignment_score': alignment_score,
            'parameters': parameters
        }
    
    def _update_domain_classifier(self):
        """Update domain classifier with new domains."""
        if len(self.known_domains) < 2:
            return  # Need at least 2 domains
        
        # Extract features and labels
        features = []
        labels = []
        
        for i, (domain_id, characteristics) in enumerate(self.known_domains.items()):
            features.append(characteristics.features)
            labels.append(i)
        
        features_array = np.array(features)
        
        # Simple classifier using PCA + distance
        if not hasattr(self, 'domain_pca'):
            self.domain_pca = PCA(n_components=min(10, features_array.shape[1]))
        
        self.domain_features_pca = self.domain_pca.fit_transform(features_array)
        self.domain_labels = list(self.known_domains.keys())
        
        logger.info("Updated domain classifier")
    
    def apply_adaptation(self, image: np.ndarray, adaptation_params: Dict[str, Any]) -> np.ndarray:
        """Apply adaptation parameters to image."""
        
        adapted_image = image.copy().astype(np.float32)
        
        # Brightness adjustment
        if 'brightness_adjustment' in adaptation_params:
            adapted_image += adaptation_params['brightness_adjustment']
        
        # Brightness scaling
        if 'brightness_scale' in adaptation_params:
            adapted_image *= adaptation_params['brightness_scale']
        
        # Contrast scaling
        if 'contrast_scale' in adaptation_params:
            mean = np.mean(adapted_image)
            adapted_image = mean + (adapted_image - mean) * adaptation_params['contrast_scale']
        
        # Gamma correction
        if 'gamma_correction' in adaptation_params:
            gamma = adaptation_params['gamma_correction']
            adapted_image = np.power(adapted_image / 255.0, gamma) * 255.0
        
        # Denoising
        if 'denoising_strength' in adaptation_params and adaptation_params['denoising_strength'] > 0:
            strength = adaptation_params['denoising_strength']
            adapted_image = cv2.GaussianBlur(adapted_image, (3, 3), strength)
        
        # Sharpening
        if 'sharpening_strength' in adaptation_params and adaptation_params['sharpening_strength'] > 0:
            strength = adaptation_params['sharpening_strength']
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]) * strength
            adapted_image = cv2.filter2D(adapted_image, -1, kernel)
        
        # Clip values
        adapted_image = np.clip(adapted_image, 0, 255)
        
        return adapted_image.astype(np.uint8)
    
    def get_domain_summary(self) -> Dict[str, Any]:
        """Get summary of all registered domains."""
        
        summary = {
            'total_domains': len(self.known_domains),
            'domain_types': {},
            'domains': {}
        }
        
        for domain_id, characteristics in self.known_domains.items():
            domain_type = characteristics.domain_type.value
            
            if domain_type not in summary['domain_types']:
                summary['domain_types'][domain_type] = 0
            summary['domain_types'][domain_type] += 1
            
            summary['domains'][domain_id] = {
                'type': domain_type,
                'sample_count': characteristics.sample_count,
                'adaptation_score': characteristics.adaptation_score,
                'metadata': characteristics.metadata
            }
        
        return summary