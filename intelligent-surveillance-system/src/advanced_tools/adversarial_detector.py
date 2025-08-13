"""Adversarial attack detection for robust surveillance."""

import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging
import time
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope

logger = logging.getLogger(__name__)

@dataclass
class AdversarialDetection:
    """Result from adversarial detection."""
    is_adversarial: bool
    confidence: float
    attack_type: Optional[str]
    perturbation_magnitude: float
    detection_method: str
    processing_time: float
    evidence: Dict[str, Any]

class StatisticalDetector:
    """Statistical methods for adversarial detection."""
    
    def __init__(self):
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.elliptic_envelope = EllipticEnvelope(contamination=0.1, random_state=42)
        self.is_trained = False
        
        # Normal image statistics for reference
        self.normal_stats = {
            'mean_brightness': 128.0,
            'std_brightness': 50.0,
            'edge_density': 0.3,
            'color_distribution': None
        }
    
    def fit_normal_data(self, normal_images: List[np.ndarray]):
        """Fit detector on normal images."""
        features = []
        for image in normal_images:
            img_features = self._extract_statistical_features(image)
            features.append(img_features)
        
        features_array = np.array(features)
        
        # Train detectors
        self.isolation_forest.fit(features_array)
        self.elliptic_envelope.fit(features_array)
        
        # Update normal statistics
        self._update_normal_stats(normal_images)
        self.is_trained = True
        
        logger.info(f"Statistical detector trained on {len(normal_images)} normal images")
    
    def detect(self, image: np.ndarray) -> Dict[str, Any]:
        """Detect adversarial perturbations using statistical methods."""
        features = self._extract_statistical_features(image)
        
        results = {
            'is_adversarial': False,
            'confidence': 0.0,
            'methods': {}
        }
        
        if self.is_trained:
            # Isolation Forest
            iso_prediction = self.isolation_forest.predict([features])[0]
            iso_anomaly_score = -self.isolation_forest.score_samples([features])[0]
            
            # Elliptic Envelope
            ellip_prediction = self.elliptic_envelope.predict([features])[0]
            
            results['methods']['isolation_forest'] = {
                'prediction': int(iso_prediction),
                'anomaly_score': float(iso_anomaly_score),
                'is_anomaly': iso_prediction == -1
            }
            
            results['methods']['elliptic_envelope'] = {
                'prediction': int(ellip_prediction),
                'is_anomaly': ellip_prediction == -1
            }
            
            # Combined decision
            anomaly_votes = 0
            if iso_prediction == -1:
                anomaly_votes += 1
            if ellip_prediction == -1:
                anomaly_votes += 1
            
            results['is_adversarial'] = anomaly_votes >= 1
            results['confidence'] = anomaly_votes / 2.0
        
        # Additional statistical tests
        stat_tests = self._run_statistical_tests(image)
        results['statistical_tests'] = stat_tests
        
        # Update decision based on statistical tests
        if stat_tests['high_frequency_anomaly'] or stat_tests['brightness_anomaly']:
            results['is_adversarial'] = True
            results['confidence'] = max(results['confidence'], 0.7)
        
        return results
    
    def _extract_statistical_features(self, image: np.ndarray) -> np.ndarray:
        """Extract statistical features from image."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        features = []
        
        # Brightness statistics
        features.extend([
            np.mean(gray),
            np.std(gray),
            np.median(gray),
            np.percentile(gray, 25),
            np.percentile(gray, 75)
        ])
        
        # Edge statistics
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        features.append(edge_density)
        
        # Gradient statistics
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        features.extend([
            np.mean(gradient_magnitude),
            np.std(gradient_magnitude),
            np.max(gradient_magnitude)
        ])
        
        # Texture features (using Local Binary Pattern-like measure)
        texture_measure = self._compute_texture_measure(gray)
        features.append(texture_measure)
        
        # Frequency domain features
        fft = np.fft.fft2(gray)
        fft_magnitude = np.abs(fft)
        high_freq_energy = np.sum(fft_magnitude[gray.shape[0]//4:, gray.shape[1]//4:])
        total_energy = np.sum(fft_magnitude)
        high_freq_ratio = high_freq_energy / (total_energy + 1e-8)
        
        features.append(high_freq_ratio)
        
        return np.array(features)
    
    def _compute_texture_measure(self, gray: np.ndarray) -> float:
        """Compute simple texture measure."""
        # Simplified local variance measure
        kernel = np.ones((3, 3), np.float32) / 9
        mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        sqr_mean = cv2.filter2D((gray.astype(np.float32))**2, -1, kernel)
        variance = sqr_mean - mean**2
        return np.mean(variance)
    
    def _run_statistical_tests(self, image: np.ndarray) -> Dict[str, bool]:
        """Run additional statistical tests."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        tests = {}
        
        # High frequency content test
        fft = np.fft.fft2(gray)
        fft_magnitude = np.abs(fft)
        high_freq_energy = np.sum(fft_magnitude[gray.shape[0]//2:, gray.shape[1]//2:])
        total_energy = np.sum(fft_magnitude)
        high_freq_ratio = high_freq_energy / (total_energy + 1e-8)
        
        tests['high_frequency_anomaly'] = high_freq_ratio > 0.3  # Threshold for high freq content
        
        # Brightness anomaly test
        mean_brightness = np.mean(gray)
        brightness_deviation = abs(mean_brightness - self.normal_stats['mean_brightness'])
        tests['brightness_anomaly'] = brightness_deviation > self.normal_stats['std_brightness'] * 2
        
        # Edge density test
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        tests['edge_anomaly'] = abs(edge_density - self.normal_stats['edge_density']) > 0.2
        
        # Gradient magnitude test
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        mean_gradient = np.mean(gradient_magnitude)
        tests['gradient_anomaly'] = mean_gradient > 100  # Threshold for high gradient
        
        return tests
    
    def _update_normal_stats(self, normal_images: List[np.ndarray]):
        """Update statistics of normal images."""
        all_brightness = []
        all_edge_densities = []
        
        for image in normal_images:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            all_brightness.append(np.mean(gray))
            
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            all_edge_densities.append(edge_density)
        
        self.normal_stats['mean_brightness'] = np.mean(all_brightness)
        self.normal_stats['std_brightness'] = np.std(all_brightness)
        self.normal_stats['edge_density'] = np.mean(all_edge_densities)

class NeuralDetector(nn.Module):
    """Neural network-based adversarial detector."""
    
    def __init__(self, input_channels: int = 3):
        super().__init__()
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.classifier(features)
        return output, features

class AdversarialDetector:
    """Main adversarial attack detector."""
    
    def __init__(self, device: str = "auto"):
        self.device = self._select_device(device)
        
        # Detection methods
        self.statistical_detector = StatisticalDetector()
        self.neural_detector = None
        
        # Attack type signatures
        self.attack_signatures = {
            'fgsm': {'high_frequency': True, 'uniform_noise': True},
            'pgd': {'high_frequency': True, 'structured_noise': True},
            'c&w': {'low_frequency': True, 'optimized_noise': True},
            'deepfool': {'minimal_noise': True, 'boundary_noise': True}
        }
        
        self._initialize_neural_detector()
    
    def _select_device(self, device: str) -> torch.device:
        """Select optimal device."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    def _initialize_neural_detector(self):
        """Initialize neural detector."""
        try:
            self.neural_detector = NeuralDetector().to(self.device)
            self.neural_detector.eval()
            logger.info("Neural adversarial detector initialized")
        except Exception as e:
            logger.error(f"Failed to initialize neural detector: {e}")
            self.neural_detector = None
    
    def train_on_normal_data(self, normal_images: List[np.ndarray]):
        """Train detector on normal images."""
        self.statistical_detector.fit_normal_data(normal_images)
        logger.info("Adversarial detector trained on normal data")
    
    def detect_adversarial(self, image: np.ndarray) -> AdversarialDetection:
        """Detect adversarial attacks in image."""
        start_time = time.time()
        
        try:
            # Statistical detection
            stat_results = self.statistical_detector.detect(image)
            
            # Neural detection (if available)
            neural_results = {}
            if self.neural_detector is not None:
                neural_results = self._neural_detection(image)
            
            # Pattern-based detection
            pattern_results = self._pattern_based_detection(image)
            
            # Combine results
            combined_results = self._combine_detection_results(
                stat_results, neural_results, pattern_results
            )
            
            # Classify attack type
            attack_type = self._classify_attack_type(combined_results['evidence'])
            
            processing_time = time.time() - start_time
            
            return AdversarialDetection(
                is_adversarial=combined_results['is_adversarial'],
                confidence=combined_results['confidence'],
                attack_type=attack_type,
                perturbation_magnitude=combined_results['perturbation_magnitude'],
                detection_method=combined_results['primary_method'],
                processing_time=processing_time,
                evidence=combined_results['evidence']
            )
            
        except Exception as e:
            logger.error(f"Adversarial detection failed: {e}")
            return AdversarialDetection(
                is_adversarial=False,
                confidence=0.0,
                attack_type=None,
                perturbation_magnitude=0.0,
                detection_method="error",
                processing_time=time.time() - start_time,
                evidence={"error": str(e)}
            )
    
    def _neural_detection(self, image: np.ndarray) -> Dict[str, Any]:
        """Neural network-based detection."""
        # Prepare image tensor
        if len(image.shape) == 3:
            image_tensor = torch.FloatTensor(image.transpose(2, 0, 1)).unsqueeze(0) / 255.0
        else:
            image_tensor = torch.FloatTensor(image).unsqueeze(0).unsqueeze(0) / 255.0
        
        image_tensor = image_tensor.to(self.device)
        
        with torch.no_grad():
            prediction, features = self.neural_detector(image_tensor)
        
        prediction_score = float(prediction.cpu().numpy().squeeze())
        
        return {
            'is_adversarial': prediction_score > 0.5,
            'confidence': prediction_score if prediction_score > 0.5 else 1 - prediction_score,
            'neural_score': prediction_score,
            'features': features.cpu().numpy()
        }
    
    def _pattern_based_detection(self, image: np.ndarray) -> Dict[str, Any]:
        """Pattern-based detection using known attack signatures."""
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        patterns = {}
        
        # High frequency noise pattern (FGSM, PGD)
        fft = np.fft.fft2(gray)
        fft_magnitude = np.abs(fft)
        high_freq_energy = np.sum(fft_magnitude[gray.shape[0]//2:, gray.shape[1]//2:])
        total_energy = np.sum(fft_magnitude)
        high_freq_ratio = high_freq_energy / (total_energy + 1e-8)
        
        patterns['high_frequency_noise'] = high_freq_ratio > 0.25
        
        # Uniform noise pattern
        noise_estimate = gray - cv2.GaussianBlur(gray, (5, 5), 0)
        noise_uniformity = np.std(noise_estimate) / (np.mean(np.abs(noise_estimate)) + 1e-8)
        patterns['uniform_noise'] = noise_uniformity < 2.0  # More uniform than natural noise
        
        # Structured noise pattern (adversarial perturbations often have structure)
        edges = cv2.Canny(gray, 50, 150)
        noise_edges = cv2.Canny(np.abs(noise_estimate).astype(np.uint8), 20, 60)
        edge_correlation = np.corrcoef(edges.flatten(), noise_edges.flatten())[0, 1]
        patterns['structured_noise'] = not np.isnan(edge_correlation) and edge_correlation > 0.3
        
        # Minimal perturbation pattern (C&W, DeepFool)
        perturbation_magnitude = np.mean(np.abs(noise_estimate))
        patterns['minimal_noise'] = perturbation_magnitude < 5.0
        
        return {
            'patterns': patterns,
            'high_freq_ratio': high_freq_ratio,
            'noise_uniformity': noise_uniformity,
            'edge_correlation': edge_correlation if not np.isnan(edge_correlation) else 0.0,
            'perturbation_magnitude': perturbation_magnitude
        }
    
    def _combine_detection_results(self, stat_results: Dict, neural_results: Dict, 
                                 pattern_results: Dict) -> Dict[str, Any]:
        """Combine results from different detection methods."""
        
        # Voting system
        votes = 0
        confidence_scores = []
        methods_used = []
        
        # Statistical detector vote
        if stat_results['is_adversarial']:
            votes += 1
            confidence_scores.append(stat_results['confidence'])
            methods_used.append('statistical')
        
        # Neural detector vote
        if neural_results and neural_results['is_adversarial']:
            votes += 1
            confidence_scores.append(neural_results['confidence'])
            methods_used.append('neural')
        
        # Pattern detector vote
        pattern_vote = sum(pattern_results['patterns'].values()) >= 2  # At least 2 patterns
        if pattern_vote:
            votes += 1
            confidence_scores.append(0.7)  # Fixed confidence for pattern detection
            methods_used.append('pattern')
        
        # Final decision
        total_methods = len([1 for r in [stat_results, neural_results, pattern_results] if r])
        is_adversarial = votes >= max(1, total_methods // 2)  # Majority vote
        
        # Combined confidence
        if confidence_scores:
            combined_confidence = np.mean(confidence_scores)
        else:
            combined_confidence = 0.0
        
        # Primary method (highest confidence)
        primary_method = "statistical"  # Default
        if neural_results and neural_results.get('confidence', 0) > combined_confidence:
            primary_method = "neural"
        elif pattern_vote:
            primary_method = "pattern"
        
        return {
            'is_adversarial': is_adversarial,
            'confidence': combined_confidence,
            'votes': votes,
            'total_methods': total_methods,
            'methods_used': methods_used,
            'primary_method': primary_method,
            'perturbation_magnitude': pattern_results.get('perturbation_magnitude', 0.0),
            'evidence': {
                'statistical': stat_results,
                'neural': neural_results,
                'patterns': pattern_results
            }
        }
    
    def _classify_attack_type(self, evidence: Dict[str, Any]) -> Optional[str]:
        """Classify the type of adversarial attack."""
        
        pattern_results = evidence.get('patterns', {})
        if not pattern_results:
            return None
        
        patterns = pattern_results.get('patterns', {})
        
        # Score each attack type
        attack_scores = {}
        
        for attack_type, signature in self.attack_signatures.items():
            score = 0
            
            if signature.get('high_frequency') and patterns.get('high_frequency_noise'):
                score += 1
            if signature.get('uniform_noise') and patterns.get('uniform_noise'):
                score += 1
            if signature.get('structured_noise') and patterns.get('structured_noise'):
                score += 1
            if signature.get('minimal_noise') and patterns.get('minimal_noise'):
                score += 1
            if signature.get('low_frequency') and not patterns.get('high_frequency_noise'):
                score += 1
            
            attack_scores[attack_type] = score
        
        # Return attack type with highest score
        if attack_scores:
            best_attack = max(attack_scores.keys(), key=lambda k: attack_scores[k])
            if attack_scores[best_attack] >= 2:  # Minimum confidence threshold
                return best_attack
        
        return "unknown_attack"
    
    def get_robustness_score(self, image: np.ndarray) -> float:
        """Get robustness score for image (0 = vulnerable, 1 = robust)."""
        
        detection_result = self.detect_adversarial(image)
        
        if detection_result.is_adversarial:
            # Image is already adversarial, low robustness
            return 0.0
        
        # For non-adversarial images, estimate robustness based on features
        stat_results = self.statistical_detector.detect(image)
        features = self.statistical_detector._extract_statistical_features(image)
        
        # Calculate stability metrics
        edge_density = features[5] if len(features) > 5 else 0.5
        gradient_stats = features[6:9] if len(features) > 8 else [50, 20, 100]
        
        # Higher edge density and moderate gradients indicate more robust images
        robustness_score = 0.0
        
        # Edge density contribution (0.4 weight)
        robustness_score += min(edge_density * 2, 1.0) * 0.4
        
        # Gradient stability contribution (0.3 weight)
        if gradient_stats[2] < 200:  # Max gradient not too high
            robustness_score += 0.3
        
        # Statistical confidence contribution (0.3 weight)
        if not stat_results['is_adversarial']:
            robustness_score += 0.3
        
        return min(robustness_score, 1.0)
    
    def generate_robustness_report(self, images: List[np.ndarray]) -> Dict[str, Any]:
        """Generate comprehensive robustness report for image set."""
        
        results = {
            'total_images': len(images),
            'adversarial_detected': 0,
            'average_robustness': 0.0,
            'attack_types': {},
            'detection_methods': {'statistical': 0, 'neural': 0, 'pattern': 0},
            'processing_time_total': 0.0
        }
        
        robustness_scores = []
        
        for image in images:
            detection = self.detect_adversarial(image)
            results['processing_time_total'] += detection.processing_time
            
            if detection.is_adversarial:
                results['adversarial_detected'] += 1
                
                if detection.attack_type:
                    if detection.attack_type not in results['attack_types']:
                        results['attack_types'][detection.attack_type] = 0
                    results['attack_types'][detection.attack_type] += 1
                
                # Count detection method
                if detection.detection_method in results['detection_methods']:
                    results['detection_methods'][detection.detection_method] += 1
            
            # Calculate robustness score
            robustness_score = self.get_robustness_score(image)
            robustness_scores.append(robustness_score)
        
        results['average_robustness'] = np.mean(robustness_scores) if robustness_scores else 0.0
        results['adversarial_rate'] = results['adversarial_detected'] / len(images)
        results['average_processing_time'] = results['processing_time_total'] / len(images)
        
        return results