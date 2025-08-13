"""Temporal transformer for sequence analysis and temporal consistency."""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import logging
import time
import math

logger = logging.getLogger(__name__)

@dataclass
class TemporalFrame:
    """Single frame in temporal sequence."""
    frame_id: str
    timestamp: float
    features: np.ndarray
    detections: List[Dict[str, Any]]
    analysis_results: Optional[Dict[str, Any]] = None

@dataclass
class TemporalAnalysis:
    """Result from temporal analysis."""
    sequence_consistency: float
    anomaly_score: float
    trend_analysis: Dict[str, Any]
    temporal_patterns: List[str]
    confidence: float
    processing_time: float

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TemporalTransformerModel(nn.Module):
    """Transformer model for temporal sequence analysis."""
    
    def __init__(self, input_dim: int, d_model: int = 256, nhead: int = 8, 
                 num_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output heads
        self.consistency_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        self.anomaly_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        self.pattern_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 8)  # 8 different temporal patterns
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, x, mask=None):
        """Forward pass through temporal transformer."""
        # x shape: [batch_size, seq_len, input_dim]
        
        # Project to model dimension
        x = self.input_projection(x)  # [batch, seq_len, d_model]
        
        # Add positional encoding
        x = x.transpose(0, 1)  # [seq_len, batch, d_model]
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # [batch, seq_len, d_model]
        
        # Apply transformer
        transformer_output = self.transformer(x, src_key_padding_mask=mask)
        
        # Apply layer normalization
        transformer_output = self.layer_norm(transformer_output)
        
        # Global average pooling
        if mask is not None:
            # Masked average
            mask_expanded = (~mask).unsqueeze(-1).float()
            pooled = (transformer_output * mask_expanded).sum(1) / mask_expanded.sum(1)
        else:
            pooled = transformer_output.mean(1)  # [batch, d_model]
        
        # Output predictions
        consistency = self.consistency_head(pooled)
        anomaly = self.anomaly_head(pooled)
        patterns = self.pattern_head(pooled)
        
        return {
            'consistency': consistency,
            'anomaly': anomaly,
            'patterns': patterns,
            'features': pooled
        }

class TemporalTransformer:
    """Temporal transformer for sequence analysis."""
    
    def __init__(self, sequence_length: int = 30, feature_dim: int = 256, device: str = "auto"):
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.device = self._select_device(device)
        
        # Temporal sequences storage
        self.temporal_sequences = {}  # stream_id -> deque of TemporalFrame
        
        # Transformer model
        self.model = None
        self._initialize_model()
        
        # Pattern definitions
        self.pattern_definitions = {
            0: "stable_normal",
            1: "gradual_increase", 
            2: "gradual_decrease",
            3: "sudden_spike",
            4: "oscillating",
            5: "trending_suspicious",
            6: "intermittent_activity",
            7: "unknown_pattern"
        }
        
        # Feature extractors for different analysis types
        self.feature_extractors = {
            'detection_based': self._extract_detection_features,
            'behavior_based': self._extract_behavior_features,
            'motion_based': self._extract_motion_features
        }
    
    def _select_device(self, device: str) -> torch.device:
        """Select optimal device."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    def _initialize_model(self):
        """Initialize temporal transformer model."""
        try:
            self.model = TemporalTransformerModel(
                input_dim=self.feature_dim,
                d_model=256,
                nhead=8,
                num_layers=4
            ).to(self.device)
            
            self.model.eval()  # Set to evaluation mode
            logger.info("Temporal transformer model initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize temporal transformer: {e}")
            self.model = None
    
    def add_frame(self, stream_id: str, frame_id: str, timestamp: float,
                  features: np.ndarray, detections: List[Dict[str, Any]], 
                  analysis_results: Optional[Dict[str, Any]] = None):
        """Add new frame to temporal sequence."""
        
        if stream_id not in self.temporal_sequences:
            self.temporal_sequences[stream_id] = deque(maxlen=self.sequence_length)
        
        # Ensure features are correct dimension
        if len(features) != self.feature_dim:
            features = self._normalize_features(features)
        
        temporal_frame = TemporalFrame(
            frame_id=frame_id,
            timestamp=timestamp,
            features=features,
            detections=detections,
            analysis_results=analysis_results
        )
        
        self.temporal_sequences[stream_id].append(temporal_frame)
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features to expected dimension."""
        if len(features) < self.feature_dim:
            # Pad with zeros
            normalized = np.zeros(self.feature_dim)
            normalized[:len(features)] = features
        else:
            # Truncate
            normalized = features[:self.feature_dim]
        
        return normalized
    
    def analyze_temporal_sequence(self, stream_id: str, 
                                analysis_type: str = 'detection_based') -> Optional[TemporalAnalysis]:
        """Analyze temporal sequence for consistency and patterns."""
        start_time = time.time()
        
        if stream_id not in self.temporal_sequences:
            return None
        
        sequence = list(self.temporal_sequences[stream_id])
        
        if len(sequence) < 5:  # Need minimum sequence length
            return None
        
        try:
            # Extract features based on analysis type
            sequence_features = self._extract_sequence_features(sequence, analysis_type)
            
            # Analyze with transformer if available
            if self.model is not None:
                analysis = self._transformer_analysis(sequence_features)
            else:
                analysis = self._statistical_analysis(sequence_features)
            
            # Additional trend analysis
            trend_analysis = self._analyze_trends(sequence)
            
            processing_time = time.time() - start_time
            
            return TemporalAnalysis(
                sequence_consistency=analysis['consistency'],
                anomaly_score=analysis['anomaly_score'],
                trend_analysis=trend_analysis,
                temporal_patterns=analysis['patterns'],
                confidence=analysis['confidence'],
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Temporal analysis failed for {stream_id}: {e}")
            return None
    
    def _extract_sequence_features(self, sequence: List[TemporalFrame], 
                                 analysis_type: str) -> np.ndarray:
        """Extract features from temporal sequence."""
        
        feature_extractor = self.feature_extractors.get(
            analysis_type, self.feature_extractors['detection_based']
        )
        
        features_list = []
        for frame in sequence:
            frame_features = feature_extractor(frame)
            features_list.append(frame_features)
        
        return np.array(features_list)
    
    def _extract_detection_features(self, frame: TemporalFrame) -> np.ndarray:
        """Extract detection-based features from frame."""
        features = []
        
        # Basic detection statistics
        num_detections = len(frame.detections)
        features.append(num_detections)
        
        if frame.detections:
            # Confidence statistics
            confidences = [d.get('confidence', 0.0) for d in frame.detections]
            features.extend([
                np.mean(confidences),
                np.std(confidences),
                np.max(confidences)
            ])
            
            # Object class distribution
            classes = [d.get('class', 'unknown') for d in frame.detections]
            class_counts = {cls: classes.count(cls) for cls in set(classes)}
            
            # Count for common classes
            common_classes = ['person', 'handbag', 'backpack', 'suitcase']
            for cls in common_classes:
                features.append(class_counts.get(cls, 0))
            
            # Spatial distribution
            bboxes = [d.get('bbox', [0, 0, 0, 0]) for d in frame.detections]
            if bboxes:
                centers_x = [(box[0] + box[2]) / 2 for box in bboxes]
                centers_y = [(box[1] + box[3]) / 2 for box in bboxes]
                
                features.extend([
                    np.mean(centers_x) / 1920,  # Normalize
                    np.mean(centers_y) / 1080,
                    np.std(centers_x) / 1920,
                    np.std(centers_y) / 1080
                ])
            else:
                features.extend([0.5, 0.5, 0.0, 0.0])  # Default center
        else:
            # No detections
            features.extend([0.0, 0.0, 0.0, 0, 0, 0, 0, 0.5, 0.5, 0.0, 0.0])
        
        # Pad to feature dimension
        while len(features) < self.feature_dim:
            features.append(0.0)
        
        return np.array(features[:self.feature_dim])
    
    def _extract_behavior_features(self, frame: TemporalFrame) -> np.ndarray:
        """Extract behavior-based features from frame."""
        features = []
        
        if frame.analysis_results:
            # Suspicion level
            suspicion = frame.analysis_results.get('suspicion_level', 0.0)
            features.append(suspicion)
            
            # Confidence
            confidence = frame.analysis_results.get('confidence', 0.0)
            features.append(confidence)
            
            # Risk factors count
            risk_factors = frame.analysis_results.get('risk_factors', [])
            features.append(len(risk_factors))
            
            # Tool results features
            tool_results = frame.analysis_results.get('tool_results', {})
            features.append(len(tool_results))
            
            # Behavior classification scores
            behavior_scores = []
            for tool_result in tool_results.values():
                if hasattr(tool_result, 'result') and tool_result.result:
                    if 'behavior_classification' in tool_result.result:
                        behavior_scores.append(1.0)
                    if 'suspicion_score' in tool_result.result:
                        behavior_scores.append(tool_result.result['suspicion_score'])
            
            if behavior_scores:
                features.extend([
                    np.mean(behavior_scores),
                    np.max(behavior_scores)
                ])
            else:
                features.extend([0.0, 0.0])
        else:
            features.extend([0.0, 0.0, 0, 0, 0.0, 0.0])
        
        # Use detection features to fill remaining dimensions
        detection_features = self._extract_detection_features(frame)
        features.extend(detection_features[:self.feature_dim - len(features)])
        
        return np.array(features[:self.feature_dim])
    
    def _extract_motion_features(self, frame: TemporalFrame) -> np.ndarray:
        """Extract motion-based features from frame."""
        features = []
        
        # Extract motion-related features from tool results
        if frame.analysis_results:
            tool_results = frame.analysis_results.get('tool_results', {})
            
            motion_data = {}
            pose_data = {}
            
            for tool_result in tool_results.values():
                if hasattr(tool_result, 'result') and tool_result.result:
                    result = tool_result.result
                    
                    if 'movement_speed' in result or 'average_velocity' in result:
                        motion_data = result
                    
                    if 'keypoints' in result or 'pose_analysis' in result:
                        pose_data = result
            
            # Motion features
            if motion_data:
                features.extend([
                    motion_data.get('average_velocity', 0.0),
                    motion_data.get('direction_changes', 0) / 10.0,
                    motion_data.get('anomaly_score', 0.0)
                ])
            else:
                features.extend([0.0, 0.0, 0.0])
            
            # Pose-based motion features
            if pose_data:
                # Simplified pose motion indicators
                indicators = pose_data.get('indicators', [])
                features.append(len(indicators) / 10.0)  # Normalize
                
                # Movement score
                movement_score = pose_data.get('movement_score', 0.0)
                features.append(movement_score)
            else:
                features.extend([0.0, 0.0])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        
        # Fill remaining with detection features
        detection_features = self._extract_detection_features(frame)
        features.extend(detection_features[:self.feature_dim - len(features)])
        
        return np.array(features[:self.feature_dim])
    
    def _transformer_analysis(self, sequence_features: np.ndarray) -> Dict[str, Any]:
        """Analyze sequence using transformer model."""
        
        # Convert to tensor
        sequence_tensor = torch.FloatTensor(sequence_features).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(sequence_tensor)
        
        # Extract results
        consistency = float(outputs['consistency'].cpu().numpy().squeeze())
        anomaly_score = float(outputs['anomaly'].cpu().numpy().squeeze())
        pattern_logits = outputs['patterns'].cpu().numpy().squeeze()
        
        # Get top patterns
        pattern_probs = torch.softmax(torch.FloatTensor(pattern_logits), dim=0).numpy()
        top_patterns = np.argsort(pattern_probs)[-3:][::-1]  # Top 3
        
        patterns = [self.pattern_definitions[idx] for idx in top_patterns if pattern_probs[idx] > 0.1]
        
        # Confidence based on model outputs
        confidence = (consistency + (1 - anomaly_score)) / 2
        
        return {
            'consistency': consistency,
            'anomaly_score': anomaly_score,
            'patterns': patterns,
            'confidence': confidence
        }
    
    def _statistical_analysis(self, sequence_features: np.ndarray) -> Dict[str, Any]:
        """Statistical analysis fallback."""
        
        # Calculate variance over time for consistency
        feature_vars = np.var(sequence_features, axis=0)
        consistency = 1.0 / (1.0 + np.mean(feature_vars))
        
        # Anomaly detection using z-scores
        z_scores = np.abs((sequence_features - np.mean(sequence_features, axis=0)) / 
                         (np.std(sequence_features, axis=0) + 1e-8))
        anomaly_score = np.mean(np.max(z_scores, axis=1) > 2.0)  # Fraction of anomalous frames
        
        # Simple pattern detection
        patterns = []
        
        # Trend detection
        mean_trajectory = np.mean(sequence_features, axis=1)
        if len(mean_trajectory) > 3:
            trend = np.polyfit(range(len(mean_trajectory)), mean_trajectory, 1)[0]
            
            if abs(trend) < 0.01:
                patterns.append("stable_normal")
            elif trend > 0.05:
                patterns.append("gradual_increase")
            elif trend < -0.05:
                patterns.append("gradual_decrease")
        
        # Spike detection
        if np.any(np.diff(mean_trajectory) > np.std(mean_trajectory) * 2):
            patterns.append("sudden_spike")
        
        # Oscillation detection
        if len(mean_trajectory) > 6:
            fft = np.fft.fft(mean_trajectory)
            dominant_freq = np.argmax(np.abs(fft[1:len(fft)//2])) + 1
            if dominant_freq > 2:  # Has periodic component
                patterns.append("oscillating")
        
        if not patterns:
            patterns = ["unknown_pattern"]
        
        confidence = consistency * (1 - anomaly_score)
        
        return {
            'consistency': consistency,
            'anomaly_score': anomaly_score,
            'patterns': patterns,
            'confidence': confidence
        }
    
    def _analyze_trends(self, sequence: List[TemporalFrame]) -> Dict[str, Any]:
        """Analyze trends in the temporal sequence."""
        
        trends = {}
        
        # Detection count trend
        detection_counts = [len(frame.detections) for frame in sequence]
        if len(detection_counts) > 3:
            detection_trend = np.polyfit(range(len(detection_counts)), detection_counts, 1)[0]
            trends['detection_trend'] = {
                'slope': float(detection_trend),
                'direction': 'increasing' if detection_trend > 0.1 else 'decreasing' if detection_trend < -0.1 else 'stable'
            }
        
        # Suspicion level trend (if available)
        suspicion_levels = []
        for frame in sequence:
            if frame.analysis_results and 'suspicion_level' in frame.analysis_results:
                suspicion_levels.append(frame.analysis_results['suspicion_level'])
        
        if len(suspicion_levels) > 3:
            suspicion_trend = np.polyfit(range(len(suspicion_levels)), suspicion_levels, 1)[0]
            trends['suspicion_trend'] = {
                'slope': float(suspicion_trend),
                'direction': 'increasing' if suspicion_trend > 0.05 else 'decreasing' if suspicion_trend < -0.05 else 'stable',
                'current_level': float(suspicion_levels[-1]) if suspicion_levels else 0.0
            }
        
        # Activity level trend (based on detection confidence)
        activity_levels = []
        for frame in sequence:
            if frame.detections:
                confidences = [d.get('confidence', 0.0) for d in frame.detections]
                activity_levels.append(np.mean(confidences))
            else:
                activity_levels.append(0.0)
        
        if len(activity_levels) > 3:
            activity_trend = np.polyfit(range(len(activity_levels)), activity_levels, 1)[0]
            trends['activity_trend'] = {
                'slope': float(activity_trend),
                'direction': 'increasing' if activity_trend > 0.02 else 'decreasing' if activity_trend < -0.02 else 'stable'
            }
        
        return trends
    
    def get_sequence_summary(self, stream_id: str) -> Optional[Dict[str, Any]]:
        """Get summary of temporal sequence."""
        
        if stream_id not in self.temporal_sequences:
            return None
        
        sequence = list(self.temporal_sequences[stream_id])
        
        if not sequence:
            return None
        
        # Basic statistics
        total_frames = len(sequence)
        time_span = sequence[-1].timestamp - sequence[0].timestamp
        
        # Detection statistics
        total_detections = sum(len(frame.detections) for frame in sequence)
        avg_detections = total_detections / total_frames
        
        # Analysis results statistics
        analyzed_frames = sum(1 for frame in sequence if frame.analysis_results)
        analysis_coverage = analyzed_frames / total_frames
        
        return {
            'stream_id': stream_id,
            'total_frames': total_frames,
            'time_span_seconds': time_span,
            'total_detections': total_detections,
            'avg_detections_per_frame': avg_detections,
            'analysis_coverage': analysis_coverage,
            'sequence_length': len(sequence)
        }
    
    def cleanup_old_sequences(self, max_age_seconds: int = 300):
        """Clean up old temporal sequences."""
        current_time = time.time()
        
        to_remove = []
        for stream_id, sequence in self.temporal_sequences.items():
            if sequence and (current_time - sequence[-1].timestamp) > max_age_seconds:
                to_remove.append(stream_id)
        
        for stream_id in to_remove:
            del self.temporal_sequences[stream_id]
        
        logger.info(f"Cleaned up {len(to_remove)} old temporal sequences")