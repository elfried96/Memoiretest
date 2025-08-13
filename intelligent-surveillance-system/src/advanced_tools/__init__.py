"""Ensemble d'outils avancés pour des capacités de surveillance améliorées."""

from .sam2_segmentation import SAM2Segmentator
from .dino_features import DinoV2FeatureExtractor
from .pose_estimation import OpenPoseEstimator
from .trajectory_analyzer import TrajectoryAnalyzer
from .multimodal_fusion import MultiModalFusion
from .temporal_transformer import TemporalTransformer
from .adversarial_detector import AdversarialDetector
from .domain_adapter import DomainAdapter

__all__ = [
    'SAM2Segmentator',
    'DinoV2FeatureExtractor', 
    'OpenPoseEstimator',
    'TrajectoryAnalyzer',
    'MultiModalFusion',
    'TemporalTransformer',
    'AdversarialDetector',
    'DomainAdapter'
]