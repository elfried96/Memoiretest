"""Ensemble d'outils avancés pour des capacités de surveillance améliorées."""

import os
import logging

logger = logging.getLogger(__name__)

# Vérification des dépendances CUDA avant import
def check_cuda_dependencies():
    """Vérifie si les dépendances CUDA sont disponibles."""
    try:
        import torch
        return torch.cuda.is_available()
    except Exception as e:
        logger.warning(f"CUDA non disponible: {e}")
        return False

# Import conditionnel des outils avancés
CUDA_AVAILABLE = check_cuda_dependencies()

if CUDA_AVAILABLE:
    try:
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
            'DomainAdapter',
            'CUDA_AVAILABLE'
        ]
        logger.info("Tous les outils avancés CUDA chargés avec succès")
        
    except Exception as e:
        logger.error(f"Erreur chargement outils avancés CUDA: {e}")
        CUDA_AVAILABLE = False

if not CUDA_AVAILABLE:
    # Créer des stubs pour les outils manquants
    logger.warning("CUDA non disponible - utilisation de stubs pour outils avancés")
    
    class AdvancedToolStub:
        """Stub pour outils avancés non disponibles."""
        def __init__(self):
            self.available = False
            logger.info(f"{self.__class__.__name__} en mode stub (CUDA requis)")
        
        def process(self, *args, **kwargs):
            return {"error": "CUDA_NOT_AVAILABLE", "tool": self.__class__.__name__}
    
    class SAM2Segmentator(AdvancedToolStub): pass
    class DinoV2FeatureExtractor(AdvancedToolStub): pass  
    class OpenPoseEstimator(AdvancedToolStub): pass
    class TrajectoryAnalyzer(AdvancedToolStub): pass
    class MultiModalFusion(AdvancedToolStub): pass
    class TemporalTransformer(AdvancedToolStub): pass
    class AdversarialDetector(AdvancedToolStub): pass
    class DomainAdapter(AdvancedToolStub): pass
    
    __all__ = [
        'SAM2Segmentator',
        'DinoV2FeatureExtractor', 
        'OpenPoseEstimator', 
        'TrajectoryAnalyzer',
        'MultiModalFusion',
        'TemporalTransformer',
        'AdversarialDetector',
        'DomainAdapter',
        'CUDA_AVAILABLE'
    ]