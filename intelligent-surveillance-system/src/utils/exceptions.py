"""Exceptions personnalisées pour le système de surveillance."""

from typing import Optional, Any


class SurveillanceSystemError(Exception):
    """Exception de base pour le système de surveillance."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Any] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details
    
    def __str__(self) -> str:
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


class ModelError(SurveillanceSystemError):
    """Erreur liée aux modèles IA."""
    pass


class DetectionError(SurveillanceSystemError):
    """Erreur de détection d'objets."""
    pass


class TrackingError(SurveillanceSystemError):
    """Erreur de suivi d'objets."""
    pass


class OrchestrationError(SurveillanceSystemError):
    """Erreur d'orchestration des outils."""
    pass


class ValidationError(SurveillanceSystemError):
    """Erreur de validation."""
    pass


class ToolError(SurveillanceSystemError):
    """Erreur d'utilisation d'un outil."""
    pass


class ProcessingError(SurveillanceSystemError):
    """Erreur de traitement générique."""
    pass


class ConfigurationError(SurveillanceSystemError):
    """Erreur de configuration."""
    pass


class StreamError(SurveillanceSystemError):
    """Erreur de flux vidéo."""
    pass


class MonitoringError(SurveillanceSystemError):
    """Erreur de monitoring."""
    pass