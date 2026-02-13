"""
Configuration du logging avec Loguru.
Logs structurés avec rotation automatique.
"""

import sys
from pathlib import Path

from loguru import logger

from src.core.config import settings


def setup_logging() -> None:
    """Configure Loguru pour l'application."""
    
    # Supprimer le handler par défaut
    logger.remove()
    
    # Format console (développement)
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )
    
    # Format fichier (production)
    file_format = (
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
        "{level: <8} | "
        "{name}:{function}:{line} | "
        "{message}"
    )
    
    # Console handler
    logger.add(
        sys.stderr,
        format=console_format,
        level=settings.log_level,
        colorize=True,
        backtrace=True,
        diagnose=settings.debug,
    )
    
    # Créer le dossier logs
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Fichier principal (rotation journalière)
    logger.add(
        log_dir / "intelli_{time:YYYY-MM-DD}.log",
        format=file_format,
        level="INFO",
        rotation="00:00",
        retention="30 days",
        compression="gz",
        encoding="utf-8",
    )
    
    # Fichier erreurs séparé
    logger.add(
        log_dir / "errors_{time:YYYY-MM-DD}.log",
        format=file_format,
        level="ERROR",
        rotation="00:00",
        retention="90 days",
        compression="gz",
        encoding="utf-8",
    )
    
    # Fichier alertes (pour audit)
    logger.add(
        log_dir / "alerts_{time:YYYY-MM-DD}.log",
        format=file_format,
        level="INFO",
        rotation="00:00",
        retention="365 days",
        compression="gz",
        encoding="utf-8",
        filter=lambda record: "alert" in record["extra"],
    )
    
    logger.info(f"Logging initialized - level: {settings.log_level}, env: {settings.env}")


def get_alert_logger():
    """Logger spécialisé pour les alertes."""
    return logger.bind(alert=True)


__all__ = ["logger", "setup_logging", "get_alert_logger"]
