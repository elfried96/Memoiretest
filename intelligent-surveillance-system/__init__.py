"""Système de Surveillance Intelligente Multimodale."""

__version__ = "1.0.0"
__author__ = "Elfried Steve David KINZOUN"
__description__ = "Système de surveillance basé sur VLM avec orchestration d'outils"

# Configuration du path pour les imports
import sys
from pathlib import Path

# Ajouter le répertoire racine au Python path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))