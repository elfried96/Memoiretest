#!/usr/bin/env python3
"""
🚀 Script de Lancement - Dashboard de Surveillance Intelligente
==============================================================

Script principal pour lancer le dashboard de surveillance avec
intégration complète au système VLM existant.
"""

import subprocess
import sys
import os
import argparse
from pathlib import Path
import logging

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_environment():
    """Configure l'environnement d'exécution."""
    # Ajout du répertoire racine du projet au PYTHONPATH
    current_dir = Path(__file__).parent
    project_root = current_dir.parent
    
    # Ajouter le répertoire racine au sys.path pour les imports
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    current_pythonpath = os.environ.get('PYTHONPATH', '')
    if str(project_root) not in current_pythonpath:
        new_pythonpath = f"{project_root}:{current_pythonpath}" if current_pythonpath else str(project_root)
        os.environ['PYTHONPATH'] = new_pythonpath
        logger.info(f"📂 PYTHONPATH mis à jour: {new_pythonpath}")

def check_dependencies():
    """Vérifie les dépendances nécessaires."""
    try:
        import streamlit
        import cv2
        import numpy
        import plotly
        logger.info("✅ Dépendances dashboard disponibles")
        return True
    except ImportError as e:
        logger.error(f"❌ Dépendance manquante: {e}")
        return False

def check_core_system():
    """Vérifie la disponibilité du système core."""
    try:
        # Tentative d'import des modules core
        sys.path.append(str(Path(__file__).parent.parent / "src"))
        from src.core.vlm.model import VisionLanguageModel
        from src.core.orchestrator.adaptive_orchestrator import AdaptiveOrchestrator
        logger.info("✅ Système VLM core disponible")
        return True
    except ImportError as e:
        logger.warning(f"⚠️ Système core non disponible: {e}")
        logger.info("🔧 Le dashboard fonctionnera en mode simulation")
        return False

def launch_dashboard(app_type="full", port=8501, host="0.0.0.0"):
    """Lance le dashboard Streamlit."""
    
    # Sélection de l'application
    app_files = {
        "full": "surveillance_dashboard.py",
        "simple": "app_simple.py", 
        "legacy": "app.py"
    }
    
    app_file = app_files.get(app_type, "surveillance_dashboard.py")
    app_path = Path(__file__).parent / app_file
    
    if not app_path.exists():
        logger.error(f"❌ Fichier d'application non trouvé: {app_path}")
        return False
    
    logger.info(f"🚀 Lancement du dashboard: {app_file}")
    logger.info(f"🌐 URL: http://{host}:{port}")
    logger.info(f"📊 Type: {app_type}")
    
    # Construction de la commande Streamlit
    cmd = [
        sys.executable,
        "-m", "streamlit",
        "run",
        str(app_path),
        "--server.port", str(port),
        "--server.address", host,
        "--browser.gatherUsageStats", "false",
        "--theme.base", "dark",
        "--theme.primaryColor", "#1f77b4",
        "--theme.backgroundColor", "#0e1117",
        "--theme.secondaryBackgroundColor", "#262730"
    ]
    
    try:
        # Lancement du processus
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Erreur lancement dashboard: {e}")
        return False
    except KeyboardInterrupt:
        logger.info("⏹️ Dashboard arrêté par l'utilisateur")
        return True

def main():
    """Fonction principale."""
    parser = argparse.ArgumentParser(
        description="🔒 Dashboard de Surveillance Intelligente",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  %(prog)s                     # Dashboard complet sur port 8501
  %(prog)s --type simple       # Version simple
  %(prog)s --port 8502         # Port personnalisé
  %(prog)s --host 127.0.0.1    # Host personnalisé
  %(prog)s --check-only        # Vérification seulement
        """
    )
    
    parser.add_argument(
        "--type", 
        choices=["full", "simple", "legacy"],
        default="full",
        help="Type de dashboard à lancer"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Port d'écoute (défaut: 8501)"
    )
    
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Adresse d'écoute (défaut: 0.0.0.0)"
    )
    
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Vérifier les dépendances seulement"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Affichage détaillé"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Configuration de l'environnement
    setup_environment()
    
    # Vérifications
    logger.info("🔍 Vérification du système...")
    
    deps_ok = check_dependencies()
    core_ok = check_core_system()
    
    if not deps_ok:
        logger.error("❌ Dépendances manquantes. Installez les requirements.")
        sys.exit(1)
    
    if args.check_only:
        if deps_ok and core_ok:
            logger.info("✅ Système prêt pour le dashboard complet")
        elif deps_ok:
            logger.info("⚠️ Dashboard disponible en mode simulation seulement")
        sys.exit(0)
    
    # Sélection automatique du type si core non disponible
    app_type = args.type
    if not core_ok and app_type == "full":
        logger.warning("⚠️ Système core non disponible, basculement vers 'simple'")
        app_type = "simple"
    
    # Lancement du dashboard
    logger.info("=" * 60)
    logger.info("🔒 DASHBOARD DE SURVEILLANCE INTELLIGENTE")
    logger.info("=" * 60)
    
    success = launch_dashboard(
        app_type=app_type,
        port=args.port,
        host=args.host
    )
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()