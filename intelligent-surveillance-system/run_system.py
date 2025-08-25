#!/usr/bin/env python3
"""
🚀 Point d'Entrée Simplifié du Système de Surveillance Intelligente
===================================================================

Ce script utilise la nouvelle configuration centralisée et structure propre.
"""

import sys
import asyncio
from pathlib import Path

# Configuration du path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config.app_config import get_config, load_config
from src.main import main_async, create_argument_parser

def main():
    """Point d'entrée principal avec nouvelle configuration."""
    
    # Chargement de la configuration selon l'environnement
    import os
    profile = os.getenv('ENV', 'development')  # development, production, testing
    config = load_config(profile)
    
    print(f"🔧 Configuration chargée: {profile}")
    print(f"🤖 VLM Principal: {config.vlm.primary_model}")
    print(f"🎯 Mode d'orchestration: {config.orchestration.mode.value}")
    
    # Délégation au main principal
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\n👋 Au revoir !")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()