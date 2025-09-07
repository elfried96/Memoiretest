#!/usr/bin/env python3
"""
🎯 MAIN HEADLESS REFACTORISÉ - Pipeline Modulaire
==================================================

Version modernisée avec architecture modulaire pour maintenance optimale.
"""

import asyncio
import argparse
import logging
import sys
from pathlib import Path

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Import du système refactorisé
from src.core.headless import HeadlessSurveillanceSystem
from src.core.orchestrator.vlm_orchestrator import OrchestrationMode


def parse_arguments():
    """Parse les arguments de ligne de commande."""
    parser = argparse.ArgumentParser(
        description="🎯 Système de surveillance intelligent headless",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  # Vidéo avec Kimi-VL en mode balanced
  python main_headless_refactored.py --video videos/surveillance.mp4 --model kimi-vl-a3b-thinking

  # Webcam avec Qwen en mode rapide  
  python main_headless_refactored.py --video 0 --model qwen2-vl-7b-instruct --mode FAST

  # Test avec frames limitées
  python main_headless_refactored.py --video videos/test.mp4 --max-frames 100 --save-frames
        """
    )
    
    # Source vidéo
    parser.add_argument(
        "--video", "-v",
        default="0",
        help="Source vidéo (fichier ou webcam, défaut: webcam)"
    )
    
    # Modèle VLM
    parser.add_argument(
        "--model", "-m",
        default="kimi-vl-a3b-thinking",
        choices=[
            "kimi-vl-a3b-thinking",
            "kimi-vl-a3b-thinking-2506", 
            "qwen2-vl-7b-instruct",
            "qwen2.5-vl-32b-instruct"
        ],
        help="Modèle VLM à utiliser (défaut: kimi-vl-a3b-thinking)"
    )
    
    # Mode d'orchestration
    parser.add_argument(
        "--mode",
        default="BALANCED",
        choices=["FAST", "BALANCED", "THOROUGH"],
        help="Mode d'orchestration (défaut: BALANCED)"
    )
    
    # Paramètres de traitement
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=1,
        help="Frames à ignorer entre analyses (défaut: 1)"
    )
    
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Nombre maximum de frames à traiter"
    )
    
    # Mode d'analyse VLM
    parser.add_argument(
        "--vlm-mode",
        default="smart",
        choices=["continuous", "periodic", "smart"],
        help="Mode d'analyse VLM (défaut: smart)"
    )
    
    # Sauvegarde
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="Sauvegarder les frames avec détections"
    )
    
    parser.add_argument(
        "--no-save-results",
        action="store_true",
        help="Ne pas sauvegarder les résultats JSON"
    )
    
    parser.add_argument(
        "--output-dir",
        default="surveillance_output",
        help="Répertoire de sortie (défaut: surveillance_output)"
    )
    
    # Debug
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Active le mode debug"
    )
    
    return parser.parse_args()


async def main():
    """Point d'entrée principal."""
    try:
        # Parsing des arguments
        args = parse_arguments()
        
        # Configuration du logging
        if args.debug:
            logging.getLogger().setLevel(logging.DEBUG)
            logger.debug("🐛 Mode debug activé")
        
        # Affichage configuration
        logger.info("🎯 SYSTÈME DE SURVEILLANCE HEADLESS REFACTORISÉ")
        logger.info("=" * 55)
        logger.info(f"📹 Source vidéo: {args.video}")
        logger.info(f"🧠 Modèle VLM: {args.model}")
        logger.info(f"⚙️ Mode orchestration: {args.mode}")
        logger.info(f"🎬 Frame skip: {args.frame_skip}")
        logger.info(f"🔍 Mode VLM: {args.vlm_mode}")
        
        if args.max_frames:
            logger.info(f"🎯 Limite frames: {args.max_frames}")
        
        # Validation de la source vidéo
        if args.video != "0" and not Path(args.video).exists():
            logger.error(f"❌ Fichier vidéo introuvable: {args.video}")
            sys.exit(1)
        
        # Conversion du mode d'orchestration
        orchestration_mode = getattr(OrchestrationMode, args.mode)
        
        # Initialisation du système
        logger.info("\n🚀 Initialisation du système...")
        
        surveillance_system = HeadlessSurveillanceSystem(
            video_source=args.video,
            vlm_model=args.model,
            orchestration_mode=orchestration_mode,
            save_results=not args.no_save_results,
            save_frames=args.save_frames,
            frame_skip=args.frame_skip,
            max_frames=args.max_frames,
            vlm_analysis_mode=args.vlm_mode,
            output_dir=args.output_dir
        )
        
        # Lancement de la surveillance
        logger.info("🎬 Démarrage de la surveillance...\n")
        
        summary = await surveillance_system.run_surveillance()
        
        # Affichage final
        logger.info("\n✅ SURVEILLANCE TERMINÉE AVEC SUCCÈS")
        logger.info("=" * 55)
        logger.info(f"📊 Résumé de session:")
        logger.info(f"  • Frames traitées: {summary.total_frames}")
        logger.info(f"  • Détections totales: {summary.total_detections}")
        logger.info(f"  • Personnes détectées: {summary.total_persons}")
        logger.info(f"  • Temps moyen/frame: {summary.average_processing_time:.2f}s")
        logger.info(f"  • Durée totale: {summary.session_duration:.1f}s")
        
        # Alertes
        if summary.alerts_by_level:
            logger.info(f"🚨 Alertes par niveau:")
            for level, count in summary.alerts_by_level.items():
                if count > 0:
                    logger.info(f"    {level}: {count}")
        
        # Événements clés
        if summary.key_events:
            logger.info(f"📋 {len(summary.key_events)} événements critiques détectés")
        
        logger.info(f"📁 Résultats sauvés dans: {args.output_dir}/")
        
    except KeyboardInterrupt:
        logger.info("\n🛑 Surveillance interrompue par l'utilisateur")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\n❌ Erreur fatale: {e}")
        if args.debug:
            import traceback
            logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    # Vérification version Python
    if sys.version_info < (3, 9):
        print("❌ Python 3.9+ requis")
        sys.exit(1)
    
    # Lancement asynchrone
    asyncio.run(main())