#!/usr/bin/env python3
"""
🚀 Script de Lancement Optimisé - Qwen2.5-VL-32B
================================================

Lancement du système de surveillance avec Qwen2.5-VL-32B comme modèle principal.
Configuration optimisée pour GPU haute performance (24GB+ VRAM).

Requirements GPU:
- NVIDIA RTX 4090 (24GB) - Recommandé
- NVIDIA RTX 3090 (24GB) - Minimum
- Tesla V100 (32GB) - Optimal

Installation préalable:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers>=4.45.0 accelerate bitsandbytes
pip install qwen-vl-utils
"""

import os
import sys
import asyncio
import argparse
import torch
from pathlib import Path
from loguru import logger

# Configuration environnement
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # GPU principal
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Ajout du chemin source
sys.path.append(str(Path(__file__).parent / "src"))

from src.core.headless.surveillance_system import HeadlessSurveillanceSystem
from src.core.vlm.dynamic_model import DynamicVisionLanguageModel


async def check_gpu_requirements():
    """Vérifie les exigences GPU pour Qwen2.5-VL-32B."""
    logger.info("🔍 Vérification des exigences GPU...")
    
    if not torch.cuda.is_available():
        logger.error("❌ CUDA non disponible - GPU requis pour Qwen2.5-VL-32B")
        return False
    
    gpu_count = torch.cuda.device_count()
    logger.info(f"📊 {gpu_count} GPU(s) détecté(s)")
    
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        vram_gb = props.total_memory / (1024**3)
        logger.info(f"GPU {i}: {props.name} - {vram_gb:.1f}GB VRAM")
        
        if vram_gb < 20:
            logger.warning(f" GPU {i} a {vram_gb:.1f}GB - 24GB recommandés pour Qwen2.5-VL-32B")
    
    return True


async def initialize_qwen_system(video_path: str, **kwargs):
    """Initialise le système avec Qwen2.5-VL-32B."""
    logger.info("🤖 Initialisation système surveillance avec Qwen2.5-VL-32B...")
    
    try:
        # Système de surveillance avec Qwen2.5-VL-32B
        surveillance_system = HeadlessSurveillanceSystem(
            vlm_model_name="Qwen/Qwen2.5-VL-32B-Instruct",
            yolo_model="yolov8n.pt",
            confidence_threshold=0.6,
            enable_tracking=True,
            vlm_mode="high_performance",
            max_frames=kwargs.get('max_frames', 20),
            frame_skip=kwargs.get('frame_skip', 2),
            enable_optimization=True,
            device="cuda"
        )
        
        # Initialisation complète
        await surveillance_system.initialize()
        
        logger.info("✅ Système initialisé avec Qwen2.5-VL-32B")
        return surveillance_system
        
    except Exception as e:
        logger.error(f"❌ Erreur initialisation: {e}")
        return None


async def run_surveillance(video_path: str, **kwargs):
    """Lance la surveillance avec Qwen2.5-VL-32B."""
    
    # Vérifications préalables
    if not await check_gpu_requirements():
        return False
    
    if not Path(video_path).exists():
        logger.error(f"❌ Vidéo non trouvée: {video_path}")
        return False
    
    # Initialisation
    system = await initialize_qwen_system(video_path, **kwargs)
    if not system:
        return False
    
    try:
        logger.info(f"🎬 Début analyse vidéo: {video_path}")
        logger.info(f"📊 Configuration: max_frames={kwargs.get('max_frames', 20)}, frame_skip={kwargs.get('frame_skip', 2)}")
        
        # Analyse complète
        results = await system.process_video(video_path)
        
        if results:
            logger.info(f" Analyse terminée - {len(results.frame_results)} frames analysées")
            
            # Statistiques
            detections = sum(len(frame.detections) for frame in results.frame_results)
            avg_confidence = sum(frame.analysis_response.confidence for frame in results.frame_results) / len(results.frame_results)
            
            logger.info(f" Statistiques:")
            logger.info(f"   • Détections YOLO: {detections}")
            logger.info(f"   • Confiance moyenne VLM: {avg_confidence:.2f}")
            logger.info(f"   • Temps total: {results.processing_time:.1f}s")
            
            # Sauvegarde résultats
            output_path = Path("surveillance_output") / f"qwen_32b_results_{int(asyncio.get_event_loop().time())}.json"
            output_path.parent.mkdir(exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                import json
                json.dump(results.to_dict(), f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Résultats sauvegardés: {output_path}")
            return True
        else:
            logger.warning(" Aucun résultat généré")
            return False
            
    except Exception as e:
        logger.error(f" Erreur pendant analyse: {e}")
        return False
    
    finally:
        await system.cleanup()
        logger.info(" Nettoyage terminé")


def main():
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(
        description=" Surveillance Intelligente avec Qwen2.5-VL-32B",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:

# Analyse vidéo standard
python launch_qwen_32b.py --video videos/surveillance01.mp4

# Analyse haute performance (plus de frames)
python launch_qwen_32b.py --video videos/test.mp4 --max-frames 50 --frame-skip 1

# Analyse rapide (moins de frames)  
python launch_qwen_32b.py --video videos/test.mp4 --max-frames 10 --frame-skip 5

Exigences matérielles:
- GPU NVIDIA avec 24GB+ VRAM (RTX 4090/3090 recommandé)
- 32GB+ RAM système
- CUDA 12.1+ installé
        """
    )
    
    parser.add_argument(
        "--video", 
        required=True,
        help="Chemin vers la vidéo à analyser"
    )
    
    parser.add_argument(
        "--max-frames",
        type=int,
        default=20,
        help="Nombre maximum de frames à analyser (défaut: 20)"
    )
    
    parser.add_argument(
        "--frame-skip", 
        type=int,
        default=2,
        help="Nombre de frames à ignorer entre analyses (défaut: 2)"
    )
    
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="ID du GPU à utiliser (défaut: 0)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Affichage détaillé"
    )
    
    args = parser.parse_args()
    
    # Configuration logging
    logger.remove()
    log_level = "DEBUG" if args.verbose else "INFO"
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=log_level
    )
    
    # Configuration GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    
    logger.info("🚀 Démarrage surveillance Qwen2.5-VL-32B")
    logger.info(f"📹 Vidéo: {args.video}")
    logger.info(f"⚙️ Config: max_frames={args.max_frames}, frame_skip={args.frame_skip}")
    
    # Lancement asynchrone
    try:
        success = asyncio.run(run_surveillance(
            video_path=args.video,
            max_frames=args.max_frames,
            frame_skip=args.frame_skip
        ))
        
        if success:
            logger.info("🎉 Analyse terminée avec succès!")
            sys.exit(0)
        else:
            logger.error("❌ Échec de l'analyse")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("⏹️ Arrêt demandé par l'utilisateur")
        sys.exit(0)
    except Exception as e:
        logger.error(f"💥 Erreur critique: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()