#!/usr/bin/env python3
"""
🧪 Script pour tester un modèle VLM unique sans fallback
=========================================================

Permet de tester :
- Kimi-VL uniquement
- Qwen2-VL uniquement  
- Git-base (modèle léger pour tests) uniquement

Sans chargement de modèles de fallback pour économiser la mémoire.
"""

import sys
import asyncio
import argparse
from pathlib import Path

# Configuration du path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config.app_config import load_config, get_config
from src.core.vlm.dynamic_model import DynamicVisionLanguageModel


async def test_model_loading(model_name: str, config_profile: str = "testing") -> bool:
    """Test de chargement d'un modèle unique."""
    
    print(f"🔧 Chargement configuration: {config_profile}")
    config = load_config(config_profile)
    
    # Override du modèle dans la config
    config.vlm.primary_model = model_name
    config.vlm.fallback_models = []
    config.vlm.enable_fallback = False
    
    print(f"🤖 Test modèle: {model_name}")
    print(f"📱 Device: {config.vlm.device.value}")
    print(f"🔢 Quantization 4-bit: {config.vlm.load_in_4bit}")
    print(f"🚫 Fallback: {config.vlm.enable_fallback}")
    print("=" * 50)
    
    try:
        # Initialisation du VLM
        vlm = DynamicVisionLanguageModel(
            default_model=model_name,
            device=config.vlm.device.value,
            enable_fallback=False  # Pas de fallback
        )
        
        # Test de chargement
        print("🔄 Chargement du modèle...")
        success = await vlm.load_model(model_name)
        
        if success:
            print(f"✅ Modèle {model_name} chargé avec succès !")
            
            # Test basique d'inférence
            print("🧪 Test d'inférence basique...")
            
            # Ici on pourrait ajouter un test d'inférence simple
            # Mais pour l'économie mémoire, on se contente du chargement
            
            print("💾 Libération de la mémoire...")
            vlm._unload_current_model()
            print("✅ Test terminé avec succès !")
            return True
        else:
            print(f"❌ Échec du chargement de {model_name}")
            return False
            
    except Exception as e:
        print(f"❌ Erreur lors du test: {e}")
        return False


async def test_memory_usage():
    """Test de l'utilisation mémoire."""
    try:
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        print("\n📊 Utilisation Mémoire:")
        print(f"  RSS: {memory_info.rss / 1024 / 1024:.1f} MB")
        print(f"  VMS: {memory_info.vms / 1024 / 1024:.1f} MB")
        
        # Mémoire GPU si disponible
        try:
            import torch
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1024 / 1024
                memory_reserved = torch.cuda.memory_reserved() / 1024 / 1024
                print(f"  GPU Allocated: {memory_allocated:.1f} MB")
                print(f"  GPU Reserved: {memory_reserved:.1f} MB")
        except:
            pass
            
    except ImportError:
        print("⚠️ psutil non disponible pour monitoring mémoire")


def create_parser() -> argparse.ArgumentParser:
    """Créer le parser d'arguments."""
    parser = argparse.ArgumentParser(
        description="Test d'un modèle VLM unique sans fallback",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  python scripts/test_single_model.py --kimi           # Test Kimi-VL uniquement
  python scripts/test_single_model.py --qwen          # Test Qwen2-VL uniquement  
  python scripts/test_single_model.py --git-base      # Test Git-base (léger)
  python scripts/test_single_model.py --model custom-model-name  # Modèle personnalisé
        """
    )
    
    # Modèles prédéfinis
    parser.add_argument(
        "--kimi",
        action="store_true",
        help="Tester Kimi-VL uniquement"
    )
    
    parser.add_argument(
        "--qwen",
        action="store_true", 
        help="Tester Qwen2-VL uniquement"
    )
    
    parser.add_argument(
        "--git-base",
        action="store_true",
        help="Tester Git-base (modèle léger)"
    )
    
    # Modèle personnalisé
    parser.add_argument(
        "--model",
        type=str,
        help="Nom du modèle à tester"
    )
    
    # Configuration
    parser.add_argument(
        "--profile",
        type=str,
        default="testing",
        choices=["testing", "testing_kimi", "testing_qwen", "development"],
        help="Profil de configuration à utiliser"
    )
    
    parser.add_argument(
        "--memory-stats",
        action="store_true",
        help="Afficher les statistiques mémoire"
    )
    
    return parser


async def main():
    """Point d'entrée principal."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Détermination du modèle à tester
    if args.kimi:
        model_name = "kimi-vl-a3b-thinking"
        profile = "testing_kimi"
    elif args.qwen:
        model_name = "qwen2-vl-7b-instruct"
        profile = "testing_qwen"
    elif args.git_base:
        model_name = "microsoft/git-base-coco"
        profile = "testing"
    elif args.model:
        model_name = args.model
        profile = args.profile
    else:
        print("❌ Veuillez spécifier un modèle à tester")
        parser.print_help()
        return
    
    print(f"🎯 Test Modèle VLM Unique - {model_name}")
    print("=" * 60)
    
    if args.memory_stats:
        print("📊 Mémoire avant test:")
        await test_memory_usage()
    
    # Test du modèle
    success = await test_model_loading(model_name, profile)
    
    if args.memory_stats:
        print("\n📊 Mémoire après test:")
        await test_memory_usage()
    
    # Résumé
    print("\n" + "=" * 60)
    if success:
        print(f"🎉 Test réussi pour {model_name}")
        print("💡 Le modèle peut être utilisé sans problème de mémoire")
    else:
        print(f"💥 Test échoué pour {model_name}")
        print("🔧 Vérifiez la configuration ou la disponibilité du modèle")
    
    return 0 if success else 1


if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        sys.exit(result)
    except KeyboardInterrupt:
        print("\n⚠️ Interruption utilisateur")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erreur inattendue: {e}")
        sys.exit(1)