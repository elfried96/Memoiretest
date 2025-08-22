#!/usr/bin/env python3
"""
Script de téléchargement et test des vrais modèles VLM.
ATTENTION: Ce script télécharge de gros modèles (plusieurs GB).
"""

import asyncio
import sys
import os
import subprocess
import logging
from pathlib import Path

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def install_dependencies():
    """Installe les dépendances nécessaires."""
    logger.info("🔧 Installation des dépendances...")
    
    dependencies = [
        "torch>=2.1.0",
        "torchvision>=0.16.0", 
        "transformers>=4.35.0",
        "accelerate>=0.21.0",
        "opencv-python>=4.8.0",
        "pillow>=10.0.0",
        "numpy>=1.24.0",
        "huggingface-hub>=0.17.0",
        "safetensors>=0.4.0"
    ]
    
    for dep in dependencies:
        try:
            logger.info(f"Installation de {dep}...")
            subprocess.run([sys.executable, "-m", "pip", "install", dep], check=True, capture_output=True)
            logger.info(f"✅ {dep} installé")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Erreur installation {dep}: {e}")
            return False
    
    return True


def check_gpu_availability():
    """Vérifie la disponibilité du GPU."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"✅ GPU disponible: {gpu_name} ({gpu_memory:.1f} GB)")
            return True, gpu_memory
        else:
            logger.warning("⚠️ Aucun GPU CUDA détecté - utilisation CPU")
            return False, 0
    except ImportError:
        logger.error("❌ PyTorch non installé")
        return False, 0


def download_model(model_name: str, model_path: str) -> bool:
    """Télécharge un modèle spécifique."""
    logger.info(f"📥 Téléchargement de {model_name}...")
    
    try:
        from transformers import AutoTokenizer, AutoProcessor
        
        # Tentative de téléchargement du tokenizer/processor
        if "llava" in model_name.lower():
            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            logger.info(f"✅ Processor LLaVA téléchargé: {model_name}")
        elif "qwen" in model_name.lower():
            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            logger.info(f"✅ Processor Qwen2-VL téléchargé: {model_name}")
        elif "kimi" in model_name.lower():
            # Kimi-VL nécessite des credentials Moonshot AI
            logger.warning(f"⚠️ Kimi-VL nécessite une API key Moonshot AI")
            return False
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            logger.info(f"✅ Tokenizer téléchargé: {model_name}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur téléchargement {model_name}: {e}")
        return False


def test_model_loading(model_name: str, model_path: str) -> bool:
    """Test le chargement d'un modèle."""
    logger.info(f"🧪 Test de chargement {model_name}...")
    
    try:
        import torch
        from transformers import AutoProcessor, AutoModelForVision2Seq, AutoModelForCausalLM
        
        # Configuration selon le modèle
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        
        logger.info(f"Device: {device}, dtype: {torch_dtype}")
        
        if "llava" in model_name.lower():
            processor = AutoProcessor.from_pretrained(model_path)
            model = AutoModelForVision2Seq.from_pretrained(
                model_path, 
                torch_dtype=torch_dtype,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True
            )
            
        elif "qwen" in model_name.lower():
            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            model = AutoModelForVision2Seq.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True
            )
            
        else:
            logger.error(f"Type de modèle non supporté: {model_name}")
            return False
        
        # Test basique
        if device != "auto":
            model = model.to(device)
        model.eval()
        
        logger.info(f"✅ Modèle {model_name} chargé avec succès")
        
        # Nettoyage mémoire
        del model
        del processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur test {model_name}: {e}")
        return False


async def test_surveillance_analysis(model_name: str, model_path: str) -> bool:
    """Test d'analyse de surveillance avec un vrai modèle."""
    logger.info(f"🎯 Test analyse surveillance avec {model_name}...")
    
    try:
        import torch
        import base64
        from PIL import Image
        from io import BytesIO
        from transformers import AutoProcessor, AutoModelForVision2Seq
        
        # Chargement du modèle
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        
        if not hasattr(model, 'device_map'):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device)
        
        model.eval()
        
        # Créer une image de test
        test_image = Image.new('RGB', (640, 480), color='white')
        # Ajouter du texte simulant une scène de magasin
        from PIL import ImageDraw
        draw = ImageDraw.Draw(test_image)
        draw.rectangle([100, 100, 200, 400], fill='blue', outline='black')
        draw.text((110, 120), "PERSON", fill='white')
        
        # Prompt de surveillance
        surveillance_prompt = """
        Analyse cette image de surveillance de magasin. 
        
        Détermine:
        1. Le niveau de suspicion (LOW/MEDIUM/HIGH/CRITICAL)
        2. Le type d'action (NORMAL_SHOPPING/SUSPICIOUS_MOVEMENT/POTENTIAL_THEFT)
        3. Ta confiance dans l'analyse (0.0 à 1.0)
        4. Une description brève
        5. Des recommandations d'action
        
        Contexte: Magasin, camera de surveillance, détection de personne présente.
        """
        
        # Préparation des inputs
        inputs = processor(
            text=surveillance_prompt,
            images=test_image,
            return_tensors="pt",
            padding=True
        )
        
        if hasattr(model, 'device_map'):
            # Le modèle gère automatiquement les devices
            pass
        else:
            device = next(model.parameters()).device
            inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        
        # Génération
        logger.info("Génération en cours...")
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True
            )
        
        # Décodage
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Extraction de la réponse
        if surveillance_prompt in generated_text:
            response = generated_text.split(surveillance_prompt)[-1].strip()
        else:
            response = generated_text.strip()
        
        logger.info(f"📊 Réponse du modèle {model_name}:")
        logger.info(f"{'='*50}")
        logger.info(response)
        logger.info(f"{'='*50}")
        
        # Nettoyage
        del model
        del processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info(f"✅ Test surveillance {model_name} réussi")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur test surveillance {model_name}: {e}")
        return False


async def main():
    """Fonction principale."""
    print("""
🎯 TÉLÉCHARGEMENT ET TEST DES MODÈLES VLM RÉELS
==============================================

⚠️  ATTENTION: Ce script télécharge de gros modèles (2-7 GB chacun)
⚠️  Assurez-vous d'avoir suffisamment d'espace disque et de bande passante

Modèles supportés:
- LLaVA-NeXT (llava-hf/llava-v1.6-mistral-7b-hf) ~7GB
- Qwen2-VL (Qwen/Qwen2-VL-7B-Instruct) ~7GB  
- Kimi-VL (moonshot-ai/kimi-vl-a3b-instruct) - Nécessite API key

""")
    
    # Vérifications préliminaires
    logger.info("🔍 Vérifications système...")
    
    # 1. Installation des dépendances
    if not install_dependencies():
        logger.error("❌ Échec installation dépendances")
        return
    
    # 2. Vérification GPU
    has_gpu, gpu_memory = check_gpu_availability()
    if has_gpu and gpu_memory < 8:
        logger.warning(f"⚠️ GPU avec {gpu_memory:.1f}GB peut être insuffisant pour les modèles 7B")
    
    # 3. Espace disque
    free_space = os.statvfs('.').f_frsize * os.statvfs('.').f_availls // (1024**3)
    logger.info(f"💾 Espace disque disponible: {free_space} GB")
    if free_space < 20:
        logger.warning("⚠️ Espace disque faible - au moins 20GB recommandés")
    
    # Modèles à tester
    models_to_test = {
        "LLaVA-NeXT-7B": "llava-hf/llava-v1.6-mistral-7b-hf",
        "Qwen2-VL-7B": "Qwen/Qwen2-VL-7B-Instruct"
    }
    
    # Demander confirmation
    response = input("\n🚀 Continuer avec le téléchargement? (y/N): ")
    if response.lower() != 'y':
        logger.info("Opération annulée")
        return
    
    # Tests séquentiels des modèles
    results = {}
    
    for model_name, model_path in models_to_test.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"📥 TRAITEMENT DE {model_name}")
        logger.info(f"{'='*60}")
        
        try:
            # 1. Téléchargement
            download_success = download_model(model_name, model_path)
            if not download_success:
                results[model_name] = "❌ Échec téléchargement"
                continue
            
            # 2. Test de chargement
            loading_success = test_model_loading(model_name, model_path)
            if not loading_success:
                results[model_name] = "❌ Échec chargement"
                continue
            
            # 3. Test d'analyse de surveillance
            analysis_success = await test_surveillance_analysis(model_name, model_path)
            if analysis_success:
                results[model_name] = "✅ Complet"
            else:
                results[model_name] = "⚠️ Partiel"
                
        except KeyboardInterrupt:
            logger.info("Interruption utilisateur")
            break
        except Exception as e:
            logger.error(f"Erreur inattendue pour {model_name}: {e}")
            results[model_name] = f"❌ Erreur: {e}"
    
    # Résumé final
    logger.info(f"\n{'='*60}")
    logger.info("📊 RÉSUMÉ DES TESTS")
    logger.info(f"{'='*60}")
    
    for model_name, status in results.items():
        logger.info(f"{model_name}: {status}")
    
    logger.info("\n🎯 Tests terminés!")
    
    # Recommandations
    logger.info("\n💡 RECOMMANDATIONS:")
    logger.info("- Pour la production: Utilisez LLaVA-NeXT (plus stable)")
    logger.info("- Pour la recherche: Qwen2-VL (meilleur raisonnement)")
    logger.info("- Pour l'edge computing: Utilisez les versions quantifiées")
    logger.info("- Surveillez l'utilisation mémoire GPU en production")


if __name__ == "__main__":
    asyncio.run(main())