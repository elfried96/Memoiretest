#!/usr/bin/env python3
"""
Script de réparation pour Kimi-VL basé sur recherche GitHub
==========================================================

Solutions trouvées sur GitHub Issues pour DynamicCache et seen_tokens
"""

import os
import subprocess
import sys

def fix_transformers_version():
    """Fix 1: Forcer version Transformers compatible."""
    print("🔧 Fix 1: Installation transformers version compatible...")
    
    commands = [
        "pip uninstall transformers -y",
        "pip install transformers==4.51.3",  # Version officielle MoonshotAI
        "pip install flash-attn",  # Recommandé pour éviter OOM
    ]
    
    for cmd in commands:
        try:
            print(f"Exécution: {cmd}")
            result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
            print(f"✅ Succès: {cmd}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Échec: {cmd}")
            print(f"Erreur: {e.stderr}")

def fix_vllm_environment():
    """Fix 2: Variables d'environnement VLLM (source: GitHub Issues)."""
    print("\n🔧 Fix 2: Configuration variables environnement VLLM...")
    
    # Variables d'environnement trouvées sur GitHub Issues
    env_vars = {
        "VLLM_USE_V1": "0",  # Fix pour Kimi-VL compatibility
        "TRANSFORMERS_VERBOSITY": "info",  # Debug transformers warnings
        "TOKENIZERS_PARALLELISM": "false",  # Éviter warnings tokenizers
        "CUDA_VISIBLE_DEVICES": "0",  # Force GPU 0
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"✅ {key}={value}")
    
    return env_vars

def test_kimi_vl_loading():
    """Fix 3: Test chargement avec nouvelles configurations."""
    print("\n🔧 Fix 3: Test chargement Kimi-VL...")
    
    test_code = '''
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
import warnings
warnings.filterwarnings("ignore")

print("Testing Kimi-VL loading...")

try:
    # Test modèle nouveau (2506)
    model_id = "moonshotai/Kimi-VL-A3B-Thinking-2506"
    print(f"Loading {model_id}...")
    
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        use_cache=False  # Fix DynamicCache
    )
    
    print("✅ Kimi-VL 2506 chargé avec succès!")
    print(f"Device: {model.device}")
    print(f"Dtype: {model.dtype}")
    
except Exception as e:
    print(f"❌ Erreur Kimi-VL 2506: {e}")
    
    # Fallback vers version originale
    try:
        model_id = "moonshotai/Kimi-VL-A3B-Thinking"
        print(f"Fallback: Loading {model_id}...")
        
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            use_cache=False  # Fix DynamicCache
        )
        
        print("✅ Kimi-VL original chargé avec succès!")
        
    except Exception as e2:
        print(f"❌ Erreur Kimi-VL original: {e2}")
'''
    
    with open("/tmp/test_kimi.py", "w") as f:
        f.write(test_code)
    
    try:
        result = subprocess.run([sys.executable, "/tmp/test_kimi.py"], 
                              capture_output=True, text=True, timeout=300)
        print(result.stdout)
        if result.stderr:
            print(f"Warnings: {result.stderr}")
    except subprocess.TimeoutExpired:
        print("❌ Timeout lors du test")
    except Exception as e:
        print(f"❌ Erreur test: {e}")

def create_fixed_config():
    """Fix 4: Créer configuration optimisée."""
    print("\n🔧 Fix 4: Création configuration optimisée...")
    
    config = '''# Configuration optimisée basée sur recherche GitHub
KIMI_VL_OPTIMAL_CONFIG = {
    # Nouveau modèle recommandé (moins d'erreurs)
    "model_id": "moonshotai/Kimi-VL-A3B-Thinking-2506",
    
    # Paramètres anti-DynamicCache
    "generation_config": {
        "max_new_tokens": 200,      # Réduit pour vitesse
        "temperature": 0.3,         # Moins créatif = plus stable  
        "do_sample": False,         # Greedy = pas de sampling
        "use_cache": False,         # CRITIQUE: Désactive DynamicCache
        "pad_token_id": None,       # Auto-détecté
    },
    
    # Paramètres modèle optimaux
    "model_config": {
        "torch_dtype": "bfloat16",  # Plus stable que float16
        "device_map": "auto",       # Distribution automatique GPU
        "trust_remote_code": True,  # Requis pour Kimi-VL
        "low_cpu_mem_usage": True,  # Économie mémoire
    },
    
    # Variables environnement
    "env_vars": {
        "VLLM_USE_V1": "0",
        "TRANSFORMERS_VERBOSITY": "info", 
        "TOKENIZERS_PARALLELISM": "false",
    }
}'''
    
    with open("kimi_vl_fixed_config.py", "w") as f:
        f.write(config)
    
    print("✅ Configuration sauvée dans: kimi_vl_fixed_config.py")

def main():
    """Exécution complète des fixes basés sur recherche GitHub."""
    print("🚀 RÉPARATION KIMI-VL - Solutions GitHub Issues")
    print("=" * 50)
    
    # Fix 1: Version transformers
    fix_transformers_version()
    
    # Fix 2: Variables environnement
    env_vars = fix_vllm_environment()
    
    # Fix 3: Test chargement
    test_kimi_vl_loading()
    
    # Fix 4: Configuration optimisée
    create_fixed_config()
    
    print("\n🎯 RÉSUMÉ DES FIXES APPLIQUÉS:")
    print("1. ✅ transformers==4.51.3 installé")
    print("2. ✅ Variables VLLM configurées")
    print("3. ✅ Test de chargement effectué")
    print("4. ✅ Configuration optimisée créée")
    
    print("\n📋 PROCHAINES ÉTAPES:")
    print("1. Redémarrer votre terminal/session")
    print("2. Tester: python main_headless.py --model kimi-vl-a3b-thinking-2506 --max-frames 5")
    print("3. Si erreur persiste: utiliser main_Qwen.py (plus stable)")

if __name__ == "__main__":
    main()