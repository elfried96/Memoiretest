#!/usr/bin/env python3
"""
Script de libération mémoire GPU pour Kimi-VL
===========================================

Résout le problème "parameters offloaded to cpu"
"""

import torch
import gc
import subprocess
import sys
import os

def check_gpu_memory():
    """Vérifier état mémoire GPU."""
    if torch.cuda.is_available():
        print(f"🔍 GPU disponible: {torch.cuda.get_device_name()}")
        print(f"💾 Mémoire totale: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"💾 Mémoire allouée: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
        print(f"💾 Mémoire réservée: {torch.cuda.memory_reserved() / 1e9:.1f} GB")
        print(f"💾 Mémoire libre: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / 1e9:.1f} GB")
    else:
        print("❌ Aucun GPU disponible")

def clear_gpu_cache():
    """Nettoyer cache GPU."""
    print("\n🧹 Nettoyage cache GPU...")
    
    if torch.cuda.is_available():
        # Vider cache PyTorch
        torch.cuda.empty_cache()
        
        # Garbage collection Python
        gc.collect()
        
        print("✅ Cache GPU vidé")
    else:
        print("❌ Pas de GPU à nettoyer")

def kill_gpu_processes():
    """Tuer processus GPU qui consomment."""
    print("\n🔧 Vérification processus GPU...")
    
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("📊 Processus GPU actifs:")
            print(result.stdout)
            
            # Optionnel: tuer processus Python gourmands
            try:
                subprocess.run(['pkill', '-f', 'python.*kimi'], check=False)
                print("✅ Processus Kimi-VL terminés")
            except:
                pass
        else:
            print("❌ nvidia-smi non disponible")
    except FileNotFoundError:
        print("❌ nvidia-smi non trouvé")

def optimize_environment_variables():
    """Optimiser variables environnement pour GPU."""
    print("\n⚙️ Optimisation variables environnement...")
    
    # Variables pour économiser mémoire
    env_vars = {
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512",  # Limite fragmentation
        "CUDA_LAUNCH_BLOCKING": "0",  # Async pour vitesse
        "TORCH_USE_CUDA_DSA": "1",    # Debug allocation
        "TOKENIZERS_PARALLELISM": "false",  # Économie mémoire
        "OMP_NUM_THREADS": "4",       # Limite threads CPU
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"✅ {key}={value}")

def suggest_optimizations():
    """Suggestions d'optimisation."""
    print("\n💡 SUGGESTIONS OPTIMISATION:")
    print("1. 🚀 Utiliser modèle plus petit:")
    print("   python main_Qwen.py (7B au lieu de 27B)")
    
    print("\n2. ⚡ Réduire paramètres:")
    print("   --max-frames 3 --frame-skip 5")
    
    print("\n3. 💾 Mode économique:")
    print("   --vlm-mode smart --summary-interval 60")
    
    print("\n4. 🔧 Si problème persiste:")
    print("   Redémarrer: sudo reboot")

def main():
    """Diagnostic complet et nettoyage."""
    print("🔍 DIAGNOSTIC MÉMOIRE GPU KIMI-VL")
    print("=" * 40)
    
    # Étape 1: Vérifier état
    check_gpu_memory()
    
    # Étape 2: Nettoyer
    clear_gpu_cache()
    
    # Étape 3: Tuer processus
    kill_gpu_processes()
    
    # Étape 4: Optimiser environnement
    optimize_environment_variables()
    
    # Étape 5: Vérifier après nettoyage
    print("\n📊 ÉTAT APRÈS NETTOYAGE:")
    check_gpu_memory()
    
    # Étape 6: Suggestions
    suggest_optimizations()
    
    print("\n🎯 COMMANDES RECOMMANDÉES:")
    print("# Test rapide Kimi-VL optimisé")
    print("python main_headless.py --model kimi-vl-a3b-thinking-2506 --video videos/surveillance01.mp4 --max-frames 3 --frame-skip 5")
    
    print("\n# Alternative stable Qwen (plus léger)")  
    print("python main_Qwen.py --video videos/surveillance01.mp4 --max-frames 5 --vlm-mode smart")

if __name__ == "__main__":
    main()