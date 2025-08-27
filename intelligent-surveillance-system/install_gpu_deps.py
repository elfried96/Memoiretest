#!/usr/bin/env python3
"""
🚀 Installation Dépendances GPU
===============================

Script pour installer toutes les dépendances nécessaires
aux tests d'intégration GPU des modèles YOLO et SAM2.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description=""):
    """Exécute une commande et affiche le résultat"""
    print(f"🔧 {description}")
    print(f"   Commande: {cmd}")
    
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True,
            timeout=300  # 5 minutes timeout
        )
        
        if result.returncode == 0:
            print(f"   ✅ Succès")
            return True
        else:
            print(f"   ❌ Échec: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"   ⏰ Timeout après 5 minutes")
        return False
    except Exception as e:
        print(f"   💥 Erreur: {e}")
        return False

def check_virtual_env():
    """Vérifie si on est dans un environnement virtuel"""
    in_venv = hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )
    
    print(f"🐍 ENVIRONNEMENT PYTHON")
    print(f"   Python: {sys.version}")
    print(f"   Environnement virtuel: {'✅ Activé' if in_venv else '❌ Non activé'}")
    print(f"   Chemin Python: {sys.executable}")
    
    if not in_venv:
        print(f"   ⚠️  Recommandation: Activer un environnement virtuel")
        
    return in_venv

def install_basic_dependencies():
    """Installation des dépendances de base"""
    print(f"\n📦 INSTALLATION DÉPENDANCES DE BASE")
    print(f"=" * 50)
    
    dependencies = [
        ("numpy", "Calculs numériques"),
        ("opencv-python", "Traitement d'images"),
        ("Pillow", "Manipulation d'images"),
        ("loguru", "Logging avancé"),
        ("tqdm", "Barres de progression"),
        ("psutil", "Monitoring système")
    ]
    
    results = []
    
    for package, desc in dependencies:
        success = run_command(
            f"pip install {package}",
            f"Installation {desc} ({package})"
        )
        results.append((package, success))
    
    return results

def install_pytorch():
    """Installation de PyTorch avec support GPU/CPU"""
    print(f"\n🔥 INSTALLATION PYTORCH")
    print(f"=" * 50)
    
    # Détection CUDA
    cuda_available = False
    try:
        result = subprocess.run("nvidia-smi", capture_output=True, text=True)
        cuda_available = result.returncode == 0
    except:
        cuda_available = False
    
    print(f"   CUDA disponible: {'✅' if cuda_available else '❌'}")
    
    if cuda_available:
        # Installation avec support CUDA
        print("   🚀 Installation PyTorch avec CUDA...")
        cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
    else:
        # Installation CPU seulement
        print("   💻 Installation PyTorch CPU seulement...")
        cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
    
    return run_command(cmd, "Installation PyTorch")

def install_yolo_dependencies():
    """Installation des dépendances YOLO"""
    print(f"\n🎯 INSTALLATION DÉPENDANCES YOLO")
    print(f"=" * 50)
    
    dependencies = [
        ("ultralytics", "Framework YOLO"),
    ]
    
    results = []
    for package, desc in dependencies:
        success = run_command(
            f"pip install {package}",
            f"Installation {desc} ({package})"
        )
        results.append((package, success))
    
    return results

def install_sam2_dependencies():
    """Installation des dépendances SAM2"""  
    print(f"\n🎭 INSTALLATION DÉPENDANCES SAM2")
    print(f"=" * 50)
    
    # SAM2 nécessite des dépendances spécifiques
    dependencies = [
        ("segment-anything", "Modèle SAM base"),
        ("timm", "Modèles de vision"),
        ("transformers", "Transformers pour vision"),
        ("accelerate", "Accélération modèles"),
    ]
    
    results = []
    for package, desc in dependencies:
        success = run_command(
            f"pip install {package}",
            f"Installation {desc} ({package})"
        )
        results.append((package, success))
    
    return results

def test_installations():
    """Test des installations"""
    print(f"\n🧪 TEST DES INSTALLATIONS")
    print(f"=" * 50)
    
    tests = [
        ("import numpy as np; print(f'NumPy: {np.__version__}')", "NumPy"),
        ("import cv2; print(f'OpenCV: {cv2.__version__}')", "OpenCV"),
        ("import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')", "PyTorch"),
        ("import ultralytics; print(f'Ultralytics: {ultralytics.__version__}')", "YOLO"),
        ("from PIL import Image; print('Pillow: OK')", "Pillow"),
        ("import loguru; print('Loguru: OK')", "Loguru"),
    ]
    
    results = []
    
    for test_code, name in tests:
        try:
            result = subprocess.run(
                [sys.executable, "-c", test_code],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                print(f"   ✅ {name}: {result.stdout.strip()}")
                results.append((name, True, result.stdout.strip()))
            else:
                print(f"   ❌ {name}: {result.stderr.strip()}")
                results.append((name, False, result.stderr.strip()))
                
        except Exception as e:
            print(f"   💥 {name}: Erreur {e}")
            results.append((name, False, str(e)))
    
    return results

def generate_report(basic_results, pytorch_success, yolo_results, sam2_results, test_results):
    """Génère un rapport d'installation"""
    print(f"\n📊 RAPPORT D'INSTALLATION")
    print(f"=" * 60)
    
    # Statistiques globales
    total_packages = len(basic_results) + 1 + len(yolo_results) + len(sam2_results)
    successful_packages = sum([1 for _, success in basic_results if success]) + \
                         (1 if pytorch_success else 0) + \
                         sum([1 for _, success in yolo_results if success]) + \
                         sum([1 for _, success in sam2_results if success])
    
    print(f"📈 RÉSULTATS GLOBAUX:")
    print(f"   Packages installés: {successful_packages}/{total_packages}")
    print(f"   Taux de succès: {(successful_packages/total_packages*100):.1f}%")
    
    # Détails par catégorie
    print(f"\n🎯 DÉTAILS PAR CATÉGORIE:")
    
    print("   📦 Dépendances de base:")
    for package, success in basic_results:
        status = "✅" if success else "❌"
        print(f"      {status} {package}")
    
    print("   🔥 PyTorch:")
    status = "✅" if pytorch_success else "❌"
    print(f"      {status} torch")
    
    print("   🎯 YOLO:")
    for package, success in yolo_results:
        status = "✅" if success else "❌"
        print(f"      {status} {package}")
    
    print("   🎭 SAM2:")
    for package, success in sam2_results:
        status = "✅" if success else "❌"
        print(f"      {status} {package}")
    
    # Tests de validation
    print(f"\n✅ TESTS DE VALIDATION:")
    for name, success, details in test_results:
        status = "✅" if success else "❌"
        print(f"   {status} {name}: {details}")
    
    # Recommandations
    print(f"\n💡 PROCHAINES ÉTAPES:")
    if successful_packages == total_packages:
        print("   🎉 Installation complète réussie!")
        print("   ▶️  Exécuter: python run_integration_tests_gpu.py")
        print("   ▶️  Les tests GPU devraient maintenant fonctionner")
    else:
        print("   ⚠️  Certaines installations ont échoué")
        print("   🔧 Réessayer les packages échoués manuellement")
        print("   📚 Consulter la documentation des packages")
    
    return successful_packages == total_packages

def main():
    """Point d'entrée principal"""
    print("🚀 INSTALLATION DÉPENDANCES GPU POUR SURVEILLANCE IA")
    print("=" * 60)
    
    # Vérification environnement
    check_virtual_env()
    
    # Installations
    basic_results = install_basic_dependencies()
    pytorch_success = install_pytorch()
    yolo_results = install_yolo_dependencies()
    sam2_results = install_sam2_dependencies()
    
    # Tests
    test_results = test_installations()
    
    # Rapport
    success = generate_report(basic_results, pytorch_success, yolo_results, sam2_results, test_results)
    
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)