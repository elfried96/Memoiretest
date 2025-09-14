#!/usr/bin/env python3
"""
🖥️ Test Dashboard avec GPU - Vérification Fonctionnelle
======================================================

Script pour tester le dashboard en conditions réelles GPU
et collecter des métriques de performance utilisateur.
"""

import os
import sys
import subprocess
import time
import psutil
import torch
from pathlib import Path

def check_gpu_availability():
    """Vérification disponibilité GPU."""
    print("🔍 VÉRIFICATION ENVIRONNEMENT GPU")
    print("=" * 40)
    
    # Check CUDA
    cuda_available = torch.cuda.is_available()
    print(f"🔥 CUDA Disponible: {cuda_available}")
    
    if cuda_available:
        gpu_count = torch.cuda.device_count()
        print(f"🔥 Nombre GPU: {gpu_count}")
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory
            print(f"🔥 GPU {i}: {gpu_name}")
            print(f"🔥 Mémoire: {gpu_memory / 1e9:.1f} GB")
            
            # Test allocation mémoire
            try:
                test_tensor = torch.randn(1000, 1000, device=f'cuda:{i}')
                print(f"✅ GPU {i} fonctionnel")
                del test_tensor
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"❌ GPU {i} problème: {e}")
    else:
        print("⚠️  Aucun GPU CUDA détecté - Mode CPU")
    
    # Check système
    memory = psutil.virtual_memory()
    print(f"💾 RAM Système: {memory.total / 1e9:.1f} GB")
    print(f"💾 RAM Libre: {memory.available / 1e9:.1f} GB")
    
    return cuda_available

def test_imports():
    """Test imports des modules principaux."""
    print("\n🔍 TEST IMPORTS MODULES")
    print("=" * 30)
    
    modules_to_test = [
        'streamlit',
        'torch', 
        'transformers',
        'opencv-python',
        'numpy',
        'pandas',
        'plotly'
    ]
    
    success_count = 0
    for module in modules_to_test:
        try:
            __import__(module.replace('-', '_'))
            print(f"✅ {module}")
            success_count += 1
        except ImportError as e:
            print(f"❌ {module}: {e}")
    
    print(f"\n📊 Modules OK: {success_count}/{len(modules_to_test)}")
    return success_count == len(modules_to_test)

def check_project_structure():
    """Vérification structure projet."""
    print("\n📁 VÉRIFICATION STRUCTURE PROJET")
    print("=" * 40)
    
    required_files = [
        'dashboard/production_dashboard.py',
        'dashboard/video_context_integration.py',
        'dashboard/vlm_chatbot_symbiosis.py',
        'src/core/vlm/dynamic_model.py',
        'src/core/vlm/prompt_builder.py',
        'src/core/orchestrator/vlm_orchestrator.py'
    ]
    
    missing_files = []
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} MANQUANT")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n⚠️  {len(missing_files)} fichiers manquants:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    
    print("✅ Structure projet complète")
    return True

def launch_dashboard_test():
    """Lance le dashboard pour test utilisateur."""
    print("\n🚀 LANCEMENT DASHBOARD TEST")
    print("=" * 35)
    
    dashboard_path = Path("dashboard/production_dashboard.py")
    
    if not dashboard_path.exists():
        print("❌ Dashboard non trouvé")
        return False
    
    print("🌐 Lancement Streamlit Dashboard...")
    print("📍 URL: http://localhost:8501")
    print("\n🔧 INSTRUCTIONS TEST:")
    print("1. Ouvrir http://localhost:8501 dans le navigateur")
    print("2. Tester l'onglet '🎥 Surveillance VLM'")
    print("3. Tester l'onglet '📤 Upload Vidéo VLM'")
    print("4. Vérifier le chat VLM avec questions test")
    print("5. Mesurer temps de réponse chat")
    print("6. Tester upload vidéo avec descriptions contextuelles")
    print("\n⏹️  Ctrl+C pour arrêter le dashboard\n")
    
    try:
        # Lancement streamlit
        cmd = [
            sys.executable, "-m", "streamlit", "run", 
            str(dashboard_path),
            "--server.port", "8501",
            "--server.address", "localhost",
            "--server.headless", "false"
        ]
        
        process = subprocess.Popen(
            cmd, 
            cwd=Path.cwd(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        print("⏳ Attente démarrage dashboard...")
        time.sleep(5)
        
        print("✅ Dashboard démarré - Tests manuels possibles")
        print("💡 Presser ENTRÉE pour arrêter le dashboard...")
        
        # Attente input utilisateur
        input()
        
        # Arrêt propre
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        
        print("⏹️  Dashboard arrêté")
        return True
        
    except Exception as e:
        print(f"❌ Erreur lancement dashboard: {e}")
        return False

def generate_test_commands():
    """Génère commandes de test recommandées."""
    print("\n📋 COMMANDES TEST RECOMMANDÉES")
    print("=" * 40)
    
    commands = [
        {
            'description': '🧪 Test Performance Complet',
            'command': 'python run_performance_tests.py',
            'purpose': 'Métriques complètes pour mémoire'
        },
        {
            'description': '🎥 Test Contexte Vidéo',
            'command': 'python test_video_context_integration.py',
            'purpose': 'Validation intégration descriptions'
        },
        {
            'description': '🤖 Test Chatbot VLM',
            'command': 'python test_vlm_chatbot.py',
            'purpose': 'Test chatbot avec GPU'
        },
        {
            'description': '🖥️ Dashboard Production',
            'command': 'streamlit run dashboard/production_dashboard.py',
            'purpose': 'Interface utilisateur complète'
        }
    ]
    
    for i, cmd in enumerate(commands, 1):
        print(f"{i}. {cmd['description']}")
        print(f"   Command: {cmd['command']}")
        print(f"   But: {cmd['purpose']}")
        print()

def main():
    """Test principal."""
    print("🚀 TEST DASHBOARD GPU - ENVIRONNEMENT COMPLET")
    print("=" * 60)
    
    # Tests préliminaires
    gpu_ok = check_gpu_availability()
    imports_ok = test_imports()
    structure_ok = check_project_structure()
    
    print(f"\n📊 RÉSUMÉ VÉRIFICATIONS:")
    print(f"   🔥 GPU: {'✅' if gpu_ok else '❌'}")
    print(f"   📦 Imports: {'✅' if imports_ok else '❌'}")
    print(f"   📁 Structure: {'✅' if structure_ok else '❌'}")
    
    if not all([imports_ok, structure_ok]):
        print("\n❌ Tests préliminaires échoués")
        print("🔧 Corrigez les problèmes avant de continuer")
        return
    
    # Proposition tests
    print(f"\n🎯 ENVIRONNEMENT {'PRÊT' if gpu_ok else 'PARTIELLEMENT PRÊT'}")
    
    generate_test_commands()
    
    # Proposition lancement dashboard
    launch_dashboard = input("🚀 Lancer le dashboard pour test manuel ? (y/N): ")
    if launch_dashboard.lower() == 'y':
        launch_dashboard_test()
    
    print("\n✅ Tests environnement terminés")
    print("🎯 Vous pouvez maintenant lancer les tests de performance:")
    print("   python run_performance_tests.py")

if __name__ == "__main__":
    main()