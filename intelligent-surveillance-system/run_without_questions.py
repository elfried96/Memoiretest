#!/usr/bin/env python3
"""
Script pour éviter les questions HuggingFace
==========================================

Lance main_headless.py sans questions de sécurité
"""

import os
import subprocess
import sys

def setup_environment():
    """Configure environnement pour éviter questions."""
    print("🔧 Configuration environnement sans questions...")
    
    # Variables pour éviter questions HuggingFace
    env_vars = {
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",  # Pas de barres de progression
        "TRANSFORMERS_VERBOSITY": "error",     # Moins de warnings
        "TOKENIZERS_PARALLELISM": "false",     # Évite warnings
        "PYTHONIOENCODING": "utf-8",          # Évite problèmes encoding
    }
    
    # Appliquer variables
    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"✅ {key}={value}")

def run_headless_auto_yes():
    """Lance main_headless.py avec réponses automatiques."""
    print("\n🚀 Lancement main_headless.py avec auto-accept...")
    
    # Commande à exécuter
    cmd = [
        sys.executable, "main_headless.py",
        "--model", "kimi-vl-a3b-thinking-2506",
        "--video", "videos/surveillance01.mp4", 
        "--max-frames", "5",
        "--vlm-mode", "smart",
        "--frame-skip", "3"
    ]
    
    print(f"📋 Commande: {' '.join(cmd)}")
    
    # Lancer avec input automatique "y\ny\n"
    try:
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Envoyer "y" automatiquement pour les questions
        auto_responses = "y\ny\ny\ny\n"  # 4x "yes" au cas où
        
        stdout, stderr = process.communicate(input=auto_responses)
        
        print("📊 SORTIE:")
        print(stdout)
        
        if process.returncode == 0:
            print("✅ Exécution réussie !")
        else:
            print(f"❌ Erreur (code {process.returncode})")
            
    except KeyboardInterrupt:
        print("🛑 Interrompu par utilisateur")
        process.terminate()
    except Exception as e:
        print(f"❌ Erreur: {e}")

def create_expect_script():
    """Crée script expect pour automatiser les réponses."""
    expect_script = '''#!/usr/bin/expect -f

# Script expect pour automatiser les réponses "y"
set timeout -1

# Lancer la commande
spawn python main_headless.py --model kimi-vl-a3b-thinking-2506 --video videos/surveillance01.mp4 --max-frames 5 --vlm-mode smart --frame-skip 3

# Répondre automatiquement aux questions
expect {
    "Do you wish to run the custom code? \\[y/N\\]" {
        send "y\\r"
        exp_continue
    }
    "trust_remote_code=True" {
        exp_continue  
    }
    eof
}
'''
    
    with open("auto_kimi.expect", "w") as f:
        f.write(expect_script)
    
    # Rendre exécutable
    os.chmod("auto_kimi.expect", 0o755)
    print("✅ Script expect créé: auto_kimi.expect")

def main():
    """Point d'entrée principal."""
    print("🤖 LAUNCHER SANS QUESTIONS HUGGINGFACE")
    print("=" * 40)
    
    # Configuration environnement
    setup_environment()
    
    # Créer script expect
    create_expect_script()
    
    print("\n🎯 MÉTHODES DISPONIBLES:")
    print("1. 🐍 Python automatique:")
    print("   python run_without_questions.py")
    
    print("\n2. 📜 Script expect (si installé):")
    print("   ./auto_kimi.expect")
    
    print("\n3. ⌨️ Réponses manuelles:")
    print("   Tapez 'y' quand demandé")
    
    print("\n💡 RECOMMANDATION:")
    print("Utilisez Qwen2-VL (pas de questions):")
    print("python main_Qwen.py --video videos/surveillance01.mp4 --max-frames 5")
    
    # Lancer automatiquement sans prompt
    print("\n🚀 Lancement automatique activé...")
    run_headless_auto_yes()

if __name__ == "__main__":
    main()