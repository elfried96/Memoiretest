#!/usr/bin/env python3
"""
🧹 Nettoyage Sécurisé du Projet
===============================

Version sécurisée qui nettoie seulement les fichiers vraiment inutiles
sans toucher au code source ou aux dépendances importantes.
"""

import os
import sys
import shutil
from pathlib import Path
from datetime import datetime

def clean_python_cache():
    """Nettoie les fichiers cache Python."""
    print("🧹 Nettoyage cache Python...")
    
    patterns = [
        "**/__pycache__",
        "**/*.pyc", 
        "**/*.pyo",
        "**/*.pyd"
    ]
    
    removed = 0
    for pattern in patterns:
        for path in Path(".").glob(pattern):
            try:
                if path.is_file():
                    path.unlink()
                    removed += 1
                elif path.is_dir():
                    shutil.rmtree(path)
                    removed += 1
                print(f"   ✅ {path}")
            except Exception as e:
                print(f"   ❌ Erreur {path}: {e}")
    
    print(f"✅ Cache Python: {removed} éléments supprimés")

def clean_test_results():
    """Nettoie les anciens résultats de tests."""
    print("\n🧹 Nettoyage résultats tests...")
    
    patterns = [
        "performance_results_*.json",
        "real_metrics_*.json", 
        "evaluation_results_*.json",
        "complete_system_video_tests_*.json",
        "real_vlm_test_results_*.json",
        "cleanup_report.json"
    ]
    
    removed = 0
    for pattern in patterns:
        for path in Path(".").glob(pattern):
            try:
                path.unlink()
                removed += 1
                print(f"   ✅ {path}")
            except Exception as e:
                print(f"   ❌ Erreur {path}: {e}")
    
    print(f"✅ Résultats tests: {removed} fichiers supprimés")

def clean_system_files():
    """Nettoie les fichiers système."""
    print("\n🧹 Nettoyage fichiers système...")
    
    patterns = [
        "**/.DS_Store",
        "**/Thumbs.db",
        "**/*.tmp",
        "**/*.temp",
        "**/*.bak",
        "**/*.swp",
        "**/*.swo"
    ]
    
    removed = 0
    for pattern in patterns:
        for path in Path(".").glob(pattern):
            try:
                path.unlink()
                removed += 1
                print(f"   ✅ {path}")
            except Exception as e:
                print(f"   ❌ Erreur {path}: {e}")
    
    print(f"✅ Fichiers système: {removed} fichiers supprimés")

def clean_logs():
    """Nettoie les logs."""
    print("\n🧹 Nettoyage logs...")
    
    patterns = [
        "**/*.log",
        "**/logs"
    ]
    
    removed = 0
    for pattern in patterns:
        for path in Path(".").glob(pattern):
            try:
                if path.is_file():
                    path.unlink()
                    removed += 1
                elif path.is_dir():
                    shutil.rmtree(path)
                    removed += 1
                print(f"   ✅ {path}")
            except Exception as e:
                print(f"   ❌ Erreur {path}: {e}")
    
    print(f"✅ Logs: {removed} éléments supprimés")

def remove_duplicate_files():
    """Supprime les fichiers dupliqués identifiés."""
    print("\n🧹 Suppression fichiers dupliqués/obsolètes...")
    
    duplicates = [
        # Tests obsolètes
        "collect_real_metrics.py",  # Intégré dans run_real_vlm_tests.py
        "test_dashboard_gpu.py",    # Fonctionnalité dans run_performance_tests.py
        "evaluate_system.py",
        "real_evaluation_guide.py",
        
        # Plans d'évaluation
        "evaluation_plan_*.json"
    ]
    
    removed = 0
    for duplicate in duplicates:
        for path in Path(".").glob(duplicate):
            try:
                path.unlink()
                removed += 1
                print(f"   ✅ {path} (obsolète)")
            except Exception as e:
                print(f"   ❌ Erreur {path}: {e}")
    
    print(f"✅ Fichiers dupliqués: {removed} fichiers supprimés")

def clean_empty_directories():
    """Supprime les dossiers vides."""
    print("\n🧹 Suppression dossiers vides...")
    
    removed = 0
    # Parcours récursif des dossiers
    for root, dirs, files in os.walk(".", topdown=False):
        for dir_name in dirs:
            dir_path = Path(root) / dir_name
            try:
                # Skip dossiers importants
                if any(important in str(dir_path) for important in ['.venv', '.git', 'node_modules']):
                    continue
                    
                if dir_path.is_dir() and not any(dir_path.iterdir()):
                    dir_path.rmdir()
                    removed += 1
                    print(f"   ✅ {dir_path}/ (vide)")
            except Exception as e:
                pass  # Ignore erreurs dossiers non vides
    
    print(f"✅ Dossiers vides: {removed} dossiers supprimés")

def show_project_structure():
    """Affiche la structure projet nettoyée."""
    print("\n📁 STRUCTURE PROJET APRÈS NETTOYAGE:")
    print("=" * 40)
    
    important_paths = [
        "src/",
        "dashboard/",
        "tests/",
        "README.md",
        "pyproject.toml"
    ]
    
    for path_str in important_paths:
        path = Path(path_str)
        if path.exists():
            if path.is_dir():
                file_count = len(list(path.rglob("*.py")))
                print(f"✅ {path}/ ({file_count} fichiers Python)")
            else:
                print(f"✅ {path}")
        else:
            print(f"⚠️  {path} (manquant)")

def main():
    """Nettoyage principal sécurisé."""
    print("🧹 NETTOYAGE SÉCURISÉ DU PROJET")
    print("=" * 45)
    print("Ce script nettoie seulement les fichiers cache et temporaires")
    print("AUCUN code source ne sera supprimé")
    print()
    
    # Confirmation (auto en mode non-interactif)
    if len(sys.argv) > 1 and "--auto" in sys.argv:
        print("Mode automatique activé")
    else:
        try:
            confirm = input("Continuer le nettoyage ? (oui/non): ").lower()
            if confirm not in ['oui', 'o', 'yes', 'y']:
                print("❌ Nettoyage annulé")
                return
        except EOFError:
            print("Mode automatique (non-interactif)")
            # Continue automatiquement
    
    start_time = datetime.now()
    
    try:
        # Nettoyages sécurisés
        clean_python_cache()
        clean_test_results()
        clean_system_files()
        clean_logs()
        remove_duplicate_files()
        clean_empty_directories()
        
        # Affichage structure finale
        show_project_structure()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"\n🎉 NETTOYAGE TERMINÉ!")
        print(f"⏱️  Durée: {duration:.1f}s")
        print(f"📁 Projet nettoyé et optimisé")
        
        # Recommandations
        print(f"\n💡 RECOMMANDATIONS POST-NETTOYAGE:")
        print(f"1. Vérifiez que tout fonctionne: python run_performance_tests.py")
        print(f"2. Testez le dashboard: streamlit run dashboard/production_dashboard.py")
        print(f"3. Si problèmes, régénérez le cache: python -m py_compile [fichier.py]")
        
    except Exception as e:
        print(f"\n❌ Erreur durant nettoyage: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()