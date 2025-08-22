#!/usr/bin/env python3
"""
🔧 Correction Automatique de Tous les Imports
============================================

Corrige automatiquement tous les problèmes d'imports
pour garantir le bon fonctionnement sur serveur GPU.
"""

import os
import re
from pathlib import Path

def fix_relative_imports():
    """Corrige tous les imports relatifs problématiques."""
    
    print("🔧 CORRECTION DES IMPORTS")
    print("=" * 40)
    
    # Corrections à appliquer
    corrections = {
        # Tests avec imports relatifs incorrects
        r'from core\.': 'from src.core.',
        r'from detection\.': 'from src.detection.',
        r'from utils\.': 'from src.utils.',
        r'from validation\.': 'from src.validation.',
        r'from testing\.': 'from src.testing.',
        r'from advanced_tools\.': 'from src.advanced_tools.',
        
        # Imports manquants de class_id dans Detection
        r'(\w+)\(bbox=': r'\1(class_id=0, class_name="unknown", bbox=',
    }
    
    # Fichiers à traiter
    files_to_fix = []
    
    # Recherche de tous les fichiers Python
    for root, dirs, files in os.walk('.'):
        # Ignorer venv et .git
        dirs[:] = [d for d in dirs if d not in ['venv', '.git', '__pycache__']]
        
        for file in files:
            if file.endswith('.py'):
                file_path = Path(root) / file
                files_to_fix.append(file_path)
    
    print(f"📁 {len(files_to_fix)} fichiers Python trouvés")
    
    fixed_files = 0
    
    for file_path in files_to_fix:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Appliquer les corrections
            for pattern, replacement in corrections.items():
                content = re.sub(pattern, replacement, content)
            
            # Si le fichier a été modifié
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"✅ Corrigé: {file_path}")
                fixed_files += 1
                
        except Exception as e:
            print(f"❌ Erreur {file_path}: {e}")
    
    print(f"\n🎉 {fixed_files} fichiers corrigés")

def ensure_detection_compatibility():
    """S'assure que tous les usages de Detection sont compatibles."""
    
    print("\n🔧 VÉRIFICATION COMPATIBILITÉ DETECTION")
    print("=" * 40)
    
    # Fichiers critiques à vérifier
    critical_files = [
        'src/core/types.py',
        'src/detection/yolo_detector.py',
        'src/detection/yolo/detector.py',
        'test_full_system_video.py',
        'main.py',
        'src/main.py'
    ]
    
    for file_path in critical_files:
        if Path(file_path).exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Vérifications
                has_detection = 'Detection' in content
                has_detectedobject = 'DetectedObject' in content
                has_class_id = 'class_id' in content
                has_class_name = 'class_name' in content
                
                status = "✅" if (has_detection or has_detectedobject) and has_class_id and has_class_name else "⚠️"
                print(f"{status} {file_path}: Detection={has_detection}, DetectedObject={has_detectedobject}")
                
            except Exception as e:
                print(f"❌ Erreur {file_path}: {e}")
        else:
            print(f"⚠️ Fichier manquant: {file_path}")

def create_missing_init_files():
    """Crée les fichiers __init__.py manquants."""
    
    print("\n🔧 CRÉATION FICHIERS __init__.py")
    print("=" * 40)
    
    # Dossiers qui doivent avoir un __init__.py
    required_dirs = [
        'src',
        'src/core',
        'src/core/vlm',
        'src/core/orchestrator',
        'src/detection',
        'src/detection/yolo',
        'src/detection/tracking',
        'src/utils',
        'src/validation',
        'src/testing',
        'src/advanced_tools',
        'tests',
        'examples'
    ]
    
    created = 0
    
    for dir_path in required_dirs:
        init_file = Path(dir_path) / '__init__.py'
        
        if not init_file.exists() and Path(dir_path).exists():
            try:
                init_file.write_text('"""Module initialization."""\n')
                print(f"✅ Créé: {init_file}")
                created += 1
            except Exception as e:
                print(f"❌ Erreur {init_file}: {e}")
    
    print(f"\n🎉 {created} fichiers __init__.py créés")

def fix_detection_instantiation():
    """Corrige les instanciations de Detection qui manquent class_id."""
    
    print("\n🔧 CORRECTION INSTANCIATIONS DETECTION")
    print("=" * 40)
    
    # Pattern pour détecter Detection() sans class_id
    pattern = r'Detection\s*\(\s*(?!class_id)'
    
    files_fixed = 0
    
    for root, dirs, files in os.walk('.'):
        dirs[:] = [d for d in dirs if d not in ['venv', '.git', '__pycache__']]
        
        for file in files:
            if file.endswith('.py'):
                file_path = Path(root) / file
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Chercher les problèmes
                    if re.search(pattern, content):
                        print(f"⚠️ Problème détecté dans: {file_path}")
                        
                        # Correction manuelle recommandée
                        lines = content.split('\n')
                        for i, line in enumerate(lines):
                            if re.search(pattern, line):
                                print(f"   Ligne {i+1}: {line.strip()}")
                
                except Exception as e:
                    continue
    
    print("💡 Vérifiez manuellement les lignes signalées")

def main():
    """Fonction principale."""
    
    print("🚀 CORRECTION AUTOMATIQUE DU SYSTÈME")
    print("=" * 50)
    
    # 1. Corriger les imports
    fix_relative_imports()
    
    # 2. Vérifier compatibilité Detection
    ensure_detection_compatibility()
    
    # 3. Créer __init__.py manquants
    create_missing_init_files()
    
    # 4. Vérifier instanciations Detection
    fix_detection_instantiation()
    
    print("\n" + "=" * 50)
    print("🎉 CORRECTIONS TERMINÉES !")
    print("\n💡 PROCHAINES ÉTAPES:")
    print("   1. bash setup_gpu_server.sh")
    print("   2. python check_gpu_system.py")
    print("   3. python run_gpu_tests.py")

if __name__ == "__main__":
    main()