#!/usr/bin/env python3
"""
🧹 Nettoyage Projet - Suppression Fichiers Inutiles
==================================================

Script pour identifier et supprimer les fichiers inutiles du projet
de surveillance intelligente.
"""

import os
import sys
import shutil
from pathlib import Path
from datetime import datetime
import json

class ProjectCleaner:
    """Nettoyeur de projet intelligent."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.files_to_remove = []
        self.dirs_to_remove = []
        self.files_to_keep = []
        
        # Fichiers essentiels à GARDER
        self.essential_files = {
            # Core système
            "src/core/vlm/dynamic_model.py",
            "src/core/vlm/prompt_builder.py", 
            "src/core/orchestrator/vlm_orchestrator.py",
            "src/core/types.py",
            
            # Dashboard principal
            "dashboard/production_dashboard.py",
            "dashboard/video_context_integration.py",
            "dashboard/vlm_chatbot_symbiosis.py",
            "dashboard/vlm_chatbot_optimizations.py",
            "dashboard/vlm_chatbot_advanced_features.py",
            
            # Configuration
            "pyproject.toml",
            "requirements.txt",
            
            # Documentation
            "README.md",
            "VLM_CHATBOT_USER_GUIDE.md",
            
            # Tests principaux
            "run_real_vlm_tests.py",
            "test_complete_system_videos.py",
            "run_performance_tests.py",
            "test_video_context_integration.py",
            
            # Scripts utilitaires
            "start_vlm_chatbot.sh",
        }
        
        # Patterns de fichiers à SUPPRIMER
        self.patterns_to_remove = [
            # Fichiers temporaires
            "**/*.pyc",
            "**/*.pyo", 
            "**/*.pyd",
            "**/__pycache__",
            "**/.pytest_cache",
            "**/.coverage",
            "**/htmlcov",
            
            # Fichiers système
            "**/.DS_Store",
            "**/Thumbs.db",
            "**/*.tmp",
            "**/*.temp",
            "**/*.bak",
            "**/*.swp",
            "**/*.swo",
            
            # Logs et cache
            "**/*.log",
            "**/logs",
            "**/.cache",
            "**/cache",
            
            # IDE
            "**/.vscode",
            "**/.idea",
            "**/*.iml",
            
            # Résultats tests (anciens)
            "**/performance_results_*.json",
            "**/real_metrics_*.json",
            "**/evaluation_results_*.json",
            "**/complete_system_video_tests_*.json",
        ]
        
        # Dossiers probablement inutiles
        self.dirs_to_check = [
            "test_videos",
            "logs",
            "cache", 
            "tmp",
            "temp",
            "__pycache__",
            ".pytest_cache",
            "htmlcov",
            ".coverage",
            "build",
            "dist",
            "*.egg-info"
        ]
    
    def scan_project(self):
        """Scan complet du projet pour identifier fichiers inutiles."""
        print("🔍 SCAN PROJET POUR NETTOYAGE")
        print("=" * 40)
        
        # Scan par patterns
        for pattern in self.patterns_to_remove:
            matches = list(self.project_root.glob(pattern))
            for match in matches:
                if match.is_file():
                    self.files_to_remove.append(match)
                elif match.is_dir():
                    self.dirs_to_remove.append(match)
        
        # Scan dossiers spécifiques
        for dir_pattern in self.dirs_to_check:
            matches = list(self.project_root.glob(f"**/{dir_pattern}"))
            for match in matches:
                if match.is_dir():
                    self.dirs_to_remove.append(match)
        
        # Identification fichiers dupliqués/obsolètes
        self._identify_duplicate_files()
        
        # Fichiers probablement obsolètes
        self._identify_obsolete_files()
        
        print(f"📊 Résultats scan:")
        print(f"   🗑️  Fichiers à supprimer: {len(self.files_to_remove)}")
        print(f"   📁 Dossiers à supprimer: {len(self.dirs_to_remove)}")
    
    def _identify_duplicate_files(self):
        """Identification fichiers dupliqués."""
        duplicates = [
            # Tests dupliqués
            "test_vlm_extended.py",  # Remplacé par run_real_vlm_tests.py
            "test_vlm_chatbot.py",   # Remplacé par vlm_chatbot_symbiosis.py
            "collect_real_metrics.py", # Intégré dans run_real_vlm_tests.py
            "test_dashboard_gpu.py",    # Fonctionnalité dans run_performance_tests.py
            
            # Scripts obsolètes
            "evaluate_system.py",
            "real_evaluation_guide.py",
            
            # Fichiers config obsolètes
            "evaluation_plan_*.json",
        ]
        
        for duplicate in duplicates:
            matches = list(self.project_root.glob(f"**/{duplicate}"))
            self.files_to_remove.extend(matches)
    
    def _identify_obsolete_files(self):
        """Identification fichiers obsolètes."""
        # Parcours tous les fichiers Python
        py_files = list(self.project_root.glob("**/*.py"))
        
        for py_file in py_files:
            relative_path = py_file.relative_to(self.project_root)
            
            # Skip fichiers essentiels
            if str(relative_path) in self.essential_files:
                self.files_to_keep.append(py_file)
                continue
            
            # Fichiers tests anciens
            if "test_" in py_file.name and "unit" not in str(py_file) and py_file not in self.files_to_remove:
                # Vérifier si fichier test obsolète
                if self._is_obsolete_test_file(py_file):
                    self.files_to_remove.append(py_file)
            
            # Fichiers avec noms génériques suspects
            if py_file.name in ["temp.py", "test.py", "debug.py", "old.py"]:
                self.files_to_remove.append(py_file)
    
    def _is_obsolete_test_file(self, file_path: Path) -> bool:
        """Vérifie si un fichier test est obsolète."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Indicateurs de fichier obsolète
            obsolete_indicators = [
                "# TODO: Remove",
                "# DEPRECATED",
                "# OLD VERSION",
                "# OBSOLETE",
                "def old_",
                "def deprecated_"
            ]
            
            for indicator in obsolete_indicators:
                if indicator in content:
                    return True
            
            # Fichier très court (probablement vide/test)
            if len(content.strip()) < 100:
                return True
                
        except Exception:
            pass
        
        return False
    
    def preview_cleanup(self):
        """Aperçu des fichiers qui seront supprimés."""
        print("\n📋 APERÇU NETTOYAGE")
        print("=" * 30)
        
        if self.files_to_remove:
            print(f"\n🗑️  FICHIERS À SUPPRIMER ({len(self.files_to_remove)}):")
            for file_path in sorted(set(self.files_to_remove)):
                relative_path = file_path.relative_to(self.project_root)
                size = file_path.stat().st_size if file_path.exists() else 0
                print(f"   - {relative_path} ({size} bytes)")
        
        if self.dirs_to_remove:
            print(f"\n📁 DOSSIERS À SUPPRIMER ({len(self.dirs_to_remove)}):")
            for dir_path in sorted(set(self.dirs_to_remove)):
                relative_path = dir_path.relative_to(self.project_root)
                print(f"   - {relative_path}/")
        
        print(f"\n✅ FICHIERS ESSENTIELS GARDÉS ({len(self.files_to_keep)}):")
        for file_path in sorted(self.files_to_keep)[:10]:  # Affiche les 10 premiers
            relative_path = file_path.relative_to(self.project_root)
            print(f"   - {relative_path}")
        if len(self.files_to_keep) > 10:
            print(f"   ... et {len(self.files_to_keep) - 10} autres fichiers essentiels")
    
    def calculate_cleanup_impact(self):
        """Calcule l'impact du nettoyage."""
        total_size_to_remove = 0
        files_count = 0
        dirs_count = 0
        
        for file_path in self.files_to_remove:
            if file_path.exists() and file_path.is_file():
                total_size_to_remove += file_path.stat().st_size
                files_count += 1
        
        for dir_path in self.dirs_to_remove:
            if dir_path.exists() and dir_path.is_dir():
                dirs_count += 1
                for item in dir_path.rglob("*"):
                    if item.is_file():
                        total_size_to_remove += item.stat().st_size
        
        print(f"\n📊 IMPACT NETTOYAGE:")
        print(f"   🗑️  Fichiers: {files_count}")
        print(f"   📁 Dossiers: {dirs_count}")
        print(f"   💾 Espace libéré: {total_size_to_remove / 1024:.1f} KB")
        
        return files_count, dirs_count, total_size_to_remove
    
    def perform_cleanup(self, confirm: bool = False):
        """Exécute le nettoyage."""
        if not confirm:
            print("\n⚠️  NETTOYAGE EN MODE APERÇU SEULEMENT")
            print("    Utilisez confirm=True pour supprimer réellement")
            return
        
        print(f"\n🧹 EXÉCUTION NETTOYAGE")
        print("=" * 30)
        
        removed_files = 0
        removed_dirs = 0
        errors = []
        
        # Suppression fichiers
        for file_path in self.files_to_remove:
            try:
                if file_path.exists():
                    file_path.unlink()
                    removed_files += 1
                    print(f"   🗑️  {file_path.name}")
            except Exception as e:
                errors.append(f"Fichier {file_path}: {e}")
        
        # Suppression dossiers
        for dir_path in self.dirs_to_remove:
            try:
                if dir_path.exists():
                    shutil.rmtree(dir_path)
                    removed_dirs += 1
                    print(f"   📁 {dir_path.name}/")
            except Exception as e:
                errors.append(f"Dossier {dir_path}: {e}")
        
        print(f"\n✅ NETTOYAGE TERMINÉ:")
        print(f"   🗑️  {removed_files} fichiers supprimés")
        print(f"   📁 {removed_dirs} dossiers supprimés")
        
        if errors:
            print(f"   ⚠️  {len(errors)} erreurs:")
            for error in errors[:5]:  # Max 5 erreurs
                print(f"      - {error}")
    
    def create_backup(self):
        """Crée une sauvegarde avant nettoyage."""
        backup_name = f"backup_before_cleanup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        backup_path = self.project_root.parent / backup_name
        
        print(f"💾 Création sauvegarde: {backup_path}")
        
        try:
            shutil.copytree(self.project_root, backup_path, 
                          ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git'))
            print(f"✅ Sauvegarde créée: {backup_path}")
            return str(backup_path)
        except Exception as e:
            print(f"❌ Erreur sauvegarde: {e}")
            return None
    
    def generate_cleanup_report(self):
        """Génère rapport de nettoyage."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'files_to_remove': [str(f.relative_to(self.project_root)) for f in self.files_to_remove],
            'dirs_to_remove': [str(d.relative_to(self.project_root)) for d in self.dirs_to_remove],
            'essential_files_kept': [str(f.relative_to(self.project_root)) for f in self.files_to_keep],
            'cleanup_impact': {
                'files_count': len(self.files_to_remove),
                'dirs_count': len(self.dirs_to_remove)
            }
        }
        
        report_file = self.project_root / "cleanup_report.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"📄 Rapport sauvegardé: {report_file}")
        return str(report_file)

def main():
    """Nettoyage principal."""
    print("🧹 NETTOYAGE PROJET INTELLIGENT")
    print("=" * 45)
    
    cleaner = ProjectCleaner()
    
    # 1. Scan projet
    cleaner.scan_project()
    
    # 2. Aperçu nettoyage
    cleaner.preview_cleanup()
    
    # 3. Calcul impact
    files_count, dirs_count, size_freed = cleaner.calculate_cleanup_impact()
    
    # 4. Génération rapport
    report_file = cleaner.generate_cleanup_report()
    
    print(f"\n🎯 RECOMMANDATIONS:")
    print(f"1. Examinez le rapport: cleanup_report.json")
    print(f"2. Créez une sauvegarde si nécessaire")
    print(f"3. Exécutez le nettoyage:")
    print(f"   - Mode aperçu: python cleanup_project.py")
    print(f"   - Mode réel: python cleanup_project.py --confirm")
    
    # Options ligne de commande
    if len(sys.argv) > 1 and "--confirm" in sys.argv:
        print(f"\n⚠️  MODE CONFIRMATION DÉTECTÉ")
        
        # Création sauvegarde
        if "--backup" in sys.argv:
            backup_path = cleaner.create_backup()
            if not backup_path:
                print("❌ Sauvegarde échouée - Nettoyage annulé")
                return
        
        confirm = input("Confirmer le nettoyage ? (oui/non): ").lower()
        if confirm in ['oui', 'o', 'yes', 'y']:
            cleaner.perform_cleanup(confirm=True)
        else:
            print("❌ Nettoyage annulé")
    else:
        print(f"\n💡 Mode aperçu - Aucun fichier supprimé")

if __name__ == "__main__":
    main()