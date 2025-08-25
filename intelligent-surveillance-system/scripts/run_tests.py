#!/usr/bin/env python3
"""Script centralisé pour exécuter les tests du système de surveillance."""

import sys
import subprocess
import argparse
from pathlib import Path
from typing import List, Optional
import os

# Configuration du path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

def run_command(cmd: List[str], description: str) -> bool:
    """Exécuter une commande et retourner le succès."""
    print(f"\n🔄 {description}")
    print(f"Commande: {' '.join(cmd)}")
    print("=" * 50)
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode == 0:
        print(f"✅ {description} - SUCCÈS")
        return True
    else:
        print(f"❌ {description} - ÉCHEC (code: {result.returncode})")
        return False

def run_unit_tests(verbose: bool = True, coverage: bool = False) -> bool:
    """Exécuter les tests unitaires."""
    cmd = ["pytest", "tests/unit/"]
    
    if verbose:
        cmd.append("-v")
    
    if coverage:
        cmd.extend(["--cov=src", "--cov-report=term-missing"])
    
    return run_command(cmd, "Tests Unitaires")

def run_integration_tests(verbose: bool = True) -> bool:
    """Exécuter les tests d'intégration."""
    cmd = ["pytest", "tests/integration/"]
    
    if verbose:
        cmd.append("-v")
    
    return run_command(cmd, "Tests d'Intégration")

def run_performance_tests(verbose: bool = True) -> bool:
    """Exécuter les tests de performance."""
    cmd = ["pytest", "tests/performance/"]
    
    if verbose:
        cmd.append("-v")
    
    return run_command(cmd, "Tests de Performance")

def run_all_tests(verbose: bool = True, coverage: bool = False, exclude_slow: bool = False) -> bool:
    """Exécuter tous les tests."""
    cmd = ["pytest", "tests/"]
    
    if verbose:
        cmd.append("-v")
    
    if coverage:
        cmd.extend(["--cov=src", "--cov-report=term-missing", "--cov-report=html"])
    
    if exclude_slow:
        cmd.extend(["-m", "not slow"])
    
    return run_command(cmd, "Tous les Tests")

def run_specific_test(test_path: str, verbose: bool = True) -> bool:
    """Exécuter un test spécifique."""
    cmd = ["pytest", test_path]
    
    if verbose:
        cmd.append("-v")
    
    return run_command(cmd, f"Test Spécifique: {test_path}")

def setup_test_environment() -> None:
    """Configuration de l'environnement de test."""
    print("🔧 Configuration de l'environnement de test...")
    
    # Variables d'environnement pour tests (modèle unique)
    test_env = {
        "TEST_MODE": "true",
        "LOG_LEVEL": "DEBUG",
        "VLM_MODEL": "microsoft/git-base-coco",  # Modèle léger uniquement
        "YOLO_MODEL": "yolov11n.pt",
        "ORCHESTRATION_MODE": "fast",
        "DISABLE_MONITORING": "true",
        "ENABLE_FALLBACK": "false",  # Pas de fallback pour économiser mémoire
        "ENV": "testing"  # Profil de configuration optimisé tests
    }
    
    for key, value in test_env.items():
        os.environ[key] = value
        print(f"  {key}={value}")
    
    print("✅ Environnement de test configuré")

def create_argument_parser() -> argparse.ArgumentParser:
    """Créer le parser d'arguments."""
    parser = argparse.ArgumentParser(
        description="Script de tests pour le système de surveillance intelligente",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  python scripts/run_tests.py --unit                    # Tests unitaires seulement
  python scripts/run_tests.py --integration             # Tests d'intégration seulement  
  python scripts/run_tests.py --all --coverage          # Tous les tests avec couverture
  python scripts/run_tests.py --all --exclude-slow      # Éviter les tests lents
  python scripts/run_tests.py --test tests/unit/test_vlm_unit.py  # Test spécifique
  python scripts/run_tests.py --performance             # Tests de performance
        """
    )
    
    # Types de tests
    parser.add_argument(
        "--unit", 
        action="store_true",
        help="Exécuter les tests unitaires"
    )
    
    parser.add_argument(
        "--integration",
        action="store_true",
        help="Exécuter les tests d'intégration"
    )
    
    parser.add_argument(
        "--performance",
        action="store_true", 
        help="Exécuter les tests de performance"
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Exécuter tous les tests"
    )
    
    parser.add_argument(
        "--test",
        type=str,
        help="Exécuter un test spécifique (chemin du fichier)"
    )
    
    # Options
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Générer un rapport de couverture de code"
    )
    
    parser.add_argument(
        "--exclude-slow",
        action="store_true",
        help="Exclure les tests marqués comme lents"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Mode silencieux (pas de verbose)"
    )
    
    parser.add_argument(
        "--no-setup",
        action="store_true",
        help="Ne pas configurer l'environnement de test"
    )
    
    return parser

def main():
    """Point d'entrée principal."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Configuration de l'environnement
    if not args.no_setup:
        setup_test_environment()
    
    verbose = not args.quiet
    success = True
    
    # Changer vers le répertoire du projet
    os.chdir(PROJECT_ROOT)
    
    print(f"\n🎯 Système de Surveillance Intelligente - Tests")
    print(f"Répertoire de travail: {PROJECT_ROOT}")
    print("=" * 60)
    
    try:
        if args.unit:
            success = run_unit_tests(verbose, args.coverage)
        
        elif args.integration:
            success = run_integration_tests(verbose)
        
        elif args.performance:
            success = run_performance_tests(verbose)
        
        elif args.all:
            success = run_all_tests(verbose, args.coverage, args.exclude_slow)
        
        elif args.test:
            success = run_specific_test(args.test, verbose)
        
        else:
            # Par défaut : tests unitaires
            print("Aucun type de test spécifié, exécution des tests unitaires...")
            success = run_unit_tests(verbose, args.coverage)
        
        # Résumé final
        print("\n" + "=" * 60)
        if success:
            print("🎉 TOUS LES TESTS ONT RÉUSSI !")
            if args.coverage:
                print("📊 Rapport de couverture généré dans htmlcov/index.html")
        else:
            print("💥 DES TESTS ONT ÉCHOUÉ !")
            print("Consultez les logs ci-dessus pour plus de détails.")
        
        sys.exit(0 if success else 1)
    
    except KeyboardInterrupt:
        print("\n⚠️ Interruption utilisateur")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erreur inattendue: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()