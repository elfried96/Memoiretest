#!/usr/bin/env python3
"""
🔍 Vérification Système GPU avec UV
=================================

Script de diagnostic pour environnement UV
"""

import sys
import subprocess
import os
from pathlib import Path

# Ajout du chemin
sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

def check_uv_environment():
    """Vérification de l'environnement UV."""
    
    console.print(Panel.fit(
        "[bold blue]🔍 DIAGNOSTIC SYSTÈME UV + GPU[/bold blue]\n"
        "[dim]Vérification environnement UV pour surveillance[/dim]",
        border_style="blue"
    ))
    
    # Table des résultats
    results_table = Table(title="📊 État du Système UV")
    results_table.add_column("Composant", style="cyan", no_wrap=True)
    results_table.add_column("Status", style="bold")
    results_table.add_column("Détails", style="dim")
    
    # 1. Vérification UV
    try:
        result = subprocess.run(['uv', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            uv_version = result.stdout.strip()
            results_table.add_row(
                "UV Package Manager", 
                "✅ Installé", 
                uv_version
            )
        else:
            results_table.add_row(
                "UV Package Manager", 
                "❌ Non trouvé", 
                "Installer avec: curl -LsSf https://astral.sh/uv/install.sh | sh"
            )
    except FileNotFoundError:
        results_table.add_row(
            "UV Package Manager", 
            "❌ Non installé", 
            "Installer UV d'abord"
        )
    
    # 2. Environnement virtuel UV
    venv_path = Path(".venv")
    if venv_path.exists():
        results_table.add_row(
            "Environnement UV", 
            "✅ Créé", 
            f".venv présent"
        )
    else:
        results_table.add_row(
            "Environnement UV", 
            "❌ Manquant", 
            "Lancer: uv sync"
        )
    
    # 3. PyTorch avec UV
    try:
        import torch
        cuda_status = "GPU" if torch.cuda.is_available() else "CPU"
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            details = f"v{torch.__version__} - {gpu_name} ({gpu_memory:.1f}GB)"
        else:
            details = f"v{torch.__version__} (CPU mode)"
            
        results_table.add_row(
            "PyTorch", 
            "✅ Installé", 
            details
        )
    except ImportError:
        results_table.add_row(
            "PyTorch", 
            "❌ Manquant", 
            "uv add torch torchvision"
        )
    
    # 4. YOLO11 avec UV
    try:
        from ultralytics import YOLO
        model = YOLO('yolov11n.pt')
        results_table.add_row(
            "YOLO11", 
            "✅ Prêt", 
            "Modèle téléchargé"
        )
    except Exception as e:
        results_table.add_row(
            "YOLO11", 
            "❌ Problème", 
            str(e)[:50]
        )
    
    # 5. Types système
    try:
        from src.core.types import Detection, BoundingBox, AnalysisRequest
        results_table.add_row(
            "Types Système", 
            "✅ Importés", 
            "Detection, BoundingBox, etc."
        )
    except ImportError as e:
        results_table.add_row(
            "Types Système", 
            "❌ Erreur", 
            str(e)[:50]
        )
    
    # 6. Dépendances UV
    try:
        result = subprocess.run(['uv', 'tree', '--depth', '1'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            dep_count = len([l for l in result.stdout.split('\n') if l.strip()])
            results_table.add_row(
                "Dépendances UV", 
                "✅ Synchronisées", 
                f"~{dep_count} packages"
            )
        else:
            results_table.add_row(
                "Dépendances UV", 
                "⚠️ À vérifier", 
                "uv sync recommandé"
            )
    except:
        results_table.add_row(
            "Dépendances UV", 
            "❌ Erreur", 
            "Problème avec uv tree"
        )
    
    console.print(results_table)
    return results_table

def test_uv_commands():
    """Test des commandes UV."""
    
    console.print("\n🧪 Test des Commandes UV")
    
    commands_to_test = [
        ("uv run python --version", "Version Python UV"),
        ("uv run python -c 'import torch; print(f\"PyTorch: {torch.__version__}\")'", "PyTorch via UV"),
        ("uv run python -c 'from src.core.types import Detection; print(\"Types OK\")'", "Types via UV"),
    ]
    
    for cmd, description in commands_to_test:
        try:
            result = subprocess.run(cmd.split(), capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                console.print(f"✅ {description}: {result.stdout.strip()}")
            else:
                console.print(f"❌ {description}: {result.stderr.strip()}")
        except subprocess.TimeoutExpired:
            console.print(f"⏱️ {description}: Timeout")
        except Exception as e:
            console.print(f"❌ {description}: {e}")

def show_uv_usage():
    """Affiche l'utilisation d'UV pour le projet."""
    
    console.print("\n💡 Guide d'Utilisation UV")
    console.print("=" * 40)
    
    usage_table = Table(title="🛠️ Commandes UV Essentielles")
    usage_table.add_column("Commande", style="cyan")
    usage_table.add_column("Description", style="dim")
    
    usage_table.add_row("uv sync", "Synchroniser l'environnement")
    usage_table.add_row("uv add package", "Ajouter une dépendance")
    usage_table.add_row("uv remove package", "Supprimer une dépendance")
    usage_table.add_row("uv run python script.py", "Exécuter un script")
    usage_table.add_row("uv run pytest", "Lancer les tests")
    usage_table.add_row("uv tree", "Voir l'arbre des dépendances")
    usage_table.add_row("uv pip list", "Lister les packages")
    usage_table.add_row("uv shell", "Activer le shell")
    
    console.print(usage_table)

def main():
    """Fonction principale."""
    
    # 1. Diagnostic système
    results_table = check_uv_environment()
    
    # 2. Test commandes UV
    test_uv_commands()
    
    # 3. Guide d'utilisation
    show_uv_usage()
    
    # 4. Recommandations finales
    console.print("\n" + "="*60)
    console.print(Panel.fit(
        "[bold green]📋 RAPPORT UV FINAL[/bold green]",
        border_style="green"
    ))
    
    console.print("🎯 [bold green]COMMANDES RECOMMANDÉES AVEC UV:[/bold green]")
    console.print("")
    console.print("# Tests de base")
    console.print("uv run python test_basic_corrections.py")
    console.print("")
    console.print("# Test surveillance")
    console.print("uv run python test_full_system_video.py --video webcam --max-frames 50")
    console.print("")
    console.print("# Lancement production")
    console.print("uv run python main.py --video webcam")
    console.print("")
    console.print("# Tests complets")
    console.print("uv run python run_gpu_tests.py")
    
    console.print(f"\n⚡ [bold blue]AVANTAGES UV:[/bold blue]")
    console.print("• Installation 10-100x plus rapide que pip")
    console.print("• Résolution de dépendances garantie")
    console.print("• Environnements reproductibles")
    console.print("• Gestion de projet moderne")

if __name__ == "__main__":
    main()