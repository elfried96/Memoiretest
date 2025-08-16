#!/usr/bin/env python3
"""Point d'entrée principal du système de surveillance intelligente."""

import asyncio
import argparse
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any
import signal

from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Configuration du path pour les imports
sys.path.insert(0, str(Path(__file__).parent))

from core.vlm.model import VisionLanguageModel
from detection.yolo.detector import YOLODetector
from detection.tracking.tracker import MultiObjectTracker, TrackerType
from core.orchestrator.manager import ToolOrchestrator, ToolRegistry
from validation.cross_validator import CrossValidator
from utils.performance import start_performance_monitoring, stop_performance_monitoring
from utils.exceptions import SurveillanceSystemError

# Configuration globale
console = Console()

class SurveillanceSystem:
    """Système de surveillance principal."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.is_running = False
        
        # Composants principaux
        self.vlm: Optional[VisionLanguageModel] = None
        self.yolo_detector: Optional[YOLODetector] = None
        self.tracker: Optional[MultiObjectTracker] = None
        self.orchestrator: Optional[ToolOrchestrator] = None
        self.validator: Optional[CrossValidator] = None
        
        # Configuration par défaut
        self.default_config = {
            "vlm_model": "llava-hf/llava-v1.6-mistral-7b-hf",
            "yolo_model": "yolov11n.pt",
            "tracker_type": "bytetrack",
            "device": "auto",
            "load_in_4bit": True,
            "target_fp_rate": 0.03
        }
        
        # Fusion des configurations
        self.config = {**self.default_config, **self.config}
    
    async def initialize(self) -> None:
        """Initialisation complète du système."""
        
        console.print(Panel.fit(
            "[bold blue]🕵️ Système de Surveillance Intelligente[/bold blue]\n"
            "[dim]Initialisation en cours...[/dim]",
            border_style="blue"
        ))
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            # VLM
            task = progress.add_task("[cyan]Chargement VLM...", total=None)
            try:
                self.vlm = VisionLanguageModel(
                    model_name=self.config["vlm_model"],
                    device=self.config["device"],
                    load_in_4bit=self.config["load_in_4bit"]
                )
                await self.vlm.load_model()
                progress.update(task, description="[green]✅ VLM chargé")
            except Exception as e:
                progress.update(task, description="[red]❌ Erreur VLM")
                logger.error(f"Erreur VLM: {e}")
                raise
            
            # YOLO Detector
            task = progress.add_task("[cyan]Initialisation YOLO...", total=None)
            try:
                self.yolo_detector = YOLODetector(
                    model_path=self.config["yolo_model"],
                    device=self.config["device"]
                )
                self.yolo_detector.load_model()
                progress.update(task, description="[green]✅ YOLO initialisé")
            except Exception as e:
                progress.update(task, description="[red]❌ Erreur YOLO")
                logger.error(f"Erreur YOLO: {e}")
                raise
            
            # Tracker
            task = progress.add_task("[cyan]Configuration tracker...", total=None)
            try:
                tracker_type = TrackerType.BYTETRACK if self.config["tracker_type"] == "bytetrack" else TrackerType.CENTROID
                self.tracker = MultiObjectTracker(tracker_type=tracker_type)
                progress.update(task, description="[green]✅ Tracker configuré")
            except Exception as e:
                progress.update(task, description="[red]❌ Erreur Tracker")
                logger.error(f"Erreur Tracker: {e}")
                raise
            
            # Orchestrateur
            task = progress.add_task("[cyan]Configuration orchestrateur...", total=None)
            try:
                tool_registry = ToolRegistry()
                self.orchestrator = ToolOrchestrator(self.vlm, tool_registry)
                self.orchestrator.register_surveillance_tools(self.yolo_detector, self.tracker)
                progress.update(task, description="[green]✅ Orchestrateur configuré")
            except Exception as e:
                progress.update(task, description="[red]❌ Erreur Orchestrateur")
                logger.error(f"Erreur Orchestrateur: {e}")
                raise
            
            # Validateur
            task = progress.add_task("[cyan]Configuration validateur...", total=None)
            try:
                self.validator = CrossValidator(
                    target_false_positive_rate=self.config["target_fp_rate"]
                )
                progress.update(task, description="[green]✅ Validateur configuré")
            except Exception as e:
                progress.update(task, description="[red]❌ Erreur Validateur")
                logger.error(f"Erreur Validateur: {e}")
                raise
        
        console.print(Panel.fit(
            "[bold green]✅ Système initialisé avec succès ![/bold green]\n"
            f"[dim]VLM: {self.config['vlm_model']}\n"
            f"YOLO: {self.config['yolo_model']}\n"
            f"Device: {self.config['device']}[/dim]",
            border_style="green"
        ))
    
    def get_system_status(self) -> Table:
        """Génère un tableau de statut du système."""
        
        table = Table(title="🔍 État du Système de Surveillance")
        table.add_column("Composant", style="cyan", no_wrap=True)
        table.add_column("État", style="bold")
        table.add_column("Informations", style="dim")
        
        # VLM
        vlm_status = "✅ Opérationnel" if self.vlm and self.vlm.is_loaded else "❌ Non disponible"
        vlm_info = f"{self.config['vlm_model']}" if self.vlm else "Non initialisé"
        table.add_row("Vision-Language Model", vlm_status, vlm_info)
        
        # YOLO
        yolo_status = "✅ Opérationnel" if self.yolo_detector and self.yolo_detector.is_loaded else "❌ Non disponible"
        yolo_info = f"{self.config['yolo_model']}" if self.yolo_detector else "Non initialisé"
        table.add_row("YOLO Detector", yolo_status, yolo_info)
        
        # Tracker
        tracker_status = "✅ Opérationnel" if self.tracker else "❌ Non disponible"
        tracker_info = f"{self.config['tracker_type']}" if self.tracker else "Non initialisé"
        table.add_row("Object Tracker", tracker_status, tracker_info)
        
        # Orchestrateur
        orch_status = "✅ Opérationnel" if self.orchestrator else "❌ Non disponible"
        orch_info = f"Outils: {len(self.orchestrator.tool_registry.get_available_tools())}" if self.orchestrator else "Non initialisé"
        table.add_row("Tool Orchestrator", orch_status, orch_info)
        
        # Validateur
        val_status = "✅ Opérationnel" if self.validator else "❌ Non disponible"
        val_info = f"Objectif FP: {self.config['target_fp_rate']*100:.1f}%" if self.validator else "Non initialisé"
        table.add_row("Cross Validator", val_status, val_info)
        
        return table
    
    def cleanup(self) -> None:
        """Nettoyage des ressources."""
        
        console.print("[yellow]🧹 Nettoyage en cours...[/yellow]")
        
        try:
            if self.vlm:
                self.vlm.unload_model()
            
            if self.yolo_detector:
                self.yolo_detector.cleanup()
            
            if self.tracker:
                self.tracker.reset()
            
            stop_performance_monitoring()
            
            console.print("[green]✅ Nettoyage terminé[/green]")
            
        except Exception as e:
            logger.error(f"Erreur lors du nettoyage: {e}")


def create_argument_parser() -> argparse.ArgumentParser:
    """Création du parser d'arguments."""
    
    parser = argparse.ArgumentParser(
        description="Système de Surveillance Intelligente Multimodale",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  surveillance-system --demo                    # Démonstration rapide
  surveillance-system --status                  # Affichage du statut
  surveillance-system --benchmark              # Benchmark de performance
  surveillance-system --vlm-model llava-7b     # Modèle VLM spécifique
  surveillance-system --device cuda            # Forcer GPU
        """
    )
    
    parser.add_argument(
        "--demo", 
        action="store_true",
        help="Mode démonstration avec données de test"
    )
    
    parser.add_argument(
        "--status",
        action="store_true", 
        help="Afficher le statut du système"
    )
    
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Lancer un benchmark de performance"
    )
    
    parser.add_argument(
        "--vlm-model",
        type=str,
        default="llava-hf/llava-v1.6-mistral-7b-hf",
        help="Modèle VLM à utiliser"
    )
    
    parser.add_argument(
        "--yolo-model", 
        type=str,
        default="yolov11n.pt",
        help="Modèle YOLO à utiliser"
    )
    
    parser.add_argument(
        "--device",
        type=str, 
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device de calcul"
    )
    
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Utiliser la quantization 4-bit"
    )
    
    parser.add_argument(
        "--target-fp-rate",
        type=float,
        default=0.03,
        help="Taux de faux positifs cible (0.03 = 3%)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Mode verbeux"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Niveau de logging"
    )
    
    return parser


async def demo_mode() -> None:
    """Mode démonstration."""
    
    console.print(Panel.fit(
        "[bold yellow]🎮 Mode Démonstration[/bold yellow]\n"
        "[dim]Test des fonctionnalités principales[/dim]",
        border_style="yellow"
    ))
    
    # Configuration légère pour la démo
    demo_config = {
        "vlm_model": "microsoft/git-base-coco",  # Modèle plus léger
        "yolo_model": "yolov11n.pt",
        "device": "auto",
        "load_in_4bit": True,
        "target_fp_rate": 0.05
    }
    
    system = SurveillanceSystem(demo_config)
    
    try:
        # Initialisation
        await system.initialize()
        
        # Affichage du statut
        console.print(system.get_system_status())
        
        console.print("\n[green]🎉 Démonstration terminée avec succès ![/green]")
        console.print("[dim]Pour des tests plus approfondis, utilisez le notebook Colab.[/dim]")
        
    except Exception as e:
        console.print(f"[red]❌ Erreur dans la démonstration: {e}[/red]")
        raise
    
    finally:
        system.cleanup()


async def status_mode() -> None:
    """Mode affichage de statut."""
    
    console.print(Panel.fit(
        "[bold cyan]📊 Statut du Système[/bold cyan]",
        border_style="cyan"
    ))
    
    # Test rapide sans initialisation complète
    try:
        import torch
        
        # Informations système
        table = Table(title="🖥️ Informations Système")
        table.add_column("Composant", style="cyan")
        table.add_column("Statut", style="bold")
        table.add_column("Détails", style="dim")
        
        # Python
        table.add_row("Python", "✅", f"{sys.version.split()[0]}")
        
        # PyTorch
        table.add_row("PyTorch", "✅", f"{torch.__version__}")
        
        # CUDA
        cuda_status = "✅ Disponible" if torch.cuda.is_available() else "❌ Non disponible"
        cuda_info = f"{torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU seulement"
        table.add_row("CUDA", cuda_status, cuda_info)
        
        # Modules
        try:
            import transformers, ultralytics, cv2
            table.add_row("Modules IA", "✅", "Transformers, YOLO, OpenCV")
        except ImportError as e:
            table.add_row("Modules IA", "❌", f"Manquant: {e}")
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]❌ Erreur lors de la vérification: {e}[/red]")


async def benchmark_mode() -> None:
    """Mode benchmark."""
    
    console.print(Panel.fit(
        "[bold magenta]🏃‍♂️ Benchmark de Performance[/bold magenta]",
        border_style="magenta"
    ))
    
    console.print("[yellow]⚠️ Benchmark complet nécessite une initialisation complète.[/yellow]")
    console.print("[dim]Pour un benchmark détaillé, utilisez le notebook Colab.[/dim]")
    
    # Benchmark rapide des imports
    import time
    
    benchmarks = []
    
    with Progress(console=console) as progress:
        task = progress.add_task("[cyan]Test des imports...", total=4)
        
        # PyTorch
        start = time.time()
        import torch
        torch_time = time.time() - start
        benchmarks.append(("PyTorch", torch_time))
        progress.advance(task)
        
        # Transformers
        start = time.time()
        import transformers
        transformers_time = time.time() - start
        benchmarks.append(("Transformers", transformers_time))
        progress.advance(task)
        
        # Ultralytics
        start = time.time()
        import ultralytics
        yolo_time = time.time() - start
        benchmarks.append(("Ultralytics", yolo_time))
        progress.advance(task)
        
        # OpenCV
        start = time.time()
        import cv2
        cv2_time = time.time() - start
        benchmarks.append(("OpenCV", cv2_time))
        progress.advance(task)
    
    # Affichage des résultats
    table = Table(title="⏱️ Temps d'Import des Modules")
    table.add_column("Module", style="cyan")
    table.add_column("Temps (s)", style="bold")
    table.add_column("État", style="green")
    
    for module, import_time in benchmarks:
        status = "✅ Rapide" if import_time < 1.0 else "⚠️ Lent" if import_time < 3.0 else "❌ Très lent"
        table.add_row(module, f"{import_time:.3f}", status)
    
    console.print(table)
    
    total_time = sum(time for _, time in benchmarks)
    console.print(f"\n[bold]Temps total d'import: {total_time:.3f}s[/bold]")


def setup_signal_handlers(system: Optional[SurveillanceSystem] = None) -> None:
    """Configuration des gestionnaires de signaux."""
    
    def signal_handler(signum, frame):
        console.print(f"\n[yellow]Signal {signum} reçu - Arrêt en cours...[/yellow]")
        if system:
            system.cleanup()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


async def main_async() -> None:
    """Point d'entrée principal asynchrone."""
    
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Configuration du logging
    logger.remove()
    log_level = args.log_level if args.verbose else "WARNING"
    logger.add(sys.stderr, level=log_level, format="<green>{time}</green> | <level>{level}</level> | {message}")
    
    # Démarrage du monitoring de performance
    start_performance_monitoring()
    
    try:
        # Banner
        console.print(Panel.fit(
            "[bold blue]🕵️ SYSTÈME DE SURVEILLANCE INTELLIGENTE MULTIMODALE[/bold blue]\n"
            "[dim]Basé sur VLM avec orchestration d'outils intelligente[/dim]\n\n"
            f"[yellow]Version: 1.0.0 | Python: {sys.version.split()[0]}[/yellow]",
            border_style="bright_blue"
        ))
        
        # Modes d'exécution
        if args.demo:
            await demo_mode()
        elif args.status:
            await status_mode()
        elif args.benchmark:
            await benchmark_mode()
        else:
            # Mode interactif par défaut
            console.print("[yellow]💡 Utilisez --help pour voir les options disponibles[/yellow]")
            console.print("[dim]Modes disponibles: --demo, --status, --benchmark[/dim]")
            
            # Configuration du système
            config = {
                "vlm_model": args.vlm_model,
                "yolo_model": args.yolo_model,
                "device": args.device,
                "load_in_4bit": args.load_in_4bit,
                "target_fp_rate": args.target_fp_rate
            }
            
            system = SurveillanceSystem(config)
            setup_signal_handlers(system)
            
            # Choix utilisateur
            choice = console.input("\n[cyan]Voulez-vous initialiser le système complet ? (y/n): [/cyan]")
            
            if choice.lower().startswith('y'):
                await system.initialize()
                console.print(system.get_system_status())
                console.print("\n[green]Système prêt ! Utilisez Ctrl+C pour arrêter.[/green]")
                
                # Boucle d'attente
                try:
                    while True:
                        await asyncio.sleep(1)
                except KeyboardInterrupt:
                    pass
                finally:
                    system.cleanup()
            else:
                console.print("[dim]Système non initialisé. Au revoir ![/dim]")
    
    except KeyboardInterrupt:
        console.print("\n[yellow]Interruption utilisateur[/yellow]")
    except Exception as e:
        console.print(f"[red]❌ Erreur: {e}[/red]")
        if args.verbose:
            console.print_exception()
        sys.exit(1)
    finally:
        stop_performance_monitoring()


def main() -> None:
    """Point d'entrée principal synchrone."""
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        console.print("\n[dim]Au revoir ![/dim]")
    except Exception as e:
        console.print(f"[red]Erreur fatale: {e}[/red]")
        sys.exit(1)


def demo_main() -> None:
    """Point d'entrée pour la démonstration rapide."""
    asyncio.run(demo_mode())


# Fonctions utilitaires pour Colab
def demo_surveillance():
    """Démonstration simple pour Colab/Jupyter."""
    
    print("🕵️ Système de Surveillance Intelligente - Démo")
    print("=" * 50)
    
    try:
        # Test des imports
        print("📦 Test des imports...")
        from core.vlm.model import VisionLanguageModel
        from detection.yolo.detector import YOLODetector
        print("✅ Imports réussis")
        
        # Info système
        import torch
        print(f"🐍 Python: {sys.version.split()[0]}")
        print(f"🔥 PyTorch: {torch.__version__}")
        print(f"🎮 CUDA: {'✅' if torch.cuda.is_available() else '❌'}")
        
        if torch.cuda.is_available():
            print(f"🎯 GPU: {torch.cuda.get_device_name(0)}")
        
        print("\n🎉 Système opérationnel !")
        print("💡 Utilisez le notebook Colab pour des tests complets.")
        
    except Exception as e:
        print(f"❌ Erreur: {e}")


if __name__ == "__main__":
    main()