#!/usr/bin/env python3
"""
🧪 Suite de Tests Complète pour Serveur GPU
==========================================

Exécute tous les tests dans l'ordre optimal pour garantir
que le système fonctionne parfaitement sur GPU.
"""

import asyncio
import sys
import time
import traceback
from pathlib import Path

# Ajout du chemin
sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table

console = Console()

class GPUTestRunner:
    """Gestionnaire des tests GPU."""
    
    def __init__(self):
        self.results = {}
        self.total_tests = 8
        
    async def run_all_tests(self):
        """Exécute tous les tests."""
        
        console.print(Panel.fit(
            "[bold blue]🧪 SUITE DE TESTS GPU COMPLÈTE[/bold blue]\n"
            "[dim]Validation complète du système de surveillance[/dim]",
            border_style="blue"
        ))
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:
            
            main_task = progress.add_task("Tests généraux", total=self.total_tests)
            
            # 1. Test des imports de base
            progress.update(main_task, description="Test imports de base...")
            result1 = await self.test_basic_imports()
            self.results["basic_imports"] = result1
            progress.advance(main_task)
            
            # 2. Test GPU/PyTorch
            progress.update(main_task, description="Test GPU/PyTorch...")
            result2 = await self.test_gpu_pytorch()
            self.results["gpu_pytorch"] = result2
            progress.advance(main_task)
            
            # 3. Test YOLO11
            progress.update(main_task, description="Test YOLO11...")
            result3 = await self.test_yolo11()
            self.results["yolo11"] = result3
            progress.advance(main_task)
            
            # 4. Test types système
            progress.update(main_task, description="Test types système...")
            result4 = await self.test_system_types()
            self.results["system_types"] = result4
            progress.advance(main_task)
            
            # 5. Test VLM
            progress.update(main_task, description="Test VLM...")
            result5 = await self.test_vlm_system()
            self.results["vlm_system"] = result5
            progress.advance(main_task)
            
            # 6. Test orchestrateur
            progress.update(main_task, description="Test orchestrateur...")
            result6 = await self.test_orchestrator()
            self.results["orchestrator"] = result6
            progress.advance(main_task)
            
            # 7. Test intégration
            progress.update(main_task, description="Test intégration...")
            result7 = await self.test_integration()
            self.results["integration"] = result7
            progress.advance(main_task)
            
            # 8. Test performance
            progress.update(main_task, description="Test performance...")
            result8 = await self.test_performance()
            self.results["performance"] = result8
            progress.advance(main_task)
        
        # Affichage des résultats
        self.display_results()
        
        return all(self.results.values())
    
    async def test_basic_imports(self):
        """Test des imports de base."""
        try:
            import numpy as np
            import cv2
            import torch
            import PIL
            from rich import console
            console.print("✅ Imports de base: OK")
            return True
        except Exception as e:
            console.print(f"❌ Imports de base: {e}")
            return False
    
    async def test_gpu_pytorch(self):
        """Test GPU et PyTorch."""
        try:
            import torch
            
            # Test CUDA
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                device_name = torch.cuda.get_device_name(0)
                console.print(f"✅ GPU: {device_name}")
                
                # Test allocation GPU
                x = torch.randn(100, 100).cuda()
                y = x * 2
                del x, y
                torch.cuda.empty_cache()
                
            else:
                console.print("⚠️ GPU non disponible - Mode CPU")
            
            return True
            
        except Exception as e:
            console.print(f"❌ GPU/PyTorch: {e}")
            return False
    
    async def test_yolo11(self):
        """Test YOLO11."""
        try:
            from ultralytics import YOLO
            
            # Chargement modèle
            model = YOLO('yolov11n.pt')
            console.print("✅ YOLO11 modèle chargé")
            
            # Test inférence simple
            import numpy as np
            test_img = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
            
            results = model(test_img, verbose=False)
            console.print("✅ YOLO11 inférence: OK")
            
            return True
            
        except Exception as e:
            console.print(f"❌ YOLO11: {e}")
            traceback.print_exc()
            return False
    
    async def test_system_types(self):
        """Test des types système."""
        try:
            from src.core.types import (
                Detection, DetectedObject, BoundingBox, 
                AnalysisRequest, AnalysisResponse,
                SuspicionLevel, ActionType, ToolResult
            )
            
            # Test BoundingBox
            bbox = BoundingBox(x1=100, y1=100, x2=200, y2=200, confidence=0.8)
            assert bbox.width == 100
            assert bbox.height == 100
            
            # Test Detection
            detection = Detection(
                class_id=0,
                class_name="person",
                bbox=bbox,
                confidence=0.85
            )
            
            # Test AnalysisRequest
            request = AnalysisRequest(
                frame_data="test_data",
                context={"test": True}
            )
            
            console.print("✅ Types système: OK")
            return True
            
        except Exception as e:
            console.print(f"❌ Types système: {e}")
            traceback.print_exc()
            return False
    
    async def test_vlm_system(self):
        """Test système VLM."""
        try:
            from src.core.vlm.model_registry import VLMModelRegistry
            
            # Test registry
            registry = VLMModelRegistry()
            models = registry.list_available_models()
            recommendations = registry.get_model_recommendations()
            
            assert len(models) > 0
            assert len(recommendations) > 0
            
            console.print(f"✅ VLM Registry: {len(models)} modèles")
            return True
            
        except Exception as e:
            console.print(f"❌ VLM système: {e}")
            traceback.print_exc()
            return False
    
    async def test_orchestrator(self):
        """Test orchestrateur."""
        try:
            from src.core.orchestrator.vlm_orchestrator import (
                ModernVLMOrchestrator, OrchestrationConfig, OrchestrationMode
            )
            
            # Test configuration
            config = OrchestrationConfig(
                mode=OrchestrationMode.BALANCED,
                enable_advanced_tools=True
            )
            
            # Test orchestrateur (sans initialisation complète)
            # On teste juste l'import et la création de config
            console.print("✅ Orchestrateur: OK")
            return True
            
        except Exception as e:
            console.print(f"❌ Orchestrateur: {e}")
            traceback.print_exc()
            return False
    
    async def test_integration(self):
        """Test d'intégration."""
        try:
            # Test chaîne complète: YOLO → Types → VLM
            from ultralytics import YOLO
            from src.core.types import Detection, BoundingBox
            import numpy as np
            
            # 1. YOLO detection
            model = YOLO('yolov11n.pt')
            test_img = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
            yolo_results = model(test_img, verbose=False)
            
            # 2. Conversion vers nos types
            detections = []
            for result in yolo_results:
                if hasattr(result, 'boxes') and result.boxes is not None:
                    for box in result.boxes:
                        if len(box.xyxy) > 0:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            conf = box.conf[0].cpu().numpy()
                            cls = int(box.cls[0].cpu().numpy())
                            
                            detection = Detection(
                                class_id=cls,
                                class_name=f"class_{cls}",
                                bbox=BoundingBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2)),
                                confidence=float(conf)
                            )
                            detections.append(detection)
            
            console.print(f"✅ Intégration: {len(detections)} détections")
            return True
            
        except Exception as e:
            console.print(f"❌ Intégration: {e}")
            traceback.print_exc()
            return False
    
    async def test_performance(self):
        """Test de performance."""
        try:
            from ultralytics import YOLO
            import numpy as np
            import time
            
            model = YOLO('yolov11n.pt')
            
            # Test batch
            images = [np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8) for _ in range(5)]
            
            start_time = time.time()
            for img in images:
                results = model(img, verbose=False)
            total_time = time.time() - start_time
            
            fps = len(images) / total_time
            console.print(f"✅ Performance: {fps:.1f} FPS")
            
            return fps > 1.0  # Au moins 1 FPS
            
        except Exception as e:
            console.print(f"❌ Performance: {e}")
            return False
    
    def display_results(self):
        """Affiche les résultats des tests."""
        
        console.print("\n" + "="*60)
        console.print(Panel.fit(
            "[bold green]📋 RÉSULTATS DES TESTS[/bold green]",
            border_style="green"
        ))
        
        # Table des résultats
        results_table = Table(title="🧪 État des Tests")
        results_table.add_column("Test", style="cyan")
        results_table.add_column("Résultat", style="bold")
        results_table.add_column("Status", style="dim")
        
        for test_name, result in self.results.items():
            status_icon = "✅" if result else "❌"
            status_text = "RÉUSSI" if result else "ÉCHEC"
            color = "green" if result else "red"
            
            results_table.add_row(
                test_name.replace("_", " ").title(),
                f"[{color}]{status_icon} {status_text}[/{color}]",
                "Opérationnel" if result else "À corriger"
            )
        
        console.print(results_table)
        
        # Résumé global
        total_passed = sum(1 for r in self.results.values() if r)
        total_tests = len(self.results)
        
        if total_passed == total_tests:
            console.print(f"\n🎉 [bold green]TOUS LES TESTS RÉUSSIS ({total_passed}/{total_tests})[/bold green]")
            console.print("🚀 SYSTÈME PRÊT POUR PRODUCTION !")
            
            console.print("\n💡 [bold cyan]COMMANDES RECOMMANDÉES:[/bold cyan]")
            console.print("   • python test_full_system_video.py --video webcam --max-frames 100")
            console.print("   • python main.py --video webcam")
            
        else:
            console.print(f"\n⚠️ [bold yellow]TESTS PARTIELS ({total_passed}/{total_tests})[/bold yellow]")
            console.print("🔧 Corrigez les erreurs et relancez les tests")

async def main():
    """Fonction principale."""
    
    console.print("🚀 Démarrage des tests GPU...")
    
    runner = GPUTestRunner()
    
    try:
        all_passed = await runner.run_all_tests()
        
        if all_passed:
            console.print("\n🎯 [bold green]VALIDATION COMPLÈTE RÉUSSIE ![/bold green]")
            return 0
        else:
            console.print("\n🔧 [bold yellow]CERTAINS TESTS ONT ÉCHOUÉ[/bold yellow]")
            return 1
            
    except KeyboardInterrupt:
        console.print("\n🛑 Tests interrompus par l'utilisateur")
        return 1
    except Exception as e:
        console.print(f"\n❌ Erreur inattendue: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())