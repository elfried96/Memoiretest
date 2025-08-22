#!/usr/bin/env python3
"""
🧪 Suite de Tests UV pour Serveur GPU
====================================

Tests optimisés pour environnement UV
"""

import asyncio
import sys
import time
import traceback
import subprocess
from pathlib import Path

# Ajout du chemin
sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table

console = Console()

class UVGPUTestRunner:
    """Gestionnaire des tests UV + GPU."""
    
    def __init__(self):
        self.results = {}
        self.total_tests = 9  # Un test supplémentaire pour UV
        
    async def run_all_tests(self):
        """Exécute tous les tests avec UV."""
        
        console.print(Panel.fit(
            "[bold blue]🧪 SUITE DE TESTS UV + GPU[/bold blue]\n"
            "[dim]Validation complète avec UV[/dim]",
            border_style="blue"
        ))
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:
            
            main_task = progress.add_task("Tests UV", total=self.total_tests)
            
            # 1. Test environnement UV
            progress.update(main_task, description="Test environnement UV...")
            result1 = await self.test_uv_environment()
            self.results["uv_environment"] = result1
            progress.advance(main_task)
            
            # 2. Test imports de base via UV
            progress.update(main_task, description="Test imports via UV...")
            result2 = await self.test_uv_imports()
            self.results["uv_imports"] = result2
            progress.advance(main_task)
            
            # 3. Test GPU/PyTorch via UV
            progress.update(main_task, description="Test GPU via UV...")
            result3 = await self.test_uv_gpu_pytorch()
            self.results["uv_gpu_pytorch"] = result3
            progress.advance(main_task)
            
            # 4. Test YOLO11 via UV
            progress.update(main_task, description="Test YOLO11 via UV...")
            result4 = await self.test_uv_yolo11()
            self.results["uv_yolo11"] = result4
            progress.advance(main_task)
            
            # 5. Test types système via UV
            progress.update(main_task, description="Test types via UV...")
            result5 = await self.test_uv_system_types()
            self.results["uv_system_types"] = result5
            progress.advance(main_task)
            
            # 6. Test VLM via UV
            progress.update(main_task, description="Test VLM via UV...")
            result6 = await self.test_uv_vlm_system()
            self.results["uv_vlm_system"] = result6
            progress.advance(main_task)
            
            # 7. Test orchestrateur via UV
            progress.update(main_task, description="Test orchestrateur via UV...")
            result7 = await self.test_uv_orchestrator()
            self.results["uv_orchestrator"] = result7
            progress.advance(main_task)
            
            # 8. Test intégration via UV
            progress.update(main_task, description="Test intégration via UV...")
            result8 = await self.test_uv_integration()
            self.results["uv_integration"] = result8
            progress.advance(main_task)
            
            # 9. Test performance via UV
            progress.update(main_task, description="Test performance via UV...")
            result9 = await self.test_uv_performance()
            self.results["uv_performance"] = result9
            progress.advance(main_task)
        
        # Affichage des résultats
        self.display_results()
        
        return all(self.results.values())
    
    async def test_uv_environment(self):
        """Test de l'environnement UV."""
        try:
            # Test UV disponible
            result = subprocess.run(['uv', '--version'], capture_output=True, text=True)
            if result.returncode != 0:
                console.print("❌ UV non disponible")
                return False
                
            # Test environnement .venv
            venv_path = Path(".venv")
            if not venv_path.exists():
                console.print("❌ Environnement .venv manquant")
                return False
                
            console.print("✅ Environnement UV: OK")
            return True
            
        except Exception as e:
            console.print(f"❌ Environnement UV: {e}")
            return False
    
    async def test_uv_imports(self):
        """Test des imports via UV."""
        try:
            cmd = ['uv', 'run', 'python', '-c', '''
import numpy as np
import cv2
import torch
import PIL
from rich import console
print("Imports de base OK")
''']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                console.print("✅ Imports via UV: OK")
                return True
            else:
                console.print(f"❌ Imports via UV: {result.stderr}")
                return False
                
        except Exception as e:
            console.print(f"❌ Imports via UV: {e}")
            return False
    
    async def test_uv_gpu_pytorch(self):
        """Test GPU PyTorch via UV."""
        try:
            cmd = ['uv', 'run', 'python', '-c', '''
import torch
cuda_available = torch.cuda.is_available()
if cuda_available:
    device_name = torch.cuda.get_device_name(0)
    print(f"GPU: {device_name}")
    # Test allocation
    x = torch.randn(100, 100).cuda()
    y = x * 2
    del x, y
    torch.cuda.empty_cache()
else:
    print("Mode CPU")
print("PyTorch OK")
''']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                console.print(f"✅ GPU PyTorch via UV: {result.stdout.strip()}")
                return True
            else:
                console.print(f"❌ GPU PyTorch via UV: {result.stderr}")
                return False
                
        except Exception as e:
            console.print(f"❌ GPU PyTorch via UV: {e}")
            return False
    
    async def test_uv_yolo11(self):
        """Test YOLO11 via UV."""
        try:
            cmd = ['uv', 'run', 'python', '-c', '''
from ultralytics import YOLO
import numpy as np

# Chargement modèle
model = YOLO("yolov11n.pt")
print("Modèle YOLO11 chargé")

# Test inférence
test_img = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
results = model(test_img, verbose=False)
print("Inférence YOLO11 OK")
''']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                console.print("✅ YOLO11 via UV: OK")
                return True
            else:
                console.print(f"❌ YOLO11 via UV: {result.stderr}")
                return False
                
        except Exception as e:
            console.print(f"❌ YOLO11 via UV: {e}")
            return False
    
    async def test_uv_system_types(self):
        """Test types système via UV."""
        try:
            cmd = ['uv', 'run', 'python', '-c', '''
import sys
sys.path.insert(0, ".")
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

print("Types système OK")
''']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                console.print("✅ Types système via UV: OK")
                return True
            else:
                console.print(f"❌ Types système via UV: {result.stderr}")
                return False
                
        except Exception as e:
            console.print(f"❌ Types système via UV: {e}")
            return False
    
    async def test_uv_vlm_system(self):
        """Test VLM via UV."""
        try:
            cmd = ['uv', 'run', 'python', '-c', '''
import sys
sys.path.insert(0, ".")
from src.core.vlm.model_registry import VLMModelRegistry

registry = VLMModelRegistry()
models = registry.list_available_models()
recommendations = registry.get_model_recommendations()

assert len(models) > 0
assert len(recommendations) > 0

print(f"VLM Registry: {len(models)} modèles")
''']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                console.print("✅ VLM via UV: OK")
                return True
            else:
                console.print(f"❌ VLM via UV: {result.stderr}")
                return False
                
        except Exception as e:
            console.print(f"❌ VLM via UV: {e}")
            return False
    
    async def test_uv_orchestrator(self):
        """Test orchestrateur via UV."""
        try:
            cmd = ['uv', 'run', 'python', '-c', '''
import sys
sys.path.insert(0, ".")
from src.core.orchestrator.vlm_orchestrator import (
    ModernVLMOrchestrator, OrchestrationConfig, OrchestrationMode
)

config = OrchestrationConfig(
    mode=OrchestrationMode.BALANCED,
    enable_advanced_tools=True
)

print("Orchestrateur OK")
''']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                console.print("✅ Orchestrateur via UV: OK")
                return True
            else:
                console.print(f"❌ Orchestrateur via UV: {result.stderr}")
                return False
                
        except Exception as e:
            console.print(f"❌ Orchestrateur via UV: {e}")
            return False
    
    async def test_uv_integration(self):
        """Test d'intégration via UV."""
        try:
            cmd = ['uv', 'run', 'python', '-c', '''
import sys
sys.path.insert(0, ".")
from ultralytics import YOLO
from src.core.types import Detection, BoundingBox
import numpy as np

# YOLO detection
model = YOLO("yolov11n.pt")
test_img = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
yolo_results = model(test_img, verbose=False)

# Conversion vers nos types
detections = []
for result in yolo_results:
    if hasattr(result, "boxes") and result.boxes is not None:
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

print(f"Intégration: {len(detections)} détections")
''']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                console.print("✅ Intégration via UV: OK")
                return True
            else:
                console.print(f"❌ Intégration via UV: {result.stderr}")
                return False
                
        except Exception as e:
            console.print(f"❌ Intégration via UV: {e}")
            return False
    
    async def test_uv_performance(self):
        """Test performance via UV."""
        try:
            cmd = ['uv', 'run', 'python', '-c', '''
from ultralytics import YOLO
import numpy as np
import time

model = YOLO("yolov11n.pt")

# Test batch
images = [np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8) for _ in range(5)]

start_time = time.time()
for img in images:
    results = model(img, verbose=False)
total_time = time.time() - start_time

fps = len(images) / total_time
print(f"Performance: {fps:.1f} FPS")

# Test réussi si FPS > 1.0
assert fps > 1.0
''']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                console.print(f"✅ Performance via UV: {result.stdout.strip()}")
                return True
            else:
                console.print(f"❌ Performance via UV: {result.stderr}")
                return False
                
        except Exception as e:
            console.print(f"❌ Performance via UV: {e}")
            return False
    
    def display_results(self):
        """Affiche les résultats des tests UV."""
        
        console.print("\n" + "="*60)
        console.print(Panel.fit(
            "[bold green]📋 RÉSULTATS TESTS UV[/bold green]",
            border_style="green"
        ))
        
        # Table des résultats
        results_table = Table(title="🧪 État des Tests UV")
        results_table.add_column("Test UV", style="cyan")
        results_table.add_column("Résultat", style="bold")
        results_table.add_column("Status", style="dim")
        
        for test_name, result in self.results.items():
            status_icon = "✅" if result else "❌"
            status_text = "RÉUSSI" if result else "ÉCHEC"
            color = "green" if result else "red"
            
            results_table.add_row(
                test_name.replace("uv_", "").replace("_", " ").title(),
                f"[{color}]{status_icon} {status_text}[/{color}]",
                "Opérationnel" if result else "À corriger"
            )
        
        console.print(results_table)
        
        # Résumé global
        total_passed = sum(1 for r in self.results.values() if r)
        total_tests = len(self.results)
        
        if total_passed == total_tests:
            console.print(f"\n🎉 [bold green]TOUS LES TESTS UV RÉUSSIS ({total_passed}/{total_tests})[/bold green]")
            console.print("🚀 SYSTÈME UV PRÊT POUR PRODUCTION !")
            
            console.print("\n💡 [bold cyan]COMMANDES UV RECOMMANDÉES:[/bold cyan]")
            console.print("   • uv run python test_full_system_video.py --video webcam --max-frames 100")
            console.print("   • uv run python main.py --video webcam")
            console.print("   • ./run_surveillance.sh --video webcam")
            
        else:
            console.print(f"\n⚠️ [bold yellow]TESTS UV PARTIELS ({total_passed}/{total_tests})[/bold yellow]")
            console.print("🔧 Corrigez les erreurs et relancez: uv sync")

async def main():
    """Fonction principale."""
    
    console.print("🚀 Démarrage des tests UV + GPU...")
    
    runner = UVGPUTestRunner()
    
    try:
        all_passed = await runner.run_all_tests()
        
        if all_passed:
            console.print("\n🎯 [bold green]VALIDATION UV COMPLÈTE RÉUSSIE ![/bold green]")
            return 0
        else:
            console.print("\n🔧 [bold yellow]CERTAINS TESTS UV ONT ÉCHOUÉ[/bold yellow]")
            return 1
            
    except KeyboardInterrupt:
        console.print("\n🛑 Tests UV interrompus")
        return 1
    except Exception as e:
        console.print(f"\n❌ Erreur inattendue UV: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())