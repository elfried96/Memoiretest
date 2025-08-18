#!/usr/bin/env python3
"""
🔍 Vérification Complète du Système GPU
=====================================

Script de diagnostic pour s'assurer que tout fonctionne parfaitement
sur serveur GPU avant les tests de production.
"""

import sys
import subprocess
import importlib
from pathlib import Path
import torch
import cv2
import numpy as np

# Ajout du chemin
sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

def check_gpu_setup():
    """Vérification de la configuration GPU."""
    
    console.print(Panel.fit(
        "[bold blue]🔍 DIAGNOSTIC SYSTÈME GPU[/bold blue]\n"
        "[dim]Vérification complète pour serveur de surveillance[/dim]",
        border_style="blue"
    ))
    
    # Table des résultats
    results_table = Table(title="📊 État du Système")
    results_table.add_column("Composant", style="cyan", no_wrap=True)
    results_table.add_column("Status", style="bold")
    results_table.add_column("Détails", style="dim")
    
    # 1. Vérification GPU
    try:
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            results_table.add_row(
                "GPU NVIDIA", 
                "✅ Détecté", 
                f"{gpu_name} ({gpu_memory:.1f} GB)"
            )
        else:
            results_table.add_row(
                "GPU NVIDIA", 
                "⚠️ Non détecté", 
                "Mode CPU activé"
            )
    except Exception as e:
        results_table.add_row(
            "GPU NVIDIA", 
            "❌ Erreur", 
            str(e)
        )
    
    # 2. PyTorch
    try:
        import torch
        cuda_status = "GPU" if torch.cuda.is_available() else "CPU"
        results_table.add_row(
            "PyTorch", 
            "✅ Installé", 
            f"v{torch.__version__} ({cuda_status})"
        )
    except ImportError:
        results_table.add_row(
            "PyTorch", 
            "❌ Manquant", 
            "pip install torch"
        )
    
    # 3. Ultralytics YOLO
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
    
    # 4. OpenCV
    try:
        import cv2
        results_table.add_row(
            "OpenCV", 
            "✅ Installé", 
            f"v{cv2.__version__}"
        )
    except ImportError:
        results_table.add_row(
            "OpenCV", 
            "❌ Manquant", 
            "pip install opencv-python"
        )
    
    # 5. Transformers
    try:
        import transformers
        results_table.add_row(
            "Transformers", 
            "✅ Installé", 
            f"v{transformers.__version__}"
        )
    except ImportError:
        results_table.add_row(
            "Transformers", 
            "❌ Manquant", 
            "pip install transformers"
        )
    
    # 6. Types système
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
    
    # 7. VLM Registry
    try:
        from src.core.vlm.model_registry import VLMModelRegistry
        registry = VLMModelRegistry()
        models = registry.list_available_models()
        results_table.add_row(
            "VLM Registry", 
            "✅ Opérationnel", 
            f"{len(models)} modèles"
        )
    except Exception as e:
        results_table.add_row(
            "VLM Registry", 
            "❌ Erreur", 
            str(e)[:50]
        )
    
    # 8. Orchestrateur
    try:
        from src.core.orchestrator.vlm_orchestrator import ModernVLMOrchestrator
        results_table.add_row(
            "Orchestrateur", 
            "✅ Prêt", 
            "ModernVLMOrchestrator"
        )
    except ImportError as e:
        results_table.add_row(
            "Orchestrateur", 
            "❌ Erreur", 
            str(e)[:50]
        )
    
    console.print(results_table)
    return results_table

def test_yolo_inference():
    """Test d'inférence YOLO11."""
    
    console.print("\n🎯 Test d'Inférence YOLO11")
    
    try:
        from ultralytics import YOLO
        
        # Chargement modèle
        model = YOLO('yolov11n.pt')
        console.print("✅ Modèle YOLO11 chargé")
        
        # Image de test
        test_image = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
        
        # Inférence
        import time
        start_time = time.time()
        results = model(test_image, verbose=False)
        inference_time = time.time() - start_time
        
        console.print(f"✅ Inférence réussie en {inference_time:.3f}s")
        
        # Vérification GPU
        if torch.cuda.is_available():
            console.print(f"🔥 GPU utilisé: {torch.cuda.get_device_name(0)}")
        else:
            console.print("💻 CPU utilisé")
            
        return True
        
    except Exception as e:
        console.print(f"❌ Erreur inférence: {e}")
        return False

def test_system_integration():
    """Test d'intégration système."""
    
    console.print("\n🧪 Test d'Intégration Système")
    
    try:
        # Import des composants
        from src.core.types import Detection, BoundingBox
        from src.core.orchestrator.vlm_orchestrator import OrchestrationConfig, OrchestrationMode
        
        # Test BoundingBox
        bbox = BoundingBox(x1=100, y1=100, x2=200, y2=200, confidence=0.8)
        console.print(f"✅ BoundingBox: {bbox.width}x{bbox.height}")
        
        # Test Detection
        detection = Detection(
            class_id=0,
            class_name="person",
            bbox=bbox,
            confidence=0.85
        )
        console.print(f"✅ Detection: {detection.class_name}")
        
        # Test Config
        config = OrchestrationConfig(
            mode=OrchestrationMode.BALANCED,
            enable_advanced_tools=True
        )
        console.print(f"✅ Config: {config.mode.value}")
        
        return True
        
    except Exception as e:
        console.print(f"❌ Erreur intégration: {e}")
        return False

def run_performance_benchmark():
    """Benchmark de performance."""
    
    console.print("\n⚡ Benchmark de Performance")
    
    # Test mémoire GPU
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        
        # Allocation test
        x = torch.randn(1000, 1000).cuda()
        y = torch.matmul(x, x)
        
        memory_used = torch.cuda.max_memory_allocated() / 1e6
        console.print(f"🔥 Mémoire GPU utilisée: {memory_used:.1f} MB")
        
        del x, y
        torch.cuda.empty_cache()
    
    # Test YOLO performance
    try:
        from ultralytics import YOLO
        model = YOLO('yolov11n.pt')
        
        # Images de test multiples
        images = [np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8) for _ in range(10)]
        
        import time
        start_time = time.time()
        
        for img in images:
            results = model(img, verbose=False)
        
        total_time = time.time() - start_time
        fps = len(images) / total_time
        
        console.print(f"⚡ Performance: {fps:.1f} FPS sur {len(images)} images")
        
    except Exception as e:
        console.print(f"❌ Erreur benchmark: {e}")

def generate_gpu_report():
    """Génération rapport GPU."""
    
    report = {
        "gpu_available": torch.cuda.is_available(),
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
    }
    
    if torch.cuda.is_available():
        report.update({
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
            "gpu_count": torch.cuda.device_count()
        })
    
    return report

def main():
    """Fonction principale."""
    
    # 1. Diagnostic système
    results_table = check_gpu_setup()
    
    # 2. Test YOLO
    yolo_ok = test_yolo_inference()
    
    # 3. Test intégration
    integration_ok = test_system_integration()
    
    # 4. Benchmark
    run_performance_benchmark()
    
    # 5. Rapport final
    console.print("\n" + "="*60)
    console.print(Panel.fit(
        "[bold green]📋 RAPPORT FINAL[/bold green]",
        border_style="green"
    ))
    
    if yolo_ok and integration_ok:
        console.print("🎉 [bold green]SYSTÈME PRÊT POUR PRODUCTION ![/bold green]")
        console.print("\n💡 Commandes recommandées:")
        console.print("   • python test_full_system_video.py --video webcam --max-frames 50")
        console.print("   • python main.py --video webcam")
    else:
        console.print("⚠️ [bold yellow]PROBLÈMES DÉTECTÉS[/bold yellow]")
        console.print("🔧 Vérifiez les erreurs ci-dessus et relancez setup_gpu_server.sh")
    
    # Génération rapport
    report = generate_gpu_report()
    console.print(f"\n📊 Configuration GPU: {report}")

if __name__ == "__main__":
    main()