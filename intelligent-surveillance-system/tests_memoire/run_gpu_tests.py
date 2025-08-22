#!/usr/bin/env python3
"""
🚀 Script d'Exécution Tests GPU - Environnement Mémoire
======================================================

Script optimisé pour environnement GPU qui :
- Vérifie la configuration GPU
- Lance les tests avec monitoring
- Génère des rapports détaillés pour le mémoire
- Collecte les métriques de performance

Usage:
    python tests_memoire/run_gpu_tests.py [--quick] [--unit-only] [--integration-only]
"""

import os
import sys
import time
import json
import subprocess
import argparse
from pathlib import Path
from typing import Dict, Any, List

# Ajout du path pour imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pytest
import numpy as np

def check_gpu_environment() -> Dict[str, Any]:
    """Vérification environnement GPU."""
    print("🔍 Vérification Environnement GPU...")
    
    env_info = {
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": 0,
        "gpu_details": []
    }
    
    if torch.cuda.is_available():
        env_info["gpu_count"] = torch.cuda.device_count()
        env_info["cuda_version"] = torch.version.cuda
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpu_info = {
                "device_id": i,
                "name": props.name,
                "memory_total_gb": props.total_memory / (1024**3),
                "compute_capability": f"{props.major}.{props.minor}",
                "multiprocessor_count": props.multi_processor_count
            }
            env_info["gpu_details"].append(gpu_info)
            print(f"   GPU {i}: {props.name} ({props.total_memory / (1024**3):.1f}GB)")
    else:
        print("   ⚠️ GPU non disponible - Tests CPU uniquement")
    
    return env_info

def setup_test_environment():
    """Configuration environnement de test."""
    print("⚙️ Configuration Environnement Tests...")
    
    # Création répertoires
    test_dirs = [
        "tests_memoire/data",
        "tests_memoire/reports", 
        "tests_memoire/logs"
    ]
    
    for dir_path in test_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # Variables d'environnement optimisées
    os.environ.update({
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512",
        "CUDA_LAUNCH_BLOCKING": "0",  # Async pour performance
        "PYTHONPATH": str(Path.cwd()),
        "TOKENIZERS_PARALLELISM": "false"  # Éviter warnings
    })
    
    print("✅ Environnement configuré")

def run_pytest_tests(test_type: str = "all", quick: bool = False) -> Dict[str, Any]:
    """Exécution tests pytest avec collecte métriques."""
    print(f"🧪 Lancement Tests: {test_type}")
    
    # Configuration arguments pytest
    pytest_args = [
        "tests_memoire/",
        "-v",
        "--tb=short",
        "--capture=no",
        "--color=yes"
    ]
    
    # Sélection tests selon type
    if test_type == "unit":
        pytest_args.extend(["-m", "unit or not integration"])
    elif test_type == "integration":
        pytest_args.extend(["-m", "integration"])
    elif test_type == "gpu":
        pytest_args.extend(["-m", "gpu"])
    elif test_type == "performance":
        pytest_args.extend(["-m", "performance"])
    
    # Mode rapide
    if quick:
        pytest_args.extend(["-x", "--maxfail=3"])  # Stop au premier échec
    
    # Rapports
    timestamp = int(time.time())
    report_file = f"tests_memoire/reports/test_report_{timestamp}.html"
    junit_file = f"tests_memoire/reports/junit_{timestamp}.xml"
    
    pytest_args.extend([
        f"--html={report_file}",
        f"--junitxml={junit_file}",
        "--self-contained-html"
    ])
    
    # Exécution
    start_time = time.time()
    
    try:
        exit_code = pytest.main(pytest_args)
        execution_time = time.time() - start_time
        
        result = {
            "exit_code": exit_code,
            "execution_time": execution_time,
            "success": exit_code == 0,
            "report_file": report_file,
            "junit_file": junit_file
        }
        
        if exit_code == 0:
            print(f"✅ Tests réussis en {execution_time:.1f}s")
        else:
            print(f"❌ Tests échoués (code {exit_code}) en {execution_time:.1f}s")
        
        return result
    
    except Exception as e:
        print(f"💥 Erreur exécution tests: {e}")
        return {
            "exit_code": -1,
            "success": False,
            "error": str(e),
            "execution_time": time.time() - start_time
        }

def collect_system_metrics() -> Dict[str, Any]:
    """Collecte métriques système pendant tests."""
    print("📊 Collecte Métriques Système...")
    
    metrics = {
        "timestamp": time.time(),
        "cpu_info": {},
        "memory_info": {},
        "gpu_info": {}
    }
    
    try:
        import psutil
        
        # CPU
        metrics["cpu_info"] = {
            "cpu_count": psutil.cpu_count(),
            "cpu_percent": psutil.cpu_percent(interval=1),
            "load_avg": os.getloadavg() if hasattr(os, 'getloadavg') else None
        }
        
        # Mémoire
        mem = psutil.virtual_memory()
        metrics["memory_info"] = {
            "total_gb": mem.total / (1024**3),
            "available_gb": mem.available / (1024**3), 
            "percent_used": mem.percent
        }
        
    except ImportError:
        print("   ⚠️ psutil non disponible pour métriques système")
    
    # GPU
    if torch.cuda.is_available():
        try:
            metrics["gpu_info"] = {
                "memory_allocated_mb": torch.cuda.memory_allocated() / (1024**2),
                "memory_reserved_mb": torch.cuda.memory_reserved() / (1024**2),
                "device_count": torch.cuda.device_count()
            }
        except Exception as e:
            print(f"   ⚠️ Erreur métriques GPU: {e}")
    
    return metrics

def generate_performance_report(results: Dict[str, Any], env_info: Dict[str, Any]):
    """Génération rapport performance pour mémoire."""
    print("📝 Génération Rapport Performance...")
    
    timestamp = int(time.time())
    report_path = f"tests_memoire/reports/performance_report_{timestamp}.json"
    
    # Collecte métriques finales
    final_metrics = collect_system_metrics()
    
    # Structure rapport
    report = {
        "metadata": {
            "timestamp": timestamp,
            "test_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "environment": env_info,
            "system_metrics": final_metrics
        },
        "test_results": results,
        "performance_analysis": {
            "gpu_acceleration": env_info["cuda_available"],
            "execution_summary": {
                "total_time": results.get("execution_time", 0),
                "success_rate": 1.0 if results.get("success", False) else 0.0,
                "exit_code": results.get("exit_code", -1)
            }
        },
        "recommendations": []
    }
    
    # Analyse et recommandations
    if env_info["cuda_available"]:
        gpu_memory_gb = env_info["gpu_details"][0]["memory_total_gb"] if env_info["gpu_details"] else 0
        if gpu_memory_gb < 8:
            report["recommendations"].append("GPU avec >8GB recommandé pour tests complets")
        
        current_memory = final_metrics["gpu_info"].get("memory_allocated_mb", 0)
        if current_memory > 4000:  # >4GB
            report["recommendations"].append("Utilisation mémoire GPU élevée détectée")
    else:
        report["recommendations"].append("Tests GPU impossibles - GPU requis pour performance optimale")
    
    # Sauvegarde
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"✅ Rapport sauvegardé: {report_path}")
    return report_path

def print_summary(results: Dict[str, Any], env_info: Dict[str, Any]):
    """Affichage résumé final."""
    print("\\n" + "="*60)
    print("📋 RÉSUMÉ TESTS MÉMOIRE")
    print("="*60)
    
    # Environnement
    print(f"🖥️  Environnement:")
    print(f"   Python: {sys.version.split()[0]}")
    print(f"   PyTorch: {torch.__version__}")
    if env_info["cuda_available"]:
        print(f"   GPU: {env_info['gpu_details'][0]['name']} ({env_info['gpu_details'][0]['memory_total_gb']:.1f}GB)")
    else:
        print("   GPU: Non disponible")
    
    # Résultats
    print(f"\\n🧪 Résultats:")
    if results.get("success", False):
        print(f"   ✅ Tests RÉUSSIS en {results.get('execution_time', 0):.1f}s")
    else:
        print(f"   ❌ Tests ÉCHOUÉS (code {results.get('exit_code', -1)})")
    
    # Fichiers générés
    if "report_file" in results:
        print(f"\\n📊 Rapports générés:")
        print(f"   HTML: {results['report_file']}")
        print(f"   JUnit: {results.get('junit_file', 'N/A')}")
    
    print("="*60)

def main():
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(description="Tests GPU pour mémoire")
    parser.add_argument("--quick", action="store_true", help="Tests rapides")
    parser.add_argument("--unit-only", action="store_true", help="Tests unitaires uniquement")
    parser.add_argument("--integration-only", action="store_true", help="Tests intégration uniquement")
    parser.add_argument("--gpu-only", action="store_true", help="Tests GPU uniquement")
    parser.add_argument("--performance-only", action="store_true", help="Tests performance uniquement")
    
    args = parser.parse_args()
    
    print("🚀 TESTS MÉMOIRE - SYSTÈME SURVEILLANCE IA")
    print("="*50)
    
    # 1. Vérification environnement
    env_info = check_gpu_environment()
    
    # 2. Configuration
    setup_test_environment()
    
    # 3. Détermination type de tests
    if args.unit_only:
        test_type = "unit"
    elif args.integration_only:
        test_type = "integration"
    elif args.gpu_only:
        test_type = "gpu"
    elif args.performance_only:
        test_type = "performance"
    else:
        test_type = "all"
    
    # 4. Exécution tests
    results = run_pytest_tests(test_type, args.quick)
    
    # 5. Génération rapport
    report_path = generate_performance_report(results, env_info)
    
    # 6. Résumé
    print_summary(results, env_info)
    
    # 7. Code de sortie
    sys.exit(results.get("exit_code", 0))

if __name__ == "__main__":
    main()