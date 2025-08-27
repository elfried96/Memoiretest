#!/usr/bin/env python3
"""
📊 Moniteur de Surveillance en Temps Réel
=========================================

Surveille les analyses vidéo en cours et affiche les statistiques.
"""

import os
import time
import json
import psutil
from pathlib import Path
from datetime import datetime

def monitor_processes():
    """Surveille les processus d'analyse en cours."""
    surveillance_processes = []
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_percent']):
        try:
            if proc.info['name'] == 'python' and proc.info['cmdline']:
                cmdline = ' '.join(proc.info['cmdline'])
                if 'main_headless.py' in cmdline:
                    surveillance_processes.append(proc.info)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    return surveillance_processes

def get_analysis_stats(output_dirs):
    """Récupère les statistiques des analyses en cours."""
    stats = {
        "total_frames_processed": 0,
        "total_detections": 0,
        "completed_analyses": 0,
        "active_processes": 0
    }
    
    for output_dir in output_dirs:
        if os.path.exists(output_dir):
            # Compter les fichiers de résultats
            results_files = list(Path(output_dir).glob("**/*.json"))
            stats["completed_analyses"] += len(results_files)
            
            # Lire les statistiques détaillées si disponibles
            for result_file in results_files:
                try:
                    with open(result_file, 'r') as f:
                        data = json.load(f)
                        if 'frames_processed' in data:
                            stats["total_frames_processed"] += data['frames_processed']
                        if 'total_detections' in data:
                            stats["total_detections"] += data['total_detections']
                except:
                    pass
    
    return stats

def main():
    print("📊 MONITEUR DE SURVEILLANCE EN TEMPS RÉEL")
    print("=" * 50)
    print("Appuyez sur Ctrl+C pour arrêter le monitoring\n")
    
    try:
        while True:
            # Clear screen (fonctionne sur la plupart des terminaux)
            os.system('clear' if os.name == 'posix' else 'cls')
            
            print("🎯 SURVEILLANCE EN TEMPS RÉEL")
            print("=" * 40)
            print(f"⏰ {datetime.now().strftime('%H:%M:%S')}")
            print()
            
            # Surveillance des processus
            processes = monitor_processes()
            
            if processes:
                print(f"🔄 Processus actifs: {len(processes)}")
                print("-" * 30)
                
                for i, proc in enumerate(processes, 1):
                    # Extraire le nom de la vidéo depuis la ligne de commande
                    cmdline = ' '.join(proc['cmdline'])
                    video_name = "Inconnue"
                    
                    if '--source' in cmdline:
                        parts = cmdline.split('--source')
                        if len(parts) > 1:
                            source_part = parts[1].split()[0]
                            video_name = Path(source_part).stem
                    
                    cpu = proc.get('cpu_percent', 0)
                    mem = proc.get('memory_percent', 0)
                    
                    print(f"  {i}. {video_name}")
                    print(f"     PID: {proc['pid']} | CPU: {cpu:.1f}% | RAM: {mem:.1f}%")
                
                print()
            else:
                print("😴 Aucun processus de surveillance actif")
                print()
            
            # Statistiques système
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            print("💻 SYSTÈME")
            print("-" * 15)
            print(f"CPU Global: {cpu_usage:.1f}%")
            print(f"RAM: {memory.used // (1024**3):.1f}GB / {memory.total // (1024**3):.1f}GB ({memory.percent:.1f}%)")
            print()
            
            # Recherche des répertoires de sortie récents
            recent_dirs = []
            for item in Path('.').iterdir():
                if item.is_dir() and 'multi_analysis_' in item.name:
                    recent_dirs.append(item)
            
            if recent_dirs:
                recent_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                latest_dir = recent_dirs[0]
                
                print("📂 DERNIÈRE ANALYSE")
                print("-" * 20)
                print(f"Répertoire: {latest_dir.name}")
                
                # Compter les fichiers de résultats
                result_files = list(latest_dir.glob("**/*.json"))
                image_files = list(latest_dir.glob("**/*.jpg")) + list(latest_dir.glob("**/*.png"))
                
                print(f"Résultats JSON: {len(result_files)}")
                print(f"Images sauvées: {len(image_files)}")
                
                # Rapport final si disponible
                report_file = latest_dir / "analysis_report.json"
                if report_file.exists():
                    try:
                        with open(report_file, 'r') as f:
                            report = json.load(f)
                            summary = report.get('summary', {})
                            print(f"Vidéos traitées: {summary.get('successful', 0)}/{summary.get('total_videos', 0)}")
                    except:
                        pass
                
                print()
            
            print("🔄 Actualisation toutes les 5 secondes...")
            print("Press Ctrl+C pour arrêter")
            
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\n\n✋ Monitoring arrêté")
        print("👋 Au revoir !")

if __name__ == "__main__":
    main()