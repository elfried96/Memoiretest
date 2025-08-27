#!/usr/bin/env python3
"""
🎬 Analyse Multiple de Vidéos de Surveillance
===========================================

Lance l'analyse de toutes les vidéos en parallèle avec des processus séparés.
"""

import os
import subprocess
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from datetime import datetime

def run_surveillance_on_video(video_path, output_dir):
    """Lance l'analyse de surveillance sur une vidéo."""
    video_name = Path(video_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Répertoire de sortie spécifique pour cette vidéo
    video_output = f"{output_dir}/{video_name}_{timestamp}"
    
    print(f"🎬 Démarrage analyse: {video_name}")
    
    try:
        # Commande pour lancer l'analyse
        cmd = [
            "uv", "run", "main_headless.py",
            "--video", video_path,
            "--save-frames",          # Sauvegarder les frames importantes
            "--max-frames", "500"     # Limiter à 500 frames pour éviter surcharge
        ]
        
        start_time = time.time()
        
        # Exécution avec capture des logs
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800  # Timeout de 30 minutes
        )
        
        duration = time.time() - start_time
        
        # Résultats
        analysis_result = {
            "video": video_name,
            "video_path": video_path,
            "output_directory": video_output,
            "duration_seconds": round(duration, 2),
            "success": result.returncode == 0,
            "stdout": result.stdout[-2000:] if result.stdout else "",  # Derniers 2000 chars
            "stderr": result.stderr[-1000:] if result.stderr else "",   # Derniers 1000 chars
            "return_code": result.returncode,
            "timestamp": datetime.now().isoformat()
        }
        
        if result.returncode == 0:
            print(f"✅ {video_name}: Analyse terminée ({duration:.1f}s)")
        else:
            print(f"❌ {video_name}: Échec après {duration:.1f}s")
            print(f"   Erreur: {result.stderr[-200:]}")  # Afficher les dernières erreurs
        
        return analysis_result
        
    except subprocess.TimeoutExpired:
        print(f"⏰ {video_name}: Timeout après 30 minutes")
        return {
            "video": video_name,
            "video_path": video_path,
            "success": False,
            "error": "Timeout après 30 minutes",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        print(f"💥 {video_name}: Erreur - {e}")
        return {
            "video": video_name,
            "video_path": video_path,
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

def main():
    print("🎯 ANALYSE MULTIPLE DE VIDÉOS DE SURVEILLANCE")
    print("=" * 50)
    
    # Répertoire des vidéos
    videos_dir = Path("videos")
    if not videos_dir.exists():
        print("❌ Répertoire 'videos' introuvable")
        return
    
    # Recherche des fichiers vidéo
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
    video_files = []
    
    for file in videos_dir.iterdir():
        if file.suffix.lower() in video_extensions:
            video_files.append(file)
    
    if not video_files:
        print("❌ Aucune vidéo trouvée dans le répertoire 'videos'")
        return
    
    print(f"📹 {len(video_files)} vidéos détectées:")
    for i, video in enumerate(video_files, 1):
        size_mb = video.stat().st_size / (1024 * 1024)
        print(f"   {i}. {video.name} ({size_mb:.1f} MB)")
    
    # Répertoire de sortie global
    output_base = f"multi_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_base, exist_ok=True)
    
    print(f"\n🗂️ Résultats dans: {output_base}")
    print(f"🚀 Lancement de {len(video_files)} analyses en parallèle...")
    print("⏳ Cela peut prendre plusieurs minutes...\n")
    
    # Analyse en parallèle avec limitation du nombre de processus simultanés
    max_workers = min(3, len(video_files))  # Max 3 processus simultanés pour éviter surcharge
    results = []
    
    start_total = time.time()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Soumission des tâches
        future_to_video = {
            executor.submit(run_surveillance_on_video, str(video), output_base): video 
            for video in video_files
        }
        
        # Collecte des résultats au fur et à mesure
        for future in as_completed(future_to_video):
            video = future_to_video[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                print(f"💥 {video.name}: Exception - {exc}")
                results.append({
                    "video": video.name,
                    "video_path": str(video),
                    "success": False,
                    "error": str(exc),
                    "timestamp": datetime.now().isoformat()
                })
    
    total_duration = time.time() - start_total
    
    # Rapport final
    print("\n" + "=" * 60)
    print("📊 RAPPORT FINAL D'ANALYSE")
    print("=" * 60)
    
    successful = [r for r in results if r.get("success", False)]
    failed = [r for r in results if not r.get("success", False)]
    
    print(f"✅ Réussies: {len(successful)}/{len(results)}")
    print(f"❌ Échouées: {len(failed)}/{len(results)}")
    print(f"⏱️ Durée totale: {total_duration:.1f}s ({total_duration/60:.1f} min)")
    
    if successful:
        print("\n🎉 Analyses réussies:")
        for result in successful:
            print(f"   • {result['video']}: {result['duration_seconds']}s")
    
    if failed:
        print("\n💥 Analyses échouées:")
        for result in failed:
            error = result.get('error', 'Erreur inconnue')
            print(f"   • {result['video']}: {error}")
    
    # Sauvegarde du rapport complet
    report_file = f"{output_base}/analysis_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump({
            "summary": {
                "total_videos": len(results),
                "successful": len(successful),
                "failed": len(failed),
                "total_duration_seconds": total_duration,
                "analysis_timestamp": datetime.now().isoformat()
            },
            "results": results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n📋 Rapport complet sauvé: {report_file}")
    print(f"📁 Tous les résultats dans: {output_base}/")

if __name__ == "__main__":
    main()