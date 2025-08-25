#!/usr/bin/env python3
"""
🎬 Test Simple de Vidéo avec Kimi-VL
===================================

Script de test simplifié pour tester les vidéos téléchargées de Kaggle
sans dépendances complexes.
"""

import sys
import cv2
import time
import argparse
from pathlib import Path

def test_video_basic(video_path: str) -> bool:
    """Test basique d'une vidéo."""
    
    print(f"🎬 Test de la vidéo: {video_path}")
    
    # Vérification existence
    if not Path(video_path).exists():
        print(f"❌ Fichier non trouvé: {video_path}")
        return False
    
    # Ouverture avec OpenCV
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Impossible d'ouvrir la vidéo: {video_path}")
        return False
    
    # Informations vidéo
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"📊 Informations vidéo:")
    print(f"  Résolution: {width}x{height}")
    print(f"  FPS: {fps:.1f}")
    print(f"  Frames: {total_frames}")
    print(f"  Durée: {duration:.1f}s")
    print(f"  Taille: {Path(video_path).stat().st_size / (1024*1024):.1f} MB")
    
    # Test lecture de quelques frames
    print(f"🔄 Test lecture des premières frames...")
    frame_count = 0
    max_test_frames = min(100, total_frames)  # Test 100 frames max
    
    start_time = time.time()
    valid_frames = 0
    
    while frame_count < max_test_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        valid_frames += 1
        frame_count += 1
        
        # Affichage progression
        if frame_count % 25 == 0:
            print(f"  Frame {frame_count}/{max_test_frames}")
    
    cap.release()
    
    read_time = time.time() - start_time
    read_fps = valid_frames / read_time if read_time > 0 else 0
    
    print(f"✅ Test terminé:")
    print(f"  Frames lues: {valid_frames}/{max_test_frames}")
    print(f"  Vitesse lecture: {read_fps:.1f} FPS")
    print(f"  Temps: {read_time:.2f}s")
    
    success = valid_frames >= max_test_frames * 0.8  # 80% des frames lues
    if success:
        print(f"🎉 Vidéo valide pour les tests !")
    else:
        print(f"⚠️ Problème potentiel avec cette vidéo")
    
    return success

def list_videos_in_directory(directory: str):
    """Liste les vidéos dans un dossier."""
    
    video_dir = Path(directory)
    if not video_dir.exists():
        print(f"❌ Dossier non trouvé: {directory}")
        return []
    
    # Extensions vidéo supportées
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(video_dir.glob(f"*{ext}"))
        video_files.extend(video_dir.glob(f"*{ext.upper()}"))
    
    return sorted(video_files)

def test_all_videos_in_directory(directory: str):
    """Test toutes les vidéos d'un dossier."""
    
    print(f"📁 Test de toutes les vidéos dans: {directory}")
    print("="*60)
    
    video_files = list_videos_in_directory(directory)
    
    if not video_files:
        print(f"❌ Aucune vidéo trouvée dans {directory}")
        return
    
    print(f"🎬 {len(video_files)} vidéos trouvées:")
    for video in video_files:
        print(f"  • {video.name}")
    
    print("\n" + "="*60)
    
    results = []
    for i, video_file in enumerate(video_files, 1):
        print(f"\n🔄 Test {i}/{len(video_files)}: {video_file.name}")
        print("-" * 40)
        
        success = test_video_basic(str(video_file))
        results.append((video_file.name, success))
    
    # Résumé
    print("\n" + "="*60)
    print("📊 RÉSUMÉ DES TESTS")
    print("="*60)
    
    successful = sum(1 for _, success in results if success)
    
    for name, success in results:
        status = "✅" if success else "❌"
        print(f"  {status} {name}")
    
    print(f"\n🏆 Résultats: {successful}/{len(results)} vidéos valides")
    
    if successful == len(results):
        print("🎉 Toutes les vidéos sont prêtes pour les tests avec Kimi-VL !")
    elif successful > 0:
        print("⚠️ Certaines vidéos ont des problèmes, mais d'autres sont utilisables")
    else:
        print("💥 Aucune vidéo utilisable trouvée")

def create_argument_parser():
    """Parser d'arguments."""
    parser = argparse.ArgumentParser(
        description="Test simple de vidéos pour le système de surveillance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  python test_simple_video.py videos/surveillance_test.mp4     # Test une vidéo
  python test_simple_video.py --directory videos/             # Test toutes vidéos du dossier
  python test_simple_video.py --list videos/                  # Lister vidéos sans tester
        """
    )
    
    parser.add_argument(
        "video_path",
        nargs="?",
        help="Chemin vers la vidéo à tester"
    )
    
    parser.add_argument(
        "--directory", "-d",
        help="Tester toutes les vidéos d'un dossier"
    )
    
    parser.add_argument(
        "--list", "-l",
        help="Lister les vidéos d'un dossier sans les tester"
    )
    
    return parser

def main():
    """Point d'entrée principal."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    print("🎬 Test Simple de Vidéos")
    print("=" * 30)
    
    if args.list:
        video_files = list_videos_in_directory(args.list)
        print(f"📁 Vidéos dans {args.list}:")
        if video_files:
            for video in video_files:
                size_mb = video.stat().st_size / (1024*1024)
                print(f"  • {video.name} ({size_mb:.1f} MB)")
        else:
            print("  Aucune vidéo trouvée")
    
    elif args.directory:
        test_all_videos_in_directory(args.directory)
    
    elif args.video_path:
        test_video_basic(args.video_path)
    
    else:
        print("❌ Veuillez spécifier une vidéo ou un dossier à tester")
        parser.print_help()

if __name__ == "__main__":
    main()