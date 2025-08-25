#!/usr/bin/env python3
"""
🎯 Tests Vidéo en Lot (Batch) avec Kimi-VL
==========================================

Script pour tester l'architecture complète sur plusieurs vidéos :
- Tests automatisés sur tous les datasets
- Comparaison de performance entre profils
- Génération de rapports consolidés
- Analyse statistique des résultats

Usage:
    python scripts/batch_video_test.py --all-categories
    python scripts/batch_video_test.py --category surveillance_basic
    python scripts/batch_video_test.py --compare-profiles
"""

import sys
import asyncio
import time
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import statistics

# Configuration du path  
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.video_test_config import (
    VIDEO_DATASETS_ROOT,
    get_video_test_config,
    get_system_config_for_video_test
)
from scripts.test_video_pipeline import VideoTestPipeline, VideoTestResult


class BatchVideoTester:
    """Testeur de vidéos en lot."""
    
    def __init__(self):
        self.datasets_root = VIDEO_DATASETS_ROOT
        self.results_dir = VIDEO_DATASETS_ROOT / "batch_results"
        self.results_dir.mkdir(exist_ok=True)
        
        self.batch_results = {
            "timestamp": datetime.now().isoformat(),
            "total_videos": 0,
            "successful_tests": 0,
            "failed_tests": 0,
            "categories_tested": [],
            "profiles_tested": [],
            "results_by_category": {},
            "results_by_profile": {},
            "performance_summary": {},
            "errors": []
        }
    
    async def test_category(self, category: str, profile: str = "standard") -> Dict[str, Any]:
        """Test d'une catégorie complète de vidéos."""
        print(f"🎬 Test catégorie: {category} (profil: {profile})")
        
        category_dir = self.datasets_root / category
        if not category_dir.exists():
            error_msg = f"Catégorie {category} non trouvée"
            print(f"❌ {error_msg}")
            self.batch_results["errors"].append(error_msg)
            return {"error": error_msg}
        
        # Trouver toutes les vidéos MP4 dans la catégorie
        video_files = list(category_dir.glob("*.mp4"))
        if not video_files:
            error_msg = f"Aucune vidéo trouvée dans {category}"
            print(f"❌ {error_msg}")
            self.batch_results["errors"].append(error_msg)
            return {"error": error_msg}
        
        print(f"📹 {len(video_files)} vidéos trouvées")
        
        # Configuration pour cette catégorie
        video_config = get_video_test_config(profile)
        system_config = get_system_config_for_video_test()
        
        # Résultats de la catégorie
        category_results = {
            "category": category,
            "profile": profile,
            "total_videos": len(video_files),
            "successful_videos": 0,
            "failed_videos": 0,
            "individual_results": [],
            "aggregated_stats": {},
            "processing_time_total": 0.0
        }
        
        # Test de chaque vidéo
        pipeline = VideoTestPipeline(video_config, system_config)
        
        try:
            for idx, video_file in enumerate(video_files, 1):
                print(f"\n🔄 Test {idx}/{len(video_files)}: {video_file.name}")
                
                try:
                    start_time = time.time()
                    result = await pipeline.process_video(str(video_file))
                    test_duration = time.time() - start_time
                    
                    if len(result.errors) == 0:
                        category_results["successful_videos"] += 1
                        self.batch_results["successful_tests"] += 1
                        print(f"  ✅ Succès en {test_duration:.1f}s")
                    else:
                        category_results["failed_videos"] += 1
                        self.batch_results["failed_tests"] += 1
                        print(f"  ⚠️ Terminé avec {len(result.errors)} erreurs")
                    
                    # Ajout du résultat individuel
                    individual_result = {
                        "video_name": video_file.name,
                        "video_path": str(video_file),
                        "test_duration": test_duration,
                        "result": {
                            "processed_frames": result.processed_frames,
                            "fps_average": result.fps_average,
                            "detections_total": result.detections_total,
                            "persons_detected": result.persons_detected,
                            "vlm_analyses": result.vlm_analyses,
                            "vlm_average_confidence": result.vlm_average_confidence,
                            "alerts_total": result.alerts_normal + result.alerts_attention + result.alerts_critique,
                            "errors_count": len(result.errors)
                        }
                    }
                    category_results["individual_results"].append(individual_result)
                    category_results["processing_time_total"] += test_duration
                    
                except Exception as e:
                    error_msg = f"Erreur test {video_file.name}: {str(e)}"
                    print(f"  ❌ {error_msg}")
                    category_results["failed_videos"] += 1
                    self.batch_results["failed_tests"] += 1
                    self.batch_results["errors"].append(error_msg)
        
        finally:
            pipeline.cleanup()
        
        # Calcul des statistiques agrégées
        if category_results["individual_results"]:
            category_results["aggregated_stats"] = self._calculate_aggregated_stats(
                category_results["individual_results"]
            )
        
        self.batch_results["total_videos"] += len(video_files)
        
        return category_results
    
    def _calculate_aggregated_stats(self, individual_results: List[Dict]) -> Dict[str, Any]:
        """Calcul des statistiques agrégées."""
        if not individual_results:
            return {}
        
        # Extraction des métriques
        metrics = {
            "fps_average": [],
            "detections_per_video": [],
            "persons_per_video": [],
            "vlm_analyses_per_video": [],
            "vlm_confidence": [],
            "alerts_per_video": [],
            "processing_time": []
        }
        
        for result in individual_results:
            r = result["result"]
            metrics["fps_average"].append(r["fps_average"])
            metrics["detections_per_video"].append(r["detections_total"])
            metrics["persons_per_video"].append(r["persons_detected"])
            metrics["vlm_analyses_per_video"].append(r["vlm_analyses"])
            if r["vlm_average_confidence"] > 0:
                metrics["vlm_confidence"].append(r["vlm_average_confidence"])
            metrics["alerts_per_video"].append(r["alerts_total"])
            metrics["processing_time"].append(result["test_duration"])
        
        # Calcul des statistiques
        stats = {}
        for metric_name, values in metrics.items():
            if values:
                stats[metric_name] = {
                    "mean": statistics.mean(values),
                    "median": statistics.median(values),
                    "min": min(values),
                    "max": max(values),
                    "std_dev": statistics.stdev(values) if len(values) > 1 else 0
                }
        
        return stats
    
    async def test_all_categories(self, profile: str = "standard") -> Dict[str, Any]:
        """Test de toutes les catégories disponibles."""
        print(f"🎯 Test de toutes les catégories (profil: {profile})")
        
        categories = [
            "surveillance_basic",
            "theft_scenarios", 
            "crowded_scenes",
            "edge_cases",
            "synthetic"
        ]
        
        all_results = {}
        
        for category in categories:
            category_dir = self.datasets_root / category
            if category_dir.exists() and list(category_dir.glob("*.mp4")):
                print(f"\n{'='*60}")
                result = await self.test_category(category, profile)
                all_results[category] = result
                self.batch_results["categories_tested"].append(category)
            else:
                print(f"⚠️ Catégorie {category} ignorée (pas de vidéos)")
        
        if profile not in self.batch_results["profiles_tested"]:
            self.batch_results["profiles_tested"].append(profile)
        
        self.batch_results["results_by_category"] = all_results
        
        return all_results
    
    async def compare_profiles(self, categories: Optional[List[str]] = None) -> Dict[str, Any]:
        """Comparaison entre différents profils de test."""
        print("⚖️ Comparaison des profils de test")
        
        profiles_to_test = ["fast", "standard", "thorough"]
        test_categories = categories or ["surveillance_basic", "synthetic"]
        
        comparison_results = {
            "profiles_compared": profiles_to_test,
            "categories_used": test_categories,
            "results_by_profile": {},
            "performance_comparison": {}
        }
        
        # Test avec chaque profil
        for profile in profiles_to_test:
            print(f"\n🔄 Test avec profil: {profile}")
            profile_results = {}
            
            for category in test_categories:
                if (self.datasets_root / category).exists():
                    result = await self.test_category(category, profile)
                    profile_results[category] = result
            
            comparison_results["results_by_profile"][profile] = profile_results
        
        # Analyse comparative
        comparison_results["performance_comparison"] = self._analyze_profile_performance(
            comparison_results["results_by_profile"]
        )
        
        return comparison_results
    
    def _analyze_profile_performance(self, results_by_profile: Dict) -> Dict[str, Any]:
        """Analyse comparative des performances entre profils."""
        analysis = {
            "average_fps": {},
            "average_processing_time": {},
            "detection_rates": {},
            "vlm_performance": {},
            "recommendations": []
        }
        
        for profile, profile_data in results_by_profile.items():
            fps_values = []
            processing_times = []
            detection_totals = []
            vlm_confidences = []
            
            for category, category_data in profile_data.items():
                if "aggregated_stats" in category_data:
                    stats = category_data["aggregated_stats"]
                    if "fps_average" in stats:
                        fps_values.append(stats["fps_average"]["mean"])
                    if "processing_time" in stats:
                        processing_times.append(stats["processing_time"]["mean"])
                    if "detections_per_video" in stats:
                        detection_totals.append(stats["detections_per_video"]["mean"])
                    if "vlm_confidence" in stats:
                        vlm_confidences.append(stats["vlm_confidence"]["mean"])
            
            if fps_values:
                analysis["average_fps"][profile] = statistics.mean(fps_values)
            if processing_times:
                analysis["average_processing_time"][profile] = statistics.mean(processing_times)
            if detection_totals:
                analysis["detection_rates"][profile] = statistics.mean(detection_totals)
            if vlm_confidences:
                analysis["vlm_performance"][profile] = statistics.mean(vlm_confidences)
        
        # Recommandations basées sur l'analyse
        if "fast" in analysis["average_fps"] and "standard" in analysis["average_fps"]:
            if analysis["average_fps"]["fast"] > analysis["average_fps"]["standard"] * 1.5:
                analysis["recommendations"].append("Profile 'fast' recommandé pour temps réel")
        
        if "thorough" in analysis["vlm_performance"] and "standard" in analysis["vlm_performance"]:
            if analysis["vlm_performance"]["thorough"] > analysis["vlm_performance"]["standard"]:
                analysis["recommendations"].append("Profile 'thorough' recommandé pour précision maximale")
        
        return analysis
    
    def generate_report(self, output_file: Optional[str] = None) -> str:
        """Génération d'un rapport consolidé."""
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"batch_test_report_{timestamp}.json"
        
        report_path = self.results_dir / output_file
        
        # Ajout de métadonnées finales
        self.batch_results["report_generated"] = datetime.now().isoformat()
        self.batch_results["success_rate"] = (
            self.batch_results["successful_tests"] / 
            max(self.batch_results["total_videos"], 1) * 100
        )
        
        # Sauvegarde du rapport
        with open(report_path, 'w') as f:
            json.dump(self.batch_results, f, indent=2, default=str)
        
        print(f"📄 Rapport sauvé: {report_path}")
        return str(report_path)
    
    def print_summary(self):
        """Affichage du résumé des tests."""
        print("\n" + "="*60)
        print("🏆 RÉSUMÉ DES TESTS BATCH")
        print("="*60)
        
        print(f"📊 Statistiques Générales:")
        print(f"  Vidéos testées: {self.batch_results['total_videos']}")
        print(f"  Tests réussis: {self.batch_results['successful_tests']}")
        print(f"  Tests échoués: {self.batch_results['failed_tests']}")
        print(f"  Taux de succès: {self.batch_results.get('success_rate', 0):.1f}%")
        
        if self.batch_results["categories_tested"]:
            print(f"\n🎬 Catégories testées: {', '.join(self.batch_results['categories_tested'])}")
        
        if self.batch_results["profiles_tested"]:
            print(f"⚙️ Profils testés: {', '.join(self.batch_results['profiles_tested'])}")
        
        if self.batch_results["errors"]:
            print(f"\n❌ Erreurs ({len(self.batch_results['errors'])}):")
            for error in self.batch_results["errors"][:5]:
                print(f"  • {error}")
            if len(self.batch_results["errors"]) > 5:
                print(f"  ... et {len(self.batch_results['errors']) - 5} autres erreurs")


def create_argument_parser() -> argparse.ArgumentParser:
    """Créer le parser d'arguments."""
    parser = argparse.ArgumentParser(
        description="Tests vidéo en lot (batch) avec Kimi-VL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  python scripts/batch_video_test.py --all-categories                    # Toutes catégories
  python scripts/batch_video_test.py --category surveillance_basic      # Une catégorie
  python scripts/batch_video_test.py --compare-profiles                 # Comparaison profils
  python scripts/batch_video_test.py --all-categories --profile thorough # Profil spécifique
        """
    )
    
    parser.add_argument(
        "--all-categories",
        action="store_true",
        help="Tester toutes les catégories disponibles"
    )
    
    parser.add_argument(
        "--category",
        type=str,
        help="Tester une catégorie spécifique"
    )
    
    parser.add_argument(
        "--compare-profiles",
        action="store_true",
        help="Comparer différents profils de test"
    )
    
    parser.add_argument(
        "--profile",
        type=str,
        default="standard",
        choices=["fast", "standard", "thorough", "demo"],
        help="Profil de configuration à utiliser"
    )
    
    parser.add_argument(
        "--report-file",
        type=str,
        help="Nom du fichier de rapport (auto si non spécifié)"
    )
    
    return parser


async def main():
    """Point d'entrée principal."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    if not any([args.all_categories, args.category, args.compare_profiles]):
        print("❌ Veuillez spécifier --all-categories, --category ou --compare-profiles")
        parser.print_help()
        return 1
    
    print("🎯 Tests Vidéo Batch avec Kimi-VL")
    print(f"⚙️ Profil: {args.profile}")
    print("="*60)
    
    tester = BatchVideoTester()
    
    try:
        if args.all_categories:
            await tester.test_all_categories(args.profile)
        
        elif args.category:
            result = await tester.test_category(args.category, args.profile)
            tester.batch_results["results_by_category"][args.category] = result
            tester.batch_results["categories_tested"].append(args.category)
        
        elif args.compare_profiles:
            comparison = await tester.compare_profiles()
            tester.batch_results["profile_comparison"] = comparison
        
        # Génération du rapport et résumé
        report_path = tester.generate_report(args.report_file)
        tester.print_summary()
        
        print(f"\n📄 Rapport détaillé: {report_path}")
        
        # Code de sortie selon les résultats
        success_rate = tester.batch_results.get("success_rate", 0)
        if success_rate >= 80:
            print("🎉 Tests majoritairement réussis !")
            return 0
        elif success_rate >= 50:
            print("⚠️ Tests partiellement réussis")
            return 1
        else:
            print("💥 Échec majoritaire des tests")
            return 2
            
    except KeyboardInterrupt:
        print("\n⚠️ Interruption utilisateur")
        return 1
    except Exception as e:
        print(f"\n❌ Erreur inattendue: {e}")
        return 1


if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        sys.exit(result)
    except Exception as e:
        print(f"Erreur fatale: {e}")
        sys.exit(1)