#!/usr/bin/env python3
"""
Script d'évaluation du système de surveillance intelligent.
Mesure les performances réelles de votre système.
"""

import sys
import time
import json
import os
from typing import Dict, List, Any
from datetime import datetime
import cv2
import numpy as np

sys.path.append('/home/elfried-kinzoun/PycharmProjects/intelligent-surveillance-system')

from main_Qwen import QwenOnlySurveillanceSystem
from src.core.types import SuspicionLevel, ActionType

class SystemEvaluator:
    def __init__(self):
        self.system = QwenOnlySurveillanceSystem()
        self.metrics = {
            'total_frames': 0,
            'processing_times': [],
            'detections': [],
            'suspicion_levels': [],
            'action_types': [],
            'errors': []
        }
    
    def evaluate_video(self, video_path: str, ground_truth: Dict = None) -> Dict[str, Any]:
        """Évalue le système sur une vidéo spécifique."""
        print(f"🎥 Évaluation de la vidéo: {video_path}")
        
        if not os.path.exists(video_path):
            print(f"❌ Vidéo non trouvée: {video_path}")
            return {}
        
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        results = []
        
        while cap.read()[0]:
            frame_count += 1
        
        cap.release()
        print(f"📊 Vidéo contient {frame_count} frames")
        
        # Analyser la vidéo avec votre système
        start_time = time.time()
        
        try:
            # Utiliser votre système existant
            analysis_result = self.system.analyze_video(video_path)
            
            processing_time = time.time() - start_time
            
            # Calculer les métriques
            metrics = self._calculate_metrics(analysis_result, processing_time, frame_count)
            
            return metrics
            
        except Exception as e:
            print(f"❌ Erreur lors de l'analyse: {e}")
            return {'error': str(e)}
    
    def _calculate_metrics(self, result: Any, processing_time: float, frame_count: int) -> Dict[str, Any]:
        """Calcule les métriques de performance."""
        
        # Métriques de base
        fps = frame_count / processing_time if processing_time > 0 else 0
        latency_per_frame = processing_time / frame_count if frame_count > 0 else 0
        
        metrics = {
            'performance': {
                'total_processing_time': round(processing_time, 2),
                'fps': round(fps, 2),
                'latency_per_frame': round(latency_per_frame * 1000, 2),  # en ms
                'total_frames': frame_count
            }
        }
        
        # Analyser le résultat si c'est un objet AnalysisResponse
        if hasattr(result, 'suspicion_level'):
            metrics['detection'] = {
                'suspicion_level': result.suspicion_level.value,
                'action_type': result.action_type.value,
                'confidence': result.confidence,
                'tools_used': result.tools_used,
                'recommendations_count': len(result.recommendations)
            }
        
        return metrics
    
    def run_benchmark(self, video_directory: str = None) -> Dict[str, Any]:
        """Lance un benchmark complet du système."""
        print("🚀 Lancement du benchmark du système de surveillance")
        
        # Vidéos de test (vous pouvez ajouter vos propres vidéos)
        test_videos = []
        
        if video_directory and os.path.exists(video_directory):
            for file in os.listdir(video_directory):
                if file.endswith(('.mp4', '.avi', '.mov')):
                    test_videos.append(os.path.join(video_directory, file))
        
        if not test_videos:
            print("⚠️ Aucune vidéo de test trouvée. Créons un test synthétique...")
            return self._synthetic_benchmark()
        
        # Évaluer chaque vidéo
        all_results = []
        for video_path in test_videos:
            result = self.evaluate_video(video_path)
            if result:
                all_results.append(result)
        
        # Calculer les statistiques globales
        global_metrics = self._aggregate_metrics(all_results)
        
        return global_metrics
    
    def _synthetic_benchmark(self) -> Dict[str, Any]:
        """Benchmark synthétique avec des données simulées."""
        print("🔬 Test synthétique du système...")
        
        # Test des composants individuels
        metrics = {}
        
        # 1. Test de l'initialisation
        start_time = time.time()
        try:
            system = QwenOnlySurveillanceSystem()
            init_time = time.time() - start_time
            metrics['initialization'] = {
                'success': True,
                'time': round(init_time, 3)
            }
        except Exception as e:
            metrics['initialization'] = {
                'success': False,
                'error': str(e),
                'time': -1
            }
        
        # 2. Test des types Pydantic
        start_time = time.time()
        try:
            from src.core.types import AnalysisResponse, SuspicionLevel, ActionType
            
            response = AnalysisResponse(
                suspicion_level=SuspicionLevel.MEDIUM,
                action_type=ActionType.SUSPICIOUS_MOVEMENT,
                confidence=0.75,
                description="Test synthétique",
                reasoning="Test de performance",
                tools_used=["test"],
                recommendations=["test recommendation"],
                timestamp=datetime.now()
            )
            
            # Test de sérialisation
            serialized = response.model_dump()
            serialization_time = time.time() - start_time
            
            metrics['pydantic_serialization'] = {
                'success': True,
                'time': round(serialization_time * 1000, 3),  # en ms
                'keys_count': len(serialized)
            }
            
        except Exception as e:
            metrics['pydantic_serialization'] = {
                'success': False,
                'error': str(e)
            }
        
        # 3. Test TemporalTransformer
        start_time = time.time()
        try:
            from src.advanced_tools.temporal_transformer import TemporalTransformer
            
            transformer = TemporalTransformer()
            result = transformer.analyze_sequence([0.1, 0.5, 0.8, 0.3, 0.9], "behavior")
            transform_time = time.time() - start_time
            
            metrics['temporal_transformer'] = {
                'success': True,
                'time': round(transform_time * 1000, 3),  # en ms
                'result_keys': list(result.keys()) if isinstance(result, dict) else []
            }
            
        except Exception as e:
            metrics['temporal_transformer'] = {
                'success': False,
                'error': str(e)
            }
        
        return {
            'benchmark_type': 'synthetic',
            'timestamp': datetime.now().isoformat(),
            'components': metrics,
            'overall_health': all(m.get('success', False) for m in metrics.values())
        }
    
    def _aggregate_metrics(self, results: List[Dict]) -> Dict[str, Any]:
        """Agrège les métriques de plusieurs évaluations."""
        if not results:
            return {}
        
        # Calculer moyennes, min, max
        processing_times = [r.get('performance', {}).get('total_processing_time', 0) for r in results]
        fps_values = [r.get('performance', {}).get('fps', 0) for r in results]
        
        return {
            'summary': {
                'total_videos_tested': len(results),
                'avg_processing_time': np.mean(processing_times),
                'avg_fps': np.mean(fps_values),
                'min_fps': np.min(fps_values),
                'max_fps': np.max(fps_values)
            },
            'detailed_results': results
        }
    
    def save_results(self, results: Dict, filename: str = None):
        """Sauvegarde les résultats d'évaluation."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_results_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Résultats sauvegardés dans: {filename}")

def main():
    """Fonction principale d'évaluation."""
    evaluator = SystemEvaluator()
    
    print("🎯 Évaluation du Système de Surveillance Intelligent")
    print("=" * 50)
    
    # Lancer le benchmark
    results = evaluator.run_benchmark()
    
    # Afficher les résultats
    print("\n📊 RÉSULTATS D'ÉVALUATION:")
    print("=" * 30)
    
    if results.get('benchmark_type') == 'synthetic':
        print("Type: Test Synthétique")
        print(f"Santé Globale: {'✅' if results.get('overall_health') else '❌'}")
        
        for component, metrics in results.get('components', {}).items():
            status = "✅" if metrics.get('success') else "❌"
            time_info = f" ({metrics.get('time', 'N/A')}ms)" if 'time' in metrics else ""
            print(f"  {component}: {status}{time_info}")
            
            if not metrics.get('success') and 'error' in metrics:
                print(f"    Erreur: {metrics['error']}")
    
    # Sauvegarder
    evaluator.save_results(results)
    
    print(f"\n🎉 Évaluation terminée!")
    return results

if __name__ == "__main__":
    main()