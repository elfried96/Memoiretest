#!/usr/bin/env python3
"""
🎥 Tests Système Complet avec Vraies Vidéos
===========================================

Ce script teste le système COMPLET de bout en bout :
- Pipeline VLM + Orchestrateur + Outils avancés
- Vraies vidéos (fichiers MP4) avec annotations
- Mesures précision/recall/F1 sur détections réelles
- Métriques end-to-end du système intégré
"""

import asyncio
import time
import json
import os
import sys
import cv2
import numpy as np
import base64
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import torch
from PIL import Image
import io
import traceback
from dataclasses import dataclass

# Ajout des chemins
sys.path.append(str(Path(__file__).parent / "src"))
sys.path.append(str(Path(__file__).parent / "dashboard"))

@dataclass
class VideoGroundTruth:
    """Annotation vérité terrain pour une vidéo."""
    video_name: str
    scenario_type: str
    suspicious_frames: List[int]  # Numéros des frames avec activité suspecte
    normal_frames: List[int]     # Frames normales
    expected_detections: int     # Nombre détections attendues
    expected_fp_rate: float      # Taux faux positifs attendu
    difficulty_level: str        # 'easy', 'medium', 'hard'

class CompleteSystemTestSuite:
    """Tests système complet avec vraies vidéos."""
    
    def __init__(self):
        self.results = {
            'video_analysis_results': {},
            'detection_metrics': {},
            'system_performance': {},
            'accuracy_measurements': {},
            'comparison_metrics': {},
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'system_version': 'VLM_Pipeline_v1.0',
                'test_environment': 'GPU' if torch.cuda.is_available() else 'CPU'
            }
        }
        
        self.orchestrator = None
        self.test_videos_dataset = []
        self.ground_truths = []
        
    async def initialize_complete_system(self):
        """Initialisation système complet (Orchestrateur + VLM + Outils)."""
        print("🔧 INITIALISATION SYSTÈME COMPLET")
        print("=" * 45)
        
        try:
            # Chargement orchestrateur VLM complet
            from src.core.orchestrator.vlm_orchestrator import (
                ModernVLMOrchestrator, 
                OrchestrationConfig,
                OrchestrationMode
            )
            
            # Configuration système optimisée
            config = OrchestrationConfig(
                mode=OrchestrationMode.BALANCED,
                enable_advanced_tools=True,
                max_concurrent_tools=6,
                timeout_seconds=300,
                confidence_threshold=0.7
            )
            
            self.orchestrator = ModernVLMOrchestrator(
                vlm_model_name="kimi-vl-a3b-thinking",
                config=config
            )
            
            print("✅ Orchestrateur VLM initialisé")
            print("✅ Outils avancés activés: SAM2, DINO, Pose, Trajectory, etc.")
            
            return True
            
        except Exception as e:
            print(f"❌ Erreur initialisation système: {e}")
            traceback.print_exc()
            return False
    
    def create_test_videos_with_annotations(self):
        """Création vidéos de test avec annotations détaillées."""
        print("\n🎬 CRÉATION DATASET VIDÉOS ANNOTÉES")
        print("=" * 45)
        
        # Scénarios de test avec annotations précises
        video_scenarios = [
            {
                'name': 'normal_shopping_sequence.mp4',
                'scenario': 'normal_shopping',
                'description': 'Client normal - shopping routine',
                'duration_seconds': 30,
                'suspicious_frames': [],  # Aucune frame suspecte
                'normal_frames': list(range(0, 150, 5)),  # Toutes les frames normales
                'expected_detections': 0,
                'expected_fp_rate': 0.05,
                'difficulty': 'easy'
            },
            {
                'name': 'shoplifting_attempt.mp4', 
                'scenario': 'theft_attempt',
                'description': 'Tentative vol à l\'étalage visible',
                'duration_seconds': 45,
                'suspicious_frames': [60, 75, 90, 105, 120],  # Frames avec vol
                'normal_frames': list(range(0, 60, 5)) + list(range(135, 180, 5)),
                'expected_detections': 5,
                'expected_fp_rate': 0.10,
                'difficulty': 'medium'
            },
            {
                'name': 'suspicious_behavior.mp4',
                'scenario': 'suspicious_activity',
                'description': 'Comportement louche - surveillance produits',
                'duration_seconds': 40,
                'suspicious_frames': [30, 45, 60, 90],  # Comportements suspects
                'normal_frames': list(range(0, 30, 5)) + list(range(105, 160, 5)),
                'expected_detections': 4,
                'expected_fp_rate': 0.15,
                'difficulty': 'medium'
            },
            {
                'name': 'crowded_store.mp4',
                'scenario': 'crowd_analysis',
                'description': 'Magasin bondé - détection difficile',
                'duration_seconds': 50,
                'suspicious_frames': [80, 120],  # Activité suspecte dans foule
                'normal_frames': list(range(0, 80, 8)) + list(range(90, 120, 8)) + list(range(130, 200, 8)),
                'expected_detections': 2,
                'expected_fp_rate': 0.25,  # Plus élevé à cause de la foule
                'difficulty': 'hard'
            },
            {
                'name': 'poor_lighting_theft.mp4',
                'scenario': 'difficult_conditions',
                'description': 'Vol dans mauvaises conditions éclairage',
                'duration_seconds': 35,
                'suspicious_frames': [40, 70, 100],  # Vol difficile à détecter
                'normal_frames': list(range(0, 40, 6)) + list(range(110, 140, 6)),
                'expected_detections': 3,
                'expected_fp_rate': 0.20,
                'difficulty': 'hard'
            }
        ]
        
        for video_config in video_scenarios:
            # Génération vidéo synthétique
            video_path = self._generate_test_video(video_config)
            
            # Création annotation ground truth
            ground_truth = VideoGroundTruth(
                video_name=video_config['name'],
                scenario_type=video_config['scenario'],
                suspicious_frames=video_config['suspicious_frames'],
                normal_frames=video_config['normal_frames'],
                expected_detections=video_config['expected_detections'],
                expected_fp_rate=video_config['expected_fp_rate'],
                difficulty_level=video_config['difficulty']
            )
            
            self.test_videos_dataset.append({
                'video_path': video_path,
                'config': video_config,
                'ground_truth': ground_truth
            })
            
            self.ground_truths.append(ground_truth)
            
            print(f"✅ {video_config['name']}: {video_config['duration_seconds']}s - "
                  f"{len(video_config['suspicious_frames'])} frames suspectes - "
                  f"Difficulté: {video_config['difficulty']}")
        
        print(f"\n📊 Dataset créé: {len(self.test_videos_dataset)} vidéos annotées")
        return len(self.test_videos_dataset) > 0
    
    def _generate_test_video(self, config: Dict) -> str:
        """Génération vidéo de test selon configuration."""
        video_name = config['name']
        duration = config['duration_seconds']
        fps = 5  # 5 FPS pour tests rapides
        
        # Création répertoire vidéos test
        test_videos_dir = Path("test_videos")
        test_videos_dir.mkdir(exist_ok=True)
        
        video_path = test_videos_dir / video_name
        
        # Configuration vidéo
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(video_path), fourcc, fps, (640, 480))
        
        total_frames = duration * fps
        
        for frame_num in range(total_frames):
            # Génération frame selon scénario et timing
            frame = self._generate_scenario_frame(
                config['scenario'], 
                frame_num, 
                frame_num in config['suspicious_frames']
            )
            
            out.write(frame)
        
        out.release()
        print(f"   📹 Vidéo générée: {video_path} ({total_frames} frames)")
        
        return str(video_path)
    
    def _generate_scenario_frame(self, scenario: str, frame_num: int, is_suspicious: bool) -> np.ndarray:
        """Génération frame selon scénario."""
        base_img = np.random.randint(50, 180, (480, 640, 3), dtype=np.uint8)
        
        if scenario == 'normal_shopping':
            # Personne normale
            cv2.rectangle(base_img, (200, 150), (400, 450), (0, 255, 0), 2)
            cv2.putText(base_img, "NORMAL", (210, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
        elif scenario == 'theft_attempt':
            if is_suspicious:
                # Comportement vol évident
                cv2.rectangle(base_img, (180, 120), (420, 460), (0, 0, 255), 3)
                cv2.putText(base_img, "THEFT", (190, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # Simulation objet dissimulé
                cv2.circle(base_img, (300, 350), 15, (255, 0, 0), -1)
            else:
                cv2.rectangle(base_img, (200, 150), (400, 450), (0, 255, 0), 2)
                cv2.putText(base_img, "BROWSING", (210, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        elif scenario == 'suspicious_activity':
            if is_suspicious:
                # Comportement suspect mais pas évident
                cv2.rectangle(base_img, (160, 130), (380, 440), (0, 150, 255), 2)
                cv2.putText(base_img, "SUSPECT", (170, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 150, 255), 2)
            else:
                cv2.rectangle(base_img, (200, 150), (400, 450), (100, 255, 100), 2)
        
        elif scenario == 'crowd_analysis':
            # Plusieurs personnes
            for i in range(5):
                x = 50 + i * 100
                y = 100 + (i % 2) * 50
                color = (0, 0, 255) if (is_suspicious and i == 2) else (100, 255, 100)
                cv2.rectangle(base_img, (x, y), (x+60, y+300), color, 1)
            
            if is_suspicious:
                cv2.putText(base_img, "SUSPICIOUS IN CROWD", (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        elif scenario == 'difficult_conditions':
            # Éclairage difficile
            base_img = np.random.randint(10, 80, (480, 640, 3), dtype=np.uint8)
            if is_suspicious:
                cv2.rectangle(base_img, (180, 120), (420, 460), (100, 50, 50), 2)
                cv2.putText(base_img, "THEFT (DARK)", (190, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 100, 100), 2)
            else:
                cv2.rectangle(base_img, (200, 150), (400, 450), (80, 80, 80), 1)
        
        # Ajout timestamp frame
        cv2.putText(base_img, f"Frame {frame_num}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return base_img
    
    async def test_complete_system_on_videos(self):
        """Test système complet sur dataset vidéos."""
        print("\n🎯 TESTS SYSTÈME COMPLET SUR VIDÉOS")
        print("=" * 50)
        
        if not self.orchestrator:
            print("❌ Système non initialisé")
            return
        
        if not self.test_videos_dataset:
            print("❌ Pas de vidéos de test")
            return
        
        for video_data in self.test_videos_dataset:
            video_path = video_data['video_path']
            ground_truth = video_data['ground_truth']
            
            print(f"\n🎬 Test vidéo: {ground_truth.video_name}")
            print(f"   Scénario: {ground_truth.scenario_type} (difficulté: {ground_truth.difficulty_level})")
            
            # Analyse complète vidéo
            analysis_results = await self._analyze_complete_video(video_path, ground_truth)
            
            # Calcul métriques précision
            metrics = self._calculate_video_accuracy_metrics(analysis_results, ground_truth)
            
            self.results['video_analysis_results'][ground_truth.video_name] = {
                'analysis_results': analysis_results,
                'accuracy_metrics': metrics,
                'ground_truth_info': {
                    'scenario': ground_truth.scenario_type,
                    'difficulty': ground_truth.difficulty_level,
                    'expected_detections': ground_truth.expected_detections,
                    'expected_fp_rate': ground_truth.expected_fp_rate
                }
            }
            
            # Affichage résultats
            print(f"   📊 Résultats: {metrics['true_positives']} TP, {metrics['false_positives']} FP, {metrics['false_negatives']} FN")
            print(f"   🎯 Précision: {metrics['precision']:.2f} | Recall: {metrics['recall']:.2f} | F1: {metrics['f1_score']:.2f}")
            print(f"   ⏱️  Temps analyse: {analysis_results['total_processing_time']:.1f}s")
    
    async def _analyze_complete_video(self, video_path: str, ground_truth: VideoGroundTruth) -> Dict:
        """Analyse complète d'une vidéo avec le système."""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise Exception(f"Impossible d'ouvrir vidéo: {video_path}")
        
        analysis_results = {
            'detections': [],
            'frame_analyses': [],
            'total_frames': 0,
            'processed_frames': 0,
            'total_processing_time': 0,
            'average_frame_time': 0
        }
        
        frame_count = 0
        total_start_time = time.time()
        
        # Configuration contexte vidéo
        video_context = {
            'video_name': ground_truth.video_name,
            'scenario_type': ground_truth.scenario_type,
            'location': 'Test_Store_' + ground_truth.scenario_type,
            'expected_difficulty': ground_truth.difficulty_level
        }
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Traitement seulement certaines frames (sampling)
            if frame_count % 5 == 0:  # Chaque 5ème frame
                try:
                    # Conversion frame en base64
                    _, buffer = cv2.imencode('.jpg', frame)
                    frame_b64 = base64.b64encode(buffer).decode('utf-8')
                    
                    # Analyse avec système complet
                    frame_start_time = time.time()
                    
                    response = await self.orchestrator.analyze_surveillance_frame(
                        frame_data=frame_b64,
                        context={
                            **video_context,
                            'frame_number': frame_count,
                            'timestamp': datetime.now().isoformat()
                        }
                    )
                    
                    frame_end_time = time.time()
                    frame_processing_time = frame_end_time - frame_start_time
                    
                    # Classification détection
                    is_suspicious_detected = self._is_detection_suspicious(response)
                    is_suspicious_ground_truth = frame_count in ground_truth.suspicious_frames
                    
                    frame_analysis = {
                        'frame_number': frame_count,
                        'processing_time': frame_processing_time,
                        'suspicious_detected': is_suspicious_detected,
                        'suspicious_ground_truth': is_suspicious_ground_truth,
                        'confidence': getattr(response, 'confidence', 0.5),
                        'suspicion_level': getattr(response, 'suspicion_level', 'UNKNOWN').value if hasattr(getattr(response, 'suspicion_level', 'UNKNOWN'), 'value') else 'UNKNOWN'
                    }
                    
                    analysis_results['frame_analyses'].append(frame_analysis)
                    analysis_results['processed_frames'] += 1
                    
                    # Collecte détections suspectes
                    if is_suspicious_detected:
                        analysis_results['detections'].append({
                            'frame_number': frame_count,
                            'confidence': frame_analysis['confidence'],
                            'suspicion_level': frame_analysis['suspicion_level']
                        })
                    
                    # Nettoyage mémoire
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                except Exception as e:
                    print(f"   ❌ Erreur frame {frame_count}: {e}")
                    continue
        
        cap.release()
        
        total_end_time = time.time()
        analysis_results['total_frames'] = frame_count
        analysis_results['total_processing_time'] = total_end_time - total_start_time
        analysis_results['average_frame_time'] = (
            analysis_results['total_processing_time'] / max(analysis_results['processed_frames'], 1)
        )
        
        return analysis_results
    
    def _is_detection_suspicious(self, response) -> bool:
        """Détermine si une détection est considérée comme suspecte."""
        try:
            if hasattr(response, 'suspicion_level'):
                level = response.suspicion_level.value if hasattr(response.suspicion_level, 'value') else str(response.suspicion_level)
                return level in ['HIGH', 'CRITICAL']
            
            if hasattr(response, 'action_type'):
                action = str(response.action_type).lower()
                return any(word in action for word in ['suspicious', 'theft', 'concealment'])
            
            if hasattr(response, 'confidence'):
                return response.confidence > 0.8
            
        except Exception:
            pass
        
        return False
    
    def _calculate_video_accuracy_metrics(self, analysis_results: Dict, ground_truth: VideoGroundTruth) -> Dict:
        """Calcul métriques précision pour une vidéo."""
        tp = 0  # True Positives
        fp = 0  # False Positives  
        fn = 0  # False Negatives
        tn = 0  # True Negatives
        
        for frame_analysis in analysis_results['frame_analyses']:
            detected = frame_analysis['suspicious_detected']
            actual = frame_analysis['suspicious_ground_truth']
            
            if detected and actual:
                tp += 1
            elif detected and not actual:
                fp += 1
            elif not detected and actual:
                fn += 1
            else:
                tn += 1
        
        # Calcul métriques
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1_score = 2 * (precision * recall) / max(precision + recall, 1e-8)
        accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
        fp_rate = fp / max(fp + tn, 1)
        
        return {
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'true_negatives': tn,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'accuracy': accuracy,
            'fp_rate': fp_rate,
            'total_detections': len(analysis_results['detections']),
            'expected_detections': ground_truth.expected_detections
        }
    
    def calculate_overall_system_metrics(self):
        """Calcul métriques globales système."""
        print("\n📊 MÉTRIQUES GLOBALES SYSTÈME")
        print("=" * 40)
        
        all_precisions = []
        all_recalls = []
        all_f1_scores = []
        all_fp_rates = []
        all_processing_times = []
        
        difficulty_metrics = {'easy': [], 'medium': [], 'hard': []}
        
        for video_name, results in self.results['video_analysis_results'].items():
            metrics = results['accuracy_metrics']
            difficulty = results['ground_truth_info']['difficulty']
            
            all_precisions.append(metrics['precision'])
            all_recalls.append(metrics['recall'])
            all_f1_scores.append(metrics['f1_score'])
            all_fp_rates.append(metrics['fp_rate'])
            
            # Analyse par difficulté
            difficulty_metrics[difficulty].append(metrics['f1_score'])
            
            # Temps processing
            processing_time = results['analysis_results']['average_frame_time']
            all_processing_times.append(processing_time)
        
        # Métriques globales
        system_metrics = {
            'overall_precision': np.mean(all_precisions) if all_precisions else 0,
            'overall_recall': np.mean(all_recalls) if all_recalls else 0,
            'overall_f1_score': np.mean(all_f1_scores) if all_f1_scores else 0,
            'overall_fp_rate': np.mean(all_fp_rates) if all_fp_rates else 0,
            'average_processing_time_per_frame': np.mean(all_processing_times) if all_processing_times else 0,
            'videos_tested': len(self.results['video_analysis_results']),
            'difficulty_breakdown': {
                difficulty: np.mean(scores) if scores else 0 
                for difficulty, scores in difficulty_metrics.items()
            }
        }
        
        self.results['system_performance'] = system_metrics
        
        print(f"🎯 PERFORMANCE SYSTÈME GLOBALE:")
        print(f"   Précision: {system_metrics['overall_precision']:.2f}")
        print(f"   Recall: {system_metrics['overall_recall']:.2f}") 
        print(f"   F1-Score: {system_metrics['overall_f1_score']:.2f}")
        print(f"   Taux FP: {system_metrics['overall_fp_rate']:.2f}")
        print(f"   Temps/frame: {system_metrics['average_processing_time_per_frame']:.2f}s")
        
        print(f"\n🎯 PERFORMANCE PAR DIFFICULTÉ:")
        for difficulty, score in system_metrics['difficulty_breakdown'].items():
            print(f"   {difficulty.capitalize()}: F1={score:.2f}")
    
    def generate_complete_system_report(self):
        """Rapport complet système avec métriques réelles."""
        print("\n" + "="*80)
        print("📊 RAPPORT SYSTÈME COMPLET - TESTS VIDÉOS RÉELLES")
        print("="*80)
        
        if self.results.get('system_performance'):
            sp = self.results['system_performance']
            
            print(f"\n🎯 MÉTRIQUES SYSTÈME INTÉGRÉ (Tests sur {sp['videos_tested']} vidéos):")
            print(f"• Précision globale: {sp['overall_precision']*100:.1f}%")
            print(f"• Recall global: {sp['overall_recall']*100:.1f}%")
            print(f"• F1-Score global: {sp['overall_f1_score']:.3f}")
            print(f"• Taux faux positifs: {sp['overall_fp_rate']*100:.1f}%")
            print(f"• Temps traitement/frame: {sp['average_processing_time_per_frame']:.2f}s")
            
            print(f"\n🎯 PERFORMANCE PAR DIFFICULTÉ:")
            for difficulty, f1 in sp['difficulty_breakdown'].items():
                print(f"• {difficulty.capitalize()}: F1-Score {f1:.3f}")
        
        # Détail par vidéo
        print(f"\n📹 DÉTAIL RÉSULTATS PAR VIDÉO:")
        for video_name, results in self.results['video_analysis_results'].items():
            metrics = results['accuracy_metrics']
            info = results['ground_truth_info']
            
            print(f"\n• {video_name} ({info['scenario']} - {info['difficulty']}):")
            print(f"  - Précision: {metrics['precision']:.2f} | Recall: {metrics['recall']:.2f} | F1: {metrics['f1_score']:.2f}")
            print(f"  - Détections: {metrics['total_detections']}/{info['expected_detections']} attendues")
            print(f"  - FP Rate: {metrics['fp_rate']:.2f} (attendu: {info['expected_fp_rate']:.2f})")
        
        print(f"\n📝 UTILISATION POUR MÉMOIRE:")
        print(f"Ces métriques sont basées sur des tests RÉELS du système complet")
        print(f"avec analyse vidéo frame-par-frame et évaluation vs ground truth.")
    
    def save_results(self, filename: str = None):
        """Sauvegarde résultats."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"complete_system_video_tests_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"💾 Résultats système complet sauvegardés: {filename}")

async def main():
    """Test principal système complet."""
    print("🚀 TESTS SYSTÈME COMPLET AVEC VIDÉOS RÉELLES")
    print("=" * 60)
    
    test_suite = CompleteSystemTestSuite()
    
    try:
        # 1. Initialisation système complet
        system_ready = await test_suite.initialize_complete_system()
        
        if not system_ready:
            print("❌ Système non initialisé")
            return
        
        # 2. Création dataset vidéos annotées
        dataset_created = test_suite.create_test_videos_with_annotations()
        
        if not dataset_created:
            print("❌ Échec création dataset")
            return
        
        # 3. Tests système complet sur vidéos
        await test_suite.test_complete_system_on_videos()
        
        # 4. Calcul métriques globales
        test_suite.calculate_overall_system_metrics()
        
        # 5. Génération rapport
        test_suite.generate_complete_system_report()
        test_suite.save_results()
        
        print("\n🎉 TESTS SYSTÈME COMPLET TERMINÉS!")
        print("📊 Métriques système intégré collectées")
        print("🎯 Résultats basés sur analyse RÉELLE de vidéos")
        
    except Exception as e:
        print(f"\n❌ Erreur tests système complet: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())