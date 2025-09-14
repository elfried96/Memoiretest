#!/usr/bin/env python3
"""
📊 Collection Métriques Réelles - Intégration VLM Pipeline
=========================================================

Script pour collecter les vraies métriques de performance 
de votre système VLM en conditions réelles GPU.
"""

import asyncio
import time
import json
import statistics
import sys
import os
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import base64
import cv2
import torch
from PIL import Image
import io

# Ajout des chemins
sys.path.append(str(Path(__file__).parent / "src"))
sys.path.append(str(Path(__file__).parent / "dashboard"))

class RealMetricsCollector:
    """Collecteur de métriques réelles avec le pipeline VLM."""
    
    def __init__(self):
        self.results = {
            'real_vlm_metrics': {},
            'pipeline_performance': {},
            'orchestrator_metrics': {},
            'component_analysis': {},
            'gpu_utilization': {},
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'gpu_info': self._get_detailed_gpu_info(),
                'system_info': self._get_system_info()
            }
        }
        self.vlm_model = None
        self.orchestrator = None
        
    def _get_detailed_gpu_info(self) -> Dict[str, Any]:
        """Informations détaillées GPU."""
        if not torch.cuda.is_available():
            return {'available': False}
        
        gpu_info = {
            'available': True,
            'device_count': torch.cuda.device_count(),
            'devices': []
        }
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpu_info['devices'].append({
                'id': i,
                'name': torch.cuda.get_device_name(i),
                'total_memory_gb': props.total_memory / 1e9,
                'multi_processor_count': props.multi_processor_count,
                'compute_capability': f"{props.major}.{props.minor}"
            })
        
        return gpu_info
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Informations système."""
        import psutil
        
        return {
            'cpu_count': psutil.cpu_count(),
            'ram_total_gb': psutil.virtual_memory().total / 1e9,
            'python_version': sys.version,
            'torch_version': torch.__version__
        }
    
    async def initialize_real_pipeline(self):
        """Initialisation du pipeline VLM réel."""
        print("🔧 INITIALISATION PIPELINE VLM RÉEL")
        print("=" * 45)
        
        try:
            # Import et initialisation VLM
            print("1️⃣ Chargement Dynamic VLM Model...")
            from src.core.vlm.dynamic_model import DynamicVisionLanguageModel
            
            self.vlm_model = DynamicVisionLanguageModel(
                default_model="kimi-vl-a3b-thinking",
                enable_fallback=True
            )
            print("✅ VLM Model initialisé")
            
            # Import et initialisation Orchestrateur
            print("2️⃣ Chargement VLM Orchestrator...")
            from src.core.orchestrator.vlm_orchestrator import (
                ModernVLMOrchestrator, 
                OrchestrationConfig,
                OrchestrationMode
            )
            
            config = OrchestrationConfig(
                mode=OrchestrationMode.BALANCED,
                enable_advanced_tools=True,
                max_concurrent_tools=4,
                timeout_seconds=600
            )
            
            self.orchestrator = ModernVLMOrchestrator(
                vlm_model_name="kimi-vl-a3b-thinking",
                config=config
            )
            print("✅ Orchestrateur initialisé")
            
            return True
            
        except Exception as e:
            print(f"❌ Erreur initialisation pipeline: {e}")
            traceback.print_exc()
            return False
    
    def create_test_frame(self, scenario: str = "default") -> str:
        """Création frame de test base64."""
        print(f"🎬 Génération frame test: {scenario}")
        
        # Création image test simple
        if scenario == "person_shopping":
            # Image avec personne + objets
            img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            # Simulation rectangle personne
            cv2.rectangle(img, (200, 150), (400, 450), (0, 255, 0), 3)
            cv2.putText(img, "PERSON", (220, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif scenario == "suspicious_activity":
            # Image avec activité suspecte simulée
            img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            cv2.rectangle(img, (150, 100), (350, 400), (0, 0, 255), 3)
            cv2.putText(img, "SUSPICIOUS", (160, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            # Image test standard
            img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Conversion base64
        _, buffer = cv2.imencode('.jpg', img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return img_base64
    
    async def test_real_vlm_performance(self):
        """Test performance VLM réel avec métriques précises."""
        print("\n🧪 TEST PERFORMANCE VLM RÉEL")
        print("=" * 40)
        
        if not self.vlm_model:
            print("❌ VLM non initialisé")
            return
        
        scenarios = [
            ("normal_shopping", "Scénario shopping normal"),
            ("suspicious_activity", "Activité suspecte"),
            ("person_shopping", "Personne avec objets")
        ]
        
        vlm_metrics = {
            'latencies': [],
            'memory_usage': [],
            'gpu_utilization': [],
            'responses_quality': [],
            'scenarios_tested': []
        }
        
        for scenario, description in scenarios:
            print(f"🎭 Test: {description}")
            
            try:
                # Création frame test
                frame_data = self.create_test_frame(scenario)
                
                # Mesure mémoire GPU avant
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gpu_memory_before = torch.cuda.memory_allocated()
                
                # Mesure performance
                start_time = time.time()
                
                # Test VLM réel
                from src.core.types import AnalysisRequest
                
                request = AnalysisRequest(
                    frame_data=frame_data,
                    context={
                        'location': 'Test Store',
                        'timestamp': datetime.now().isoformat(),
                        'scenario': scenario
                    },
                    tools_available=['sam2_segmentator', 'dino_features', 'pose_estimator']
                )
                
                response = await self.vlm_model.analyze_with_tools(
                    request, 
                    use_advanced_tools=True
                )
                
                end_time = time.time()
                latency = (end_time - start_time) * 1000  # en ms
                
                # Mesure mémoire GPU après
                if torch.cuda.is_available():
                    gpu_memory_after = torch.cuda.memory_allocated()
                    memory_usage = (gpu_memory_after - gpu_memory_before) / 1e6  # MB
                else:
                    memory_usage = 0
                
                # Évaluation qualité réponse
                quality_score = self._evaluate_response_quality(response, scenario)
                
                vlm_metrics['latencies'].append(latency)
                vlm_metrics['memory_usage'].append(memory_usage)
                vlm_metrics['responses_quality'].append(quality_score)
                vlm_metrics['scenarios_tested'].append(scenario)
                
                print(f"   ✅ Latence: {latency:.1f}ms")
                print(f"   ✅ Mémoire: {memory_usage:.1f}MB")
                print(f"   ✅ Qualité: {quality_score:.2f}")
                
                # Nettoyage GPU
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Pause entre tests
                await asyncio.sleep(1)
                
            except Exception as e:
                print(f"   ❌ Erreur test {scenario}: {e}")
                continue
        
        # Calcul métriques finales
        if vlm_metrics['latencies']:
            final_metrics = {
                'average_latency_ms': statistics.mean(vlm_metrics['latencies']),
                'min_latency_ms': min(vlm_metrics['latencies']),
                'max_latency_ms': max(vlm_metrics['latencies']),
                'latency_std': statistics.stdev(vlm_metrics['latencies']) if len(vlm_metrics['latencies']) > 1 else 0,
                'average_memory_mb': statistics.mean(vlm_metrics['memory_usage']),
                'average_quality_score': statistics.mean(vlm_metrics['responses_quality']),
                'tests_completed': len(vlm_metrics['latencies']),
                'success_rate': len(vlm_metrics['latencies']) / len(scenarios) * 100
            }
            
            self.results['real_vlm_metrics'] = final_metrics
            
            print(f"\n📊 MÉTRIQUES VLM RÉEL:")
            print(f"   🚀 Latence moyenne: {final_metrics['average_latency_ms']:.1f}ms")
            print(f"   🚀 Latence min/max: {final_metrics['min_latency_ms']:.1f}/{final_metrics['max_latency_ms']:.1f}ms")
            print(f"   💾 Mémoire moyenne: {final_metrics['average_memory_mb']:.1f}MB")
            print(f"   🎯 Qualité moyenne: {final_metrics['average_quality_score']:.2f}")
            print(f"   ✅ Taux succès: {final_metrics['success_rate']:.1f}%")
    
    def _evaluate_response_quality(self, response, scenario: str) -> float:
        """Évaluation qualité réponse VLM."""
        quality_score = 0.5  # Base
        
        try:
            # Vérification présence champs obligatoires
            if hasattr(response, 'suspicion_level'):
                quality_score += 0.2
            if hasattr(response, 'description') and response.description:
                quality_score += 0.2
            if hasattr(response, 'confidence') and response.confidence > 0:
                quality_score += 0.1
            
            # Cohérence avec scénario
            if scenario == "suspicious_activity" and hasattr(response, 'suspicion_level'):
                if response.suspicion_level.value in ['HIGH', 'CRITICAL']:
                    quality_score += 0.2
            elif scenario == "normal_shopping":
                if hasattr(response, 'suspicion_level') and response.suspicion_level.value in ['LOW', 'MEDIUM']:
                    quality_score += 0.2
            
        except Exception:
            pass
        
        return min(quality_score, 1.0)
    
    async def test_orchestrator_performance(self):
        """Test performance orchestrateur réel."""
        print("\n🎯 TEST ORCHESTRATEUR RÉEL")
        print("=" * 35)
        
        if not self.orchestrator:
            print("❌ Orchestrateur non initialisé")
            return
        
        modes_configs = [
            ('FAST', 3),
            ('BALANCED', 5), 
            ('THOROUGH', 8)
        ]
        
        orchestrator_metrics = {}
        
        for mode_name, expected_tools in modes_configs:
            print(f"🔧 Test mode {mode_name}...")
            
            try:
                # Configuration mode
                from src.core.orchestrator.vlm_orchestrator import OrchestrationMode
                
                if mode_name == 'FAST':
                    self.orchestrator.config.mode = OrchestrationMode.FAST
                elif mode_name == 'BALANCED':
                    self.orchestrator.config.mode = OrchestrationMode.BALANCED
                else:
                    self.orchestrator.config.mode = OrchestrationMode.THOROUGH
                
                # Test avec frame réel
                frame_data = self.create_test_frame("person_shopping")
                
                start_time = time.time()
                
                # Analyse complète orchestrateur
                response = await self.orchestrator.analyze_surveillance_frame(
                    frame_data=frame_data,
                    context={
                        'location': 'Test Store',
                        'timestamp': datetime.now().isoformat()
                    }
                )
                
                end_time = time.time()
                latency = (end_time - start_time) * 1000
                
                # Métriques mode
                mode_metrics = {
                    'latency_ms': latency,
                    'response_valid': response is not None,
                    'tools_estimated': expected_tools,
                    'mode': mode_name
                }
                
                orchestrator_metrics[mode_name.lower()] = mode_metrics
                
                print(f"   ✅ Latence: {latency:.1f}ms")
                print(f"   ✅ Réponse valide: {response is not None}")
                
                # Pause entre modes
                await asyncio.sleep(2)
                
            except Exception as e:
                print(f"   ❌ Erreur mode {mode_name}: {e}")
                continue
        
        self.results['orchestrator_metrics'] = orchestrator_metrics
        
        print(f"\n📊 MÉTRIQUES ORCHESTRATEUR:")
        for mode, metrics in orchestrator_metrics.items():
            print(f"   {mode.upper()}: {metrics['latency_ms']:.1f}ms")
    
    async def measure_gpu_utilization(self):
        """Mesure utilisation GPU pendant les tests."""
        print("\n🔥 MESURE UTILISATION GPU")
        print("=" * 35)
        
        if not torch.cuda.is_available():
            print("⚠️  Pas de GPU - Skip mesure")
            return
        
        try:
            gpu_metrics = {
                'memory_allocated_mb': torch.cuda.memory_allocated() / 1e6,
                'memory_cached_mb': torch.cuda.memory_reserved() / 1e6,
                'memory_total_mb': torch.cuda.get_device_properties(0).total_memory / 1e6,
                'device_name': torch.cuda.get_device_name()
            }
            
            self.results['gpu_utilization'] = gpu_metrics
            
            print(f"🔥 Device: {gpu_metrics['device_name']}")
            print(f"🔥 Mémoire allouée: {gpu_metrics['memory_allocated_mb']:.1f}MB")
            print(f"🔥 Mémoire en cache: {gpu_metrics['memory_cached_mb']:.1f}MB")
            print(f"🔥 Mémoire totale: {gpu_metrics['memory_total_mb']:.1f}MB")
            
        except Exception as e:
            print(f"❌ Erreur mesure GPU: {e}")
    
    def generate_thesis_metrics(self):
        """Génère métriques formatées pour le mémoire."""
        print("\n" + "="*70)
        print("📊 MÉTRIQUES RÉELLES POUR MÉMOIRE ACADÉMIQUE")
        print("="*70)
        
        # Métriques VLM réelles
        if 'real_vlm_metrics' in self.results and self.results['real_vlm_metrics']:
            vlm = self.results['real_vlm_metrics']
            print(f"\n🧠 MODULE VLM RÉEL (Kimi-VL-A3B-Thinking):")
            print(f"• Latence moyenne: {vlm['average_latency_ms']:.1f}ms")
            print(f"• Précision estimée: {vlm['average_quality_score']*100:.1f}%")
            print(f"• Taux succès: {vlm['success_rate']:.1f}%")
            print(f"• Utilisation mémoire: {vlm['average_memory_mb']:.1f}MB")
        
        # Métriques orchestrateur réelles
        if 'orchestrator_metrics' in self.results and self.results['orchestrator_metrics']:
            print(f"\n🎯 ORCHESTRATEUR RÉEL:")
            for mode, metrics in self.results['orchestrator_metrics'].items():
                print(f"• Mode {mode.upper()}: {metrics['latency_ms']:.1f}ms latence")
        
        # Métriques GPU réelles
        if 'gpu_utilization' in self.results and self.results['gpu_utilization']:
            gpu = self.results['gpu_utilization']
            print(f"\n🔥 UTILISATION GPU RÉELLE:")
            print(f"• Device: {gpu['device_name']}")
            print(f"• Mémoire utilisée: {gpu['memory_allocated_mb']:.1f}MB")
        
        # Recommandations mise à jour mémoire
        print(f"\n📝 RECOMMANDATIONS MISE À JOUR MÉMOIRE:")
        if self.results.get('real_vlm_metrics'):
            real_latency = self.results['real_vlm_metrics']['average_latency_ms']
            print(f"• Remplacer '180ms latence' par '{real_latency:.0f}ms latence' pour VLM")
            
        if self.results.get('orchestrator_metrics', {}).get('balanced'):
            real_balanced = self.results['orchestrator_metrics']['balanced']['latency_ms']
            print(f"• Remplacer '285ms' par '{real_balanced:.0f}ms' pour mode BALANCED")
        
        print(f"\n🎯 Métriques collectées à {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    def save_results(self, filename: str = None):
        """Sauvegarde résultats."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"real_metrics_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"💾 Métriques sauvegardées: {filename}")

async def main():
    """Collecte métriques réelles."""
    print("🚀 COLLECTION MÉTRIQUES RÉELLES")
    print("=" * 50)
    
    collector = RealMetricsCollector()
    
    # Initialisation pipeline
    pipeline_ready = await collector.initialize_real_pipeline()
    
    if not pipeline_ready:
        print("❌ Pipeline non disponible - Utiliser run_performance_tests.py pour simulation")
        return
    
    try:
        # Séquence tests réels
        await collector.test_real_vlm_performance()
        await collector.test_orchestrator_performance()
        await collector.measure_gpu_utilization()
        
        # Génération rapport
        collector.generate_thesis_metrics()
        collector.save_results()
        
        print("\n🎉 COLLECTION MÉTRIQUES TERMINÉE!")
        print("📊 Utilisez ces valeurs pour mise à jour mémoire")
        
    except Exception as e:
        print(f"\n❌ Erreur durant collection: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())