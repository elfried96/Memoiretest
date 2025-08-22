"""
⚡ Tests de Performance et Benchmarks - Métriques pour Mémoire
=============================================================

Benchmarks spécialisés pour quantifier :
- Performance comparative des modèles VLM
- Efficacité des outils selon contexte
- Scalabilité du système sous charge
- Optimisations GPU vs CPU
- Métriques temps réel vs batch

Objectif : Données quantitatives précises pour validation mémoire
"""

import pytest
import asyncio
import time
import numpy as np
import torch
import cv2
import statistics
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict
import json
from pathlib import Path

from src.core.vlm.dynamic_model import DynamicVisionLanguageModel
from src.core.vlm.model_registry import VLMModelRegistry
from src.detection.yolo_detector import YOLODetector
from src.core.orchestrator.adaptive_orchestrator import AdaptiveVLMOrchestrator
from src.core.orchestrator.vlm_orchestrator import OrchestrationConfig, OrchestrationMode
from src.core.types import AnalysisRequest, AnalysisResponse
from src.utils.performance import get_current_performance


@dataclass
class BenchmarkResult:
    """Résultat de benchmark standardisé."""
    test_name: str
    execution_time: float
    throughput: float  # images/sec ou operations/sec
    memory_usage_mb: float
    gpu_utilization: float
    success_rate: float
    confidence_avg: float
    metadata: Dict[str, Any]


class TestPerformanceBenchmarks:
    """Benchmarks de performance pour validation mémoire."""
    
    @pytest.fixture
    def benchmark_images(self, sample_images):
        """Images standardisées pour benchmarks."""
        return {
            "low_res": cv2.resize(sample_images["normal"], (320, 240)),
            "standard": sample_images["normal"],  # 640x480
            "hd": cv2.resize(sample_images["normal"], (1280, 720)),
            "full_hd": cv2.resize(sample_images["normal"], (1920, 1080))
        }
    
    @pytest.fixture
    def benchmark_images_b64(self, benchmark_images, test_utils):
        """Images benchmark en base64."""
        return {
            name: test_utils.numpy_to_b64(img) 
            for name, img in benchmark_images.items()
        }
    
    # =================== BENCHMARKS VLM ===================
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_vlm_models_comparative_benchmark(self, benchmark_images_b64, metrics_collector):
        """Benchmark 5.1: Comparaison performance modèles VLM."""
        print("\\n🧠 Benchmark Comparatif Modèles VLM")
        
        models_to_test = ["kimi-vl-a3b-thinking", "qwen2-vl-7b-instruct"]
        resolutions = ["standard", "hd"]  # Tests sur 2 résolutions
        
        results = {}
        
        for model_name in models_to_test:
            print(f"   Testing {model_name}...")
            
            vlm = DynamicVisionLanguageModel(default_model=model_name, enable_fallback=False)
            
            try:
                # Chargement avec mesure de temps
                load_start = time.time()
                load_success = await vlm.load_model(model_name)
                load_time = time.time() - load_start
                
                if not load_success:
                    results[model_name] = {"error": "Échec chargement", "load_time": load_time}
                    continue
                
                model_results = {"load_time": load_time, "inferences": {}}
                
                for res_name in resolutions:
                    print(f"      Resolution {res_name}...")
                    
                    # Mesures multiples pour moyenne
                    inference_times = []
                    confidences = []
                    
                    for run in range(3):  # 3 runs pour moyenner
                        request = AnalysisRequest(
                            frame_data=benchmark_images_b64[res_name],
                            context={"benchmark": True, "model": model_name, "resolution": res_name},
                            tools_available=[]
                        )
                        
                        start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                        
                        start_time = time.time()
                        result = await vlm.analyze_with_tools(request, use_advanced_tools=False)
                        inference_time = time.time() - start_time
                        
                        end_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                        memory_used = (end_memory - start_memory) / (1024**2)  # MB
                        
                        inference_times.append(inference_time)
                        confidences.append(result.confidence)
                    
                    # Statistiques
                    model_results["inferences"][res_name] = {
                        "avg_time": statistics.mean(inference_times),
                        "std_time": statistics.stdev(inference_times) if len(inference_times) > 1 else 0,
                        "min_time": min(inference_times),
                        "max_time": max(inference_times),
                        "avg_confidence": statistics.mean(confidences),
                        "memory_per_inference_mb": memory_used,
                        "throughput_fps": 1.0 / statistics.mean(inference_times)
                    }
                
                results[model_name] = model_results
                
                # Nettoyage
                vlm._unload_current_model()
                
            except Exception as e:
                results[model_name] = {"error": str(e)}
                print(f"      Erreur {model_name}: {e}")
        
        # Analyse comparative
        comparison = self._analyze_vlm_comparison(results)
        
        # Sauvegarde métriques
        metrics_collector.add_performance_data("vlm_comparative", {
            "results": results,
            "comparison": comparison
        })
        
        print(f"✅ Benchmark VLM: {len(results)} modèles testés")
        self._print_vlm_comparison(comparison)
        
        return results
    
    def _analyze_vlm_comparison(self, results: Dict) -> Dict:
        """Analyse comparative des résultats VLM."""
        comparison = {
            "fastest_model": None,
            "most_accurate": None,
            "performance_ratios": {},
            "recommendations": []
        }
        
        valid_models = {k: v for k, v in results.items() if "error" not in v}
        
        if len(valid_models) >= 2:
            # Modèle le plus rapide (résolution standard)
            avg_times = {}
            avg_confidences = {}
            
            for model, data in valid_models.items():
                if "standard" in data.get("inferences", {}):
                    avg_times[model] = data["inferences"]["standard"]["avg_time"]
                    avg_confidences[model] = data["inferences"]["standard"]["avg_confidence"]
            
            if avg_times:
                comparison["fastest_model"] = min(avg_times, key=avg_times.get)
                comparison["most_accurate"] = max(avg_confidences, key=avg_confidences.get)
                
                # Ratios de performance
                fastest_time = min(avg_times.values())
                for model, time_val in avg_times.items():
                    comparison["performance_ratios"][model] = time_val / fastest_time
        
        return comparison
    
    def _print_vlm_comparison(self, comparison: Dict):
        """Affichage résultats comparaison VLM."""
        print("   📊 Analyse Comparative:")
        if comparison["fastest_model"]:
            print(f"      Plus rapide: {comparison['fastest_model']}")
        if comparison["most_accurate"]:
            print(f"      Plus précis: {comparison['most_accurate']}")
        
        for model, ratio in comparison.get("performance_ratios", {}).items():
            print(f"      {model}: {ratio:.2f}x temps de référence")
    
    # =================== BENCHMARKS YOLO ===================
    
    @pytest.mark.performance
    def test_yolo_resolution_scaling_benchmark(self, benchmark_images, metrics_collector):
        """Benchmark 5.2: Scalabilité YOLO selon résolution."""
        print("\\n🎯 Benchmark Scalabilité YOLO")
        
        detector = YOLODetector(model_name="yolov8n.pt", device="auto")
        
        results = {}
        
        for res_name, image in benchmark_images.items():
            print(f"   Testing résolution {res_name} ({image.shape[1]}x{image.shape[0]})...")
            
            # Benchmarks multiples
            times = []
            detection_counts = []
            
            for run in range(5):  # 5 runs pour stabilité
                start_time = time.time()
                detections = detector.detect(image)
                detection_time = time.time() - start_time
                
                times.append(detection_time)
                detection_counts.append(len(detections))
            
            # Calcul throughput
            avg_time = statistics.mean(times)
            pixels = image.shape[0] * image.shape[1]
            
            results[res_name] = {
                "resolution": f"{image.shape[1]}x{image.shape[0]}",
                "pixels": pixels,
                "avg_time": avg_time,
                "std_time": statistics.stdev(times),
                "min_time": min(times),
                "max_time": max(times),
                "avg_detections": statistics.mean(detection_counts),
                "fps": 1.0 / avg_time,
                "pixels_per_second": pixels / avg_time
            }
            
            print(f"      {avg_time:.3f}s, {1.0/avg_time:.1f} FPS, {statistics.mean(detection_counts):.1f} détections")
        
        # Analyse scalabilité
        scaling_analysis = self._analyze_yolo_scaling(results)
        
        metrics_collector.add_performance_data("yolo_scaling", {
            "results": results,
            "scaling_analysis": scaling_analysis
        })
        
        print(f"✅ Benchmark YOLO: {len(results)} résolutions testées")
        return results
    
    def _analyze_yolo_scaling(self, results: Dict) -> Dict:
        """Analyse de la scalabilité YOLO."""
        resolutions = list(results.keys())
        
        if len(resolutions) < 2:
            return {"insufficient_data": True}
        
        # Calcul du facteur de scaling
        base_res = resolutions[0]  # Plus petite résolution
        base_pixels = results[base_res]["pixels"]
        base_time = results[base_res]["avg_time"]
        
        scaling_factors = {}
        for res in resolutions[1:]:
            pixel_ratio = results[res]["pixels"] / base_pixels
            time_ratio = results[res]["avg_time"] / base_time
            scaling_factors[res] = {
                "pixel_ratio": pixel_ratio,
                "time_ratio": time_ratio,
                "efficiency": pixel_ratio / time_ratio  # >1 = super-linéaire, <1 = sous-linéaire
            }
        
        return {
            "base_resolution": base_res,
            "scaling_factors": scaling_factors,
            "linearity_assessment": "sous-linéaire" if all(sf["efficiency"] < 1 for sf in scaling_factors.values()) else "super-linéaire"
        }
    
    # =================== BENCHMARKS ORCHESTRATION ===================
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_orchestration_modes_benchmark(self, benchmark_images_b64, metrics_collector):
        """Benchmark 5.3: Comparaison modes orchestration."""
        print("\\n🎭 Benchmark Modes Orchestration")
        
        modes = [OrchestrationMode.FAST, OrchestrationMode.BALANCED, OrchestrationMode.THOROUGH]
        
        results = {}
        
        for mode in modes:
            print(f"   Testing mode {mode.value}...")
            
            config = OrchestrationConfig(
                mode=mode,
                max_concurrent_tools=8,  # Max pour THOROUGH
                enable_advanced_tools=True
            )
            
            orchestrator = AdaptiveVLMOrchestrator(
                vlm_model_name="kimi-vl-a3b-thinking",
                config=config,
                enable_adaptive_learning=False  # Désactivé pour benchmark pur
            )
            
            mode_times = []
            tools_used_counts = []
            confidences = []
            
            # Test sur plusieurs images
            test_images = ["standard", "hd"]
            
            for img_name in test_images:
                for run in range(2):  # 2 runs par image
                    start_time = time.time()
                    
                    result = await orchestrator.analyze_surveillance_frame(
                        frame_data=benchmark_images_b64[img_name],
                        context={"benchmark": True, "mode": mode.value, "image": img_name}
                    )
                    
                    orchestration_time = time.time() - start_time
                    
                    mode_times.append(orchestration_time)
                    tools_used_counts.append(len(result.tools_used))
                    confidences.append(result.confidence)
            
            results[mode.value] = {
                "avg_time": statistics.mean(mode_times),
                "std_time": statistics.stdev(mode_times) if len(mode_times) > 1 else 0,
                "min_time": min(mode_times),
                "max_time": max(mode_times),
                "avg_tools_used": statistics.mean(tools_used_counts),
                "avg_confidence": statistics.mean(confidences),
                "throughput_fps": 1.0 / statistics.mean(mode_times),
                "samples": len(mode_times)
            }
            
            print(f"      {statistics.mean(mode_times):.2f}s, {statistics.mean(tools_used_counts):.1f} outils, confiance={statistics.mean(confidences):.2f}")
        
        # Analyse des modes
        modes_analysis = self._analyze_orchestration_modes(results)
        
        metrics_collector.add_performance_data("orchestration_modes", {
            "results": results,
            "analysis": modes_analysis
        })
        
        print(f"✅ Benchmark Orchestration: {len(results)} modes testés")
        return results
    
    def _analyze_orchestration_modes(self, results: Dict) -> Dict:
        """Analyse des modes d'orchestration."""
        analysis = {
            "speed_ranking": [],
            "accuracy_ranking": [],
            "efficiency_ranking": [],
            "trade_offs": {}
        }
        
        # Classement par vitesse
        speed_sorted = sorted(results.items(), key=lambda x: x[1]["avg_time"])
        analysis["speed_ranking"] = [mode for mode, _ in speed_sorted]
        
        # Classement par précision
        accuracy_sorted = sorted(results.items(), key=lambda x: x[1]["avg_confidence"], reverse=True)
        analysis["accuracy_ranking"] = [mode for mode, _ in accuracy_sorted]
        
        # Analyse trade-offs vitesse/précision
        for mode, data in results.items():
            time_score = 1.0 / data["avg_time"]  # Plus rapide = meilleur score
            accuracy_score = data["avg_confidence"]
            efficiency = (time_score * accuracy_score) / data["avg_tools_used"]  # Score composite
            
            analysis["trade_offs"][mode] = {
                "time_score": time_score,
                "accuracy_score": accuracy_score,
                "efficiency_score": efficiency,
                "tools_overhead": data["avg_tools_used"]
            }
        
        # Classement efficacité
        efficiency_sorted = sorted(analysis["trade_offs"].items(), key=lambda x: x[1]["efficiency_score"], reverse=True)
        analysis["efficiency_ranking"] = [mode for mode, _ in efficiency_sorted]
        
        return analysis
    
    # =================== BENCHMARKS CHARGE ===================
    
    @pytest.mark.performance
    @pytest.mark.asyncio  
    async def test_load_stress_benchmark(self, benchmark_images_b64, metrics_collector):
        """Benchmark 5.4: Tests de charge et stress."""
        print("\\n⚡ Benchmark Charge et Stress")
        
        orchestrator = AdaptiveVLMOrchestrator(
            vlm_model_name="kimi-vl-a3b-thinking",
            config=OrchestrationConfig(mode=OrchestrationMode.FAST)  # Mode rapide pour charge
        )
        
        # Configuration test de charge
        load_configs = [
            {"concurrent": 2, "duration": 10},   # Charge légère
            {"concurrent": 5, "duration": 15},   # Charge moyenne
            {"concurrent": 8, "duration": 10}    # Charge élevée
        ]
        
        results = {}
        
        for config in load_configs:
            concurrent = config["concurrent"]
            duration = config["duration"]
            
            print(f"   Testing charge: {concurrent} requêtes concurrentes, {duration}s...")
            
            async def process_continuous():
                """Traitement continu pendant la durée."""
                processed = 0
                errors = 0
                times = []
                
                end_time = time.time() + duration
                
                while time.time() < end_time:
                    try:
                        # Sélection image aléatoire
                        img_key = np.random.choice(list(benchmark_images_b64.keys()))
                        
                        start = time.time()
                        result = await orchestrator.analyze_surveillance_frame(
                            frame_data=benchmark_images_b64[img_key],
                            context={"load_test": True, "concurrent": concurrent}
                        )
                        process_time = time.time() - start
                        
                        times.append(process_time)
                        processed += 1
                        
                    except Exception as e:
                        errors += 1
                        print(f"      Erreur: {e}")
                
                return {"processed": processed, "errors": errors, "times": times}
            
            # Lancement concurrent
            start_load = time.time()
            
            tasks = [process_continuous() for _ in range(concurrent)]
            task_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            total_load_time = time.time() - start_load
            
            # Agrégation résultats
            total_processed = sum(r["processed"] for r in task_results if isinstance(r, dict))
            total_errors = sum(r["errors"] for r in task_results if isinstance(r, dict))
            all_times = []
            for r in task_results:
                if isinstance(r, dict):
                    all_times.extend(r["times"])
            
            throughput = total_processed / total_load_time if total_load_time > 0 else 0
            error_rate = total_errors / (total_processed + total_errors) if (total_processed + total_errors) > 0 else 0
            
            results[f"concurrent_{concurrent}"] = {
                "concurrent_streams": concurrent,
                "duration_sec": duration,
                "total_processed": total_processed,
                "total_errors": total_errors,
                "throughput_rps": throughput,
                "error_rate": error_rate,
                "avg_response_time": statistics.mean(all_times) if all_times else 0,
                "max_response_time": max(all_times) if all_times else 0,
                "p95_response_time": np.percentile(all_times, 95) if all_times else 0
            }
            
            print(f"      {total_processed} traitées, {throughput:.1f} req/s, {error_rate:.1%} erreurs")
        
        # Analyse scalabilité
        scalability_analysis = self._analyze_load_scalability(results)
        
        metrics_collector.add_performance_data("load_stress", {
            "results": results,
            "scalability": scalability_analysis
        })
        
        print(f"✅ Benchmark Charge: {len(results)} configurations testées")
        return results
    
    def _analyze_load_scalability(self, results: Dict) -> Dict:
        """Analyse de la scalabilité sous charge."""
        analysis = {
            "throughput_progression": {},
            "error_rate_progression": {},
            "response_time_degradation": {},
            "scalability_assessment": ""
        }
        
        sorted_configs = sorted(results.items(), key=lambda x: x[1]["concurrent_streams"])
        
        for config_name, data in sorted_configs:
            concurrent = data["concurrent_streams"]
            analysis["throughput_progression"][concurrent] = data["throughput_rps"]
            analysis["error_rate_progression"][concurrent] = data["error_rate"]
            analysis["response_time_degradation"][concurrent] = data["avg_response_time"]
        
        # Évaluation globale
        throughputs = list(analysis["throughput_progression"].values())
        error_rates = list(analysis["error_rate_progression"].values())
        
        if len(throughputs) >= 2:
            if throughputs[-1] > throughputs[0] * 0.8:  # Maintient >80% du throughput
                if max(error_rates) < 0.1:  # <10% erreurs
                    analysis["scalability_assessment"] = "Excellente"
                else:
                    analysis["scalability_assessment"] = "Bonne"
            else:
                analysis["scalability_assessment"] = "Limitée"
        
        return analysis


@pytest.mark.gpu
class TestGPUPerformanceBenchmarks:
    """Benchmarks spécifiques GPU."""
    
    @pytest.mark.performance
    def test_gpu_memory_efficiency_benchmark(self, benchmark_images, metrics_collector):
        """Benchmark 5.5: Efficacité mémoire GPU."""
        if not torch.cuda.is_available():
            pytest.skip("GPU non disponible")
        
        print("\\n💾 Benchmark Efficacité Mémoire GPU")
        
        detector = YOLODetector(device="cuda")
        
        memory_results = {}
        
        for res_name, image in benchmark_images.items():
            print(f"   Testing mémoire {res_name}...")
            
            # Nettoyage initial
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()
            
            # Mesures avec batch sizes différents
            batch_sizes = [1, 2, 4, 8]
            
            for batch_size in batch_sizes:
                try:
                    # Création batch
                    batch_images = [image] * batch_size
                    
                    # Mesure mémoire
                    pre_memory = torch.cuda.memory_allocated()
                    
                    # Détection
                    start_time = time.time()
                    results = detector.detect_batch(batch_images)
                    detection_time = time.time() - start_time
                    
                    post_memory = torch.cuda.memory_allocated()
                    memory_used = (post_memory - pre_memory) / (1024**2)  # MB
                    
                    memory_per_image = memory_used / batch_size
                    throughput = batch_size / detection_time
                    
                    key = f"{res_name}_batch_{batch_size}"
                    memory_results[key] = {
                        "resolution": res_name,
                        "batch_size": batch_size,
                        "total_memory_mb": memory_used,
                        "memory_per_image_mb": memory_per_image,
                        "detection_time": detection_time,
                        "throughput_ips": throughput,
                        "memory_efficiency": throughput / memory_used if memory_used > 0 else 0
                    }
                    
                    print(f"      Batch {batch_size}: {memory_per_image:.1f}MB/img, {throughput:.1f} img/s")
                    
                except Exception as e:
                    print(f"      Batch {batch_size} failed: {e}")
            
            # Nettoyage
            torch.cuda.empty_cache()
        
        metrics_collector.add_performance_data("gpu_memory_efficiency", memory_results)
        
        print(f"✅ Benchmark Mémoire GPU: {len(memory_results)} configurations testées")
        return memory_results


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "performance"])