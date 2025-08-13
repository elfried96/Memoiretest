"""A/B Testing framework for pipeline comparisons."""

import time
import threading
import queue
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Callable, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

logger = logging.getLogger(__name__)

class PipelineVariant(Enum):
    """Different pipeline configurations to test."""
    BASELINE = "yolo_only"
    VLM_FIRST = "vlm_first" 
    PARALLEL = "parallel_fusion"
    VALIDATION_EARLY = "validation_early"
    ADVANCED_TOOLS = "advanced_tools"
    FULL_PIPELINE = "full_pipeline"

@dataclass
class TestVariant:
    """Configuration for a single test variant."""
    name: str
    pipeline_type: PipelineVariant
    components: List[str]
    parameters: Dict[str, Any]
    enabled_tools: List[str]
    description: str

@dataclass
class TestResult:
    """Result from running a single test."""
    variant_name: str
    execution_time: float
    memory_usage: float
    cpu_usage: float
    gpu_usage: float
    accuracy_metrics: Dict[str, float]
    error_count: int
    throughput: float
    latency_p95: float
    success_rate: float

class ABTestFramework:
    """Framework for conducting A/B tests between different pipeline configurations."""
    
    def __init__(self, output_dir: str = "test_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.test_variants = {}
        self.results = []
        self.current_test_id = None
        
    def register_variant(self, variant: TestVariant):
        """Register a test variant."""
        self.test_variants[variant.name] = variant
        logger.info(f"Registered test variant: {variant.name}")
    
    def setup_default_variants(self):
        """Setup default test variants for comparison."""
        
        # Baseline: YOLO only
        self.register_variant(TestVariant(
            name="baseline_yolo",
            pipeline_type=PipelineVariant.BASELINE,
            components=["yolo_detector", "basic_tracker"],
            parameters={
                "yolo_confidence": 0.5,
                "nms_threshold": 0.45,
                "tracker_type": "simple"
            },
            enabled_tools=[],
            description="Basic YOLO detection with simple tracking"
        ))
        
        # VLM First approach
        self.register_variant(TestVariant(
            name="vlm_first",
            pipeline_type=PipelineVariant.VLM_FIRST,
            components=["vlm_analyzer", "yolo_detector", "cross_validator"],
            parameters={
                "vlm_temperature": 0.7,
                "vlm_max_tokens": 256,
                "yolo_confidence": 0.3
            },
            enabled_tools=["object_detector", "behavior_classifier"],
            description="VLM analysis first, then YOLO refinement"
        ))
        
        # Parallel processing
        self.register_variant(TestVariant(
            name="parallel_fusion",
            pipeline_type=PipelineVariant.PARALLEL,
            components=["yolo_detector", "vlm_analyzer", "multimodal_fusion", "cross_validator"],
            parameters={
                "fusion_weight_yolo": 0.6,
                "fusion_weight_vlm": 0.4,
                "parallel_processing": True
            },
            enabled_tools=["object_detector", "motion_analyzer", "behavior_classifier"],
            description="Parallel YOLO and VLM with fusion"
        ))
        
        # Advanced tools enabled
        self.register_variant(TestVariant(
            name="advanced_tools",
            pipeline_type=PipelineVariant.ADVANCED_TOOLS,
            components=["yolo_detector", "sam2_segmentation", "dino_features", "pose_estimation", "vlm_analyzer"],
            parameters={
                "sam2_enabled": True,
                "dino_enabled": True,
                "pose_enabled": True,
                "feature_fusion": True
            },
            enabled_tools=["object_detector", "segmentation_tool", "feature_extractor", "pose_analyzer", "behavior_classifier"],
            description="Full advanced tools pipeline"
        ))
        
        # Full production pipeline
        self.register_variant(TestVariant(
            name="full_pipeline",
            pipeline_type=PipelineVariant.FULL_PIPELINE,
            components=[
                "yolo_detector", "advanced_tracker", "sam2_segmentation", 
                "dino_features", "pose_estimation", "vlm_analyzer", 
                "cross_validator", "temporal_transformer"
            ],
            parameters={
                "all_optimizations": True,
                "ensemble_validation": True,
                "temporal_analysis": True
            },
            enabled_tools=[
                "object_detector", "segmentation_tool", "feature_extractor",
                "pose_analyzer", "motion_analyzer", "behavior_classifier",
                "crowd_counter", "trajectory_analyzer"
            ],
            description="Complete production-ready pipeline"
        ))
    
    def run_ab_test(self, test_dataset: List[str], test_duration: int = 300,
                   concurrent_variants: int = 2, metrics_to_collect: List[str] = None) -> Dict[str, Any]:
        """Run A/B test comparing different pipeline variants."""
        
        if metrics_to_collect is None:
            metrics_to_collect = [
                "execution_time", "memory_usage", "accuracy", 
                "false_positive_rate", "throughput", "latency"
            ]
        
        self.current_test_id = f"ab_test_{int(time.time())}"
        test_results = {}
        
        logger.info(f"Starting A/B test {self.current_test_id} with {len(self.test_variants)} variants")
        
        # Run tests for each variant
        with ThreadPoolExecutor(max_workers=concurrent_variants) as executor:
            futures = {}
            
            for variant_name, variant in self.test_variants.items():
                future = executor.submit(
                    self._run_single_variant_test, 
                    variant, test_dataset, test_duration, metrics_to_collect
                )
                futures[variant_name] = future
            
            # Collect results
            for variant_name, future in futures.items():
                try:
                    result = future.result(timeout=test_duration + 60)
                    test_results[variant_name] = result
                    self.results.append(result)
                    logger.info(f"Completed test for variant: {variant_name}")
                except Exception as e:
                    logger.error(f"Test failed for variant {variant_name}: {e}")
                    test_results[variant_name] = None
        
        # Analyze and compare results
        comparison_report = self._analyze_test_results(test_results)
        
        # Save results
        self._save_test_results(test_results, comparison_report)
        
        return {
            "test_id": self.current_test_id,
            "variant_results": test_results,
            "comparison_report": comparison_report,
            "winner": comparison_report.get("best_variant"),
            "significant_differences": comparison_report.get("statistical_significance", {})
        }
    
    def _run_single_variant_test(self, variant: TestVariant, test_dataset: List[str],
                               test_duration: int, metrics_to_collect: List[str]) -> TestResult:
        """Run test for a single pipeline variant."""
        
        logger.info(f"Starting test for variant: {variant.name}")
        start_time = time.time()
        
        # Initialize pipeline with variant configuration
        pipeline = self._create_pipeline_from_variant(variant)
        
        # Metrics collection
        execution_times = []
        memory_usages = []
        cpu_usages = []
        gpu_usages = []
        accuracy_scores = []
        errors = 0
        processed_frames = 0
        
        # Performance monitoring
        performance_monitor = self._create_performance_monitor()
        
        try:
            for frame_path in test_dataset:
                if time.time() - start_time >= test_duration:
                    break
                
                frame_start = time.time()
                
                # Process frame
                try:
                    result = pipeline.process_frame(frame_path)
                    
                    # Collect metrics
                    execution_times.append(time.time() - frame_start)
                    
                    # System metrics
                    sys_metrics = performance_monitor.get_current_metrics()
                    memory_usages.append(sys_metrics["memory_mb"])
                    cpu_usages.append(sys_metrics["cpu_percent"])
                    gpu_usages.append(sys_metrics["gpu_percent"])
                    
                    # Accuracy metrics (if ground truth available)
                    if hasattr(result, 'accuracy_score'):
                        accuracy_scores.append(result.accuracy_score)
                    
                    processed_frames += 1
                    
                except Exception as e:
                    logger.warning(f"Frame processing error in {variant.name}: {e}")
                    errors += 1
            
            # Calculate final metrics
            total_time = time.time() - start_time
            
            result = TestResult(
                variant_name=variant.name,
                execution_time=np.mean(execution_times) if execution_times else 0,
                memory_usage=np.mean(memory_usages) if memory_usages else 0,
                cpu_usage=np.mean(cpu_usages) if cpu_usages else 0,
                gpu_usage=np.mean(gpu_usages) if gpu_usages else 0,
                accuracy_metrics={
                    "mean_accuracy": np.mean(accuracy_scores) if accuracy_scores else 0,
                    "accuracy_std": np.std(accuracy_scores) if accuracy_scores else 0
                },
                error_count=errors,
                throughput=processed_frames / total_time if total_time > 0 else 0,
                latency_p95=np.percentile(execution_times, 95) if execution_times else 0,
                success_rate=(processed_frames - errors) / processed_frames if processed_frames > 0 else 0
            )
            
            logger.info(f"Test completed for {variant.name}: {processed_frames} frames processed")
            return result
            
        finally:
            # Cleanup
            if hasattr(pipeline, 'cleanup'):
                pipeline.cleanup()
    
    def _create_pipeline_from_variant(self, variant: TestVariant):
        """Create pipeline instance from variant configuration."""
        # This would create the actual pipeline based on variant config
        # For now, return a mock pipeline
        
        class MockPipeline:
            def __init__(self, variant_config):
                self.config = variant_config
                self.processing_delay = np.random.uniform(0.1, 0.5)  # Simulate different processing times
            
            def process_frame(self, frame_path):
                # Simulate processing
                time.sleep(self.processing_delay)
                
                # Mock result
                class MockResult:
                    def __init__(self):
                        self.accuracy_score = np.random.uniform(0.7, 0.95)
                        self.detections = []
                        self.processing_time = self.processing_delay
                
                return MockResult()
            
            def cleanup(self):
                pass
        
        return MockPipeline(variant)
    
    def _create_performance_monitor(self):
        """Create system performance monitor."""
        import psutil
        
        class PerformanceMonitor:
            def get_current_metrics(self):
                return {
                    "memory_mb": psutil.virtual_memory().used / 1024 / 1024,
                    "cpu_percent": psutil.cpu_percent(),
                    "gpu_percent": self._get_gpu_usage()
                }
            
            def _get_gpu_usage(self):
                try:
                    import GPUtil
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        return gpus[0].load * 100
                except:
                    pass
                return 0.0
        
        return PerformanceMonitor()
    
    def _analyze_test_results(self, test_results: Dict[str, TestResult]) -> Dict[str, Any]:
        """Analyze and compare test results."""
        
        if not test_results or all(result is None for result in test_results.values()):
            return {"error": "No valid test results"}
        
        # Filter valid results
        valid_results = {name: result for name, result in test_results.items() if result is not None}
        
        if len(valid_results) < 2:
            return {"error": "Need at least 2 valid results for comparison"}
        
        # Create comparison dataframe
        comparison_data = []
        for name, result in valid_results.items():
            comparison_data.append({
                "variant": name,
                **asdict(result)
            })
        
        df = pd.DataFrame(comparison_data)
        
        # Statistical analysis
        statistical_tests = {}
        metrics_for_stats = ["execution_time", "memory_usage", "throughput", "latency_p95"]
        
        for metric in metrics_for_stats:
            if metric in df.columns:
                # Perform ANOVA test
                groups = [group[metric].values for name, group in df.groupby("variant")]
                if len(groups) >= 2 and all(len(group) > 0 for group in groups):
                    f_stat, p_value = stats.f_oneway(*groups)
                    statistical_tests[metric] = {
                        "f_statistic": float(f_stat),
                        "p_value": float(p_value),
                        "significant": p_value < 0.05
                    }
        
        # Ranking
        rankings = {}
        
        # Performance ranking (lower is better for time/memory, higher for throughput/accuracy)
        performance_score = {}
        for _, row in df.iterrows():
            score = 0
            score -= row["execution_time"] * 10  # Lower execution time is better
            score -= row["memory_usage"] * 0.01  # Lower memory usage is better  
            score += row["throughput"] * 5  # Higher throughput is better
            score += row["success_rate"] * 100  # Higher success rate is better
            score -= row["latency_p95"] * 10  # Lower latency is better
            
            performance_score[row["variant"]] = score
        
        best_variant = max(performance_score.keys(), key=lambda k: performance_score[k])
        
        rankings = {
            "performance_ranking": sorted(performance_score.items(), key=lambda x: x[1], reverse=True),
            "best_variant": best_variant
        }
        
        return {
            "summary_statistics": df.describe().to_dict(),
            "statistical_significance": statistical_tests,
            "rankings": rankings,
            "best_variant": best_variant,
            "performance_scores": performance_score
        }
    
    def _save_test_results(self, test_results: Dict[str, TestResult], comparison_report: Dict[str, Any]):
        """Save test results to files."""
        
        # Save raw results as JSON
        results_file = self.output_dir / f"{self.current_test_id}_results.json"
        
        serializable_results = {}
        for name, result in test_results.items():
            if result is not None:
                serializable_results[name] = asdict(result)
        
        with open(results_file, 'w') as f:
            json.dump({
                "test_id": self.current_test_id,
                "results": serializable_results,
                "comparison": comparison_report
            }, f, indent=2)
        
        # Save comparison report
        report_file = self.output_dir / f"{self.current_test_id}_report.json"
        with open(report_file, 'w') as f:
            json.dump(comparison_report, f, indent=2)
        
        logger.info(f"Test results saved to {results_file}")
    
    def visualize_results(self, test_results: Dict[str, TestResult], output_file: Optional[str] = None):
        """Create visualizations of test results."""
        
        if not test_results or all(result is None for result in test_results.values()):
            logger.warning("No valid results to visualize")
            return
        
        # Filter valid results
        valid_results = {name: result for name, result in test_results.items() if result is not None}
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'A/B Test Results Comparison - {self.current_test_id}', fontsize=16)
        
        # Prepare data
        variants = list(valid_results.keys())
        execution_times = [result.execution_time for result in valid_results.values()]
        memory_usages = [result.memory_usage for result in valid_results.values()]
        throughputs = [result.throughput for result in valid_results.values()]
        success_rates = [result.success_rate for result in valid_results.values()]
        latencies = [result.latency_p95 for result in valid_results.values()]
        accuracies = [result.accuracy_metrics.get("mean_accuracy", 0) for result in valid_results.values()]
        
        # Plot 1: Execution Time
        axes[0, 0].bar(variants, execution_times, color='skyblue')
        axes[0, 0].set_title('Average Execution Time (s)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Memory Usage
        axes[0, 1].bar(variants, memory_usages, color='lightgreen')
        axes[0, 1].set_title('Average Memory Usage (MB)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Throughput
        axes[0, 2].bar(variants, throughputs, color='orange')
        axes[0, 2].set_title('Throughput (frames/s)')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # Plot 4: Success Rate
        axes[1, 0].bar(variants, success_rates, color='lightcoral')
        axes[1, 0].set_title('Success Rate')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].set_ylim([0, 1])
        
        # Plot 5: Latency P95
        axes[1, 1].bar(variants, latencies, color='gold')
        axes[1, 1].set_title('95th Percentile Latency (s)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Plot 6: Accuracy
        axes[1, 2].bar(variants, accuracies, color='mediumpurple')
        axes[1, 2].set_title('Mean Accuracy')
        axes[1, 2].tick_params(axis='x', rotation=45)
        axes[1, 2].set_ylim([0, 1])
        
        plt.tight_layout()
        
        # Save plot
        if output_file is None:
            output_file = self.output_dir / f"{self.current_test_id}_comparison.png"
        
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Visualization saved to {output_file}")
    
    def generate_report(self, test_results: Dict[str, TestResult], 
                       comparison_report: Dict[str, Any]) -> str:
        """Generate detailed text report."""
        
        report_lines = [
            f"# A/B Test Report - {self.current_test_id}",
            f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Test Configuration",
            f"Number of variants tested: {len(self.test_variants)}",
            f"Variants: {', '.join(self.test_variants.keys())}",
            "",
            "## Results Summary"
        ]
        
        if comparison_report.get("best_variant"):
            report_lines.extend([
                f"**Winner: {comparison_report['best_variant']}**",
                ""
            ])
        
        # Add detailed results for each variant
        report_lines.append("## Detailed Results")
        
        valid_results = {name: result for name, result in test_results.items() if result is not None}
        
        for variant_name, result in valid_results.items():
            variant_config = self.test_variants.get(variant_name)
            report_lines.extend([
                f"### {variant_name}",
                f"Description: {variant_config.description if variant_config else 'N/A'}",
                f"- Execution Time: {result.execution_time:.3f}s",
                f"- Memory Usage: {result.memory_usage:.1f}MB", 
                f"- Throughput: {result.throughput:.2f} frames/s",
                f"- Success Rate: {result.success_rate:.2%}",
                f"- 95th Percentile Latency: {result.latency_p95:.3f}s",
                f"- Mean Accuracy: {result.accuracy_metrics.get('mean_accuracy', 0):.3f}",
                f"- Error Count: {result.error_count}",
                ""
            ])
        
        # Statistical significance
        if "statistical_significance" in comparison_report:
            report_lines.append("## Statistical Analysis")
            
            for metric, stats in comparison_report["statistical_significance"].items():
                significance = "Significant" if stats.get("significant", False) else "Not Significant"
                report_lines.extend([
                    f"### {metric}",
                    f"- F-statistic: {stats.get('f_statistic', 0):.4f}",
                    f"- P-value: {stats.get('p_value', 1):.4f}",
                    f"- Result: {significance}",
                    ""
                ])
        
        # Rankings
        if "rankings" in comparison_report:
            report_lines.append("## Performance Rankings")
            
            for rank, (variant, score) in enumerate(comparison_report["rankings"]["performance_ranking"], 1):
                report_lines.append(f"{rank}. {variant} (Score: {score:.2f})")
            
            report_lines.append("")
        
        # Recommendations
        report_lines.extend([
            "## Recommendations",
            "",
            f"Based on the test results, **{comparison_report.get('best_variant', 'N/A')}** shows the best overall performance.",
            "Consider the following factors when choosing a variant:",
            "",
            "- **Performance**: Balance between speed and accuracy",
            "- **Resource Usage**: Memory and CPU constraints", 
            "- **Reliability**: Success rate and error handling",
            "- **Scalability**: Throughput under load",
            ""
        ])
        
        return "\n".join(report_lines)