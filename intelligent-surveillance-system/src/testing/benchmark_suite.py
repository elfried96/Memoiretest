"""Complete benchmark suite for pipeline testing and comparison."""

import time
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import cpu_count

from .ab_testing import ABTestFramework, TestVariant
from .performance_metrics import PerformanceCollector, MetricsVisualizer
from ..advanced_tools import *

logger = logging.getLogger(__name__)

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark tests."""
    test_name: str
    dataset_path: str
    test_duration: int = 300  # seconds
    concurrent_streams: int = 1
    repeat_count: int = 3
    metrics_collection_interval: float = 0.5
    output_dir: str = "benchmark_results"
    hardware_profiling: bool = True
    memory_profiling: bool = True
    
@dataclass
class BenchmarkResult:
    """Complete benchmark result."""
    config: BenchmarkConfig
    variant_results: Dict[str, Any]
    performance_summary: Dict[str, float]
    hardware_utilization: Dict[str, float]
    comparison_report: Dict[str, Any]
    execution_time: float
    timestamp: str

class BenchmarkSuite:
    """Complete benchmark suite for testing different pipeline configurations."""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize testing components
        self.ab_framework = ABTestFramework(str(self.output_dir / "ab_tests"))
        self.performance_collector = PerformanceCollector()
        self.visualizer = MetricsVisualizer(str(self.output_dir / "visualizations"))
        
        # Benchmark configurations
        self.benchmark_configs = {}
        
        # Results storage
        self.benchmark_history = []
        
        self._setup_default_benchmarks()
    
    def _setup_default_benchmarks(self):
        """Setup default benchmark configurations."""
        
        # Performance benchmark
        self.add_benchmark_config(BenchmarkConfig(
            test_name="performance_benchmark",
            dataset_path="test_data/performance_test",
            test_duration=300,
            concurrent_streams=5,
            repeat_count=3,
            hardware_profiling=True,
            memory_profiling=True
        ))
        
        # Accuracy benchmark
        self.add_benchmark_config(BenchmarkConfig(
            test_name="accuracy_benchmark", 
            dataset_path="test_data/accuracy_test",
            test_duration=600,
            concurrent_streams=1,
            repeat_count=5,
            hardware_profiling=False,
            memory_profiling=False
        ))
        
        # Scalability benchmark
        self.add_benchmark_config(BenchmarkConfig(
            test_name="scalability_benchmark",
            dataset_path="test_data/scalability_test",
            test_duration=180,
            concurrent_streams=20,
            repeat_count=2,
            hardware_profiling=True,
            memory_profiling=True
        ))
        
        # Robustness benchmark
        self.add_benchmark_config(BenchmarkConfig(
            test_name="robustness_benchmark",
            dataset_path="test_data/robustness_test",
            test_duration=450,
            concurrent_streams=3,
            repeat_count=4,
            hardware_profiling=True,
            memory_profiling=False
        ))
    
    def add_benchmark_config(self, config: BenchmarkConfig):
        """Add a new benchmark configuration."""
        self.benchmark_configs[config.test_name] = config
        logger.info(f"Added benchmark configuration: {config.test_name}")
    
    def setup_test_variants(self):
        """Setup comprehensive test variants."""
        self.ab_framework.setup_default_variants()
        
        # Add additional advanced variants
        advanced_variants = [
            TestVariant(
                name="sam2_enhanced",
                pipeline_type="advanced_tools",
                components=["yolo_detector", "sam2_segmentation", "vlm_analyzer", "cross_validator"],
                parameters={
                    "sam2_confidence": 0.8,
                    "segmentation_enabled": True,
                    "pixel_level_analysis": True
                },
                enabled_tools=["object_detector", "segmentation_tool", "behavior_classifier"],
                description="SAM2-enhanced pipeline with pixel-level analysis"
            ),
            
            TestVariant(
                name="dino_features",
                pipeline_type="advanced_tools", 
                components=["yolo_detector", "dino_features", "vlm_analyzer", "cross_validator"],
                parameters={
                    "dino_model": "dinov2_vitb14",
                    "feature_similarity_threshold": 0.8,
                    "attention_analysis": True
                },
                enabled_tools=["object_detector", "feature_extractor", "behavior_classifier"],
                description="DINO v2 feature-enhanced pipeline"
            ),
            
            TestVariant(
                name="pose_behavior_analysis",
                pipeline_type="advanced_tools",
                components=["yolo_detector", "pose_estimation", "behavior_classifier", "vlm_analyzer"],
                parameters={
                    "pose_model": "mediapipe",
                    "behavior_analysis_window": 30,
                    "pose_confidence_threshold": 0.5
                },
                enabled_tools=["object_detector", "pose_analyzer", "behavior_classifier"],
                description="Pose-based behavioral analysis pipeline"
            ),
            
            TestVariant(
                name="temporal_transformer",
                pipeline_type="advanced_tools",
                components=["yolo_detector", "temporal_transformer", "vlm_analyzer", "cross_validator"],
                parameters={
                    "sequence_length": 30,
                    "temporal_analysis": True,
                    "pattern_detection": True
                },
                enabled_tools=["object_detector", "behavior_classifier", "trajectory_analyzer"],
                description="Temporal transformer for sequence analysis"
            ),
            
            TestVariant(
                name="multimodal_fusion",
                pipeline_type="advanced_tools",
                components=["yolo_detector", "dino_features", "pose_estimation", "multimodal_fusion", "vlm_analyzer"],
                parameters={
                    "fusion_method": "attention",
                    "modality_weights": {"visual": 0.3, "detection": 0.25, "pose": 0.2, "motion": 0.25},
                    "fusion_threshold": 0.7
                },
                enabled_tools=["object_detector", "feature_extractor", "pose_analyzer", "behavior_classifier"],
                description="Multimodal fusion of all analysis types"
            ),
            
            TestVariant(
                name="adversarial_robust",
                pipeline_type="advanced_tools",
                components=["adversarial_detector", "yolo_detector", "vlm_analyzer", "cross_validator"],
                parameters={
                    "adversarial_detection": True,
                    "robustness_threshold": 0.8,
                    "defensive_preprocessing": True
                },
                enabled_tools=["object_detector", "behavior_classifier"],
                description="Adversarial attack robust pipeline"
            ),
            
            TestVariant(
                name="domain_adaptive",
                pipeline_type="advanced_tools",
                components=["domain_adapter", "yolo_detector", "vlm_analyzer", "cross_validator"],
                parameters={
                    "domain_adaptation": True,
                    "adaptation_threshold": 0.7,
                    "auto_domain_detection": True
                },
                enabled_tools=["object_detector", "behavior_classifier"],
                description="Domain-adaptive pipeline for different environments"
            ),
            
            TestVariant(
                name="complete_advanced",
                pipeline_type="full_pipeline",
                components=[
                    "adversarial_detector", "domain_adapter", "yolo_detector",
                    "sam2_segmentation", "dino_features", "pose_estimation", 
                    "trajectory_analyzer", "multimodal_fusion", "temporal_transformer",
                    "vlm_analyzer", "cross_validator"
                ],
                parameters={
                    "all_advanced_tools": True,
                    "optimal_configuration": True,
                    "adaptive_thresholds": True,
                    "real_time_optimization": True
                },
                enabled_tools=[
                    "object_detector", "segmentation_tool", "feature_extractor",
                    "pose_analyzer", "motion_analyzer", "behavior_classifier",
                    "crowd_counter", "trajectory_analyzer"
                ],
                description="Complete advanced pipeline with all tools"
            )
        ]
        
        # Register advanced variants
        for variant in advanced_variants:
            self.ab_framework.register_variant(variant)
        
        logger.info(f"Setup {len(advanced_variants)} additional advanced variants")
    
    def run_comprehensive_benchmark(self, benchmark_name: str, 
                                  custom_dataset: Optional[str] = None) -> BenchmarkResult:
        """Run comprehensive benchmark with all variants."""
        
        if benchmark_name not in self.benchmark_configs:
            raise ValueError(f"Unknown benchmark: {benchmark_name}")
        
        config = self.benchmark_configs[benchmark_name]
        if custom_dataset:
            config.dataset_path = custom_dataset
        
        start_time = time.time()
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        logger.info(f"Starting comprehensive benchmark: {benchmark_name}")
        
        # Setup test variants
        self.setup_test_variants()
        
        # Prepare test dataset
        test_dataset = self._prepare_test_dataset(config.dataset_path)
        
        # Run multiple iterations for statistical significance
        all_results = []
        
        for iteration in range(config.repeat_count):
            logger.info(f"Running benchmark iteration {iteration + 1}/{config.repeat_count}")
            
            # Start performance collection
            self.performance_collector.start_collection()
            
            # Run A/B test
            ab_results = self.ab_framework.run_ab_test(
                test_dataset=test_dataset,
                test_duration=config.test_duration,
                concurrent_variants=min(config.concurrent_streams, cpu_count()),
                metrics_to_collect=[
                    "execution_time", "memory_usage", "accuracy", 
                    "false_positive_rate", "throughput", "latency", 
                    "cpu_usage", "gpu_usage"
                ]
            )
            
            # Stop performance collection
            perf_report = self.performance_collector.stop_collection()
            
            all_results.append({
                'iteration': iteration,
                'ab_results': ab_results,
                'performance_report': perf_report
            })
            
            time.sleep(10)  # Cool-down between iterations
        
        # Aggregate results
        aggregated_results = self._aggregate_iteration_results(all_results)
        
        # Hardware utilization analysis
        hardware_utilization = self._analyze_hardware_utilization(all_results)
        
        # Performance summary
        performance_summary = self._generate_performance_summary(aggregated_results)
        
        # Comparison report
        comparison_report = self._generate_comparison_report(aggregated_results)
        
        execution_time = time.time() - start_time
        
        # Create benchmark result
        benchmark_result = BenchmarkResult(
            config=config,
            variant_results=aggregated_results,
            performance_summary=performance_summary,
            hardware_utilization=hardware_utilization,
            comparison_report=comparison_report,
            execution_time=execution_time,
            timestamp=timestamp
        )
        
        # Save results
        self._save_benchmark_result(benchmark_result)
        
        # Generate visualizations
        self._generate_benchmark_visualizations(benchmark_result)
        
        # Add to history
        self.benchmark_history.append(benchmark_result)
        
        logger.info(f"Completed comprehensive benchmark: {benchmark_name} in {execution_time:.2f}s")
        
        return benchmark_result
    
    def _prepare_test_dataset(self, dataset_path: str) -> List[str]:
        """Prepare test dataset from path."""
        dataset_path = Path(dataset_path)
        
        if not dataset_path.exists():
            # Create mock dataset for testing
            logger.warning(f"Dataset path {dataset_path} not found, creating mock dataset")
            return self._create_mock_dataset(100)  # 100 mock frames
        
        # Load real dataset
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(dataset_path.glob(f"*{ext}"))
            image_files.extend(dataset_path.glob(f"**/*{ext}"))
        
        return [str(f) for f in image_files[:1000]]  # Limit to 1000 images
    
    def _create_mock_dataset(self, size: int) -> List[str]:
        """Create mock dataset for testing."""
        mock_dataset = []
        for i in range(size):
            # Create mock file paths
            mock_dataset.append(f"mock_frame_{i:04d}.jpg")
        return mock_dataset
    
    def _aggregate_iteration_results(self, all_results: List[Dict]) -> Dict[str, Any]:
        """Aggregate results across multiple iterations."""
        
        aggregated = {}
        
        # Get all variant names
        variant_names = set()
        for result in all_results:
            if 'ab_results' in result and 'variant_results' in result['ab_results']:
                variant_names.update(result['ab_results']['variant_results'].keys())
        
        # Aggregate for each variant
        for variant_name in variant_names:
            variant_data = []
            
            for result in all_results:
                ab_results = result.get('ab_results', {})
                variant_results = ab_results.get('variant_results', {})
                
                if variant_name in variant_results and variant_results[variant_name] is not None:
                    variant_data.append(variant_results[variant_name])
            
            if variant_data:
                # Calculate statistics across iterations
                aggregated[variant_name] = self._calculate_variant_statistics(variant_data)
        
        return aggregated
    
    def _calculate_variant_statistics(self, variant_data: List[Any]) -> Dict[str, Any]:
        """Calculate statistics for variant across iterations."""
        
        if not variant_data:
            return {}
        
        # Extract numeric metrics
        metrics = {}
        
        for data in variant_data:
            if hasattr(data, '__dict__'):
                data_dict = asdict(data)
            else:
                data_dict = data
            
            for key, value in data_dict.items():
                if isinstance(value, (int, float)):
                    if key not in metrics:
                        metrics[key] = []
                    metrics[key].append(value)
        
        # Calculate statistics
        statistics = {}
        for metric_name, values in metrics.items():
            statistics[metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values),
                'p95': np.percentile(values, 95),
                'count': len(values)
            }
        
        return statistics
    
    def _analyze_hardware_utilization(self, all_results: List[Dict]) -> Dict[str, float]:
        """Analyze hardware utilization across all tests."""
        
        cpu_usage = []
        memory_usage = []
        gpu_usage = []
        
        for result in all_results:
            if 'performance_report' in result:
                perf_report = result['performance_report']
                if hasattr(perf_report, 'average_cpu'):
                    cpu_usage.append(perf_report.average_cpu)
                if hasattr(perf_report, 'peak_memory_mb'):
                    memory_usage.append(perf_report.peak_memory_mb)
                # Add GPU usage if available
        
        utilization = {}
        
        if cpu_usage:
            utilization['cpu'] = {
                'average': np.mean(cpu_usage),
                'peak': np.max(cpu_usage),
                'efficiency': np.mean(cpu_usage) / 100.0  # Normalize
            }
        
        if memory_usage:
            utilization['memory'] = {
                'average_mb': np.mean(memory_usage),
                'peak_mb': np.max(memory_usage),
                'efficiency': np.mean(memory_usage) / 16000  # Assume 16GB baseline
            }
        
        return utilization
    
    def _generate_performance_summary(self, aggregated_results: Dict[str, Any]) -> Dict[str, float]:
        """Generate performance summary across all variants."""
        
        summary = {}
        
        # Overall performance metrics
        all_execution_times = []
        all_memory_usage = []
        all_throughputs = []
        all_error_rates = []
        
        for variant_name, variant_stats in aggregated_results.items():
            if 'execution_time' in variant_stats:
                all_execution_times.append(variant_stats['execution_time']['mean'])
            
            if 'memory_usage' in variant_stats:
                all_memory_usage.append(variant_stats['memory_usage']['mean'])
            
            if 'throughput' in variant_stats:
                all_throughputs.append(variant_stats['throughput']['mean'])
            
            if 'error_rate' in variant_stats:
                all_error_rates.append(variant_stats['error_rate']['mean'])
        
        # Calculate summary statistics
        if all_execution_times:
            summary['avg_execution_time'] = np.mean(all_execution_times)
            summary['best_execution_time'] = np.min(all_execution_times)
        
        if all_memory_usage:
            summary['avg_memory_usage'] = np.mean(all_memory_usage)
            summary['peak_memory_usage'] = np.max(all_memory_usage)
        
        if all_throughputs:
            summary['avg_throughput'] = np.mean(all_throughputs)
            summary['best_throughput'] = np.max(all_throughputs)
        
        if all_error_rates:
            summary['avg_error_rate'] = np.mean(all_error_rates)
            summary['best_error_rate'] = np.min(all_error_rates)
        
        # Performance efficiency score
        if all_execution_times and all_memory_usage and all_throughputs:
            efficiency_scores = []
            for i in range(len(all_execution_times)):
                # Higher throughput, lower execution time, lower memory = higher efficiency
                efficiency = (all_throughputs[i] / (all_execution_times[i] * all_memory_usage[i] / 1000))
                efficiency_scores.append(efficiency)
            
            summary['avg_efficiency_score'] = np.mean(efficiency_scores)
            summary['best_efficiency_score'] = np.max(efficiency_scores)
        
        return summary
    
    def _generate_comparison_report(self, aggregated_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed comparison report."""
        
        if not aggregated_results:
            return {}
        
        # Ranking variants by different criteria
        rankings = {}
        
        # Execution time ranking (lower is better)
        execution_times = {}
        for variant, stats in aggregated_results.items():
            if 'execution_time' in stats:
                execution_times[variant] = stats['execution_time']['mean']
        
        if execution_times:
            rankings['execution_time'] = sorted(execution_times.items(), key=lambda x: x[1])
        
        # Throughput ranking (higher is better) 
        throughputs = {}
        for variant, stats in aggregated_results.items():
            if 'throughput' in stats:
                throughputs[variant] = stats['throughput']['mean']
        
        if throughputs:
            rankings['throughput'] = sorted(throughputs.items(), key=lambda x: x[1], reverse=True)
        
        # Memory efficiency ranking (lower is better)
        memory_usage = {}
        for variant, stats in aggregated_results.items():
            if 'memory_usage' in stats:
                memory_usage[variant] = stats['memory_usage']['mean']
        
        if memory_usage:
            rankings['memory_efficiency'] = sorted(memory_usage.items(), key=lambda x: x[1])
        
        # Overall performance ranking (composite score)
        overall_scores = {}
        for variant, stats in aggregated_results.items():
            score = 0
            factors = 0
            
            if 'execution_time' in stats:
                # Lower is better, so take reciprocal
                score += 1.0 / (stats['execution_time']['mean'] + 0.1)
                factors += 1
            
            if 'throughput' in stats:
                # Higher is better
                score += stats['throughput']['mean']
                factors += 1
            
            if 'memory_usage' in stats:
                # Lower is better, so take reciprocal and scale
                score += 1000.0 / (stats['memory_usage']['mean'] + 100)
                factors += 1
            
            if factors > 0:
                overall_scores[variant] = score / factors
        
        if overall_scores:
            rankings['overall_performance'] = sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Statistical significance testing
        significance_tests = {}
        
        # Compare top 2 variants if available
        if 'overall_performance' in rankings and len(rankings['overall_performance']) >= 2:
            top_variant = rankings['overall_performance'][0][0]
            second_variant = rankings['overall_performance'][1][0]
            
            # Get execution time data for both
            top_times = []
            second_times = []
            
            top_stats = aggregated_results[top_variant]
            second_stats = aggregated_results[second_variant]
            
            # This is a simplified statistical test
            # In practice, you'd want to use the raw data points
            if 'execution_time' in top_stats and 'execution_time' in second_stats:
                top_mean = top_stats['execution_time']['mean']
                top_std = top_stats['execution_time']['std']
                second_mean = second_stats['execution_time']['mean']
                second_std = second_stats['execution_time']['std']
                
                # Simple t-test approximation
                pooled_std = np.sqrt((top_std**2 + second_std**2) / 2)
                t_stat = abs(top_mean - second_mean) / (pooled_std + 1e-8)
                
                significance_tests['top_vs_second'] = {
                    'variants': [top_variant, second_variant],
                    't_statistic': t_stat,
                    'significant': t_stat > 2.0,  # Simplified threshold
                    'confidence_level': 'high' if t_stat > 2.5 else 'medium' if t_stat > 2.0 else 'low'
                }
        
        return {
            'rankings': rankings,
            'statistical_significance': significance_tests,
            'variant_count': len(aggregated_results),
            'best_overall_variant': rankings['overall_performance'][0][0] if 'overall_performance' in rankings else None
        }
    
    def _save_benchmark_result(self, result: BenchmarkResult):
        """Save benchmark result to file."""
        
        # Create output file path
        timestamp = result.timestamp
        filename = f"benchmark_{result.config.test_name}_{timestamp}.json"
        filepath = self.output_dir / filename
        
        # Convert result to serializable format
        result_dict = {
            'config': asdict(result.config),
            'variant_results': result.variant_results,
            'performance_summary': result.performance_summary,
            'hardware_utilization': result.hardware_utilization,
            'comparison_report': result.comparison_report,
            'execution_time': result.execution_time,
            'timestamp': result.timestamp
        }
        
        # Save to JSON
        with open(filepath, 'w') as f:
            json.dump(result_dict, f, indent=2, default=str)
        
        logger.info(f"Saved benchmark result to {filepath}")
    
    def _generate_benchmark_visualizations(self, result: BenchmarkResult):
        """Generate visualizations for benchmark results."""
        
        # Create performance comparison plots
        if result.variant_results:
            # Convert to format expected by visualizer
            variant_reports = {}
            
            for variant_name, variant_stats in result.variant_results.items():
                # Create mock PerformanceReport for visualization
                class MockPerformanceReport:
                    def __init__(self, stats):
                        self.average_fps = stats.get('throughput', {}).get('mean', 0)
                        self.average_latency = stats.get('execution_time', {}).get('mean', 0) * 1000
                        self.peak_memory_mb = stats.get('memory_usage', {}).get('max', 0)
                        self.average_cpu = stats.get('cpu_usage', {}).get('mean', 0)
                        self.average_gpu = stats.get('gpu_usage', {}).get('mean', 0)
                        self.error_rate = stats.get('error_rate', {}).get('mean', 0)
                        self.resource_efficiency = 75.0  # Mock value
                
                variant_reports[variant_name] = MockPerformanceReport(variant_stats)
            
            # Generate comparison plots
            timestamp = result.timestamp
            
            self.visualizer.plot_performance_comparison(
                variant_reports,
                output_file=self.output_dir / f"performance_comparison_{timestamp}.png"
            )
            
            self.visualizer.create_efficiency_radar_chart(
                variant_reports,
                output_file=self.output_dir / f"efficiency_radar_{timestamp}.html"
            )
            
            self.visualizer.generate_performance_heatmap(
                variant_reports,
                output_file=self.output_dir / f"performance_heatmap_{timestamp}.png"
            )
        
        logger.info(f"Generated visualizations for benchmark {result.config.test_name}")
    
    def generate_summary_report(self) -> str:
        """Generate text summary of all benchmark results."""
        
        if not self.benchmark_history:
            return "No benchmark results available."
        
        report_lines = [
            "# Surveillance System Benchmark Summary",
            f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total benchmarks run: {len(self.benchmark_history)}",
            ""
        ]
        
        for i, result in enumerate(self.benchmark_history):
            report_lines.extend([
                f"## Benchmark {i+1}: {result.config.test_name}",
                f"Timestamp: {result.timestamp}",
                f"Execution time: {result.execution_time:.2f}s",
                f"Variants tested: {len(result.variant_results)}",
                ""
            ])
            
            # Best performance summary
            if result.comparison_report.get('best_overall_variant'):
                best_variant = result.comparison_report['best_overall_variant']
                report_lines.extend([
                    f"**Best performing variant: {best_variant}**",
                    ""
                ])
            
            # Key metrics
            if result.performance_summary:
                report_lines.append("### Key Performance Metrics:")
                for metric, value in result.performance_summary.items():
                    if isinstance(value, float):
                        report_lines.append(f"- {metric}: {value:.4f}")
                    else:
                        report_lines.append(f"- {metric}: {value}")
                report_lines.append("")
            
            # Hardware utilization
            if result.hardware_utilization:
                report_lines.append("### Hardware Utilization:")
                for resource, stats in result.hardware_utilization.items():
                    report_lines.append(f"- {resource.upper()}:")
                    for stat_name, stat_value in stats.items():
                        if isinstance(stat_value, float):
                            report_lines.append(f"  - {stat_name}: {stat_value:.2f}")
                        else:
                            report_lines.append(f"  - {stat_name}: {stat_value}")
                report_lines.append("")
        
        # Overall insights
        if len(self.benchmark_history) > 1:
            report_lines.extend([
                "## Overall Insights",
                "",
                "### Consistent Top Performers:",
                # This would analyze which variants perform well across benchmarks
                "",
                "### Performance Trends:",
                # This would show trends across multiple benchmark runs
                "",
                "### Recommendations:",
                "- Consider the best overall variant for production deployment",
                "- Monitor hardware utilization for optimal resource allocation", 
                "- Regular benchmarking recommended to track performance changes",
                ""
            ])
        
        return "\n".join(report_lines)
    
    def export_results_csv(self, output_file: Optional[str] = None) -> str:
        """Export benchmark results to CSV for analysis."""
        
        if not self.benchmark_history:
            logger.warning("No benchmark results to export")
            return ""
        
        if output_file is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_file = str(self.output_dir / f"benchmark_results_{timestamp}.csv")
        
        # Prepare data for CSV export
        data_rows = []
        
        for result in self.benchmark_history:
            base_row = {
                'benchmark_name': result.config.test_name,
                'timestamp': result.timestamp,
                'execution_time': result.execution_time,
                'test_duration': result.config.test_duration,
                'concurrent_streams': result.config.concurrent_streams,
                'repeat_count': result.config.repeat_count
            }
            
            # Add performance summary
            base_row.update(result.performance_summary)
            
            # Add variant results
            for variant_name, variant_stats in result.variant_results.items():
                row = base_row.copy()
                row['variant_name'] = variant_name
                
                # Add variant statistics
                for metric_name, stats in variant_stats.items():
                    if isinstance(stats, dict) and 'mean' in stats:
                        row[f'{metric_name}_mean'] = stats['mean']
                        row[f'{metric_name}_std'] = stats['std']
                        row[f'{metric_name}_min'] = stats['min']
                        row[f'{metric_name}_max'] = stats['max']
                
                data_rows.append(row)
        
        # Create DataFrame and save
        df = pd.DataFrame(data_rows)
        df.to_csv(output_file, index=False)
        
        logger.info(f"Exported benchmark results to {output_file}")
        return output_file