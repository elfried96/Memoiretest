"""Performance metrics collection and visualization system."""

import time
import threading
import psutil
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from collections import deque
import logging
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

logger = logging.getLogger(__name__)

@dataclass
class MetricsSnapshot:
    """Single point-in-time metrics snapshot."""
    timestamp: float
    cpu_percent: float
    memory_mb: float
    memory_percent: float
    gpu_percent: float
    gpu_memory_mb: float
    fps: float
    latency_ms: float
    queue_size: int
    error_count: int
    custom_metrics: Dict[str, float] = field(default_factory=dict)

@dataclass
class PerformanceReport:
    """Complete performance analysis report."""
    test_duration: float
    total_frames: int
    average_fps: float
    average_latency: float
    peak_memory_mb: float
    average_cpu: float
    average_gpu: float
    error_rate: float
    percentiles: Dict[str, float]
    resource_efficiency: float

class PerformanceCollector:
    """Real-time performance metrics collector."""
    
    def __init__(self, collection_interval: float = 0.5, history_size: int = 1000):
        self.collection_interval = collection_interval
        self.history = deque(maxlen=history_size)
        self.is_collecting = False
        self.collection_thread = None
        self.custom_metrics = {}
        self.callbacks = []
        
        # Counters
        self.total_frames = 0
        self.total_errors = 0
        self.start_time = None
        
    def start_collection(self):
        """Start metrics collection."""
        if self.is_collecting:
            logger.warning("Performance collection already running")
            return
        
        self.is_collecting = True
        self.start_time = time.time()
        self.total_frames = 0
        self.total_errors = 0
        
        self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collection_thread.start()
        
        logger.info("Started performance metrics collection")
    
    def stop_collection(self) -> PerformanceReport:
        """Stop collection and return final report."""
        if not self.is_collecting:
            logger.warning("Performance collection not running")
            return None
        
        self.is_collecting = False
        if self.collection_thread:
            self.collection_thread.join(timeout=2.0)
        
        report = self._generate_report()
        logger.info("Stopped performance metrics collection")
        return report
    
    def _collection_loop(self):
        """Main collection loop running in separate thread."""
        while self.is_collecting:
            try:
                snapshot = self._collect_snapshot()
                self.history.append(snapshot)
                
                # Call registered callbacks
                for callback in self.callbacks:
                    try:
                        callback(snapshot)
                    except Exception as e:
                        logger.error(f"Metrics callback error: {e}")
                
                time.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                time.sleep(self.collection_interval)
    
    def _collect_snapshot(self) -> MetricsSnapshot:
        """Collect single metrics snapshot."""
        current_time = time.time()
        
        # System metrics
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        memory_mb = memory.used / 1024 / 1024
        memory_percent = memory.percent
        
        # GPU metrics
        gpu_percent, gpu_memory_mb = self._get_gpu_metrics()
        
        # Calculate FPS
        fps = self._calculate_current_fps()
        
        # Get latest latency
        latency_ms = self.custom_metrics.get('latest_latency_ms', 0.0)
        
        # Queue size (if available)
        queue_size = self.custom_metrics.get('queue_size', 0)
        
        return MetricsSnapshot(
            timestamp=current_time,
            cpu_percent=cpu_percent,
            memory_mb=memory_mb,
            memory_percent=memory_percent,
            gpu_percent=gpu_percent,
            gpu_memory_mb=gpu_memory_mb,
            fps=fps,
            latency_ms=latency_ms,
            queue_size=queue_size,
            error_count=self.total_errors,
            custom_metrics=self.custom_metrics.copy()
        )
    
    def _get_gpu_metrics(self) -> tuple:
        """Get GPU utilization and memory usage."""
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Use first GPU
                return gpu.load * 100, gpu.memoryUsed
        except ImportError:
            pass
        
        # Try nvidia-ml-py
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return util.gpu, memory.used / 1024 / 1024
        except:
            pass
        
        return 0.0, 0.0
    
    def _calculate_current_fps(self) -> float:
        """Calculate current FPS based on recent frame processing."""
        if len(self.history) < 2:
            return 0.0
        
        # Look at last few snapshots to calculate FPS
        recent_snapshots = list(self.history)[-10:]  # Last 10 snapshots
        if len(recent_snapshots) < 2:
            return 0.0
        
        time_window = recent_snapshots[-1].timestamp - recent_snapshots[0].timestamp
        frame_count = len(recent_snapshots) - 1
        
        return frame_count / time_window if time_window > 0 else 0.0
    
    def record_frame_processed(self):
        """Record that a frame was processed."""
        self.total_frames += 1
    
    def record_error(self):
        """Record an error occurrence."""
        self.total_errors += 1
    
    def record_latency(self, latency_ms: float):
        """Record processing latency."""
        self.custom_metrics['latest_latency_ms'] = latency_ms
    
    def record_custom_metric(self, name: str, value: float):
        """Record custom metric."""
        self.custom_metrics[name] = value
    
    def add_callback(self, callback: Callable[[MetricsSnapshot], None]):
        """Add callback to be called on each metrics snapshot."""
        self.callbacks.append(callback)
    
    def _generate_report(self) -> PerformanceReport:
        """Generate comprehensive performance report."""
        if not self.history:
            return PerformanceReport(0, 0, 0, 0, 0, 0, 0, 0, {}, 0)
        
        # Convert to arrays for analysis
        snapshots = list(self.history)
        
        cpu_values = [s.cpu_percent for s in snapshots]
        memory_values = [s.memory_mb for s in snapshots]
        gpu_values = [s.gpu_percent for s in snapshots]
        latency_values = [s.latency_ms for s in snapshots if s.latency_ms > 0]
        fps_values = [s.fps for s in snapshots if s.fps > 0]
        
        # Calculate statistics
        test_duration = time.time() - self.start_time if self.start_time else 0
        average_fps = np.mean(fps_values) if fps_values else 0
        average_latency = np.mean(latency_values) if latency_values else 0
        peak_memory = np.max(memory_values) if memory_values else 0
        average_cpu = np.mean(cpu_values) if cpu_values else 0
        average_gpu = np.mean(gpu_values) if gpu_values else 0
        error_rate = self.total_errors / max(self.total_frames, 1)
        
        # Calculate percentiles for latency
        percentiles = {}
        if latency_values:
            percentiles = {
                "p50": np.percentile(latency_values, 50),
                "p90": np.percentile(latency_values, 90),
                "p95": np.percentile(latency_values, 95),
                "p99": np.percentile(latency_values, 99)
            }
        
        # Resource efficiency score
        efficiency = self._calculate_efficiency_score(average_cpu, peak_memory, average_fps, error_rate)
        
        return PerformanceReport(
            test_duration=test_duration,
            total_frames=self.total_frames,
            average_fps=average_fps,
            average_latency=average_latency,
            peak_memory_mb=peak_memory,
            average_cpu=average_cpu,
            average_gpu=average_gpu,
            error_rate=error_rate,
            percentiles=percentiles,
            resource_efficiency=efficiency
        )
    
    def _calculate_efficiency_score(self, cpu: float, memory: float, fps: float, error_rate: float) -> float:
        """Calculate overall efficiency score (0-100)."""
        # Normalize metrics
        cpu_score = max(0, 100 - cpu)  # Lower CPU usage is better
        memory_score = max(0, 100 - (memory / 1000))  # Lower memory usage is better  
        fps_score = min(100, fps * 5)  # Higher FPS is better, cap at 20 FPS = 100
        error_score = max(0, 100 - (error_rate * 1000))  # Lower error rate is better
        
        # Weighted average
        weights = [0.25, 0.25, 0.3, 0.2]  # CPU, Memory, FPS, Error
        scores = [cpu_score, memory_score, fps_score, error_score]
        
        return sum(w * s for w, s in zip(weights, scores))

class MetricsVisualizer:
    """Advanced metrics visualization system."""
    
    def __init__(self, output_dir: str = "metrics_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def create_real_time_dashboard(self, collector: PerformanceCollector, 
                                 update_interval: float = 1.0) -> None:
        """Create real-time metrics dashboard using Plotly."""
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=['CPU Usage', 'Memory Usage', 'GPU Usage', 'FPS', 'Latency', 'Error Count'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Initialize empty traces
        fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name='CPU %'), row=1, col=1)
        fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name='Memory MB'), row=1, col=2)
        fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name='GPU %'), row=2, col=1)
        fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name='FPS'), row=2, col=2)
        fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name='Latency ms'), row=3, col=1)
        fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name='Errors'), row=3, col=2)
        
        fig.update_layout(
            title="Real-time Performance Metrics",
            showlegend=False,
            height=800
        )
        
        # This would integrate with a web dashboard framework
        # For now, save static plots
        logger.info("Real-time dashboard would be displayed here")
    
    def plot_performance_comparison(self, results: Dict[str, PerformanceReport], 
                                  output_file: Optional[str] = None) -> None:
        """Create comprehensive performance comparison plots."""
        
        if not results:
            logger.warning("No results to plot")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Performance Metrics Comparison', fontsize=16, fontweight='bold')
        
        # Prepare data
        variants = list(results.keys())
        metrics = {
            'Average FPS': [r.average_fps for r in results.values()],
            'Average Latency (ms)': [r.average_latency for r in results.values()],
            'Peak Memory (MB)': [r.peak_memory_mb for r in results.values()],
            'Average CPU (%)': [r.average_cpu for r in results.values()],
            'Average GPU (%)': [r.average_gpu for r in results.values()],
            'Error Rate (%)': [r.error_rate * 100 for r in results.values()]
        }
        
        # Colors for variants
        colors = plt.cm.Set3(np.linspace(0, 1, len(variants)))
        
        # Plot each metric
        for i, (metric_name, values) in enumerate(metrics.items()):
            row, col = divmod(i, 3)
            ax = axes[row, col]
            
            bars = ax.bar(variants, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
            ax.set_title(metric_name, fontsize=12, fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{value:.2f}', ha='center', va='bottom', fontsize=10)
            
            # Set y-axis to start from 0 for better comparison
            ax.set_ylim(bottom=0)
            
            # Add grid for better readability
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save plot
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.output_dir / f"performance_comparison_{timestamp}.png"
        
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        logger.info(f"Performance comparison plot saved to {output_file}")
    
    def plot_time_series(self, collector: PerformanceCollector, 
                        output_file: Optional[str] = None) -> None:
        """Plot time series of metrics."""
        
        if not collector.history:
            logger.warning("No metrics history to plot")
            return
        
        # Convert history to DataFrame
        snapshots = list(collector.history)
        data = []
        
        for snapshot in snapshots:
            data.append({
                'timestamp': datetime.fromtimestamp(snapshot.timestamp),
                'cpu_percent': snapshot.cpu_percent,
                'memory_mb': snapshot.memory_mb,
                'gpu_percent': snapshot.gpu_percent,
                'fps': snapshot.fps,
                'latency_ms': snapshot.latency_ms
            })
        
        df = pd.DataFrame(data)
        
        # Create time series plots
        fig, axes = plt.subplots(2, 3, figsize=(20, 10))
        fig.suptitle('Performance Metrics Over Time', fontsize=16, fontweight='bold')
        
        # Plot each metric
        metrics = [
            ('cpu_percent', 'CPU Usage (%)', 'tab:blue'),
            ('memory_mb', 'Memory Usage (MB)', 'tab:green'),
            ('gpu_percent', 'GPU Usage (%)', 'tab:red'),
            ('fps', 'FPS', 'tab:orange'),
            ('latency_ms', 'Latency (ms)', 'tab:purple')
        ]
        
        for i, (metric, title, color) in enumerate(metrics):
            if i >= 5:  # Only 5 plots in 2x3 grid
                break
                
            row, col = divmod(i, 3)
            ax = axes[row, col]
            
            ax.plot(df['timestamp'], df[metric], color=color, linewidth=2, alpha=0.8)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Format x-axis
            ax.tick_params(axis='x', rotation=45)
            
            # Add moving average if enough data points
            if len(df) > 10:
                window = min(10, len(df) // 4)
                moving_avg = df[metric].rolling(window=window, center=True).mean()
                ax.plot(df['timestamp'], moving_avg, color='red', linewidth=2, 
                       alpha=0.6, linestyle='--', label=f'Moving Avg ({window})')
                ax.legend()
        
        # Remove empty subplot
        if len(metrics) < 6:
            fig.delaxes(axes[1, 2])
        
        plt.tight_layout()
        
        # Save plot
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.output_dir / f"metrics_timeseries_{timestamp}.png"
        
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        logger.info(f"Time series plot saved to {output_file}")
    
    def create_efficiency_radar_chart(self, results: Dict[str, PerformanceReport],
                                    output_file: Optional[str] = None) -> None:
        """Create radar chart comparing efficiency across variants."""
        
        if not results:
            logger.warning("No results for radar chart")
            return
        
        # Define metrics for radar chart (normalized to 0-100 scale)
        categories = ['FPS', 'Low Latency', 'Memory Efficiency', 'CPU Efficiency', 
                     'GPU Efficiency', 'Reliability']
        
        fig = go.Figure()
        
        for variant_name, report in results.items():
            # Normalize metrics to 0-100 scale
            fps_score = min(100, report.average_fps * 5)  # 20 FPS = 100
            latency_score = max(0, 100 - (report.average_latency / 10))  # Lower is better
            memory_score = max(0, 100 - (report.peak_memory_mb / 100))  # Lower is better
            cpu_score = max(0, 100 - report.average_cpu)  # Lower is better
            gpu_score = max(0, 100 - report.average_gpu)  # Lower is better
            reliability_score = max(0, 100 - (report.error_rate * 1000))  # Lower is better
            
            values = [fps_score, latency_score, memory_score, cpu_score, gpu_score, reliability_score]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=variant_name,
                opacity=0.6
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=True,
            title="Performance Efficiency Comparison (Radar Chart)"
        )
        
        # Save plot
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.output_dir / f"efficiency_radar_{timestamp}.html"
        
        fig.write_html(output_file)
        fig.show()
        
        logger.info(f"Efficiency radar chart saved to {output_file}")
    
    def generate_performance_heatmap(self, results: Dict[str, PerformanceReport],
                                   output_file: Optional[str] = None) -> None:
        """Generate performance metrics heatmap."""
        
        if not results:
            logger.warning("No results for heatmap")
            return
        
        # Create metrics matrix
        variants = list(results.keys())
        metrics_names = ['Average FPS', 'Average Latency', 'Peak Memory', 
                        'Average CPU', 'Average GPU', 'Error Rate', 'Efficiency Score']
        
        data_matrix = []
        for variant in variants:
            report = results[variant]
            row = [
                report.average_fps,
                report.average_latency,
                report.peak_memory_mb / 1000,  # Convert to GB for better scale
                report.average_cpu,
                report.average_gpu,
                report.error_rate * 100,  # Convert to percentage
                report.resource_efficiency
            ]
            data_matrix.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(data_matrix, index=variants, columns=metrics_names)
        
        # Normalize data for better visualization (0-1 scale)
        df_normalized = df.copy()
        for col in df.columns:
            if col in ['Average Latency', 'Peak Memory', 'Average CPU', 'Average GPU', 'Error Rate']:
                # For these metrics, lower is better - invert the scale
                df_normalized[col] = 1 - (df[col] - df[col].min()) / (df[col].max() - df[col].min() + 1e-8)
            else:
                # For these metrics, higher is better
                df_normalized[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min() + 1e-8)
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(df_normalized, annot=True, cmap='RdYlGn', center=0.5, 
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        
        plt.title('Performance Metrics Heatmap (Normalized)', fontsize=16, fontweight='bold')
        plt.xlabel('Metrics', fontsize=12)
        plt.ylabel('Variants', fontsize=12)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        # Save plot
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.output_dir / f"performance_heatmap_{timestamp}.png"
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        logger.info(f"Performance heatmap saved to {output_file}")