"""
Collecteur de métriques système (CPU, mémoire, GPU).
"""

import time
import logging
from typing import Dict, Optional
import psutil

from .performance_monitor import MetricsCollector, PerformanceMetric

logger = logging.getLogger(__name__)

# Import conditionnel pour GPU
try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
except ImportError:
    torch = None
    GPU_AVAILABLE = False

try:
    import nvidia_ml_py as nvidia_ml
    nvidia_ml.nvmlInit()
    NVIDIA_ML_AVAILABLE = True
except (ImportError, Exception):
    nvidia_ml = None
    NVIDIA_ML_AVAILABLE = False


class SystemMetricsCollector(MetricsCollector):
    """
    Collecteur de métriques système pour monitoring des ressources.
    
    Collecte:
    - CPU : utilisation, fréquence, température si disponible
    - Mémoire : utilisation RAM, swap
    - GPU : mémoire, utilisation, température si disponible
    - Disque : espace disponible, I/O
    - Réseau : trafic si applicable
    """
    
    def __init__(self, include_gpu: bool = True, include_detailed: bool = True):
        """
        Initialise le collecteur de métriques système.
        
        Args:
            include_gpu: Inclure les métriques GPU si disponible
            include_detailed: Inclure métriques détaillées (températures, etc.)
        """
        self.include_gpu = include_gpu and GPU_AVAILABLE
        self.include_detailed = include_detailed
        
        # Initialisation des compteurs de base
        self._last_net_io = None
        self._last_disk_io = None
        self._last_timestamp = None
        
        logger.info(f"📊 Collecteur système initialisé (GPU: {self.include_gpu})")
    
    def get_collector_name(self) -> str:
        """Nom du collecteur."""
        return "system_metrics"
    
    async def collect_metrics(self) -> Dict[str, PerformanceMetric]:
        """Collecte toutes les métriques système."""
        timestamp = time.time()
        metrics = {}
        
        # Métriques CPU
        cpu_metrics = self._collect_cpu_metrics(timestamp)
        metrics.update(cpu_metrics)
        
        # Métriques mémoire
        memory_metrics = self._collect_memory_metrics(timestamp)
        metrics.update(memory_metrics)
        
        # Métriques GPU
        if self.include_gpu:
            gpu_metrics = self._collect_gpu_metrics(timestamp)
            metrics.update(gpu_metrics)
        
        # Métriques disque
        disk_metrics = self._collect_disk_metrics(timestamp)
        metrics.update(disk_metrics)
        
        # Métriques réseau
        network_metrics = self._collect_network_metrics(timestamp)
        metrics.update(network_metrics)
        
        self._last_timestamp = timestamp
        
        return metrics
    
    def _collect_cpu_metrics(self, timestamp: float) -> Dict[str, PerformanceMetric]:
        """Collecte les métriques CPU."""
        metrics = {}
        
        try:
            # Utilisation CPU globale
            cpu_percent = psutil.cpu_percent(interval=None)
            metrics["cpu_usage"] = PerformanceMetric(
                name="cpu_usage",
                value=cpu_percent,
                timestamp=timestamp,
                unit="%",
                category="system"
            )
            
            # Utilisation par CPU
            cpu_percents = psutil.cpu_percent(interval=None, percpu=True)
            for i, percent in enumerate(cpu_percents):
                metrics[f"cpu_{i}_usage"] = PerformanceMetric(
                    name=f"cpu_{i}_usage",
                    value=percent,
                    timestamp=timestamp,
                    unit="%",
                    category="system"
                )
            
            # Fréquence CPU
            cpu_freq = psutil.cpu_freq()
            if cpu_freq:
                metrics["cpu_frequency"] = PerformanceMetric(
                    name="cpu_frequency",
                    value=cpu_freq.current,
                    timestamp=timestamp,
                    unit="MHz",
                    category="system",
                    metadata={"max": cpu_freq.max, "min": cpu_freq.min}
                )
            
            # Charge système (load average)
            try:
                load_avg = psutil.getloadavg()
                metrics["load_average_1m"] = PerformanceMetric(
                    name="load_average_1m",
                    value=load_avg[0],
                    timestamp=timestamp,
                    unit="",
                    category="system"
                )
            except (AttributeError, OSError):
                # getloadavg() pas disponible sur Windows
                pass
            
            # Température CPU si disponible
            if self.include_detailed:
                try:
                    temps = psutil.sensors_temperatures()
                    if temps:
                        for name, entries in temps.items():
                            if "cpu" in name.lower() or "core" in name.lower():
                                for i, entry in enumerate(entries):
                                    metrics[f"cpu_temp_{name}_{i}"] = PerformanceMetric(
                                        name=f"cpu_temp_{name}_{i}",
                                        value=entry.current,
                                        timestamp=timestamp,
                                        unit="°C",
                                        category="system"
                                    )
                except (AttributeError, OSError):
                    pass
            
        except Exception as e:
            logger.warning(f"⚠️ Erreur collecte métriques CPU: {e}")
        
        return metrics
    
    def _collect_memory_metrics(self, timestamp: float) -> Dict[str, PerformanceMetric]:
        """Collecte les métriques mémoire."""
        metrics = {}
        
        try:
            # Mémoire virtuelle (RAM)
            memory = psutil.virtual_memory()
            
            metrics["memory_usage"] = PerformanceMetric(
                name="memory_usage",
                value=memory.percent,
                timestamp=timestamp,
                unit="%",
                category="system"
            )
            
            metrics["memory_available"] = PerformanceMetric(
                name="memory_available",
                value=memory.available / (1024**3),  # GB
                timestamp=timestamp,
                unit="GB",
                category="system"
            )
            
            metrics["memory_used"] = PerformanceMetric(
                name="memory_used", 
                value=memory.used / (1024**3),  # GB
                timestamp=timestamp,
                unit="GB",
                category="system"
            )
            
            # Mémoire swap
            swap = psutil.swap_memory()
            if swap.total > 0:
                metrics["swap_usage"] = PerformanceMetric(
                    name="swap_usage",
                    value=swap.percent,
                    timestamp=timestamp,
                    unit="%",
                    category="system"
                )
            
        except Exception as e:
            logger.warning(f"⚠️ Erreur collecte métriques mémoire: {e}")
        
        return metrics
    
    def _collect_gpu_metrics(self, timestamp: float) -> Dict[str, PerformanceMetric]:
        """Collecte les métriques GPU."""
        metrics = {}
        
        if not self.include_gpu:
            return metrics
        
        try:
            # Métriques PyTorch
            if torch and torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                
                for i in range(device_count):
                    # Mémoire GPU
                    memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)  # GB
                    memory_reserved = torch.cuda.memory_reserved(i) / (1024**3)    # GB
                    
                    # Propriétés du device
                    props = torch.cuda.get_device_properties(i)
                    memory_total = props.total_memory / (1024**3)  # GB
                    
                    metrics[f"gpu_{i}_memory_allocated"] = PerformanceMetric(
                        name=f"gpu_{i}_memory_allocated",
                        value=memory_allocated,
                        timestamp=timestamp,
                        unit="GB",
                        category="gpu"
                    )
                    
                    metrics[f"gpu_{i}_memory_reserved"] = PerformanceMetric(
                        name=f"gpu_{i}_memory_reserved",
                        value=memory_reserved,
                        timestamp=timestamp,
                        unit="GB",
                        category="gpu"
                    )
                    
                    memory_usage_percent = (memory_reserved / memory_total) * 100
                    metrics[f"gpu_{i}_memory_usage"] = PerformanceMetric(
                        name=f"gpu_{i}_memory_usage",
                        value=memory_usage_percent,
                        timestamp=timestamp,
                        unit="%",
                        category="gpu"
                    )
            
            # Métriques nvidia-ml (plus détaillées)
            if NVIDIA_ML_AVAILABLE and nvidia_ml:
                device_count = nvidia_ml.nvmlDeviceGetCount()
                
                for i in range(device_count):
                    handle = nvidia_ml.nvmlDeviceGetHandleByIndex(i)
                    
                    # Utilisation GPU
                    try:
                        utilization = nvidia_ml.nvmlDeviceGetUtilizationRates(handle)
                        metrics[f"gpu_{i}_utilization"] = PerformanceMetric(
                            name=f"gpu_{i}_utilization",
                            value=utilization.gpu,
                            timestamp=timestamp,
                            unit="%",
                            category="gpu"
                        )
                        
                        metrics[f"gpu_{i}_memory_utilization"] = PerformanceMetric(
                            name=f"gpu_{i}_memory_utilization",
                            value=utilization.memory,
                            timestamp=timestamp,
                            unit="%",
                            category="gpu"
                        )
                    except Exception:
                        pass
                    
                    # Température
                    try:
                        temp = nvidia_ml.nvmlDeviceGetTemperature(
                            handle, 
                            nvidia_ml.NVML_TEMPERATURE_GPU
                        )
                        metrics[f"gpu_{i}_temperature"] = PerformanceMetric(
                            name=f"gpu_{i}_temperature",
                            value=temp,
                            timestamp=timestamp,
                            unit="°C",
                            category="gpu"
                        )
                    except Exception:
                        pass
                    
                    # Fréquence
                    try:
                        clock_graphics = nvidia_ml.nvmlDeviceGetClockInfo(
                            handle,
                            nvidia_ml.NVML_CLOCK_GRAPHICS
                        )
                        metrics[f"gpu_{i}_clock_graphics"] = PerformanceMetric(
                            name=f"gpu_{i}_clock_graphics",
                            value=clock_graphics,
                            timestamp=timestamp,
                            unit="MHz",
                            category="gpu"
                        )
                    except Exception:
                        pass
                    
                    # Consommation énergétique
                    try:
                        power = nvidia_ml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Watts
                        metrics[f"gpu_{i}_power_usage"] = PerformanceMetric(
                            name=f"gpu_{i}_power_usage",
                            value=power,
                            timestamp=timestamp,
                            unit="W",
                            category="gpu"
                        )
                    except Exception:
                        pass
                        
        except Exception as e:
            logger.warning(f"⚠️ Erreur collecte métriques GPU: {e}")
        
        return metrics
    
    def _collect_disk_metrics(self, timestamp: float) -> Dict[str, PerformanceMetric]:
        """Collecte les métriques disque."""
        metrics = {}
        
        try:
            # Utilisation des disques
            disk_usage = psutil.disk_usage('/')
            
            metrics["disk_usage"] = PerformanceMetric(
                name="disk_usage",
                value=(disk_usage.used / disk_usage.total) * 100,
                timestamp=timestamp,
                unit="%",
                category="system"
            )
            
            metrics["disk_free"] = PerformanceMetric(
                name="disk_free",
                value=disk_usage.free / (1024**3),  # GB
                timestamp=timestamp,
                unit="GB",
                category="system"
            )
            
            # I/O disque
            disk_io = psutil.disk_io_counters()
            if disk_io and self._last_disk_io and self._last_timestamp:
                time_delta = timestamp - self._last_timestamp
                
                read_rate = (disk_io.read_bytes - self._last_disk_io.read_bytes) / time_delta
                write_rate = (disk_io.write_bytes - self._last_disk_io.write_bytes) / time_delta
                
                metrics["disk_read_rate"] = PerformanceMetric(
                    name="disk_read_rate",
                    value=read_rate / (1024**2),  # MB/s
                    timestamp=timestamp,
                    unit="MB/s",
                    category="system"
                )
                
                metrics["disk_write_rate"] = PerformanceMetric(
                    name="disk_write_rate",
                    value=write_rate / (1024**2),  # MB/s
                    timestamp=timestamp,
                    unit="MB/s",
                    category="system"
                )
            
            self._last_disk_io = disk_io
            
        except Exception as e:
            logger.warning(f"⚠️ Erreur collecte métriques disque: {e}")
        
        return metrics
    
    def _collect_network_metrics(self, timestamp: float) -> Dict[str, PerformanceMetric]:
        """Collecte les métriques réseau."""
        metrics = {}
        
        try:
            # I/O réseau
            net_io = psutil.net_io_counters()
            if net_io and self._last_net_io and self._last_timestamp:
                time_delta = timestamp - self._last_timestamp
                
                bytes_sent_rate = (net_io.bytes_sent - self._last_net_io.bytes_sent) / time_delta
                bytes_recv_rate = (net_io.bytes_recv - self._last_net_io.bytes_recv) / time_delta
                
                metrics["network_sent_rate"] = PerformanceMetric(
                    name="network_sent_rate",
                    value=bytes_sent_rate / (1024**2),  # MB/s
                    timestamp=timestamp,
                    unit="MB/s",
                    category="system"
                )
                
                metrics["network_recv_rate"] = PerformanceMetric(
                    name="network_recv_rate",
                    value=bytes_recv_rate / (1024**2),  # MB/s
                    timestamp=timestamp,
                    unit="MB/s",
                    category="system"
                )
            
            self._last_net_io = net_io
            
        except Exception as e:
            logger.warning(f"⚠️ Erreur collecte métriques réseau: {e}")
        
        return metrics