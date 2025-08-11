"""
Evaluation Benchmarks for XL-Share System

Comprehensive benchmarking suite comparing XL-Share against baseline approaches
for large-scale AI model inference and training.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import json
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict

from xlshare import XLShareInferenceEngine, InferenceRequest
from xlshare.memory_manager import CXLMemoryManager, LocalCache
from xlshare.emulator import CXLEmulator


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark experiments"""
    model_sizes: List[Tuple[str, int, int]]  # (name, layers, hidden_size)
    batch_sizes: List[int]
    cache_sizes_mb: List[int]
    cxl_pool_sizes_gb: List[float]
    num_iterations: int = 10
    warmup_iterations: int = 3


@dataclass
class BenchmarkResult:
    """Results from a benchmark experiment"""
    config_name: str
    model_name: str
    batch_size: int
    cache_size_mb: int
    pool_size_gb: float
    
    # Performance metrics
    avg_latency_ms: float
    p99_latency_ms: float
    throughput_tokens_per_sec: float
    
    # Memory metrics
    cache_hit_rate: float
    memory_utilization: float
    prefetch_efficiency: float
    
    # System metrics
    gpu_utilization: float
    memory_bandwidth_gbps: float
    
    # Raw measurements
    latency_samples: List[float]
    # Additional stats
    warm_latency_ms: float = 0.0
    median_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0


class BaselineComparison:
    """
    Implements baseline comparison systems for evaluation:
    1. No disaggregation (full model replication)
    2. CPU offloading (ZeRO-style)
    3. Model parallelism simulation
    """
    
    def __init__(self):
        self.stats = {
            'baseline_local': {'latencies': [], 'memory_usage': 0},
            'baseline_offload': {'latencies': [], 'memory_usage': 0, 'transfer_overhead': 0},
            'baseline_parallel': {'latencies': [], 'memory_usage': 0, 'communication_overhead': 0}
        }
    
    def run_baseline_local(self, model_config, requests: List[InferenceRequest]) -> List[float]:
        """
        Simulate baseline with full model replication on each GPU
        
        Args:
            model_config: Model configuration
            requests: List of inference requests
            
        Returns:
            List of latency measurements
        """
        latencies = []
        
        # Assume model fully fits in GPU memory (best case)
        local_memory_latency_ns = 80  # HBM access latency
        
        for request in requests:
            start_time = time.time()
            
            # Simulate layer execution with local memory access
            for layer in model_config.layers:
                # Local memory access (very fast)
                time.sleep(local_memory_latency_ns / 1e9)
                
                # Layer computation
                time.sleep(layer.computation_time_ms / 1000)
            
            latency_ms = (time.time() - start_time) * 1000
            latencies.append(latency_ms)
        
        # Calculate memory usage (full model replication)
        self.stats['baseline_local']['memory_usage'] = model_config.total_size_mb
        self.stats['baseline_local']['latencies'] = latencies
        
        return latencies
    
    def run_baseline_offload(self, model_config, requests: List[InferenceRequest]) -> List[float]:
        """
        Simulate CPU offloading baseline (ZeRO-style)
        
        Args:
            model_config: Model configuration
            requests: List of inference requests
            
        Returns:
            List of latency measurements
        """
        latencies = []
        
        # PCIe transfer characteristics
        pcie_bandwidth_gbps = 16.0  # PCIe 4.0 x16
        cpu_memory_latency_ns = 200  # CPU DRAM latency
        
        for request in requests:
            start_time = time.time()
            transfer_overhead = 0
            
            for layer in model_config.layers:
                # Transfer weights from CPU to GPU
                transfer_time = (layer.weight_size_bytes / (1024**3)) / pcie_bandwidth_gbps
                time.sleep(transfer_time)
                transfer_overhead += transfer_time
                
                # Layer computation
                time.sleep(layer.computation_time_ms / 1000)
                
                # Transfer results back (smaller)
                result_size = layer.weight_size_bytes * 0.01  # Assume 1% result size
                result_transfer_time = (result_size / (1024**3)) / pcie_bandwidth_gbps
                time.sleep(result_transfer_time)
                transfer_overhead += result_transfer_time
            
            latency_ms = (time.time() - start_time) * 1000
            latencies.append(latency_ms)
        
        # Calculate metrics
        avg_transfer_overhead = np.mean([transfer_overhead]) if latencies else 0
        self.stats['baseline_offload']['memory_usage'] = model_config.total_size_mb * 0.3  # Reduced GPU usage
        self.stats['baseline_offload']['latencies'] = latencies
        self.stats['baseline_offload']['transfer_overhead'] = avg_transfer_overhead
        
        return latencies
    
    def run_baseline_parallel(self, model_config, requests: List[InferenceRequest], 
                            num_gpus: int = 4) -> List[float]:
        """
        Simulate model parallelism baseline
        
        Args:
            model_config: Model configuration
            requests: List of inference requests
            num_gpus: Number of GPUs for parallelism
            
        Returns:
            List of latency measurements
        """
        latencies = []
        
        # Inter-GPU communication characteristics
        nvlink_bandwidth_gbps = 300.0  # NVLink bandwidth
        layers_per_gpu = len(model_config.layers) // num_gpus
        
        for request in requests:
            start_time = time.time()
            communication_overhead = 0
            
            # Execute layers across GPUs
            for gpu_id in range(num_gpus):
                gpu_layers = model_config.layers[gpu_id * layers_per_gpu:(gpu_id + 1) * layers_per_gpu]
                
                for layer in gpu_layers:
                    # Local execution on assigned GPU
                    time.sleep(layer.computation_time_ms / 1000)
                
                # Communication between GPUs (activations transfer)
                if gpu_id < num_gpus - 1:
                    activation_size = 512 * 1024 * 4  # Assume 512K float32 activations
                    comm_time = (activation_size / (1024**3)) / nvlink_bandwidth_gbps
                    time.sleep(comm_time)
                    communication_overhead += comm_time
            
            latency_ms = (time.time() - start_time) * 1000
            latencies.append(latency_ms)
        
        # Calculate metrics
        avg_comm_overhead = np.mean([communication_overhead]) if latencies else 0
        self.stats['baseline_parallel']['memory_usage'] = model_config.total_size_mb / num_gpus
        self.stats['baseline_parallel']['latencies'] = latencies
        self.stats['baseline_parallel']['communication_overhead'] = avg_comm_overhead
        
        return latencies


class BenchmarkSuite:
    """
    Comprehensive benchmark suite for evaluating XL-Share system
    """
    
    def __init__(self, config: BenchmarkConfig, latency_profile: Dict[str, Any] | None = None, use_torch: bool = False):
        """
        Initialize benchmark suite
        
        Args:
            config: Benchmark configuration
        """
        self.config = config
        self.results: List[BenchmarkResult] = []
        self.baseline_comparison = BaselineComparison()
        self.latency_profile = latency_profile
        self.use_torch = use_torch
        
        print("Benchmark Suite initialized")
        print(f"Model sizes: {[f'{name}({layers}L,{hidden}H)' for name, layers, hidden in config.model_sizes]}")
        print(f"Batch sizes: {config.batch_sizes}")
        print(f"Cache sizes: {config.cache_sizes_mb}MB")
        print(f"Pool sizes: {config.cxl_pool_sizes_gb}GB")
    
    def run_xlshare_benchmark(self, model_name: str, layers: int, hidden_size: int,
                             batch_size: int, cache_size_mb: int, 
                             pool_size_gb: float) -> BenchmarkResult:
        """
        Run XL-Share benchmark for specific configuration
        
        Args:
            model_name: Name of model variant
            layers: Number of model layers
            hidden_size: Hidden dimension size
            batch_size: Batch size for inference
            cache_size_mb: Local cache size in MB
            pool_size_gb: CXL pool size in GB
            
        Returns:
            Benchmark result
        """
        print(f"  Running XL-Share: {model_name}, batch={batch_size}, "
              f"cache={cache_size_mb}MB, pool={pool_size_gb}GB")
        
        # Initialize XL-Share system
        engine = XLShareInferenceEngine(
            cxl_pool_size_gb=pool_size_gb,
            gpu_cache_size_mb=cache_size_mb,
            emulate_cxl=True,
            latency_profile=self.latency_profile
        )
        
        # Create and register model
        model_config, weights = engine.create_sample_transformer_model(
            num_layers=layers,
            hidden_size=hidden_size
        )
        model_config.name = model_name
        
        success = engine.register_model(model_config, weights)
        if not success:
            raise RuntimeError(f"Failed to register model {model_name}")
        
        # Create inference requests
        requests = []
        for i in range(self.config.num_iterations + self.config.warmup_iterations):
            input_shape = [batch_size, 512]  # 512 token sequence
            input_data = np.random.randn(*input_shape).astype(np.float32)
            
            request = InferenceRequest(
                request_id=f"bench_{i}",
                input_data=input_data,
                model_name=model_name,
                timestamp=time.time()
            )
            requests.append(request)
        
        # Warmup
        print("    Warming up...")
        for i in range(self.config.warmup_iterations):
            engine.inference(requests[i])
        
        # Actual benchmark
        print("    Running benchmark...")
        latencies = []
        throughputs = []
        
        for i in range(self.config.warmup_iterations, len(requests)):
            result = engine.inference(requests[i])
            latencies.append(result.latency_ms)
            throughputs.append(result.throughput_tokens_per_sec)
        
        # Collect system statistics
        system_stats = engine.get_system_stats()
        cache_hit_rate = system_stats['local_cache']['hit_rate']
        memory_utilization = system_stats['memory_manager']['pool_utilization']
        prefetch_efficiency = system_stats['prefetcher']['efficiency']
        
        # Calculate performance metrics
        avg_latency = np.mean(latencies)
        p99_latency = np.percentile(latencies, 99)
        avg_throughput = np.mean(throughputs)
        warm_only = latencies[1:] if len(latencies) > 1 else latencies
        warm_latency = float(np.mean(warm_only)) if warm_only else float(avg_latency)
        median_latency = float(np.median(latencies)) if len(latencies) else 0.0
        p95_latency = float(np.percentile(latencies, 95)) if len(latencies) else 0.0
        
        # Estimate GPU utilization and memory bandwidth
        gpu_utilization = 0.85  # Simulated value
        memory_bandwidth = system_stats.get('cxl_emulator', {}).get('avg_bandwidth_gbps', 0)
        
        return BenchmarkResult(
            config_name=f"xlshare_{cache_size_mb}mb_{pool_size_gb}gb",
            model_name=model_name,
            batch_size=batch_size,
            cache_size_mb=cache_size_mb,
            pool_size_gb=pool_size_gb,
            avg_latency_ms=avg_latency,
            throughput_tokens_per_sec=avg_throughput,
            cache_hit_rate=cache_hit_rate,
            memory_utilization=memory_utilization,
            prefetch_efficiency=prefetch_efficiency,
            gpu_utilization=gpu_utilization,
            memory_bandwidth_gbps=memory_bandwidth,
            latency_samples=latencies,
            warm_latency_ms=warm_latency,
            median_latency_ms=median_latency,
            p95_latency_ms=p95_latency,
            p99_latency_ms=p99_latency
        )
    
    def run_baseline_benchmarks(self, model_name: str, layers: int, hidden_size: int,
                               batch_size: int) -> Dict[str, List[float]]:
        """
        Run baseline comparison benchmarks
        
        Args:
            model_name: Name of model variant
            layers: Number of model layers
            hidden_size: Hidden dimension size
            batch_size: Batch size for inference
            
        Returns:
            Dictionary of baseline results
        """
        print(f"  Running baselines for {model_name}, batch={batch_size}")
        
        # Create model configuration
        engine = XLShareInferenceEngine(cxl_pool_size_gb=1.0, gpu_cache_size_mb=512)
        model_config, _ = engine.create_sample_transformer_model(layers, hidden_size)
        model_config.name = model_name
        
        # Create inference requests
        requests = []
        for i in range(self.config.num_iterations):
            input_shape = [batch_size, 512]
            input_data = np.random.randn(*input_shape).astype(np.float32)
            
            request = InferenceRequest(
                request_id=f"baseline_{i}",
                input_data=input_data,
                model_name=model_name,
                timestamp=time.time()
            )
            requests.append(request)
        
        # Run baseline comparisons
        baseline_results = {}
        
        baseline_results['local'] = self.baseline_comparison.run_baseline_local(
            model_config, requests
        )
        
        baseline_results['offload'] = self.baseline_comparison.run_baseline_offload(
            model_config, requests
        )
        
        baseline_results['parallel'] = self.baseline_comparison.run_baseline_parallel(
            model_config, requests, num_gpus=4
        )
        
        return baseline_results
    
    def run_full_benchmark(self) -> List[BenchmarkResult]:
        """
        Run complete benchmark suite
        
        Returns:
            List of all benchmark results
        """
        print("Starting full benchmark suite...")
        
        all_results = []
        baseline_results_cache = {}
        
        for model_name, layers, hidden_size in self.config.model_sizes:
            print(f"\nBenchmarking model: {model_name} ({layers} layers, {hidden_size} hidden)")
            
            for batch_size in self.config.batch_sizes:
                print(f"\n  Batch size: {batch_size}")
                
                # Run baseline benchmarks (once per model-batch combination)
                baseline_key = f"{model_name}_{batch_size}"
                if baseline_key not in baseline_results_cache:
                    baseline_results_cache[baseline_key] = self.run_baseline_benchmarks(
                        model_name, layers, hidden_size, batch_size
                    )
                
                # Run XL-Share configurations
                for cache_size_mb in self.config.cache_sizes_mb:
                    for pool_size_gb in self.config.cxl_pool_sizes_gb:
                        try:
                            result = self.run_xlshare_benchmark(
                                model_name, layers, hidden_size,
                                batch_size, cache_size_mb, pool_size_gb
                            )
                            all_results.append(result)
                            self.results.append(result)
                            
                        except Exception as e:
                            print(f"    ERROR: {e}")
                            continue
        
        # Store baseline results for comparison
        self.baseline_results = baseline_results_cache
        
        print(f"\nBenchmark suite completed: {len(all_results)} configurations tested")
        return all_results
    
    def generate_performance_plots(self, output_dir: str = "."):
        """
        Generate performance comparison plots
        
        Args:
            output_dir: Directory to save plots
        """
        if not self.results:
            print("No results to plot")
            return
        
        # Group results by model
        model_results = {}
        for result in self.results:
            if result.model_name not in model_results:
                model_results[result.model_name] = []
            model_results[result.model_name].append(result)
        
        # Plot 1: Latency vs Batch Size
        plt.figure(figsize=(12, 8))
        
        for model_name, results in model_results.items():
            if len(results) < 2:
                continue
            
            batch_sizes = [r.batch_size for r in results]
            latencies = [r.avg_latency_ms for r in results]
            
            plt.subplot(2, 2, 1)
            plt.plot(batch_sizes, latencies, 'o-', label=f'XL-Share {model_name}')
            
            # Add baseline comparison if available
            baseline_key = f"{model_name}_{batch_sizes[0]}"
            if baseline_key in self.baseline_results:
                baselines = self.baseline_results[baseline_key]
                plt.axhline(np.mean(baselines['local']), linestyle='--', alpha=0.7, 
                           label=f'Local {model_name}')
                plt.axhline(np.mean(baselines['offload']), linestyle=':', alpha=0.7,
                           label=f'Offload {model_name}')
        
        plt.xlabel('Batch Size')
        plt.ylabel('Average Latency (ms)')
        plt.title('Inference Latency vs Batch Size')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Throughput Comparison
        plt.subplot(2, 2, 2)
        for model_name, results in model_results.items():
            if len(results) < 2:
                continue
            
            batch_sizes = [r.batch_size for r in results]
            throughputs = [r.throughput_tokens_per_sec for r in results]
            
            plt.plot(batch_sizes, throughputs, 's-', label=f'XL-Share {model_name}')
        
        plt.xlabel('Batch Size')
        plt.ylabel('Throughput (tokens/sec)')
        plt.title('Inference Throughput vs Batch Size')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Cache Hit Rate vs Cache Size
        plt.subplot(2, 2, 3)
        cache_sizes = list(set(r.cache_size_mb for r in self.results))
        cache_hit_rates = {}
        
        for cache_size in cache_sizes:
            cache_results = [r for r in self.results if r.cache_size_mb == cache_size]
            avg_hit_rate = np.mean([r.cache_hit_rate for r in cache_results])
            cache_hit_rates[cache_size] = avg_hit_rate
        
        plt.plot(list(cache_hit_rates.keys()), list(cache_hit_rates.values()), 'D-')
        plt.xlabel('Cache Size (MB)')
        plt.ylabel('Cache Hit Rate')
        plt.title('Cache Performance vs Cache Size')
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Memory Utilization
        plt.subplot(2, 2, 4)
        for model_name, results in model_results.items():
            if len(results) < 2:
                continue
            
            pool_sizes = [r.pool_size_gb for r in results]
            memory_utils = [r.memory_utilization for r in results]
            
            plt.plot(pool_sizes, memory_utils, '^-', label=f'{model_name}')
        
        plt.xlabel('CXL Pool Size (GB)')
        plt.ylabel('Memory Utilization')
        plt.title('Memory Utilization vs Pool Size')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/xlshare_performance_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Performance plots saved to {output_dir}/xlshare_performance_analysis.png")
    
    def save_results(self, filename: str = "benchmark_results.json"):
        """
        Save benchmark results to JSON file
        
        Args:
            filename: Output filename
        """
        results_data = {
            'config': asdict(self.config),
            'xlshare_results': [asdict(result) for result in self.results],
            'baseline_stats': self.baseline_comparison.stats,
            'summary': self.get_summary_statistics()
        }
        
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        print(f"Results saved to {filename}")
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Calculate summary statistics across all benchmark results
        
        Returns:
            Dictionary of summary statistics
        """
        if not self.results:
            return {}
        
        # XL-Share performance
        xlshare_latencies = [r.avg_latency_ms for r in self.results]
        xlshare_throughputs = [r.throughput_tokens_per_sec for r in self.results]
        xlshare_hit_rates = [r.cache_hit_rate for r in self.results]
        xlshare_warm = [r.warm_latency_ms for r in self.results]
        
        # Best performing configuration
        best_throughput_idx = np.argmax(xlshare_throughputs)
        best_config = self.results[best_throughput_idx]
        
        summary = {
            'xlshare_performance': {
                'avg_latency_ms': float(np.mean(xlshare_latencies)),
                'min_latency_ms': float(np.min(xlshare_latencies)),
                'avg_throughput_tokens_per_sec': float(np.mean(xlshare_throughputs)),
                'max_throughput_tokens_per_sec': float(np.max(xlshare_throughputs)),
                'avg_cache_hit_rate': float(np.mean(xlshare_hit_rates)),
                'avg_warm_latency_ms': float(np.mean(xlshare_warm)) if xlshare_warm else 0.0,
            },
            'best_configuration': {
                'model_name': best_config.model_name,
                'batch_size': best_config.batch_size,
                'cache_size_mb': best_config.cache_size_mb,
                'pool_size_gb': best_config.pool_size_gb,
                'throughput_tokens_per_sec': best_config.throughput_tokens_per_sec,
                'latency_ms': best_config.avg_latency_ms
            },
            'configurations_tested': len(self.results),
            'models_tested': len(set(r.model_name for r in self.results))
        }
        
        return summary
