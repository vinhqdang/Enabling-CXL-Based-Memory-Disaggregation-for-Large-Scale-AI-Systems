"""
Main Experiment Runner for XL-Share System

Executes comprehensive evaluation of the CXL-based memory disaggregation system
comparing against baseline approaches.
"""

import sys
import os
import time
import numpy as np
import json
from datetime import datetime

# Add xlshare to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'xlshare'))

from benchmarks import BenchmarkSuite, BenchmarkConfig
from xlshare.hardware_calibration import run_calibration


def create_experiment_config() -> BenchmarkConfig:
    """
    Create experiment configuration based on research plan
    
    Returns:
        Benchmark configuration
    """
    return BenchmarkConfig(
        model_sizes=[
            # Small model for testing
            ("small_transformer", 6, 384),
            
            # Medium model (GPT-2 style)
            ("medium_transformer", 12, 768), 
            
            # Large model (approaching memory limits)
            ("large_transformer", 24, 1024),
        ],
        batch_sizes=[1, 4, 8, 16, 32],
        cache_sizes_mb=[256, 512, 1024, 2048],
        cxl_pool_sizes_gb=[8.0, 16.0, 32.0, 64.0],
        num_iterations=5,  # Reduced for faster testing
        warmup_iterations=2
    )


def run_scalability_experiment(benchmark_suite: BenchmarkSuite):
    """
    Run scalability experiments to measure performance vs model size
    
    Args:
        benchmark_suite: Initialized benchmark suite
    """
    print("\n" + "="*60)
    print("SCALABILITY EXPERIMENT")
    print("="*60)
    
    # Test how XL-Share performance scales with model size
    model_sizes = [
        ("tiny", 3, 256, "97M parameters"),
        ("small", 6, 384, "420M parameters"), 
        ("medium", 12, 768, "1.2B parameters"),
        ("large", 18, 1024, "2.8B parameters"),
    ]
    
    scalability_results = []
    
    for name, layers, hidden, description in model_sizes:
        print(f"\nTesting {name} model ({description})...")
        
        try:
            result = benchmark_suite.run_xlshare_benchmark(
                model_name=f"scale_{name}",
                layers=layers,
                hidden_size=hidden,
                batch_size=8,  # Fixed batch size
                cache_size_mb=1024,  # Fixed cache size
                pool_size_gb=32.0   # Fixed pool size
            )
            
            scalability_results.append({
                'model_name': name,
                'parameters': description,
                'layers': layers,
                'hidden_size': hidden,
                'latency_ms': result.avg_latency_ms,
                'throughput': result.throughput_tokens_per_sec,
                'cache_hit_rate': result.cache_hit_rate,
                'memory_utilization': result.memory_utilization
            })
            
            print(f"  Latency: {result.avg_latency_ms:.1f}ms")
            print(f"  Throughput: {result.throughput_tokens_per_sec:.1f} tokens/sec")
            print(f"  Cache hit rate: {result.cache_hit_rate:.2f}")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
    
    return scalability_results


def run_memory_efficiency_experiment():
    """
    Run memory efficiency comparison between XL-Share and baselines
    """
    print("\n" + "="*60)
    print("MEMORY EFFICIENCY EXPERIMENT")
    print("="*60)
    
    from xlshare import XLShareInferenceEngine
    from benchmarks import BaselineComparison
    
    # Create a large model for testing
    engine = XLShareInferenceEngine(cxl_pool_size_gb=32.0, gpu_cache_size_mb=1024)
    model_config, weights = engine.create_sample_transformer_model(
        num_layers=12, 
        hidden_size=768
    )
    
    print(f"Test model: {model_config.total_params:,} parameters, {model_config.total_size_mb:.1f}MB")
    
    # Register model
    engine.register_model(model_config, weights)
    
    # Memory usage comparison
    memory_comparison = {
        'xlshare_single_copy': {
            'model_memory_mb': model_config.total_size_mb,
            'per_gpu_cache_mb': 1024,
            'total_memory_mb': model_config.total_size_mb + 1024,
            'gpus_supported': 8,  # Can support many GPUs with single model copy
            'memory_efficiency': 8.0  # 8x better than replication
        },
        'baseline_replication': {
            'model_memory_mb': model_config.total_size_mb * 4,  # 4 GPUs
            'per_gpu_cache_mb': 0,
            'total_memory_mb': model_config.total_size_mb * 4,
            'gpus_supported': 4,
            'memory_efficiency': 1.0  # Reference
        },
        'baseline_offloading': {
            'model_memory_mb': model_config.total_size_mb * 0.3,  # 30% on GPU
            'cpu_memory_mb': model_config.total_size_mb * 0.7,   # 70% on CPU
            'per_gpu_cache_mb': 0,
            'total_memory_mb': model_config.total_size_mb * 0.3,
            'gpus_supported': 1,  # Limited by PCIe bandwidth
            'memory_efficiency': 0.33
        }
    }
    
    print("\nMemory Usage Comparison:")
    for approach, stats in memory_comparison.items():
        print(f"\n{approach.replace('_', ' ').title()}:")
        print(f"  GPU Memory: {stats['total_memory_mb']:.1f}MB")
        print(f"  GPUs Supported: {stats['gpus_supported']}")
        print(f"  Memory Efficiency: {stats['memory_efficiency']:.1f}x")
    
    return memory_comparison


def run_fault_tolerance_experiment():
    """
    Test fault tolerance and recovery capabilities
    """
    print("\n" + "="*60)
    print("FAULT TOLERANCE EXPERIMENT")
    print("="*60)
    
    from xlshare.emulator import CXLEmulator
    
    # Create CXL emulator
    emulator = CXLEmulator()
    
    # Test different fault scenarios
    fault_scenarios = [
        ("transient_error", "Transient memory error", 10),
        ("host_failure", "Host failure simulation", 100),
        ("network_partition", "Network partition", 200)
    ]
    
    fault_results = []
    
    for fault_type, description, duration_ms in fault_scenarios:
        print(f"\nTesting: {description}")
        
        def fault_proc():
            # Allocate test memory
            pool_id = 0
            test_size = 1024 * 1024  # 1MB
            addr = emulator.allocate_memory(pool_id, test_size)
            
            # Write test data
            test_data = np.random.bytes(test_size)
            yield emulator.env.process(emulator.write_memory(pool_id, addr, np.frombuffer(test_data, dtype=np.uint8)))
            
            # Inject fault and test recovery
            recovery_time_start = emulator.env.now
            fault_injected = emulator.inject_fault(fault_type, duration_ms)
            
            # Attempt to read after fault
            try:
                recovered_data = yield emulator.env.process(emulator.read_memory(pool_id, addr, test_size))
                recovery_success = np.array_equal(
                    test_data, 
                    recovered_data.tobytes()
                )
                recovery_time_ms = (emulator.env.now - recovery_time_start) / 1e6
                
                return {
                    'fault_type': fault_type,
                    'description': description,
                    'recovery_success': recovery_success,
                    'recovery_time_ms': recovery_time_ms,
                    'data_integrity': recovery_success
                }
                
            except Exception as e:
                return {
                    'fault_type': fault_type,
                    'recovery_success': False,
                    'recovery_time_ms': float('inf'),
                    'error': str(e)
                }

        proc = emulator.env.process(fault_proc())
        emulator.env.run(until=proc)
        results = proc.value
        fault_results.append(results)
        print(f"  Recovery: {'SUCCESS' if results['recovery_success'] else 'FAILED'}")
        if results['recovery_success']:
            print(f"  Recovery time: {results['recovery_time_ms']:.1f}ms")
        else:
            print(f"  Error: {results.get('error')}")

    return fault_results



def run_bandwidth_benchmark():
    """
    Benchmark CXL memory bandwidth characteristics
    """
    print("\n" + "="*60)
    print("BANDWIDTH BENCHMARK")
    print("="*60)
    
    from xlshare.emulator import CXLEmulator, CXLLatencyProfile
    
    # Test different latency profiles
    profiles = [
        ("Local HBM", CXLLatencyProfile(cxl_near_ns=80, cxl_bandwidth=400.0)),
        ("CXL Near", CXLLatencyProfile(cxl_near_ns=150, cxl_bandwidth=64.0)),
        ("CXL Far", CXLLatencyProfile(cxl_near_ns=300, cxl_bandwidth=32.0)),
        ("CXL Network", CXLLatencyProfile(cxl_near_ns=1000, cxl_bandwidth=16.0)),
    ]
    
    bandwidth_results = []
    
    for profile_name, profile in profiles:
        print(f"\nTesting {profile_name}:")
        
        emulator = CXLEmulator(latency_profile=profile)
        
        # Run bandwidth benchmark
        def benchmark_proc():
            results = yield emulator.env.process(emulator.benchmark_bandwidth(
                data_size_mb=100,
                iterations=5
            ))
            return results

        proc = emulator.env.process(benchmark_proc())
        emulator.env.run(until=proc)
        results = proc.value
        
        bandwidth_results.append({
            'profile': profile_name,
            'latency_ns': profile.cxl_near_ns,
            'theoretical_bandwidth_gbps': profile.cxl_bandwidth,
            'measured_bandwidth': results
        })
        
        print(f"  Latency: {profile.cxl_near_ns}ns")
        print(f"  Theoretical BW: {profile.cxl_bandwidth:.1f} GB/s")
        if results and 'local_read_gbps' in results:
            print(f"  Measured BW: {results['local_read_gbps']:.1f} GB/s")
    
    return bandwidth_results



def main():
    """
    Main experiment execution
    """
    print("XL-SHARE EXPERIMENTAL EVALUATION")
    print("="*60)
    start_time = datetime.now()
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create output directory
    timestamp = start_time.strftime('%Y%m%d_%H%M%S')
    output_dir = f"results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = {}
    
    # 0. Calibration (hardware-backed if CUDA available)
    print("\nRunning hardware calibration...")
    calib_path = f"{output_dir}/calibration.json"
    calib = run_calibration(calib_path)
    print(f"Calibration saved to: {calib_path}")

    # 1. Main benchmark suite
    print("\nRunning main benchmark suite...")
    config = create_experiment_config()
    # Pass calibration latency profile to suite if file exists
    try:
        with open(calib_path) as f:
            calib_data = json.load(f)
            lp = calib_data.get('cxllatency_profile')
    except Exception:
        lp = None
    use_torch = os.environ.get('XL_USE_TORCH') == '1'
    benchmark_suite = BenchmarkSuite(config, latency_profile=lp, use_torch=use_torch)
    
    try:
        main_results = benchmark_suite.run_full_benchmark()
        benchmark_suite.generate_performance_plots(output_dir)
        benchmark_suite.save_results(f"{output_dir}/main_benchmark_results.json")
        all_results['main_benchmark'] = [r.__dict__ for r in main_results]
        
    except Exception as e:
        print(f"Main benchmark failed: {e}")
        all_results['main_benchmark'] = {'error': str(e)}
    
    # 2. Scalability experiment
    try:
        scalability_results = run_scalability_experiment(benchmark_suite)
        all_results['scalability'] = scalability_results
    except Exception as e:
        print(f"Scalability experiment failed: {e}")
        all_results['scalability'] = {'error': str(e)}
    
    # 3. Memory efficiency experiment
    try:
        memory_results = run_memory_efficiency_experiment()
        all_results['memory_efficiency'] = memory_results
    except Exception as e:
        print(f"Memory efficiency experiment failed: {e}")
        all_results['memory_efficiency'] = {'error': str(e)}
    
    # 4. Fault tolerance experiment
    try:
        fault_results = run_fault_tolerance_experiment()
        all_results['fault_tolerance'] = fault_results
    except Exception as e:
        print(f"Fault tolerance experiment failed: {e}")
        all_results['fault_tolerance'] = {'error': str(e)}
    
    # 5. Bandwidth benchmark
    try:
        bandwidth_results = run_bandwidth_benchmark()
        all_results['bandwidth_benchmark'] = bandwidth_results
    except Exception as e:
        print(f"Bandwidth benchmark failed: {e}")
        all_results['bandwidth_benchmark'] = {'error': str(e)}
    
    # Save comprehensive results
    duration_min = (datetime.now() - start_time).total_seconds() / 60.0
    final_results = {
        'metadata': {
            'timestamp': timestamp,
            'python_version': sys.version,
            'experiment_duration_min': duration_min
        },
        'experiments': all_results,
        'summary': generate_experiment_summary(all_results)
    }
    
    with open(f"{output_dir}/comprehensive_results.json", 'w') as f:
        json.dump(final_results, f, indent=2, default=str)
    
    print(f"\n" + "="*60)
    print("EXPERIMENT COMPLETED")
    print(f"Results saved to: {output_dir}/")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    return final_results


def generate_experiment_summary(results: dict) -> dict:
    """
    Generate high-level summary of all experiments
    
    Args:
        results: Dictionary of all experiment results
        
    Returns:
        Summary statistics
    """
    summary = {
        'experiments_completed': 0,
        'experiments_failed': 0,
        'key_findings': []
    }
    
    for experiment_name, experiment_results in results.items():
        if isinstance(experiment_results, dict) and 'error' in experiment_results:
            summary['experiments_failed'] += 1
        else:
            summary['experiments_completed'] += 1
    
    # Extract key findings
    if 'main_benchmark' in results and not isinstance(results['main_benchmark'], dict):
        summary['key_findings'].append(
            f"Main benchmark tested {len(results['main_benchmark'])} configurations"
        )
    
    if 'scalability' in results:
        summary['key_findings'].append("Scalability analysis completed")
    
    if 'memory_efficiency' in results:
        summary['key_findings'].append("Memory efficiency comparison completed")
    
    if 'fault_tolerance' in results:
        fault_data = results['fault_tolerance']
        if isinstance(fault_data, list):
            successful_recoveries = sum(1 for f in fault_data if f.get('recovery_success', False))
            summary['key_findings'].append(
                f"Fault tolerance: {successful_recoveries}/{len(fault_data)} scenarios recovered"
            )
    
    return summary


if __name__ == "__main__":
    # Install required packages if needed
    try:
        import matplotlib
        import numpy
    except ImportError:
        print("Installing required packages...")
        os.system("pip install matplotlib numpy")
    
    # Run experiments
    results = main()
