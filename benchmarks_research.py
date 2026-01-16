"""
Research-Grade Benchmarking Suite for XL-Share.
Performs sensitivity analysis and model scaling experiments.
"""

import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any

from xlshare import XLShareInferenceEngine, InferenceRequest, ModelConfig
from xlshare.emulator import CXLLatencyProfile

FIG_DIR = "figs"
NUM_DIR = "numerical_results"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(NUM_DIR, exist_ok=True)

def run_sensitivity_analysis():
    """
    Analyze impact of CXL Latency and Bandwidth on Inference Latency.
    Using a synthetic LLaMA-7B-like model structure.
    """
    print("--- Starting Sensitivity Analysis ---")
    
    # Define ranges
    latencies = [100, 200, 300, 500, 1000] # ns
    bandwidths = [32.0, 64.0, 128.0] # GB/s
    
    results = {
        "metadata": {"model": "Synthetic-LLaMA-7B-Layer"},
        "data": []
    }
    
    # We use a synthetic model that mimics LLaMA-7B layer sizes
    # 7B Params: ~14GB (FP16). 32 Layers. ~440MB per layer.
    # Cache size 2GB (very constrained).
    
    for bw in bandwidths:
        lat_curve = []
        for lat in latencies:
            print(f"Testing BW={bw}GB/s, Lat={lat}ns...")
            
            profile = {
                "cxl_near_ns": lat,
                "cxl_far_ns": lat * 1.5,
                "cxl_bandwidth": bw,
                "local_bandwidth": 400.0,
                "coherence_overhead_ns": 50
            }
            
            engine = XLShareInferenceEngine(
                cxl_pool_size_gb=32.0,
                gpu_cache_size_mb=2048, # 2GB Cache
                emulate_cxl=True,
                latency_profile=profile
            )
            
            # Enable advanced features
            engine.local_cache.set_eviction_policy("graph_aware")
            
            # Create synthetic LLaMA-like model
            # 32 Layers of ~400MB each
            model_config, weights = engine.create_sample_transformer_model(
                num_layers=10, # Reduced to 10 for speed in prototype
                hidden_size=4096, # LLaMA size
                vocab_size=32000
            )
            engine.register_model(model_config, weights)
            
            # Run inference
            # Warmup
            req = InferenceRequest("warmup", np.random.randn(1, 128), model_config.name, time.time())
            engine.env.run(until=engine.inference(req))
            
            # Measure
            latencies_ms = []
            for i in range(5):
                req = InferenceRequest(f"req_{i}", np.random.randn(1, 128), model_config.name, 0)
                process = engine.inference(req)
                engine.env.run(until=process)
                res = process.value
                latencies_ms.append(res.latency_ms)
                
            avg_lat = np.mean(latencies_ms)
            lat_curve.append(avg_lat)
            
            results["data"].append({
                "bandwidth": bw,
                "latency": lat,
                "avg_inference_ms": avg_lat,
                "throughput_tok_sec": 128 / (avg_lat / 1000)
            })
            
            print(f"  Result: {avg_lat:.2f}ms")

    with open(f"{RESULT_DIR}/sensitivity.json", "w") as f:
        json.dump(results, f, indent=2)
    
    plot_sensitivity(results)

    with open(f"{NUM_DIR}/sensitivity.json", "w") as f:
        json.dump(results, f, indent=2)
    
    plot_sensitivity(results)

def plot_sensitivity(results):
    data = results["data"]
    bandwidths = sorted(list(set(d["bandwidth"] for d in data)))
    
    plt.figure(figsize=(10, 6))
    
    for bw in bandwidths:
        subset = [d for d in data if d["bandwidth"] == bw]
        subset.sort(key=lambda x: x["latency"])
        
        x = [d["latency"] for d in subset]
        y = [d["avg_inference_ms"] for d in subset]
        
        plt.plot(x, y, marker='o', label=f"BW {bw} GB/s")
        
    plt.xlabel("CXL Latency (ns)")
    plt.ylabel("Inference Latency (ms)")
    plt.title("Impact of CXL Characteristics on Inference Latency")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{FIG_DIR}/sensitivity_plot.png")
    print(f"Plot saved to {FIG_DIR}/sensitivity_plot.png")

def run_scaling_analysis():
    """
    Analyze performance scaling with Model Size (Layers).
    """
    print("--- Starting Scaling Analysis ---")
    layer_counts = [4, 8, 12, 16] # 24 layers is full GPT-2 XL approx
    
    results = {
        "metadata": {"hidden_size": 2048},
        "data": []
    }
    
    # Use fixed characteristics
    profile = {
        "cxl_near_ns": 300,
        "cxl_far_ns": 450,
        "cxl_bandwidth": 64.0,
        "local_bandwidth": 400.0,
        "coherence_overhead_ns": 50
    }
    
    # Reuse engine to save alloc time?
    # Actually need to re-alloc for different model sizes anyway.
    # But we can use smaller hidden size for speed.
    
    for layers in layer_counts:
        print(f"Testing Layers={layers}...")
        
        engine = XLShareInferenceEngine(
            cxl_pool_size_gb=32.0,
            gpu_cache_size_mb=2048,
            emulate_cxl=True,
            latency_profile=profile
        )
        engine.local_cache.set_eviction_policy("graph_aware")
        
        # Create model (smaller hidden size for speed)
        model_config, weights = engine.create_sample_transformer_model(
            num_layers=layers,
            hidden_size=2048, 
            vocab_size=32000
        )
        engine.register_model(model_config, weights)
        
        # Measure
        req = InferenceRequest("warmup", np.random.randn(1, 128), model_config.name, time.time())
        engine.env.run(until=engine.inference(req))
        
        latencies = []
        for i in range(3): # Fewer iters
            req = InferenceRequest(f"req_{i}", np.random.randn(1, 128), model_config.name, 0)
            process = engine.inference(req)
            engine.env.run(until=process)
            latencies.append(process.value.latency_ms)
            
        avg_lat = np.mean(latencies)
        results["data"].append({
            "layers": int(layers),
            "avg_inference_ms": float(avg_lat),
            "params_b": int(model_config.total_params * 4)
        })
        print(f"  Result: {avg_lat:.2f}ms")
        
    with open(f"{NUM_DIR}/scaling.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    # Optimize Sensitivity: Reduce hidden size slightly
    # run_sensitivity_analysis() # Commented out to focus on scaling for now or run both
    # We will redefine run_sensitivity_analysis to be faster too
    pass

def run_sensitivity_fast():
    print("--- Starting Sensitivity Analysis (Fast) ---")
    latencies = [100, 300, 600]
    bandwidths = [32.0, 64.0]
    
    results = {"data":[]}
    
    # Helper to update engine without reload
    engine = XLShareInferenceEngine(
        cxl_pool_size_gb=32.0,
        gpu_cache_size_mb=2048,
        emulate_cxl=True,
        latency_profile={"cxl_near_ns": 300, "cxl_bandwidth": 64.0}
    )
    engine.local_cache.set_eviction_policy("graph_aware")
    
    # Load model ONCE
    model_config, weights = engine.create_sample_transformer_model(
        num_layers=8,
        hidden_size=2048,
        vocab_size=32000
    )
    engine.register_model(model_config, weights)
    
    for bw in bandwidths:
        for lat in latencies:
            print(f"Testing BW={bw}, Lat={lat}...")
            # Update emulator stats in place
            engine.cxl_emulator.latency_profile.cxl_bandwidth = bw
            engine.cxl_emulator.latency_profile.cxl_near_ns = lat
            engine.memory_manager.latency_ns = lat
            
            # Run
            lat_sum = 0
            for i in range(3):
                req = InferenceRequest(f"req_{bw}_{lat}_{i}", np.random.randn(1, 128), model_config.name, i) # timestamp monotonic
                process = engine.inference(req)
                engine.env.run(until=process)
                lat_sum += process.value.latency_ms
            
            avg = lat_sum / 3
            results["data"].append({
                "bandwidth": float(bw), 
                "latency": int(lat), 
                "avg_inference_ms": float(avg)
            })
            print(f"  -> {avg:.2f}ms")
            
    with open(f"{NUM_DIR}/sensitivity.json", "w") as f:
        json.dump(results, f, indent=2)
    plot_sensitivity(results)

if __name__ == "__main__":
    run_sensitivity_fast()
    run_scaling_analysis()
