
import json
import matplotlib.pyplot as plt
import numpy as np
import os

# Configuration
NUM_DIR = "numerical_results"
FIG_DIR = "figs"
os.makedirs(FIG_DIR, exist_ok=True)

def plot_comprehensive_results():
    print("Loading results from numerical_results/comprehensive_scenarios.json...")
    with open(f"{NUM_DIR}/comprehensive_scenarios.json", "r") as f:
        data = json.load(f)
    
    scenarios = ["thinking", "streaming", "thrashing"]
    scenario_labels = ["Thinking (Generative)", "Streaming (Throughput)", "Thrashing (Constrained)"]
    
    modes = ["no_prefetch", "static", "tmo", "melody", "limoncello", "camp", "expand"]
    mode_labels = ["No Prefetch", "Static (N=2)", "TMO '22", "Melody '25", "Limoncello '24", "CAMP (Ours)", "Oracle"]
    colors = ['#9E9E9E', '#FFC107', '#2196F3', '#03A9F4', '#FF5722', '#4CAF50', '#673AB7']
    
    # 1. Plot Latency (Grouped Bar Chart)
    print("Generating comprehensive_latency.png...")
    plt.figure(figsize=(14, 8))
    
    bar_width = 0.12
    x = np.arange(len(scenarios))
    
    for i, mode in enumerate(modes):
        latencies = []
        for sc in scenarios:
            # Find the result for this mode in this scenario
            res = next((r for r in data[sc] if r["mode"] == mode), None)
            if res:
                latencies.append(res["latency"])
            else:
                latencies.append(0)
        
        plt.bar(x + i*bar_width, latencies, width=bar_width, label=mode_labels[i], color=colors[i], edgecolor='black', alpha=0.9)

    plt.xlabel("Benchmark Scenario", fontsize=14, fontweight='bold')
    plt.ylabel("Inference Latency (ms) [Lower is Better]", fontsize=14, fontweight='bold')
    plt.title("Comprehensive 7-Way Comparison: Latency Analysis", fontsize=16, fontweight='bold')
    plt.xticks(x + bar_width * 3, scenario_labels, fontsize=12)
    plt.legend(ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.1), fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/comprehensive_latency.png", dpi=300)
    plt.close()

    # 2. Plot Hit Rate (Deep Analysis)
    print("Generating deep_telemetry_hit_rate.png...")
    plt.figure(figsize=(14, 8))
    
    for i, mode in enumerate(modes):
        rates = []
        for sc in scenarios:
            res = next((r for r in data[sc] if r["mode"] == mode), None)
            rates.append(res.get("hit_rate_pct", 0) if res else 0)
            
        plt.bar(x + i*bar_width, rates, width=bar_width, label=mode_labels[i], color=colors[i], edgecolor='black', alpha=0.9)

    plt.xlabel("Benchmark Scenario", fontsize=14, fontweight='bold')
    plt.ylabel("Cache Hit Rate (%) [Higher is Better]", fontsize=14, fontweight='bold')
    plt.title("Micro-Architecture Analysis: Cache Hit Rate", fontsize=16, fontweight='bold')
    plt.xticks(x + bar_width * 3, scenario_labels, fontsize=12)
    plt.ylim(0, 105)
    plt.legend(ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.1), fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Annotate CAMP vs Limoncello in Thrashing
    # x[2] is Thrashing. CAMP is index 5. Limoncello is index 4.
    camp_thrashing_x = x[2] + 5*bar_width
    limo_thrashing_x = x[2] + 4*bar_width
    
    # Add annotation if data exists
    try:
        camp_res = next(r for r in data["thrashing"] if r["mode"] == "camp")
        limo_res = next(r for r in data["thrashing"] if r["mode"] == "limoncello")
        
        plt.annotate(f"CAMP: {camp_res['hit_rate_pct']:.1f}%", 
                     xy=(camp_thrashing_x, camp_res['hit_rate_pct']), 
                     xytext=(camp_thrashing_x, camp_res['hit_rate_pct']+10),
                     arrowprops=dict(facecolor='black', shrink=0.05),
                     ha='center', fontsize=10, fontweight='bold')
                     
        plt.annotate(f"Limoncello: {limo_res['hit_rate_pct']:.1f}%", 
                     xy=(limo_thrashing_x, limo_res['hit_rate_pct']), 
                     xytext=(limo_thrashing_x, limo_res['hit_rate_pct']+20),
                     arrowprops=dict(facecolor='red', shrink=0.05),
                     ha='center', fontsize=10, color='red')
    except:
        pass

    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/deep_telemetry_hit_rate.png", dpi=300)
    plt.close()

    # 3. Plot Stalls (Deep Analysis)
    print("Generating deep_telemetry_stalls.png...")
    plt.figure(figsize=(14, 8))
    
    for i, mode in enumerate(modes):
        stalls = []
        for sc in scenarios:
            res = next((r for r in data[sc] if r["mode"] == mode), None)
            stalls.append(res.get("stall_count", 0) if res else 0)
            
        plt.bar(x + i*bar_width, stalls, width=bar_width, label=mode_labels[i], color=colors[i], edgecolor='black', alpha=0.9)

    plt.xlabel("Benchmark Scenario", fontsize=14, fontweight='bold')
    plt.ylabel("Stall Cycles (Count) [Lower is Better]", fontsize=14, fontweight='bold')
    plt.title("Micro-Architecture Analysis: Pipeline Stalls", fontsize=16, fontweight='bold')
    plt.xticks(x + bar_width * 3, scenario_labels, fontsize=12)
    plt.legend(ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.1), fontsize=11)
    
    # Annotate Thinking Phase (CAMP vs TMO)
    # x[0] is Thinking. CAMP is 5. TMO is 2.
    try:
        camp_think = next(r for r in data["thinking"] if r["mode"] == "camp")
        tmo_think = next(r for r in data["thinking"] if r["mode"] == "tmo")
        text_x = x[0] + 3.5*bar_width
        plt.text(text_x, 5, f"CAMP Stalls: {camp_think['stall_count']}\nTMO Stalls: {tmo_think.get('stall_count', '?')}", 
                 ha='center', bbox=dict(facecolor='white', alpha=0.8))
    except:
        pass
        
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/deep_telemetry_stalls.png", dpi=300)
    plt.close()
    
    print("Done! Figures saved to figs/")

if __name__ == "__main__":
    plot_comprehensive_results()
