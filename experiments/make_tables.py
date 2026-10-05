"""Generate manuscript/tables.tex (all numeric tables) from results/*.json."""
import json
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from camp_sim.models import PRESETS
from camp_sim.hw import H100, LINKS

RES = os.path.join(ROOT, "results")


def load(n):
    return json.load(open(os.path.join(RES, n + ".json")))


def ms(x):
    return f"{x * 1e3:,.0f}".replace(",", "\\,")


def sec(x):
    return f"{x:.2f}"


out = []


def emit(s):
    out.append(s)


# ---------------------------------------------------------------- platform parameters
emit(r"""\begin{table}[t]
\centering
\caption{Simulated platform. ``Measured'' entries are anchored on published device measurements or on our own T4 micro-benchmarks (Section~\ref{sec:validation}); all values are exposed as parameters of the simulator.}
\label{tab:platform}
\footnotesize
\resizebox{\textwidth}{!}{%
\begin{tabular}{L{3.4cm}L{4.2cm}L{6.4cm}}
\toprule
\textbf{Component} & \textbf{Value} & \textbf{Basis}\\
\midrule
GPU & H100-class: 989~TFLOP/s FP16 dense, 3.35~TB/s HBM, 80~GB & vendor data sheet; true efficiencies (GEMM 0.62, memory 0.86, attention 0.42) are hidden from CAMP\\
Host-to-GPU path & pinned DMA via root complex (``direct''); optional bounce-buffer path (``staged'') & \cite{yoon2025tract}; staged/pinned ratio $0.37$ measured on T4\\
Effective CXL$\to$GPU bandwidth & 18~GB/s (default, ``measured''); 45~GB/s (``expected'', x16 Gen5); sweep 8--64~GB/s & $\approx$10~GB/s device-level \cite{yoon2025tract}, $\approx$18~GB/s DMA to GPU \cite{jang2026itme}; PCIe Gen5 pinned copies are the upper bound\\
Per-copy fixed cost & $12~\mu$s API/driver/first-byte $+\,6~\mu$s event sync $+\,3~\mu$s per 32~MB chunk & T4: $10$--$19~\mu$s fit of copy time vs.\ size; CXL.mem load latency $\approx$0.3--0.6~$\mu$s \cite{melody2025,yoon2025tract}\\
Staging hop (staged mode) & 8~GB/s CPU copy, serial with DMA & T4 pageable-copy measurement ($4.6$ vs.\ $12.4$~GB/s)\\
Noise & per-copy log-normal $\sigma=0.06$; slow AR(1) bandwidth drift $\sigma=0.05$ ($\tau\approx2.5$~s); per-layer compute $\sigma=0.03$ & varied up to $8\times$ in Section~\ref{sec:res-robust}\\
Host sampling gap & 150~$\mu$s between decode iterations (link keeps working) & typical scheduler/sampler cost\\
\bottomrule
\end{tabular}}
\end{table}
""")

# ---------------------------------------------------------------- models
rows = []
for k in ("llama2-7b", "llama2-13b", "llama3-8b", "llama3-70b", "llama3-405b", "gemma2-9b"):
    m = PRESETS[k]
    rows.append(f"{m.name} & {m.total_params / 1e9:.2f} & {m.n_layers} & {m.d_model} & {m.n_heads}/{m.n_kv_heads} & {m.d_ff} & {m.vocab // 1000}k & "
                f"{'tied' if m.tied else 'untied'} & {m.weight_bytes / 1e9:.1f}\\\\")
emit(r"""\begin{table}[t]
\centering
\caption{Model configurations used in the evaluation (public model-card hyper-parameters; FP16 weights). Layer units are one attention block, one feed-forward block, the embedding table and the output head.}
\label{tab:models}
\small
\resizebox{\textwidth}{!}{%
\begin{tabular}{lrrrrrrlr}
\toprule
\textbf{Model} & \textbf{Params (B)} & \textbf{Layers} & $d$ & \textbf{Heads/KV} & $d_{ff}$ & \textbf{Vocab} & \textbf{Embed.} & \textbf{Weights (GB)}\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\end{table}
""")

# ---------------------------------------------------------------- main results
d = load("e2_main")


def g(res, prefix):
    for k, v in res.items():
        if k.startswith(prefix):
            return v
    raise KeyError(prefix)


rows = []
last = None
for r in d:
    res = r["results"]
    name = {"llama2-13b": "Llama-2-13B", "llama3-70b": "Llama-3-70B", "gemma2-9b (tied)": "Gemma-2-9B"}[r["model"]]
    wl = (r["workload"].replace("decode", "dec.").replace("prefill", "pre.").replace("mixed 64 dec + 512 chunk", "mixed 64+512")
          .replace(" ctx=2k", ",2k").replace("B=", "$B{=}$").replace("S=", "$S{=}$"))
    cache = f"{r['cache_ratio'] * 100:.0f}\\%"
    key = (name, r["workload"])
    c2 = res["CAMP-v2"]["mean"]
    vals = {k: g(res, k)["mean"] for k in ("Demand", "Static-k", "FlexGen", "Hot/Cold", "CAMP-v1", "CAMP-v2")}
    best_base = min(vals[k] for k in ("Demand", "Static-k", "FlexGen", "Hot/Cold", "CAMP-v1"))
    lb = max(r["lb"], r["ideal"])
    cells = []
    for k in ("Demand", "Static-k", "FlexGen", "Hot/Cold", "CAMP-v1"):
        cells.append(ms(vals[k]))
    cells.append(f"\\textbf{{{ms(c2)}}}")
    rows.append(f"{name if key[0] != last else ''} & {wl} & {cache} & {r['bw']:.0f} & " + " & ".join(cells) +
                f" & {ms(lb)} & {best_base / c2:.2f}$\\times$ & {c2 / lb:.2f}\\\\")
    last = key[0]
emit(r"""\begin{table}[t]
\centering
\caption{Iteration latency (ms, mean of 10 seeds) of CAMP-v2 and the baselines. ``Cache'' is the share of weight bytes that fit in the HBM weight cache. Speed-up is relative to the best baseline in the row; ``/LB'' is CAMP-v2 latency divided by the lower bound of Section~\ref{sec:method-lb}. Static-$k$ uses the best $k\in\{1,2,4,8\}$ per configuration. 95\% confidence half-widths are $<3\%$ of the mean for every entry (Fig.~\ref{fig:main}).}
\label{tab:main}
\scriptsize
\setlength{\tabcolsep}{3.2pt}
\resizebox{\textwidth}{!}{%
\begin{tabular}{llrrrrrrrrrrr}
\toprule
\textbf{Model} & \textbf{Workload} & \textbf{Cache} & \textbf{GB/s} & \textbf{Demand} & \textbf{Static-$k$} & \textbf{FlexGen} & \textbf{Hot/Cold} & \textbf{CAMP-v1} & \textbf{CAMP-v2} & \textbf{LB} & \textbf{Speed-up} & \textbf{/LB}\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\end{table}
""")

# ---------------------------------------------------------------- optimality gap
gp = load("e3_optimality_gap")
rows = []
for r in gp:
    rows.append(f"{r['workload'].replace('B=', '$B{=}$').replace('S=', '$S{=}$')} & {r['cache_ratio']:.1f} & {r['scp_gap_pct']:.1f} & {r['freq_gap_pct']:.1f} & {r['sim_gap_pct']:.1f}\\\\")
emit(r"""\begin{table}[t]
\centering
\caption{Optimality gap of the pin-set selection on an 18-unit model (8 layers, $d{=}2048$), where all $2^{18}$ pin sets can be enumerated. Planner gap: predicted iteration time of the selected set relative to the exhaustive optimum of the planner objective; simulated gap: the same sets replayed in the event engine.}
\label{tab:gap}
\small
\begin{tabular}{lrrrr}
\toprule
\textbf{Workload} & \textbf{Cache ratio} & \textbf{SCP gap (\%)} & \textbf{Frequency gap (\%)} & \textbf{SCP, simulated (\%)}\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---------------------------------------------------------------- eviction (uniform)
ev = load("e4_reuse_heterogeneity")["eviction_uniform"]
rows = []
for e in ("lru", "fifo", "lfu", "random", "belady"):
    a = [r for r in ev if r["evict"] == e and r["prefetch"] == "demand"][0]
    b = [r for r in ev if r["evict"] == e and r["prefetch"] == "static-2"][0]
    nm = {"lru": "LRU", "fifo": "FIFO", "lfu": "LFU", "random": "Random", "belady": "Next-use (graph-aware)"}[e]
    rows.append(f"{nm} & {ms(a['mean'])} & {a['hit_rate'] * 100:.1f} & {ms(b['mean'])} & {b['hit_rate'] * 100:.1f}\\\\")
emit(r"""\begin{table}[t]
\centering
\caption{Eviction policies on a uniform dense decode loop (Llama-2-13B, $B{=}16$, 2k context, 40\% cached, 18~GB/s, no pinning). On a cyclic access pattern with equal frequencies, LRU, FIFO and LFU are indistinguishable (zero hits); only next-use eviction with the known layer order retains useful layers.}
\label{tab:evict}
\small
\begin{tabular}{lrrrr}
\toprule
 & \multicolumn{2}{c}{\textbf{demand paging}} & \multicolumn{2}{c}{\textbf{prefetch, depth 2}}\\
\textbf{Eviction} & latency (ms) & hit (\%) & latency (ms) & hit (\%)\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---------------------------------------------------------------- heterogeneity
het = load("e4_reuse_heterogeneity")["heterogeneous"]
cols = [("demand + LRU", "LRU"), ("demand + LFU", "LFU"), ("CAMP prefetch + none pins", "none"), ("CAMP prefetch + prefix pins", "prefix"),
        ("CAMP prefetch + frequency pins", "freq."), ("CAMP prefetch + knapsack-DP pins", "DP"), ("CAMP prefetch + SCP pins", "SCP")]
rows = []
for c in het:
    nm = c["case"].replace("gemma2-9b tied decode", "Gemma-2-9B, tied embedding (per step)").replace(
        "spec. decoding (13B target + 1B draft, k=4)", "speculative decoding, 13B+1B, $k{=}4$ (per step)").replace(
        "encoder-decoder (24+24 layers, 64 output tokens)", "encoder--decoder, 64 tokens (per request)")
    rows.append(nm + " & " + " & ".join(f"{c['res'][k]['mean']:.2f}" for k, _ in cols) + "\\\\")
emit(r"""\begin{table}[t]
\centering
\caption{Workloads with non-uniform reuse (latency in seconds; 18~GB/s; 35--40\% of weights cached). The first two columns are demand paging with LRU/LFU eviction; the others use CAMP's prefetcher with the stated pinning rule.}
\label{tab:hetero}
\small
\begin{tabular}{lrrrrrrr}
\toprule
\textbf{Workload} & \textbf{LRU} & \textbf{LFU} & \textbf{none} & \textbf{prefix} & \textbf{freq.} & \textbf{DP} & \textbf{SCP}\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---------------------------------------------------------------- hardware validation
hv = load("hw_validation")
by = {}
for r in hv["rows"]:
    ratio = hv["comp_per_unit_ms"][str(r["T"])] / hv["copy_per_unit_ms"]
    if r["depth"] == 0:
        cat = "demand paging"
    elif ratio < 0.3:
        cat = "link-bound (compute $<0.3\\times$ copy)"
    elif ratio < 1.2:
        cat = "balanced ($0.3$--$1.2\\times$)"
    else:
        cat = "compute-bound ($>1.2\\times$)"
    by.setdefault(cat, []).append(r["err_pct"])
rows = []
for cat in ("link-bound (compute $<0.3\\times$ copy)", "balanced ($0.3$--$1.2\\times$)", "compute-bound ($>1.2\\times$)", "demand paging"):
    xs = by[cat]
    rows.append(f"{cat} & {len(xs)} & {statistics.mean(abs(x) for x in xs):.1f} & {statistics.mean(xs):+.1f} & {max(abs(x) for x in xs):.1f}\\\\")
allx = [r["err_pct"] for r in hv["rows"]]
rows.append(f"\\midrule all configurations & {len(allx)} & {statistics.mean(abs(x) for x in allx):.1f} & {statistics.mean(allx):+.1f} & {max(abs(x) for x in allx):.1f}\\\\")
emit(r"""\begin{table}[t]
\centering
\caption{Simulator versus a real GPU (Tesla T4, PCIe Gen3 x16, 24 weight tensors of 64~MiB streamed from pinned host memory). The simulator is composed from separately measured compute-only and copy parameters and asked to predict the measured composite iteration time; no parameter is fitted to the composites.}
\label{tab:hw}
\small
\begin{tabular}{lrrrr}
\toprule
\textbf{Regime} & \textbf{Configs} & \textbf{MAPE (\%)} & \textbf{Mean signed (\%)} & \textbf{Max $|$err$|$ (\%)}\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---------------------------------------------------------------- cost model
cm = load("e5_cost_model")["cases"]
rows = []
for c in cm:
    est = c["estimator"]
    rows.append(f"{c['name']} & {est['static']['before'] * 100:.1f} & {est['online']['after'] * 100:.1f} & {ms(c['ref']['no_pins'][0])} & {ms(c['ref']['nominal'][0])} & {ms(c['ref']['exact'][0])} & {ms(c['lb'])}\\\\")
emit(r"""\begin{table}[t]
\centering
\caption{Runtime cost model. Mean absolute percentage error (MAPE) of the per-layer compute-time estimate before and after online calibration with one probe iteration, and the resulting iteration latency (ms) with SCP pins planned from the generic roofline estimate (``generic''), from the exact device model (``exact''), and without pinning.}
\label{tab:costmodel}
\small
\begin{tabular}{lrrrrrr}
\toprule
\textbf{Workload} & \textbf{MAPE generic (\%)} & \textbf{MAPE calibrated (\%)} & \textbf{no pins} & \textbf{generic} & \textbf{exact} & \textbf{LB}\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

import re
for block in out:
    m = re.search(r"\\label\{tab:(\w+)\}", block)
    with open(os.path.join(ROOT, "manuscript", f"tab_{m.group(1)}.tex"), "w") as f:
        f.write(block)
    print("wrote manuscript/tab_%s.tex" % m.group(1))
