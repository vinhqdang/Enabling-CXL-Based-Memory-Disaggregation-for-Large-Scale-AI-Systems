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
Per-copy fixed cost & $12~\mu$s API/driver/first-byte $+\,6~\mu$s event sync $+\,3~\mu$s per 32~MB chunk ($\approx$21~$\mu$s for a 64~MiB copy) & T4: fitted intercept of copy time vs.\ size is $10.6~\mu$s (the extra simulator terms are $<0.1\%$ of a 64~MiB copy); CXL.mem load latency $\approx$0.3--0.6~$\mu$s \cite{melody2025,yoon2025tract}\\
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


NAMES = {"llama2-13b": "Llama-2-13B", "llama3-70b": "Llama-3-70B", "llama3-405b": "Llama-3-405B", "gemma2-9b (tied)": "Gemma-2-9B"}


def short_wl(w):
    return (w.replace("decode", "dec.").replace("prefill", "pre.").replace("mixed 32 dec + 512 chunk", "mixed 32+512")
            .replace(" ctx=2k", ",2k").replace("B=", "$B{=}$").replace("S=", "$S{=}$"))


def main_table(group, label, caption):
    rows, last = [], None
    for r in d:
        if r["group"] != group:
            continue
        res = r["results"]
        name = NAMES[r["model"]]
        cells = {k: g(res, k)["mean"] for k in ("Demand", "Static-k", "FlexGen", "Hot/Cold", "Stride", "CAMP-v1", "CAMP-v2")}
        ext = min(cells[k] for k in ("Demand", "Static-k", "FlexGen", "Hot/Cold", "CAMP-v1"))
        best = min(cells[k] for k in ("Demand", "Static-k", "FlexGen", "Hot/Cold", "Stride", "CAMP-v1", "CAMP-v2"))
        lb = max(r["lb"], r["ideal"])
        txt = []
        for k in ("Demand", "Static-k", "FlexGen", "Hot/Cold", "Stride", "CAMP-v1", "CAMP-v2"):
            v = ms(cells[k])
            txt.append(f"\\textbf{{{v}}}" if cells[k] <= best * 1.0005 else v)
        rows.append(f"{name if r['model'] != last else ''} & {short_wl(r['workload'])} & {r['cache_ratio'] * 100:.0f}\\% & {r['bw']:.0f} & " + " & ".join(txt) +
                    f" & {ms(lb)} & {r['det_ratio_lb']:.3f} & {cells['Demand'] / cells['CAMP-v2']:.2f}$\\times$ & {ext / cells['CAMP-v2']:.2f}$\\times$\\\\")
        last = r["model"]
    emit(r"""\begin{table}[t]
\centering
\caption{""" + caption + r"""}
\label{tab:""" + label + r"""}
\scriptsize
\setlength{\tabcolsep}{2.8pt}
\resizebox{\textwidth}{!}{%
\begin{tabular}{llrrrrrrrrrrrrr}
\toprule
\textbf{Model} & \textbf{Workload} & \textbf{Cache} & \textbf{GB/s} & \textbf{Demand} & \textbf{Static-$k$} & \textbf{FlexGen} & \textbf{Hot/Cold} & \textbf{Stride} & \textbf{CAMP-v1} & \textbf{CAMP-v2} & \textbf{LB} & \textbf{det./LB} & \textbf{vs.\ Demand} & \textbf{vs.\ best ext.}\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\end{table}
""")


COMMON = (r"Iteration latency in ms (mean of 10 seeds with common random numbers; best entry per row in bold). ``Cache'' is the share of weight bytes that fit in the HBM weight cache. "
          r"FlexGen = per-layer fractional split with double buffering; Hot/Cold and Stride = equal-budget pinning with CAMP's or a tuned fixed lookahead (Section~\ref{sec:eval-setup}); "
          r"LB = lower bound of Section~\ref{sec:method-lb}; det./LB = noise-free CAMP-v2 latency divided by LB; ``vs.\ best ext.'' = best of Demand/Static-$k$/FlexGen/Hot-Cold/CAMP-v1 divided by CAMP-v2.")
main_table("main", "main", r"Main comparison on the large-model configurations that do not fit one 80~GB GPU. " + COMMON)
main_table("constrained", "constrained", r"Constrained-cache configurations (a smaller model with 40\% of its weights in HBM, as for a GPU shared with other services). " + COMMON)

# ---------------------------------------------------------------- optimality gap
gp = load("e3_optimality_gap")
rows = []
for r in gp:
    rows.append(f"{r['workload'].replace('B=', '$B{=}$').replace('S=', '$S{=}$')} & {r['cache_ratio']:.1f} & {r['scp_gap_pct']:.1f} & {r['stride_gap_pct']:.1f} & {r['freq_gap_pct']:.1f} & {r['sim_gap_pct']:.1f}\\\\")
emit(r"""\begin{table}[t]
\centering
\caption{Optimality gap of the pin-set selection on an 18-unit model (8 layers, $d{=}2048$), where all $2^{18}$ pin sets can be enumerated. Planner gap: predicted iteration time of the selected set relative to the exhaustive optimum of the planner objective; the last column replays the SCP set in the event engine and compares it with the best set (by simulation) among the enumerated sets.}
\label{tab:gap}
\small
\begin{tabular}{lrrrrr}
\toprule
\textbf{Workload} & \textbf{Cache ratio} & \textbf{SCP (\%)} & \textbf{Stride (\%)} & \textbf{Frequency (\%)} & \textbf{SCP, simulated (\%)}\\
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
cols = [("demand + LRU", "LRU"), ("demand + LFU", "LFU"), ("FlexGen-style layer split", "FlexGen"), ("CAMP prefetch + none pins", "none"),
        ("CAMP prefetch + prefix (conv.) pins", "prefix"), ("CAMP prefetch + frequency (equal budget) pins", "freq."),
        ("CAMP prefetch + stride (equal budget) pins", "stride"), ("CAMP prefetch + knapsack-DP pins", "DP"), ("CAMP prefetch + SCP pins", "SCP")]
rows = []
for c in het:
    nm = c["case"].replace("gemma2-9b tied decode", "Gemma-2-9B, tied embedding (per step)").replace(
        "spec. decoding (13B target + 1B draft, k=4)", "speculative decoding, 13B+1B, $k{=}4$ (per step)").replace(
        "encoder-decoder (24+24 layers, 64 output tokens)", "encoder--decoder, 64 tokens (per request)")
    vals = [c["res"][k]["mean"] for k, _ in cols]
    best = min(vals)
    rows.append(nm + " & " + " & ".join((f"\\textbf{{{v:.2f}}}" if v <= best * 1.005 else f"{v:.2f}") for v in vals) + "\\\\")
emit(r"""\begin{table}[t]
\centering
\caption{Workloads with non-uniform reuse or layer sizes (latency in seconds; 18~GB/s; 35--40\% of weights cached; best in bold, ties within 0.5\%). The first three columns are not CAMP variants; the remaining columns use CAMP's prefetcher with the stated pinning rule. Equal-budget rules keep one streaming slot free; ``prefix'' uses the conventional two-slot reserve.}
\label{tab:hetero}
\scriptsize
\setlength{\tabcolsep}{3pt}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lrrrrrrrrr}
\toprule
\textbf{Workload} & \textbf{LRU} & \textbf{LFU} & \textbf{FlexGen} & \textbf{none} & \textbf{prefix} & \textbf{freq.} & \textbf{stride} & \textbf{DP} & \textbf{SCP}\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\end{table}
""")

# ---------------------------------------------------------------- hardware validation
hv = load("hw_validation")
by = {}
for r in hv["rows"]:
    ratio = hv["comp_per_unit_ms"][str(r["T"])] / hv["copy_per_unit_ms"]
    if r["depth"] == 0:
        cat = "demand paging (serial copy, compute)"
    elif ratio < 0.3:
        cat = "overlapped, link-bound (compute $<0.3\\times$ copy)"
    elif ratio < 1.2:
        cat = "overlapped, balanced ($0.3$--$1.2\\times$)"
    else:
        cat = "overlapped, compute-bound ($>1.2\\times$)"
    by.setdefault(cat, []).append(r)
rows = []
for cat in ("overlapped, link-bound (compute $<0.3\\times$ copy)", "overlapped, balanced ($0.3$--$1.2\\times$)", "overlapped, compute-bound ($>1.2\\times$)", "demand paging (serial copy, compute)"):
    xs = by[cat]
    e = [x["err_pct"] for x in xs]
    cfe = [x["cf_err_pct"] for x in xs]
    rows.append(f"{cat} & {len(xs)} & {statistics.mean(abs(x) for x in e):.1f} & {statistics.mean(e):+.1f} & {max(abs(x) for x in e):.1f} & {statistics.mean(abs(x) for x in cfe):.1f}\\\\")
allx = [r["err_pct"] for r in hv["rows"]]
allc = [r["cf_err_pct"] for r in hv["rows"]]
rows.append(f"\\midrule all configurations & {len(allx)} & {statistics.mean(abs(x) for x in allx):.1f} & {statistics.mean(allx):+.1f} & {max(abs(x) for x in allx):.1f} & {statistics.mean(abs(x) for x in allc):.1f}\\\\")
rows.append(f"\\quad with pinned units only & {hv['pinned_n']} & {hv['pinned_mape']:.1f} & & & \\\\")
emit(r"""\begin{table}[t]
\centering
\caption{Simulator versus a real GPU (Tesla T4, PCIe Gen3 x16; 24 weight tensors of 64~MiB streamed from pinned host memory with a fixed-depth ring). The simulator is composed from separately measured compute-only and copy parameters and asked to predict the measured composite iteration time; no parameter is fitted to the composites. The last column gives the error of the closed-form expression $\max(n_s t_{copy}, N t_{comp})$ (demand paging: $n_s t_{copy}+N t_{comp}$), which uses the same measured inputs but no simulator.}
\label{tab:hw}
\small
\begin{tabular}{lrrrrr}
\toprule
\textbf{Regime} & \textbf{Configs} & \textbf{MAPE (\%)} & \textbf{Signed (\%)} & \textbf{Max $|$err$|$ (\%)} & \textbf{Closed-form MAPE (\%)}\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---------------------------------------------------------------- cost model
cmd = load("e5_cost_model")
cm = cmd["cases"]
rows = []
for c in cm:
    est = c["estimator"]
    rows.append(f"{c['name']} & {est['static']['before'] * 100:.1f} & {est['online']['after'] * 100:.1f} & {ms(c['ref']['no_pins'][0])} & {ms(c['ref']['generic'][0])} & {ms(c['ref']['calibrated'][0])} & {ms(c['ref']['exact'][0])} & {ms(c['lb'])}\\\\")
emit(r"""\begin{table}[t]
\centering
\caption{Runtime cost model. Mean absolute percentage error (MAPE) of the per-layer compute-time estimate before and after online calibration with one probe iteration, and the resulting iteration latency (ms) when SCP pins are planned without pinning, from the generic roofline estimate, from the calibrated estimate and from the exact device model.}
\label{tab:costmodel}
\small
\begin{tabular}{lrrrrrrr}
\toprule
\textbf{Workload} & \textbf{MAPE generic (\%)} & \textbf{MAPE calib. (\%)} & \textbf{no pins} & \textbf{generic} & \textbf{calib.} & \textbf{exact} & \textbf{LB}\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# shape-dependent ground truth
rows = []
for r in cmd["shape"]:
    if r["variant"].startswith("oracle"):
        continue
    rows.append(f"{r['evaluate'].replace('B=', '$B{=}$').replace('S=', '$S{=}$')} & {r['probe'].replace('B=', '$B{=}$').replace('S=', '$S{=}$')} & {r['variant']} & {r['mape'] * 100 if r['mape'] < 1 else r['mape']:.1f} & {ms(r['mean'])} & {ms(r['lb'])}\\\\")
emit(r"""\begin{table}[t]
\centering
\caption{Cost-model error under a shape-dependent ground truth (GEMM efficiency ramps with arithmetic intensity, which the runtime estimator does not model), Llama-3-70B, 18~GB/s. ``Probe'' is the shape on which the online calibration is performed; a probe from a different shape leaves a residual error, and the plan is evaluated on the other shape.}
\label{tab:shape}
\scriptsize
\resizebox{\textwidth}{!}{%
\begin{tabular}{lllrrr}
\toprule
\textbf{Evaluated on} & \textbf{Probe} & \textbf{Estimator} & \textbf{MAPE (\%)} & \textbf{Latency (ms)} & \textbf{LB (ms)}\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\end{table}
""")

# ---------------------------------------------------------------- alternatives
alt = load("e11_alternatives")
rows = []
for r in alt:
    if r["model"] == "llama3-405b" and r["quant"].startswith("INT4") is False and False:
        continue
    fit = "yes" if r["fits_one_gpu"] else "no"
    tg = "--" if r["two_gpu_latency_optimistic"] is None else ms(r["two_gpu_latency_optimistic"])
    if r["two_gpu_latency_optimistic"] is None:
        tg = "does not fit"
    rows.append(f"{r['model'].replace('llama3-', 'Llama-3-').upper().replace('LLAMA-3-', 'Llama-3-')} & {r['quant'].replace(' (weight-only)', '')} & {r['workload'].replace('decode B=16 ctx=2k', 'dec.').replace('prefill B=4 S=2k', 'pre.')} & "
                f"{r['weight_gb']:.0f} & {fit} & {ms(r['cxl18']) if not r['fits_one_gpu'] else ms(r['ideal'])} & {ms(r['cxl45']) if not r['fits_one_gpu'] else ms(r['ideal'])} & {ms(r['hostdram52']) if not r['fits_one_gpu'] else ms(r['ideal'])} & {tg}\\\\")
emit(r"""\begin{table}[t]
\centering
\caption{Alternatives to streaming from a CXL expander: weight quantisation, pinned host DRAM, and two GPUs with tensor parallelism (latency in ms; CAMP-v2 where weights are streamed; ``resident'' = all weights fit one 80~GB GPU together with the KV cache). The two-GPU column is an optimistic estimate (perfect scaling, no communication) and is shown only to indicate when a second GPU makes streaming unnecessary.}
\label{tab:alt}
\scriptsize
\resizebox{\textwidth}{!}{%
\begin{tabular}{lllrcrrrr}
\toprule
\textbf{Model} & \textbf{Weights} & \textbf{Phase} & \textbf{GB} & \textbf{1 GPU fits} & \textbf{CXL 18} & \textbf{CXL 45} & \textbf{host DRAM 52} & \textbf{2 GPUs (opt.)}\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\end{table}
""")

import re
for block in out:
    m = re.search(r"\\label\{tab:(\w+)\}", block)
    with open(os.path.join(ROOT, "manuscript", f"tab_{m.group(1)}.tex"), "w") as f:
        f.write(block)
    print("wrote manuscript/tab_%s.tex" % m.group(1))
