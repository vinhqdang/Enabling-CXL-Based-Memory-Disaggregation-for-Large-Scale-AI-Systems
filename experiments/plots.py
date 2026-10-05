"""Generate all manuscript figures from results/*.json  (python experiments/plots.py)."""
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.ticker import NullFormatter, ScalarFormatter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "manuscript")

plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 7.8,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130,
                     "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5})
# Okabe-Ito colour-blind-safe palette
C = dict(demand="#999999", static="#B0B0B0", reactive="#CC79A7", aggr="#E69F00", flexgen="#56B4E9", hotcold="#0072B2",
         v1="#009E73", v2="#D55E00", bound="#000000", ideal="#000000")


def load(n):
    return json.load(open(os.path.join(RES, n + ".json")))


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# ------------------------------------------------------------------ fig: regime map
def fig_regime():
    d = load("e1_regime_map")
    Ts, BWs = d["Ts"], d["BWs"]
    Z = np.zeros((len(Ts), len(BWs)))
    for c in d["cells"]:
        Z[Ts.index(c["T"]), BWs.index(c["bw"])] = c["camp"] / c["ideal"]
    fig, ax = plt.subplots(figsize=(6.4, 3.7))
    im = ax.imshow(Z, aspect="auto", origin="lower", cmap="viridis_r", norm=LogNorm(vmin=1.0, vmax=200))
    ax.set_xticks(range(len(BWs)), [str(b) for b in BWs])
    ax.set_yticks(range(len(Ts)), [f"{t:,}" for t in Ts])
    ax.set_xlabel("effective CXL-to-GPU bandwidth (GB/s)")
    ax.set_ylabel("tokens per iteration")
    ax.grid(False)
    for i in range(len(Ts)):
        for j in range(len(BWs)):
            v = Z[i, j]
            ax.text(j, i, f"{v:.1f}" if v < 10 else f"{v:.0f}", ha="center", va="center", fontsize=7,
                    color="white" if v > 6 else "black")
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("CAMP latency / all-weights-in-HBM latency")
    ax.set_title("Regime map, Llama-2-13B, 40% of weights fit in HBM (iteration = one GEMM batch)")
    save(fig, "regime_map")


# ------------------------------------------------------------------ fig: main comparison
MAIN_KEYS = [("Demand+LRU", "Demand + LRU", C["demand"]), ("Static-k tuned", "Static-k (tuned)", C["static"]),
             ("FlexGen-style", "FlexGen-style", C["flexgen"]), ("Hot/Cold (PowerInfer-style)", "Hot/Cold (PowerInfer-style)", C["hotcold"]),
             ("CAMP-v1 (orig.)", "CAMP-v1 (orig.)", C["v1"]), ("CAMP-v2", "CAMP-v2 (this work)", C["v2"])]


def _find(res, prefix):
    for k, v in res.items():
        if k.startswith(prefix):
            return v
    raise KeyError(prefix)


def fig_main():
    d = load("e2_main")
    sel = [("llama2-13b", "decode B=16 ctx=2k", "13B decode\nB=16, 2k ctx"),
           ("llama2-13b", "mixed 64 dec + 512 chunk", "13B mixed\n64 dec+512 ch."),
           ("llama2-13b", "prefill B=4 S=4k", "13B prefill\nB=4, S=4k"),
           ("llama3-70b", "decode B=16 ctx=2k", "70B decode\nB=16, 2k ctx"),
           ("llama3-70b", "prefill B=4 S=2k", "70B prefill\nB=4, S=2k"),
           ("llama3-70b", "prefill B=4 S=4k", "70B prefill\nB=4, S=4k")]
    fig, axes = plt.subplots(2, 1, figsize=(7.6, 5.2), sharex=True)
    for ax, bw in zip(axes, (18.0, 45.0)):
        w = 0.13
        for ki, (key, label, col) in enumerate(MAIN_KEYS):
            ys, es = [], []
            for (m, wl, _) in sel:
                r = [x for x in d if x["model"] == m and x["workload"] == wl and x["bw"] == bw][0]
                v = _find(r["results"], key)
                lb = max(r["lb"], r["ideal"])
                ys.append(v["mean"] / lb)
                es.append(v["ci"] / lb)
            ax.bar(np.arange(len(sel)) + (ki - 2.5) * w, ys, w, yerr=es, color=col, label=label, error_kw=dict(lw=0.7))
        ax.axhline(1.0, color="k", lw=0.8, ls="--")
        ax.set_ylabel(f"latency / lower bound\n({bw:g} GB/s link)")
        ax.set_yscale("log")
        ax.set_ylim(0.95, 4.2)
        ax.set_yticks([1, 1.5, 2, 3, 4], ["1", "1.5", "2", "3", "4"])
    axes[0].legend(ncol=3, loc="upper left", frameon=False, bbox_to_anchor=(0.0, 1.02))
    axes[1].set_xticks(range(len(sel)), [s[2] for s in sel], fontsize=8)
    fig.suptitle("Iteration latency relative to max(compute-only time, link-time lower bound); lower is better", y=0.93, fontsize=9)
    save(fig, "main_comparison")


# ------------------------------------------------------------------ fig: pin strategies
def fig_pins():
    d = load("e3_pin_strategies")
    sel = [("llama2-13b", "decode B=16"), ("llama2-13b", "prefill B=1 S=2k"), ("llama2-13b", "prefill B=4 S=4k"),
           ("llama3-70b", "prefill B=4 S=2k"), ("llama3-70b", "prefill B=4 S=4k")]
    names = [("none", "no pins", C["demand"]), ("prefix (FlexGen-style)", "prefix", C["flexgen"]), ("frequency (orig. Alg. 2)", "frequency (orig.)", C["hotcold"]),
             ("random", "random", "#CC79A7"), ("stride (interleaved)", "stride", C["aggr"]), ("knapsack-DP (Eq. 2)", "knapsack-DP", C["v1"]),
             ("SCP (plan+verify)", "SCP (this work)", C["v2"])]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), sharey=True)
    for ax, bw in zip(axes, (18.0, 45.0)):
        w = 0.115
        for ni, (key, label, col) in enumerate(names):
            ys, es = [], []
            for (m, wl) in sel:
                r = [x for x in d if x["model"] == m and x["workload"] == wl and x["bw"] == bw][0]
                lb = max(r["lb"], r["ideal"])
                ys.append(r["res"][key]["mean"] / lb)
                es.append(r["res"][key]["ci"] / lb)
            ax.bar(np.arange(len(sel)) + (ni - 3) * w, ys, w, yerr=es, color=col, label=label, error_kw=dict(lw=0.6))
        ax.axhline(1.0, color="k", lw=0.8, ls="--")
        ax.set_title(f"{bw:g} GB/s")
        ax.set_xticks(range(len(sel)), ["13B\ndecode", "13B\npre. 1x2k", "13B\npre. 4x4k", "70B\npre. 4x2k", "70B\npre. 4x4k"], fontsize=7.5)
        ax.set_yscale("log")
        ax.set_ylim(0.95, 2.1)
        ax.set_yticks([1, 1.25, 1.5, 2], ["1", "1.25", "1.5", "2"])
    axes[0].set_ylabel("latency / lower bound")
    axes[0].legend(ncol=2, frameon=False, loc="upper left")
    save(fig, "pin_strategies")


# ------------------------------------------------------------------ fig: heterogeneity
def fig_hetero():
    d = load("e4_reuse_heterogeneity")["heterogeneous"]
    keys = [("demand + LRU", "demand + LRU", C["demand"]), ("demand + LFU", "demand + LFU", "#CC79A7"),
            ("CAMP prefetch + none pins", "CAMP, no pins", C["static"]), ("CAMP prefetch + prefix pins", "CAMP + prefix pins", C["flexgen"]),
            ("CAMP prefetch + frequency pins", "CAMP + frequency pins", C["hotcold"]), ("CAMP prefetch + knapsack-DP pins", "CAMP + knapsack-DP", C["v1"]),
            ("CAMP prefetch + SCP pins", "CAMP + SCP", C["v2"])]
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.1))
    titles = ["Gemma-2-9B (tied embedding)\nper decode step", "speculative decoding\n13B target + 1B draft, per step",
              "encoder-decoder, 64 tokens\nper request"]
    for ax, c, t in zip(axes, d, titles):
        vals = [c["res"][k]["mean"] for k, _, _ in keys]
        errs = [c["res"][k]["ci"] for k, _, _ in keys]
        base = vals[0]
        ax.barh(range(len(keys)), [v / base for v in vals], xerr=[e / base for e in errs], color=[k[2] for k in keys], error_kw=dict(lw=0.6))
        ax.set_yticks(range(len(keys)), [k[1] for k in keys] if ax is axes[0] else [""] * len(keys))
        ax.invert_yaxis()
        ax.set_title(t)
        ax.set_xlim(0, 1.3)
        ax.set_xlabel("latency relative to demand+LRU (absolute at bar end)")
        for i, v in enumerate(vals):
            ax.text(v / base + 0.02, i, f"{v:.2f} s" if v < 10 else f"{v:.1f} s", va="center", fontsize=7)
    save(fig, "reuse_heterogeneity")


# ------------------------------------------------------------------ fig: cost model
def fig_costmodel():
    d = load("e5_cost_model")["cases"]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2))
    for ax, ci in zip(axes, (0, 1)):
        c = d[ci]
        for online, lab, col, ls in ((False, "static roofline (no calibration)", C["demand"], "-"), (True, "online-calibrated (1 probe iteration)", C["v2"], "-")):
            xs = sorted({g["bias"] for g in c["grid"]})
            for sg, mk in ((0.0, "o"), (0.5, "s")):
                ys = [[g for g in c["grid"] if g["bias"] == b and g["online"] == online and g["sigma"] == sg][0] for b in xs]
                ax.plot(xs, [y["mean"] * 1e3 for y in ys], marker=mk, ms=4, lw=1.3 if sg == 0 else 0.8, color=col, ls=ls if sg == 0 else "--",
                        label=f"{lab}, noise $\\sigma$={sg}")
        ax.axhline(c["ref"]["no_pins"][0] * 1e3, color="k", lw=0.7, ls=":")
        ax.text(0.27, c["ref"]["no_pins"][0] * 1e3 * 1.005, "no pinning", fontsize=7)
        ax.axhline(c["lb"] * 1e3, color="k", lw=0.7, ls="--")
        ax.set_xscale("log", base=2)
        ax.set_xticks([0.25, 0.5, 1, 2, 4], ["1/4", "1/2", "1", "2", "4"])
        ax.set_xlabel("systematic bias of the compute-time estimate")
        ax.set_ylabel("iteration latency (ms)")
        ax.set_title(c["name"])
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=2, frameon=False, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, -0.12))
    save(fig, "cost_model_sensitivity")


# ------------------------------------------------------------------ fig: scaling
def fig_scaling():
    d = load("e6_batch_scaling")
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.3))
    ax = axes[0]
    for bw, ls in ((18.0, "-"), (45.0, "--")):
        rows = [r for r in d["rows"] if r["bw"] == bw]
        ax.plot([r["T"] for r in rows], [r["T"] / r["camp"] for r in rows], ls, color=C["v2"], marker="o", ms=3, label=f"CAMP-v2, {bw:g} GB/s")
        ax.plot([r["T"] for r in rows], [r["T"] / r["demand"] for r in rows], ls, color=C["demand"], marker="o", ms=3, label=f"demand+LRU, {bw:g} GB/s")
    rows = [r for r in d["rows"] if r["bw"] == 18.0]
    ax.plot([r["T"] for r in rows], [r["T"] / r["ideal"] for r in rows], "k:", label="all weights in HBM")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("tokens per iteration"); ax.set_ylabel("throughput (tokens/s)")
    ax.set_title("(a) tokens per iteration (13B)")
    ax.legend(frameon=False, fontsize=6.5)
    ax = axes[1]
    for bw, ls in ((18.0, "-"), (45.0, "--")):
        rows = [r for r in d["decode_rows"] if r["bw"] == bw]
        ax.plot([r["B"] for r in rows], [r["B"] / r["camp"] for r in rows], ls, color=C["v2"], marker="o", ms=3, label=f"CAMP-v2, {bw:g} GB/s")
        ax.plot([r["B"] for r in rows], [r["B"] / r["demand"] for r in rows], ls, color=C["demand"], marker="o", ms=3, label=f"demand+LRU, {bw:g} GB/s")
    rows = [r for r in d["decode_rows"] if r["bw"] == 18.0]
    ax.plot([r["B"] for r in rows], [r["B"] / r["ideal"] for r in rows], "k:", label="all weights in HBM")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("decode batch size (512-token contexts)"); ax.set_ylabel("throughput (tokens/s)")
    ax.set_title("(b) decode batch size (13B)")
    ax = axes[2]
    for bw, ls in ((18.0, "-"), (45.0, "--")):
        rows = [r for r in d["decode_rows"] if r["bw"] == bw]
        ax.plot([r["B"] for r in rows], [r["camp"] / r["ideal"] for r in rows], ls, color=C["v2"], marker="o", ms=3, label=f"{bw:g} GB/s")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("decode batch size"); ax.set_ylabel("slowdown vs. all-in-HBM")
    ax.set_title("(c) decode slowdown")
    ax.set_yticks([4, 10, 20, 40, 80], ["4", "10", "20", "40", "80"])
    ax.minorticks_off()
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, "throughput_scaling")


def fig_modelscale():
    d = load("e7_model_scale")
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2))
    for ax, wl, title in zip(axes, ("decode B=16 ctx=2k", "prefill B=4 S=2k"), ("decode, B=16, 2k context", "prefill, B=4, S=2k")):
        for bw, ls in ((18.0, "-"), (45.0, "--")):
            rows = [r for r in d if r["workload"] == wl and r["bw"] == bw]
            xs = [r["params_b"] for r in rows]
            for key, col, lab in (("Demand+LRU", C["demand"], "demand+LRU"), ("Hot/Cold (PowerInfer-style)", C["hotcold"], "Hot/Cold"), ("CAMP-v2", C["v2"], "CAMP-v2")):
                ax.plot(xs, [r["results"][key]["mean"] for r in rows], ls, marker="o", ms=3, color=col, label=f"{lab}, {bw:g} GB/s")
        rows = [r for r in d if r["workload"] == wl and r["bw"] == 18.0]
        ax.plot([r["params_b"] for r in rows], [r["ideal"] for r in rows], "k:", marker="o", ms=3, label="weights resident (needs more HBM)")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("parameters (billions)"); ax.set_ylabel("iteration latency (s)")
        ax.set_title(title)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=3, frameon=False, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, -0.14))
    save(fig, "model_scale")


def fig_link():
    d = load("e8_link_sensitivity")
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.2))
    for ax, wl in zip(axes[:2], ("decode B=16", "prefill B=4 S=4k")):
        rows = [r for r in d["bw"] if r["workload"] == wl]
        xs = [r["bw"] for r in rows]
        ax.plot(xs, [r["Demand+LRU"] for r in rows], marker="o", ms=3, color=C["demand"], label="demand+LRU")
        ax.plot(xs, [r["Hot/Cold (PowerInfer-style)"] for r in rows], marker="o", ms=3, color=C["hotcold"], label="Hot/Cold")
        ax.plot(xs, [r["CAMP-v2"] for r in rows], marker="o", ms=3, color=C["v2"], label="CAMP-v2")
        ax.plot(xs, [max(r["lb"], r["ideal"]) for r in rows], "k--", lw=0.9, label="lower bound")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xticks([8, 12, 18, 24, 32, 45, 64], ["8", "12", "18", "24", "32", "45", "64"])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(ScalarFormatter()); ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("effective bandwidth (GB/s)"); ax.set_ylabel("iteration latency (s)")
        ax.set_title(f"({'a' if wl.startswith('dec') else 'b'}) {wl}, 13B, 40% cached")
    axes[0].legend(frameon=False)
    ax = axes[2]
    for wl, ls in (("decode B=16", "-"), ("prefill B=4 S=4k", "--")):
        rows = [r for r in d["cache"] if r["workload"] == wl]
        xs = [r["ratio"] for r in rows]
        ax.plot(xs, [r["CAMP-v2"] / max(r["lb"], r["ideal"]) for r in rows], ls, marker="o", ms=3, color=C["v2"], label=f"CAMP-v2, {wl}")
        ax.plot(xs, [r["Hot/Cold (PowerInfer-style)"] / max(r["lb"], r["ideal"]) for r in rows], ls, marker="o", ms=3, color=C["hotcold"], label=f"Hot/Cold, {wl}")
    ax.set_xlabel("fraction of weights cached in HBM"); ax.set_ylabel("latency / lower bound")
    ax.set_title("(c) cache-size sweep, 18 GB/s")
    ax.legend(frameon=False, fontsize=6.3, loc="upper right", bbox_to_anchor=(1.0, 1.0))
    save(fig, "link_sensitivity")


def fig_hw():
    h = load("hw_validation")
    raw = json.load(open(os.path.join(RES, "hw", "hw_T4_v2.json")))
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.2))
    ax = axes[0]
    for kind, col, lab in (("pinned", C["v2"], "pinned (direct DMA)"), ("pageable", C["demand"], "pageable (bounce buffer)")):
        sz = sorted(int(k) for k in raw["h2d"][kind])
        ax.plot([s / 1e6 for s in sz], [raw["h2d"][kind][str(s)]["median"] * 1e3 for s in sz], marker="o", ms=3, color=col, label=lab)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("copy size (MB)"); ax.set_ylabel("host-to-GPU copy time (ms)")
    ax.set_title(f"(a) measured copies, {h['device']['name']}\npinned {h['pinned_fit']['bw_GBps']:.1f} GB/s, pageable {h['pageable_fit']['bw_GBps']:.1f} GB/s")
    ax.legend(frameon=False)
    ax = axes[1]
    rows = h["rows"]
    colmap = {0: "#999999", 1: C["hotcold"], 2: C["v1"], 4: C["aggr"]}
    for r in rows:
        ax.scatter(r["hw_ms"], r["sim_ms"], color=colmap[r["depth"]], s=22 + 10 * (r["n_pinned"] > 0), edgecolor="k", lw=0.4)
    lim = [50, 600]
    ax.plot(lim, lim, "k-", lw=0.8)
    ax.fill_between(lim, [x * 0.9 for x in lim], [x * 1.1 for x in lim], color="k", alpha=0.08)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks([60, 100, 200, 400], ["60", "100", "200", "400"]); ax.set_yticks([60, 100, 200, 400], ["60", "100", "200", "400"])
    ax.xaxis.set_minor_formatter(NullFormatter()); ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("measured iteration time (ms)"); ax.set_ylabel("simulated (ms)")
    ax.set_title("(b) 30 streaming configurations\n(grey band: $\\pm$10%; colour: prefetch depth)")
    for dp, col in colmap.items():
        ax.scatter([], [], color=col, label=f"depth {dp}", edgecolor="k", lw=0.4, s=22)
    ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    ax = axes[2]
    cats = {}
    for r in rows:
        ratio = h["comp_per_unit_ms"][str(r["T"])] / h["copy_per_unit_ms"]
        cat = "link-\nbound" if ratio < 0.3 else ("balanced" if ratio < 1.2 else "compute-\nbound")
        if r["depth"] == 0:
            cat = "demand\npaging"
        cats.setdefault(cat, []).append(abs(r["err_pct"]))
    order = ["link-\nbound", "balanced", "compute-\nbound", "demand\npaging"]
    ax.bar(range(len(order)), [np.mean(cats[o]) for o in order], color=[C["v1"], C["aggr"], C["flexgen"], C["demand"]])
    ax.set_xticks(range(len(order)), order, fontsize=8)
    ax.set_ylabel("mean |error| (%)")
    ax.set_title(f"(c) error by regime; overall MAPE {h['mape']:.1f}%")
    for i, o in enumerate(order):
        ax.text(i, np.mean(cats[o]) + 0.3, f"n={len(cats[o])}", ha="center", fontsize=7)
    save(fig, "hardware_validation")


def fig_robust():
    e = load("e9_robustness")
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.1))
    ax = axes[0]
    for wl, col in (("decode B=16", C["v1"]), ("prefill B=4 S=4k", C["v2"])):
        rows = [r for r in e["noise"] if r["workload"] == wl]
        xs = [r["jitter"] for r in rows]
        ax.plot(xs, [r["CAMP-v2"]["mean"] / r["Hot/Cold"]["mean"] for r in rows], marker="o", ms=3, color=col, label=f"CAMP-v2 / Hot-Cold, {wl}")
        ax.plot(xs, [r["CAMP-v2"]["mean"] / r["Demand+LRU"]["mean"] for r in rows], marker="s", ms=3, ls="--", color=col, label=f"CAMP-v2 / demand+LRU, {wl}")
    ax.set_xlabel("link jitter $\\sigma$ (log-normal; drift and compute noise scaled alongside)")
    ax.set_ylabel("latency ratio (paired seeds)")
    ax.set_title("(a) conclusions under increasing noise")
    ax.legend(frameon=False, fontsize=6.2)
    ax = axes[1]
    pr = e["planner"]
    ax.scatter([r["sim"] for r in pr], [r["pred"] for r in pr], s=12, color=C["hotcold"], edgecolor="k", lw=0.3)
    lim = [min(r["sim"] for r in pr) * 0.8, max(r["sim"] for r in pr) * 1.2]
    ax.plot(lim, lim, "k-", lw=0.8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("engine iteration time (s)"); ax.set_ylabel("planner prediction (s)")
    ax.set_title(f"(b) planner vs. engine, 80 random configs\nMAPE {e['planner_mape']:.2f}%, max {max(abs(r['err_pct']) for r in pr):.1f}%")
    save(fig, "robustness")


if __name__ == "__main__":
    for f in (fig_regime, fig_main, fig_pins, fig_hetero, fig_costmodel, fig_scaling, fig_modelscale, fig_link, fig_hw, fig_robust):
        f()
