#!/usr/bin/env python3
"""Data and regime figures of the placement study (extra to analyze.py and diagrams.py).

Writes into paper/figs/:
    fig_regime.pdf     theory maps over f/c and r/L for L=28, k=3, with the measured workloads
    fig_decode8b.pdf   measured and predicted period / lower bound for four workloads
    fig_bandwidth.pdf  predicted iteration time against path bandwidth for a 16-layer budget

Inputs: results/summary/stream_table.csv (written by analyze.py) and results/raw/t4_full_A.log
(per-layer compute and copy records of the T4 run). Colours follow diagrams.py: resident blue,
streamed orange, copies grey.

Usage:
    python3 paper/analysis/extra_figs.py
"""
import csv
import json
import math
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import NullFormatter, NullLocator  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

from placement import balanced, contiguous_first, cycle_time  # noqa: E402
from placement.theory import contiguous_uniform, lb_uniform  # noqa: E402

FIGS = os.path.join(ROOT, "paper", "figs")
SUMMARY = os.path.join(ROOT, "results", "summary", "stream_table.csv")
RAW_A = os.path.join(ROOT, "results", "raw", "t4_full_A.log")
BLUE = "#1f77b4"    # resident layers / balanced placement
ORANGE = "#ff7f0e"  # streamed layers / contiguous placement
GREY = "#7f7f7f"    # copies, measured markers
INK = "#222222"
COLOR_OF = {"contiguous_first": ORANGE, "balanced": BLUE}
RC = {"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42,
      "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6}


def geomspace(a, b, n):
    """n points from a to b, equally spaced in log."""
    return [a * (b / a) ** (i / (n - 1)) for i in range(n)]


def _float_row(r):
    out = dict(r)
    for key, val in r.items():
        if val in ("", None):
            out[key] = None
            continue
        try:
            out[key] = float(val)
        except ValueError:
            pass
    return out


def load_summary():
    with open(SUMMARY, newline="") as fh:
        return [_float_row(r) for r in csv.DictReader(fh)]


def read_rec(path):
    """Yield the JSON payload of every REC line; truncated lines are skipped."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.startswith("REC"):
                continue
            try:
                yield json.loads(line[3:].strip())
            except json.JSONDecodeError:
                continue


def savefig(fig, out):
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)
    return out


# ------------------------------------------------------------------ regime maps

def fig_regime(out, rows):
    L, k, c = 28, 3, 1.0
    fc = geomspace(0.5, 6.0, 300)  # f/c, log-spaced
    rs = list(range(L + 1))         # resident units r; y = r / L
    ratio = [[1.0] * len(fc) for _ in rs]  # contiguous / balanced
    bal = [[1.0] * len(fc) for _ in rs]    # balanced / all-resident
    for i, r in enumerate(rs):
        m = L - r
        for j, x in enumerate(fc):
            f = x * c
            bal[i][j] = lb_uniform(L, r, c, f) / (L * c)
            if f > c and m >= k:
                ratio[i][j] = contiguous_uniform(L, r, k, c, f) / lb_uniform(L, r, c, f)
    y = [r / L for r in rs]
    # cell edges for pcolormesh: geometric midpoints in x, midpoints in y
    fe = [math.sqrt(fc[i] * fc[i + 1]) for i in range(len(fc) - 1)]
    fe = [fc[0] ** 2 / fe[0]] + fe + [fc[-1] ** 2 / fe[-1]]
    ye = [y[0] - 0.5 / L] + [(y[i] + y[i + 1]) / 2 for i in range(len(y) - 1)] + [y[-1] + 0.5 / L]
    marks = sorted(_mean_fc(rows).items(), key=lambda kv: kv[1])

    with plt.rc_context(RC):
        fig = plt.figure(figsize=(7.0, 2.6))
        ax_a = fig.add_axes([0.085, 0.20, 0.33, 0.58])
        ax_b = fig.add_axes([0.575, 0.20, 0.33, 0.58])
        panels = [
            (ax_a, ratio, "contiguous / balanced", "Oranges", (1.0, 2.0), None),
            (ax_b, bal, "balanced / all-resident", "Oranges", (1.0, 3.0), "max"),
        ]
        for ax, z, label, cmap, lim, extend in panels:
            mesh = ax.pcolormesh(fe, ye, z, cmap=cmap, vmin=lim[0], vmax=lim[1], shading="flat")
            cs = ax.contour(fc, y, z, levels=[1.05, 1.2, 1.5], colors=INK, linewidths=0.6)
            ax.clabel(cs, fmt="%.2f", fontsize=6, inline=True, inline_spacing=2)
            xs = geomspace(1.0, 6.0, 100)
            ax.plot(xs, [1 - 1 / x for x in xs], color=INK, lw=0.9, ls="--")
            ax.text(0.53, 0.93, "dashed: $Lc=mf$", fontsize=6.5, color=INK, ha="left", va="top")
            for wl, x in marks:
                ax.axvline(x, color=INK, lw=0.5, ls=":", alpha=0.8)
            ax.set_xscale("log")
            ax.set_xlim(0.5, 6.0)
            ax.set_ylim(0.0, 1.0)
            ax.set_xticks([0.5, 1, 2, 3, 6])
            ax.set_xticklabels(["0.5", "1", "2", "3", "6"])
            ax.xaxis.set_minor_locator(NullLocator())
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_xlabel("$f/c$ (log scale)")
            ax.set_ylabel("resident fraction $r/L$")
            cb = fig.colorbar(mesh, ax=ax, fraction=0.05, pad=0.03, extend=extend)
            cb.set_label(label, fontsize=7)
            cb.ax.tick_params(labelsize=6.5)
            for wl, x in marks:
                ax.text(x, 1.015, _mark_label(wl), transform=blended_transform_factory(ax.transData,
                                                                                    ax.transAxes),
                        rotation=90, ha="center", va="bottom", fontsize=6.5, color=INK)
        fig.text(0.005, 0.995, "(a)", fontsize=8, weight="bold", va="top")
        fig.text(0.5, 0.995, "(b)", fontsize=8, weight="bold", va="top")
        return savefig(fig, out)


def _mark_label(wl):
    kind, _, val = wl.partition(":")
    if kind == "pre":
        return f"T={int(val)}"
    return f"B={int(wl.split(':')[1])}"


def _mean_fc(rows):
    """Mean f/c of each measured llama3.2-3b workload (main group)."""
    acc = defaultdict(list)
    for r in rows:
        if r["group"] == "main" and r["model"] == "llama3.2-3b" and r["f_over_c"] is not None:
            if r["workload"].startswith(("pre:", "dec:")):
                acc[r["workload"]].append(r["f_over_c"])
    return {wl: sum(v) / len(v) for wl, v in acc.items()}


# ------------------------------------------------------------------ decode and prefill panels

DECODE8B_PANELS = [
    ("llama3.2-3b", "dec:16:512", "3B decode, $B$=16"),
    ("llama3.2-3b", "dec:64:512", "3B decode, $B$=64"),
    ("llama3.1-8b-16L", "pre:512", "8B (16 layers), $T$=512"),
    ("llama3.1-8b-16L", "pre:1024", "8B (16 layers), $T$=1024"),
]


def _panel_rows(rows, model, workload):
    """Rows of the main group for one workload, taken from the source file with most rows."""
    cand = [r for r in rows if r["group"] == "main" and r["model"] == model
            and r["workload"] == workload and r["exp"] == "grid"
            and r["placement"] in COLOR_OF]
    if not cand:
        return None, []
    count = defaultdict(int)
    for r in cand:
        count[r["source"]] += 1
    src = max(count, key=lambda s: count[s])
    return src, [r for r in cand if r["source"] == src]


def fig_decode8b(out, rows):
    panels = []
    for model, wl, title in DECODE8B_PANELS:
        src, sel = _panel_rows(rows, model, wl)
        if not sel:
            print("SKIP fig_decode8b:", model, wl, "no main-group rows")
            return None
        panels.append((src, title, sel))

    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, len(panels), figsize=(7.0, 2.4), sharey=False, squeeze=False)
        for ax, (src, title, sel) in zip(axes[0], panels):
            top = 0.0
            for placement in ("contiguous_first", "balanced"):
                col = COLOR_OF[placement]
                for k in sorted({int(r["k"]) for r in sel}):
                    pts = sorted([r for r in sel if r["placement"] == placement and int(r["k"]) == k],
                                 key=lambda r: r["r"])
                    if not pts:
                        continue
                    xs = [r["r"] for r in pts]
                    model_y = [r["pred_cal_ms"] / r["lb_cal_ms"] for r in pts]
                    meas_y = [r["period_ms"] / r["lb_cal_ms"] for r in pts]
                    top = max(top, max(model_y), max(meas_y))
                    ax.plot(xs, model_y, color=col, lw=0.9, ls="-" if k == 3 else "--")
                    ax.plot(xs, meas_y, ls="none", marker="o", ms=3.5,
                            mfc=col if k == 3 else "white", mec=col, mew=0.9, zorder=3)
            ax.axhline(1.0, color=INK, lw=0.5, ls=":")
            ax.set_xlabel("resident units $r$")
            ax.text(0.03, 0.97, title, transform=ax.transAxes, va="top", ha="left", fontsize=7)
            ax.set_ylim(0.98, top + 0.03)
            ax.set_xticks(sorted({r["r"] for r in sel}))
            ax.tick_params(axis="x", labelsize=6.5)
        axes[0, 0].set_ylabel("period / lower bound")
        handles = [
            Line2D([], [], color=ORANGE, lw=1.0, label="contiguous first"),
            Line2D([], [], color=BLUE, lw=1.0, label="balanced"),
            Line2D([], [], color=GREY, marker="o", ms=4, ls="none", label="measured, $k$=3"),
            Line2D([], [], color=GREY, marker="o", ms=4, ls="none", mfc="white", label="measured, $k$=2"),
            Line2D([], [], color=GREY, lw=1.0, ls="-", label="model, $k$=3"),
            Line2D([], [], color=GREY, lw=1.0, ls="--", label="model, $k$=2"),
        ]
        fig.legend(handles=handles, frameon=False, fontsize=7, loc="lower center", ncol=6,
                   bbox_to_anchor=(0.5, 0.0), handlelength=2.2, columnspacing=1.2)
        fig.subplots_adjust(left=0.085, right=0.99, bottom=0.27, top=0.97, wspace=0.30)
        return savefig(fig, out)


# ------------------------------------------------------------------ bandwidth prediction

def t4_unit_times(path, workload):
    """Per-unit compute times (ms) of one workload, copy sizes (bytes), and the all-resident period."""
    comp, nbytes, allres = {}, {}, None
    for d in read_rec(path):
        kind = d.get("kind")
        if kind == "compute" and d.get("workload") == workload:
            comp.setdefault(d["unit_index"], d["c_ms"])
        elif kind == "h2d":
            nbytes.setdefault(d["unit_index"], d["bytes"])
        elif kind == "stream" and d.get("exp") == "allres" and d.get("workload") == workload:
            allres = allres or d["period_ms"]
    units = sorted(comp)
    if not units or units != sorted(nbytes) or allres is None:
        raise RuntimeError(f"incomplete T4 records for {workload} in {path}")
    return [comp[u] for u in units], [nbytes[u] for u in units], allres


def fig_bandwidth(out):
    c_raw, nbytes, allres = t4_unit_times(RAW_A, "pre:512")
    scale = allres / sum(c_raw)  # calibrated: compute times sum to the all-resident period
    c = [x * scale for x in c_raw]
    L = len(c)
    B_MEASURED = 12.3  # GB/s, page-locked PCIe Gen3 path on the T4
    grid = geomspace(8.0, 64.0, 90)
    curves = [
        ("contiguous first, $r$=13, $k$=3", contiguous_first(L, 13), 3, ORANGE, "-"),
        ("balanced, $r$=13, $k$=3", balanced(L, 13), 3, BLUE, "-"),
        ("balanced, $r$=14, $k$=2", balanced(L, 14), 2, BLUE, "--"),
    ]
    series = {}
    for label, res, k, col, ls in curves:
        ys = []
        for B in grid:
            f = [0.02 + b / (B * 1e6) for b in nbytes]  # ms: bytes / (GB/s) plus 0.02 ms per copy
            ys.append(cycle_time(c, f, res, k))
        series[label] = ys

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(3.4, 2.6))
        for label, res, k, col, ls in curves:
            ax.plot(grid, series[label], color=col, lw=1.1, ls=ls, label=label)
        ax.axhline(allres, color=INK, lw=0.8, ls="-.")
        ax.text(grid[0] * 1.05, allres + 6, f"all resident, {allres:.0f} ms", fontsize=6.5, va="bottom")
        ax.axvline(B_MEASURED, color=INK, lw=0.8, ls=":")
        ax.text(B_MEASURED * 1.06, 40, "measured T4 path\n12.3 GB/s", fontsize=6.5, va="bottom",
                ha="left")
        ax.set_xscale("log")
        ax.set_xlim(grid[0], grid[-1])
        ax.set_xticks([8, 12.3, 16, 32, 64])
        ax.set_xticklabels(["8", "12.3", "16", "32", "64"])
        ax.minorticks_off()
        ax.set_xlabel("path bandwidth $B$ (GB/s, log scale)")
        ax.set_ylabel("iteration time (ms)")
        ax.set_ylim(0, max(max(v) for v in series.values()) * 1.05)
        ax.legend(frameon=False, fontsize=6.5, loc="upper right")
        ax.grid(axis="y", color="#dddddd", linewidth=0.4)
        fig.tight_layout(pad=0.4)
        return savefig(fig, out)


def main():
    os.makedirs(FIGS, exist_ok=True)
    rows = load_summary()
    fig_regime(os.path.join(FIGS, "fig_regime.pdf"), rows)
    fig_decode8b(os.path.join(FIGS, "fig_decode8b.pdf"), rows)
    fig_bandwidth(os.path.join(FIGS, "fig_bandwidth.pdf"))


if __name__ == "__main__":
    main()
