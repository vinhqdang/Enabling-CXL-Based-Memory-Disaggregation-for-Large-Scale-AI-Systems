#!/usr/bin/env python3
"""Compare measured T4 weight-streaming periods with the placement model.

Usage: python3 paper/analysis/analyze.py [--include-quick]

Reads REC lines from results/raw/t4_full_*.log (and results/raw/t4_quick.log with
--include-quick). Writes results/summary/{stream_table.csv, summary.json},
paper/numbers.tex and the figures under paper/figs/. Missing data skips the affected
output and is reported on stdout.
"""
import argparse
import csv
import glob
import json
import math
import os
import statistics
import sys
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from placement import cycle_time, cycle_time_contention, schedule_contention  # noqa: E402
from placement.placements import balanced, contiguous_first  # noqa: E402

RAW = os.path.join(ROOT, "results", "raw")
SUMMARY = os.path.join(ROOT, "results", "summary")
FIGS = os.path.join(ROOT, "paper", "figs")
NUMBERS = os.path.join(ROOT, "paper", "numbers.tex")

PLACEMENTS_GRID = ["contiguous_first", "contiguous_last", "balanced", "random0", "random1"]
RANK_THRESHOLD = 0.02
MARKERS = {
    "contiguous_first": "o",
    "contiguous_last": "s",
    "balanced": "^",
    "random0": "D",
    "random1": "v",
    "none": "x",
    "all_resident": "P",
}
# Okabe-Ito hues for workloads (CVD-checked with scripts/validate_palette.js).
COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9"]
SKIPPED = []
NOTES = []


def skip(msg):
    SKIPPED.append(msg)
    print("SKIP:", msg)


def note(msg):
    NOTES.append(msg)
    print("NOTE:", msg)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def pct(x):
    return None if x is None else 100.0 * x


# ---------------------------------------------------------------- parsing

def read_records(path, stats):
    """Parse REC lines; lines that are not valid JSON (truncated runs) are counted and dropped."""
    recs = []
    with open(path, "rb") as fh:
        for raw in fh:
            if not raw.startswith(b"REC "):
                continue
            try:
                recs.append(json.loads(raw[4:].decode("utf-8", "replace")))
            except ValueError:
                stats["truncated_lines"] += 1
    return recs


def build_context(path, recs):
    """Per-file lookup tables: f per (model, unit), c and c_conc per (model, workload, unit)."""
    src = os.path.basename(path)
    ctx = {"source": src, "L": {}, "f": defaultdict(lambda: defaultdict(list)),
           "bytes": [], "c": defaultdict(lambda: defaultdict(list)),
           "c_conc": defaultdict(lambda: defaultdict(list)), "stream": [], "kinds": defaultdict(int),
           "device": None}
    for r in recs:
        kind = r.get("kind", r.get("type"))
        ctx["kinds"][kind] += 1
        if kind == "device" and ctx["device"] is None:
            ctx["device"] = {k: r.get(k) for k in ("name", "cuda", "torch", "capability", "pcie_max")}
        elif kind == "model":
            ctx["L"][r["model"]] = r.get("L")
        elif kind == "h2d":
            ctx["f"][r["model"]][r["unit_index"]].append(r["f_ms"])
            if r.get("bytes"):
                ctx["bytes"].append((r["bytes"], r["f_ms"], r.get("sm_clock_mean"), r.get("power_mean")))
        elif kind == "compute":
            ctx["c"][(r["model"], r["workload"])][r["unit_index"]].append(r["c_ms"])
            ctx.setdefault("clock_power", []).append((r["workload"], r.get("sm_clock_mean"), r.get("power_mean")))
        elif kind == "compute_conc":
            ctx["c_conc"][(r["model"], r["workload"])][r["unit_index"]].append(r["c_conc_ms"])
        elif kind == "stream":
            ctx["stream"].append(r)
            ctx.setdefault("clock_power", []).append((r["workload"], r.get("sm_clock_mean"), r.get("power_mean")))
    return ctx


def avg_map(d):
    return {u: mean(v) for u, v in d.items() if v}


# ---------------------------------------------------------------- analysis rows

def analyze_stream(ctx, r, stats):
    model, wl = r["model"], r["workload"]
    L = ctx["L"].get(model)
    fmap = avg_map(ctx["f"].get(model, {}))
    cmap = avg_map(ctx["c"].get((model, wl), {}))
    cc = avg_map(ctx["c_conc"].get((model, wl), {}))
    if L is None:
        L = len(fmap)
    if L == 0 or any(u not in fmap for u in range(L)) or any(u not in cmap for u in range(L)):
        stats["no_c_or_f"] += 1
        skip(f"{ctx['source']}: {wl} {r.get('placement')} r={r.get('r')} k={r.get('k')}: "
             f"missing per-unit f or c (have f {len(fmap)}/{L}, c {len(cmap)}/{L})")
        return None
    c = [cmap[u] for u in range(L)]
    f = [fmap[u] for u in range(L)]
    res = set(r["resident"])
    k = r["k"]
    streamed = [u for u in range(L) if u not in res]
    sum_c = sum(c)
    sum_f = sum(f[u] for u in streamed)
    lb = max(sum_c, sum_f)
    if streamed and k < 1:
        stats["bad_k"] += 1
        skip(f"{ctx['source']}: {wl} k={k} with streamed units")
        return None
    measured = r["period_ms"]
    pred_base = cycle_time(c, f, res, k)
    pred_cont = None
    if all(u in cc for u in range(L)):
        pred_cont = cycle_time_contention(c, [cc[u] for u in range(L)], f, res, k)
    else:
        stats["no_c_conc"] += 1
    row = {
        "source": ctx["source"], "model": model, "workload": wl, "exp": r.get("exp"),
        "placement": r.get("placement"), "r": r.get("r"), "k": k, "m": len(streamed),
        "period_ms": measured, "period_median_ms": r.get("period_median_ms"),
        "host_enqueue_ms": r.get("host_enqueue_ms"), "lb_ms": lb, "lb_bench_ms": r.get("lb_ms"),
        "sum_c_ms": sum_c, "sum_f_streamed_ms": sum_f,
        "f_over_c": mean(f) / mean(c),
        "pred_base_ms": pred_base, "pred_cont_ms": pred_cont,
        "err_base": measured / pred_base - 1.0,
        "err_cont": None if pred_cont is None else measured / pred_cont - 1.0,
        "has_timeline": bool(r.get("timeline_compute")),
        "sm_clock_mean": r.get("sm_clock_mean"), "power_mean": r.get("power_mean"),
        "temp_max": r.get("temp_max"), "resident": sorted(res),
        "timeline_compute": r.get("timeline_compute") or [], "timeline_copy": r.get("timeline_copy") or [],
        "c": c, "c_conc": [cc[u] for u in range(L)] if pred_cont is not None else None, "f": f,
    }
    return row


CSV_FIELDS = ["source", "model", "workload", "exp", "placement", "r", "k", "m", "period_ms",
              "period_median_ms", "host_enqueue_ms", "lb_ms", "lb_bench_ms", "sum_c_ms",
              "sum_f_streamed_ms", "f_over_c", "pred_base_ms", "pred_cont_ms", "err_base",
              "err_cont", "has_timeline", "sm_clock_mean", "power_mean", "temp_max"]


def regime_label(x):
    if x < 1.0:
        return "compute-bound (f < c)"
    if x < 2.0:
        return "band (c <= f < 2c)"
    return "transfer-bound (f >= 2c)"


def mape_block(rows, key):
    e = [abs(r[key]) for r in rows if r.get(key) is not None]
    if not e:
        return {"n": 0, "mape_pct": None, "max_abs_err_pct": None}
    return {"n": len(e), "mape_pct": round(100 * mean(e), 3), "max_abs_err_pct": round(100 * max(e), 3)}


def group_mape(rows, key, field):
    out = {}
    for name in sorted({r[field] for r in rows}):
        out[name] = mape_block([r for r in rows if r[field] == name], key)
    return out


def ranking(rows):
    """Pairwise ordering agreement among placements sharing (source, workload, r, k, exp)."""
    groups = defaultdict(dict)
    for r in rows:
        if r["placement"] in PLACEMENTS_GRID and r["exp"] in ("grid", "sweep"):
            groups[(r["source"], r["workload"], r["r"], r["k"])][r["placement"]] = r
    tot = base_all = base_paired = cont = 0
    base_agree_all = base_agree_paired = cont_agree = 0
    for g in groups.values():
        names = sorted(g)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = g[names[i]], g[names[j]]
                ma, mb = a["period_ms"], b["period_ms"]
                if abs(ma - mb) / min(ma, mb) <= RANK_THRESHOLD:
                    continue
                tot += 1
                meas_sign = (ma > mb) - (ma < mb)
                pb = (a["pred_base_ms"] > b["pred_base_ms"]) - (a["pred_base_ms"] < b["pred_base_ms"])
                base_agree_all += int(pb == meas_sign)
                if a["pred_cont_ms"] is not None and b["pred_cont_ms"] is not None:
                    base_paired += 1
                    base_agree_paired += int(pb == meas_sign)
                    cont += 1
                    pc = (a["pred_cont_ms"] > b["pred_cont_ms"]) - (a["pred_cont_ms"] < b["pred_cont_ms"])
                    cont_agree += int(pc == meas_sign)
    return {
        "threshold_rel": RANK_THRESHOLD,
        "pairs_over_threshold": tot,
        "base_agree_pct_all_pairs": pct(base_agree_all / tot) if tot else None,
        "base_agree_pct_paired_pairs": pct(base_agree_paired / base_paired) if base_paired else None,
        "contention_agree_pct": pct(cont_agree / cont) if cont else None,
        "pairs_with_contention": cont,
    }


def grid_table(rows):
    groups = defaultdict(dict)
    for r in rows:
        if r["exp"] == "grid" and r["placement"] in PLACEMENTS_GRID:
            groups[(r["source"], r["workload"], r["r"], r["k"])][r["placement"]] = r
    out = []
    for (src, wl, rr, k), g in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1], x[0][2], x[0][3])):
        ent = {"source": src, "workload": wl, "r": rr, "k": k}
        for p, rec in g.items():
            ent[p] = {"measured_ms": rec["period_ms"], "base_ms": rec["pred_base_ms"],
                      "contention_ms": rec["pred_cont_ms"], "lb_ms": rec["lb_ms"],
                      "has_timeline": rec["has_timeline"]}
        if "contiguous_first" in g and "balanced" in g:
            cf, bal = g["contiguous_first"], g["balanced"]
            ent["cf_over_balanced"] = {
                "measured": cf["period_ms"] / bal["period_ms"],
                "base": cf["pred_base_ms"] / bal["pred_base_ms"],
                "contention": None if cf["pred_cont_ms"] is None or bal["pred_cont_ms"] is None
                else cf["pred_cont_ms"] / bal["pred_cont_ms"],
            }
        out.append(ent)
    return out


def sweep_table(rows):
    groups = defaultdict(list)
    for r in rows:
        if r["exp"] == "sweep" and r["placement"] in ("balanced", "contiguous_first"):
            groups[(r["source"], r["workload"], r["r"])].append(r)
    out = []
    for (src, wl, rr), lst in sorted(groups.items()):
        for p in ("balanced", "contiguous_first"):
            pts = sorted([x for x in lst if x["placement"] == p], key=lambda x: x["k"])
            out.append({"source": src, "workload": wl, "r": rr, "placement": p,
                        "points": [{"k": x["k"], "measured_ms": x["period_ms"],
                                    "base_ms": x["pred_base_ms"], "contention_ms": x["pred_cont_ms"]}
                                   for x in pts]})
    return out


# ---------------------------------------------------------------- figures

def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "legend.fontsize": 6, "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.6, "lines.linewidth": 1.0, "lines.markersize": 4,
        "pdf.fonttype": 42, "figure.dpi": 150, "axes.grid": True,
        "grid.color": "#dddddd", "grid.linewidth": 0.4,
    })
    return plt


def fig_accuracy(plt, rows):
    pts = [r for r in rows if r["pred_cont_ms"] is not None]
    if not pts:
        skip("fig_accuracy.pdf: no stream record has a contention prediction")
        return
    fig, ax = plt.subplots(figsize=(3.4, 4.3))
    wls = sorted({r["workload"] for r in pts})
    lo = min(min(r["period_ms"], r["pred_cont_ms"]) for r in pts)
    hi = max(max(r["period_ms"], r["pred_cont_ms"]) for r in pts)
    for i, wl in enumerate(wls):
        for p in sorted({r["placement"] for r in pts}):
            sel = [r for r in pts if r["workload"] == wl and r["placement"] == p]
            if sel:
                filled = MARKERS.get(p, "o") not in ("x", "+")
                ax.scatter([r["pred_cont_ms"] for r in sel], [r["period_ms"] for r in sel],
                           s=18, marker=MARKERS.get(p, "o"), color=COLORS[i % len(COLORS)],
                           edgecolor="black" if filled else None, linewidth=0.3,
                           label=f"{wl}, {p}", zorder=3)
    ax.plot([lo * 0.9, hi * 1.1], [lo * 0.9, hi * 1.1], color="black", linewidth=0.6, linestyle="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo * 0.9, hi * 1.1)
    ax.set_ylim(lo * 0.9, hi * 1.1)
    ax.set_xlabel("Predicted period, contention model (ms)")
    ax.set_ylabel("Measured period (ms)")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=5.5)
    fig.tight_layout(rect=[0, 0.36, 1, 1])
    fig.savefig(os.path.join(FIGS, "fig_accuracy.pdf"))
    plt.close(fig)
    print("wrote", os.path.join(FIGS, "fig_accuracy.pdf"), f"({len(pts)} points)")


def dense_curves(ctx_by_source, src, wl, k, placement):
    """Model predictions for every resident count r at one k (base and contention)."""
    ctx = ctx_by_source[src]
    rec = None
    base_c = None
    c = f = cc = None
    for r in ctx["stream"]:
        if r["workload"] == wl and r["model"]:
            rec = r
            break
    if rec is None:
        return None
    model = rec["model"]
    L = ctx["L"].get(model) or len(avg_map(ctx["f"][model]))
    fmap = avg_map(ctx["f"][model])
    cmap = avg_map(ctx["c"][(model, wl)])
    ccm = avg_map(ctx["c_conc"].get((model, wl), {}))
    if any(u not in fmap or u not in cmap for u in range(L)):
        return None
    c = [cmap[u] for u in range(L)]
    f = [fmap[u] for u in range(L)]
    has_cc = all(u in ccm for u in range(L))
    cc = [ccm[u] for u in range(L)] if has_cc else None
    xs, base, cont = [], [], []
    for r in range(1, L):
        res = balanced(L, r) if placement == "balanced" else contiguous_first(L, r)
        streamed = [u for u in range(L) if u not in res]
        lb = max(sum(c), sum(f[u] for u in streamed))
        xs.append(r)
        base.append(cycle_time(c, f, res, k) / lb)
        cont.append(cycle_time_contention(c, cc, f, res, k) / lb if has_cc else None)
    return xs, base, cont


def fig_penalty(plt, rows, grid, ctx_by_source):
    wls = sorted({g["workload"] for g in grid if g["workload"].startswith("pre:")})
    if not wls:
        skip("fig_penalty.pdf: no prefill workload with contiguous_first and balanced grid records")
        return
    fig, axes = plt.subplots(len(wls), 1, figsize=(3.4, 1.9 * len(wls) + 1.1), squeeze=False)
    for ax, wl in zip(axes[:, 0], wls):
        ent = [g for g in grid if g["workload"] == wl]
        ks = sorted({g["k"] for g in ent})
        src = ent[0]["source"]
        for placement, col in (("contiguous_first", COLORS[0]), ("balanced", COLORS[1])):
            for k in ks:
                cur = dense_curves(ctx_by_source, src, wl, k, placement)
                if cur is None:
                    continue
                xs, base, cont = cur
                ax.plot(xs, cont if cont[0] is not None else base, color=col, linewidth=0.9,
                        linestyle="-" if k == ks[-1] else ":", label=None)
                if cont[0] is not None:
                    ax.plot(xs, base, color=col, linewidth=0.6, linestyle="--", alpha=0.6)
            meas = [(g["r"], g["k"], g[placement]["measured_ms"] / g[placement]["lb_ms"])
                    for g in ent if placement in g]
            for r_, k_, y in meas:
                ax.scatter([r_], [y], s=14, marker="o" if k_ == ks[0] else "s",
                           color=col, edgecolor="black", linewidth=0.3, zorder=3)
        ax.axhline(1.0, color="black", linewidth=0.5, linestyle=":")
        ax.set_xlabel("Resident units r")
        ax.set_ylabel("Period / lower bound")
        ax.text(0.02, 0.96, wl.replace(":", " "), transform=ax.transAxes, va="top", fontsize=6)
    handles = [plt.Line2D([], [], color=COLORS[0], label="contiguous first"),
               plt.Line2D([], [], color=COLORS[1], label="balanced"),
               plt.Line2D([], [], color="black", linestyle="-", label="model, contention"),
               plt.Line2D([], [], color="black", linestyle="--", label="model, base"),
               plt.Line2D([], [], color="black", marker="o", linestyle="", label="measured, k=min"),
               plt.Line2D([], [], color="black", marker="s", linestyle="", label="measured, k=max")]
    fig.legend(handles=handles, frameon=False, fontsize=5.5, loc="lower center", ncol=3)
    fig.tight_layout(rect=[0, 0.1, 1, 1])
    fig.savefig(os.path.join(FIGS, "fig_penalty.pdf"))
    plt.close(fig)
    print("wrote", os.path.join(FIGS, "fig_penalty.pdf"), f"({len(wls)} panels)")


def fig_timeline(plt, grid, rows):
    cands = [g for g in grid if g["k"] == 3 and "cf_over_balanced" in g]
    if not cands:
        skip("fig_timeline.pdf: no (workload, r, k=3) configuration with both contiguous_first and balanced")
        return
    cands.sort(key=lambda g: -g["cf_over_balanced"]["measured"])
    chosen = None
    for g in cands:
        if g["contiguous_first"]["has_timeline"] and g["balanced"]["has_timeline"]:
            chosen = g
            break
    if chosen is None:
        skip("fig_timeline.pdf: no k=3 configuration with timelines for both placements")
        return
    if chosen is not cands[0]:
        note(f"timeline: largest ratio {cands[0]['workload']} r={cands[0]['r']} has no timeline; "
             f"using next best with timelines")
    print(f"timeline configuration: {chosen['workload']} r={chosen['r']} k=3 "
          f"ratio={chosen['cf_over_balanced']['measured']:.3f} source={chosen['source']}")
    recs = {r["placement"]: r for r in rows if r["source"] == chosen["source"]
            and r["workload"] == chosen["workload"] and r["r"] == chosen["r"]
            and r["k"] == 3 and r["exp"] == "grid"}
    fig, axes = plt.subplots(2, 1, figsize=(3.4, 3.7), sharex=True)
    xmin, xmax = 0.0, 0.0
    for ax, p in zip(axes, ("contiguous_first", "balanced")):
        rec = recs[p]
        for u, s, e in rec["timeline_compute"]:
            res = u in rec["resident"]
            ax.broken_barh([(s, e - s)], (1.2, 0.8), facecolors=COLORS[0] if res else COLORS[1],
                           edgecolor="none")
            xmin, xmax = min(xmin, s), max(xmax, e)
        for u, s, e in rec["timeline_copy"]:
            ax.broken_barh([(s, e - s)], (0.0, 0.8), facecolors="#999999", edgecolor="none")
            xmin, xmax = min(xmin, s), max(xmax, e)
        ax.axvline(0.0, color="black", linewidth=0.5)
        ax.axvline(rec["period_ms"], color="black", linewidth=0.5, linestyle=":")
        ax.set_yticks([0.4, 1.6])
        ax.set_yticklabels(["copy", "compute"])
        ax.set_ylim(-0.1, 2.2)
        ax.text(0.01, 0.98, f"{p}, period {rec['period_ms']:.1f} ms", transform=ax.transAxes,
                va="top", fontsize=6)
    axes[1].set_xlabel("Time from iteration start (ms)")
    axes[1].set_xlim(xmin * 1.02, xmax * 1.02)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[0], label="resident unit compute"),
               plt.Rectangle((0, 0), 1, 1, color=COLORS[1], label="streamed unit compute"),
               plt.Rectangle((0, 0), 1, 1, color="#999999", label="host-to-GPU copy")]
    fig.legend(handles=handles, frameon=False, fontsize=5.5, loc="lower center", ncol=3)
    fig.tight_layout(rect=[0, 0.1, 1, 1])
    fig.savefig(os.path.join(FIGS, "fig_timeline.pdf"))
    plt.close(fig)
    print("wrote", os.path.join(FIGS, "fig_timeline.pdf"))


def fig_slots(plt, sweeps):
    groups = defaultdict(dict)
    for s in sweeps:
        groups[(s["source"], s["workload"], s["r"])][s["placement"]] = s["points"]
    if not groups:
        skip("fig_slots.pdf: no slot sweep records (exp == 'sweep')")
        return
    n = len(groups)
    fig, axes = plt.subplots(n, 1, figsize=(3.4, 2.3 * n + 0.2), squeeze=False)
    cols = {"balanced": COLORS[1], "contiguous_first": COLORS[0]}
    for ax, (key, plc) in zip(axes[:, 0], groups.items()):
        for p in ("contiguous_first", "balanced"):
            pts = plc.get(p, [])
            if not pts:
                continue
            ks = [x["k"] for x in pts]
            ax.scatter(ks, [x["measured_ms"] for x in pts], s=18, marker=MARKERS[p],
                       color=cols[p], edgecolor="black", linewidth=0.3, zorder=3)
            ax.plot(ks, [x["base_ms"] for x in pts], color=cols[p], linestyle="--", linewidth=0.7)
            if all(x["contention_ms"] is not None for x in pts):
                ax.plot(ks, [x["contention_ms"] for x in pts], color=cols[p], linestyle="-", linewidth=0.9)
        ax.set_xlabel("Slots k")
        ax.set_ylabel("Period (ms)")
        ax.text(0.02, 0.96, f"{key[1].replace(':', ' ')}, r={key[2]}", transform=ax.transAxes, va="top",
                fontsize=6)
    handles = [plt.Line2D([], [], color=cols["contiguous_first"], label="contiguous first"),
               plt.Line2D([], [], color=cols["balanced"], label="balanced"),
               plt.Line2D([], [], color="black", linestyle="-", label="model, contention"),
               plt.Line2D([], [], color="black", linestyle="--", label="model, base"),
               plt.Line2D([], [], color="black", marker="o", linestyle="", label="measured")]
    axes[0, 0].legend(handles=handles, frameon=False, fontsize=5.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig_slots.pdf"))
    plt.close(fig)
    print("wrote", os.path.join(FIGS, "fig_slots.pdf"))


# ---------------------------------------------------------------- numbers

def write_numbers(macros):
    lines = ["% Generated by paper/analysis/analyze.py. Values are numbers (percent signs are added in the text)."]
    for name, val in macros:
        assert name.isalpha(), name
        lines.append(f"\\newcommand{{\\{name}}}{{{val}}}")
    with open(NUMBERS, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote", NUMBERS, f"({len(macros)} macros)")


def fmt(x, nd):
    return "n/a" if x is None else f"{x:.{nd}f}"


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-quick", action="store_true")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(RAW, "t4_full_*.log")))
    if args.include_quick:
        q = os.path.join(RAW, "t4_quick.log")
        if os.path.exists(q):
            files.append(q)
    if not files:
        print("no input logs found under", RAW)
        return 1
    print("inputs:", [os.path.basename(f) for f in files])

    os.makedirs(SUMMARY, exist_ok=True)
    os.makedirs(FIGS, exist_ok=True)

    stats = defaultdict(int)
    all_rows, ctx_by_source, devices, kinds_total = [], {}, {}, defaultdict(int)
    h2d_rates = []
    clock_power = defaultdict(lambda: {"sm": [], "pw": []})
    for path in files:
        recs = read_records(path, stats)
        if not recs:
            skip(f"{os.path.basename(path)}: no REC lines (run did not start or failed)")
            continue
        ctx = build_context(path, recs)
        ctx_by_source[ctx["source"]] = ctx
        devices[ctx["source"]] = ctx["device"]
        for k_, v in ctx["kinds"].items():
            kinds_total[k_] += v
        for b, ms, sm, pw in ctx["bytes"]:
            h2d_rates.append((ctx["source"], b / (ms * 1e6)))
        for wl, sm, pw in ctx.get("clock_power", []):
            if sm is not None:
                clock_power[wl]["sm"].append(sm)
            if pw is not None:
                clock_power[wl]["pw"].append(pw)
        if not ctx["stream"]:
            note(f"{ctx['source']}: no stream records (kinds seen: {dict(ctx['kinds'])})")
        for r in ctx["stream"]:
            row = analyze_stream(ctx, r, stats)
            if row is not None:
                all_rows.append(row)
        if ctx["kinds"].get("stream", 0) == 0 and ctx["kinds"].get("h2d", 0) > 0:
            note(f"{ctx['source']}: measurements only (h2d/compute), no stream records yet")
    print(f"stream rows: {len(all_rows)}, truncated REC lines dropped: {stats['truncated_lines']}")

    with_cont = [r for r in all_rows if r["pred_cont_ms"] is not None]

    # CSV
    with open(os.path.join(SUMMARY, "stream_table.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    print("wrote", os.path.join(SUMMARY, "stream_table.csv"))

    # accuracy
    acc_cont = mape_block(with_cont, "err_cont")
    acc_base_sub = mape_block(with_cont, "err_base")
    acc_base_all = mape_block(all_rows, "err_base")
    worst = max(with_cont, key=lambda r: abs(r["err_cont"]), default=None)
    worst_base = max(with_cont, key=lambda r: abs(r["err_base"]), default=None)
    accuracy = {
        "contention": dict(acc_cont, by_workload=group_mape(with_cont, "err_cont", "workload"),
                           by_placement=group_mape(with_cont, "err_cont", "placement"),
                           worst={"workload": worst["workload"], "placement": worst["placement"],
                                  "r": worst["r"], "k": worst["k"], "source": worst["source"],
                                  "err_pct": round(100 * worst["err_cont"], 3)} if worst else None),
        "base_same_records": dict(acc_base_sub, by_workload=group_mape(with_cont, "err_base", "workload"),
                                  by_placement=group_mape(with_cont, "err_base", "placement"),
                                  worst={"workload": worst_base["workload"], "placement": worst_base["placement"],
                                         "r": worst_base["r"], "k": worst_base["k"],
                                         "source": worst_base["source"],
                                         "err_pct": round(100 * worst_base["err_base"], 3)}
                                  if worst_base else None),
        "base_all_records": dict(acc_base_all, by_workload=group_mape(all_rows, "err_base", "workload"),
                                 by_placement=group_mape(all_rows, "err_base", "placement")),
    }
    if not with_cont:
        note("no stream record has per-unit c_conc in the same file: contention accuracy not computed")

    # regimes
    regimes = {}
    for wl in sorted({r["workload"] for r in all_rows}):
        vals = [r["f_over_c"] for r in all_rows if r["workload"] == wl]
        regimes[wl] = {"mean_f_over_c": round(mean(vals), 4), "regime": regime_label(mean(vals)),
                       "n": len(vals)}

    rank = ranking(all_rows)
    grid = grid_table(all_rows)
    sweeps = sweep_table(all_rows)

    # max measured cf/balanced
    maxcfb = max((g for g in grid if "cf_over_balanced" in g),
                 key=lambda g: g["cf_over_balanced"]["measured"], default=None)
    max_info = None
    if maxcfb:
        max_info = {"ratio": round(maxcfb["cf_over_balanced"]["measured"], 4), "workload": maxcfb["workload"],
                    "r": maxcfb["r"], "k": maxcfb["k"], "source": maxcfb["source"]}

    # copy bandwidth
    bw = [v for _, v in h2d_rates]
    bandwidth = {
        "mean_GBps": round(mean(bw), 3) if bw else None,
        "median_GBps": round(statistics.median(bw), 3) if bw else None,
        "min_GBps": round(min(bw), 3) if bw else None,
        "max_GBps": round(max(bw), 3) if bw else None,
        "n": len(bw),
        "per_file_mean_GBps": {s: round(mean([v for ss, v in h2d_rates if ss == s]), 3)
                               for s in sorted({s for s, _ in h2d_rates})},
    }

    clocks = {wl: {"sm_clock_mean_MHz": round(mean(v["sm"]), 1) if v["sm"] else None,
                   "power_mean_W": round(mean(v["pw"]), 2) if v["pw"] else None,
                   "n": len(v["sm"]) or len(v["pw"])} for wl, v in sorted(clock_power.items())}

    summary = {
        "inputs": {"files": [os.path.basename(f) for f in files], "include_quick": args.include_quick,
                   "record_kinds": dict(kinds_total), "stream_rows": len(all_rows),
                   "stream_rows_with_contention": len(with_cont),
                   "truncated_lines_dropped": stats["truncated_lines"],
                   "skipped_stream_records": stats["no_c_or_f"]},
        "devices": devices,
        "accuracy": accuracy,
        "regimes": regimes,
        "max_measured_cf_over_balanced": max_info,
        "grid": grid,
        "ranking": rank,
        "slot_sweep": sweeps,
        "copy_bandwidth": bandwidth,
        "clock_power_by_workload": clocks,
        "skipped": SKIPPED,
        "notes": NOTES,
    }
    with open(os.path.join(SUMMARY, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print("wrote", os.path.join(SUMMARY, "summary.json"))

    # numbers
    macros = [
        ("AccMape", fmt(acc_cont["mape_pct"], 1)),
        ("AccMapeBase", fmt(acc_base_sub["mape_pct"], 1)),
        ("AccMapeBaseAll", fmt(acc_base_all["mape_pct"], 1)),
        ("AccMax", fmt(acc_cont["max_abs_err_pct"], 1)),
        ("AccMaxBase", fmt(acc_base_sub["max_abs_err_pct"], 1)),
        ("AccN", str(acc_cont["n"])),
        ("AccNAll", str(len(all_rows))),
        ("AccRankPct", "n/a" if rank["contention_agree_pct"] is None
         else str(int(round(rank["contention_agree_pct"])))),
        ("AccRankBasePct", "n/a" if rank["base_agree_pct_paired_pairs"] is None
         else str(int(round(rank["base_agree_pct_paired_pairs"])))),
        ("AccRankPairs", str(rank["pairs_over_threshold"])),
        ("HwBw", fmt(bandwidth["mean_GBps"], 1)),
    ]
    if max_info:
        macros += [
            ("MeasPenaltyMax", f"{max_info['ratio']:.2f}"),
            ("MeasPenaltyMaxR", str(max_info["r"])),
            ("MeasPenaltyMaxK", str(max_info["k"])),
            ("MeasPenaltyMaxWorkload", max_info["workload"].replace(":", " ")),
        ]
    else:
        macros.append(("MeasPenaltyMax", "n/a"))
    write_numbers(macros)

    # figures
    plt = setup_matplotlib()
    fig_accuracy(plt, all_rows)
    fig_penalty(plt, all_rows, grid, ctx_by_source)
    fig_timeline(plt, grid, all_rows)
    fig_slots(plt, sweeps)

    print("\n--- skipped:", len(SKIPPED))
    for s in SKIPPED:
        print("  ", s)
    print("--- notes:", len(NOTES))
    for s in NOTES:
        print("  ", s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
