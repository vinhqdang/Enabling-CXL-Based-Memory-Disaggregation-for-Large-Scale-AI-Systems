#!/usr/bin/env python3
"""Compare measured T4 weight-streaming periods with the placement model.

Usage:
    python3 paper/analysis/analyze.py [--raw-dir DIR] [--out-root DIR]
                                      [--nonuniform-iters N] [--nonuniform-seeds N]

Inputs (groups):
    main    results/raw/t4_full_*.log          locked clocks
    capped  results/raw/t4_power_capped_*.log  clock fell below the lock (power cap)
    quick   results/raw/t4_quick.log           unlocked, no clock samples (reported separately)

Only REC lines are parsed; truncated JSON lines are counted and dropped. Outputs go to
results/summary/ (stream_table.csv, summary.json), paper/numbers.tex and paper/figs/.
Missing data skips the affected output and is reported on stdout.
"""
import argparse
import csv
import glob
import json
import math
import os
import statistics
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

from placement import (  # noqa: E402
    balanced,
    contiguous_first,
    cycle_time,
    cycle_time_contention,
    local_search,
    schedule_contention,
)

PLACEMENTS_GRID = ["contiguous_first", "contiguous_last", "balanced", "random0", "random1"]
RANK_THRESHOLD = 0.02
BUDGETS_LAYERS = list(range(6, 25, 2))
KS_NONUNIFORM = (2, 3)
MARKERS = {"contiguous_first": "o", "contiguous_last": "s", "balanced": "^", "random0": "D",
           "random1": "v", "none": "x", "all_resident": "P"}
# Okabe-Ito hues (CVD-checked with scripts/validate_palette.js).
COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9"]
SWEEP_SOURCE = "t4_full_Sweep.log"
REPEAT_BASE = "t4_full_B.log"
REPEAT_CHECK_SOURCE = "t4_repeat_B768.log"
MEDIAN_BUDGET = 15.0  # median of BUDGETS_LAYERS (14 and 16 bracket it; periods interpolated)
SKIPPED, NOTES = [], []


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


def avg_map(d):
    return {u: mean(v) for u, v in d.items() if v}


def regime_label(x):
    if x < 1.0:
        return "compute-bound (f < c)"
    if x < 2.0:
        return "band (c <= f < 2c)"
    return "transfer-bound (f >= 2c)"


# ---------------------------------------------------------------- parsing

def read_records(path, stats):
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


def build_context(path, group, recs):
    ctx = {"source": os.path.basename(path), "group": group, "models": {},
           "f": defaultdict(lambda: defaultdict(list)), "c": defaultdict(lambda: defaultdict(list)),
           "c_conc": defaultdict(lambda: defaultdict(list)), "clock_c": defaultdict(list),
           "h2d": [], "stream": [], "kinds": Counter(), "device": None}
    for r in recs:
        kind = r.get("kind", r.get("type"))
        ctx["kinds"][str(kind)] += 1
        if kind == "device" and ctx["device"] is None:
            ctx["device"] = {k: r.get(k) for k in ("name", "cuda", "torch", "capability", "pcie_max")}
            ctx["device"]["lock_mhz"] = (r.get("clocks") or {}).get("target_mhz")
        elif kind == "model":
            ctx["models"][r["model"]] = {"L": r.get("L"), "N": r.get("N"), "layer_bytes": r.get("layer_bytes"),
                                         "head_bytes": r.get("head_bytes") or 0,
                                         "head": bool(r.get("head"))}
        elif kind == "h2d":
            ctx["f"][r["model"]][r["unit_index"]].append(r["f_ms"])
            if r.get("bytes"):
                ctx["h2d"].append((r["bytes"], r["f_ms"]))
        elif kind == "compute":
            key = (r["model"], r["workload"])
            ctx["c"][key][r["unit_index"]].append(r["c_ms"])
            if r.get("sm_clock_mean") is not None:
                ctx["clock_c"][key].append(r["sm_clock_mean"])
        elif kind == "compute_conc":
            ctx["c_conc"][(r["model"], r["workload"])][r["unit_index"]].append(r["c_conc_ms"])
        elif kind == "stream":
            ctx["stream"].append(r)
    return ctx


def nunits(ctx, model):
    m = ctx["models"].get(model, {})
    return m.get("N") or m.get("L") or 0


def unit_maps(ctx, model, wl):
    """Per-unit f, c (isolated), c_conc and clock for one (file, model, workload), or None."""
    N = nunits(ctx, model)
    fmap = avg_map(ctx["f"].get(model, {}))
    cmap = avg_map(ctx["c"].get((model, wl), {}))
    if N == 0 or any(u not in fmap for u in range(N)) or any(u not in cmap for u in range(N)):
        return None
    ccmap = avg_map(ctx["c_conc"].get((model, wl), {}))
    cc = [ccmap[u] for u in range(N)] if all(u in ccmap for u in range(N)) else None
    return {"N": N, "f": [fmap[u] for u in range(N)], "c": [cmap[u] for u in range(N)],
            "cc": cc, "clock": mean(ctx["clock_c"].get((model, wl), []))}


def calibration(ctx, model, wl, um):
    """Scale s = measured all-resident period / sum of isolated c_i for this (file, model, workload)."""
    ar = [r["period_ms"] for r in ctx["stream"]
          if r.get("exp") == "allres" and r["model"] == model and r["workload"] == wl]
    if not ar:
        return None
    allres = mean(ar)
    return {"s": allres / sum(um["c"]), "allres_ms": allres}


# ---------------------------------------------------------------- rows

def stream_row(ctx, r):
    model, wl = r["model"], r["workload"]
    um = unit_maps(ctx, model, wl)
    tag = f"{ctx['source']}: {wl} {r.get('placement')} r={r.get('r')} k={r.get('k')}"
    if um is None:
        skip(f"{tag}: missing per-unit f or c")
        return None
    N, f, c, cc = um["N"], um["f"], um["c"], um["cc"]
    res = set(r["resident"])
    k = r["k"]
    streamed = [u for u in range(N) if u not in res]
    if streamed and k < 1:
        skip(f"{tag}: k={k} with streamed units")
        return None
    sum_f = sum(f[u] for u in streamed)
    cal = calibration(ctx, model, wl, um)
    s = cal["s"] if cal else None
    minfo = ctx["models"].get(model, {})
    is_head = bool(minfo.get("head")) and N == (minfo.get("L") or N) + 1
    row = {
        "group": ctx["group"], "source": ctx["source"], "model": model, "workload": wl, "head": is_head,
        "exp": r.get("exp"), "placement": r.get("placement"), "r": r.get("r"), "k": k,
        "m": len(streamed), "N": N, "period_ms": r["period_ms"],
        "period_median_ms": r.get("period_median_ms"), "host_enqueue_ms": r.get("host_enqueue_ms"),
        "sm_clock_stream": r.get("sm_clock_mean"), "sm_clock_compute": um["clock"],
        "s": s, "allres_ms": cal["allres_ms"] if cal else None,
        "sum_c_ms": sum(c), "sum_f_streamed_ms": sum_f, "f_over_c": mean(f) / mean(c),
        "lb_raw_ms": max(sum(c), sum_f),
        "has_timeline": bool(r.get("timeline_compute")),
        "timeline_compute": r.get("timeline_compute") or [], "timeline_copy": r.get("timeline_copy") or [],
        "resident": sorted(res),
    }
    measured = r["period_ms"]
    row["pred_base_raw_ms"] = cycle_time(c, f, res, k)
    row["pred_cont_raw_ms"] = cycle_time_contention(c, cc, f, res, k) if cc else None
    if s is not None:
        cs = [s * x for x in c]
        row["pred_cal_ms"] = cycle_time(cs, f, res, k)
        row["pred_cal_cont_ms"] = cycle_time_contention(cs, [s * x for x in cc], f, res, k) if cc else None
        row["lb_cal_ms"] = max(sum(cs), sum_f)
    else:
        row["pred_cal_ms"] = row["pred_cal_cont_ms"] = row["lb_cal_ms"] = None
    xc = None
    if um["clock"] and r.get("sm_clock_mean"):
        xc = um["clock"] / r["sm_clock_mean"]
        cck = [x * xc for x in c]
        row["pred_clk_ms"] = cycle_time(cck, f, res, k)
    else:
        row["pred_clk_ms"] = None
    row["clock_ratio"] = xc
    for key, pk in (("err_base_raw", "pred_base_raw_ms"), ("err_cont_raw", "pred_cont_raw_ms"),
                    ("err_cal", "pred_cal_ms"), ("err_cal_cont", "pred_cal_cont_ms"),
                    ("err_clk", "pred_clk_ms")):
        row[key] = None if row[pk] is None else measured / row[pk] - 1.0
    return row


CSV_FIELDS = ["group", "source", "model", "workload", "exp", "placement", "r", "k", "m", "N", "period_ms",
              "period_median_ms", "host_enqueue_ms", "s", "allres_ms", "sum_c_ms", "sum_f_streamed_ms",
              "f_over_c", "lb_raw_ms", "lb_cal_ms", "pred_base_raw_ms", "pred_cont_raw_ms", "pred_cal_ms",
              "pred_cal_cont_ms", "pred_clk_ms", "clock_ratio", "err_base_raw", "err_cont_raw", "err_cal",
              "err_cal_cont", "err_clk", "sm_clock_stream", "sm_clock_compute", "has_timeline"]


def evaluated(rows):
    return [r for r in rows if r["exp"] != "allres"]


def err_block(rows, key):
    """MAPE and max |err| (percent) over rows with a value for key."""
    vals = [(abs(r[key]), r) for r in rows if r.get(key) is not None]
    if not vals:
        return {"n": 0, "mape_pct": None, "max_abs_err_pct": None, "worst": None}
    worst = max(vals, key=lambda x: x[0])[1]
    return {"n": len(vals), "mape_pct": round(100 * mean([v for v, _ in vals]), 3),
            "max_abs_err_pct": round(100 * abs(worst[key]), 3),
            "worst": {"workload": worst["workload"], "model": worst["model"], "placement": worst["placement"],
                      "r": worst["r"], "k": worst["k"], "source": worst["source"],
                      "err_pct": round(100 * worst[key], 3)}}


def err_by(rows, key, field):
    return {name: {kk: vv for kk, vv in err_block([r for r in rows if r[field] == name], key).items()
                   if kk != "worst"}
            for name in sorted({r[field] for r in rows})}


def ranking(rows, key_base, key_cont=None):
    """Pairwise ordering agreement among placements sharing (source, model, workload, r, k)."""
    groups = defaultdict(dict)
    for r in rows:
        if r["placement"] in PLACEMENTS_GRID and r["exp"] in ("grid", "sweep"):
            groups[(r["source"], r["model"], r["workload"], r["r"], r["k"])][r["placement"]] = r
    tot = agree = 0
    agree_c = tot_c = 0
    for g in groups.values():
        names = sorted(g)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = g[names[i]], g[names[j]]
                ma, mb = a["period_ms"], b["period_ms"]
                if abs(ma - mb) / min(ma, mb) <= RANK_THRESHOLD:
                    continue
                ms = (ma > mb) - (ma < mb)
                if a.get(key_base) is not None and b.get(key_base) is not None:
                    tot += 1
                    pa, pb = a[key_base], b[key_base]
                    agree += int(((pa > pb) - (pa < pb)) == ms)
                if key_cont and a.get(key_cont) is not None and b.get(key_cont) is not None:
                    tot_c += 1
                    pa, pb = a[key_cont], b[key_cont]
                    agree_c += int(((pa > pb) - (pa < pb)) == ms)
    return {"threshold_rel": RANK_THRESHOLD, "pairs": tot, "agree": agree,
            "agree_pct": pct(agree / tot) if tot else None,
            "pairs_cont": tot_c, "agree_cont_pct": pct(agree_c / tot_c) if tot_c else None}


def penalty_table(rows):
    groups = defaultdict(dict)
    for r in rows:
        if r["exp"] == "grid" and r["placement"] in ("contiguous_first", "balanced"):
            groups[(r["group"], r["source"], r["model"], r["workload"], r["r"], r["k"])][r["placement"]] = r
    out = []
    for key, g in sorted(groups.items()):
        if "contiguous_first" not in g or "balanced" not in g:
            continue
        cf, bal = g["contiguous_first"], g["balanced"]
        ent = {"group": key[0], "source": key[1], "model": key[2], "workload": key[3], "r": key[4], "k": key[5],
               "head": cf["head"],
               "measured_cf_ms": cf["period_ms"], "measured_bal_ms": bal["period_ms"],
               "measured_ratio": cf["period_ms"] / bal["period_ms"],
               "pred_cal_cf_ms": cf["pred_cal_ms"], "pred_cal_bal_ms": bal["pred_cal_ms"],
               "pred_cal_ratio": None if cf["pred_cal_ms"] is None or bal["pred_cal_ms"] is None
               else cf["pred_cal_ms"] / bal["pred_cal_ms"],
               "pred_cal_cont_ratio": None if cf["pred_cal_cont_ms"] is None or bal["pred_cal_cont_ms"] is None
               else cf["pred_cal_cont_ms"] / bal["pred_cal_cont_ms"],
               "lb_cal_cf_ms": cf["lb_cal_ms"], "lb_cal_bal_ms": bal["lb_cal_ms"],
               "has_timeline": cf["has_timeline"] and bal["has_timeline"]}
        out.append(ent)
    return out


def sweep_table(rows):
    groups = defaultdict(list)
    for r in rows:
        if r["exp"] == "sweep" and r["placement"] in ("balanced", "contiguous_first"):
            groups[(r["source"], r["model"], r["workload"], r["r"])].append(r)
    out = []
    for key, lst in sorted(groups.items()):
        for p in ("balanced", "contiguous_first"):
            pts = sorted([x for x in lst if x["placement"] == p], key=lambda x: x["k"])
            if pts:
                out.append({"source": key[0], "model": key[1], "workload": key[2], "r": key[3],
                            "placement": p,
                            "points": [{"k": x["k"], "measured_ms": x["period_ms"],
                                        "pred_cal_ms": x["pred_cal_ms"], "pred_cal_cont_ms": x["pred_cal_cont_ms"]}
                                       for x in pts]})
    return out


# ---------------------------------------------------------------- dense model curves

def dense_curves(ctx, model, wl, k, placement):
    """Calibrated model predictions over every layer-resident count r (head, if any, streamed)."""
    um = unit_maps(ctx, model, wl)
    cal = calibration(ctx, model, wl, um) if um else None
    if um is None or cal is None:
        return None
    L = ctx["models"][model]["L"] or um["N"]
    s = cal["s"]
    cs = [s * x for x in um["c"]]
    f = um["f"]
    has_cc = um["cc"] is not None
    cc = [s * x for x in um["cc"]] if has_cc else None
    xs, base, cont = [], [], []
    for r in range(1, L):
        res = balanced(L, r) if placement == "balanced" else contiguous_first(L, r)
        lb = max(sum(cs), sum(f[u] for u in range(um["N"]) if u not in res))
        xs.append(r)
        base.append(cycle_time(cs, f, res, k) / lb)
        cont.append(cycle_time_contention(cs, cc, f, res, k) / lb if has_cc else None)
    return xs, base, cont


# ---------------------------------------------------------------- non-uniform study

def nonuniform_study(ctx, model, wl, iters, seeds):
    um = unit_maps(ctx, model, wl)
    if um is None:
        return None, "missing per-unit c or f"
    cal = calibration(ctx, model, wl, um)
    calibrated = cal is not None
    if not calibrated:
        note(f"{ctx['source']}: no all_resident record for {wl}; non-uniform study uses s = 1")
        cal = {"s": 1.0, "allres_ms": None}
    minfo = ctx["models"][model]
    L = minfo["L"]
    N = um["N"]
    head = minfo["head"] and N == L + 1
    lb = minfo["layer_bytes"]
    hb = minfo["head_bytes"] if head else 0
    if not lb:
        return None, "layer_bytes missing"
    sizes = [lb] * N if not head else [lb] * L + [hb]
    s = cal["s"]
    cs = [s * x for x in um["c"]]
    f = um["f"]
    rows = []
    for B in BUDGETS_LAYERS:
        budget = B * lb
        for k in KS_NONUNIFORM:
            row = {"budget_layers": B, "k": k}
            slot_all = max(sizes)
            # (a) contiguous layers, head streamed
            ra = min(L, math.floor((budget - k * slot_all) / lb)) if head else \
                min(L, math.floor((budget - k * lb) / lb))
            res_a = res_b = res_c = None
            if ra >= 0:
                res_a = contiguous_first(L, ra)
                row["a"] = {"feasible": True, "r": ra, "period_ms": cycle_time(cs, f, res_a, k)}
            else:
                row["a"] = {"feasible": False}
            # (b) balanced layers, head streamed
            if ra >= 0:
                res_b = balanced(L, ra) if ra > 0 else set()
                row["b"] = {"feasible": True, "r": ra, "period_ms": cycle_time(cs, f, res_b, k)}
            else:
                row["b"] = {"feasible": False}
            # (c) head resident + balanced layers with the remaining budget
            if head:
                rc = min(L - 1, math.floor((budget - hb - k * lb) / lb))
                if rc >= 0:
                    res_c = (balanced(L, rc) if rc > 0 else set()) | {N - 1}
                    row["c"] = {"feasible": True, "r": rc, "period_ms": cycle_time(cs, f, res_c, k)}
                else:
                    row["c"] = {"feasible": False}
            else:
                row["c"] = {"feasible": False, "note": "no head unit"}
            # (d) local search over all units, started from the best feasible heuristic (a, b or c)
            starts = [res for res, key in ((res_c if head and row["c"].get("feasible") else None, "c"),
                                           (res_b if row["b"].get("feasible") else None, "b"),
                                           (res_a if row["a"].get("feasible") else None, "a"))
                      if res is not None]
            best = None
            if starts:
                start_set = min(starts, key=lambda st: cycle_time(cs, f, st, k))
                for seed in seeds:
                    p, res_d = local_search(cs, f, sizes, budget, k, start=set(start_set), iters=iters, seed=seed)
                    if best is None or p < best[0]:
                        best = (p, res_d)
            if best is None:
                row["d"] = {"feasible": False}
            else:
                row["d"] = {"feasible": True, "period_ms": best[0], "residents": sorted(best[1]),
                            "r": len(best[1])}
            rows.append(row)
            print(f"  nonuniform B={B} k={k}: " + " ".join(
                f"{x}={row[x]['period_ms']:.1f}" if row[x].get("feasible") else f"{x}=infeasible"
                for x in ("a", "b", "c", "d")))
    info = {"source": ctx["source"], "model": model, "workload": wl, "L": L, "N": N, "head": head,
            "calibrated_from_allres": calibrated,
            "layer_bytes": lb, "head_bytes": hb, "head_over_layer": (hb / lb) if head else None,
            "scale_s": s, "seeds": list(seeds), "iters": iters, "rows": rows}
    return info, None


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


ACC_REGIME_COLORS = {"prefill, f/c > 1": "#D55E00", "prefill, f/c <= 1": "#009E73", "decode": "#CC79A7"}
ACC_PLACEMENTS = [("contiguous_first", "o", "contiguous_first"), ("contiguous_last", "s", "contiguous_last"),
                  ("balanced", "^", "balanced"), ("random0", "D", "random0, random1"),
                  ("none", "x", "none (all streamed)")]


def accuracy_regime(r):
    if r["workload"].startswith("dec:"):
        return "decode"
    return "prefill, f/c > 1" if float(r["f_over_c"]) > 1.0 else "prefill, f/c <= 1"


def fig_accuracy(plt, figs, rows):
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter, NullLocator

    marker_of = {p: (m, lab) for p, m, lab in ACC_PLACEMENTS}
    pts = []
    for r in rows:
        if r.get("pred_cal_ms") is None or r.get("period_ms") is None:
            continue
        key = "random0" if r["placement"] == "random1" else r["placement"]
        if key in marker_of:
            pts.append((r, accuracy_regime(r), key))
    if not pts:
        skip("fig_accuracy.pdf: no evaluated main record with a calibrated prediction")
        return
    fig = plt.figure(figsize=(7.0, 2.6))
    ax_a = fig.add_axes([0.085, 0.19, 0.31, 0.72])
    ax_b = fig.add_axes([0.53, 0.19, 0.22, 0.72])

    lo = min(min(r["period_ms"], r["pred_cal_ms"]) for r, _, _ in pts)
    hi = max(max(r["period_ms"], r["pred_cal_ms"]) for r, _, _ in pts)
    for reg, col in ACC_REGIME_COLORS.items():
        for key, (m, _) in marker_of.items():
            sel = [r for r, g, p in pts if g == reg and p == key]
            if not sel:
                continue
            ax_a.scatter([r["pred_cal_ms"] for r in sel], [r["period_ms"] for r in sel], s=10, marker=m,
                         color=col, alpha=0.8, linewidths=0.8 if m == "x" else 0, zorder=3)
    ax_a.plot([lo * 0.9, hi * 1.1], [lo * 0.9, hi * 1.1], color="black", linewidth=0.6, linestyle="--",
              zorder=2)
    ax_a.set_xscale("log")
    ax_a.set_yscale("log")
    ax_a.set_xlim(lo * 0.9, hi * 1.1)
    ax_a.set_ylim(lo * 0.9, hi * 1.1)
    for axis in (ax_a.xaxis, ax_a.yaxis):
        axis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        axis.set_minor_locator(NullLocator())
        axis.set_minor_formatter(NullFormatter())
    ax_a.set_xlabel("Predicted period, calibrated (ms)")
    ax_a.set_ylabel("Measured period (ms)")
    ax_a.text(0.02, 0.98, "(a)", transform=ax_a.transAxes, va="top", ha="left", weight="bold")

    edges = [-6 + 0.5 * i for i in range(25)]
    mae = {}
    for reg, col in ACC_REGIME_COLORS.items():
        errs = [100.0 * (r["period_ms"] / r["pred_cal_ms"] - 1.0) for r, g, _ in pts if g == reg]
        if not errs:
            continue
        inside = [e for e in errs if -6 <= e <= 6]
        if len(inside) < len(errs):
            note(f"fig_accuracy.pdf: {len(errs) - len(inside)} {reg} errors outside +-6 % not in histogram")
        mae[reg] = sum(abs(e) for e in errs) / len(errs)
        ax_b.hist(inside, bins=edges, histtype="step", color=col, linewidth=1.1)
    ax_b.axvline(0.0, ymax=0.6, color="black", linewidth=0.6, linestyle="--")
    ax_b.set_xlim(-6, 6)
    ax_b.set_xlabel("Relative error, measured / predicted - 1 (%)")
    ax_b.set_yscale("log")
    ax_b.set_ylim(0.7, 10000)
    ax_b.set_ylabel("Configurations (log scale)")
    ax_b.text(0.02, 0.98, "(b)", transform=ax_b.transAxes, va="top", ha="left", weight="bold")
    short = {"prefill, f/c > 1": "prefill f/c > 1", "prefill, f/c <= 1": "prefill f/c <= 1", "decode": "decode"}
    ax_b.text(0.03, 0.87, "mean absolute error", transform=ax_b.transAxes, va="top", ha="left", fontsize=6)
    for i, (reg, col) in enumerate(ACC_REGIME_COLORS.items()):
        if reg in mae:
            ax_b.text(0.03, 0.79 - 0.08 * i, f"{short[reg]}: {mae[reg]:.2f} %", transform=ax_b.transAxes,
                      va="top", ha="left", color=col, fontsize=6)

    handles = [Line2D([], [], ls="none", marker="o", markersize=4, color=col, label=reg)
               for reg, col in ACC_REGIME_COLORS.items()]
    handles += [Line2D([], [], ls="none", marker=m, markersize=4, color="black", markerfacecolor="none"
                       if m == "x" else "black", label=lab) for _, m, lab in ACC_PLACEMENTS]
    fig.legend(handles=handles, labels=[h.get_label() for h in handles], loc="center left",
               bbox_to_anchor=(0.79, 0.52), frameon=False, fontsize=6, handletextpad=0.4,
               labelspacing=0.6)
    out = os.path.join(figs, "fig_accuracy.pdf")
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out, f"({len(pts)} points)")


def fig_penalty(plt, figs, ctx_by_source, ptable):
    ents = [e for e in ptable if e["group"] == "main" and e["workload"].startswith("pre:")
            and e["model"] == "llama3.2-3b" and not e["head"]]
    # one panel per prefill T; when a T appears in two files, use the file with more grid points
    best = {}
    for e in ents:
        best.setdefault(e["workload"], {}).setdefault(e["source"], []).append(e)
    keys = []
    for wl, per_src in best.items():
        src = max(per_src, key=lambda x: len(per_src[x]))
        keys.append((src, "llama3.2-3b", wl))
    keys.sort(key=lambda x: prefill_t(x[2]))
    ents = [e for e in ents if (e["source"], e["model"], e["workload"]) in set(keys)]
    if not keys:
        skip("fig_penalty.pdf: no prefill workload with contiguous_first and balanced in the main group")
        return
    ncol = 3 if len(keys) > 3 else len(keys)
    nrow = (len(keys) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.0, 2.2 * nrow + 0.6), squeeze=False)
    for ax in axes.flat[len(keys):]:
        ax.set_visible(False)
    for ax, (src, model, wl) in zip(axes.flat, keys):
        ctx = ctx_by_source[src]
        sub = [e for e in ents if (e["source"], e["model"], e["workload"]) == (src, model, wl)]
        ks = sorted({e["k"] for e in sub})
        for placement, col in (("contiguous_first", COLORS[0]), ("balanced", COLORS[1])):
            for k in ks:
                cur = dense_curves(ctx, model, wl, k, placement)
                if cur is None:
                    continue
                xs, base, cont = cur
                ax.plot(xs, base, color=col, linewidth=0.7, linestyle=":" if k == ks[0] else "-.")
                if cont[0] is not None:
                    ax.plot(xs, cont, color=col, linewidth=0.9, linestyle="-" if k == ks[-1] else "--")
            for e in sub:
                y = e["measured_cf_ms"] if placement == "contiguous_first" else e["measured_bal_ms"]
                lb = e["lb_cal_cf_ms"] if placement == "contiguous_first" else e["lb_cal_bal_ms"]
                if lb is None:
                    continue
                ax.scatter([e["r"]], [y / lb], s=14, marker="o" if e["k"] == ks[0] else "s",
                           color=col, edgecolor="black", linewidth=0.3, zorder=3)
        ax.axhline(1.0, color="black", linewidth=0.5, linestyle=":")
        ax.set_xlabel("Resident units r")
        ax.set_ylabel("Period / lower bound")
        ax.text(0.02, 0.96, f"prefill T = {prefill_t(wl)}", transform=ax.transAxes, va="top", fontsize=6)
    ytop = max(ax.get_ylim()[1] for ax in axes.flat[:len(keys)])
    for ax in axes.flat[:len(keys)]:
        ax.set_ylim(0.97, ytop)
    handles = [plt.Line2D([], [], color=COLORS[0], label="contiguous first"),
               plt.Line2D([], [], color=COLORS[1], label="balanced"),
               plt.Line2D([], [], color="black", linestyle="-", label="calibrated model, contention"),
               plt.Line2D([], [], color="black", linestyle=":", label="calibrated model, base"),
               plt.Line2D([], [], color="black", marker="o", linestyle="", label="measured, smaller k"),
               plt.Line2D([], [], color="black", marker="s", linestyle="", label="measured, larger k")]
    fig.legend(handles=handles, frameon=False, fontsize=6.5, loc="lower center", ncol=3)
    fig.tight_layout(rect=[0, 0.1, 1, 1])
    out = os.path.join(figs, "fig_penalty.pdf")
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out, f"({len(keys)} panels)")


def fig_timeline(plt, figs, ptable, rows):
    cands = [e for e in ptable if e["group"] == "main" and e["k"] == 3]
    if not cands:
        skip("fig_timeline.pdf: no k=3 configuration in the main group with both placements")
        return
    cands.sort(key=lambda e: -e["measured_ratio"])
    chosen = next((e for e in cands if e["has_timeline"]), None)
    if chosen is None:
        skip("fig_timeline.pdf: no k=3 configuration with timelines for both placements")
        return
    if chosen is not cands[0]:
        note(f"timeline: largest ratio {cands[0]['workload']} r={cands[0]['r']} has no timeline; using next best")
    print(f"timeline configuration: {chosen['model']} {chosen['workload']} r={chosen['r']} k=3 "
          f"ratio={chosen['measured_ratio']:.3f} source={chosen['source']}")
    recs = {r["placement"]: r for r in rows if r["exp"] == "grid" and r["source"] == chosen["source"]
            and r["model"] == chosen["model"] and r["workload"] == chosen["workload"]
            and r["r"] == chosen["r"] and r["k"] == 3}
    fig, axes = plt.subplots(2, 1, figsize=(3.4, 3.7), sharex=True)
    xmin, xmax = 0.0, 0.0
    for ax, p in zip(axes, ("contiguous_first", "balanced")):
        rec = recs[p]
        for u, s, e in rec["timeline_compute"]:
            ax.broken_barh([(s, e - s)], (1.2, 0.8),
                           facecolors=COLORS[0] if u in rec["resident"] else COLORS[1], edgecolor="none")
            xmin, xmax = min(xmin, s), max(xmax, e)
        for u, s, e in rec["timeline_copy"]:
            ax.broken_barh([(s, e - s)], (0.0, 0.8), facecolors="#999999", edgecolor="none")
            xmin, xmax = min(xmin, s), max(xmax, e)
        ax.axvline(0.0, color="black", linewidth=0.5)
        ax.axvline(rec["period_ms"], color="black", linewidth=0.5, linestyle=":")
        ax.set_yticks([0.4, 1.6])
        ax.set_yticklabels(["copy", "compute"])
        ax.set_ylim(-0.1, 2.2)
        ax.text(0.01, 0.98, f"{p}, period {rec['period_ms']:.1f} ms", transform=ax.transAxes, va="top",
                fontsize=6)
    axes[1].set_xlabel("Time from iteration start (ms)")
    axes[1].set_xlim(xmin * 1.02, xmax * 1.02)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[0], label="resident unit compute"),
               plt.Rectangle((0, 0), 1, 1, color=COLORS[1], label="streamed unit compute"),
               plt.Rectangle((0, 0), 1, 1, color="#999999", label="host-to-GPU copy")]
    fig.legend(handles=handles, frameon=False, fontsize=5.5, loc="lower center", ncol=3)
    fig.tight_layout(rect=[0, 0.1, 1, 1])
    out = os.path.join(figs, "fig_timeline.pdf")
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


def fig_slots(plt, figs, sweeps):
    groups = defaultdict(dict)
    for s in sweeps:
        groups[(s["source"], s["model"], s["workload"], s["r"])][s["placement"]] = s["points"]
    if not groups:
        skip("fig_slots.pdf: no slot sweep records (exp == 'sweep') in the main group")
        return
    n = len(groups)
    fig, axes = plt.subplots(n, 1, figsize=(3.4, 2.3 * n + 1.0), squeeze=False)
    cols = {"balanced": COLORS[1], "contiguous_first": COLORS[0]}
    for ax, (key, plc) in zip(axes[:, 0], groups.items()):
        for p in ("contiguous_first", "balanced"):
            pts = plc.get(p, [])
            if not pts:
                continue
            ks = [x["k"] for x in pts]
            ax.scatter(ks, [x["measured_ms"] for x in pts], s=18, marker=MARKERS[p], color=cols[p],
                       edgecolor="black", linewidth=0.3, zorder=3)
            if all(x["pred_cal_ms"] is not None for x in pts):
                ax.plot(ks, [x["pred_cal_ms"] for x in pts], color=cols[p], linestyle=":", linewidth=0.8)
            if all(x["pred_cal_cont_ms"] is not None for x in pts):
                ax.plot(ks, [x["pred_cal_cont_ms"] for x in pts], color=cols[p], linestyle="-", linewidth=0.9)
        ax.set_xlabel("Slots k")
        ax.set_ylabel("Period (ms)")
        ax.text(0.02, 0.96, f"{key[1]}, {key[2].replace(':', ' ')}, r={key[3]}", transform=ax.transAxes,
                va="top", fontsize=6)
    handles = [plt.Line2D([], [], color=cols["contiguous_first"], label="contiguous first"),
               plt.Line2D([], [], color=cols["balanced"], label="balanced"),
               plt.Line2D([], [], color="black", linestyle="-", label="calibrated model, contention"),
               plt.Line2D([], [], color="black", linestyle=":", label="calibrated model, base"),
               plt.Line2D([], [], color="black", marker="o", linestyle="", label="measured")]
    fig.legend(handles=handles, frameon=False, fontsize=5.5, loc="lower center", ncol=2)
    fig.tight_layout(rect=[0, 0.1, 1, 1])
    out = os.path.join(figs, "fig_slots.pdf")
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


def fig_nonuniform(plt, figs, nu):
    if nu is None:
        skip("fig_nonuniform.pdf: no non-uniform (head) study data")
        return
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    names = {"a": ("contiguous layers, head streamed", COLORS[0]),
             "b": ("balanced layers, head streamed", COLORS[1]),
             "c": ("head resident, balanced layers", COLORS[2]),
             "d": ("local search (all units)", COLORS[3])}
    for key, (label, col) in names.items():
        for k, ls in zip(KS_NONUNIFORM, ("-", "--")):
            pts = [(r["budget_layers"], r[key]["period_ms"]) for r in nu["rows"]
                   if r["k"] == k and r[key].get("feasible")]
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], color=col, linestyle=ls,
                        marker="o", markersize=2.5, linewidth=0.9, label=f"{label}, k={k}")
    ax.set_xlabel("Memory budget (layer equivalents)")
    ax.set_ylabel("Calibrated period (ms)")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=5)
    fig.tight_layout(rect=[0, 0.3, 1, 1])
    out = os.path.join(figs, "fig_nonuniform.pdf")
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


# ---------------------------------------------------------------- numbers

def write_numbers(path, macros):
    lines = ["% Generated by paper/analysis/analyze.py. Values are numbers; add % in the text where quoted."]
    for name, val in macros:
        assert name.isalpha(), name
        lines.append(f"\\newcommand{{\\n{name}}}{{{val}}}")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote", path, f"({len(macros)} macros)")


def fmt(x, nd):
    return "n/a" if x is None else f"{x:.{nd}f}"


def fmt_int(x):
    return "n/a" if x is None else str(int(round(x)))


# ---------------------------------------------------------------- additional analyses

def interp_at(nu_rows, cfg, k, B):
    """Period of configuration cfg at budget B, linearly interpolated between grid budgets."""
    pts = sorted((r["budget_layers"], r[cfg]["period_ms"]) for r in nu_rows
                 if r["k"] == k and r[cfg].get("feasible"))
    lo = [p for p in pts if p[0] <= B]
    hi = [p for p in pts if p[0] >= B]
    if not lo or not hi:
        return None
    (b0, y0), (b1, y1) = lo[-1], hi[0]
    return y0 if b1 == b0 else y0 + (y1 - y0) * (B - b0) / (b1 - b0)


def prefill_t(wl):
    return int(wl.split(":")[1]) if wl.startswith("pre:") else None


def repeat_check(rows):
    base = {(r["model"], r["workload"], r["placement"], r["r"], r["k"]): r["period_ms"]
            for r in rows if r["group"] == "main" and r["source"] == REPEAT_BASE and r["exp"] == "grid"}
    diffs = []
    for r in rows:
        if r["group"] == "repeat" and r["exp"] == "grid":
            key = (r["model"], r["workload"], r["placement"], r["r"], r["k"])
            if key in base:
                diffs.append(100 * abs(r["period_ms"] / base[key] - 1))
    return {"base": REPEAT_BASE, "repeat": REPEAT_CHECK_SOURCE, "n": len(diffs),
            "mean_pct": round(mean(diffs), 3) if diffs else None,
            "max_pct": round(max(diffs), 3) if diffs else None}


def regime_table(rows):
    fc = defaultdict(list)
    for r in rows:
        fc[(r["model"], r["workload"], r["head"])].append(r["f_over_c"])
    fc = {k: mean(v) for k, v in fc.items()}

    def regime(r):
        if r["workload"].startswith("dec:"):
            return "decode"
        return ("prefill transfer-bound (f/c > 1)" if fc[(r["model"], r["workload"], r["head"])] > 1
                else "prefill compute-bound (f/c <= 1)")

    out = {}
    for name in ("prefill transfer-bound (f/c > 1)", "prefill compute-bound (f/c <= 1)", "decode"):
        sel = [r for r in rows if regime(r) == name]
        cal_cont = [r for r in sel if r["err_cal_cont"] is not None]
        out[name] = {"n": len(sel), "models": sorted({r["model"] for r in sel}),
                     "calibrated": {k: v for k, v in err_block(sel, "err_cal").items() if k != "worst"},
                     "calibrated_contention": {k: v for k, v in err_block(cal_cont, "err_cal_cont").items()
                                               if k != "worst"},
                     "uncalibrated_base": {k: v for k, v in err_block(sel, "err_base_raw").items() if k != "worst"}}
    return out


def spread_and_worst(rows):
    """Placement spread for compute-bound 3B prefill, and random0 vs balanced worst case."""
    groups = defaultdict(dict)
    for r in rows:
        if r["exp"] == "grid" and r["placement"] in PLACEMENTS_GRID:
            groups[(r["source"], r["model"], r["workload"], r["r"], r["k"])][r["placement"]] = r["period_ms"]
    compute_spread = []
    worst_rb = []
    for key, pl in groups.items():
        if len(pl) >= 2:
            compute_spread.append((100 * (max(pl.values()) / min(pl.values()) - 1), key))
        if "random0" in pl and "balanced" in pl:
            worst_rb.append((100 * (pl["random0"] / pl["balanced"] - 1), key))
    return compute_spread, worst_rb


def read_interference(path):
    """Ratio records of the interference diagnostic, grouped by clock state ('locked' / 'unlocked')."""
    if not os.path.exists(path):
        return None
    out = defaultdict(list)
    with open(path, "rb") as fh:
        for raw in fh:
            if raw.startswith(b"REC "):
                try:
                    r = json.loads(raw[4:].decode("utf-8", "replace"))
                except ValueError:
                    continue
                if r.get("type") == "ratio":
                    out[r.get("clock")].append(r)
    return out


def band_excess(rows, k):
    """Largest measured (or predicted) period over the lower bound for balanced placements at pre:768."""
    sel = [r for r in rows if r["group"] == "main" and r["model"] == "llama3.2-3b" and not r["head"]
           and r["workload"] == "pre:768" and r["placement"] == "balanced" and r["k"] == k
           and r["exp"] == "grid" and r["lb_cal_ms"] is not None and r["pred_cal_ms"] is not None]
    if not sel:
        return None
    meas = max(sel, key=lambda r: r["period_ms"] / r["lb_cal_ms"])
    pred = max(sel, key=lambda r: r["pred_cal_ms"] / r["lb_cal_ms"])
    return {"k": k, "measured_pct": 100 * (meas["period_ms"] / meas["lb_cal_ms"] - 1), "r_at_measured_max": meas["r"],
            "predicted_pct": 100 * (pred["pred_cal_ms"] / pred["lb_cal_ms"] - 1), "r_at_predicted_max": pred["r"],
            "n": len(sel)}


def compute_bound_deviation(rows, compute3):
    """|period / all_resident period - 1| over streaming configurations of compute-bound 3B prefill."""
    vals = [abs(r["period_ms"] / r["allres_ms"] - 1) for r in rows
            if r["group"] == "main" and r["model"] == "llama3.2-3b" and not r["head"] and r["exp"] != "allres"
            and prefill_t(r["workload"]) in compute3 and r["allres_ms"]]
    return 100 * max(vals) if vals else None


def sweep_model_cf(ctx, model, wl, r_res, k_values, tol=0.005):
    """Calibrated model: contiguous_first period at r_res for each k, and the smallest k within tol of the bound."""
    um = unit_maps(ctx, model, wl)
    cal = calibration(ctx, model, wl, um) if um else None
    if um is None or cal is None:
        return None
    cs = [cal["s"] * x for x in um["c"]]
    f = um["f"]
    res = contiguous_first(um["N"], r_res)
    lb = max(sum(cs), sum(f[u] for u in range(um["N"]) if u not in res))
    ratios = {k: cycle_time(cs, f, res, k) / lb for k in k_values}
    ok = [k for k in k_values if ratios[k] - 1 <= tol]
    return {"r": r_res, "ratio_to_bound": {str(k): v for k, v in ratios.items()},
            "smallest_k_within_tol": min(ok) if ok else None, "tol_pct": 100 * tol}


def unlocked_block(unl_rows):
    """Unlocked group: calibrated / clock-corrected errors, clock per workload, penalty and reversals."""
    ev = evaluated(unl_rows)
    cal = [r for r in ev if r["pred_cal_ms"] is not None]
    pen = penalty_table(unl_rows)
    rev = [e for e in pen if e["measured_bal_ms"] > e["measured_cf_ms"] * 1.01]
    clocks = {}
    for r in ev:
        if r["sm_clock_stream"] is not None:
            clocks.setdefault(r["workload"], []).append(r["sm_clock_stream"])
    max_pen = max(pen, key=lambda e: e["measured_ratio"], default=None)
    return {
        "n": len(ev),
        "files": sorted({r["source"] for r in unl_rows}),
        "calibrated": {k: v for k, v in err_block(cal, "err_cal").items() if k != "worst"},
        "uncalibrated_base": {k: v for k, v in err_block(ev, "err_base_raw").items() if k != "worst"},
        "clock_corrected": {k: v for k, v in err_block(ev, "err_clk").items() if k != "worst"},
        "mean_sm_clock_MHz_by_workload": {w: round(mean(v), 1) for w, v in sorted(clocks.items())},
        "penalty_table": pen,
        "penalty_max": max_pen,
        "reversals_over_1pct": len(rev),
        "reversal_entries": rev,
    }


def fig_unlocked(plt, figs, unl_rows):
    grid = [r for r in unl_rows if r["exp"] == "grid" and r["placement"] in ("contiguous_first", "balanced")
            and r["lb_cal_ms"] is not None]
    if not grid:
        skip("fig_unlocked.pdf: no calibrated grid records in the unlocked group")
        return
    wls = sorted({r["workload"] for r in grid}, key=lambda w: prefill_t(w) or 0)
    cols = {"contiguous_first": "#ff7f0e", "balanced": "#1f77b4"}  # streamed orange, resident blue
    dodge = {"contiguous_first": -0.3, "balanced": 0.3}  # offsets keep coinciding markers apart
    ks = sorted({r["k"] for r in grid})
    with plt.rc_context({"font.size": 8, "xtick.labelsize": 7, "ytick.labelsize": 7}):
        fig, axes = plt.subplots(1, len(wls), figsize=(3.4, 2.6), sharey=True, squeeze=False)
        for ax, wl in zip(axes[0], wls):
            sel = [r for r in grid if r["workload"] == wl]
            for p in ("contiguous_first", "balanced"):
                for r in sorted([x for x in sel if x["placement"] == p], key=lambda x: x["r"]):
                    filled = r["k"] == ks[-1]
                    ax.scatter([r["r"] + dodge[p]], [r["period_ms"] / r["lb_cal_ms"]], s=14,
                               marker="o", facecolor=cols[p] if filled else "white",
                               edgecolor=cols[p], linewidth=0.9, zorder=3)
            ax.axhline(1.0, color="black", linewidth=0.5, linestyle=":")
            ax.set_xticks(sorted({r["r"] for r in sel}))
            ax.set_xlim(min(r["r"] for r in sel) - 2, max(r["r"] for r in sel) + 2)
            ax.set_title(f"prefill $T$={prefill_t(wl)}", fontsize=7, pad=3)
        ymax = max(r["period_ms"] / r["lb_cal_ms"] for r in grid)
        axes[0, 0].set_ylim(0.98, ymax + 0.02)
        axes[0, 0].set_ylabel("period / lower bound")
        handles = [
            plt.Line2D([], [], color=cols["contiguous_first"], marker="o", linestyle="", label="contiguous first"),
            plt.Line2D([], [], color=cols["balanced"], marker="o", linestyle="", label="balanced"),
            plt.Line2D([], [], color="#555555", marker="o", markerfacecolor="white", linestyle="", label=f"k={ks[0]}"),
            plt.Line2D([], [], color="#555555", marker="o", linestyle="", label=f"k={ks[-1]}"),
        ]
        fig.legend(handles=handles, frameon=False, fontsize=7, loc="lower center", ncol=2,
                   bbox_to_anchor=(0.5, 0.0))
        fig.supxlabel("resident units $r$", fontsize=8, y=0.23)
        fig.subplots_adjust(left=0.2, right=0.985, bottom=0.36, top=0.9, wspace=0.12)
        out = os.path.join(figs, "fig_unlocked.pdf")
        fig.savefig(out)
        plt.close(fig)
    print("wrote", out, f"({len(wls)} panels)")


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=os.path.join(ROOT, "results", "raw"))
    ap.add_argument("--out-root", default=ROOT)
    ap.add_argument("--nonuniform-iters", type=int, default=3000)
    ap.add_argument("--nonuniform-seeds", type=int, default=3)
    args = ap.parse_args()

    raw = args.raw_dir
    summ = os.path.join(args.out_root, "results", "summary")
    figs = os.path.join(args.out_root, "paper", "figs")
    numbers = os.path.join(args.out_root, "paper", "numbers.tex")
    os.makedirs(summ, exist_ok=True)
    os.makedirs(figs, exist_ok=True)

    groups = {"main": sorted(glob.glob(os.path.join(raw, "t4_full_*.log"))),
              "repeat": sorted(glob.glob(os.path.join(raw, "t4_repeat_*.log"))),
              "unlocked": sorted(glob.glob(os.path.join(raw, "t4_unlocked_*.log"))),
              "capped": sorted(glob.glob(os.path.join(raw, "t4_power_capped_*.log"))),
              "quick": [p for p in [os.path.join(raw, "t4_quick.log")] if os.path.exists(p)]}
    all_files = [f for g in groups.values() for f in g]
    if not all_files:
        print("no input logs found under", raw)
        return 1
    for g, fl in groups.items():
        print(f"group {g}:", [os.path.basename(x) for x in fl] or "(none yet)")

    stats = defaultdict(int)
    ctx_by_source, rows = {}, []
    file_info = {}
    for g, fl in groups.items():
        for path in fl:
            recs = read_records(path, stats)
            if not recs:
                skip(f"{os.path.basename(path)}: no REC lines (run did not start or failed)")
                continue
            ctx = build_context(path, g, recs)
            ctx_by_source[ctx["source"]] = ctx
            file_info[ctx["source"]] = {"group": g, "kinds": dict(ctx["kinds"]),
                                        "models": ctx["models"], "device": ctx["device"]}
            if not ctx["stream"]:
                note(f"{ctx['source']}: no stream records yet (kinds: {dict(ctx['kinds'])})")
            for r in ctx["stream"]:
                if r.get("exp") == "sweep" and os.path.basename(path) != SWEEP_SOURCE:
                    stats["sweep_outside_sweep_file"] += 1
                    continue
                row = stream_row(ctx, r)
                if row is not None:
                    rows.append(row)
    main_rows = [r for r in rows if r["group"] == "main"]
    cap_rows = [r for r in rows if r["group"] == "capped"]
    quick_rows = [r for r in rows if r["group"] == "quick"]
    print(f"stream rows: main {len(main_rows)}, capped {len(cap_rows)}, quick {len(quick_rows)}")

    # CSV (all groups)
    with open(os.path.join(summ, "stream_table.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("wrote", os.path.join(summ, "stream_table.csv"))

    # calibration table
    calib = []
    for src, ctx in ctx_by_source.items():
        for r in ctx["stream"]:
            if r.get("exp") != "allres":
                continue
            um = unit_maps(ctx, r["model"], r["workload"])
            if um is None:
                continue
            cal = calibration(ctx, r["model"], r["workload"], um)
            calib.append({"source": src, "group": ctx["group"], "model": r["model"], "workload": r["workload"],
                          "allres_ms": cal["allres_ms"], "sum_c_ms": sum(um["c"]), "s": cal["s"]})
    calib = sorted({(c["source"], c["workload"]): c for c in calib}.values(), key=lambda c: (c["source"], c["workload"]))
    s_vals = [c["s"] for c in calib if c["group"] == "main"]
    s_by_workload = {}
    for c in calib:
        if c["group"] == "main":
            s_by_workload.setdefault(c["workload"], []).append(c["s"])
    s_by_model = {}
    for c in calib:
        if c["group"] == "main":
            s_by_model.setdefault(c["model"], []).append(c["s"])

    # main group accuracy
    main_eval = [r for r in evaluated(main_rows) if r["pred_cal_ms"] is not None]
    main_cal_cont = [r for r in main_eval if r["pred_cal_cont_ms"] is not None]
    accuracy = {
        "n_evaluated": len(main_eval),
        "n_allres_excluded": len([r for r in main_rows if r["exp"] == "allres"]),
        "calibrated": dict(err_block(main_eval, "err_cal"),
                           by_workload=err_by(main_eval, "err_cal", "workload"),
                           by_placement=err_by(main_eval, "err_cal", "placement"),
                           by_model=err_by(main_eval, "err_cal", "model")),
        "calibrated_contention": dict(err_block(main_cal_cont, "err_cal_cont"),
                                      by_workload=err_by(main_cal_cont, "err_cal_cont", "workload"),
                                      by_placement=err_by(main_cal_cont, "err_cal_cont", "placement")),
        "uncalibrated_base": dict(err_block(main_eval, "err_base_raw"),
                                  by_workload=err_by(main_eval, "err_base_raw", "workload"),
                                  by_placement=err_by(main_eval, "err_base_raw", "placement")),
        "uncalibrated_contention": dict(err_block(main_cal_cont, "err_cont_raw"),
                                        by_workload=err_by(main_cal_cont, "err_cont_raw", "workload")),
    }
    rank = ranking(main_eval, "pred_cal_ms", "pred_cal_cont_ms")
    rank_uncal = ranking(main_eval, "pred_base_raw_ms")
    print("main: calibrated MAPE", accuracy["calibrated"]["mape_pct"], "max", accuracy["calibrated"]["max_abs_err_pct"],
          "| uncal", accuracy["uncalibrated_base"]["mape_pct"], "| n", len(main_eval))

    # head and 8B validation
    head_rows = [r for r in main_eval if r["head"]]
    eightb_rows = [r for r in main_eval if "8b" in r["model"]]
    head_block = {"n": len(head_rows), "calibrated": err_block(head_rows, "err_cal"),
                  "by_placement": err_by(head_rows, "err_cal", "placement")}
    eight_block = {"n": len(eightb_rows), "calibrated": err_block(eightb_rows, "err_cal"),
                   "uncalibrated": err_block(eightb_rows, "err_base_raw")}

    # penalty
    ptable = penalty_table(main_rows)
    regimes = {}
    for wl in sorted({r["workload"] for r in main_rows}):
        vals = [r["f_over_c"] for r in main_rows if r["workload"] == wl]
        regimes[wl] = {"mean_f_over_c": round(mean(vals), 4), "regime": regime_label(mean(vals)), "n": len(vals)}
    ratio_vals = [v["mean_f_over_c"] for v in regimes.values()]
    max_pen = max((e for e in ptable), key=lambda e: e["measured_ratio"], default=None)
    penalty = {"table": ptable, "max_measured": max_pen,
               "ratio_min": min(ratio_vals) if ratio_vals else None,
               "ratio_max": max(ratio_vals) if ratio_vals else None, "regimes": regimes}

    # slot sweep
    sweeps = sweep_table(main_rows)
    sweep_summary = {}
    for s_ in sweeps:
        if s_["placement"] == "balanced":
            pts = [p for p in s_["points"] if p["measured_ms"] is not None]
            if pts:
                sweep_summary["balanced_min_k"] = min(pts, key=lambda p: p["measured_ms"])["k"]
        if s_["placement"] == "contiguous_first":
            pts = [p for p in s_["points"] if p["measured_ms"] is not None]
            if pts:
                sweep_summary["contiguous_min_k"] = min(pts, key=lambda p: p["measured_ms"])["k"]
    sweep_errs = [p for s_ in sweeps for p in s_["points"] if p["pred_cal_ms"] is not None]
    sweep_mape = 100 * mean([abs(p["measured_ms"] / p["pred_cal_ms"] - 1) for p in sweep_errs]) if sweep_errs else None

    # capped group
    cap_eval = evaluated(cap_rows)
    cap_block = {
        "n": len(cap_eval),
        "base_uncalibrated": err_block(cap_eval, "err_base_raw"),
        "calibrated": err_block(cap_eval, "err_cal"),
        "clock_corrected": err_block(cap_eval, "err_clk"),
        "rows": [{"source": r["source"], "model": r["model"], "workload": r["workload"], "placement": r["placement"],
                  "r": r["r"], "k": r["k"], "measured_ms": r["period_ms"],
                  "pred_base_raw_ms": r["pred_base_raw_ms"], "pred_cal_ms": r["pred_cal_ms"],
                  "pred_clk_ms": r["pred_clk_ms"], "sm_clock_compute": r["sm_clock_compute"],
                  "sm_clock_stream": r["sm_clock_stream"], "clock_ratio": r["clock_ratio"]} for r in cap_eval],
    }

    # quick group (reported separately)
    quick_eval = evaluated(quick_rows)
    quick_block = {"n": len(quick_eval), "calibrated": err_block([r for r in quick_eval if r["pred_cal_ms"] is not None],
                                                                  "err_cal"),
                   "uncalibrated": err_block(quick_eval, "err_base_raw")}
    ex = [r for r in quick_eval if r["workload"] == "pre:1024" and r["r"] == 14 and r["k"] == 3
          and r["placement"] in PLACEMENTS_GRID]
    quick_block["pre1024_r14_k3"] = [
        {"placement": r["placement"], "measured_ms": r["period_ms"], "pred_base_raw_ms": r["pred_base_raw_ms"],
         "pred_cal_ms": r["pred_cal_ms"]} for r in sorted(ex, key=lambda r: r["placement"])]
    if not ex:
        skip("quick example pre:1024 r=14 k=3: not present in the quick log")
    quick_block["note"] = ("quick log has no compute_conc and no clock samples; "
                           "contention and clock-corrected predictions are not available")

    # copy bandwidth and clocks
    def bw_block(files):
        vals = [b / (ms * 1e6) for src in files for b, ms in ctx_by_source[src]["h2d"]]
        return {"mean_GBps": round(mean(vals), 3) if vals else None,
                "median_GBps": round(statistics.median(vals), 3) if vals else None, "n": len(vals)}
    main_srcs = [s for s, i in file_info.items() if i["group"] == "main"]
    bandwidth = {"main": bw_block(main_srcs), "per_file": {s: bw_block([s]) for s in file_info}}

    clock_power = {}
    for src, ctx in ctx_by_source.items():
        for r in ctx["stream"]:
            d = clock_power.setdefault(ctx["group"], {}).setdefault(r["workload"], {"sm": [], "pw": []})
            if r.get("sm_clock_mean") is not None:
                d["sm"].append(r["sm_clock_mean"])
            if r.get("power_mean") is not None:
                d["pw"].append(r["power_mean"])
    clock_power = {g: {w: {"sm_clock_mean_MHz": round(mean(v["sm"]), 1) if v["sm"] else None,
                           "power_mean_W": round(mean(v["pw"]), 2) if v["pw"] else None, "n": len(v["sm"])}
                       for w, v in d.items()} for g, d in clock_power.items()}

    # concurrent inflation on locked workloads
    infl = {}
    for src, ctx in ctx_by_source.items():
        if ctx["group"] != "main":
            continue
        for key in ctx["c_conc"]:
            um = unit_maps(ctx, key[0], key[1])
            if um is None or um["cc"] is None:
                continue
            infl[f"{src}|{key[0]}|{key[1]}"] = mean([cc / c - 1 for cc, c in zip(um["cc"], um["c"])])
    infl_max = max(infl.values()) * 100 if infl else None

    # device / versions
    dev_main = next((file_info[s]["device"] for s in main_srcs if file_info[s]["device"]), None)
    torch_v = dev_main["torch"].split("+")[0] if dev_main and dev_main.get("torch") else None
    cuda_v = str(dev_main["cuda"]) if dev_main and dev_main.get("cuda") else None

    # non-uniform study on the head file
    nu_info = None
    head_src = [r for r in main_eval if r["head"] and r["workload"] == "pre:512"]
    if head_src:
        src = head_src[0]["source"]
        ctx = ctx_by_source[src]
        model = head_src[0]["model"]
        wl = "pre:512"
        print(f"non-uniform study: {src} {model} {wl} (iters {args.nonuniform_iters}, seeds {args.nonuniform_seeds})")
        nu_info, why = nonuniform_study(ctx, model, wl, args.nonuniform_iters, list(range(args.nonuniform_seeds)))
        if nu_info is None:
            skip(f"non-uniform study: {why}")
    else:
        skip("non-uniform study: no head model with measured streaming records in the main group")
    nu_block = None
    if nu_info:
        # gains at the median budget (interpolated between 14 and 16 layer equivalents)
        gains = {}
        for k in KS_NONUNIFORM:
            pb = interp_at(nu_info["rows"], "b", k, MEDIAN_BUDGET)
            pc = interp_at(nu_info["rows"], "c", k, MEDIAN_BUDGET)
            pd = interp_at(nu_info["rows"], "d", k, MEDIAN_BUDGET)
            gains[k] = {"b_over_c_minus_1_pct": None if None in (pb, pc) else 100 * (pb / pc - 1),
                        "c_over_d_minus_1_pct": None if None in (pc, pd) else 100 * (pc / pd - 1),
                        "period_ms": {"b": pb, "c": pc, "d": pd}}
        nu_block = dict(nu_info, median_budget_layers=MEDIAN_BUDGET,
                        definition=("at budget 15 layer equivalents (median of the grid, linearly interpolated "
                                    "between 14 and 16), (b)/(c)-1 and (c)/(d)-1 in percent; "
                                    "macros use k=2"),
                        gains_at_median=gains,
                        head_resident_gain_pct=gains[2]["b_over_c_minus_1_pct"],
                        local_search_gain_pct=gains[2]["c_over_d_minus_1_pct"])

    # regimes, repeatability, prefill T and worst-case placements, decode
    base3b = [r for r in main_rows if r["model"] == "llama3.2-3b" and not r["head"]]
    regimes_table = regime_table(main_eval)
    repeat = repeat_check(rows)
    fc3 = {}
    for r in base3b:
        if prefill_t(r["workload"]) is not None:
            fc3.setdefault(r["workload"], []).append(r["f_over_c"])
    fc3 = {w: mean(v) for w, v in fc3.items()}
    transfer3 = {prefill_t(w): v for w, v in fc3.items() if v > 1}
    compute3 = {prefill_t(w): v for w, v in fc3.items() if v <= 1}
    spread, worst_rb = spread_and_worst(main_rows)
    head_sources = {r["source"] for r in main_rows if r["head"]}
    comp_spread = [v for v, key in spread if key[1] == "llama3.2-3b" and key[0] not in head_sources
                   and prefill_t(key[2]) in compute3]
    rb_max = max(worst_rb, key=lambda x: x[0]) if worst_rb else None
    dec_rows = [r for r in main_eval if r["workload"].startswith("dec:")]
    dec_block = {"n": len(dec_rows), "calibrated": err_block(dec_rows, "err_cal")}
    prefill_T = {"f_over_c_by_T": {prefill_t(w): round(v, 4) for w, v in sorted(fc3.items(), key=lambda x: prefill_t(x[0]))},
                 "transfer_bound_max_T": max(transfer3) if transfer3 else None,
                 "transfer_ratio_min": min(transfer3.values()) if transfer3 else None,
                 "transfer_ratio_max": max(transfer3.values()) if transfer3 else None,
                 "compute_bound_min_T": min(compute3) if compute3 else None,
                 "compute_bound_spread_max_pct": max(comp_spread) if comp_spread else None,
                 "random_vs_balanced_worst": {"pct": rb_max[0], "where": rb_max[1]} if rb_max else None}

    # new diagnostics requested for the paper
    interf = read_interference(os.path.join(raw, "t4_interference.log"))
    slow_unl = []
    slow_locked = []
    if interf:
        for r in interf.get("unlocked", []):
            for key in ("compute_d_over_b", "compute_e_over_b"):
                if r.get(key) is not None:
                    slow_unl.append(100 * (r[key] - 1))
        for r in interf.get("locked", []):
            if r.get("compute_d_over_b") is not None:
                slow_locked.append(100 * (r["compute_d_over_b"] - 1))
    lock_vals = [file_info[src]["device"]["lock_mhz"] for src in main_srcs
                 if file_info[src]["device"] and file_info[src]["device"].get("lock_mhz") is not None]
    clock_by_file = {}
    for src in file_info:
        sm = [r["sm_clock_mean"] for c in [ctx_by_source[src]] for r in c["stream"] if r.get("sm_clock_mean") is not None]
        clock_by_file[src] = {"group": file_info[src]["group"],
                              "lock_mhz": file_info[src]["device"]["lock_mhz"] if file_info[src]["device"] else None,
                              "stream_sm_clock_mean_MHz": round(mean(sm), 1) if sm else None}
    band_a = band_excess(main_rows, 2)
    band_b = band_excess(main_rows, 3)
    dev_pct = compute_bound_deviation(main_rows, compute3)
    e_1024 = [r["f_over_c"] for r in main_rows if r["model"] == "llama3.1-8b-16L" and r["workload"] == "pre:1024"]
    e_ratio = mean(e_1024)
    sweep_k = sweep_model_cf(ctx_by_source[SWEEP_SOURCE], "llama3.2-3b", "pre:768", 14, list(range(1, 9))) \
        if SWEEP_SOURCE in ctx_by_source else None
    unl_rows = [r for r in rows if r["group"] == "unlocked"]
    unl = unlocked_block(unl_rows) if unl_rows else None
    if unl is None:
        skip("unlocked group: results/raw/t4_unlocked_*.log not present yet")
    diag = {"interference": {"unlocked_slowdown_min_pct": min(slow_unl) if slow_unl else None,
                             "unlocked_slowdown_max_pct": max(slow_unl) if slow_unl else None,
                             "locked_slowdown_max_pct": max(slow_locked) if slow_locked else None,
                             "n_unlocked": len(slow_unl), "n_locked": len(slow_locked)},
            "clock_lock_MHz": {"low": min(lock_vals) if lock_vals else None,
                               "high": max(lock_vals) if lock_vals else None},
            "clock_by_file": clock_by_file,
            "compute_bound_deviation_max_pct": dev_pct,
            "eight_b_ratio_pre1024": e_ratio,
            "band_excess_k2": band_a, "band_excess_k3": band_b,
            "slot_model_cf_r14": sweep_k}

    summary = {
        "inputs": {"groups": {g: [os.path.basename(x) for x in fl] for g, fl in groups.items()},
                   "files": file_info, "truncated_lines_dropped": stats["truncated_lines"],
                   "stream_rows": {"main": len(main_rows), "capped": len(cap_rows), "quick": len(quick_rows)}},
        "calibration": calib,
        "scale_s": {"by_workload": {w: {"values": v, "mean": mean(v)} for w, v in s_by_workload.items()},
                    "by_model": {m: {"values": v, "mean": mean(v)} for m, v in s_by_model.items()},
                    "min": min(s_vals) if s_vals else None, "max": max(s_vals) if s_vals else None},
        "main": {"accuracy": accuracy, "ranking_calibrated": rank, "ranking_uncalibrated": rank_uncal,
                 "head_validation": head_block, "eight_b_validation": eight_block,
                 "regimes": regimes_table, "decode": dec_block, "prefill_T_3B": prefill_T},
        "repeatability": repeat,
        "diagnostics": diag,
        "unlocked": unl,
        "penalty": penalty,
        "slot_sweep": {"points": sweeps, "summary": sweep_summary, "calibrated_mape_pct": sweep_mape},
        "nonuniform": nu_block,
        "power_capped": cap_block,
        "quick": quick_block,
        "copy_bandwidth": bandwidth,
        "clock_power_by_group": clock_power,
        "conc_inflation_locked_max_pct": infl_max,
        "conc_inflation_by_workload": infl,
        "skipped": SKIPPED,
        "notes": NOTES,
    }

    def clean(o):
        if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
            return None
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        return o

    with open(os.path.join(summ, "summary.json"), "w") as fh:
        json.dump(clean(summary), fh, indent=2)
    print("wrote", os.path.join(summ, "summary.json"))

    # numbers
    cal = accuracy["calibrated"]
    unc = accuracy["uncalibrated_base"]
    def sweep_point(placement, k):
        for sw in sweeps:
            if sw["placement"] == placement:
                for pt in sw["points"]:
                    if pt["k"] == k:
                        return pt["measured_ms"]
        return None

    macros = [
        ("AccMape", fmt(cal["mape_pct"], 2)),
        ("AccMax", fmt(cal["max_abs_err_pct"], 2)),
        ("AccMapeUncal", fmt(unc["mape_pct"], 2)),
        ("AccMaxUncal", fmt(unc["max_abs_err_pct"], 2)),
        ("AccMapeCont", fmt(accuracy["calibrated_contention"]["mape_pct"], 2)),
        ("AccMaxCont", fmt(accuracy["calibrated_contention"]["max_abs_err_pct"], 2)),
        ("AccN", str(len(main_eval))),
        ("AccNAllres", str(accuracy["n_allres_excluded"])),
        ("AccRankPct", fmt_int(rank["agree_pct"])),
        ("AccRankContPct", fmt_int(rank["agree_cont_pct"])),
        ("AccRankPairs", str(rank["pairs"])),
        ("AccRankUncalPct", fmt_int(rank_uncal["agree_pct"])),
        ("HwBw", fmt(bandwidth["main"]["mean_GBps"], 1)),
        ("RatioMin", fmt(penalty["ratio_min"], 2)),
        ("RatioMax", fmt(penalty["ratio_max"], 2)),
        ("TorchVersion", torch_v or "n/a"),
        ("CudaVersion", cuda_v or "n/a"),
        ("ConcInflationLockedPct", fmt(infl_max, 1)),
        ("CapMape", fmt(cap_block["base_uncalibrated"]["mape_pct"], 2)),
        ("CapMax", fmt(cap_block["base_uncalibrated"]["max_abs_err_pct"], 2)),
        ("CapMapeCal", fmt(cap_block["calibrated"]["mape_pct"], 2)),
        ("CapMaxCal", fmt(cap_block["calibrated"]["max_abs_err_pct"], 2)),
        ("CapClockMape", fmt(cap_block["clock_corrected"]["mape_pct"], 2)),
        ("CapClockMax", fmt(cap_block["clock_corrected"]["max_abs_err_pct"], 2)),
        ("CapMaxErrInt", fmt_int(cap_block["base_uncalibrated"]["max_abs_err_pct"])),
        ("EightBMape", fmt(eight_block["calibrated"]["mape_pct"], 2)),
        ("EightBMax", fmt(eight_block["calibrated"]["max_abs_err_pct"], 2)),
        ("HeadMape", fmt(head_block["calibrated"]["mape_pct"], 2)),
        ("HeadMax", fmt(head_block["calibrated"]["max_abs_err_pct"], 2)),
        ("QuickMapeCal", fmt(quick_block["calibrated"]["mape_pct"], 2)),
        ("SweepMape", fmt(sweep_mape, 2)),
        ("CalScaleMin", fmt(summary["scale_s"]["min"], 2)),
        ("CalScaleMax", fmt(summary["scale_s"]["max"], 2)),
        ("NonUnifHeadGainPct", fmt(nu_block["head_resident_gain_pct"] if nu_block else None, 1)),
        ("NonUnifLocalGainPct", fmt(nu_block["local_search_gain_pct"] if nu_block else None, 1)),
        ("RepeatMeanPct", fmt(repeat["mean_pct"], 2)),
        ("RepeatMaxPct", fmt(repeat["max_pct"], 2)),
        ("RepeatN", str(repeat["n"])),
        ("TransferBoundMaxT", fmt_int(prefill_T["transfer_bound_max_T"])),
        ("RatioTransferMin", fmt(prefill_T["transfer_ratio_min"], 2)),
        ("RatioTransferMax", fmt(prefill_T["transfer_ratio_max"], 2)),
        ("ComputeBoundMinT", fmt_int(prefill_T["compute_bound_min_T"])),
        ("ComputeBoundSpreadPct", fmt(prefill_T["compute_bound_spread_max_pct"], 1)),
        ("RandomWorstPct", fmt_int(prefill_T["random_vs_balanced_worst"]["pct"]
                                   if prefill_T["random_vs_balanced_worst"] else None)),
        ("DecodeHostMaxErr", fmt(dec_block["calibrated"]["max_abs_err_pct"], 2)),
        ("DecodeMape", fmt(dec_block["calibrated"]["mape_pct"], 2)),
        ("SweepBalTwoMs", fmt(sweep_point("balanced", 2), 1)),
        ("SweepBalThreeMs", fmt(sweep_point("balanced", 3), 1)),
        ("SweepCfTwoMs", fmt(sweep_point("contiguous_first", 2), 1)),
        ("SweepCfSixMs", fmt(sweep_point("contiguous_first", 6), 1)),
        ("SweepCfSixOverTwo", fmt(None if sweep_point("contiguous_first", 6) is None or sweep_point("contiguous_first", 2) is None
                                  else sweep_point("contiguous_first", 6) / sweep_point("contiguous_first", 2), 2)),
        ("SweepRatio", fmt(fc3.get("pre:768"), 2)),
    ]
    if max_pen:
        macros += [
            ("MeasPenaltyMax", f"{max_pen['measured_ratio']:.2f}"),
            ("MeasPenaltyMaxWorkload", f"{max_pen['model']} {max_pen['workload'].replace(':', ' ')}"),
            ("MeasPenaltyMaxR", str(max_pen["r"])),
            ("MeasPenaltyMaxK", str(max_pen["k"])),
            ("PredPenaltyAtMax", fmt(max_pen["pred_cal_ratio"], 2)),
        ]
    else:
        macros += [("MeasPenaltyMax", "n/a"), ("MeasPenaltyMaxWorkload", "n/a"), ("MeasPenaltyMaxR", "n/a"),
                   ("MeasPenaltyMaxK", "n/a"), ("PredPenaltyAtMax", "n/a")]
    di = diag["interference"]
    lk = diag["clock_lock_MHz"]
    ba = diag["band_excess_k2"] or {}
    bb = diag["band_excess_k3"] or {}
    sk = diag["slot_model_cf_r14"] or {}
    nu_k3 = nu_block["gains_at_median"][3]["b_over_c_minus_1_pct"] if nu_block else None
    u = unl or {}
    up = u.get("penalty_max") or {}
    macros += [
        ("ConcSlowMinPct", fmt_int(di["unlocked_slowdown_min_pct"])),
        ("ConcSlowMaxPct", fmt_int(di["unlocked_slowdown_max_pct"])),
        ("ConcSlowLockedMaxPct", fmt(di["locked_slowdown_max_pct"], 1)),
        ("ClockLowMHz", fmt_int(lk["low"])),
        ("ClockHighMHz", fmt_int(lk["high"])),
        ("ComputeBoundDevMaxPct", fmt(diag["compute_bound_deviation_max_pct"], 2)),
        ("EightBRatioTenTwentyFour", fmt(diag["eight_b_ratio_pre1024"], 2)),
        ("BandExcessMaxPct", fmt(ba.get("measured_pct"), 1)),
        ("BandExcessPredPct", fmt(ba.get("predicted_pct"), 1)),
        ("BandExcessR", fmt_int(ba.get("r_at_measured_max"))),
        ("BandKThreeMaxPct", fmt(bb.get("measured_pct"), 2)),
        ("AccRankExact", fmt(rank["agree_pct"], 1)),
        ("AccRankWrong", "n/a" if rank.get("agree") is None else str(rank["pairs"] - rank["agree"])),
        ("MapeTransferPre", fmt(regimes_table["prefill transfer-bound (f/c > 1)"]["calibrated"]["mape_pct"], 2)),
        ("MapeComputePre", fmt(regimes_table["prefill compute-bound (f/c <= 1)"]["calibrated"]["mape_pct"], 2)),
        ("MapeDecode", fmt(regimes_table["decode"]["calibrated"]["mape_pct"], 2)),
        ("SweepCfMinKModel", str(sk.get("smallest_k_within_tol", "n/a")) if sk else "n/a"),
        ("SweepCfFourMs", fmt(sweep_point("contiguous_first", 4), 1)),
        ("NonUnifBudget", fmt(MEDIAN_BUDGET, 0)),
        ("NonUnifHeadGainKThreePct", fmt(nu_k3, 1)),
        ("UnlockedN", str(u.get("n", 0)) if unl else "n/a"),
        ("UnlockedMape", fmt(u.get("calibrated", {}).get("mape_pct"), 2)),
        ("UnlockedMax", fmt(u.get("calibrated", {}).get("max_abs_err_pct"), 2)),
        ("UnlockedMapeUncal", fmt(u.get("uncalibrated_base", {}).get("mape_pct"), 2)),
        ("UnlockedMaxUncal", fmt(u.get("uncalibrated_base", {}).get("max_abs_err_pct"), 2)),
        ("UnlockedMapeClk", fmt(u.get("clock_corrected", {}).get("mape_pct"), 2)),
        ("UnlockedMaxClk", fmt(u.get("clock_corrected", {}).get("max_abs_err_pct"), 2)),
        ("UnlockedPenaltyMax", fmt(up.get("measured_ratio"), 2)),
        ("UnlockedPenaltyMaxWorkload", (up["workload"].replace(":", " ") if up else "n/a")),
        ("UnlockedPenaltyMaxR", fmt_int(up.get("r"))),
        ("UnlockedPenaltyMaxK", fmt_int(up.get("k"))),
        ("UnlockedReversals", fmt_int(u.get("reversals_over_1pct")) if unl else "n/a"),
    ]
    write_numbers(numbers, macros)

    # figures
    plt = setup_matplotlib()
    fig_accuracy(plt, figs, main_eval)
    fig_penalty(plt, figs, ctx_by_source, ptable)
    fig_timeline(plt, figs, ptable, main_rows)
    fig_slots(plt, figs, sweeps)
    fig_nonuniform(plt, figs, nu_block)
    if unl_rows:
        fig_unlocked(plt, figs, unl_rows)

    print("\n--- skipped:", len(SKIPPED))
    for s in SKIPPED:
        print("  ", s)
    print("--- notes:", len(NOTES))
    for s in NOTES:
        print("  ", s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
