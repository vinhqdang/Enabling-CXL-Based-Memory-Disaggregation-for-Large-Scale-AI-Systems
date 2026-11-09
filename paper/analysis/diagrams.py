#!/usr/bin/env python3
"""Schematic and timeline figures of the placement model.

Writes paper/figs/fig_overview.pdf (system schematic and the two example placements) and
paper/figs/fig_schedule.pdf (one steady-state iteration for L=12, r=6, k=2, c=1, f=2).
The timelines come from placement.schedule, and the iteration times are checked against
placement.cycle_time and the closed forms of Theorems 3 and 4.

Usage:
    python3 paper/analysis/diagrams.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

from placement import balanced, contiguous_first, cycle_time, schedule  # noqa: E402
from placement.placements import gaps  # noqa: E402

FIGS = os.path.join(ROOT, "paper", "figs")
BLUE = "#1f77b4"      # resident layers
ORANGE = "#ff7f0e"    # streamed layers
GREY = "#7f7f7f"      # copy engine / copies
IDLE_FACE = "#fbd5d5"  # idle copy engine and GPU stalls
IDLE_EDGE = "#d9534f"
INK = "#222222"

L, R_RES, K, C, F = 12, 6, 2, 1.0, 2.0

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "pdf.fonttype": 42,
    "hatch.linewidth": 0.5,
})


def rect(ax, x, y, w, h, fc, ec="none", lw=0.6, ls="-", z=2, hatch=None):
    p = Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls,
                  zorder=z, hatch=hatch)
    ax.add_patch(p)
    return p


def text(ax, x, y, s, color=INK, ha="center", va="center", z=5, **kw):
    return ax.text(x, y, s, color=color, ha=ha, va=va, zorder=z, **kw)


def arrow(ax, x0, y0, x1, y1, color=INK, lw=0.9, rad=0.0, ms=8):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=ms,
                                 linewidth=lw, color=color, connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=0, shrinkB=0, zorder=4))


# ------------------------------------------------------------------ overview

def fig_overview(out):
    W, H = 7.0, 2.8
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    res = balanced(L, R_RES)  # residents 1, 3, ..., 11

    # host-attached memory
    hx, hy, hw, hh = 0.0, 1.42, 2.6, 1.30
    rect(ax, hx, hy, hw, hh, fc="white", ec=INK, lw=0.8, z=1)
    text(ax, hx + 0.12, hy + hh - 0.12,
         "Host-attached memory\n(page-locked DRAM or CXL expander)", ha="left", va="top")
    step, bw = 0.19, 0.17
    x0 = hx + (hw - ((L - 1) * step + bw)) / 2
    for i in range(L):
        x = x0 + i * step
        if i in res:
            rect(ax, x, 1.62, bw, 0.30, fc="white", ec=BLUE, lw=0.9, ls="--", z=2)
            text(ax, x + bw / 2, 1.77, str(i), color=BLUE)
        else:
            rect(ax, x, 1.62, bw, 0.30, fc=ORANGE, z=2)
            text(ax, x + bw / 2, 1.77, str(i))

    # copy engine
    arrow(ax, 2.72, 1.92, 3.84, 1.92, color=GREY, lw=1.6, ms=10)
    text(ax, 3.28, 2.06, "one copy engine\n(PCIe / CXL path)\nf per layer", va="bottom")

    # GPU memory
    gx, gy, gw, gh = 3.9, 1.42, 3.1, 1.30
    rect(ax, gx, gy, gw, gh, fc="white", ec=INK, lw=0.8, z=1)
    text(ax, gx + 0.12, gy + gh - 0.12, "GPU memory", ha="left", va="top")
    for j, i in enumerate(sorted(res)):
        x = 4.02 + j * 0.26
        rect(ax, x, 2.14, 0.20, 0.26, fc=BLUE, z=2)
        text(ax, x + 0.10, 2.27, str(i), color="white")
    text(ax, 5.66, 2.27, "resident layers", color=BLUE, ha="left")
    # ring of k = 2 slots with reuse arc (transfer n+1 goes to slot (n+1) mod k)
    for s in range(K):
        x = 4.02 + s * 0.96
        rect(ax, x, 1.56, 0.72, 0.30, fc="white", ec=ORANGE, lw=1.6, z=2)
        text(ax, x + 0.36, 1.71, f"slot {s}")
    arrow(ax, 4.98 + 0.36, 1.88, 4.02 + 0.36, 1.88, rad=0.35, lw=0.9, ms=7)
    text(ax, 5.84, 1.71, "ring of\nk = 2 slots", ha="left")

    # compute stream under the GPU box
    arrow(ax, 4.02, 1.20, 4.92, 1.20, lw=1.0, ms=8)
    text(ax, 4.02, 1.25, "compute stream: layers in order, c per layer", ha="left", va="bottom")

    # layer sequence strips
    cw, step_s, ch = 0.40, 0.42, 0.18
    contig = set(range(R_RES))
    bal = res

    def strip(y_cells, resident):
        for i in range(L):
            x = i * step_s
            if i in resident:
                rect(ax, x, y_cells, cw, ch, fc=BLUE, z=2)
                text(ax, x + cw / 2, y_cells + ch / 2, str(i), color="white")
            else:
                rect(ax, x, y_cells, cw, ch, fc=ORANGE, z=2)
                text(ax, x + cw / 2, y_cells + ch / 2, str(i))

    text(ax, 0.0, 0.99, "contiguous (model-order filling)", ha="left", va="bottom", weight="bold")
    strip(0.76, contig)
    text(ax, 0.0, 0.59, "balanced", ha="left", va="bottom", weight="bold")
    strip(0.36, bal)

    # gaps g_j of the balanced placement, between consecutive streamed layers (cyclic)
    streamed = [i for i in range(L) if i not in bal]
    g = gaps(L, bal)
    cen = [i * step_s + cw / 2 for i in range(L)]
    yb, yt = 0.31, 0.35
    for j in range(len(streamed)):
        a, b = streamed[j], streamed[(j + 1) % len(streamed)]
        if b > a:
            xa, xb = cen[a], cen[b]
            ax.plot([xa, xb], [yb, yb], color=INK, lw=0.6, zorder=3)
            ax.plot([xa, xa], [yb, yt], color=INK, lw=0.6, zorder=3)
            ax.plot([xb, xb], [yb, yt], color=INK, lw=0.6, zorder=3)
            text(ax, (xa + xb) / 2, 0.22, f"$g_{{{j}}}$ = {g[j]}")
        else:  # wrap-around gap: from the last streamed layer to the first one
            xa, xb = cen[a], cen[-1] + 0.12
            ax.plot([xa, xb], [yb, yb], color=INK, lw=0.6, zorder=3)
            ax.plot([xa, xa], [yb, yt], color=INK, lw=0.6, zorder=3)
            ax.plot([0.0, cen[b]], [yb, yb], color=INK, lw=0.6, zorder=3)
            ax.plot([cen[b], cen[b]], [yb, yt], color=INK, lw=0.6, zorder=3)
            text(ax, cen[11], 0.22, f"$g_{{{j}}}$ = {g[j]}")

    # legend and definition of the gaps
    rect(ax, 5.35, 0.86, 0.18, 0.14, fc=BLUE)
    text(ax, 5.62, 0.93, "resident (blue)", ha="left")
    rect(ax, 5.35, 0.66, 0.18, 0.14, fc=ORANGE)
    text(ax, 5.62, 0.73, "streamed (orange)", ha="left")
    text(ax, 5.35, 0.52, "$g_j$: resident layers\nbetween streamed j and j+1\n"
                         "(cyclic); $\\sum_j g_j = r = 6$", ha="left", va="top")

    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


# ------------------------------------------------------------------ schedule

def iteration_window(res, c, f, k, it=40):
    """Compute and copy intervals of steady-state iteration `it`, relative to its start.

    Returns a dict with the iteration time T, compute intervals (unit, start, end), stall
    intervals (GPU waiting for a copy), copy intervals overlapping the iteration (unit, start,
    end; unclipped) and the idle intervals of the copy engine inside [0, T].
    """
    L_ = len(c)
    tr = schedule(c, f, res, k, iters=it + 3)
    comp = tr["compute"]
    origin = comp[it * L_ - 1][2]
    T = comp[(it + 1) * L_ - 1][2] - origin
    compute = [(u, s - origin, e - origin) for (u, s, e) in comp[it * L_:(it + 1) * L_]]
    stalls = []
    for n in range(it * L_, (it + 1) * L_):
        u, s, _ = comp[n]
        prev_end = comp[n - 1][2]
        if u not in res and s > prev_end + 1e-12:
            stalls.append((prev_end - origin, s - origin))
    copies = [(u, s - origin, e - origin) for (u, s, e) in tr["transfer"]
              if e > origin + 1e-12 and s < origin + T - 1e-12]
    # copy engine idle time inside the window: complement of the copy intervals
    busy = sorted((max(s, 0.0), min(e, T)) for (_, s, e) in copies if min(e, T) > max(s, 0.0))
    idle, cur = [], 0.0
    for s, e in busy:
        if s > cur + 1e-12:
            idle.append((cur, s))
        cur = max(cur, e)
    if cur < T - 1e-12:
        idle.append((cur, T))
    return {"T": T, "compute": compute, "stalls": stalls, "copies": copies, "idle": idle,
            "res": set(res)}


def draw_row_panel(ax, win, res):
    yc, yg, h = 1.0, 0.0, 0.56
    T = win["T"]
    for a, b in win["idle"]:
        rect(ax, a, yc - h / 2, b - a, h, fc=IDLE_FACE, ec=IDLE_EDGE, lw=0.4, hatch="////", z=1)
    for u, s, e in win["copies"]:
        a, b = max(s, 0.0), min(e, T)
        if b - a <= 1e-12:
            continue
        rect(ax, a, yc - h / 2, b - a, h, fc=GREY, ec="white", lw=0.6, z=3)
        text(ax, (a + b) / 2, yc, str(u), color=INK)
    for a, b in win["stalls"]:
        rect(ax, a, yg - h / 2, b - a, h, fc=IDLE_FACE, ec=IDLE_EDGE, lw=0.4, hatch="////", z=1)
    for u, s, e in win["compute"]:
        if u in res:
            rect(ax, s, yg - h / 2, e - s, h, fc=BLUE, ec="white", lw=0.6, z=3)
            text(ax, (s + e) / 2, yg, str(u), color="white")
        else:
            rect(ax, s, yg - h / 2, e - s, h, fc=ORANGE, ec="white", lw=0.6, z=3)
            text(ax, (s + e) / 2, yg, str(u), color=INK)


def draw_schedule_panel(fig, rect_pos, res, label, expected, win, last):
    ax = fig.add_axes(rect_pos)
    T = win["T"]
    assert abs(T - expected) < 1e-9, (label, T, expected)
    ax.set_xlim(0, 16.4)
    ax.set_ylim(-0.5, 2.75)
    ax.set_xticks(range(0, 17, 2))
    ax.set_yticks([1.0, 0.0])
    ax.set_yticklabels(["copy engine", "GPU compute"])
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=2.5, labelsize=8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    if last:
        ax.set_xlabel("time (units of c)")
    else:
        ax.set_xticklabels([])
    draw_row_panel(ax, win, res)
    # iteration time and its end
    ax.annotate("", xy=(T, 1.85), xytext=(0, 1.85),
                arrowprops=dict(arrowstyle="<->", lw=0.8, color=INK, shrinkA=0, shrinkB=0))
    ax.plot([T, T], [-0.5, 1.85], color=INK, lw=0.7, ls="--", zorder=2)
    text(ax, T / 2, 1.92, label[1], va="bottom", ha="center")
    text(ax, 0.0, 2.62, label[0], ha="left", va="center", weight="bold")
    return ax


def fig_schedule(out):
    fig = plt.figure(figsize=(7.0, 3.0))
    res_a = contiguous_first(L, R_RES)
    res_b = balanced(L, R_RES)
    per_a = cycle_time([C] * L, [F] * L, res_a, K)
    per_b = cycle_time([C] * L, [F] * L, res_b, K)
    exp_a = max(L * C, (L - R_RES) * F, R_RES * C + 2 * C + (L - R_RES - K) * F)
    exp_b = max(L * C, (L - R_RES) * F)
    assert abs(per_a - exp_a) < 1e-9 and abs(per_b - exp_b) < 1e-9
    win_a = iteration_window(res_a, [C] * L, [F] * L, K)
    win_b = iteration_window(res_b, [C] * L, [F] * L, K)
    assert abs(win_a["T"] - per_a) < 1e-9 and abs(win_b["T"] - per_b) < 1e-9
    draw_schedule_panel(
        fig, [0.125, 0.57, 0.86, 0.38], res_a,
        ("(a) contiguous placement: residents 0-5",
         f"iteration = {win_a['T']:g}c = max(Lc, mf, rc+2c+(m-k)f)"),
        exp_a, win_a, last=False)
    ax_a = fig.axes[-1]
    ax_a.annotate("copy engine idle while the resident block computes",
                  xy=(5.0, 1.31), xytext=(6.9, 1.6), fontsize=7, color=INK, va="center", ha="left",
                  arrowprops=dict(arrowstyle="-|>", lw=0.7, color=INK, shrinkA=0, shrinkB=2, mutation_scale=7))
    draw_schedule_panel(
        fig, [0.125, 0.14, 0.86, 0.38], res_b,
        ("(b) balanced placement: residents 1, 3, ..., 11",
         f"iteration = {win_b['T']:g}c = max(Lc, mf)"),
        exp_b, win_b, last=True)
    text(fig.axes[-1], 12.5, 0.5, "no copy-engine idle,\nno GPU stall", ha="left")
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out, f"(T_contiguous={win_a['T']:g}, T_balanced={win_b['T']:g})")


def main():
    os.makedirs(FIGS, exist_ok=True)
    fig_overview(os.path.join(FIGS, "fig_overview.pdf"))
    fig_schedule(os.path.join(FIGS, "fig_schedule.pdf"))


if __name__ == "__main__":
    main()
