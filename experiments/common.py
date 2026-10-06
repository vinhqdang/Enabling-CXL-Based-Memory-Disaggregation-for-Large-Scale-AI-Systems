"""Shared helpers for the CAMP experiments."""
import json, math, os, statistics, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from camp_sim.hw import H100, LINKS, GPU, Link, with_bw
from camp_sim.models import PRESETS, trace_decode, trace_prefill, trace_mixed, weights_budget
from camp_sim.sim import Engine
from camp_sim.policies import (NoPrefetch, StaticK, Reactive, CAMPPrefetch, camp_pins, make_cm, build_plan_seq,
                               pins_first, pins_freq, pins_scp, pins_knapsack_dp, pins_none, plan_time, _unique_units)

def _t95(df):
    try:
        from scipy import stats
        return float(stats.t.ppf(0.975, df))
    except Exception:  # pragma: no cover
        tab = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
               11: 2.201, 12: 2.179, 15: 2.131, 19: 2.093, 29: 2.045}
        keys = sorted(tab)
        for k in keys:
            if df <= k:
                return tab[k]
        return 1.96


def ci95(xs):
    n = len(xs)
    if n < 2:
        return 0.0
    return _t95(n - 1) * statistics.stdev(xs) / math.sqrt(n)


def run_many(trace, gpu, link, cap, make_policy, seeds=range(8), skip=1, **ekw):
    """Return (mean, ci95, last Result) of steady-state iteration latency over seeds."""
    vals, last = [], None
    for s in seeds:
        r = Engine(trace, gpu, link, cap, make_policy(s), seed=s, **ekw).run()
        vals.append(r.steady(skip))
        last = r
    return statistics.mean(vals), ci95(vals), last


def lower_bound(trace, cap, link, gpu, skip=0):
    """Implementation-independent lower bound on one iteration (valid for non-uniform unit sizes).

    The iteration cannot finish before the compute-only time, nor before the minimum number of
    streamed bytes has crossed the link.  If any unit is streamed, one streaming slot as large as the
    largest streamed unit must stay free, so with W total weight bytes the streamed bytes are at least
    W - (cap - s) where s is the size of the largest streamed unit; units larger than s must then be
    pinned (their total must fit in cap - s).  The bound takes the smallest feasible s and a fractional
    (relaxed) choice of the remaining pinned bytes, and is exact for uniform unit sizes.
    """
    n0 = trace.iter_starts[1] if len(trace.iter_starts) > 1 else len(trace.acc)
    acc = trace.acc[:n0]
    comp = sum(a.t_true for a in acc)
    uniq = {}
    for a in acc:
        if a.fetch_bytes is None:
            uniq[a.uid] = trace.units[a.uid].nbytes
    sizes = sorted(uniq.values())
    W = sum(sizes)
    if W <= cap:
        streamed = 0.0
    else:
        streamed = float("inf")
        for s in sorted(set(sizes)):
            larger = sum(x for x in sizes if x > s)
            if larger <= cap - s:
                streamed = max(0.0, W - (cap - s))
                break
    return max(comp, streamed / link.effective_bw())


# --------------------------------------------------------------------------------------
# policy suite
# --------------------------------------------------------------------------------------
from camp_sim.policies import pins_stride, pins_scp, pins_greedy_marginal, pins_scp_verified
from camp_sim.cost import CostModel


def exact_cm(gpu):
    """Cost model that knows the device's true efficiencies (upper bound on estimator quality)."""
    return CostModel(peak_flops=gpu.peak_flops, hbm_bw=gpu.hbm_bw, nominal_gemm=gpu.eff_gemm,
                     nominal_mem=gpu.eff_mem, nominal_attn=gpu.eff_attn, launch_us=gpu.launch_us,
                     online=False)


def trace_slice(tr, n_iters):
    from camp_sim.models import Trace
    end = tr.iter_starts[n_iters] if n_iters < len(tr.iter_starts) else len(tr.acc)
    return Trace(tr.units, tr.acc[:end], tr.iter_starts[:n_iters], tr.meta)


def calibrated_cm(trace, cap, link, gpu, seed=10_001, **cm_kw):
    """Runtime cost model after ONE probe iteration (the first iteration of the workload, executed with the
    generic roofline estimate and no pinning): per-layer-kind correction factors are learned online from the
    observed layer durations.  No offline profiling is involved."""
    cm = make_cm(gpu, online=True, **cm_kw)
    Engine(trace_slice(trace, 1), gpu, link, cap, CAMPPrefetch(cm=cm, pins=set()), seed=seed).run()
    return cm


from camp_sim.models import Unit, Trace as _Trace


def flexgen_split(trace, cap):
    """FlexGen-style placement: a uniform fraction ``phi`` of *every* layer stays resident and the remaining
    (1-phi) of each layer is streamed through two slots (double buffering).  Returns (trace', cap', phi).

    The resident part is charged to the cache, so the streamed pieces see cap' = cap - phi * W.  This is the
    per-layer percentage split of FlexGen's policy, modelled at fractional granularity."""
    full = {a.uid for a in trace.acc if a.fetch_bytes is None}
    W = sum(trace.units[u].nbytes for u in full)
    smax = max(trace.units[u].nbytes for u in full)
    if W <= cap:
        return trace, cap, 1.0
    if cap >= 2 * smax:
        phi = (cap - 2 * smax) / (W - 2 * smax)
    else:
        phi = max(0.0, (cap - smax) / (W - smax))
    phi = min(max(phi, 0.0), 0.999)
    units2 = {u: Unit(u, un.name, un.kind, (1 - phi) * un.nbytes if u in full else un.nbytes) for u, un in trace.units.items()}
    return _Trace(units2, trace.acc, trace.iter_starts, trace.meta), cap - phi * W, phi


def tuned_k(trace, cap, link, gpu, make, ks=(1, 2, 4, 8), seeds=(100, 101, 102)):
    best_k, best_v = None, float("inf")
    for k in ks:
        v = statistics.mean(Engine(trace, gpu, link, cap, make(k), seed=s).run().steady() for s in seeds)
        if v < best_v:
            best_k, best_v = k, v
    return best_k


def build_suite(trace, cap, link, gpu, tune_seeds=(100, 101, 102), which=None):
    """Policy factories (seed -> fresh Policy) for one (trace, cache, link) configuration.

    All pinning baselines use the *equal budget* rule (pin up to cap - 1 streaming slot) and a tuned lookahead;
    the conventional 2-slot / position-tie-break rules appear only in the attribution experiment."""
    cm0 = make_cm(gpu)
    seq = build_plan_seq(trace, 0, cm0)
    p_hot = pins_freq(seq, cap, gamma=1.0, slots=1.0, tiebreak="random", seed=0)
    p_stride = pins_stride(seq, cap, gamma=1.0, slots=1.0)
    p_freq1 = pins_freq(seq, cap)                          # CAMP-v1 pinning rule (conventional budget)
    p_scp_nom, info = pins_scp_verified(trace, cap, link, gpu, cm0, return_info=True)
    cm_cal = calibrated_cm(trace, cap, link, gpu)
    p_scp = pins_scp_verified(trace, cap, link, gpu, cm_cal)
    p_scp_x = pins_scp_verified(trace, cap, link, gpu, exact_cm(gpu))
    k_static = tuned_k(trace, cap, link, gpu, lambda k: StaticK(k))
    k_hot = tuned_k(trace, cap, link, gpu, lambda k: StaticK(k, wrap=True, pins=p_hot, evict="fifo"))
    tr_fg, cap_fg, phi = flexgen_split(trace, cap)
    k_fg = tuned_k(tr_fg, cap_fg, link, gpu, lambda k: StaticK(k, wrap=True, evict="fifo"))

    suite = {
        "Demand+LRU": lambda s: NoPrefetch(),
        f"Static-k tuned (k={k_static})": lambda s: StaticK(k_static),
        "Reactive (TMO-like)": lambda s: Reactive(),
        "Aggressive-10 (Limoncello-like)": lambda s: StaticK(10),
        "FlexGen-style (layer split)": ("split", lambda s: StaticK(k_fg, wrap=True, evict="fifo")),
        f"Hot/Cold (equal budget, k={k_hot})": lambda s: StaticK(k_hot, wrap=True, pins=p_hot, evict="fifo"),
        "Stride pins + CAMP prefetch": lambda s: CAMPPrefetch(cm=make_cm(gpu), pins=p_stride),
        "CAMP-v1 (orig.)": lambda s: CAMPPrefetch(cm=make_cm(gpu), mode="horizon", wrap=False, pins=p_freq1),
        "CAMP-v2": lambda s: CAMPPrefetch(cm=make_cm(gpu), pins=p_scp),
        "CAMP-v2 (uncalibrated)": lambda s: CAMPPrefetch(cm=make_cm(gpu), pins=p_scp_nom),
        "CAMP-v2 (exact constants)": lambda s: CAMPPrefetch(cm=exact_cm(gpu), pins=p_scp_x),
    }
    suite["_info"] = dict(k_static=k_static, k_hot=k_hot, k_fg=k_fg, phi_fg=phi, scp=info, n_pin=len(p_scp),
                          pin_gb=sum(trace.units[u].nbytes for u in p_scp) / 1e9, split=(tr_fg, cap_fg))
    return suite


def eval_suite(trace, cap, link, gpu, seeds=range(10), names=None, **ekw):
    suite = build_suite(trace, cap, link, gpu)
    info = suite.pop("_info")
    tr_fg, cap_fg = info.pop("split")
    out = {}
    for n, f in suite.items():
        if names and not any(n.startswith(x) for x in names):
            continue
        tr_use, cap_use = trace, cap
        if isinstance(f, tuple):
            tr_use, cap_use, f = tr_fg, cap_fg, f[1]
        vals, stalls, utils = [], [], []
        for s_ in seeds:
            r = Engine(tr_use, gpu, link, cap_use, f(s_), seed=s_, **ekw).run()
            vals.append(r.steady())
            stalls.append(r.steady_stall())
            utils.append(r.link_util)
        out[n] = dict(vals=vals, mean=statistics.mean(vals), ci=ci95(vals), stall=statistics.mean(stalls),
                      util=statistics.mean(utils))
    out["_lb"] = lower_bound(trace, cap, link, gpu)
    out["_info"] = info
    return out


def det_ratio_to_lb(trace, cap, link, gpu, pins, cm=None):
    """Deterministic (noise-free) CAMP latency divided by the lower bound (max with the compute-only time)."""
    n0 = trace.iter_starts[1] if len(trace.iter_starts) > 1 else len(trace.acc)
    ideal = sum(a.t_true for a in trace.acc[:n0]) + gpu.sampling_gap_us * 1e-6
    r = Engine(trace, gpu, link, cap, CAMPPrefetch(cm=cm or make_cm(gpu), pins=pins), seed=0, deterministic=True).run()
    return r.steady() / max(lower_bound(trace, cap, link, gpu), ideal)
