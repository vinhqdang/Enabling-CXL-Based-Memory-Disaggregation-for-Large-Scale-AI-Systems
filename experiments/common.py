"""Shared helpers for the CAMP experiments."""
import json, math, os, statistics, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from camp_sim.hw import H100, LINKS, GPU, Link, with_bw
from camp_sim.models import PRESETS, trace_decode, trace_prefill, trace_mixed, weights_budget
from camp_sim.sim import Engine
from camp_sim.policies import (NoPrefetch, StaticK, Reactive, CAMPPrefetch, camp_pins, make_cm, build_plan_seq,
                               pins_first, pins_freq, pins_scp, pins_knapsack_dp, pins_none, plan_time, _unique_units)

T95 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26, 10: 2.23, 15: 2.13, 20: 2.09, 30: 2.05}


def ci95(xs):
    n = len(xs)
    if n < 2:
        return 0.0
    t = T95.get(n - 1, 1.96)
    return t * statistics.stdev(xs) / math.sqrt(n)


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


def calibrated_cm(trace, cap, link, gpu, seed=0, **cm_kw):
    """Runtime cost model after ONE probe iteration (the first iteration of the workload, executed with the
    generic roofline estimate and no pinning): per-layer-kind correction factors are learned online from the
    observed layer durations.  No offline profiling is involved."""
    cm = make_cm(gpu, online=True, **cm_kw)
    Engine(trace_slice(trace, 1), gpu, link, cap, CAMPPrefetch(cm=cm, pins=set()), seed=seed).run()
    return cm


def build_suite(trace, cap, link, gpu, tune_seeds=(100, 101, 102), which=None):
    """Policy factories (seed -> fresh Policy) for one (trace, cache, link) configuration."""
    gap = gpu.sampling_gap_us * 1e-6
    cm0 = make_cm(gpu)
    seq = build_plan_seq(trace, 0, cm0)
    p_first = pins_first(seq, cap)
    p_freq = pins_freq(seq, cap)
    p_scp_nom, info = pins_scp_verified(trace, cap, link, gpu, cm0, return_info=True)
    cm_cal = calibrated_cm(trace, cap, link, gpu)
    p_scp = pins_scp_verified(trace, cap, link, gpu, cm_cal)
    seq_x = build_plan_seq(trace, 0, exact_cm(gpu))
    p_scp_x = pins_scp_verified(trace, cap, link, gpu, exact_cm(gpu))

    suite = {
        "Demand+LRU": lambda s: NoPrefetch(),
        "Reactive (TMO-like)": lambda s: Reactive(),
        "Aggressive-10 (Limoncello-like)": lambda s: StaticK(10),
        "FlexGen-style": lambda s: StaticK(2, wrap=True, pins=p_first, evict="fifo"),
        "Hot/Cold (PowerInfer-style)": lambda s: StaticK(2, pins=p_freq),
        "CAMP-v1 (orig.)": lambda s: CAMPPrefetch(cm=make_cm(gpu), mode="horizon", wrap=False, pins=p_freq),
        "CAMP-v2": lambda s: CAMPPrefetch(cm=make_cm(gpu), pins=p_scp),
        "CAMP-v2 (uncalibrated)": lambda s: CAMPPrefetch(cm=make_cm(gpu), pins=p_scp_nom),
        "CAMP-v2 (exact cost model)": lambda s: CAMPPrefetch(cm=exact_cm(gpu), pins=p_scp_x),
    }
    # best fixed lookahead (oracle-tuned per configuration, strongest static baseline)
    best_k, best_v = None, float("inf")
    for k in (1, 2, 4, 8):
        v = statistics.mean(Engine(trace, gpu, link, cap, StaticK(k), seed=s).run().steady() for s in tune_seeds)
        if v < best_v:
            best_k, best_v = k, v
    suite[f"Static-k tuned (k={best_k})"] = lambda s, k=best_k: StaticK(k)
    suite["_info"] = dict(best_k=best_k, scp=info, n_pin=len(p_scp),
                          pin_gb=sum(trace.units[u].nbytes for u in p_scp) / 1e9)
    return suite


def eval_suite(trace, cap, link, gpu, seeds=range(10), names=None, **ekw):
    suite = build_suite(trace, cap, link, gpu)
    info = suite.pop("_info")
    out = {}
    for n, f in suite.items():
        if names and n not in names:
            continue
        vals, stalls = [], []
        for s in seeds:
            r = Engine(trace, gpu, link, cap, f(s), seed=s, **ekw).run()
            vals.append(r.steady())
            stalls.append(r.steady_stall())
        out[n] = dict(vals=vals, mean=statistics.mean(vals), ci=ci95(vals), stall=statistics.mean(stalls))
    out["_lb"] = lower_bound(trace, cap, link, gpu)
    out["_info"] = info
    return out
