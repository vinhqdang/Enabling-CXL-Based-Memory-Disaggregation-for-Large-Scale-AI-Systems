"""
Run the CAMP evaluation.  Usage:  python experiments/run_all.py [e1 e2 ...]   (default: all)

Every experiment writes results/<name>.json; figures are produced by experiments/plots.py and every number
quoted in the manuscript is generated from these files by experiments/numbers.py / make_tables.py.
"""
import json
import os
import random
import statistics
import sys
import time
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *  # noqa
from camp_sim.cost import CostModel, true_time
from camp_sim.models import ModelSpec, trace_specdec, trace_encdec
from camp_sim.policies import pins_exhaustive, pins_random, pins_greedy_marginal, pins_scp_verified

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
GPU_ = H100
LINK18 = LINKS["cxl_x8_measured"]
LINK45 = LINKS["cxl_x16_expected"]
HOSTDRAM = LINKS["host_dram_pinned_gen5"]
SEEDS = range(10)
DEC_STEPS = 6
M13, M70, M405, MG = PRESETS["llama2-13b"], PRESETS["llama3-70b"], PRESETS["llama3-405b"], PRESETS["gemma2-9b"]


def save(name, obj):
    with open(os.path.join(RES, name + ".json"), "w") as f:
        json.dump(obj, f, indent=1)
    print(f"[saved] {name}.json", flush=True)


def ideal_time(trace, gpu):
    """All-weights-resident time per iteration (no CXL traffic): compute only + sampling gap."""
    n0 = trace.iter_starts[1] if len(trace.iter_starts) > 1 else len(trace.acc)
    return sum(a.t_true for a in trace.acc[:n0]) + gpu.sampling_gap_us * 1e-6


def camp_pins_cal(trace, cap, link, gpu):
    cm = calibrated_cm(trace, cap, link, gpu)
    return pins_scp_verified(trace, cap, link, gpu, cm), cm


def camp_v2(trace, cap, link, gpu, **kw):
    pins, cm = camp_pins_cal(trace, cap, link, gpu)
    return lambda s: CAMPPrefetch(cm=make_cm(gpu), pins=pins, **kw)


def dec(m, B, ctx, steps=DEC_STEPS):
    return trace_decode(m, GPU_, B, ctx, steps)


# ----------------------------------------------------------------------------------------
# E1  regime map: when does streaming from CXL pay off?  (tokens per iteration x link BW)
# ----------------------------------------------------------------------------------------
def workload_by_tokens(m, T):
    """One iteration that processes T tokens through the GEMM layers (no long KV context), so that the
    x-axis isolates the weight-streaming arithmetic intensity (FLOP per streamed byte = T)."""
    if T <= 16384:
        return trace_prefill(m, GPU_, 1, T, reps=4)
    return trace_prefill(m, GPU_, T // 16384, 16384, reps=4)


def e1():
    m = M70
    cap = 0.4 * m.weight_bytes
    Ts = [1, 4, 16, 64, 256, 1024, 4096, 16384, 65536]
    BWs = [8, 12, 18, 24, 32, 45, 64]
    cells = []
    for T in Ts:
        tr = workload_by_tokens(m, T)
        ideal = ideal_time(tr, GPU_)
        seq = build_plan_seq(tr, 0, make_cm(GPU_))
        for bw in BWs:
            link = with_bw(LINK18, bw)
            mc, cc, r = run_many(tr, GPU_, link, cap, camp_v2(tr, cap, link, GPU_), seeds=range(4))
            md, cd, _ = run_many(tr, GPU_, link, cap, lambda s: NoPrefetch(), seeds=range(4))
            P = pins_stride(seq, cap, gamma=1.0, slots=1.0)
            ms_, cs_, _ = run_many(tr, GPU_, link, cap, lambda s: CAMPPrefetch(cm=make_cm(GPU_), pins=P), seeds=range(4))
            tfg, cfg_, phi, kfg, sfg = flexgen_best(tr, cap, link, GPU_)
            mf, cf, _ = run_many(tfg, GPU_, link, cfg_, lambda s: StaticK(kfg, wrap=True, evict="fifo"), seeds=range(4))
            cells.append(dict(T=T, bw=bw, ideal=ideal, camp=mc, demand=md, stride=ms_, flexgen=mf,
                              lb=lower_bound(tr, cap, link, GPU_), stall_frac=r.steady_stall() / mc))
        print("e1 T", T, flush=True)
    save("e1_regime_map", dict(model=m.name, cache_ratio=0.4, Ts=Ts, BWs=BWs, cells=cells))


# ----------------------------------------------------------------------------------------
# E2  main comparison on prefill / decode / mixed workloads (70B and 405B lead; 13B/9B are capacity-constrained cases)
# ----------------------------------------------------------------------------------------
def main_configs():
    cfgs = [
        ("llama3-70b", "decode B=16 ctx=2k", dec(M70, 16, 2048), weights_budget(M70, GPU_, 16, 4096), "main"),
        ("llama3-70b", "mixed 32 dec + 512 chunk", trace_mixed(M70, GPU_, 32, 2048, 512, DEC_STEPS), weights_budget(M70, GPU_, 32, 4096), "main"),
        ("llama3-70b", "prefill B=4 S=2k", trace_prefill(M70, GPU_, 4, 2048), weights_budget(M70, GPU_, 4, 2048), "main"),
        ("llama3-70b", "prefill B=4 S=4k", trace_prefill(M70, GPU_, 4, 4096), weights_budget(M70, GPU_, 4, 4096), "main"),
        ("llama3-405b", "decode B=16 ctx=2k", dec(M405, 16, 2048, 4), weights_budget(M405, GPU_, 16, 4096), "main"),
        ("llama3-405b", "prefill B=4 S=2k", trace_prefill(M405, GPU_, 4, 2048, reps=3), weights_budget(M405, GPU_, 4, 2048), "main"),
        ("llama2-13b", "decode B=16 ctx=2k", dec(M13, 16, 2048), 0.4 * M13.weight_bytes, "constrained"),
        ("llama2-13b", "prefill B=4 S=4k", trace_prefill(M13, GPU_, 4, 4096), 0.4 * M13.weight_bytes, "constrained"),
        ("gemma2-9b (tied)", "decode B=32 ctx=2k", dec(MG, 32, 2048), 0.4 * MG.weight_bytes, "constrained"),
    ]
    return cfgs


def e2():
    out = []
    for (mn, wl, tr, cap, group) in main_configs():
        for link in (LINK18, LINK45):
            t0 = time.time()
            res = eval_suite(tr, cap, link, GPU_, seeds=SEEDS)
            info = res.pop("_info")
            lb = res.pop("_lb")
            pins, cm = camp_pins_cal(tr, cap, link, GPU_)
            det = det_ratio_to_lb(tr, cap, link, GPU_, pins, cm=make_cm(GPU_))
            out.append(dict(model=mn, workload=wl, group=group, link=link.name, bw=link.bw_gbs, cache_gb=cap / 1e9,
                            cache_ratio=cap / sum(u.nbytes for u in tr.units.values()), ideal=ideal_time(tr, GPU_),
                            lb=lb, det_ratio_lb=det,
                            info=dict(k_static=info["k_static"], k_hot=info["k_hot"], k_fg=info["k_fg"], phi_fg=info["phi_fg"],
                                      n_pin=info["n_pin"], pin_gb=info["pin_gb"], scp_pick=info["scp"]["picked"]),
                            results=res))
            print(f"e2 {mn:18s} {wl:28s} {link.name:20s} {time.time()-t0:.1f}s", flush=True)
    save("e2_main", out)


# ----------------------------------------------------------------------------------------
# E3  pinning study: what does the pin-selection rule buy?
# ----------------------------------------------------------------------------------------
def pin_strategies(trace, cap, link, gpu):
    gap = gpu.sampling_gap_us * 1e-6
    seq = build_plan_seq(trace, 0, make_cm(gpu))
    S = {"none": set(),
         "prefix (conventional budget)": pins_first(seq, cap),
         "frequency (conventional budget)": pins_freq(seq, cap),
         "frequency (equal budget, random tie-break)": pins_freq(seq, cap, gamma=1.0, slots=1.0, tiebreak="random"),
         "stride (equal budget)": pins_stride(seq, cap, gamma=1.0, slots=1.0),
         "knapsack-DP (equal budget)": pins_knapsack_dp(seq, cap, link, gamma=1.0, slots=1.0),
         "marginal greedy": pins_greedy_marginal(seq, cap, link, gap),
         "SCP (planner only)": pins_scp(seq, cap, link, gap),
         "SCP (plan+verify)": pins_scp_verified(trace, cap, link, gpu, calibrated_cm(trace, cap, link, gpu))}
    return S, seq


def e3():
    out = []
    wl = [("llama3-70b", "decode B=16", dec(M70, 16, 2048), weights_budget(M70, GPU_, 16, 4096)),
          ("llama3-70b", "prefill B=4 S=2k", trace_prefill(M70, GPU_, 4, 2048), weights_budget(M70, GPU_, 4, 2048)),
          ("llama3-70b", "prefill B=4 S=4k", trace_prefill(M70, GPU_, 4, 4096), weights_budget(M70, GPU_, 4, 4096)),
          ("llama2-13b", "decode B=16", dec(M13, 16, 2048), 0.4 * M13.weight_bytes),
          ("llama2-13b", "prefill B=1 S=2k", trace_prefill(M13, GPU_, 1, 2048), 0.4 * M13.weight_bytes),
          ("llama2-13b", "prefill B=4 S=4k", trace_prefill(M13, GPU_, 4, 4096), 0.4 * M13.weight_bytes),
          ("llama2-13b", "prefill B=8 S=4k", trace_prefill(M13, GPU_, 8, 4096), 0.4 * M13.weight_bytes)]
    for (mn, name, tr, cap) in wl:
        for link in (LINK18, LINK45):
            S, seq = pin_strategies(tr, cap, link, GPU_)
            row = dict(model=mn, workload=name, bw=link.bw_gbs, lb=lower_bound(tr, cap, link, GPU_), ideal=ideal_time(tr, GPU_), res={})
            for k, P in S.items():
                vals, utils = [], []
                for s in SEEDS:
                    r = Engine(tr, GPU_, link, cap, CAMPPrefetch(cm=make_cm(GPU_), pins=P), seed=s).run()
                    vals.append(r.steady())
                    utils.append(r.link_util)
                row["res"][k] = dict(mean=statistics.mean(vals), ci=ci95(vals), vals=vals, util=statistics.mean(utils),
                                     n_pin=len(P), pin_gb=sum(tr.units[u].nbytes for u in P) / 1e9)
            # random pin sets: 20 independent draws (equal budget), 3 seeds each
            draws = []
            for d in range(20):
                P = pins_random(seq, cap, gamma=1.0, slots=1.0, seed=d)
                draws.append(statistics.mean(Engine(tr, GPU_, link, cap, CAMPPrefetch(cm=make_cm(GPU_), pins=P), seed=s).run().steady()
                                             for s in range(3)))
            row["random_draws"] = dict(mean=statistics.mean(draws), min=min(draws), max=max(draws),
                                       p25=statistics.quantiles(draws, n=4)[0], p75=statistics.quantiles(draws, n=4)[2], vals=draws)
            out.append(row)
            print("e3", mn, name, link.bw_gbs, flush=True)
    save("e3_pin_strategies", out)

    # ---- optimality gap on a small instance (exhaustive search over all pin sets) ----
    toy = ModelSpec("toy-8L", 8, 2048, 16, 16, 128, 8192, 32000)
    gap_rows = []
    gap = GPU_.sampling_gap_us * 1e-6
    for (name, tr_f) in [("decode B=16", lambda: trace_decode(toy, GPU_, 16, 2048, 4)),
                         ("prefill B=1 S=2k", lambda: trace_prefill(toy, GPU_, 1, 2048)),
                         ("prefill B=4 S=4k", lambda: trace_prefill(toy, GPU_, 4, 4096)),
                         ("prefill B=8 S=2k", lambda: trace_prefill(toy, GPU_, 8, 2048))]:
        tr = tr_f()
        for ratio in (0.3, 0.5, 0.7):
            cap = ratio * toy.weight_bytes
            link = with_bw(LINK18, 4.0)   # scaled so that the small model sits in the same regime as the large ones
            seq = build_plan_seq(tr, 0, make_cm(GPU_))
            Pe = pins_exhaustive(seq, cap, link, gap, max_units=18)
            Ps = pins_scp(seq, cap, link, gap)
            Pf = pins_freq(seq, cap)
            Pst = pins_stride(seq, cap, gamma=1.0, slots=1.0)
            te, ts, tf, tst = (plan_time(seq, P, cap, link, gap) for P in (Pe, Ps, Pf, Pst))
            sim = {}
            for k, P in (("exhaustive", Pe), ("scp", Ps), ("freq", Pf), ("stride", Pst)):
                vals = [Engine(tr, GPU_, link, cap, CAMPPrefetch(cm=make_cm(GPU_), pins=P), seed=s).run().steady() for s in range(6)]
                sim[k] = statistics.mean(vals)
            gap_rows.append(dict(workload=name, cache_ratio=ratio, planner=dict(exhaustive=te, scp=ts, freq=tf, stride=tst),
                                 sim=sim, scp_gap_pct=100 * (ts - te) / te, stride_gap_pct=100 * (tst - te) / te,
                                 sim_gap_pct=100 * (sim["scp"] - sim["exhaustive"]) / sim["exhaustive"],
                                 freq_gap_pct=100 * (tf - te) / te))
            print("e3 gap", name, ratio, flush=True)
    save("e3_optimality_gap", gap_rows)


# ----------------------------------------------------------------------------------------
# E4  non-uniform reuse (tied embedding, speculative decoding, encoder-decoder) and eviction
# ----------------------------------------------------------------------------------------
def e4():
    out = {}
    tr = dec(M13, 16, 2048)
    cap = 0.4 * M13.weight_bytes
    rows = []
    for ev in ("lru", "fifo", "lfu", "random", "belady"):
        for pf, mk in (("demand", lambda e: NoPrefetch(evict=e)), ("static-2", lambda e: StaticK(2, evict=e))):
            vals = [Engine(tr, GPU_, LINK18, cap, mk(ev), seed=s).run().steady() for s in range(6)]
            r = Engine(tr, GPU_, LINK18, cap, mk(ev), seed=0).run()
            rows.append(dict(evict=ev, prefetch=pf, mean=statistics.mean(vals), ci=ci95(vals), hit_rate=r.hits / max(1, r.accesses)))
    out["eviction_uniform"] = rows

    link = LINK18
    gap = GPU_.sampling_gap_us * 1e-6
    het = []
    d1b = ModelSpec("draft-1b", 16, 2048, 16, 16, 128, 5632, 32000)
    enc = ModelSpec("enc-24L", 24, 4096, 32, 32, 128, 10240, 32000, gated=False)
    dec_m = ModelSpec("dec-24L", 24, 4096, 32, 32, 128, 10240, 32000, gated=False)
    cases = [("gemma2-9b tied decode", dec(MG, 32, 2048), 0.4),
             ("spec. decoding (13B target + 1B draft, k=4)", trace_specdec(M13, d1b, GPU_, 8, 2048, 4, 4), 0.35),
             ("encoder-decoder (24+24 layers, 64 output tokens)", trace_encdec(enc, dec_m, GPU_, 8, 2048, 64, 2), 0.35)]
    for (name, tr, ratio) in cases:
        total = sum(u.nbytes for u in tr.units.values())
        cap = ratio * total
        seq = build_plan_seq(tr, 0, make_cm(GPU_))
        is_ed = tr.meta["kind"] == "encdec"
        if is_ed:
            end = tr.iter_starts[1 + 64] if len(tr.iter_starts) > 65 else len(tr.acc)
            cm = make_cm(GPU_)
            seq = [(a.uid, tr.units[a.uid].nbytes, cm.predict(a.feat, a.kind), a.fetch_bytes) for a in tr.acc[:end]]
        P = {"none": set(), "prefix (conv.)": pins_first(seq, cap), "frequency (conv.)": pins_freq(seq, cap),
             "frequency (equal budget)": pins_freq(seq, cap, gamma=1.0, slots=1.0, tiebreak="random"),
             "stride (equal budget)": pins_stride(seq, cap, gamma=1.0, slots=1.0),
             "knapsack-DP": pins_knapsack_dp(seq, cap, link, gamma=1.0, slots=1.0),
             "SCP": (pins_scp(seq, cap, link, gap) if is_ed else pins_scp_verified(tr, cap, link, GPU_, calibrated_cm(tr, cap, link, GPU_)))}
        row = dict(case=name, cache_ratio=ratio, res={})

        def per_unit(r):
            if is_ed:
                ipr = tr.meta["iters_per_request"]
                return sum(r.iter_times[ipr:2 * ipr])
            return r.steady()

        for k, pins in P.items():
            vals = [per_unit(Engine(tr, GPU_, link, cap, CAMPPrefetch(cm=make_cm(GPU_), pins=pins), seed=s).run()) for s in range(6)]
            row["res"][f"CAMP prefetch + {k} pins"] = dict(mean=statistics.mean(vals), ci=ci95(vals),
                                                         pin_gb=sum(tr.units[u].nbytes for u in pins) / 1e9)
        for ev in ("lru", "lfu"):
            vals = [per_unit(Engine(tr, GPU_, link, cap, NoPrefetch(evict=ev), seed=s).run()) for s in range(6)]
            row["res"][f"demand + {ev.upper()}"] = dict(mean=statistics.mean(vals), ci=ci95(vals), pin_gb=0.0)
        # FlexGen-style layer split on the same trace
        trs, caps, phi, kfg, sfg = flexgen_best(tr, cap, link, GPU_)
        vals = [per_unit(Engine(trs, GPU_, link, caps, StaticK(kfg, wrap=True, evict="fifo"), seed=s).run()) for s in range(6)]
        row["res"]["FlexGen-style layer split"] = dict(mean=statistics.mean(vals), ci=ci95(vals), pin_gb=phi * sum(
            tr.units[u].nbytes for u in tr.full_fetch_uids()) / 1e9)
        het.append(row)
        print("e4", name, flush=True)
    out["heterogeneous"] = het
    save("e4_reuse_heterogeneity", out)


# ----------------------------------------------------------------------------------------
# E5  runtime cost-model sensitivity
# ----------------------------------------------------------------------------------------
class TrueCM(CostModel):
    """Oracle estimator: returns the device's true layer time (used only as an upper bound)."""
    def __init__(self, gpu):
        super().__init__(peak_flops=gpu.peak_flops, hbm_bw=gpu.hbm_bw, online=False)
        self._gpu = gpu

    def predict(self, feat, kind):
        return true_time(feat, self._gpu)


def e5():
    gap = GPU_.sampling_gap_us * 1e-6
    out = {"cases": [], "shape": []}
    wl = [("llama2-13b prefill B=4 S=4k", trace_prefill(M13, GPU_, 4, 4096), 0.4 * M13.weight_bytes),
          ("llama3-70b prefill B=4 S=2k", trace_prefill(M70, GPU_, 4, 2048), weights_budget(M70, GPU_, 4, 2048)),
          ("llama3-70b decode B=16", dec(M70, 16, 2048), weights_budget(M70, GPU_, 16, 4096))]
    link = LINK18
    for (name, tr, cap) in wl:
        base = {}

        def lat(P):
            vals = [Engine(tr, GPU_, link, cap, CAMPPrefetch(cm=make_cm(GPU_), pins=P), seed=s).run().steady() for s in range(8)]
            return statistics.mean(vals), ci95(vals)

        base["exact"] = lat(pins_scp_verified(tr, cap, link, GPU_, exact_cm(GPU_)))
        base["generic"] = lat(pins_scp_verified(tr, cap, link, GPU_, make_cm(GPU_, online=False)))
        base["calibrated"] = lat(pins_scp_verified(tr, cap, link, GPU_, calibrated_cm(tr, cap, link, GPU_)))
        base["no_pins"] = lat(set())
        rows = []
        for bias in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0):
            for sigma in (0.0, 0.5):
                for online in (False, True):
                    cm = make_cm(GPU_, bias=bias, noise_sigma=sigma, online=online, seed=3)
                    if online:
                        cm = calibrated_cm(tr, cap, link, GPU_, bias=bias, noise_sigma=sigma, seed=3)
                    P = pins_scp_verified(tr, cap, link, GPU_, cm)
                    m_, c_ = lat(P)
                    rows.append(dict(bias=bias, sigma=sigma, online=online, mean=m_, ci=c_, n_pin=len(P)))
        # estimator accuracy on the same iteration
        acc = {}
        n0 = tr.iter_starts[1] if len(tr.iter_starts) > 1 else len(tr.acc)
        for online in (False, True):
            cm = make_cm(GPU_, online=online)
            e0 = [abs(cm.predict(a.feat, a.kind) - a.t_true) / a.t_true for a in tr.acc[:n0]]
            if online:
                Engine(trace_slice(tr, 1), GPU_, link, cap, CAMPPrefetch(cm=cm, pins=set()), seed=10_001).run()
            e1_ = [abs(cm.predict(a.feat, a.kind) - a.t_true) / a.t_true for a in tr.acc[:n0]]
            acc["online" if online else "static"] = dict(before=statistics.mean(e0), after=statistics.mean(e1_), corr=dict(cm.corr))
        pf = []
        pins_x = pins_scp_verified(tr, cap, link, GPU_, exact_cm(GPU_))
        for bias in (0.25, 1.0, 4.0):
            for mode in ("horizon", "wc"):
                vals = [Engine(tr, GPU_, link, cap, CAMPPrefetch(cm=make_cm(GPU_, bias=bias, online=False), mode=mode, wrap=True, pins=pins_x),
                               seed=s).run().steady() for s in range(8)]
                pf.append(dict(bias=bias, mode=mode, mean=statistics.mean(vals), ci=ci95(vals)))
        out["cases"].append(dict(name=name, ref=base, grid=rows, estimator=acc, prefetch=pf, lb=lower_bound(tr, cap, link, GPU_)))
        print("e5", name, flush=True)

    # ---- shape-dependent ground truth: GEMM efficiency ramps with arithmetic intensity ----
    gpu_r = replace(GPU_, ramp_ai0=1024.0)
    shapes = {"decode B=16": lambda g: trace_decode(M70, g, 16, 2048, DEC_STEPS),
              "prefill B=4 S=2k": lambda g: trace_prefill(M70, g, 4, 2048)}
    caps = {"decode B=16": weights_budget(M70, gpu_r, 16, 4096), "prefill B=4 S=2k": weights_budget(M70, gpu_r, 4, 2048)}
    for ev_name in shapes:
        tr_eval = shapes[ev_name](gpu_r)
        cap = caps[ev_name]
        for probe_name in shapes:
            tr_probe = shapes[probe_name](gpu_r)
            cm_probe = calibrated_cm(tr_probe, caps[probe_name], link, gpu_r)       # learned on the *probe* shape
            variants = {"generic (no calibration)": make_cm(gpu_r, online=False),
                        "calibrated on probe shape": cm_probe,
                        "constants-exact (no ramp knowledge)": exact_cm(gpu_r),
                        "oracle (true layer times)": TrueCM(gpu_r)}
            for vn, cm in variants.items():
                if probe_name != ev_name and vn not in ("calibrated on probe shape",):
                    continue
                P = pins_scp_verified(tr_eval, cap, link, gpu_r, cm)
                vals = [Engine(tr_eval, gpu_r, link, cap, CAMPPrefetch(cm=make_cm(gpu_r), pins=P), seed=s).run().steady() for s in range(8)]
                mape = statistics.mean(abs(cm.predict(a.feat, a.kind) - a.t_true) / a.t_true
                                       for a in tr_eval.acc[:tr_eval.iter_starts[1] if len(tr_eval.iter_starts) > 1 else len(tr_eval.acc)])
                out["shape"].append(dict(evaluate=ev_name, probe=probe_name, variant=vn, mean=statistics.mean(vals), ci=ci95(vals),
                                         mape=mape, lb=lower_bound(tr_eval, cap, link, gpu_r)))
        print("e5 shape", ev_name, flush=True)
    save("e5_cost_model", out)


# ----------------------------------------------------------------------------------------
# E6  throughput versus batch (GEMM-batch view and KV-feasible decode batching)
# ----------------------------------------------------------------------------------------
def e6():
    m = M70
    cap = 0.4 * m.weight_bytes
    Ts = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    rows = []
    for link in (LINK18, LINK45):
        for T in Ts:
            tr = workload_by_tokens(m, T)
            mc, cc, _ = run_many(tr, GPU_, link, cap, camp_v2(tr, cap, link, GPU_), seeds=range(6))
            md, cd, _ = run_many(tr, GPU_, link, cap, lambda s: NoPrefetch(), seeds=range(6))
            rows.append(dict(bw=link.bw_gbs, T=T, camp=mc, camp_ci=cc, demand=md, ideal=ideal_time(tr, GPU_)))
    # decode batching with the weight cache derived from HBM - KV(B, 512 tokens) - workspace (infeasible B are skipped)
    drows = []
    for link in (LINK18, LINK45):
        for B in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512):
            cap_b = weights_budget(m, GPU_, B, 512)
            kv_gb = B * 512 * m.kv_bytes_per_token() / 1e9
            tr = trace_decode(m, GPU_, B, 512, DEC_STEPS)
            full = {a.uid for a in tr.acc if a.fetch_bytes is None}
            smax = max(tr.units[u].nbytes for u in full)
            if cap_b < 2 * smax:
                drows.append(dict(bw=link.bw_gbs, B=B, feasible=False, kv_gb=kv_gb, cache_gb=cap_b / 1e9))
                continue
            mc, cc, _ = run_many(tr, GPU_, link, cap_b, camp_v2(tr, cap_b, link, GPU_), seeds=range(6))
            md, cd, _ = run_many(tr, GPU_, link, cap_b, lambda s: NoPrefetch(), seeds=range(6))
            drows.append(dict(bw=link.bw_gbs, B=B, feasible=True, kv_gb=kv_gb, cache_gb=cap_b / 1e9, cache_ratio=cap_b / m.weight_bytes,
                              camp=mc, camp_ci=cc, demand=md, ideal=ideal_time(tr, GPU_)))
    save("e6_batch_scaling", dict(model=m.name, rows=rows, decode_rows=drows))


# ----------------------------------------------------------------------------------------
# E7  model scale
# ----------------------------------------------------------------------------------------
def e7():
    rows = []
    names = ("Demand+LRU", "FlexGen-style", "Hot/Cold", "Stride", "CAMP-v2")
    for mn in ("llama2-7b", "llama2-13b", "llama3-70b", "llama3-405b"):
        m = PRESETS[mn]
        for wl, mk, ctxmax in (("decode B=16 ctx=2k", lambda: dec(m, 16, 2048, 4), 4096),
                               ("prefill B=4 S=2k", lambda: trace_prefill(m, GPU_, 4, 2048, reps=3), 2048)):
            B = 16 if "decode" in wl else 4
            cap = min(weights_budget(m, GPU_, B, ctxmax), m.weight_bytes)
            tr = mk()
            for link in (LINK18, LINK45):
                res = eval_suite(tr, cap, link, GPU_, seeds=range(6), names=names)
                info = res.pop("_info"); lb = res.pop("_lb")
                rows.append(dict(model=mn, params_b=m.total_params / 1e9, weight_gb=m.weight_bytes / 1e9, workload=wl, bw=link.bw_gbs,
                                 cache_gb=cap / 1e9, cache_ratio=cap / m.weight_bytes, ideal=ideal_time(tr, GPU_), lb=lb,
                                 tokens=(16 if "decode" in wl else 4 * 2048), results=res))
            print("e7", mn, wl, flush=True)
    save("e7_model_scale", rows)


# ----------------------------------------------------------------------------------------
# E8  link / path sensitivity (bandwidth, staged vs direct DMA path, cache ratio) -- 70B
# ----------------------------------------------------------------------------------------
def e8():
    out = {"bw": [], "path": [], "cache": []}
    names = ("Demand+LRU", "FlexGen-style", "Hot/Cold", "Stride", "CAMP-v2")
    for wl, tr, cap in (("decode B=16", dec(M70, 16, 2048), weights_budget(M70, GPU_, 16, 4096)),
                        ("prefill B=4 S=4k", trace_prefill(M70, GPU_, 4, 4096), weights_budget(M70, GPU_, 4, 4096))):
        for bw in (8, 10, 12, 18, 24, 32, 45, 64):
            link = with_bw(LINK18, bw)
            res = eval_suite(tr, cap, link, GPU_, seeds=range(6), names=names)
            res.pop("_info")
            out["bw"].append(dict(workload=wl, bw=bw, lb=res.pop("_lb"), ideal=ideal_time(tr, GPU_),
                                  **{k: v["mean"] for k, v in res.items()}))
        for link in (LINKS["cxl_x8_measured"], LINKS["cxl_x8_staged"], LINKS["cxl_x16_expected"], replace(LINKS["cxl_x16_expected"], mode="staged", name="x16_staged")):
            res = eval_suite(tr, cap, link, GPU_, seeds=range(6), names=("Demand+LRU", "CAMP-v2"))
            res.pop("_info"); res.pop("_lb")
            out["path"].append(dict(workload=wl, link=link.name, mode=link.mode, eff_bw=link.effective_bw() / 1e9,
                                    **{k: v["mean"] for k, v in res.items()}))
        for ratio in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0):
            cp = ratio * M70.weight_bytes
            res = eval_suite(tr, cp, LINK18, GPU_, seeds=range(6), names=names)
            res.pop("_info")
            out["cache"].append(dict(workload=wl, ratio=ratio, lb=res.pop("_lb"), ideal=ideal_time(tr, GPU_),
                                     **{k: v["mean"] for k, v in res.items()}))
        print("e8", wl, flush=True)
    save("e8_link_sensitivity", out)


# ----------------------------------------------------------------------------------------
# E9  simulator robustness: noise levels, planner-vs-engine, link-model mismatch
# ----------------------------------------------------------------------------------------
def e9():
    out = {"noise": [], "planner": [], "linkbias": []}
    cfgs = (("decode B=16", dec(M70, 16, 2048), weights_budget(M70, GPU_, 16, 4096)),
            ("prefill B=4 S=2k", trace_prefill(M70, GPU_, 4, 2048), weights_budget(M70, GPU_, 4, 2048)))
    for wl, tr, cap in cfgs:
        for js, ds, cs in ((0.0, 0.0, 0.0), (0.03, 0.03, 0.02), (0.06, 0.05, 0.03), (0.12, 0.10, 0.06), (0.25, 0.20, 0.12), (0.5, 0.4, 0.25)):
            link = replace(LINK18, jitter_sigma=js, drift_sigma=ds)
            P = pins_scp_verified(tr, cap, link, GPU_, calibrated_cm(tr, cap, link, GPU_))
            seq = build_plan_seq(tr, 0, make_cm(GPU_))
            Pst = pins_stride(seq, cap, gamma=1.0, slots=1.0)
            tfg, cfg_, _, kfg, sfg = flexgen_best(tr, cap, link, GPU_)
            row = dict(workload=wl, jitter=js, drift=ds, compute=cs)
            for nm, trx, capx, mk in (("CAMP-v2", tr, cap, lambda s: CAMPPrefetch(cm=make_cm(GPU_), pins=P)),
                                      ("FlexGen-style", tfg, cfg_, lambda s: StaticK(kfg, wrap=True, evict="fifo")),
                                      ("Stride", tr, cap, lambda s: CAMPPrefetch(cm=make_cm(GPU_), pins=Pst)),
                                      ("Demand+LRU", tr, cap, lambda s: NoPrefetch())):
                vals = [Engine(trx, GPU_, link, capx, mk(s), seed=s, compute_sigma=cs).run().steady() for s in range(12)]
                row[nm] = dict(mean=statistics.mean(vals), ci=ci95(vals), cv=statistics.stdev(vals) / statistics.mean(vals))
            out["noise"].append(row)
        print("e9 noise", wl, flush=True)
    # planner vs engine over random configurations (consistency check of two implementations of one model)
    rng = random.Random(5)
    models = [PRESETS[k] for k in ("llama2-7b", "llama2-13b", "llama3-8b", "llama3-70b", "gemma2-9b")]
    for _ in range(80):
        mm = rng.choice(models)
        kind = rng.choice(["decode", "prefill"])
        if kind == "decode":
            tr = trace_decode(mm, GPU_, rng.choice([1, 4, 16, 64, 256]), rng.choice([512, 2048]), 4)
        else:
            tr = trace_prefill(mm, GPU_, rng.choice([1, 2, 4]), rng.choice([512, 1024, 2048, 4096]), reps=4)
        cap = rng.uniform(0.15, 0.7) * mm.weight_bytes
        link = with_bw(LINK18, rng.choice([8, 12, 18, 32, 45, 64]))
        gap = GPU_.sampling_gap_us * 1e-6
        seq = build_plan_seq(tr, 0, exact_cm(GPU_))
        P = pins_scp(seq, cap, link, gap)
        pred = plan_time(seq, P, cap, link, gap)
        sim = Engine(tr, GPU_, link, cap, CAMPPrefetch(cm=exact_cm(GPU_), pins=P), seed=0, deterministic=True).run().steady()
        out["planner"].append(dict(model=mm.name, kind=kind, pred=pred, sim=sim, err_pct=100 * (pred - sim) / sim))
    errs = [abs(r["err_pct"]) for r in out["planner"]]
    out["planner_mape"] = statistics.mean(errs)
    out["planner_median"] = statistics.median(errs)
    out["planner_max"] = max(errs)
    # link-model mismatch: plan pins with a wrong bandwidth / latency, evaluate on the true link
    for wl, tr, cap in cfgs:
        cm = calibrated_cm(tr, cap, LINK18, GPU_)
        ref = None
        for fbw in (0.7, 0.85, 1.0, 1.15, 1.3):
            for flat in (1.0, 4.0):
                lk_plan = replace(LINK18, bw_gbs=LINK18.bw_gbs * fbw, latency_us=LINK18.latency_us * flat)
                P = pins_scp_verified(tr, cap, lk_plan, GPU_, cm)
                vals = [Engine(tr, GPU_, LINK18, cap, CAMPPrefetch(cm=make_cm(GPU_), pins=P), seed=s).run().steady() for s in range(8)]
                out["linkbias"].append(dict(workload=wl, bw_factor=fbw, lat_factor=flat, mean=statistics.mean(vals), ci=ci95(vals), n_pin=len(P)))
        print("e9 linkbias", wl, flush=True)
    print("e9 planner MAPE", out["planner_mape"], flush=True)
    save("e9_robustness", out)


# ----------------------------------------------------------------------------------------
# E10  baseline attribution: from the conventional Hot/Cold baseline to CAMP, one change at a time
# ----------------------------------------------------------------------------------------
def e10():
    out = []
    cfgs = [("llama3-70b", "decode B=16", dec(M70, 16, 2048), weights_budget(M70, GPU_, 16, 4096)),
            ("llama3-70b", "prefill B=4 S=2k", trace_prefill(M70, GPU_, 4, 2048), weights_budget(M70, GPU_, 4, 2048)),
            ("llama3-70b", "prefill B=4 S=4k", trace_prefill(M70, GPU_, 4, 4096), weights_budget(M70, GPU_, 4, 4096)),
            ("llama2-13b", "prefill B=4 S=4k", trace_prefill(M13, GPU_, 4, 4096), 0.4 * M13.weight_bytes),
            ("llama2-13b", "decode B=16", dec(M13, 16, 2048), 0.4 * M13.weight_bytes)]
    for (mn, wl, tr, cap) in cfgs:
        for link in (LINK18, LINK45):
            seq = build_plan_seq(tr, 0, make_cm(GPU_))
            cm_cal = calibrated_cm(tr, cap, link, GPU_)
            steps = {
                "0 conventional Hot/Cold (2-slot reserve, position tie-break, lookahead 2, LRU)":
                    (tr, cap, lambda s: StaticK(2, pins=pins_freq(seq, cap))),
                "1 + equal budget (1-slot reserve, pin up to cap - 1 slot)":
                    (tr, cap, lambda s: StaticK(2, pins=pins_freq(seq, cap, gamma=1.0, slots=1.0))),
                "2 + random tie-break":
                    (tr, cap, lambda s: StaticK(2, pins=pins_freq(seq, cap, gamma=1.0, slots=1.0, tiebreak="random"))),
                "3 + FIFO eviction and cross-iteration lookahead":
                    (tr, cap, lambda s: StaticK(2, wrap=True, evict="fifo", pins=pins_freq(seq, cap, gamma=1.0, slots=1.0, tiebreak="random"))),
                "4 + stride (interleaved) pins instead of random":
                    (tr, cap, lambda s: StaticK(2, wrap=True, evict="fifo", pins=pins_stride(seq, cap, gamma=1.0, slots=1.0))),
                "5 + CAMP prefetcher (work-conserving, next-use admission)":
                    (tr, cap, lambda s: CAMPPrefetch(cm=make_cm(GPU_), pins=pins_stride(seq, cap, gamma=1.0, slots=1.0))),
                "6 + SCP pins (planner, dry-run verification, calibrated cost model)":
                    (tr, cap, lambda s, P=pins_scp_verified(tr, cap, link, GPU_, cm_cal): CAMPPrefetch(cm=make_cm(GPU_), pins=P)),
            }
            tfg, cfg_, phi, kfg, sfg = flexgen_best(tr, cap, link, GPU_)
            steps["ref FlexGen-style per-layer split (tuned buffers)"] = (tfg, cfg_, lambda s: StaticK(kfg, wrap=True, evict="fifo"))
            row = dict(model=mn, workload=wl, bw=link.bw_gbs, lb=lower_bound(tr, cap, link, GPU_), steps={})
            for k, (trx, capx, mk) in steps.items():
                vals = [Engine(trx, GPU_, link, capx, mk(s), seed=s).run().steady() for s in SEEDS]
                row["steps"][k] = dict(mean=statistics.mean(vals), ci=ci95(vals))
            out.append(row)
            print("e10", mn, wl, link.bw_gbs, flush=True)
    save("e10_attribution", out)


# ----------------------------------------------------------------------------------------
# E11  alternatives that remove the need for CXL streaming: quantisation, host DRAM, second GPU
# ----------------------------------------------------------------------------------------
def e11():
    rows = []
    for mn in ("llama3-70b", "llama3-405b"):
        for bpp, qn in ((2, "FP16"), (1, "FP8/INT8 (weight-only)"), (0.5, "INT4 (weight-only)")):
            m = replace(PRESETS[mn], bpp=bpp)
            for wl in ("decode B=16 ctx=2k", "prefill B=4 S=2k"):
                B = 16 if "decode" in wl else 4
                ctxmax = 4096 if "decode" in wl else 2048
                cap = min(weights_budget(m, GPU_, B, ctxmax), m.weight_bytes)
                tr = dec(m, 16, 2048, 4) if "decode" in wl else trace_prefill(m, GPU_, 4, 2048, reps=3)
                tokens = 16 if "decode" in wl else 4 * 2048
                ideal = ideal_time(tr, GPU_)
                fits = m.weight_bytes <= cap + 1
                kv_gb = B * ctxmax * m.kv_bytes_per_token() / 1e9
                two_gpu_fits = m.weight_bytes / 1e9 + kv_gb + 8 <= 160
                row = dict(model=mn, quant=qn, workload=wl, weight_gb=m.weight_bytes / 1e9, kv_gb=kv_gb, cache_gb=cap / 1e9, fits_one_gpu=fits,
                           ideal=ideal, tokens=tokens, two_gpu_fits=two_gpu_fits, two_gpu_latency_optimistic=ideal / 2 if two_gpu_fits else None)
                for ln, link in (("cxl18", LINK18), ("cxl45", LINK45), ("hostdram52", HOSTDRAM)):
                    if fits:
                        row[ln] = ideal
                    else:
                        mc, cc, _ = run_many(tr, GPU_, link, cap, camp_v2(tr, cap, link, GPU_), seeds=range(4))
                        row[ln] = mc
                rows.append(row)
        print("e11", mn, flush=True)
    save("e11_alternatives", rows)


ALL = dict(e1=e1, e2=e2, e3=e3, e4=e4, e5=e5, e6=e6, e7=e7, e8=e8, e9=e9, e10=e10, e11=e11)

if __name__ == "__main__":
    which = sys.argv[1:] or list(ALL)
    for k in which:
        t0 = time.time()
        ALL[k]()
        print(f"== {k} done in {time.time()-t0:.0f}s", flush=True)
