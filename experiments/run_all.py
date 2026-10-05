"""
Run the CAMP evaluation.  Usage:  python experiments/run_all.py [e1 e2 ...]   (default: all)

Every experiment writes results/<name>.json; figures are produced by experiments/plots.py.
All numbers quoted in the paper come from these files.
"""
import itertools
import json
import os
import random
import sys
import time
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *  # noqa
from camp_sim.models import ModelSpec, trace_specdec, trace_encdec, build_units
from camp_sim.policies import pins_exhaustive, pins_random, pins_greedy_marginal, pins_scp_verified

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
GPU_ = H100
LINK18 = LINKS["cxl_x8_measured"]
LINK45 = LINKS["cxl_x16_expected"]
SEEDS = range(10)


def save(name, obj):
    with open(os.path.join(RES, name + ".json"), "w") as f:
        json.dump(obj, f, indent=1)
    print(f"[saved] {name}.json")


def ideal_time(trace, gpu):
    """All-weights-resident time per iteration (no CXL traffic): compute only + sampling gap."""
    n0 = trace.iter_starts[1] if len(trace.iter_starts) > 1 else len(trace.acc)
    return sum(a.t_true for a in trace.acc[:n0]) + gpu.sampling_gap_us * 1e-6


def camp_v2(trace, cap, link, gpu, **kw):
    cm = make_cm(gpu)
    seq = build_plan_seq(trace, 0, cm)
    pins = pins_scp_verified(trace, cap, link, gpu, cm)
    return lambda s: CAMPPrefetch(cm=make_cm(gpu), pins=pins, **kw)


# ----------------------------------------------------------------------------------------
# E1  regime map: when does streaming from CXL pay off?  (tokens per iteration x link BW)
# ----------------------------------------------------------------------------------------
def workload_by_tokens(m, T):
    """One iteration that processes T tokens through the GEMM layers (no long KV context), so that the
    x-axis isolates the weight-streaming arithmetic intensity (FLOP per streamed byte = T)."""
    if T <= 16384:
        return trace_prefill(m, GPU_, 1, T, reps=3)
    return trace_prefill(m, GPU_, T // 16384, 16384, reps=3)


def e1():
    m = PRESETS["llama2-13b"]
    cap = 0.4 * m.weight_bytes
    Ts = [1, 4, 16, 64, 256, 1024, 4096, 16384, 65536]
    BWs = [8, 12, 18, 24, 32, 45, 64]
    cells = []
    for T in Ts:
        tr = workload_by_tokens(m, T)
        ideal = ideal_time(tr, GPU_)
        for bw in BWs:
            link = with_bw(LINK18, bw)
            f_camp = camp_v2(tr, cap, link, GPU_)
            mc, cc, r = run_many(tr, GPU_, link, cap, f_camp, seeds=range(4))
            md, cd, _ = run_many(tr, GPU_, link, cap, lambda s: NoPrefetch(), seeds=range(4))
            mh, ch, _ = run_many(tr, GPU_, link, cap,
                                 lambda s: StaticK(2, pins=pins_freq(build_plan_seq(tr, 0, make_cm(GPU_)), cap)), seeds=range(4))
            cells.append(dict(T=T, bw=bw, ideal=ideal, camp=mc, demand=md, hotcold=mh, lb=lower_bound(tr, cap, link, GPU_),
                              stall_frac=r.steady_stall() / mc))
        print("e1 T", T)
    save("e1_regime_map", dict(model=m.name, cache_ratio=0.4, Ts=Ts, BWs=BWs, cells=cells))


# ----------------------------------------------------------------------------------------
# E2  main comparison on prefill / decode / mixed workloads
# ----------------------------------------------------------------------------------------
def e2():
    out = []
    cfgs = []
    m13, m70, mg = PRESETS["llama2-13b"], PRESETS["llama3-70b"], PRESETS["gemma2-9b"]
    cfgs += [("llama2-13b", "decode B=16 ctx=2k", trace_decode(m13, GPU_, 16, 2048, 4), 0.4 * m13.weight_bytes),
             ("llama2-13b", "mixed 64 dec + 512 chunk", trace_mixed(m13, GPU_, 64, 2048, 512, 4), 0.4 * m13.weight_bytes),
             ("llama2-13b", "prefill B=1 S=2k", trace_prefill(m13, GPU_, 1, 2048), 0.4 * m13.weight_bytes),
             ("llama2-13b", "prefill B=4 S=4k", trace_prefill(m13, GPU_, 4, 4096), 0.4 * m13.weight_bytes)]
    cfgs += [("llama3-70b", "decode B=16 ctx=2k", trace_decode(m70, GPU_, 16, 2048, 4), weights_budget(m70, GPU_, 16, 4096)),
             ("llama3-70b", "prefill B=4 S=2k", trace_prefill(m70, GPU_, 4, 2048), weights_budget(m70, GPU_, 4, 2048)),
             ("llama3-70b", "prefill B=4 S=4k", trace_prefill(m70, GPU_, 4, 4096), weights_budget(m70, GPU_, 4, 4096))]
    cfgs += [("gemma2-9b (tied)", "decode B=32 ctx=2k", trace_decode(mg, GPU_, 32, 2048, 4), 0.4 * mg.weight_bytes)]
    for (mn, wl, tr, cap) in cfgs:
        for link in (LINK18, LINK45):
            t0 = time.time()
            res = eval_suite(tr, cap, link, GPU_, seeds=SEEDS)
            info = res.pop("_info")
            lb = res.pop("_lb")
            out.append(dict(model=mn, workload=wl, link=link.name, bw=link.bw_gbs, cache_gb=cap / 1e9,
                            cache_ratio=cap / sum(u.nbytes for u in tr.units.values()), ideal=ideal_time(tr, GPU_),
                            lb=lb, info=dict(best_k=info["best_k"], n_pin=info["n_pin"], pin_gb=info["pin_gb"],
                                             scp_start=info["scp"]["picked"]), results=res))
            print(f"e2 {mn:18s} {wl:28s} {link.name:20s} {time.time()-t0:.1f}s")
    save("e2_main", out)


# ----------------------------------------------------------------------------------------
# E3  pinning study: what does the pin-selection rule buy?
# ----------------------------------------------------------------------------------------
def pin_strategies(trace, cap, link, gpu):
    gap = gpu.sampling_gap_us * 1e-6
    seq = build_plan_seq(trace, 0, make_cm(gpu))
    S = {"none": set(), "prefix (FlexGen-style)": pins_first(seq, cap), "frequency (orig. Alg. 2)": pins_freq(seq, cap),
         "random": pins_random(seq, cap), "stride (interleaved)": pins_stride(seq, cap),
         "knapsack-DP (Eq. 2)": pins_knapsack_dp(seq, cap, link), "marginal greedy": pins_greedy_marginal(seq, cap, link, gap),
         "SCP (planner only)": pins_scp(seq, cap, link, gap),
         "SCP (plan+verify)": pins_scp_verified(trace, cap, link, gpu, calibrated_cm(trace, cap, link, gpu))}
    best_g, best_t = None, 1e18
    for g in (0.3, 0.5, 0.7, 0.8, 0.9, 0.95):
        P = pins_freq(seq, cap, g)
        t = plan_time(seq, P, cap, link, gap)
        if t < best_t:
            best_g, best_t, Pb = g, t, P
    S[f"frequency, best gamma={best_g}"] = Pb
    return S


def e3():
    out = []
    m13, m70 = PRESETS["llama2-13b"], PRESETS["llama3-70b"]
    wl = [("llama2-13b", "decode B=16", trace_decode(m13, GPU_, 16, 2048, 4), 0.4 * m13.weight_bytes),
          ("llama2-13b", "prefill B=1 S=2k", trace_prefill(m13, GPU_, 1, 2048), 0.4 * m13.weight_bytes),
          ("llama2-13b", "prefill B=4 S=4k", trace_prefill(m13, GPU_, 4, 4096), 0.4 * m13.weight_bytes),
          ("llama2-13b", "prefill B=8 S=4k", trace_prefill(m13, GPU_, 8, 4096), 0.4 * m13.weight_bytes),
          ("llama3-70b", "prefill B=4 S=2k", trace_prefill(m70, GPU_, 4, 2048), weights_budget(m70, GPU_, 4, 2048)),
          ("llama3-70b", "prefill B=4 S=4k", trace_prefill(m70, GPU_, 4, 4096), weights_budget(m70, GPU_, 4, 4096))]
    for (mn, name, tr, cap) in wl:
        for link in (LINK18, LINK45):
            S = pin_strategies(tr, cap, link, GPU_)
            row = dict(model=mn, workload=name, bw=link.bw_gbs, lb=lower_bound(tr, cap, link, GPU_), ideal=ideal_time(tr, GPU_), res={})
            for k, P in S.items():
                vals = []
                for s in SEEDS:
                    vals.append(Engine(tr, GPU_, link, cap, CAMPPrefetch(cm=make_cm(GPU_), pins=P), seed=s).run().steady())
                row["res"][k] = dict(mean=statistics.mean(vals), ci=ci95(vals), vals=vals, n_pin=len(P),
                                     pin_gb=sum(tr.units[u].nbytes for u in P) / 1e9)
            out.append(row)
            print("e3", mn, name, link.bw_gbs)
    save("e3_pin_strategies", out)

    # ---- optimality gap on a small instance (exhaustive search over all pin sets) ----
    toy = ModelSpec("toy-8L", 8, 2048, 16, 16, 128, 8192, 32000)
    gap_rows = []
    gap = GPU_.sampling_gap_us * 1e-6
    for (name, tr_f) in [("decode B=16", lambda: trace_decode(toy, GPU_, 16, 2048, 3)),
                         ("prefill B=1 S=2k", lambda: trace_prefill(toy, GPU_, 1, 2048)),
                         ("prefill B=4 S=4k", lambda: trace_prefill(toy, GPU_, 4, 4096)),
                         ("prefill B=8 S=2k", lambda: trace_prefill(toy, GPU_, 8, 2048))]:
        tr = tr_f()
        for ratio in (0.3, 0.5, 0.7):
            cap = ratio * toy.weight_bytes
            link = with_bw(LINK18, 4.0)   # scaled so that the small model sits in the same regime as the large ones
            seq = build_plan_seq(tr, 0, make_cm(GPU_))
            t0 = time.time()
            Pe = pins_exhaustive(seq, cap, link, gap, max_units=18)
            Ps, info = pins_scp(seq, cap, link, gap, return_info=True)
            te, ts = plan_time(seq, Pe, cap, link, gap), plan_time(seq, Ps, cap, link, gap)
            tf = plan_time(seq, pins_freq(seq, cap), cap, link, gap)
            sim = {}
            for k, P in (("exhaustive", Pe), ("scp", Ps), ("freq", pins_freq(seq, cap))):
                vals = [Engine(tr, GPU_, link, cap, CAMPPrefetch(cm=make_cm(GPU_), pins=P), seed=s).run().steady() for s in range(6)]
                sim[k] = statistics.mean(vals)
            gap_rows.append(dict(workload=name, cache_ratio=ratio, planner=dict(exhaustive=te, scp=ts, freq=tf),
                                 sim=sim, scp_gap_pct=100 * (ts - te) / te, sim_gap_pct=100 * (sim["scp"] - sim["exhaustive"]) / sim["exhaustive"],
                                 freq_gap_pct=100 * (tf - te) / te, search_s=time.time() - t0))
            print("e3 gap", name, ratio, round(gap_rows[-1]["scp_gap_pct"], 2), round(gap_rows[-1]["freq_gap_pct"], 2))
    save("e3_optimality_gap", gap_rows)


# ----------------------------------------------------------------------------------------
# E4  non-uniform reuse (tied embedding, speculative decoding, encoder-decoder) and eviction
# ----------------------------------------------------------------------------------------
def e4():
    out = {}
    # (a) eviction policies on a uniform dense decode loop (no pins): who escapes cyclic thrash?
    m13 = PRESETS["llama2-13b"]
    tr = trace_decode(m13, GPU_, 16, 2048, 6)
    cap = 0.4 * m13.weight_bytes
    rows = []
    for ev in ("lru", "fifo", "lfu", "random", "belady"):
        for pf, mk in (("demand", lambda e: NoPrefetch(evict=e)), ("static-2", lambda e: StaticK(2, evict=e))):
            vals = [Engine(tr, GPU_, LINK18, cap, mk(ev), seed=s).run().steady() for s in range(6)]
            r = Engine(tr, GPU_, LINK18, cap, mk(ev), seed=0).run()
            rows.append(dict(evict=ev, prefetch=pf, mean=statistics.mean(vals), ci=ci95(vals), hit_rate=r.hits / max(1, r.accesses)))
    out["eviction_uniform"] = rows

    # (b) workloads with real reuse heterogeneity
    link = LINK18
    gap = GPU_.sampling_gap_us * 1e-6
    het = []
    mg = PRESETS["gemma2-9b"]
    d1b = ModelSpec("draft-1b", 16, 2048, 16, 16, 128, 5632, 32000)
    enc = ModelSpec("enc-24L", 24, 4096, 32, 32, 128, 10240, 32000, gated=False)
    dec = ModelSpec("dec-24L", 24, 4096, 32, 32, 128, 10240, 32000, gated=False)
    cases = [("gemma2-9b tied decode", trace_decode(mg, GPU_, 32, 2048, 6), 0.4, 6),
             ("spec. decoding (13B target + 1B draft, k=4)", trace_specdec(m13, d1b, GPU_, 8, 2048, 4, 4), 0.35, 4),
             ("encoder-decoder (24+24 layers, 64 output tokens)", trace_encdec(enc, dec, GPU_, 8, 2048, 64, 2), 0.35, 65)]
    for (name, tr, ratio, ips) in cases:
        total = sum(u.nbytes for u in tr.units.values())
        cap = ratio * total
        seq = build_plan_seq(tr, 0, make_cm(GPU_))
        if tr.meta["kind"] == "encdec":
            # plan over the whole request (encoder pass + all decoder steps)
            seq = []
            for i in range(tr.iter_starts[1] if False else 0, 0):
                pass
            end = tr.iter_starts[1 + 64] if len(tr.iter_starts) > 65 else len(tr.acc)
            cm = make_cm(GPU_)
            seq = [(a.uid, tr.units[a.uid].nbytes, cm.predict(a.feat, a.kind), a.fetch_bytes) for a in tr.acc[:end]]
        P = {"none": set(), "prefix": pins_first(seq, cap), "frequency": pins_freq(seq, cap),
             "knapsack-DP": pins_knapsack_dp(seq, cap, link),
             "SCP": (pins_scp(seq, cap, link, gap) if tr.meta["kind"] == "encdec" else pins_scp_verified(tr, cap, link, GPU_, calibrated_cm(tr, cap, link, GPU_)))}
        row = dict(case=name, cache_ratio=ratio, res={})

        def per_unit(r):
            # latency per decoded token-step (specdec/tied) or per request (encdec)
            if tr.meta["kind"] == "encdec":
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
        het.append(row)
        print("e4", name)
    out["heterogeneous"] = het
    save("e4_reuse_heterogeneity", out)


# ----------------------------------------------------------------------------------------
# E5  runtime cost-model sensitivity (answers: how is T_comp obtained, and does it matter?)
# ----------------------------------------------------------------------------------------
def e5():
    gap = GPU_.sampling_gap_us * 1e-6
    out = {"cases": []}
    m13, m70 = PRESETS["llama2-13b"], PRESETS["llama3-70b"]
    wl = [("llama2-13b prefill B=4 S=4k", trace_prefill(m13, GPU_, 4, 4096), 0.4 * m13.weight_bytes),
          ("llama3-70b prefill B=4 S=2k", trace_prefill(m70, GPU_, 4, 2048), weights_budget(m70, GPU_, 4, 2048)),
          ("llama2-13b decode B=16", trace_decode(m13, GPU_, 16, 2048, 4), 0.4 * m13.weight_bytes)]
    link = LINK18
    for (name, tr, cap) in wl:
        base = {}
        # reference points
        P_exact = pins_scp_verified(tr, cap, link, GPU_, exact_cm(GPU_))
        P_nom = pins_scp_verified(tr, cap, link, GPU_, make_cm(GPU_))
        P_freq = pins_freq(build_plan_seq(tr, 0, make_cm(GPU_)), cap)

        def lat(P, mk=lambda: make_cm(GPU_)):
            vals = [Engine(tr, GPU_, link, cap, CAMPPrefetch(cm=mk(), pins=P), seed=s).run().steady() for s in range(8)]
            return statistics.mean(vals), ci95(vals)

        base["exact"] = lat(P_exact)
        base["nominal"] = lat(P_nom)
        base["freq"] = lat(P_freq)
        base["no_pins"] = lat(set())
        rows = []
        for bias in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0):
            for sigma in (0.0, 0.25, 0.5):
                for online in (False, True):
                    cm = make_cm(GPU_, bias=bias, noise_sigma=sigma, online=online, seed=3)
                    P = pins_scp_verified(tr, cap, link, GPU_, cm)
                    if online:
                        # one probe iteration to calibrate, then re-plan with the corrected model
                        probe = trace_slice(tr, 1)
                        cmp_ = make_cm(GPU_, bias=bias, noise_sigma=0.0, online=True)
                        Engine(probe, GPU_, link, cap, CAMPPrefetch(cm=cmp_, pins=P), seed=0).run()
                        cm2 = make_cm(GPU_, bias=bias, noise_sigma=sigma, online=True, seed=3)
                        cm2.corr = dict(cmp_.corr)
                        P = pins_scp_verified(tr, cap, link, GPU_, cm2)
                    m_, c_ = lat(P)
                    rows.append(dict(bias=bias, sigma=sigma, online=online, mean=m_, ci=c_, n_pin=len(P)))
        # estimator accuracy (per-layer MAPE of predicted vs true compute)
        acc = {}
        for online in (False, True):
            cm = make_cm(GPU_, online=online)
            errs0 = [abs(cm.predict(a.feat, a.kind) - a.t_true) / a.t_true for a in tr.acc[: tr.iter_starts[1] if len(tr.iter_starts) > 1 else len(tr.acc)]]
            if online:
                Engine(trace_slice(tr, 1), GPU_, link, cap, CAMPPrefetch(cm=cm, pins=set()), seed=0).run()
            errs1 = [abs(cm.predict(a.feat, a.kind) - a.t_true) / a.t_true for a in tr.acc[: tr.iter_starts[1] if len(tr.iter_starts) > 1 else len(tr.acc)]]
            acc["online" if online else "static"] = dict(before=statistics.mean(errs0), after=statistics.mean(errs1), corr=dict(cm.corr))
        # prefetch horizon (orig. Alg. 1) vs work-conserving under estimator error
        pf = []
        for bias in (0.25, 0.5, 1.0, 2.0, 4.0):
            for mode in ("horizon", "wc"):
                vals = [Engine(tr, GPU_, link, cap, CAMPPrefetch(cm=make_cm(GPU_, bias=bias, online=False), mode=mode, wrap=True, pins=P_exact),
                               seed=s).run().steady() for s in range(8)]
                pf.append(dict(bias=bias, mode=mode, mean=statistics.mean(vals), ci=ci95(vals)))
        out["cases"].append(dict(name=name, ref=base, grid=rows, estimator=acc, prefetch=pf, lb=lower_bound(tr, cap, link, GPU_)))
        print("e5", name)
    save("e5_cost_model", out)


# ----------------------------------------------------------------------------------------
# E6  throughput vs. tokens per iteration (batch amortisation) and E7 model scale
# ----------------------------------------------------------------------------------------
def e6():
    m = PRESETS["llama2-13b"]
    cap = 0.4 * m.weight_bytes
    Ts = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    rows = []
    for link in (LINK18, LINK45):
        for T in Ts:
            tr = workload_by_tokens(m, T)
            mc, cc, _ = run_many(tr, GPU_, link, cap, camp_v2(tr, cap, link, GPU_), seeds=range(6))
            md, cd, _ = run_many(tr, GPU_, link, cap, lambda s: NoPrefetch(), seeds=range(6))
            rows.append(dict(bw=link.bw_gbs, T=T, camp=mc, camp_ci=cc, demand=md, demand_ci=cd, ideal=ideal_time(tr, GPU_)))
    # decode batch scaling: batch of B sequences, 512-token contexts, KV cache resident in HBM
    drows = []
    for link in (LINK18, LINK45):
        for B in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512):
            tr = trace_decode(m, GPU_, B, 512, 3)
            mc, cc, _ = run_many(tr, GPU_, link, cap, camp_v2(tr, cap, link, GPU_), seeds=range(6))
            md, cd, _ = run_many(tr, GPU_, link, cap, lambda s: NoPrefetch(), seeds=range(6))
            drows.append(dict(bw=link.bw_gbs, B=B, camp=mc, camp_ci=cc, demand=md, ideal=ideal_time(tr, GPU_)))
    save("e6_batch_scaling", dict(model=m.name, rows=rows, decode_rows=drows))


def e7():
    rows = []
    for mn in ("llama2-7b", "llama2-13b", "llama3-70b", "llama3-405b"):
        m = PRESETS[mn]
        for wl, mk, ctxmax in (("decode B=16 ctx=2k", lambda: trace_decode(m, GPU_, 16, 2048, 4), 4096),
                               ("prefill B=4 S=2k", lambda: trace_prefill(m, GPU_, 4, 2048), 2048)):
            B = 16 if "decode" in wl else 4
            cap = min(weights_budget(m, GPU_, B, ctxmax), m.weight_bytes)
            tr = mk()
            for link in (LINK18, LINK45):
                res = eval_suite(tr, cap, link, GPU_, seeds=range(6),
                                 names=("Demand+LRU", "Hot/Cold (PowerInfer-style)", "FlexGen-style", "CAMP-v2"))
                info = res.pop("_info"); lb = res.pop("_lb")
                rows.append(dict(model=mn, params_b=m.total_params / 1e9, weight_gb=m.weight_bytes / 1e9, workload=wl, bw=link.bw_gbs,
                                 cache_gb=cap / 1e9, cache_ratio=cap / m.weight_bytes, ideal=ideal_time(tr, GPU_), lb=lb,
                                 tokens=(16 if "decode" in wl else 4 * 2048), results=res))
            print("e7", mn, wl)
    save("e7_model_scale", rows)


# ----------------------------------------------------------------------------------------
# E8  link / path sensitivity (bandwidth, staged vs direct DMA path, cache ratio)
# ----------------------------------------------------------------------------------------
def e8():
    m = PRESETS["llama2-13b"]
    out = {"bw": [], "path": [], "cache": []}
    cap = 0.4 * m.weight_bytes
    for wl, tr in (("decode B=16", trace_decode(m, GPU_, 16, 2048, 4)), ("prefill B=4 S=4k", trace_prefill(m, GPU_, 4, 4096))):
        for bw in (8, 10, 12, 18, 24, 32, 45, 64):
            link = with_bw(LINK18, bw)
            res = eval_suite(tr, cap, link, GPU_, seeds=range(6), names=("Demand+LRU", "Hot/Cold (PowerInfer-style)", "CAMP-v2"))
            res.pop("_info")
            out["bw"].append(dict(workload=wl, bw=bw, lb=res.pop("_lb"), ideal=ideal_time(tr, GPU_),
                                  **{k: v["mean"] for k, v in res.items()}))
        for link in (LINKS["cxl_x8_measured"], LINKS["cxl_x8_staged"], LINKS["cxl_x16_expected"], replace(LINKS["cxl_x16_expected"], mode="staged", name="x16_staged")):
            res = eval_suite(tr, cap, link, GPU_, seeds=range(6), names=("Demand+LRU", "CAMP-v2"))
            res.pop("_info"); res.pop("_lb")
            out["path"].append(dict(workload=wl, link=link.name, mode=link.mode, eff_bw=link.effective_bw() / 1e9,
                                    **{k: v["mean"] for k, v in res.items()}))
        for ratio in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0):
            cp = ratio * m.weight_bytes
            res = eval_suite(tr, cp, LINK18, GPU_, seeds=range(6), names=("Demand+LRU", "Hot/Cold (PowerInfer-style)", "CAMP-v2"))
            res.pop("_info")
            out["cache"].append(dict(workload=wl, ratio=ratio, lb=res.pop("_lb"), ideal=ideal_time(tr, GPU_),
                                     **{k: v["mean"] for k, v in res.items()}))
        print("e8", wl)
    save("e8_link_sensitivity", out)


# ----------------------------------------------------------------------------------------
# E9  simulator robustness: noise levels, and planner-vs-engine consistency
# ----------------------------------------------------------------------------------------
def e9():
    m = PRESETS["llama2-13b"]
    cap = 0.4 * m.weight_bytes
    out = {"noise": [], "planner": []}
    for wl, tr in (("decode B=16", trace_decode(m, GPU_, 16, 2048, 4)), ("prefill B=4 S=4k", trace_prefill(m, GPU_, 4, 4096))):
        for js, ds, cs in ((0.0, 0.0, 0.0), (0.03, 0.03, 0.02), (0.06, 0.05, 0.03), (0.12, 0.10, 0.06), (0.25, 0.20, 0.12), (0.5, 0.4, 0.25)):
            link = replace(LINK18, jitter_sigma=js, drift_sigma=ds)
            seq = build_plan_seq(tr, 0, make_cm(GPU_))
            P = pins_scp_verified(tr, cap, link, GPU_, calibrated_cm(tr, cap, link, GPU_))
            Pf = pins_freq(seq, cap)
            row = dict(workload=wl, jitter=js, drift=ds, compute=cs)
            for nm, mk in (("CAMP-v2", lambda s: CAMPPrefetch(cm=make_cm(GPU_), pins=P)),
                           ("Hot/Cold", lambda s: StaticK(2, pins=Pf)), ("Demand+LRU", lambda s: NoPrefetch())):
                vals = [Engine(tr, GPU_, link, cap, mk(s), seed=s, compute_sigma=cs).run().steady() for s in range(12)]
                row[nm] = dict(mean=statistics.mean(vals), ci=ci95(vals), cv=statistics.stdev(vals) / statistics.mean(vals))
            out["noise"].append(row)
        print("e9 noise", wl)
    # planner vs engine (deterministic engine) over random configurations
    rng = random.Random(5)
    models = [PRESETS[k] for k in ("llama2-7b", "llama2-13b", "llama3-8b", "llama3-70b", "gemma2-9b")]
    for _ in range(80):
        mm = rng.choice(models)
        kind = rng.choice(["decode", "prefill"])
        if kind == "decode":
            tr = trace_decode(mm, GPU_, rng.choice([1, 4, 16, 64, 256]), rng.choice([512, 2048]), 3)
        else:
            tr = trace_prefill(mm, GPU_, rng.choice([1, 2, 4]), rng.choice([512, 1024, 2048, 4096]), reps=3)
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
    print("e9 planner MAPE", out["planner_mape"])
    save("e9_robustness", out)


ALL = dict(e1=e1, e2=e2, e3=e3, e4=e4, e5=e5, e6=e6, e7=e7, e8=e8, e9=e9)

if __name__ == "__main__":
    which = sys.argv[1:] or list(ALL)
    for k in which:
        t0 = time.time()
        ALL[k]()
        print(f"== {k} done in {time.time()-t0:.0f}s")
