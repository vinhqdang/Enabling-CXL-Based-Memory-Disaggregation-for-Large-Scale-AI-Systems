"""Additional experiments: e12 pool sharing, e13 cross-check against a published weight-streaming configuration."""
import os, sys, statistics
from dataclasses import replace
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_all import *  # noqa
from camp_sim.hw import GPU


def tstar(m, cap, bw_gbs, gpu=GPU_):
    phi = min(1.0, cap / m.weight_bytes)
    return (1 - phi) * gpu.peak_flops * gpu.eff_gemm / (bw_gbs * 1e9) * (m.bpp / 2)


def e12():
    """G GPUs behind one expander share its bandwidth; every GPU streams at the same time (worst case)."""
    rows = []
    m = M70
    for pool in (18, 45):
        for T in (4096, 16384, 65536):
            tr = workload_by_tokens(m, T)
            cap = 0.4 * m.weight_bytes
            ideal = ideal_time(tr, GPU_)
            for G in (1, 2, 4, 8, 16):
                link = with_bw(LINK18, pool / G)
                mc, cc, _ = run_many(tr, GPU_, link, cap, camp_v2(tr, cap, link, GPU_), seeds=range(4))
                rows.append(dict(pool=pool, T=T, G=G, ideal=ideal, camp=mc, lb=lower_bound(tr, cap, link, GPU_),
                                 tstar=tstar(m, cap, pool / G)))
            print("e12", pool, T, flush=True)
    save("e12_pool_sharing", dict(model=m.name, cache_ratio=0.4, rows=rows))


def e13():
    """Cross-check of the regime model against a published streaming configuration (ITME: A100-80GB, Llama-3.1,
    18 GB/s CXL-hybrid prototype, batch 16 x 4K-token prompts, prefetch buffer of 4 GB with a three-deep pipeline)."""
    a100 = GPU(name="A100-80GB", peak_flops=312e12, hbm_bw=2.039e12, hbm_bytes=80e9, eff_gemm=0.62)
    out = []
    for mn in ("llama3-70b", "llama3-8b"):
        m = PRESETS[mn]
        tr = trace_prefill(m, a100, 16, 4096, reps=3)
        for bw in (18, 22):
            link = with_bw(LINK18, bw)
            for depth, buf in ((1, 1.5e9), (2, 3e9), (3, 4.5e9)):
                cap = depth * max(u.nbytes for u in tr.units.values())
                n0 = tr.iter_starts[1]
                ideal = sum(a.t_true for a in tr.acc[:n0]) + a100.sampling_gap_us * 1e-6
                vals = [Engine(tr, a100, link, cap, CAMPPrefetch(cm=make_cm(a100), pins=[]), seed=s).run().steady() for s in range(4)]
                out.append(dict(model=mn, bw=bw, depth=depth, ideal=ideal, t=statistics.mean(vals),
                                over=statistics.mean(vals) / ideal - 1, tstar_full=tstar(m, 0, bw, a100)))
                print("e13", mn, bw, depth, out[-1]["over"], flush=True)
    save("e13_itme_check", out)


if __name__ == "__main__":
    for n in (sys.argv[1:] or ["e12", "e13"]):
        globals()[n]()
