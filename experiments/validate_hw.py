"""
Validate the simulator's pipeline/link machinery against a real GPU (Tesla T4, PCIe Gen3 x16).

Protocol: the micro-benchmark (validation/hw_validate.py) measures, in the same CUDA streaming
loop, (i) a compute-only iteration (all 24 weight tensors resident in HBM) per token count,
(ii) the H2D copy cost of one weight tensor (pinned host memory standing in for the CXL pool).
The simulator is *composed* from these separately measured parts and asked to predict the
measured composite iteration time (copy + compute with a given prefetch depth and pinned
count).  Nothing is fitted to the composites.
"""
import json, statistics, sys
from collections import defaultdict
sys.path.insert(0, '.')
from camp_sim.hw import GPU, Link
from camp_sim.models import Trace, Unit, Access
from camp_sim.sim import Engine
from camp_sim.policies import NoPrefetch, StaticK

hw = json.load(open('results/hw/hw_T4_v2.json'))
fit = hw['h2d_fit_pinned']
N, UB = 24, 64 * 2**20
g = defaultdict(list)
for e in hw['stream']:
    if 'error' not in e:
        g[(e['mode'], e['T'], e['depth'], e['n_pinned'])].append(e['mean'])
comp_per_unit = {T: statistics.mean(v) / N for (m, T, d, p), v in g.items() if m == 'compute_only'}
copy_per_unit = statistics.mean(g[('copy_only', 0, 4, 0)]) / N
print('compute per unit (ms):', {k: round(v * 1e3, 3) for k, v in comp_per_unit.items()})
print('copy per unit (ms): measured loop %.3f ; fit %.3f' % (copy_per_unit * 1e3, (fit['latency_us'] * 1e-6 + UB / (fit['bw_GBps'] * 1e9)) * 1e3))

gpu = GPU(name='T4', hbm_bytes=15e9, sampling_gap_us=0.0)
link = Link(name='T4-pinned', bw_gbs=fit['bw_GBps'], latency_us=fit['latency_us'], sync_us=0.0,
            chunk_mb=1e9, per_chunk_us=0.0, jitter_sigma=0.005, drift_sigma=0.003)
rows = []
for (mode, T, depth, npin), v in sorted(g.items()):
    if mode != 'composite':
        continue
    hw_t = statistics.mean(v)
    c = comp_per_unit[T]
    units = {i: Unit(i, f'u{i}', 'mlp', UB) for i in range(N)}
    reps = 8
    acc = [Access(u, 'mlp', None, c, (0, 0, 0, 0, 0), r) for r in range(reps) for u in range(N)]
    tr = Trace(units, acc, [r * N for r in range(reps)])
    cap = (npin + max(depth, 0) + 1) * UB
    pins = set(range(npin))
    pol = NoPrefetch(pins=pins, evict='fifo') if depth == 0 else StaticK(depth, wrap=True, pins=pins, evict='fifo')
    sim = statistics.mean(Engine(tr, gpu, link, cap, pol, seed=s, compute_sigma=0.0).run().steady(2) for s in range(5))
    ns = N - npin
    cf = ns * copy_per_unit + N * c if depth == 0 else max(ns * copy_per_unit, N * c)   # closed-form (no simulator)
    rows.append(dict(T=T, depth=depth, n_pinned=npin, hw_ms=hw_t * 1e3, hw_spread_ms=(max(v) - min(v)) * 1e3,
                     sim_ms=sim * 1e3, err_pct=100 * (sim - hw_t) / hw_t, cf_ms=cf * 1e3, cf_err_pct=100 * (cf - hw_t) / hw_t))
    print(f"T={T:5d} depth={depth} pinned={npin:2d}  hw={hw_t*1e3:7.1f} ms  sim={sim*1e3:7.1f} ms  err={rows[-1]['err_pct']:+5.1f}%")
errs = [abs(r['err_pct']) for r in rows]
cf_errs = [abs(r['cf_err_pct']) for r in rows]
pin_errs = [abs(r['err_pct']) for r in rows if r['n_pinned'] > 0]
nopin_errs = [abs(r['err_pct']) for r in rows if r['n_pinned'] == 0]
print(f"closed-form MAPE={statistics.mean(cf_errs):.2f}% max={max(cf_errs):.2f}% ; pinned-config simulator MAPE={statistics.mean(pin_errs):.2f}% (n={len(pin_errs)}), unpinned {statistics.mean(nopin_errs):.2f}% (n={len(nopin_errs)})")
print(f"MAPE={statistics.mean(errs):.2f}%  median={statistics.median(errs):.2f}%  max={max(errs):.2f}%  n={len(errs)}")
# staged (pageable) path: predicted vs measured copy bandwidth
pg = hw['h2d_fit_pageable']
print('pageable measured %.2f GB/s ; pinned %.2f GB/s ; ratio %.2f' % (pg['bw_GBps'], fit['bw_GBps'], pg['bw_GBps'] / fit['bw_GBps']))
staged_bw = 1.0 / (1.0 / pg['bw_GBps'] - 1.0 / fit['bw_GBps'])
print('implied staging-hop (CPU memcpy) rate for serial two-hop model: %.2f GB/s' % staged_bw)
json.dump(dict(rows=rows, cf_mape=statistics.mean(cf_errs), cf_max=max(cf_errs), pinned_mape=statistics.mean(pin_errs), pinned_n=len(pin_errs),
               unpinned_mape=statistics.mean(nopin_errs), mape=statistics.mean(errs), median=statistics.median(errs), max=max(errs),
               comp_per_unit_ms={str(k): v * 1e3 for k, v in comp_per_unit.items()}, copy_per_unit_ms=copy_per_unit * 1e3,
               pinned_fit=fit, pageable_fit=pg, implied_staging_gbs=staged_bw, device=hw['device']),
          open('results/hw_validation.json', 'w'), indent=1)
