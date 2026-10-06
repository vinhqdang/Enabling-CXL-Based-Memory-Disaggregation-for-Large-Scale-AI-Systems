"""Analytic cross-checks of the event engine (deterministic mode)."""
import sys; sys.path.insert(0, '.')
from camp_sim.hw import GPU, Link
from camp_sim.models import Trace, Unit, Access
from camp_sim.sim import Engine
from camp_sim.policies import NoPrefetch, StaticK, CAMPPrefetch, pins_first

LINK = Link(bw_gbs=10.0, latency_us=0.0, sync_us=0.0, per_chunk_us=0.0, chunk_mb=1e9)
GPU0 = GPU(sampling_gap_us=0.0)


def uniform(N, nbytes, c, reps):
    units = {i: Unit(i, f"u{i}", "mlp", nbytes) for i in range(N)}
    acc = [Access(u, "mlp", None, c, (0, 0, 0, 0, 0), r) for r in range(reps) for u in range(N)]
    return Trace(units, acc, [r * N for r in range(reps)])


def run(tr, cap, pol):
    return Engine(tr, GPU0, LINK, cap, pol, deterministic=True).run()


def check(name, got, want, tol=0.005):
    ok = abs(got - want) <= tol * want
    print(f"{'PASS' if ok else 'FAIL'}  {name}: got {got*1e3:.3f} ms want {want*1e3:.3f} ms")
    assert ok


nb, N = 100e6, 20
f = nb / 10e9           # 10 ms per copy
# 1. demand paging, no reuse: every layer = copy + compute (serial)
for c in (2e-3, 15e-3):
    tr = uniform(N, nb, c, 4)
    check(f"demand c={c*1e3:g}ms", run(tr, 1 * nb, NoPrefetch()).steady(), N * (f + c))
# 2. deep prefetch, ring buffer of 3 units, no pins: iteration = N * max(f, c)
for c in (2e-3, 15e-3):
    tr = uniform(N, nb, c, 6)
    check(f"prefetch c={c*1e3:g}ms", run(tr, 3 * nb, StaticK(2, wrap=True, evict="fifo")).steady(2), N * max(f, c), 0.02)
# 3. pinning P units removes their transfers: link-bound => (N-P)*f
tr = uniform(N, nb, 2e-3, 6)
pins = set(range(8))
check("pin 8/20 link-bound", run(tr, (8 + 3) * nb, CAMPPrefetch(pins=pins)).steady(2), (N - 8) * f, 0.02)
# 4. compute-bound with pins: pinned units do not extend iteration => N*c
tr = uniform(N, nb, 30e-3, 6)
check("compute-bound", run(tr, 7 * nb, CAMPPrefetch(pins=set(range(4)))).steady(2), N * 30e-3, 0.02)

# 5. non-uniform sizes: alternating big/small units, link-bound, prefetch ring large enough => sum of copy times
units = {0: Unit(0, "big", "mlp", 200e6), 1: Unit(1, "small", "mlp", 50e6)}
acc = [Access(u, "mlp", None, 1e-3, (0, 0, 0, 0, 0), r) for r in range(6) for u in (0, 1, 0, 1, 0, 1)]
tr = Trace(units, acc, [r * 6 for r in range(6)])
# cache of 220 MB cannot hold both units (250 MB): every access misses under demand paging
want = 3 * (200e6 / 10e9 + 1e-3) + 3 * (50e6 / 10e9 + 1e-3)
check("non-uniform sizes, demand paging", run(tr, 220e6, NoPrefetch()).steady(2), want, 0.005)

# 6. next-use eviction retains layers on a cyclic scan (LRU does not): with cache for K=10 of N=20 units and
#    demand paging, Belady keeps K-1 units resident (hits), LRU keeps none.
tr = uniform(20, nb, 1e-3, 8)
r_lru = run(tr, 10 * nb, NoPrefetch(evict="lru"))
r_bel = run(tr, 10 * nb, NoPrefetch(evict="belady"))
assert r_lru.hits == 0 and r_bel.hits >= 8 * 9 - 20, (r_lru.hits, r_bel.hits)
print(f"PASS  Belady retains layers on a cyclic scan: hits LRU={r_lru.hits}, next-use={r_bel.hits}")

# 7. noise: reproducible per seed, and common random numbers across policies for the same layer executions
noisy_link = Link(bw_gbs=10.0, latency_us=0.0, sync_us=0.0, per_chunk_us=0.0, chunk_mb=1e9, jitter_sigma=0.1, drift_sigma=0.0)
tr = uniform(10, nb, 5e-3, 3)
def run_noisy(pol, seed):
    return Engine(tr, GPU0, noisy_link, 3 * nb, pol, seed=seed, compute_sigma=0.1).run()
a1, a2 = run_noisy(NoPrefetch(), 5), run_noisy(NoPrefetch(), 5)
assert a1.total_time == a2.total_time
b = run_noisy(StaticK(2, wrap=True, evict="fifo"), 5)
assert abs(a1.compute_time - b.compute_time) < 1e-12      # same seed -> identical compute noise for the same layers
assert run_noisy(NoPrefetch(), 6).total_time != a1.total_time
print("PASS  noise is reproducible per seed and shared across policies (compute)")
print("all engine checks passed")
