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
print("all engine checks passed")
