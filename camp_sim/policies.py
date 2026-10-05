"""
Prefetch, eviction and pinning policies.

Baselines
---------
NoPrefetch      demand paging (no lookahead), LRU/LFU/FIFO eviction
StaticK         fixed lookahead of ``k`` layers (within one forward pass), naive eviction
Reactive        lookahead that grows on observed stalls and decays when stall-free
                (a PSI/pressure-driven, TMO-like controller)
Aggressive      fixed deep lookahead (Limoncello-like, k=10)
FlexGenStyle    layer-prefix residency + double-buffered streaming (static placement)
HotColdStyle    static partition by access frequency (PowerInfer-style hot/cold) + LRU stream
CAMP            runtime cost model + work-conserving prefetch with next-use admission +
                stall-cost-aware pinning

All CAMP planning uses only (i) the model's static unit sequence and (ii) the online
``CostModel``; it never reads the simulator's ground-truth compute times.
"""
import heapq
import itertools
import math
import random
from collections import deque
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .cost import CostModel
from .hw import Link

INF = float("inf")


# =============================================================================
# Planner: analytic steady-state pipeline model used by CAMP's pinning stage
# =============================================================================
def plan_time(seq, pinned: Set[int], cap: float, link: Link, gap: float, reps: int = 3) -> float:
    """Predicted steady-state iteration time (s) for one iteration repeated ``reps`` times.

    ``seq`` is a list of ``(uid, nbytes, c_est, partial_bytes)``.  Streamed (non-pinned)
    units are fetched in order over a FIFO link as early as the streaming buffer
    ``R = cap - pinned_bytes`` allows; compute starts when the previous layer finished and
    the fetch completed.
    """
    R = cap - sum(nb for (u, nb, c, pb) in _unique_units(seq) if u in pinned)
    link_free = 0.0
    t = 0.0
    occ: deque = deque()          # (release_time, bytes) of streamed units still held
    occ_bytes = 0.0
    t_prev_iter_end = 0.0
    last = 0.0
    for r in range(reps):
        for (u, nb, c, pb) in seq:
            if u in pinned:
                start = t
            elif pb is not None:
                f = link_free + link.nominal_time(pb)
                link_free = f
                start = max(t, f)
            else:
                ts = link_free
                while occ and occ[0][0] <= ts:
                    occ_bytes -= occ.popleft()[1]
                while occ and occ_bytes + nb > R:
                    rel, b = occ.popleft()
                    occ_bytes -= b
                    ts = max(ts, rel)
                f = ts + link.nominal_time(nb)
                link_free = f
                start = max(t, f)
            end = start + c
            if u not in pinned and pb is None:
                occ.append((end, nb))
                occ_bytes += nb
            t = end
        last = t - t_prev_iter_end
        t_prev_iter_end = t + gap
        t += gap
    return last


def _unique_units(seq):
    seen = {}
    for e in seq:
        if e[3] is None and e[0] not in seen:
            seen[e[0]] = e
    return list(seen.values())


def build_plan_seq(trace, it_index: int, cm: CostModel):
    """Representative iteration of ``trace`` as planner input, with *estimated* compute times."""
    start = trace.iter_starts[it_index] if trace.iter_starts else 0
    end = trace.iter_starts[it_index + 1] if it_index + 1 < len(trace.iter_starts) else len(trace.acc)
    seq = []
    for a in trace.acc[start:end]:
        nb = trace.units[a.uid].nbytes
        seq.append((a.uid, nb, cm.predict(a.feat, a.kind), a.fetch_bytes))
    return seq


def freq_per_iteration(seq) -> Dict[int, int]:
    f: Dict[int, int] = {}
    for (u, nb, c, pb) in seq:
        if pb is None:
            f[u] = f.get(u, 0) + 1
    return f


# =============================================================================
# Pinning strategies (return a set of unit ids)
# =============================================================================
def pin_budget(cap: float, smax: float, gamma: float) -> float:
    return max(0.0, min(gamma * cap, cap - 2.0 * smax))


def pins_none(*a, **k) -> Set[int]:
    return set()


def pins_first(seq, cap, gamma=0.9) -> Set[int]:
    """FlexGen-style static placement: resident prefix of the layer sequence."""
    uniq = _unique_units(seq)
    smax = max(e[1] for e in uniq)
    budget = pin_budget(cap, smax, gamma)
    out, used = set(), 0.0
    for (u, nb, c, pb) in uniq:
        if used + nb > budget:
            break
        out.add(u)
        used += nb
    return out


def pins_freq(seq, cap, gamma=0.9) -> Set[int]:
    """Frequency-ranked greedy (CAMP's original Algorithm 2 / PowerInfer-style hot set)."""
    F = freq_per_iteration(seq)
    uniq = _unique_units(seq)
    order = {e[0]: k for k, e in enumerate(uniq)}
    smax = max(e[1] for e in uniq)
    budget = pin_budget(cap, smax, gamma)
    out, used = set(), 0.0
    for (u, nb, c, pb) in sorted(uniq, key=lambda e: (-F[e[0]], order[e[0]])):
        if used + nb <= budget:
            out.add(u)
            used += nb
    return out


def pins_stride(seq, cap, gamma=0.9) -> Set[int]:
    """Spread pins evenly over the layer sequence (heuristic: interleave pinned and streamed units)."""
    uniq = _unique_units(seq)
    smax = max(e[1] for e in uniq)
    budget = pin_budget(cap, smax, gamma)
    total = sum(e[1] for e in uniq)
    frac = min(1.0, budget / total)
    out, used, acc_ = set(), 0.0, 0.0
    for (u, nb, c, pb) in uniq:
        acc_ += frac
        if acc_ >= 1.0 - 1e-12 and used + nb <= budget:
            out.add(u)
            used += nb
            acc_ -= 1.0
    return out


def pins_random(seq, cap, gamma=0.9, seed=0) -> Set[int]:
    rng = random.Random(seed)
    uniq = _unique_units(seq)
    smax = max(e[1] for e in uniq)
    budget = pin_budget(cap, smax, gamma)
    rng.shuffle(uniq)
    out, used = set(), 0.0
    for (u, nb, c, pb) in uniq:
        if used + nb <= budget:
            out.add(u)
            used += nb
    return out


def pins_knapsack_dp(seq, cap, link: Link, gamma=0.9, quantum=16e6) -> Set[int]:
    """Exact 0-1 knapsack on the additive proxy  benefit(v) = F(v) * T_fetch(v).

    This is the optimisation problem that Eq. (2) of the paper states; it ignores pipeline
    interactions (those are handled by ``pins_scp``).  Weights are discretised to ``quantum`` bytes.
    """
    F = freq_per_iteration(seq)
    uniq = _unique_units(seq)
    smax = max(e[1] for e in uniq)
    budget = pin_budget(cap, smax, gamma)
    W = int(budget // quantum)
    items = [(e[0], max(1, int(round(e[1] / quantum))), F[e[0]] * link.nominal_time(e[1])) for e in uniq]
    n = len(items)
    best = [0.0] * (W + 1)
    take = [[False] * (W + 1) for _ in range(n)]
    for k, (u, w, v) in enumerate(items):
        for c in range(W, w - 1, -1):
            cand = best[c - w] + v
            if cand > best[c]:
                best[c] = cand
                take[k][c] = True
    out, c = set(), W
    for k in range(n - 1, -1, -1):
        if take[k][c]:
            out.add(items[k][0])
            c -= items[k][1]
    return out


def pins_greedy_marginal(seq, cap, link: Link, gap: float, reps: int = 3, eps: float = 1e-9) -> Set[int]:
    """Lazy-greedy on the planner's marginal iteration-time reduction per pinned byte."""
    uniq = _unique_units(seq)
    sizes = {e[0]: e[1] for e in uniq}
    pinned: Set[int] = set()
    cur = plan_time(seq, pinned, cap, link, gap, reps)

    def feasible(u, P):
        used = sum(sizes[x] for x in P) + sizes[u]
        streamed = [sizes[x] for x in sizes if x not in P and x != u]
        smax = max(streamed) if streamed else 0.0
        return cap - used >= smax - 1e-6

    heap = []
    for u in sizes:
        if feasible(u, pinned):
            g = cur - plan_time(seq, {u}, cap, link, gap, reps)
            heap.append((-g / sizes[u], u))
    heapq.heapify(heap)
    while heap:
        negd, u = heapq.heappop(heap)
        if not feasible(u, pinned):
            continue
        t_new = plan_time(seq, pinned | {u}, cap, link, gap, reps)
        g = cur - t_new
        d = g / sizes[u]
        if g <= eps:
            continue
        if heap and d < -heap[0][0] - 1e-18:
            heapq.heappush(heap, (-d, u))
            continue
        pinned.add(u)
        cur = t_new
    return pinned


def _feasible(sizes, P, cap):
    used = sum(sizes[x] for x in P)
    rest = [sizes[x] for x in sizes if x not in P]
    return used <= cap + 1e-6 and cap - used >= (max(rest) if rest else 0.0) - 1e-6


def pins_scp(seq, cap, link: Link, gap: float, reps: int = 3, max_evals: int = 400,
             return_info: bool = False, sa_evals: int = 1200):
    """Stall-cost-aware pinning (CAMP): plan-and-verify with the pipeline planner.

    1. Generate structured candidates: no pinning, prefix / frequency / knapsack-DP sets, evenly
       interleaved ("stride") sets at several pinned fractions, and the lazy marginal-greedy set.
    2. Score every candidate with ``plan_time`` (predicted steady-state iteration time).
    3. Refine the best candidate by first-improvement local search (add / remove / local swap),
       bounded by ``max_evals`` planner evaluations.
    Objective is the predicted iteration time, so the result accounts for compute overlap, buffer
    shrinkage and layer heterogeneity; pinning nothing is a valid outcome when the iteration is
    compute-bound.
    """
    uniq = _unique_units(seq)
    ids = [e[0] for e in uniq]
    sizes = {e[0]: e[1] for e in uniq}
    evals = [0]

    def score(P):
        evals[0] += 1
        return plan_time(seq, P, cap, link, gap, reps)

    cands = {"none": set()}
    for g in (0.3, 0.45, 0.6, 0.75, 0.9):
        cands[f"stride{g}"] = pins_stride(seq, cap, g)
    cands["first"] = pins_first(seq, cap, 0.9)
    cands["freq"] = pins_freq(seq, cap, 0.9)
    cands["dp"] = pins_knapsack_dp(seq, cap, link, 0.9)
    cands["greedy"] = pins_greedy_marginal(seq, cap, link, gap, reps)
    scored = {k: (score(P), P) for k, P in cands.items() if _feasible(sizes, P, cap)}
    best_name = min(scored, key=lambda k: scored[k][0])
    best_t, best = scored[best_name]
    best = set(best)
    pos = {u: k for k, u in enumerate(ids)}
    improved = True
    while improved and evals[0] < max_evals:
        improved = False
        moves = []
        for u in ids:
            if u in best:
                moves.append(best - {u})
            else:
                moves.append(best | {u})
        for p in list(best):
            for q in ids:
                if q not in best and abs(pos[q] - pos[p]) <= 3:
                    moves.append((best - {p}) | {q})
        for M in moves:
            if evals[0] >= max_evals:
                break
            if not _feasible(sizes, M, cap):
                continue
            t = score(M)
            if t < best_t - 1e-9:
                best_t, best, improved = t, set(M), True
                break
    # phase 3: short simulated-annealing polish (escapes the add/remove/swap local optimum)
    if sa_evals > 0 and len(ids) > 1:
        rng = random.Random(7)
        cur, cur_t = set(best), best_t
        T0 = 0.03 * best_t
        for k in range(sa_evals):
            T = T0 * (1.0 - k / sa_evals) + 1e-12
            M = set(cur)
            r = rng.random()
            if r < 0.34:
                u = rng.choice(ids)
                M ^= {u}
            else:
                ins = [x for x in ids if x in M]
                outs = [x for x in ids if x not in M]
                if not ins or not outs:
                    continue
                M.remove(rng.choice(ins))
                M.add(rng.choice(outs))
                if r > 0.8:
                    outs2 = [x for x in ids if x not in M]
                    if outs2:
                        M.add(rng.choice(outs2))
            if not _feasible(sizes, M, cap):
                continue
            t = score(M)
            if t < cur_t or rng.random() < math.exp(-(t - cur_t) / T):
                cur, cur_t = M, t
                if t < best_t - 1e-12:
                    best, best_t = set(M), t
    if return_info:
        return best, dict(start=best_name, predicted=best_t, evals=evals[0],
                          candidates={k: v[0] for k, v in scored.items()},
                          sets={k: set(v[1]) for k, v in scored.items()})
    return best


def pins_exhaustive(seq, cap, link: Link, gap: float, reps: int = 3, max_units: int = 18) -> Set[int]:
    """Brute-force optimum of the planner objective (small instances only)."""
    uniq = _unique_units(seq)
    ids = [e[0] for e in uniq]
    if len(ids) > max_units:
        raise ValueError("too many units for exhaustive search")
    sizes = {e[0]: e[1] for e in uniq}
    best, best_t = set(), plan_time(seq, set(), cap, link, gap, reps)
    for r in range(1, len(ids) + 1):
        for comb in itertools.combinations(ids, r):
            used = sum(sizes[x] for x in comb)
            rest = [sizes[x] for x in ids if x not in comb]
            if cap - used < (max(rest) if rest else 0.0) - 1e-6:
                continue
            t = plan_time(seq, set(comb), cap, link, gap, reps)
            if t < best_t - 1e-12:
                best_t, best = t, set(comb)
    return best


def pins_scp_verified(trace, cap, link: Link, gpu, cm: CostModel, it_index: int = 0, dry_iters: int = 3,
                      return_info: bool = False):
    """SCP with an engine dry-run verification step.

    The analytic planner shortlists candidates (its own optimum plus every structured candidate);
    each is then replayed for ``dry_iters`` iterations in the deterministic event engine using the
    runtime cost model's *estimated* layer times (never the device's true times), and the fastest
    replay wins.  This removes the planner's approximations (e.g. ignoring buffer reuse across
    iterations) from the final decision at the cost of a few engine replays (milliseconds each).
    """
    from .sim import Engine
    from .models import Access, Trace
    gap = gpu.sampling_gap_us * 1e-6
    seq = build_plan_seq(trace, it_index, cm)
    best, info = pins_scp(seq, cap, link, gap, return_info=True)
    cands = dict(info["sets"])
    cands["scp"] = best
    start = trace.iter_starts[it_index] if trace.iter_starts else 0
    end_it = min(it_index + dry_iters, len(trace.iter_starts)) if trace.iter_starts else 1
    end = trace.iter_starts[end_it] if trace.iter_starts and end_it < len(trace.iter_starts) else len(trace.acc)
    est_acc = [Access(a.uid, a.kind, a.fetch_bytes, cm.predict(a.feat, a.kind), a.feat, a.it) for a in trace.acc[start:end]]
    est_trace = Trace(trace.units, est_acc, [i - start for i in trace.iter_starts[it_index:end_it]] or [0], trace.meta)
    scored = {}
    for name, P in cands.items():
        if not _feasible({u: trace.units[u].nbytes for u in {a.uid for a in est_acc if a.fetch_bytes is None}}, P, cap):
            continue
        try:
            r = Engine(est_trace, gpu, link, cap, _DryPolicy(P), seed=0, deterministic=True, compute_sigma=0.0).run()
        except RuntimeError:
            continue
        scored[name] = r.steady(1 if len(r.iter_times) > 1 else 0)
    pick = min(scored, key=scored.get)
    if return_info:
        return set(cands[pick]), dict(picked=pick, dry_run=scored, planner_best=info["predicted"])
    return set(cands[pick])


class _DryPolicy:
    """Policy used only for the verification replays (CAMP prefetch controller, no learning)."""
    name = "dry"
    evict = "belady"

    def __init__(self, pins):
        self.pins = set(pins)
        self._inner = None

    def setup(self, eng):
        self._inner = CAMPPrefetch(pins=self.pins)

    def tick(self, eng, t, kind):
        self._inner.tick(eng, t, kind)

    def observe(self, eng, a, d):
        pass

    def choose_victim(self, eng, now, admit_before):
        return self._inner.choose_victim(eng, now, admit_before)


# =============================================================================
# Runtime policies
# =============================================================================
class Policy:
    name = "base"
    evict = "lru"

    def __init__(self, pins: Optional[Set[int]] = None, evict: Optional[str] = None, label: Optional[str] = None):
        self.pins = set(pins or [])
        if evict:
            self.evict = evict
        if label:
            self.name = label
        self._rng = random.Random(1)

    def setup(self, eng):
        pass

    def tick(self, eng, t, kind):
        pass

    def observe(self, eng, access, duration):
        pass

    # -- victim selection ---------------------------------------------------
    def choose_victim(self, eng, now, admit_before):
        cands = eng.evictable()
        if not cands:
            return None
        e = self.evict
        if e == "lru":
            return min(cands, key=lambda u: eng.last_use[u])
        if e == "fifo":
            return min(cands, key=lambda u: eng.ins_seq[u])
        if e == "lfu":
            return min(cands, key=lambda u: (eng.freq[u], eng.last_use[u]))
        if e == "random":
            return self._rng.choice(cands)
        if e == "belady":
            cur = eng.i
            tr = eng.trace
            best = max(cands, key=lambda u: tr.next_use(u, cur))
            nu = tr.next_use(best, cur)
            if admit_before is not None and nu <= admit_before:
                return None
            return best
        raise ValueError(e)


class NoPrefetch(Policy):
    name = "Demand"


class StaticK(Policy):
    def __init__(self, k: int, wrap: bool = False, **kw):
        super().__init__(**kw)
        self.k = k
        self.wrap = wrap
        self.name = kw.get("label") or f"Static-{k}"

    def window(self, eng, kind):
        i = eng.i
        lo = 0 if kind == "start" else i + 1
        hi = lo + self.k - 1 if kind == "start" else i + self.k
        return lo, hi

    def tick(self, eng, t, kind):
        acc = eng.trace.acc
        lo, hi = self.window(eng, kind)
        it = acc[min(eng.i, len(acc) - 1)].it
        for j in range(lo, min(hi, len(acc) - 1) + 1):
            a = acc[j]
            if not self.wrap and kind != "start" and a.it != it:
                break
            if a.fetch_bytes is None and eng.state[a.uid] == 0:
                if not eng.issue(a.uid, t, 1):
                    break


class Reactive(StaticK):
    """Lookahead controller driven by observed stalls (pressure-stall style)."""

    def __init__(self, kmax: int = 16, **kw):
        super().__init__(1, **kw)
        self.name = kw.get("label") or "Reactive"
        self.kmax = kmax
        self.quiet = 0
        self._last_stall = 0.0

    def observe(self, eng, a, d):
        s = eng.stall - self._last_stall
        self._last_stall = eng.stall
        if s > 1e-6:
            self.k = min(self.kmax, self.k + 1)
            self.quiet = 0
        else:
            self.quiet += 1
            if self.quiet >= 6:
                self.k = max(1, self.k - 1)
                self.quiet = 0


class CAMPPrefetch(Policy):
    """Work-conserving, next-use-admitted lookahead (CAMP's Semantic Opportunity Seizing, v2)."""
    evict = "belady"

    def __init__(self, cm: Optional[CostModel] = None, mode: str = "wc", wrap: bool = True,
                 max_ahead: int = 64, **kw):
        super().__init__(**kw)
        self.cm = cm
        self.mode = mode
        self.wrap = wrap
        self.max_ahead = max_ahead
        self.name = kw.get("label") or "CAMP"

    def tick(self, eng, t, kind):
        acc = eng.trace.acc
        n = len(acc)
        i = eng.i
        start = 0 if kind == "start" else i + 1
        it = acc[min(i, n - 1)].it
        budget = INF
        if self.mode == "horizon":
            # original Algorithm 1: only prefetch while the *estimated* remaining compute
            # of the current layer covers the cumulative transfer time.
            if eng.computing and self.cm is not None:
                a = acc[i]
                est_end = eng.cur_start + self.cm.predict(a.feat, a.kind)
                budget = max(0.0, est_end - t)
            else:
                budget = 0.0 if kind != "start" else 1e-9
        j = start
        while j < n and j - start < self.max_ahead:
            a = acc[j]
            if not self.wrap and a.it != it and kind != "start":
                break
            if a.fetch_bytes is None and eng.state[a.uid] == 0:
                if self.mode == "horizon":
                    if budget <= 0:
                        break
                    budget -= eng.link.nominal_time(eng.units[a.uid].nbytes)
                if not eng.issue(a.uid, t, 1, admit_before=j if self.evict == "belady" else None):
                    break
            j += 1

    def observe(self, eng, a, d):
        if self.cm is not None:
            self.cm.observe(a.feat, a.kind, d)


# =============================================================================
# Factory
# =============================================================================
def camp_pins(trace, cap, link, gpu, cm: CostModel, plan: str = "scp", it_index: int = 0,
              gamma: float = 0.9) -> Set[int]:
    gap = gpu.sampling_gap_us * 1e-6
    seq = build_plan_seq(trace, it_index, cm)
    if plan == "scp":
        return pins_scp_verified(trace, cap, link, gpu, cm, it_index)
    if plan == "freq":
        return pins_freq(seq, cap, gamma)
    if plan == "dp":
        return pins_knapsack_dp(seq, cap, link, gamma)
    if plan == "first":
        return pins_first(seq, cap, gamma)
    if plan == "none":
        return set()
    raise ValueError(plan)


def make_cm(gpu, **kw) -> CostModel:
    return CostModel(peak_flops=gpu.peak_flops, hbm_bw=gpu.hbm_bw, **kw)
