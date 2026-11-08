"""Resident-set constructions, structural predicates and placement searches."""
import math
import random
from itertools import combinations

from .model import cycle_time


def contiguous_first(L, r):
    """Residents are the first r units."""
    return set(range(r))


def contiguous_last(L, r):
    """Residents are the last r units."""
    return set(range(L - r, L))


def balanced(L, r):
    """r residents spread evenly over the cycle: positions floor((j + 0.5) L / r)."""
    pos = {math.floor((j + 0.5) * L / r) for j in range(r)}
    assert len(pos) == r
    return pos


def random_placement(L, r, seed):
    """r residents chosen uniformly at random with the given seed."""
    return set(random.Random(seed).sample(range(L), r))


def gaps(L, resident):
    """Resident counts between consecutive streamed units, cyclic, starting at the first streamed unit."""
    res = set(resident)
    s = [i for i in range(L) if i not in res]
    return [(s[(t + 1) % len(s)] - s[t] - 1) % L for t in range(len(s))]


def is_balanced(L, resident):
    """True iff every cyclic window of q consecutive gaps holds floor or ceil of q r / m residents."""
    res = set(resident)
    r = len(res)
    m = L - r
    if m == 0:
        return True
    g = gaps(L, res)
    for t in range(m):
        total = 0
        for q in range(1, m + 1):
            total += g[(t + q - 1) % m]
            lo, hi = q * r // m, -(-q * r // m)
            if total not in (lo, hi):
                return False
    return True


def brute_force_optimal(c, f, r, k):
    """Minimum cycle_time over all resident sets of size r (L <= 16). Returns (period, set)."""
    L = len(c)
    assert L <= 16
    best = (math.inf, None)
    for comb in combinations(range(L), r):
        p = cycle_time(c, f, comb, k)
        if p < best[0]:
            best = (p, set(comb))
    return best


def local_search(c, f, sizes, budget, k, start=None, iters=2000, seed=0):
    """Resident set minimizing cycle_time subject to memory use within budget.

    Memory = sum of sizes of residents + k * max size of streamed units.
    Moves: swap a resident with a streamed unit, add a unit, remove a resident.
    Improvements are accepted; equal-period moves with probability 0.1. Returns (period, set).
    """
    L = len(c)
    rng = random.Random(seed)

    def cost(R):
        streamed = [sizes[i] for i in range(L) if i not in R]
        return sum(sizes[i] for i in R) + k * max(streamed, default=0.0)

    if start is None:
        for t in range(L, -1, -1):
            start = balanced(L, t)
            if cost(start) <= budget:
                break
        else:
            raise ValueError("budget is infeasible even with no residents")
        for i in sorted(set(range(L)) - start, key=lambda i: -sizes[i]):
            if cost(start | {i}) <= budget:
                start = start | {i}
    cur = set(start)
    assert cost(cur) <= budget, "start placement exceeds the budget"
    cur_p = cycle_time(c, f, cur, k)
    best, best_p = set(cur), cur_p

    for _ in range(iters):
        out = [i for i in range(L) if i not in cur]
        kinds = (["swap"] if out and cur else []) + (["add"] if out else []) + (["remove"] if cur else [])
        if not kinds:
            break
        kind = rng.choice(kinds)
        R = set(cur)
        if kind == "swap":
            R.remove(rng.choice(sorted(cur)))
            R.add(rng.choice(out))
        elif kind == "add":
            R.add(rng.choice(out))
        else:
            R.remove(rng.choice(sorted(cur)))
        if cost(R) > budget:
            continue
        p = cycle_time(c, f, R, k)
        if p < cur_p * (1 - 1e-12) or (abs(p - cur_p) <= 1e-12 * cur_p and rng.random() < 0.1):
            cur, cur_p = R, p
            if cur_p < best_p * (1 - 1e-12):
                best, best_p = set(cur), cur_p
    return best_p, best
