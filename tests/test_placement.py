"""Checks for the placement package: lower bound, closed-form theorems, and helpers."""
import random

import pytest

from placement import (
    balanced,
    brute_force_optimal,
    contiguous_first,
    contiguous_uniform,
    cycle_time,
    cycle_time_contention,
    gaps,
    is_balanced,
    local_search,
    lower_bound,
    random_placement,
    schedule,
    schedule_contention,
)


def close(a, b, tol=1e-9):
    return abs(a - b) <= tol * max(1.0, abs(b))


def test_lower_bound_holds_nonuniform():
    rng = random.Random(1)
    for _ in range(300):
        L = rng.randint(2, 12)
        c = [rng.uniform(0.1, 5) for _ in range(L)]
        f = [rng.uniform(0.1, 5) for _ in range(L)]
        r = rng.randint(0, L)
        resident = random_placement(L, r, rng.randrange(10**6))
        k = rng.randint(1, 5)
        assert cycle_time(c, f, resident, k) >= lower_bound(c, f, resident) * (1 - 1e-9)


def test_T1_compute_bound_any_placement():
    rng = random.Random(2)
    for _ in range(500):
        L = rng.randint(2, 40)
        r = rng.randint(0, L - 1)
        k = rng.randint(2, 5)
        c = rng.uniform(0.5, 3)
        f = rng.uniform(0.1, 1) * c
        resident = random_placement(L, r, rng.randrange(10**6))
        p = cycle_time([c] * L, [f] * L, resident, k)
        assert close(p, L * c), (L, r, k, c, f, p)


def test_T2_balanced_is_optimal():
    rng = random.Random(3)
    for _ in range(500):
        L = rng.randint(4, 60)
        r = rng.randint(0, L - 1)
        k = rng.randint(3, 5)
        c = rng.uniform(0.5, 2)
        f = c * rng.uniform(1.0001, 4)
        p = cycle_time([c] * L, [f] * L, balanced(L, r), k)
        target = max(L * c, (L - r) * f)
        assert close(p, target), (L, r, k, c, f, p, target)


def test_single_streamed_unit_outside_theorem():
    # Theorems assume 2 <= k <= m. With m = 1 only one slot can be used and the
    # streamed unit is copied and computed back to back: period max(Lc, f + c).
    c, f = 1.0, 3.5
    p = cycle_time([c] * 4, [f] * 4, balanced(4, 3), 3)
    assert close(p, max(4 * c, f + c))


def test_T3_contiguous_closed_form():
    rng = random.Random(4)
    checked = 0
    while checked < 500:
        L = rng.randint(4, 60)
        r = rng.randint(0, L - 1)
        m = L - r
        if m < 2:
            continue
        k = rng.randint(2, m)
        c = rng.uniform(0.5, 2)
        f = c * rng.uniform(0.2, 4)
        p = cycle_time([c] * L, [f] * L, contiguous_first(L, r), k)
        assert close(p, contiguous_uniform(L, r, k, c, f)), (L, r, k, c, f, p)
        checked += 1


def test_balanced_satisfies_is_balanced():
    for L in range(1, 61):
        for r in range(L + 1):
            assert is_balanced(L, balanced(L, r)), (L, r)


def test_balanced_returns_r_distinct_units():
    for L in range(1, 61):
        for r in range(L + 1):
            assert len(balanced(L, r)) == r


def test_brute_force_optimum_attained_by_balanced():
    rng = random.Random(5)
    for L in range(4, 11):
        for _ in range(3):
            c = rng.uniform(0.5, 2)
            f = c * rng.uniform(1.0001, 4)
            for r in range(L):
                best, _ = brute_force_optimal([c] * L, [f] * L, r, 3)
                p = cycle_time([c] * L, [f] * L, balanced(L, r), 3)
                assert close(p, best), (L, r, c, f, p, best)


def test_k1_no_residents_is_serial():
    rng = random.Random(6)
    for L in range(2, 20):
        c = rng.uniform(0.5, 2)
        f = rng.uniform(0.5, 3)
        assert close(cycle_time([c] * L, [f] * L, set(), 1), L * (c + f))


def test_all_resident_returns_compute_sum():
    c = [1.0, 2.5, 0.5]
    assert cycle_time(c, [9.0] * 3, {0, 1, 2}, 2) == sum(c)
    assert lower_bound(c, [9.0] * 3, {0, 1, 2}) == sum(c)


def test_gaps_and_is_balanced_small_example():
    # L=6, residents {0, 3}: streamed 1,2,4,5; gaps after 1,2,4,5 are 0,1,0,1 (cyclic)
    assert gaps(6, {0, 3}) == [0, 1, 0, 1]
    assert is_balanced(6, {0, 3})
    assert not is_balanced(6, {0, 1})  # gaps 0,0,0,2: a single gap of 2 is not floor/ceil of r/m = 0.5


def test_schedule_shapes_and_compute_order():
    c = [1.0, 2.0, 1.5, 0.5]
    f = [2.0, 1.0, 3.0, 1.0]
    resident = {1}
    iters = 10
    s = schedule(c, f, resident, 2, iters)
    assert len(s["compute"]) == iters * len(c)
    assert len(s["transfer"]) == iters * (len(c) - len(resident))
    ends = [e for _, _, e in s["compute"]]
    assert all(b >= a for a, b in zip(ends, ends[1:]))
    assert all(e >= st for _, st, e in s["compute"])


def test_local_search_respects_budget_and_improves_start():
    rng = random.Random(7)
    L = 10
    c = [rng.uniform(0.5, 2) for _ in range(L)]
    f = [rng.uniform(0.5, 3) for _ in range(L)]
    sizes = [rng.uniform(1, 4) for _ in range(L)]
    k, budget = 2, 30.0
    start = contiguous_first(L, 2)
    period, best = local_search(c, f, sizes, budget, k, start=start, iters=300, seed=0)
    used = sum(sizes[i] for i in best) + k * max(sizes[i] for i in range(L) if i not in best)
    assert used <= budget
    assert period <= cycle_time(c, f, start, k) + 1e-12
    assert close(period, cycle_time(c, f, best, k))


def test_contention_equals_base_when_rates_equal():
    rng = random.Random(8)
    for _ in range(200):
        L = rng.randint(2, 30)
        r = rng.randint(0, L - 1)
        k = rng.randint(1, 5)
        c = [rng.uniform(0.2, 5) for _ in range(L)]
        f = [rng.uniform(0.2, 5) for _ in range(L)]
        resident = random_placement(L, r, rng.randrange(10**6))
        base = cycle_time(c, f, resident, k)
        cont = cycle_time_contention(c, c, f, resident, k)
        assert close(cont, base, 1e-9), (L, r, k, base, cont)


def test_contention_sandwiched_between_base_models():
    rng = random.Random(9)
    for _ in range(200):
        L = rng.randint(2, 30)
        r = rng.randint(0, L - 1)
        k = rng.randint(1, 5)
        c = [rng.uniform(0.2, 5) for _ in range(L)]
        c_conc = [x * rng.uniform(1.0, 2.5) for x in c]
        f = [rng.uniform(0.2, 5) for _ in range(L)]
        resident = random_placement(L, r, rng.randrange(10**6))
        cont = cycle_time_contention(c, c_conc, f, resident, k)
        lo = cycle_time(c, f, resident, k)
        hi = cycle_time(c_conc, f, resident, k)
        assert cont >= lo * (1 - 1e-9), (L, r, k, lo, cont)
        assert cont <= hi * (1 + 1e-9), (L, r, k, cont, hi)


def test_contention_all_resident_is_compute_sum():
    rng = random.Random(10)
    for _ in range(50):
        L = rng.randint(1, 12)
        c = [rng.uniform(0.2, 5) for _ in range(L)]
        c_conc = [x * rng.uniform(1.0, 3.0) for x in c]
        f = [rng.uniform(0.2, 5) for _ in range(L)]
        assert close(cycle_time_contention(c, c_conc, f, set(range(L)), 2), sum(c))


def test_contention_hand_example():
    # L=2, unit 0 resident, one slot. Copy of unit 1 runs during [0, 1]; unit 0 computes at
    # rate 1/2 for that interval (progress 0.5), then at rate 1 (ends at 1.5). Period 2.5.
    assert close(cycle_time_contention([1.0, 1.0], [2.0, 2.0], [1.0, 1.0], {0}, 1), 2.5)


def test_schedule_contention_intervals():
    c = [1.0, 2.0, 1.5, 0.5]
    c_conc = [1.2, 2.5, 1.5, 0.6]
    f = [2.0, 1.0, 3.0, 1.0]
    resident = {1}
    s = schedule_contention(c, c_conc, f, resident, 2, iters=10)
    assert [u for u, _, _ in s["compute"]] == [0, 1, 2, 3]
    starts = [st for _, st, _ in s["compute"]]
    assert starts[0] >= 0.0 and all(b >= a for a, b in zip(starts, starts[1:]))
    for _, st, en in s["compute"]:
        assert en >= st
    assert sorted(u for u, _, _ in s["copy"]) == [0, 2, 3]
    assert all(en >= st for _, st, en in s["copy"])


def test_balanced_is_balanced_up_to_130():
    # every L is covered; for large L a few residents counts per L (cost is quadratic in m)
    for L in range(1, 131):
        rs = range(L + 1) if L <= 40 else sorted({0, 1, L // 3, L // 2, L - 2, L - 1, L} & set(range(L + 1)))
        for r in rs:
            assert is_balanced(L, balanced(L, r)), (L, r)
            assert len(balanced(L, r)) == r


def test_brute_force_k2_uniform_balanced_optimal():
    for L in range(4, 13):
        for fc in (1.1, 1.3, 1.6, 1.9, 2.5, 4.0):
            c, f = 1.0, fc
            for r in range(L):
                best, _ = brute_force_optimal([c] * L, [f] * L, r, 2)
                p = cycle_time([c] * L, [f] * L, balanced(L, r), 2)
                assert close(p, best, 1e-9), (L, fc, r, p, best)


def test_proposition2_band_k2_balanced_bounds():
    rng = random.Random(11)
    done = 0
    while done < 1000:
        L = rng.randint(4, 40)
        r = rng.randint(0, L - 2)  # m >= 2
        m = L - r
        c = rng.uniform(0.5, 2.0)
        f = c * rng.uniform(1.0001, 1.9999)
        p = cycle_time([c] * L, [f] * L, balanced(L, r), 2)
        lo = max(L * c, m * f)
        hi = lo + (m / 2) * (2 * c - f)
        assert lo * (1 - 1e-9) <= p <= hi * (1 + 1e-9), (L, r, c, f, p, lo, hi)
        done += 1


def test_corollary_contiguous_balanced_ratio_bound():
    rng = random.Random(12)
    done = 0
    while done < 1000:
        k = rng.randint(2, 5)
        L = rng.randint(max(4, k), 40)
        r = rng.randint(0, L - k)  # m >= k
        c = rng.uniform(0.5, 2.0)
        f = c * rng.uniform(1.0001, 4.0)
        if (k - 1) * f < 2 * c:
            continue
        cf = cycle_time([c] * L, [f] * L, contiguous_first(L, r), k)
        bal = cycle_time([c] * L, [f] * L, balanced(L, r), k)
        assert cf / bal < 2 - c / f + 1e-9, (k, L, r, c, f, cf / bal)
        done += 1
