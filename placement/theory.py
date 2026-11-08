"""Closed forms for uniform units (c, f per unit, L units, r residents, m = L - r, k slots)."""
import math


def lb_uniform(L, r, c, f):
    """Lower bound max(L c, m f) for any placement."""
    return max(L * c, (L - r) * f)


def contiguous_uniform(L, r, k, c, f):
    """Exact period for contiguous-first residents (valid for k >= 2)."""
    m = L - r
    third = r * c + 2 * c + (m - k) * f if f > c else -math.inf
    return max(L * c, m * f, third)


def budget_optimal_uniform(L, M, c, f):
    """Best (k, r, P) for a budget of M units of GPU memory (residents + slots)."""
    if f > c:
        k = 3
        r = min(max(M - k, 0), L)
    else:
        k = 2
        r = min(max(M - k, 0), L)
    m = L - r
    P = max(L * c, m * f) if f > c else L * c
    return k, r, P
