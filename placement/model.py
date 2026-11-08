"""Max-plus steady-state period of weight streaming with a resident set (paper/DESIGN.md)."""


def _default_iters(L):
    return max(200, 8 * L)


def _run(c, f, resident, k, iters, trace=None):
    """Simulate the recurrences for `iters` iterations, all initial times 0.

    Returns the compute end time after each iteration. If `trace` is a dict, it receives
    'compute' and 'transfer' lists of (unit, start, end).
    """
    L = len(c)
    res = set(resident)
    k = min(k, L - len(res))
    X, Ystr = [], []  # transfer end times and compute end times of streamed units, by global index
    y = 0.0
    E = []
    for _ in range(iters):
        for i in range(L):
            if i in res:
                cs = y
                y = y + c[i]
            else:
                g = len(X)
                xs = max(X[g - 1] if g > 0 else 0.0, Ystr[g - k] if g >= k else 0.0)
                x = xs + f[i]
                X.append(x)
                cs = max(y, x)
                y = cs + c[i]
                Ystr.append(y)
                if trace is not None:
                    trace["transfer"].append((i, xs, x))
            if trace is not None:
                trace["compute"].append((i, cs, y))
        E.append(y)
    return E


def cycle_time(c, f, resident, k, iters=None):
    """Exact steady-state period of the max-plus recurrences.

    c, f: per-unit compute and transfer times; resident: unit indices kept on the GPU;
    k: number of streaming slots (>= 1, capped at the number of streamed units).
    """
    L = len(c)
    res = set(resident)
    if len(res) >= L:
        return float(sum(c))
    assert k >= 1
    if iters is None:
        iters = _default_iters(L)
    E = _run(c, f, res, k, iters)
    h = len(E) // 2
    return (E[-1] - E[h]) / (len(E) - 1 - h)


def schedule(c, f, resident, k, iters=None):
    """Start/end times of every compute and transfer event, for plotting.

    Returns {'compute': [(unit, start, end), ...], 'transfer': [(unit, start, end), ...]}.
    """
    trace = {"compute": [], "transfer": []}
    _run(c, f, resident, k, _default_iters(len(c)) if iters is None else iters, trace)
    return trace


def lower_bound(c, f, resident):
    """max(total compute, total transfer of streamed units): valid for every placement."""
    res = set(resident)
    return max(sum(c), sum(f[i] for i in range(len(c)) if i not in res))
