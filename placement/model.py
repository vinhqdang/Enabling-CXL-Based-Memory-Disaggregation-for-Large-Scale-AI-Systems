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


def _run_contention(c, c_conc, f, resident, k, iters, trace=None):
    """Discrete-event fluid simulation with compute slowed while a copy is in progress.

    Same ordering rules as _run: one copy engine serving streamed transfers in order, a ring of
    k slots (transfer g starts after the compute of streamed transfer g - k has ended), compute
    units in order, a streamed unit computes only after its transfer has ended. A unit's work is
    1; it progresses at rate 1 / c[i] while the copy engine is idle and at rate 1 / c_conc[i]
    while a copy is in progress. Copy durations f are unaffected. Between events the rates are
    constant, so the remaining work is integrated exactly.

    Returns the list of compute end times, one per iteration. If `trace` is a dict it receives
    'compute' entries (iteration, unit, start, end) and 'copy' entries (g, unit, start, end).
    """
    L = len(c)
    res = set(resident)
    streamed = [i for i in range(L) if i not in res]
    m = len(streamed)
    k = min(k, m)
    slot_of = {u: j for j, u in enumerate(streamed)}
    X_end = []  # end time of completed transfers, by global transfer index g
    Y_end = []  # compute end time of streamed units, by global transfer index g
    E = []
    t = 0.0
    next_g = 0  # next transfer to issue
    copy = None  # (g, unit, end) of the copy in progress
    n = 0  # global compute index: iteration n // L, unit n % L
    running = False
    cs = 0.0
    rem = 0.0
    while len(E) < iters:
        pos = n % L
        it = n // L
        # 1. start the next copy if the copy engine is idle and its slot is free
        if copy is None and (next_g < k or next_g - k < len(Y_end)):
            u = streamed[next_g % m]
            copy = (next_g, u, t + f[u])
            if trace is not None:
                trace["copy"].append((next_g, u, t, t + f[u]))
            next_g += 1
        # 2. start the current unit if its input is ready
        if not running:
            if pos in res:
                ready = True
            else:
                g = it * m + slot_of[pos]
                ready = g < len(X_end)
            if ready:
                running = True
                cs = t
                rem = 1.0
        # 3. advance to the next event (copy end or compute end)
        cand = []
        if copy is not None:
            cand.append((copy[2], 0))
        if running:
            cand.append((t + rem * (c_conc[pos] if copy is not None else c[pos]), 1))
        if not cand:
            raise RuntimeError("contention simulation stalled")
        t_new, kind = min(cand)
        if running:
            rem -= (t_new - t) / (c_conc[pos] if copy is not None else c[pos])
        t = t_new
        # 4. process completions
        if copy is not None and copy[2] <= t:
            X_end.append(copy[2])
            copy = None
        if running and (kind == 1 or rem <= 1e-12):
            running = False
            if pos not in res:
                Y_end.append(t)
            if trace is not None:
                trace["compute"].append((it, pos, cs, t))
            if pos == L - 1:
                E.append(t)
            n += 1
    return E


def cycle_time_contention(c, c_conc, f, resident, k, iters=None):
    """Steady-state period with compute slowed by concurrent copies.

    c: compute time per unit when the copy engine is idle; c_conc: compute time per unit when a
    copy is in progress; f: transfer time per unit. Other arguments as in cycle_time. With
    c_conc equal to c this equals cycle_time; with c_conc >= c it lies between cycle_time(c)
    and cycle_time(c_conc).
    """
    L = len(c)
    res = set(resident)
    if len(res) >= L:
        return float(sum(c))
    assert k >= 1
    if iters is None:
        iters = _default_iters(L)
    E = _run_contention(c, c_conc, f, res, k, iters)
    h = len(E) // 2
    return (E[-1] - E[h]) / (len(E) - 1 - h)


def schedule_contention(c, c_conc, f, resident, k, iters=None):
    """Compute and copy intervals of one late iteration, for plotting.

    The late iteration is the last one simulated. Times are shifted so that the iteration starts
    at 0 (the end of the previous iteration's last unit). Returns
    {'compute': [(unit, start, end), ...], 'copy': [(unit, start, end), ...]}, where the copy
    list holds the transfers of that iteration's streamed units (they may start before 0).
    """
    L = len(c)
    res = set(resident)
    if len(res) >= L:
        return {
            "compute": [(i, sum(c[:i]), sum(c[: i + 1])) for i in range(L)],
            "copy": [],
        }
    assert k >= 1
    if iters is None:
        iters = _default_iters(L)
    trace = {"compute": [], "copy": []}
    E = _run_contention(c, c_conc, f, res, k, iters, trace)
    it = iters - 1
    origin = E[it - 1] if it > 0 else 0.0
    m = L - len(res)
    compute = [(u, s - origin, e - origin) for (j, u, s, e) in trace["compute"] if j == it]
    copy = [
        (u, s - origin, e - origin)
        for (g, u, s, e) in trace["copy"]
        if g // m == it
    ]
    return {"compute": compute, "copy": copy}
