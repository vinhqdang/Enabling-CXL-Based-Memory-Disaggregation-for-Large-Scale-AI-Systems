"""
Hardware micro-benchmarks used to calibrate and validate the CAMP simulator.

Run on any CUDA machine (for example `colab run --gpu T4 --timeout 1500 validation/hw_validate.py`).
The script prints one JSON document between the markers HWJSON_BEGIN / HWJSON_END.

What it measures (all with real CUDA streams/events, no simulation):
  A. device identity and PCIe link state
  B. host->device copy time vs. size for pinned and pageable host memory
     (pinned = direct DMA path; pageable = driver bounce buffer / "staged" path)
  C. launch + event-synchronisation overheads
  D. GEMM time for transformer-shaped weights vs. number of tokens (roofline check)
  E. an end-to-end layer-streaming loop: N weight tensors live in pinned host memory
     (standing in for the CXL pool), a small set is pinned in HBM, the rest are
     streamed through a ring of device slots with a configurable prefetch depth.
     Iteration times for several (tokens, depth, pinned-fraction) points are
     compared offline against the simulator.
"""
import json
import statistics
import subprocess
import sys
import time

import torch

OUT = {}


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()
    except Exception as e:  # pragma: no cover
        return str(e)


def spec_for(name):
    n = name.lower()
    table = [  # (substring, fp16 dense tensor FLOP/s, HBM/GDDR bytes/s)
        ("h100 pcie", 756e12, 2.0e12),
        ("h100", 989e12, 3.35e12),
        ("a100", 312e12, 1.935e12),
        ("l4", 121e12, 300e9),
        ("t4", 65e12, 320e9),
    ]
    for key, f, b in table:
        if key in n:
            return f, b
    return None, None


def time_gpu(fn, reps=20, warm=5):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) / 1e3)
    return ts


def summarize(ts):
    return {
        "mean": statistics.mean(ts),
        "std": statistics.pstdev(ts),
        "min": min(ts),
        "median": statistics.median(ts),
        "max": max(ts),
        "n": len(ts),
    }


def main():
    if not torch.cuda.is_available():
        print("No CUDA device available.")
        sys.exit(1)
    dev = torch.cuda.get_device_name(0)
    peak_f, peak_b = spec_for(dev)
    OUT["device"] = {
        "name": dev,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "mem_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
        "pcie": sh("nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.gen.max,"
                   "pcie.link.width.current,pcie.link.width.max --format=csv,noheader"),
        "spec_peak_flops": peak_f,
        "spec_mem_bw": peak_b,
    }

    # ---------------- B: host->device copy characteristics ----------------
    sizes = [4 << 10, 64 << 10, 1 << 20, 4 << 20, 16 << 20, 64 << 20, 256 << 20]
    B = {"pinned": {}, "pageable": {}}
    for nbytes in sizes:
        n = nbytes // 2
        d = torch.empty(n, dtype=torch.float16, device="cuda")
        hp = torch.empty(n, dtype=torch.float16).pin_memory()
        hq = torch.empty(n, dtype=torch.float16)
        hp.normal_(); hq.normal_()
        for kind, h in (("pinned", hp), ("pageable", hq)):
            reps = 30 if nbytes <= (64 << 20) else 10
            ts = time_gpu(lambda: d.copy_(h, non_blocking=True), reps=reps)
            B[kind][str(nbytes)] = summarize(ts)
        del d, hp, hq
    OUT["h2d"] = B
    # linear fit t = a + bytes / bw on large pinned sizes
    xs = [float(k) for k in B["pinned"] if int(k) >= (1 << 20)]
    ys = [B["pinned"][str(int(k))]["median"] for k in xs]
    n = len(xs); mx = sum(xs) / n; my = sum(ys) / n
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    OUT["h2d_fit_pinned"] = {"bw_GBps": 1 / slope / 1e9, "latency_us": (my - slope * mx) * 1e6}
    xs = [float(k) for k in B["pageable"] if int(k) >= (1 << 20)]
    ys = [B["pageable"][str(int(k))]["median"] for k in xs]
    n = len(xs); mx = sum(xs) / n; my = sum(ys) / n
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    OUT["h2d_fit_pageable"] = {"bw_GBps": 1 / slope / 1e9, "latency_us": (my - slope * mx) * 1e6}

    # ---------------- C: launch / event overheads ----------------
    a = torch.randn(16, 16, device="cuda", dtype=torch.float16)
    tiny = time_gpu(lambda: a @ a, reps=200, warm=20)
    OUT["tiny_kernel_s"] = summarize(tiny)
    ev_t = []
    for _ in range(200):
        t0 = time.perf_counter()
        e = torch.cuda.Event(); e.record(); e.synchronize()
        ev_t.append(time.perf_counter() - t0)
    OUT["event_sync_s"] = summarize(ev_t)

    # ---------------- D: GEMM roofline ----------------
    d_model, ff = 4096, 8192
    W = torch.randn(d_model, ff, device="cuda", dtype=torch.float16)
    G = {}
    for T in [1, 4, 16, 64, 256, 1024, 2048, 4096, 8192]:
        x = torch.randn(T, d_model, device="cuda", dtype=torch.float16)
        ts = time_gpu(lambda: x @ W, reps=30, warm=5)
        flops = 2.0 * T * d_model * ff
        wbytes = d_model * ff * 2
        G[str(T)] = dict(summarize(ts), flops=flops, weight_bytes=wbytes)
        if peak_f:
            tr = max(flops / peak_f, (wbytes + 2 * T * (d_model + ff) * 2) / peak_b)
            G[str(T)]["roofline_s"] = tr
            G[str(T)]["roofline_over_measured"] = tr / G[str(T)]["median"]
    OUT["gemm"] = G
    del W

    # ---------------- E: layer-streaming loop ----------------
    N, U_BYTES = 24, d_model * ff * 2  # 24 units of 64 MiB (stand-in for the CXL pool)
    host = [torch.empty(d_model, ff, dtype=torch.float16).pin_memory() for _ in range(N)]
    for h in host:
        h.normal_()
    comp = torch.cuda.current_stream()
    cstream = torch.cuda.Stream()

    def run_stream(T, depth, n_pinned, iters=12, warm=3, no_compute=False):
        x = torch.randn(T, d_model, device="cuda", dtype=torch.float16)
        slots = [torch.empty(d_model, ff, device="cuda", dtype=torch.float16) for _ in range(depth + 1)]
        pinned = {i: host[i].to("cuda") for i in range(n_pinned)}
        free_evt = [torch.cuda.Event() for _ in slots]
        for ev in free_evt:
            ev.record(comp)
        seq = list(range(N)) * (iters + warm)
        ready = {}
        used_slot = {}
        nslot = [0]
        t_iter = []

        def enqueue_copy(pos):
            u = seq[pos]
            if u in pinned or pos in ready:
                return
            s = nslot[0] % len(slots); nslot[0] += 1
            with torch.cuda.stream(cstream):
                cstream.wait_event(free_evt[s])
                slots[s].copy_(host[u], non_blocking=True)
                ev = torch.cuda.Event(); ev.record(cstream)
            ready[pos] = ev
            used_slot[pos] = s

        torch.cuda.synchronize()
        marks = []
        for pos in range(len(seq)):
            if pos % N == 0 and pos // N >= warm:
                torch.cuda.synchronize()
                marks.append(time.perf_counter())
            if depth == 0:
                enqueue_copy(pos)
            else:
                for j in range(pos, min(pos + depth + 1, len(seq))):
                    enqueue_copy(j)
            u = seq[pos]
            if u in pinned:
                w = pinned[u]
            else:
                comp.wait_event(ready[pos])
                w = slots[used_slot[pos]]
            if not no_compute:
                _ = x @ w
            if u not in pinned:
                free_evt[used_slot[pos]].record(comp)
            if depth == 0 and u not in pinned:
                torch.cuda.synchronize()
        torch.cuda.synchronize()
        marks.append(time.perf_counter())
        per_iter = [b - a for a, b in zip(marks[:-1], marks[1:])]
        del slots, pinned
        return per_iter

    E = []
    # components measured in the same context: compute-only (all units pinned in HBM)
    # and copy-only (stream through the ring without compute)
    for T in [16, 512, 1024, 2048, 4096]:
        for rep in range(3):
            try:
                per = run_stream(T, 0, N)
                E.append({"T": T, "depth": 0, "n_pinned": N, "N": N, "unit_bytes": U_BYTES, "mode": "compute_only", "rep": rep,
                          "iter_s": per, "mean": statistics.mean(per), "std": statistics.pstdev(per)})
            except Exception as ex:  # pragma: no cover
                E.append({"T": T, "mode": "compute_only", "error": str(ex)})
            torch.cuda.empty_cache()
    for rep in range(3):
        per = run_stream(16, 4, 0, no_compute=True)
        E.append({"T": 0, "depth": 4, "n_pinned": 0, "N": N, "unit_bytes": U_BYTES, "mode": "copy_only", "rep": rep,
                  "iter_s": per, "mean": statistics.mean(per), "std": statistics.pstdev(per)})
    # composites
    for T in [16, 512, 1024, 2048, 4096]:
        for depth, n_pinned in [(0, 0), (1, 0), (2, 0), (4, 0), (2, 6), (2, 12)]:
            for rep in range(3):
                try:
                    per = run_stream(T, depth, n_pinned)
                    E.append({"T": T, "depth": depth, "n_pinned": n_pinned, "N": N, "unit_bytes": U_BYTES, "mode": "composite",
                              "rep": rep, "iter_s": per, "mean": statistics.mean(per), "std": statistics.pstdev(per)})
                except Exception as ex:  # pragma: no cover
                    E.append({"T": T, "depth": depth, "n_pinned": n_pinned, "mode": "composite", "error": str(ex)})
                torch.cuda.empty_cache()
    OUT["stream"] = E

    print("HWJSON_BEGIN")
    print(json.dumps(OUT))
    print("HWJSON_END")


main()
