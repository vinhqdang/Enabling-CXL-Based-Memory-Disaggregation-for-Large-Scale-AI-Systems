#!/usr/bin/env python3
"""Interference between layer compute and host-to-device copies on one GPU.

Question: why does layer compute get slower while a host-to-device copy runs
on another stream? For each workload the script measures

  a. idle                       (GPU clocks and power at rest)
  b. compute only               (8 resident layers, per-layer CUDA-event time)
  c. copy only                  (one pinned 201 MB host buffer -> GPU, back to back)
  d. compute + copy concurrent  (both loops on separate streams and threads)
  e. compute + copy at ~50%     (the copy loop idles on the host for as long as
                                 each copy took, so the copy engine is busy about half the time)

and reports the slowdown ratios compute(d)/compute(b) and copy(d)/copy(c).
A background nvidia-smi sampler records clocks, power, temperature, throttle
reasons and PCIe generation, tagged with the phase that was running. The
script then tries to lock the GPU clocks (nvidia-smi -lgc), repeats b and d,
resets the clocks, and finally measures compute right after an idle period
versus after sustained compute (thermal and power warm-up).

Layer shapes and layer compute follow runtime/stream_bench.py (llama3.2-3b:
d=3072, 24 query heads, 8 KV heads, head_dim 128, ff 8192; one flat fp16
weight buffer per layer). Attention on sm < 80 uses the memory-efficient
SDPA kernel, as in stream_bench.

Single file, standard library + torch only.

Examples (on a GPU VM):
  colab run --gpu T4 --timeout 600 runtime/interference.py

CPU control-flow check (no CUDA, no streams, no nvidia-smi):
  python3 runtime/interference.py --dry
"""

import argparse
import json
import math
import statistics
import subprocess
import threading
import time
from collections import deque
from contextlib import nullcontext

import torch
import torch.nn.functional as F

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except ImportError:  # older torch: no explicit backend selection
    SDPBackend = None
    sdpa_kernel = None


MODEL = dict(d=3072, n_heads=24, n_kv=8, head_dim=128, ff=8192)   # llama3.2-3b
DRY_MODEL = dict(d=64, n_heads=4, n_kv=2, head_dim=16, ff=128)
DRY_CAP = dict(T=128, B=4, ctx=64)

FP16 = torch.float16
ALIGN = 256
RMS_EPS = 1e-5
FILL_CHUNK = 1 << 24

N_RES = 8                 # resident layers in the compute loop
CMP_WINDOW = 2            # compute iterations kept in flight by the host
CPY_WINDOW = 4            # copies kept in flight by the host
PHASE_S = 15.0
IDLE_S = 15.0
WARM_S = 30.0
EDGE_S = 3.0
BUDGET_S = 540.0
SAMPLE_PERIOD_S = 0.1
DEFAULT_WORKLOADS = "pre:1024,pre:4096,dec:16:512"

NVSMI_FIELDS = ("timestamp,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu,"
                "clocks_throttle_reasons.active,pcie.link.gen.current,utilization.gpu")
NVSMI_SAMPLE = ["nvidia-smi", f"--query-gpu={NVSMI_FIELDS}", "--format=csv,noheader,nounits"]
THROTTLE_NAMES = {
    0x01: "gpu_idle",
    0x02: "applications_clocks_setting",
    0x04: "sw_power_cap",
    0x08: "hw_slowdown",
    0x10: "sync_boost",
    0x20: "sw_thermal_slowdown",
    0x40: "hw_thermal_slowdown",
    0x80: "hw_power_brake_slowdown",
    0x100: "display_clock_setting",
}

T0 = time.time()
RECORDS = []
RESULTS = {}
SUMMARY = {}
STATE = {"stage": "startup"}
SAMPLER = None


# ----------------------------------------------------------------- logging

def elapsed():
    return time.time() - T0


def say(msg):
    print(f"PROGRESS t={elapsed():.1f}s {msg}", flush=True)


def set_stage(text):
    STATE["stage"] = f"{text} (t={elapsed():.0f}s)"


def emit(kind, **fields):
    rec = dict(type=kind, **fields)
    RECORDS.append(rec)
    print("REC " + json.dumps(rec, default=str), flush=True)
    return rec


def heartbeat():
    while True:
        time.sleep(20)
        print(f"HEARTBEAT t={elapsed():.0f}s stage={STATE['stage']}", flush=True)


def rnd(x, nd=4):
    return None if x is None else round(float(x), nd)


def stats(vals):
    vals = [float(v) for v in vals if v is not None]
    if not vals:
        return dict(n=0)
    return dict(n=len(vals), mean=rnd(statistics.fmean(vals)), median=rnd(statistics.median(vals)),
                min=rnd(min(vals)), max=rnd(max(vals)))


def split_list(text):
    return [t.strip() for t in text.split(",") if t.strip()]


def run_cmd(args, cap=2000):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=30)
        return dict(cmd=" ".join(args), returncode=p.returncode,
                    stdout=p.stdout.strip()[:cap], stderr=p.stderr.strip()[:cap])
    except Exception as exc:  # noqa: BLE001 - recorded, not fatal
        return dict(cmd=" ".join(args), returncode=None, error=repr(exc))


# ----------------------------------------------------------------- dry-run stand-ins

class DryEvent:
    def __init__(self):
        self.t = None

    def record(self, stream=None):
        self.t = time.perf_counter()

    def synchronize(self):
        pass

    def elapsed_time(self, other):
        return (other.t - self.t) * 1000.0


# ----------------------------------------------------------------- layout and tensors

def _align(x, a=ALIGN):
    return (x + a - 1) // a * a


def layer_layout(cfg):
    """Byte offsets of the fields inside one layer's flat weight buffer."""
    d, nh, nkv, hd, ff = cfg["d"], cfg["n_heads"], cfg["n_kv"], cfg["head_dim"], cfg["ff"]
    specs = [
        ("wqkv", (d, (nh + 2 * nkv) * hd)),
        ("wo", (nh * hd, d)),
        ("wgu", (d, 2 * ff)),
        ("wdown", (ff, d)),
        ("n1", (d,)),
        ("n2", (d,)),
    ]
    off = 0
    lay = {}
    for name, shape in specs:
        off = _align(off)
        lay[name] = (off, shape)
        off += math.prod(shape) * 2
    return lay, _align(off)


def views_layer(buf, lay):
    out = {}
    for name, (off, shape) in lay.items():
        n = math.prod(shape) * 2
        out[name] = buf[off:off + n].view(FP16).view(shape)
    return out


def fill_fp16(t_u8):
    """Fill a uint8 CPU tensor with random fp16 values."""
    f = t_u8.view(FP16)
    n = f.numel()
    for s in range(0, n, FILL_CHUNK):
        e = min(n, s + FILL_CHUNK)
        f[s:e].copy_(torch.randn(e - s, dtype=torch.float32).mul_(0.02))


def rms(x, w):
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + RMS_EPS)
    return y.to(x.dtype) * w


# ----------------------------------------------------------------- GPU runner

class Runner:
    def __init__(self, dry, cfg):
        self.dry = dry
        self.dev = "cpu" if dry else "cuda"
        self.d = cfg["d"]
        self.nh = cfg["n_heads"]
        self.nkv = cfg["n_kv"]
        self.hd = cfg["head_dim"]
        self.ff = cfg["ff"]
        self.lay, self.layer_bytes = layer_layout(cfg)
        self.comp = None
        self.cpy = None
        self.cap = None
        self.gpu_name = None
        self.attn_mode = None
        self.attn_probe = {}
        self.src = None
        self.dst = None
        self.resident = []

    # stream / event / sync helpers
    def sctx(self, s):
        return nullcontext() if self.dry else torch.cuda.stream(s)

    def event(self):
        return DryEvent() if self.dry else torch.cuda.Event(enable_timing=True)

    def sync(self):
        if not self.dry:
            torch.cuda.synchronize()

    def setup(self):
        if not self.dry:
            torch.cuda.set_device(0)
            self.cap = tuple(torch.cuda.get_device_capability(0))
            self.gpu_name = torch.cuda.get_device_name(0)
            self.comp = torch.cuda.Stream()
            self.cpy = torch.cuda.Stream()
        # one pinned host buffer of one layer's size, as the copy source
        self.src = torch.empty(self.layer_bytes, dtype=torch.uint8, pin_memory=not self.dry)
        if not self.dry:
            assert self.src.is_pinned(), "host buffer is not page-locked"
        fill_fp16(self.src)
        self.dst = torch.empty(self.layer_bytes, dtype=torch.uint8, device=self.dev)
        for _ in range(N_RES):
            buf = torch.empty(self.layer_bytes, dtype=torch.uint8, device=self.dev)
            buf.copy_(self.src)
            self.resident.append(views_layer(buf, self.lay))
        self.sync()
        self.probe_attn()
        for _ in range(2):  # untimed warm-up of the copy path
            _, a, b = self._enqueue_copy()
            b.synchronize()
        self.sync()

    def make_wl(self, spec):
        p = spec.split(":")
        if p[0] == "pre":
            T = int(p[1])
            if self.dry:
                T = min(T, DRY_CAP["T"])
            wl = dict(name=spec, kind="pre", T=T)
            x_shape = (T, self.d)
        else:
            B, ctx = int(p[1]), int(p[2])
            if self.dry:
                B, ctx = min(B, DRY_CAP["B"]), min(ctx, DRY_CAP["ctx"])
            wl = dict(name=spec, kind="dec", B=B, ctx=ctx)
            x_shape = (B, self.d)
            kv_shape = (B, self.nkv, ctx, self.hd)
            wl["K"] = (torch.randn(kv_shape, dtype=torch.float32, device=self.dev) * 0.1).to(FP16)
            wl["V"] = (torch.randn(kv_shape, dtype=torch.float32, device=self.dev) * 0.1).to(FP16)
        wl["x0"] = (torch.randn(x_shape, dtype=torch.float32, device=self.dev) * 0.1).to(FP16)
        self.sync()
        # untimed warm-up: first-use kernel and library initialisation
        for _ in range(2):
            _, a, b = self._enqueue_compute(wl)
            b.synchronize()
        self.sync()
        return wl

    # attention backend (same selection as stream_bench)
    def probe_attn(self):
        nh, nkv, hd = self.nh, self.nkv, self.hd
        q = (torch.randn(1, nh, 4, hd, dtype=torch.float32, device=self.dev) * 0.1).to(FP16)
        kv = (torch.randn(1, nkv, 4, hd, dtype=torch.float32, device=self.dev) * 0.1).to(FP16)
        kvr = kv.repeat_interleave(nh // nkv, dim=1)
        results = {}

        def attempt(label, fn):
            try:
                fn()
                self.sync()
                results[label] = True
            except Exception as exc:  # noqa: BLE001 - a failing mode is recorded, not fatal
                results[label] = f"{type(exc).__name__}: {str(exc)[:160]}"

        def gqa_call():
            F.scaled_dot_product_attention(q, kv, kv, is_causal=True, enable_gqa=True)

        def repeat_call():
            F.scaled_dot_product_attention(q, kvr, kvr, is_causal=True)

        def efficient_call():
            with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                F.scaled_dot_product_attention(q, kvr, kvr, is_causal=True)

        attempt("enable_gqa", gqa_call)
        attempt("repeat_default", repeat_call)
        if not self.dry and sdpa_kernel is not None:
            attempt("repeat_efficient", efficient_call)

        if self.cap is None or self.cap >= (8, 0):
            prefs = ["enable_gqa", "repeat_default"]
        else:
            # sm < 80: enable_gqa is not used; force the memory-efficient kernel
            prefs = ["repeat_efficient", "repeat_default"]
        mode = next((p for p in prefs if results.get(p) is True), None)
        if mode is None:
            raise RuntimeError(f"no attention mode works: {results}")
        self.attn_mode = mode
        self.attn_probe = results

    def attn(self, q, k, v, causal):
        if self.attn_mode == "enable_gqa":
            return F.scaled_dot_product_attention(q, k, v, is_causal=causal, enable_gqa=True)
        rep = self.nh // self.nkv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        if self.attn_mode == "repeat_efficient":
            with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                return F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        return F.scaled_dot_product_attention(q, k, v, is_causal=causal)

    # compute (as in stream_bench.layer_fwd)
    def layer_fwd(self, x, W, wl):
        nh, nkv, hd, ff = self.nh, self.nkv, self.hd, self.ff
        h = rms(x, W["n1"])
        qkv = h @ W["wqkv"]
        if wl["kind"] == "pre":
            T = x.shape[0]
            q = qkv[:, :nh * hd].reshape(T, nh, hd).transpose(0, 1).unsqueeze(0)
            k = qkv[:, nh * hd:(nh + nkv) * hd].reshape(T, nkv, hd).transpose(0, 1).unsqueeze(0)
            v = qkv[:, (nh + nkv) * hd:].reshape(T, nkv, hd).transpose(0, 1).unsqueeze(0)
            a = self.attn(q, k, v, True).squeeze(0).transpose(0, 1).reshape(T, nh * hd)
        else:
            B = x.shape[0]
            q = qkv[:, :nh * hd].reshape(B, nh, 1, hd)
            a = self.attn(q, wl["K"], wl["V"], False).reshape(B, nh * hd)
        x = x + a @ W["wo"]
        h = rms(x, W["n2"])
        gu = h @ W["wgu"]
        x = x + (F.silu(gu[:, :ff]) * gu[:, ff:]) @ W["wdown"]
        return x

    # loops: each returns records of (enqueue wall time, value)
    def _enqueue_compute(self, wl):
        a, b = self.event(), self.event()
        with self.sctx(self.comp):
            a.record(self.comp)
            x = wl["x0"]
            for W in self.resident:
                x = self.layer_fwd(x, W, wl)
            b.record(self.comp)
        return time.time(), a, b

    def compute_loop(self, wl, until, rec, errs):
        """Run N_RES-layer iterations until wall time `until`; record ms per layer."""
        try:
            pend = deque()
            while time.time() < until:
                pend.append(self._enqueue_compute(wl))
                while len(pend) > CMP_WINDOW:
                    t, pa, pb = pend.popleft()
                    pb.synchronize()
                    rec.append((t, pa.elapsed_time(pb) / N_RES))
            while pend:
                t, pa, pb = pend.popleft()
                pb.synchronize()
                rec.append((t, pa.elapsed_time(pb) / N_RES))
        except Exception as exc:  # noqa: BLE001 - reported by the caller
            errs.append(f"compute_loop: {exc!r}")

    def _enqueue_copy(self):
        a, b = self.event(), self.event()
        with self.sctx(self.cpy):
            a.record(self.cpy)
            self.dst.copy_(self.src, non_blocking=True)
            b.record(self.cpy)
        return time.time(), a, b

    def copy_loop(self, until, rec, duty, errs):
        """Copy one layer-sized pinned buffer to the GPU until `until`.

        duty=False: back to back, with CPY_WINDOW copies in flight.
        duty=True: wait for each copy, then idle on the host for the same
        duration, so the copy engine is busy about half the time.
        """
        try:
            pend = deque()
            while time.time() < until:
                item = self._enqueue_copy()
                if duty:
                    t, a, b = item
                    b.synchronize()
                    ms = a.elapsed_time(b)
                    rec.append((t, ms))
                    time.sleep(ms / 1000.0)
                else:
                    pend.append(item)
                    while len(pend) > CPY_WINDOW:
                        t, a, b = pend.popleft()
                        b.synchronize()
                        rec.append((t, a.elapsed_time(b)))
            while pend:
                t, a, b = pend.popleft()
                b.synchronize()
                rec.append((t, a.elapsed_time(b)))
        except Exception as exc:  # noqa: BLE001 - reported by the caller
            errs.append(f"copy_loop: {exc!r}")

    # phase drivers
    def run_compute(self, wl, dur):
        rec, errs = [], []
        self.compute_loop(wl, time.time() + dur, rec, errs)
        check(errs)
        return rec

    def run_copy(self, dur):
        rec, errs = [], []
        self.copy_loop(time.time() + dur, rec, False, errs)
        check(errs)
        return rec

    def run_both(self, wl, dur, duty):
        crec, prec, errs = [], [], []
        until = time.time() + dur
        ct = threading.Thread(target=self.copy_loop, args=(until, crec, duty, errs), daemon=True)
        pt = threading.Thread(target=self.compute_loop, args=(wl, until, prec, errs), daemon=True)
        ct.start()
        pt.start()
        ct.join()
        pt.join()
        check(errs)
        return prec, crec


def check(errs):
    if errs:
        raise RuntimeError("; ".join(errs))


# ----------------------------------------------------------------- nvidia-smi sampler

def parse_sample(line):
    f = [x.strip() for x in line.split(",")]
    if len(f) != 9:
        return None

    def num(x):
        try:
            return float(x)
        except ValueError:
            return None

    thr = f[6]
    try:
        bits = int(thr, 16) if thr.lower().startswith("0x") else int(float(thr))
    except ValueError:
        bits = None
    return dict(ts=f[0], sm=num(f[1]), mem=num(f[2]), power=num(f[3]), power_limit=num(f[4]),
                temp=num(f[5]), throttle=bits, pcie_gen=num(f[7]), util=num(f[8]))


class Sampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.phase = "startup"
        self.samples = []
        self.failures = []
        self.stop_ev = threading.Event()

    def run(self):
        nxt = time.time()
        while not self.stop_ev.is_set():
            phase = self.phase
            try:
                p = subprocess.run(NVSMI_SAMPLE, capture_output=True, text=True, timeout=5)
                lines = p.stdout.strip().splitlines() if p.returncode == 0 else []
                row = parse_sample(lines[0]) if lines else None
                if row is not None:
                    self.samples.append((phase, row))
                elif len(self.failures) < 20:
                    self.failures.append(f"rc={p.returncode} err={p.stderr.strip()[:200]}")
            except Exception as exc:  # noqa: BLE001
                if len(self.failures) < 20:
                    self.failures.append(repr(exc)[:200])
            nxt += SAMPLE_PERIOD_S
            delay = nxt - time.time()
            if delay > 0:
                self.stop_ev.wait(delay)
            else:
                nxt = time.time()

    def stop(self):
        self.stop_ev.set()
        self.join(10)

    def summary(self, phase):
        rows = [r for p, r in self.samples if p == phase]
        if not rows:
            return dict(n=0)

        def vals(k):
            return [r[k] for r in rows if r[k] is not None]

        names = set()
        bits_seen = set()
        for r in rows:
            b = r["throttle"]
            if b is None:
                continue
            bits_seen.add(hex(b))
            if b == 0:
                names.add("none")
            for bit, name in THROTTLE_NAMES.items():
                if b & bit:
                    names.add(name)
        return dict(n=len(rows), sm_clock_mhz=stats(vals("sm")), mem_clock_mhz=stats(vals("mem")),
                    power_w=stats(vals("power")), power_limit_w=stats(vals("power_limit")),
                    temp_c=stats(vals("temp")), util_pct=stats(vals("util")),
                    throttle_reasons=sorted(names), throttle_bits=sorted(bits_seen),
                    pcie_gen=sorted(set(vals("pcie_gen"))))


# ----------------------------------------------------------------- phase bookkeeping

def begin(tag):
    if SAMPLER is not None:
        SAMPLER.phase = tag
    set_stage(tag)
    say(f"start {tag}")
    return time.time()


def end(tag, t0, spec, clock, cmp=None, cpy=None, nbytes=0):
    t1 = time.time()
    dur = t1 - t0
    if SAMPLER is not None:
        SAMPLER.phase = "gap"
    rec = dict(tag=tag, workload=spec, clock=clock, duration_s=rnd(dur, 2))
    if cmp is not None:
        rec["compute_ms_per_layer"] = stats([v for _, v in cmp])
    if cpy is not None:
        ms = [v for _, v in cpy]
        rec["copy_ms"] = stats(ms)
        if ms and dur > 0:
            rec["copy_GBps_mean"] = rnd(nbytes / 1e9 / (statistics.fmean(ms) / 1000.0), 2)
            rec["copy_busy_frac"] = rnd(sum(ms) / 1000.0 / dur, 3)
    if SAMPLER is not None:
        rec["nvsmi"] = SAMPLER.summary(tag)
    SUMMARY[tag] = rec
    say(f"done {tag} dur={dur:.1f}s")
    return emit("phase", **rec)


def mean_of(tag, key):
    rec = SUMMARY.get(tag)
    if not rec or key not in rec:
        return None
    return rec[key].get("mean")


def ratio(a, b):
    if not a or not b:
        return None
    return rnd(a / b, 3)


def ratios(spec, clock):
    b, c = f"{spec}/b/{clock}", f"{spec}/c/{clock}"
    d, e = f"{spec}/d/{clock}", f"{spec}/e/{clock}"
    out = dict(
        compute_d_over_b=ratio(mean_of(d, "compute_ms_per_layer"), mean_of(b, "compute_ms_per_layer")),
        compute_e_over_b=ratio(mean_of(e, "compute_ms_per_layer"), mean_of(b, "compute_ms_per_layer")),
        copy_d_over_c=ratio(mean_of(d, "copy_ms"), mean_of(c, "copy_ms")),
        copy_e_over_c=ratio(mean_of(e, "copy_ms"), mean_of(c, "copy_ms")),
    )
    out = {k: v for k, v in out.items() if v is not None}
    return emit("ratio", workload=spec, clock=clock, **out)


# ----------------------------------------------------------------- measurement blocks

def measure_block(R, spec, clock, P, full):
    """Phases b and d (and, if full, a, c and e) for one workload."""
    wl = R.make_wl(spec)
    if full:
        tag = f"{spec}/a/{clock}"
        t0 = begin(tag)
        time.sleep(P["idle_s"])
        end(tag, t0, spec, clock)

    tag = f"{spec}/b/{clock}"
    t0 = begin(tag)
    cmp = R.run_compute(wl, P["phase_s"])
    end(tag, t0, spec, clock, cmp=cmp)

    if full:
        tag = f"{spec}/c/{clock}"
        t0 = begin(tag)
        cpy = R.run_copy(P["phase_s"])
        end(tag, t0, spec, clock, cpy=cpy, nbytes=R.layer_bytes)

    tag = f"{spec}/d/{clock}"
    t0 = begin(tag)
    cmp, cpy = R.run_both(wl, P["phase_s"], duty=False)
    end(tag, t0, spec, clock, cmp=cmp, cpy=cpy, nbytes=R.layer_bytes)
    if not full:
        ratios(spec, clock)

    if full:
        tag = f"{spec}/e/{clock}"
        t0 = begin(tag)
        cmp, cpy = R.run_both(wl, P["phase_s"], duty=True)
        end(tag, t0, spec, clock, cmp=cmp, cpy=cpy, nbytes=R.layer_bytes)
        ratios(spec, clock)


def warm_test(R, spec, P):
    """Compute right after an idle phase versus after sustained compute."""
    clock = "unlocked"
    wl = R.make_wl(spec)
    tag = f"{spec}/warm_idle/{clock}"
    t0 = begin(tag)
    time.sleep(P["idle_s"])
    end(tag, t0, spec, clock)

    tag = f"{spec}/warm_run/{clock}"
    t0 = begin(tag)
    rec = R.run_compute(wl, P["warm_s"])
    t1 = time.time()
    end(tag, t0, spec, clock, cmp=rec)

    edge = P["edge_s"]
    buckets = {}
    for t, v in rec:
        buckets.setdefault(int((t - t0) // edge), []).append(v)
    series = [rnd(statistics.fmean(buckets[k]), 3) for k in sorted(buckets)]
    first = [v for t, v in rec if t - t0 < edge]
    last = [v for t, v in rec if t1 - t < edge]
    fm = statistics.fmean(first) if first else None
    lm = statistics.fmean(last) if last else None
    emit("warm", workload=spec, bucket_s=edge,
         first_edge_ms_per_layer=rnd(fm, 3), last_edge_ms_per_layer=rnd(lm, 3),
         last_over_first=ratio(lm, fm), bucket_ms_per_layer=series)


# ----------------------------------------------------------------- clock lock

def clock_pass(R, specs, P):
    lock = run_cmd(["nvidia-smi", "-lgc", "1200,1200"])
    emit("clock_lock", **lock)
    ok = lock.get("returncode") == 0
    if not ok:
        emit("skipped", what="locked measurements", reason="nvidia-smi -lgc did not succeed")
        return
    try:
        for spec in specs:
            measure_block(R, spec, "locked", P, full=False)
    finally:
        reset = run_cmd(["nvidia-smi", "-rgc"])
        emit("clock_reset", **reset)


# ----------------------------------------------------------------- main

def main():
    global SAMPLER
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry", action="store_true", help="CPU control-flow check with tiny shapes")
    ap.add_argument("--workloads", default=DEFAULT_WORKLOADS)
    ap.add_argument("--phase-s", type=float, default=PHASE_S)
    ap.add_argument("--idle-s", type=float, default=IDLE_S)
    ap.add_argument("--warm-s", type=float, default=WARM_S)
    ap.add_argument("--budget", type=float, default=BUDGET_S,
                    help="seconds after which optional blocks are skipped")
    args = ap.parse_args()

    P = dict(phase_s=args.phase_s, idle_s=args.idle_s, warm_s=args.warm_s,
             edge_s=min(EDGE_S, args.warm_s / 4.0), budget_s=args.budget)
    if args.dry:
        P.update(phase_s=0.4, idle_s=0.3, warm_s=2.0, edge_s=0.4, budget_s=1e9)
    specs = split_list(args.workloads)
    cfg = DRY_MODEL if args.dry else MODEL

    threading.Thread(target=heartbeat, daemon=True).start()
    set_stage("setup")
    R = Runner(args.dry, cfg)
    R.setup()
    setup = dict(dry=args.dry, torch=torch.__version__, gpu=R.gpu_name,
                 capability=R.cap, attn_mode=R.attn_mode, attn_probe=R.attn_probe,
                 layer_bytes=R.layer_bytes, resident_layers=N_RES, model=cfg,
                 phase_s=P["phase_s"], idle_s=P["idle_s"], warm_s=P["warm_s"],
                 budget_s=args.budget, workloads=specs)
    emit("setup", **setup)
    RESULTS["setup"] = setup

    if args.dry:
        RESULTS["nvidia_smi_q"] = "skipped (dry)"
        RESULTS["clock_lock"] = "skipped (dry)"
    else:
        SAMPLER = Sampler()
        SAMPLER.start()
        q = run_cmd(["nvidia-smi", "-q", "-d", "CLOCK,POWER,PERFORMANCE"], cap=30000)
        RESULTS["nvidia_smi_q"] = q
        emit("nvsmi_query", returncode=q.get("returncode"), error=q.get("error"),
             chars=len(q.get("stdout", "")))

    block_s = 5 * P["phase_s"] + 5
    try:
        for spec in specs:
            if elapsed() + block_s > P["budget_s"]:
                emit("skipped", what=spec, reason="budget")
                continue
            measure_block(R, spec, "unlocked", P, full=True)

        warm_spec = specs[0] if specs else None
        if warm_spec and elapsed() + P["idle_s"] + P["warm_s"] + 10 <= P["budget_s"]:
            warm_test(R, warm_spec, P)
        elif warm_spec:
            emit("skipped", what="warm test", reason="budget")

        lock_s = 2 * len(specs) * P["phase_s"] + 15
        if not args.dry:
            if elapsed() + lock_s <= P["budget_s"]:
                clock_pass(R, specs, P)
            else:
                emit("skipped", what="locked clocks", reason="budget")
        else:
            emit("skipped", what="locked clocks", reason="dry run")
    except Exception as exc:  # noqa: BLE001 - keep the partial results
        RESULTS["error"] = repr(exc)
        say(f"ERROR {exc!r}")
    finally:
        if SAMPLER is not None:
            SAMPLER.stop()

    if SAMPLER is not None:
        phases = sorted({p for p, _ in SAMPLER.samples})
        RESULTS["nvsmi_phases"] = {p: SAMPLER.summary(p) for p in phases}
        RESULTS["nvsmi_failures"] = SAMPLER.failures
    RESULTS["records"] = RECORDS
    RESULTS["elapsed_s"] = rnd(elapsed(), 1)
    print("JSON_BEGIN", flush=True)
    print(json.dumps(RESULTS, indent=1, default=str), flush=True)
    print("JSON_END", flush=True)
    say(f"finished in {elapsed():.1f}s")


if __name__ == "__main__":
    main()
