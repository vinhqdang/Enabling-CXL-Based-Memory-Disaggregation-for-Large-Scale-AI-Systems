#!/usr/bin/env python3
"""Steady-state iteration time of weight-streamed transformer inference.

Some transformer units are resident in GPU memory. The remaining units are
streamed from page-locked host memory through a ring of k GPU slots: a copy
stream fills slot g % k with streamed unit S[g % m] while the compute stream
runs the units in order. The script measures per-unit copy time f_u, per-unit
compute time c_u (alone and with a concurrent copy loop), the all-resident
iteration period, and the steady-state period for placements, slot counts and
workloads. GPU clocks are locked (nvidia-smi -lgc) and sampled in the background
so that every record carries the clock, power, temperature and throttle state.

Single-file, standard library + torch only. One model per process is
recommended: each model gets one pinned host slab, released between models.

Examples (on a GPU VM):
  colab run --gpu T4 --timeout 600 runtime/stream_bench.py --quick
  colab run --gpu T4 --timeout 600 runtime/stream_bench.py --models llama3.2-3b \
      --workloads pre:768 --budget 480

CPU control-flow check (no CUDA needed):
  python3 runtime/stream_bench.py --dry
"""

import argparse
import atexit
import gc
import json
import math
import random
import statistics
import subprocess
import sys
import threading
import time
from contextlib import nullcontext

import torch
import torch.nn.functional as F

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except ImportError:  # older torch: no explicit backend selection
    SDPBackend = None
    sdpa_kernel = None


MODELS = {
    "llama3.2-3b": dict(L=28, d=3072, n_heads=24, n_kv=8, head_dim=128, ff=8192, vocab=128256),
    "llama3.1-8b-16L": dict(L=16, d=4096, n_heads=32, n_kv=8, head_dim=128, ff=14336, vocab=128256),
}
DRY_SHAPE = dict(d=64, n_heads=4, n_kv=2, head_dim=16, ff=128, vocab=512)
DRY_MAX_L = 6
DRY_CAP = {"T": 128, "B": 4, "ctx": 64}

REF_L = 28
DEFAULT_WORKLOADS = ("pre:256,pre:512,pre:768,pre:1024,pre:1536,pre:2048,"
                     "dec:16:512,dec:64:512")
DEFAULT_RS = "0,4,7,10,14,18,21,24"
DEFAULT_PLACEMENTS = "contiguous_first,contiguous_last,balanced,random0"
DEFAULT_KS = "2,3"
QUICK_DEFAULTS = dict(workloads="pre:1024,pre:4096,dec:16:512", rs="0,14", ks="3")
SWEEP_WL = "pre:768"
SWEEP_R_REF = 14
SWEEP_KS = [1, 2, 3, 4, 6]
SWEEP_PLACEMENTS = ["balanced", "contiguous_first"]
KNOWN_PLACEMENTS = {"contiguous_first", "contiguous_last", "balanced", "random0", "random1"}

PRE_ITERS, PRE_WARM = 12, 4
DEC_ITERS, DEC_WARM = 40, 10
WARM_ITERS = 2            # untimed pass before every timed run
RESET_EVERY = 16          # reset the activation to its saved input every 16 repeats
WARM_SECONDS = 10.0       # resident-compute warm-up before each workload
LOCK_CAP_MHZ = 570        # auto clock lock: highest supported graphics clock <= this
SAMPLE_MS = 200

FP16 = torch.float16
ALIGN = 256               # alignment of fields inside a unit
SLAB_ALIGN = 4096         # alignment of units inside the host slab
RMS_EPS = 1e-5
FILL_CHUNK = 1 << 24

T0 = time.time()
BUDGET = 0.0
RECORDS = []
NOTES = []
STAGE = ["startup"]
CLOCKS = {"mode": None, "locked": False, "cmds": []}
SAMPLES = []              # (time, stage, sm_clock, power, temp, throttle_bits)
SAMPLES_LOCK = threading.Lock()


# ----------------------------------------------------------------- logging

def elapsed():
    return time.time() - T0


def set_stage(text):
    STAGE[0] = f"{text} (t={elapsed():.0f}s)"


def say(msg):
    print(f"PROGRESS t={elapsed():.1f}s {msg}", flush=True)


def emit(rec):
    RECORDS.append(rec)
    print("REC " + json.dumps(rec), flush=True)


def heartbeat():
    while True:
        time.sleep(30)
        print(f"HEARTBEAT t={elapsed():.0f}s stage={STAGE[0]}", flush=True)


def over_budget():
    return BUDGET > 0 and elapsed() > BUDGET


def rnd(x, nd=4):
    return None if x is None else round(float(x), nd)


def split_list(text):
    return [t.strip() for t in text.split(",") if t.strip()]


def meminfo():
    info = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                p = line.split()
                if len(p) >= 2 and p[0].endswith(":"):
                    info[p[0][:-1]] = int(p[1]) * 1024
    except OSError:
        pass
    return info


# ----------------------------------------------------------------- GPU clocks and sampling

def run_smi(args_list):
    try:
        p = subprocess.run(["nvidia-smi"] + args_list, capture_output=True, text=True, timeout=60)
        return p.returncode, p.stdout.strip(), p.stderr.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return -1, "", f"{type(exc).__name__}: {exc}"


def _record_cmd(args_list, rc, out, err):
    CLOCKS["cmds"].append(dict(cmd="nvidia-smi " + " ".join(args_list), rc=rc,
                               out=out[:400], err=err[:400]))


def setup_clocks(mode, dry):
    CLOCKS["mode"] = mode
    if dry or mode == "none":
        CLOCKS["reason"] = "dry run" if dry else "disabled"
        return
    q = ["--query-supported-clocks=graphics", "--format=csv,noheader,nounits"]
    rc, out, err = run_smi(q)
    _record_cmd(q, rc, out, err)
    if rc != 0:
        CLOCKS["reason"] = "clock query failed"
        return
    supported = []
    for line in out.splitlines():
        try:
            supported.append(int(float(line.strip())))
        except ValueError:
            pass
    if mode == "auto":
        cands = [c for c in supported if c <= LOCK_CAP_MHZ]
        if not cands:
            CLOCKS["reason"] = f"no supported graphics clock <= {LOCK_CAP_MHZ} MHz"
            return
        target = max(cands)
    else:
        try:
            target = int(float(mode))
        except ValueError:
            sys.exit(f"--lock-clocks must be none, auto or a MHz value, got {mode}")
    CLOCKS["target_mhz"] = target
    CLOCKS["supported_max_mhz"] = max(supported) if supported else None
    lock = ["-lgc", f"{target},{target}"]
    rc, out, err = run_smi(lock)
    _record_cmd(lock, rc, out, err)
    CLOCKS["locked"] = rc == 0
    if rc != 0:
        CLOCKS["reason"] = "lock failed; running unlocked"
    else:
        atexit.register(reset_clocks)


def reset_clocks():
    if not CLOCKS.get("locked") or CLOCKS.get("reset_done"):
        return
    rc, out, err = run_smi(["-rgc"])
    _record_cmd(["-rgc"], rc, out, err)
    CLOCKS["reset_done"] = True
    CLOCKS["reset_rc"] = rc


def _parse_sample_line(line):
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 4:
        return None

    def num(s):
        try:
            return float(s)
        except ValueError:
            return None

    bits = 0
    try:
        bits = int(parts[3], 16) if parts[3].lower().startswith("0x") else int(parts[3])
    except ValueError:
        bits = 0
    return num(parts[0]), num(parts[1]), num(parts[2]), bits


def start_sampler(dry):
    if dry:
        return None
    cmd = ["nvidia-smi", "--query-gpu=clocks.sm,power.draw,temperature.gpu,clocks_throttle_reasons.active",
           "--format=csv,noheader,nounits", "-lms", str(SAMPLE_MS)]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    except OSError as exc:
        NOTES.append(f"sampler unavailable ({type(exc).__name__}); clock/power stats omitted")
        return None

    def reader():
        for line in proc.stdout:
            parsed = _parse_sample_line(line)
            if parsed is None:
                continue
            clk, pwr, temp, bits = parsed
            with SAMPLES_LOCK:
                SAMPLES.append((time.time(), STAGE[0], clk, pwr, temp, bits))

    threading.Thread(target=reader, daemon=True).start()
    return proc


def stop_sampler(proc):
    if proc is not None and proc.poll() is None:
        proc.terminate()


class Window:
    """Context manager that brackets a measurement block for the sampler."""

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        self.t1 = time.time()
        return False

    def stats(self):
        with SAMPLES_LOCK:
            rows = [r for r in SAMPLES if self.t0 <= r[0] <= self.t1]
        clk = [r[2] for r in rows if r[2] is not None]
        pwr = [r[3] for r in rows if r[3] is not None]
        tmp = [r[4] for r in rows if r[4] is not None]
        masks = sorted({hex(r[5]) for r in rows if r[5]})
        return dict(
            sm_clock_mean=rnd(statistics.mean(clk), 1) if clk else None,
            sm_clock_min=rnd(min(clk), 1) if clk else None,
            power_mean=rnd(statistics.mean(pwr), 2) if pwr else None,
            temp_max=rnd(max(tmp), 1) if tmp else None,
            throttle_masks=masks,
            n_samples=len(rows),
        )


# ----------------------------------------------------------------- dry-run stand-ins

class DryStream:
    def wait_event(self, ev):
        pass

    def synchronize(self):
        pass


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
    """Byte offsets of the fields inside one layer's host unit."""
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


def slab_total(L, layer_bytes, head_bytes, head):
    """Size of the single pinned slab holding all units, each 4096-byte aligned."""
    sizes = [layer_bytes] * L + ([head_bytes] if head else [])
    off = 0
    for nb in sizes:
        off = _align(off, SLAB_ALIGN) + nb
    return _align(off, SLAB_ALIGN)


def views_layer(buf, lay):
    out = {}
    for name, (off, shape) in lay.items():
        n = math.prod(shape) * 2
        out[name] = buf[off:off + n].view(FP16).view(shape)
    return out


def fill_fp16(t_u8):
    """Fill a uint8 CPU tensor (view) with random fp16 values."""
    f = t_u8.view(FP16)
    n = f.numel()
    for s in range(0, n, FILL_CHUNK):
        e = min(n, s + FILL_CHUNK)
        f[s:e].copy_(torch.randn(e - s, dtype=torch.float32).mul_(0.02))


def rms(x, w):
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + RMS_EPS)
    return y.to(x.dtype) * w


# ----------------------------------------------------------------- placements

def scale_r(r_ref, L):
    return int(round(r_ref * L / REF_L))


def layer_set(placement, L, r):
    if r == 0:
        return []
    if placement == "contiguous_first":
        return list(range(r))
    if placement == "contiguous_last":
        return list(range(L - r, L))
    if placement == "balanced":
        return sorted({(2 * j + 1) * L // (2 * r) for j in range(r)})
    if placement.startswith("random"):
        seed = int(placement[len("random"):] or 0)
        rng = random.Random(1000 + seed)
        return sorted(rng.sample(range(L), r))
    raise ValueError(f"unknown placement {placement}")


# ----------------------------------------------------------------- benchmark object

class Bench:
    def __init__(self, name, cfg, dry, head, head_resident):
        self.name = name
        self.cfg = cfg
        self.dry = dry
        self.dev = "cpu" if dry else "cuda"
        self.L = cfg["L"]
        self.d = cfg["d"]
        self.vocab = cfg["vocab"]
        self.nh = cfg["n_heads"]
        self.nkv = cfg["n_kv"]
        self.hd = cfg["head_dim"]
        self.ff = cfg["ff"]
        self.lay, self.layer_bytes = layer_layout(cfg)
        self.head_bytes = self.vocab * self.d * 2
        self.head = head
        self.head_resident = head_resident
        self.N = self.L + (1 if head else 0)
        self.max_inner = 3 if dry else 500
        self.attn_mode = None
        self.attn_probe = {}
        self.cap = None
        self.slab = None
        self.slab_bytes = 0
        self.host = None
        self.comp = DryStream() if dry else torch.cuda.Stream()
        self.cpy = DryStream() if dry else torch.cuda.Stream()

    # unit helpers
    def ubytes(self, u):
        return self.layer_bytes if u < self.L else self.head_bytes

    def kind(self, u):
        return "layer" if u < self.L else "head"

    def uname(self, u):
        return f"layer{u}" if u < self.L else "head"

    # stream / event / sync helpers (plain calls in dry mode)
    def sctx(self, s):
        return nullcontext() if self.dry else torch.cuda.stream(s)

    def event(self, timing=False):
        return DryEvent() if self.dry else torch.cuda.Event(enable_timing=timing)

    def sync(self):
        if not self.dry:
            torch.cuda.synchronize()

    def empty_cache(self):
        gc.collect()
        if not self.dry:
            torch.cuda.empty_cache()

    # host memory: one pinned slab per model, units sliced from it
    def alloc_host(self):
        offs, off = [], 0
        for u in range(self.N):
            off = _align(off, SLAB_ALIGN)
            offs.append(off)
            off += self.ubytes(u)
        total = _align(off, SLAB_ALIGN)
        if self.dry:
            slab = torch.empty(total, dtype=torch.uint8)
        else:
            slab = torch.empty(total, dtype=torch.uint8, pin_memory=True)
            assert slab.is_pinned(), "host slab is not page-locked"
        self.slab = slab
        self.slab_bytes = total
        self.host = []
        for u in range(self.N):
            t = slab[offs[u]:offs[u] + self.ubytes(u)]
            fill_fp16(t)
            self.host.append(t)

    def free_host(self):
        self.host = None
        self.slab = None
        self.empty_cache()
        if hasattr(torch._C, "_host_emptyCache"):
            torch._C._host_emptyCache()

    def load_units(self, idx):
        bufs = {}
        for u in idx:
            b = torch.empty(self.ubytes(u), dtype=torch.uint8, device=self.dev)
            b.copy_(self.host[u])
            bufs[u] = b
        self.sync()
        return bufs

    def views(self, u, buf):
        if u < self.L:
            return views_layer(buf, self.lay)
        return {"W": buf[:self.head_bytes].view(FP16).view(self.vocab, self.d)}

    def slot_views(self, buf, S):
        sv = {}
        if any(u < self.L for u in S):
            sv["layer"] = views_layer(buf, self.lay)
        if any(u >= self.L for u in S):
            sv["head"] = {"W": buf[:self.head_bytes].view(FP16).view(self.vocab, self.d)}
        return sv

    def make_wl(self, spec):
        p = spec.split(":")
        if p[0] == "pre":
            wl = dict(name=spec, kind="pre", T=int(p[1]), n_iters=PRE_ITERS, warmup=PRE_WARM)
            x_shape = (wl["T"], self.d)
        else:
            wl = dict(name=spec, kind="dec", B=int(p[1]), ctx=int(p[2]), n_iters=DEC_ITERS, warmup=DEC_WARM)
            x_shape = (wl["B"], self.d)
            kv_shape = (wl["B"], self.nkv, wl["ctx"], self.hd)
            wl["K"] = (torch.randn(kv_shape, dtype=torch.float32, device=self.dev) * 0.1).to(FP16)
            wl["V"] = (torch.randn(kv_shape, dtype=torch.float32, device=self.dev) * 0.1).to(FP16)
        wl["x0"] = (torch.randn(x_shape, dtype=torch.float32, device=self.dev) * 0.1).to(FP16)
        self.sync()
        return wl

    # attention backend
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

        self.cap = None if self.dry else tuple(torch.cuda.get_device_capability(0))
        if self.cap is None or self.cap >= (8, 0):
            prefs = ["enable_gqa", "repeat_default"]
        else:
            # sm < 80: enable_gqa is not used; force the memory-efficient kernel
            prefs = ["repeat_efficient", "repeat_default"]
        mode = next((p for p in prefs if results.get(p) is True), None)
        if mode is None:
            raise RuntimeError(f"no attention mode works: {results}")
        if self.cap is not None and self.cap < (8, 0) and mode == "repeat_default":
            NOTES.append("memory-efficient attention unavailable on sm<80; default backend used")
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

    # compute
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

    def head_fwd(self, x, Wt, wl):
        x_last = x[-1:] if wl["kind"] == "pre" else x
        return x_last @ Wt.t()

    def compute(self, u, W, st, wl):
        if u < self.L:
            st["x"] = self.layer_fwd(st["x"], W, wl)
        else:
            st["logits"] = self.head_fwd(st["x"], W["W"], wl)

    # timing primitives
    def _timed_compute(self, u, W, wl, inner):
        a, b = self.event(True), self.event(True)
        with self.sctx(self.comp):
            st = {"x": wl["x0"]}
            a.record(self.comp)
            for r in range(inner):
                if r > 0 and r % RESET_EVERY == 0:
                    st = {"x": wl["x0"]}
                self.compute(u, W, st, wl)
            b.record(self.comp)
        self.comp.synchronize()
        return a.elapsed_time(b) / inner

    def time_compute_unit(self, u, W, wl, n_rep=9, warm=2):
        with self.sctx(self.comp):
            self.compute(u, W, {"x": wl["x0"]}, wl)
        self.comp.synchronize()
        self._timed_compute(u, W, wl, 1)          # first sizing pass (discarded)
        est = self._timed_compute(u, W, wl, 1)    # second sizing pass is used
        inner = 1 if est >= 20.0 else min(self.max_inner, max(1, math.ceil(20.0 / max(est, 1e-4))))
        samples = [self._timed_compute(u, W, wl, inner) for _ in range(n_rep)]
        return statistics.median(samples[warm:]), inner, samples

    def time_h2d_alone(self, u, slot, n_rep=9, warm=2):
        nb = self.ubytes(u)
        samples = []
        for _ in range(n_rep):
            a, b = self.event(True), self.event(True)
            with self.sctx(self.cpy):
                a.record(self.cpy)
                slot[:nb].copy_(self.host[u], non_blocking=True)
                b.record(self.cpy)
            self.cpy.synchronize()
            samples.append(a.elapsed_time(b))
        return statistics.median(samples[warm:]), samples

    def time_h2d_concurrent(self, u, slot, Wv, wl, n_comp, n_rep=9, warm=2):
        """Copy unit u while n_comp resident layers run on the compute stream.

        The copy is enqueued on the copy stream behind a start event recorded on
        comp, and before the compute loop, so it starts together with the compute.
        """
        nb = self.ubytes(u)
        samples, ovs = [], []
        for _ in range(n_rep):
            cs, ce = self.event(True), self.event(True)
            a, b = self.event(True), self.event(True)
            with self.sctx(self.comp):
                cs.record(self.comp)
            with self.sctx(self.cpy):
                self.cpy.wait_event(cs)
                a.record(self.cpy)
                slot[:nb].copy_(self.host[u], non_blocking=True)
                b.record(self.cpy)
            with self.sctx(self.comp):
                st = {"x": wl["x0"]}
                for j in range(n_comp):
                    if j > 0 and j % RESET_EVERY == 0:
                        st = {"x": wl["x0"]}
                    self.compute(j % self.L, Wv[j % self.L], st, wl)
                ce.record(self.comp)
            self.sync()
            ta = cs.elapsed_time(a)
            tb = cs.elapsed_time(b)
            tc = cs.elapsed_time(ce)
            ov = max(0.0, min(tb, tc) - max(ta, 0.0)) / max(tb - ta, 1e-9)
            samples.append(a.elapsed_time(b))
            ovs.append(ov)
        return statistics.median(samples[warm:]), statistics.median(ovs[warm:]), samples

    def warm_resident(self, wl, Wv, seconds):
        """Resident-compute loop for a fixed wall time, to reach a steady thermal/power state."""
        t_end = time.time() + seconds
        passes = 0
        while time.time() < t_end:
            st = {"x": wl["x0"]}
            with self.sctx(self.comp):
                for i in range(self.L):
                    self.compute(i, Wv[i], st, wl)
            self.comp.synchronize()
            passes += 1
        return passes

    def start_pump(self, src, dst):
        """Back-to-back copies src -> dst on the copy stream until stopped."""
        stop = threading.Event()
        nb = src.numel()

        def run():
            with self.sctx(self.cpy):
                while not stop.is_set():
                    for _ in range(4):
                        dst[:nb].copy_(src, non_blocking=True)
                    ev = self.event()
                    ev.record(self.cpy)
                    ev.synchronize()

        th = threading.Thread(target=run, daemon=True)
        th.start()
        return th, stop

    @staticmethod
    def stop_pump(th, stop):
        stop.set()
        th.join(timeout=120)

    # executor
    def _issue_copy(self, g, S, k, slots, comp_done, copy_done, total, tl=None):
        if g >= total:
            return
        m = len(S)
        u = S[g % m]
        nb = self.ubytes(u)
        slot = slots[g % k]
        with self.sctx(self.cpy):
            if g >= k:
                self.cpy.wait_event(comp_done[g - k])
            cs = self.event(True) if tl is not None else None
            if cs is not None:
                cs.record(self.cpy)
            slot[:nb].copy_(self.host[u], non_blocking=True)
            ev = self.event()
            ev.record(self.cpy)
            copy_done[g] = ev
            if cs is not None:
                ce = self.event(True)
                ce.record(self.cpy)
                tl.append(("copy", u, cs, ce, g))

    def _pass(self, wl, S, k, res_set, Wres, slots, sviews, n_iters, tl=None):
        """One pass over n_iters iterations.

        Returns iteration-boundary events, host enqueue times, and (if tl is a
        list) the per-unit events of the last iteration.
        """
        N = self.N
        m = len(S)
        total = n_iters * m
        comp_done, copy_done = {}, {}
        t0 = self.event(True)
        self.sync()
        t0.record(self.comp)
        for g in range(min(k, total)):
            self._issue_copy(g, S, k, slots, comp_done, copy_done, total, tl)
        ends = [t0]
        enq = []
        g = 0
        for it in range(n_iters):
            rec_now = tl is not None and it == n_iters - 1
            st = {"x": wl["x0"]}
            t_enq = time.perf_counter()
            for u in range(N):
                if u in res_set:
                    with self.sctx(self.comp):
                        s = self.event(True) if rec_now else None
                        if s is not None:
                            s.record(self.comp)
                        self.compute(u, Wres[u], st, wl)
                        if s is not None:
                            e = self.event(True)
                            e.record(self.comp)
                            tl.append(("compute", u, s, e))
                else:
                    W = sviews[g % k][self.kind(u)]
                    with self.sctx(self.comp):
                        self.comp.wait_event(copy_done[g])
                        s = self.event(True) if rec_now else None
                        if s is not None:
                            s.record(self.comp)
                        self.compute(u, W, st, wl)
                        if s is not None:
                            e = self.event(True)
                            e.record(self.comp)
                            tl.append(("compute", u, s, e))
                        ev = self.event()
                        ev.record(self.comp)
                        comp_done[g] = ev
                    self._issue_copy(g + k, S, k, slots, comp_done, copy_done, total, tl)
                    g += 1
                    for old in [x for x in comp_done if x < g - 2 * k]:
                        del comp_done[old]
                    for old in [x for x in copy_done if x < g - 2 * k]:
                        del copy_done[old]
            enq.append((time.perf_counter() - t_enq) * 1000.0)
            e = self.event(True)
            e.record(self.comp)
            ends.append(e)
        self.sync()
        return ends, enq, tl

    def run_stream(self, resident, k, wl, n_iters, warmup, gpu_res=None, timeline=False):
        res = sorted(set(resident))
        res_set = set(res)
        S = [u for u in range(self.N) if u not in res_set]
        m = len(S)
        k = min(k, m) if m > 0 else 0
        owned = gpu_res is None
        if owned:
            gpu_res = self.load_units(res)
        Wres = {u: self.views(u, gpu_res[u]) for u in res}
        max_b = max((self.ubytes(u) for u in S), default=0)
        slots = [torch.empty(max_b, dtype=torch.uint8, device=self.dev) for _ in range(k)]
        sviews = [self.slot_views(buf, S) for buf in slots]
        self.sync()

        self._pass(wl, S, k, res_set, Wres, slots, sviews, WARM_ITERS)   # untimed warm run
        ends, enq, _ = self._pass(wl, S, k, res_set, Wres, slots, sviews, n_iters)

        tl_out = None
        if timeline:
            # extra timed run of 2 iterations; only the second (steady) iteration is recorded
            tl_ends, _, tl_events = self._pass(wl, S, k, res_set, Wres, slots, sviews, 2, tl=[])
            base = tl_ends[1]                      # start event of the recorded (second) iteration
            tl_out = dict(compute=[], copy=[])
            for ent in tl_events:
                kind_, u, s, e = ent[:4]
                if kind_ == "compute":
                    tl_out["compute"].append([u, rnd(base.elapsed_time(s)), rnd(base.elapsed_time(e))])
                elif m <= ent[4] < 2 * m:          # transfers that feed the recorded iteration
                    tl_out["copy"].append([u, rnd(base.elapsed_time(s)), rnd(base.elapsed_time(e))])

        durs = [ends[i].elapsed_time(ends[i + 1]) for i in range(n_iters)]
        post = durs[warmup:] if len(durs) > warmup else durs
        window = ends[warmup].elapsed_time(ends[n_iters]) / (n_iters - warmup)
        out = dict(
            period=window,                               # mean over post-warmup window
            period_median=statistics.median(post),
            mean=statistics.mean(post),
            min=min(post),
            max=max(post),
            durations=durs,
            host_enqueue_ms=statistics.mean(enq[warmup:]) if enq[warmup:] else 0.0,
            k=k,
            m=m,
            streamed=S,
            slot_bytes=max_b,
            timeline=tl_out,
        )
        del slots, sviews, Wres, ends
        if owned:
            del gpu_res
        self.empty_cache()
        return out


# ----------------------------------------------------------------- measurement helpers

def stream_fields(out, lb):
    d = dict(
        period_ms=rnd(out["period"]), period_median_ms=rnd(out["period_median"]),
        mean_ms=rnd(out["mean"]), min_ms=rnd(out["min"]), max_ms=rnd(out["max"]),
        host_enqueue_ms=rnd(out["host_enqueue_ms"]),
        durations_ms=[rnd(x) for x in out["durations"]], slot_bytes=out["slot_bytes"],
    )
    if out.get("timeline") is not None:
        d["timeline_compute"] = out["timeline"]["compute"]   # [unit, start_ms, end_ms]
        d["timeline_copy"] = out["timeline"]["copy"]         # [unit, start_ms, end_ms]
    if lb is not None and lb > 0:
        d["lb_ms"] = rnd(lb)
        d["period_over_lb"] = rnd(out["period"] / lb)
    return d


def stream_run(bench, wl, res, k, placement, r, exp, f_alone, c_map, timeline):
    N = bench.N
    res = sorted(res)
    res_set = set(res)
    S = [u for u in range(N) if u not in res_set]
    m = len(S)
    set_stage(f"{bench.name} {wl['name']} {placement} r={r} k={k} m={m}")
    w = Window()
    with w:
        out = bench.run_stream(res, k, wl, wl["n_iters"], wl["warmup"], timeline=timeline)
    lb = None
    if c_map is not None and f_alone is not None:
        lb = max(sum(c_map[u] for u in range(N)), sum(f_alone[u] for u in S))
    rec = dict(exp=exp, model=bench.name, kind="stream", workload=wl["name"], placement=placement,
               r=r, k=out["k"], k_req=k, resident=res, streamed=S, m=m,
               n_iters=wl["n_iters"], warmup=wl["warmup"])
    rec.update(stream_fields(out, lb))
    rec.update(w.stats())
    emit(rec)
    return out


def grid(bench, wl, rs, placements, ks, f_alone, c_map, timeline):
    N, L = bench.N, bench.L
    seen = set()
    for r_ref in rs:
        r = scale_r(r_ref, L)
        if r < 0 or r > L:
            NOTES.append(f"skipped r_ref={r_ref} (scaled r={r}, L={L})")
            continue
        pls = ["none"] if r == 0 else placements
        for pl in pls:
            lay = [] if pl == "none" else layer_set(pl, L, r)
            res = lay + ([L] if bench.head_resident else [])
            m = N - len(res)
            for k in ks:
                keff = min(k, m) if m > 0 else 0
                key = (tuple(sorted(res)), keff)
                if key in seen:
                    continue
                seen.add(key)
                if over_budget():
                    NOTES.append(f"budget reached before {bench.name} {wl['name']} {pl} r={r} k={k}")
                    return
                stream_run(bench, wl, res, k, pl, len(lay), "grid", f_alone, c_map, timeline)


def sweep_spec(dry):
    return cap_spec(SWEEP_WL) if dry else SWEEP_WL


def sweep(bench, f_alone, c_map, timeline):
    L = bench.L
    wl = bench.make_wl(sweep_spec(bench.dry))
    r = scale_r(SWEEP_R_REF, L)
    seen = set()
    for pl in SWEEP_PLACEMENTS:
        lay = layer_set(pl, L, r)
        res = lay + ([L] if bench.head_resident else [])
        m = bench.N - len(res)
        for k in SWEEP_KS:
            keff = min(k, m) if m > 0 else 0
            key = (tuple(sorted(res)), keff)
            if key in seen:
                continue
            seen.add(key)
            if over_budget():
                NOTES.append("budget reached inside slot sweep")
                return
            stream_run(bench, wl, res, k, pl, len(lay), "sweep", f_alone, c_map, timeline)
    del wl
    bench.empty_cache()


# ----------------------------------------------------------------- per-model driver

def setup_model(name, head, dry):
    cfg = dict(MODELS[name])
    if dry:
        cfg.update(DRY_SHAPE)
        cfg["L"] = min(cfg["L"], DRY_MAX_L)
    lay, layer_bytes = layer_layout(cfg)
    head_bytes = cfg["vocab"] * cfg["d"] * 2
    avail = meminfo().get("MemAvailable", 12 * 2 ** 30)
    L_nom = cfg["L"]
    L = L_nom
    while L > 1 and slab_total(L, layer_bytes, head_bytes, head) > 0.7 * avail:
        L -= 1
    total = slab_total(L, layer_bytes, head_bytes, head)
    if total > 0.7 * avail:
        NOTES.append(f"{name}: skipped, slab {total / 2**30:.2f} GiB exceeds 70% of available host RAM")
        emit(dict(kind="host_skip", model=name, need_bytes=total, avail_bytes=avail))
        return None
    if L != L_nom:
        NOTES.append(f"{name}: L reduced {L_nom}->{L} to fit host RAM")
        emit(dict(kind="host_reduction", model=name, L_nominal=L_nom, L_used=L,
                  need_bytes=total, avail_bytes=avail))
    cfg["L"] = L
    return cfg


def run_model(name, args, dry, workloads, rs, placements, ks):
    head = args.head or args.head_resident
    cfg = setup_model(name, head, dry)
    if cfg is None:
        return {"skipped": True}
    bench = Bench(name, cfg, dry, head, args.head_resident)
    N, L = bench.N, bench.L
    emit(dict(kind="model", model=name, L=L, d=cfg["d"], n_heads=cfg["n_heads"], n_kv=cfg["n_kv"],
              head_dim=cfg["head_dim"], ff=cfg["ff"], vocab=cfg["vocab"], N=N,
              layer_bytes=bench.layer_bytes, head_bytes=bench.head_bytes if head else 0,
              head=head, head_resident=args.head_resident))

    set_stage(f"{name}: host alloc")
    say(f"{name}: allocating pinned slab for {N} units ({L} layers)")
    bench.alloc_host()
    slab_bytes = bench.slab_bytes
    emit(dict(kind="host_slab", model=name, slab_bytes=slab_bytes, pinned=not dry))
    bench.probe_attn()
    emit(dict(kind="attn", model=name, mode=bench.attn_mode, capability=bench.cap,
              probe=bench.attn_probe))
    max_b = max(bench.ubytes(u) for u in range(N))
    slot = torch.empty(max_b, dtype=torch.uint8, device=bench.dev)
    timeline = args.timeline

    f_alone = {}
    for u in range(N):
        set_stage(f"{name}: h2d alone {bench.uname(u)}")
        w = Window()
        with w:
            med, samples = bench.time_h2d_alone(u, slot)
        f_alone[u] = med
        emit(dict(kind="h2d", model=name, unit=bench.uname(u), unit_index=u,
                  bytes=bench.ubytes(u), f_ms=rnd(med), durations_ms=[rnd(x) for x in samples],
                  **w.stats()))

    c_cache = {}
    for spec in workloads:
        if over_budget():
            NOTES.append(f"budget reached before workload {spec} of {name}")
            break
        say(f"{name}: workload {spec}")
        wl = bench.make_wl(spec)
        set_stage(f"{name} {spec}: all-resident load")
        gpu_all = bench.load_units(range(N))
        Wv = {u: bench.views(u, gpu_all[u]) for u in range(N)}

        set_stage(f"{name} {spec}: warm-up")
        warm_s = 0.0 if dry else WARM_SECONDS
        w = Window()
        with w:
            passes = bench.warm_resident(wl, Wv, warm_s)
        emit(dict(kind="warmup", model=name, workload=spec, seconds=warm_s, passes=passes,
                  **w.stats()))

        c_map = {}
        for u in range(N):
            set_stage(f"{name} {spec}: compute {bench.uname(u)}")
            w = Window()
            with w:
                med, inner, samples = bench.time_compute_unit(u, Wv[u], wl)
            c_map[u] = med
            emit(dict(kind="compute", model=name, workload=spec, unit=bench.uname(u),
                      unit_index=u, c_ms=rnd(med), inner=inner, durations_ms=[rnd(x) for x in samples],
                      **w.stats()))
        c_cache[spec] = c_map

        # compute with a continuous copy loop on the copy stream
        set_stage(f"{name} {spec}: compute under copy load")
        pump = None if dry else bench.start_pump(bench.host[0], slot)
        try:
            for u in range(N):
                set_stage(f"{name} {spec}: compute_conc {bench.uname(u)}")
                w = Window()
                with w:
                    med, inner, samples = bench.time_compute_unit(u, Wv[u], wl)
                emit(dict(kind="compute_conc", model=name, workload=spec, unit=bench.uname(u),
                          unit_index=u, c_conc_ms=rnd(med), c_ms=rnd(c_map[u]), inner=inner,
                          durations_ms=[rnd(x) for x in samples], **w.stats()))
        finally:
            if pump is not None:
                Bench.stop_pump(*pump)

        c_layer = max(c_map[0], 1e-6)
        for u in range(N):
            set_stage(f"{name} {spec}: copy-under-compute {bench.uname(u)}")
            n_comp = min(4000, max(1, math.ceil(1.5 * f_alone[u] / c_layer)))
            w = Window()
            with w:
                fc, ov, samples = bench.time_h2d_concurrent(u, slot, Wv, wl, n_comp)
            emit(dict(kind="h2d_concurrent", model=name, workload=spec, unit=bench.uname(u),
                      unit_index=u, f_alone_ms=rnd(f_alone[u]), f_conc_ms=rnd(fc),
                      overlap_frac=rnd(ov, 3), n_comp=n_comp,
                      durations_ms=[rnd(x) for x in samples], **w.stats()))

        set_stage(f"{name} {spec}: all-resident period")
        w = Window()
        with w:
            allres = bench.run_stream(list(range(N)), 2, wl, wl["n_iters"], wl["warmup"], gpu_res=gpu_all,
                                      timeline=timeline)
        lb_all = sum(c_map.values())
        rec = dict(exp="allres", model=name, kind="stream", workload=spec, placement="all_resident",
                   r=L, k=0, k_req=0, resident=list(range(N)), streamed=[], m=0,
                   n_iters=wl["n_iters"], warmup=wl["warmup"])
        rec.update(stream_fields(allres, lb_all))
        rec.update(w.stats())
        emit(rec)
        del Wv, gpu_all
        bench.empty_cache()

        if not over_budget():
            grid(bench, wl, rs, placements, ks, f_alone, c_map, timeline)
        del wl
        bench.empty_cache()

    if not over_budget():
        say(f"{name}: slot sweep {sweep_spec(dry)}")
        sweep(bench, f_alone, c_cache.get(sweep_spec(dry)), timeline)
    else:
        NOTES.append(f"budget reached; slot sweep for {name} skipped")

    del slot
    bench.free_host()
    return {"L": L, "N": N, "attn_mode": bench.attn_mode, "slab_bytes": slab_bytes}


# ----------------------------------------------------------------- device info

def device_info(dry):
    mi = meminfo()
    info = dict(python=sys.version.split()[0], torch=torch.__version__, dry=dry,
                host_mem_total_bytes=mi.get("MemTotal"), host_mem_avail_bytes=mi.get("MemAvailable"))
    if dry:
        info["cuda"] = None
        return info
    p = torch.cuda.get_device_properties(0)
    info.update(name=torch.cuda.get_device_name(0), cuda=torch.version.cuda,
                total_mem_bytes=p.total_memory, capability=f"{p.major}.{p.minor}")
    rc, out, _ = run_smi(["--query-gpu=pcie.link.gen.max,pcie.link.width.max", "--format=csv,noheader"])
    info["pcie_max"] = out if rc == 0 else "unavailable"
    return info


# ----------------------------------------------------------------- main

def cap_spec(spec):
    p = spec.split(":")
    if p[0] == "pre":
        return f"pre:{min(int(p[1]), DRY_CAP['T'])}"
    if p[0] == "dec" and len(p) == 3:
        return f"dec:{min(int(p[1]), DRY_CAP['B'])}:{min(int(p[2]), DRY_CAP['ctx'])}"
    raise ValueError(f"bad workload {spec}")


def parse_workload(spec):
    p = spec.split(":")
    if p[0] == "pre" and len(p) == 2:
        int(p[1])
        return spec
    if p[0] == "dec" and len(p) == 3:
        int(p[1])
        int(p[2])
        return spec
    raise ValueError(f"bad workload {spec} (use pre:T or dec:B:ctx)")


def main():
    global BUDGET
    ap = argparse.ArgumentParser(description="Weight-streaming steady-state iteration benchmark.")
    ap.add_argument("--models", default="llama3.2-3b")
    ap.add_argument("--workloads", default=None)
    ap.add_argument("--rs", default=None, help="resident layer counts, reference L=28, scaled by L/28")
    ap.add_argument("--ks", default=None)
    ap.add_argument("--placements", default=None)
    ap.add_argument("--head", action="store_true", help="append the LM head unit")
    ap.add_argument("--head-resident", action="store_true", help="keep the LM head resident (implies --head)")
    ap.add_argument("--quick", action="store_true", help="small grid unless options are given")
    ap.add_argument("--dry", action="store_true", help="CPU control-flow check with tiny shapes")
    ap.add_argument("--budget", type=float, default=0.0,
                    help="seconds after which no new measurement starts (0 = unlimited)")
    ap.add_argument("--lock-clocks", default="auto",
                    help="none, auto (highest supported graphics clock <= 570 MHz) or a MHz value")
    ap.add_argument("--timeline", action=argparse.BooleanOptionalAction, default=True,
                    help="record per-unit compute and copy event times in stream records (default on)")
    args = ap.parse_args()

    if args.quick:
        for key in ("workloads", "rs", "ks"):
            if getattr(args, key) is None:
                setattr(args, key, QUICK_DEFAULTS[key])
    BUDGET = args.budget

    models = split_list(args.models)
    for name in models:
        if name not in MODELS:
            sys.exit(f"unknown model {name}; choices: {', '.join(MODELS)}")
    workloads = []
    for spec in split_list(args.workloads or DEFAULT_WORKLOADS):
        spec = parse_workload(spec)
        if args.dry:
            spec = cap_spec(spec)
        if spec not in workloads:
            workloads.append(spec)
    rs = [int(x) for x in split_list(args.rs or DEFAULT_RS)]
    ks = [int(x) for x in split_list(args.ks or DEFAULT_KS)]
    placements = split_list(args.placements or DEFAULT_PLACEMENTS)
    for pl in placements:
        if pl not in KNOWN_PLACEMENTS:
            sys.exit(f"unknown placement {pl}")
    if any(k < 1 for k in ks):
        sys.exit("ks must be >= 1")

    dry = args.dry
    if not dry and not torch.cuda.is_available():
        sys.exit("CUDA is not available; use --dry for a CPU control-flow check")

    threading.Thread(target=heartbeat, daemon=True).start()
    if len(models) > 1:
        msg = ("WARNING: several models in one process; they run sequentially with the host slab "
               "released between them. One model per run is recommended for clean host memory.")
        NOTES.append(msg)
        print(msg, flush=True)

    sampler = None
    models_out = {}
    try:
        setup_clocks(args.lock_clocks, dry)
        sampler = start_sampler(dry)
        info = device_info(dry)
        info["clocks"] = dict(CLOCKS)
        emit(dict(kind="device", **info))
        say(f"models={models} workloads={workloads} rs={rs} ks={ks} placements={placements} "
            f"head={args.head or args.head_resident} head_resident={args.head_resident} "
            f"timeline={args.timeline} clocks={CLOCKS.get('target_mhz', 'unlocked')} dry={dry}")
        for name in models:
            models_out[name] = run_model(name, args, dry, workloads, rs, placements, ks)
    finally:
        stop_sampler(sampler)
        reset_clocks()

    result = dict(device=info, clocks=CLOCKS, models=models_out, records=RECORDS, notes=NOTES,
                  budget_s=BUDGET, elapsed_s=round(elapsed(), 1))
    print("JSON_BEGIN", flush=True)
    print(json.dumps(result), flush=True)
    print("JSON_END", flush=True)


if __name__ == "__main__":
    main()
