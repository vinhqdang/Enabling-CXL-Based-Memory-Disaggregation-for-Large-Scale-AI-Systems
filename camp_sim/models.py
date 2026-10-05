"""
Model descriptions and execution traces.

A model is decomposed into *units* -- the granularity at which weights are fetched and
pinned: one attention block (QKV + output projection + norms), one feed-forward block,
the embedding table and the output head.  A *trace* is the ordered list of unit accesses
of one or more iterations (a prefill pass, a decode step, ...), each carrying

* the bytes that must be present in HBM (a full unit, or a small gathered slice for an
  embedding lookup),
* the true execution time on the simulated GPU (roofline, ``cost.true_time``), and
* the layer features the runtime cost model sees (``cost.Feat``).

Architecture hyper-parameters follow the public model cards (Llama-2 / Llama-3 / Gemma-2);
the encoder-decoder and speculative-decoding models are synthetic but built from the same
unit types.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .cost import Feat, true_time


@dataclass(frozen=True)
class ModelSpec:
    name: str
    n_layers: int
    d_model: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    d_ff: int
    vocab: int
    gated: bool = True
    tied: bool = False
    bpp: int = 2            # bytes per weight parameter (2 = FP16/BF16)
    kv_bpp: int = 2         # bytes per KV element

    # ---- parameter counts ------------------------------------------------
    @property
    def attn_params(self) -> int:
        d, h, hk, dh = self.d_model, self.n_heads, self.n_kv_heads, self.head_dim
        return d * h * dh + 2 * d * hk * dh + h * dh * d + 2 * d  # + 2 norm vectors

    @property
    def mlp_params(self) -> int:
        return (3 if self.gated else 2) * self.d_model * self.d_ff

    @property
    def embed_params(self) -> int:
        return self.vocab * self.d_model

    @property
    def total_params(self) -> int:
        n = self.n_layers * (self.attn_params + self.mlp_params) + self.embed_params
        if not self.tied:
            n += self.embed_params
        return n

    @property
    def weight_bytes(self) -> float:
        return self.total_params * self.bpp

    def kv_bytes_per_token(self) -> float:
        return 2 * self.n_layers * self.n_kv_heads * self.head_dim * self.kv_bpp


PRESETS: Dict[str, ModelSpec] = {
    "llama2-7b": ModelSpec("llama2-7b", 32, 4096, 32, 32, 128, 11008, 32000),
    "llama2-13b": ModelSpec("llama2-13b", 40, 5120, 40, 40, 128, 13824, 32000),
    "llama3-8b": ModelSpec("llama3-8b", 32, 4096, 32, 8, 128, 14336, 128256),
    "llama3-70b": ModelSpec("llama3-70b", 80, 8192, 64, 8, 128, 28672, 128256),
    "llama3-405b": ModelSpec("llama3-405b", 126, 16384, 128, 8, 128, 53248, 128256),
    "gemma2-9b": ModelSpec("gemma2-9b", 42, 3584, 16, 8, 256, 14336, 256000, tied=True),
    "gpt2-124m": ModelSpec("gpt2-124m", 12, 768, 12, 12, 64, 3072, 50257, gated=False, tied=True),
}


@dataclass
class Unit:
    uid: int
    name: str
    kind: str          # 'attn' | 'mlp' | 'embed' | 'head' | 'xattn'
    nbytes: float


@dataclass
class Access:
    uid: int
    kind: str
    fetch_bytes: Optional[float]   # None -> whole unit must be resident; else gather of this many bytes
    t_true: float                  # seconds, noise-free
    feat: Feat
    it: int = 0                    # iteration index


@dataclass
class Trace:
    units: Dict[int, Unit]
    acc: List[Access]
    iter_starts: List[int] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.pos: Dict[int, List[int]] = {}
        for i, a in enumerate(self.acc):
            self.pos.setdefault(a.uid, []).append(i)
        self._ptr: Dict[int, int] = {}

    def next_use(self, uid: int, after: int) -> float:
        """Index of the first access to ``uid`` strictly after ``after`` (inf if none)."""
        import bisect
        p = self.pos.get(uid)
        if not p:
            return float("inf")
        k = bisect.bisect_right(p, after)
        return p[k] if k < len(p) else float("inf")

    def full_fetch_uids(self):
        return {a.uid for a in self.acc if a.fetch_bytes is None}

    @property
    def total_unit_bytes(self) -> float:
        return sum(u.nbytes for u in self.units.values())


# --------------------------------------------------------------------------
# unit construction
# --------------------------------------------------------------------------
def build_units(m: ModelSpec, offset: int = 0, prefix: str = "") -> Tuple[Dict[int, Unit], Dict[str, int]]:
    units: Dict[int, Unit] = {}
    idx: Dict[str, int] = {}

    def add(name, kind, params):
        uid = offset + len(units)
        units[uid] = Unit(uid, prefix + name, kind, params * m.bpp)
        idx[name] = uid
        return uid

    add("embed", "embed", m.embed_params)
    for l in range(m.n_layers):
        add(f"attn{l}", "attn", m.attn_params)
        add(f"mlp{l}", "mlp", m.mlp_params)
    if m.tied:
        idx["head"] = idx["embed"]
    else:
        add("head", "head", m.embed_params)
    return units, idx


# --------------------------------------------------------------------------
# per-layer features for a batch descriptor
#   B: sequences, S: new tokens per sequence, ctx: total context length incl. new tokens
#   prefix: tokens already in KV before this step (so ctx = prefix + S)
# --------------------------------------------------------------------------
def feat_attn(m: ModelSpec, B: int, S: int, ctx: int) -> Feat:
    T = B * S
    gemm_flops = 2.0 * m.attn_params * T
    gemm_bytes = m.attn_params * m.bpp + 2.0 * T * m.d_model * m.bpp * 3
    # causal attention: each new token attends to (ctx - S) + (S+1)/2 positions on average
    avg_ctx = (ctx - S) + (S + 1) / 2.0
    core_flops = 4.0 * B * S * avg_ctx * m.n_heads * m.head_dim
    kv_read = B * ctx * 2.0 * m.n_kv_heads * m.head_dim * m.kv_bpp
    kv_write = B * S * 2.0 * m.n_kv_heads * m.head_dim * m.kv_bpp
    return (gemm_flops, gemm_bytes, core_flops, kv_read + kv_write, 7)


def feat_mlp(m: ModelSpec, B: int, S: int) -> Feat:
    T = B * S
    flops = 2.0 * m.mlp_params * T
    nbytes = m.mlp_params * m.bpp + 2.0 * T * (m.d_model + m.d_ff) * m.bpp
    return (flops, nbytes, 0.0, 0.0, 4)


def feat_head(m: ModelSpec, B: int) -> Feat:
    flops = 2.0 * m.embed_params * B          # logits for the last position of each sequence
    nbytes = m.embed_params * m.bpp
    return (flops, nbytes, 0.0, 0.0, 3)


def feat_embed(m: ModelSpec, B: int, S: int) -> Feat:
    return (0.0, 2.0 * B * S * m.d_model * m.bpp, 0.0, 0.0, 1)


def iteration_accesses(m: ModelSpec, units: Dict[int, Unit], idx: Dict[str, int], gpu,
                       B: int, S: int, ctx: int, it: int = 0) -> List[Access]:
    out: List[Access] = []
    T = B * S
    f = feat_embed(m, B, S)
    out.append(Access(idx["embed"], "embed", min(units[idx["embed"]].nbytes, T * m.d_model * m.bpp),
                      true_time(f, gpu), f, it))
    for l in range(m.n_layers):
        fa = feat_attn(m, B, S, ctx)
        out.append(Access(idx[f"attn{l}"], "attn", None, true_time(fa, gpu), fa, it))
        fm = feat_mlp(m, B, S)
        out.append(Access(idx[f"mlp{l}"], "mlp", None, true_time(fm, gpu), fm, it))
    fh = feat_head(m, B)
    out.append(Access(idx["head"], "head", None, true_time(fh, gpu), fh, it))
    return out


# --------------------------------------------------------------------------
# trace builders for the workloads used in the paper
# --------------------------------------------------------------------------
def trace_decode(m: ModelSpec, gpu, B: int, ctx: int, steps: int) -> Trace:
    units, idx = build_units(m)
    acc: List[Access] = []
    starts = []
    for s in range(steps):
        starts.append(len(acc))
        acc += iteration_accesses(m, units, idx, gpu, B, 1, ctx + s, it=s)
    return Trace(units, acc, starts, dict(kind="decode", B=B, ctx=ctx, steps=steps, tokens_per_iter=B))


def trace_prefill(m: ModelSpec, gpu, B: int, S: int, reps: int = 3) -> Trace:
    units, idx = build_units(m)
    acc: List[Access] = []
    starts = []
    for r in range(reps):
        starts.append(len(acc))
        acc += iteration_accesses(m, units, idx, gpu, B, S, S, it=r)
    return Trace(units, acc, starts, dict(kind="prefill", B=B, S=S, reps=reps, tokens_per_iter=B * S))


def trace_mixed(m: ModelSpec, gpu, B_dec: int, ctx: int, chunk: int, steps: int) -> Trace:
    """Chunked-prefill serving iteration: ``B_dec`` decode tokens piggy-back on a prefill chunk.

    The GEMM layers see ``B_dec + chunk`` tokens; attention is split into the decode part
    (context ``ctx``) and the prefill-chunk part (context ``chunk``); features are summed.
    """
    units, idx = build_units(m)
    acc: List[Access] = []
    starts = []
    for s in range(steps):
        starts.append(len(acc))
        T = B_dec + chunk
        f = feat_embed(m, T, 1)
        acc.append(Access(idx["embed"], "embed", min(units[idx["embed"]].nbytes, T * m.d_model * m.bpp),
                          true_time(f, gpu), f, s))
        for l in range(m.n_layers):
            fd = feat_attn(m, B_dec, 1, ctx + s)
            fc = feat_attn(m, 1, chunk, chunk)
            fa = (2.0 * m.attn_params * T,
                  m.attn_params * m.bpp + 6.0 * T * m.d_model * m.bpp,
                  fd[2] + fc[2], fd[3] + fc[3], 7)
            acc.append(Access(idx[f"attn{l}"], "attn", None, true_time(fa, gpu), fa, s))
            fm = feat_mlp(m, 1, T)
            acc.append(Access(idx[f"mlp{l}"], "mlp", None, true_time(fm, gpu), fm, s))
        fh = feat_head(m, B_dec + 1)
        acc.append(Access(idx["head"], "head", None, true_time(fh, gpu), fh, s))
    return Trace(units, acc, starts, dict(kind="mixed", B_dec=B_dec, ctx=ctx, chunk=chunk, steps=steps,
                                          tokens_per_iter=B_dec + chunk))


def weights_budget(m: ModelSpec, gpu, B: int, ctx_max: int, workspace: float = 4e9) -> float:
    """HBM bytes left for the weight cache after KV cache and workspace."""
    kv = B * ctx_max * m.kv_bytes_per_token()
    return max(0.0, gpu.hbm_bytes - kv - workspace)


# --------------------------------------------------------------------------
# multi-model workloads with genuinely non-uniform reuse
# --------------------------------------------------------------------------
def trace_specdec(target: ModelSpec, draft: ModelSpec, gpu, B: int, ctx: int, k: int, steps: int) -> Trace:
    """Speculative decoding: ``k`` draft decode steps followed by one target verification of k+1 tokens.

    Draft-model weights are touched ``k`` times per iteration, target weights once.
    """
    ut, it_ = build_units(target, 0, "T.")
    ud, id_ = build_units(draft, len(ut), "D.")
    units = {**ut, **ud}
    acc: List[Access] = []
    starts = []
    for s in range(steps):
        starts.append(len(acc))
        c0 = ctx + s * (k + 1)
        for j in range(k):
            acc += iteration_accesses(draft, ud, id_, gpu, B, 1, c0 + j, it=s)
        acc += iteration_accesses(target, ut, it_, gpu, B, k + 1, c0 + k + 1, it=s)
    return Trace(units, acc, starts, dict(kind="specdec", B=B, k=k, tokens_per_iter=B * (k + 1)))


def trace_encdec(enc: ModelSpec, dec: ModelSpec, gpu, B: int, s_enc: int, t_dec: int, requests: int) -> Trace:
    """Encoder-decoder request stream: one encoder pass then ``t_dec`` decoder steps per request
    (cross-attention is folded into the decoder blocks; this is a structural, not a numerical, model)."""
    ue, ie = build_units(enc, 0, "E.")
    ud, id_ = build_units(dec, len(ue), "D.")
    units = {**ue, **ud}
    acc: List[Access] = []
    starts = []
    it = 0
    for r in range(requests):
        starts.append(len(acc))
        acc += iteration_accesses(enc, ue, ie, gpu, B, s_enc, s_enc, it=it)
        it += 1
        for t in range(t_dec):
            starts.append(len(acc))
            acc += iteration_accesses(dec, ud, id_, gpu, B, 1, s_enc + t, it=it)
            it += 1
    return Trace(units, acc, starts, dict(kind="encdec", B=B, t_dec=t_dec, requests=requests,
                                          iters_per_request=1 + t_dec, tokens_per_iter=B))
