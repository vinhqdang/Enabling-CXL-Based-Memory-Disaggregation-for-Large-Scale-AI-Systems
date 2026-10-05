"""
Compute-time models.

``true_time``   -- ground-truth execution time of one layer on the *simulated device*.
                   Used only by the simulator engine.
``CostModel``   -- the runtime estimator available to CAMP.  It receives the same layer
                   features (FLOPs and bytes, derived from tensor shapes and the live batch
                   descriptor) but uses *nominal* efficiency constants and then calibrates
                   itself online from observed layer durations.  It never reads the true
                   efficiencies or any offline profile.
"""
from dataclasses import dataclass
from typing import Dict, Tuple

# Features of one layer execution:
#   (gemm_flops, gemm_bytes, core_flops, core_bytes, n_kernels)
Feat = Tuple[float, float, float, float, int]


def true_time(feat: Feat, gpu) -> float:
    gf, gb, cf, cb, nk = feat
    t_gemm = max(gf / (gpu.peak_flops * gpu.eff_gemm), gb / (gpu.hbm_bw * gpu.eff_mem)) if (gf or gb) else 0.0
    t_core = max(cf / (gpu.peak_flops * gpu.eff_attn), cb / (gpu.hbm_bw * gpu.eff_mem)) if (cf or cb) else 0.0
    return t_gemm + t_core + nk * gpu.launch_us * 1e-6


@dataclass
class CostModel:
    """Online-calibrated roofline estimator (no offline profiling).

    ``nominal_*`` are generic, device-class constants (what a runtime can assume without
    measuring the specific model); ``bias`` and ``noise_sigma`` let experiments degrade the
    estimator to study sensitivity; ``alpha`` is the EMA rate of the online correction.
    """
    peak_flops: float
    hbm_bw: float
    nominal_gemm: float = 0.50
    nominal_mem: float = 0.80
    nominal_attn: float = 0.35
    launch_us: float = 5.0
    alpha: float = 0.3
    online: bool = True
    bias: float = 1.0
    noise_sigma: float = 0.0
    seed: int = 0

    def __post_init__(self):
        import random
        self.corr: Dict[str, float] = {}
        self._rng = random.Random(self.seed)

    def nominal(self, feat: Feat) -> float:
        gf, gb, cf, cb, nk = feat
        t_gemm = max(gf / (self.peak_flops * self.nominal_gemm), gb / (self.hbm_bw * self.nominal_mem)) if (gf or gb) else 0.0
        t_core = max(cf / (self.peak_flops * self.nominal_attn), cb / (self.hbm_bw * self.nominal_mem)) if (cf or cb) else 0.0
        return (t_gemm + t_core + nk * self.launch_us * 1e-6) * self.bias

    def predict(self, feat: Feat, kind: str) -> float:
        t = self.nominal(feat) * self.corr.get(kind, 1.0)
        if self.noise_sigma:
            t *= self._rng.lognormvariate(0.0, self.noise_sigma)
        return t

    def observe(self, feat: Feat, kind: str, observed: float):
        if not self.online:
            return
        nom = self.nominal(feat)
        if nom <= 0:
            return
        r = observed / nom
        prev = self.corr.get(kind)
        self.corr[kind] = r if prev is None else (1 - self.alpha) * prev + self.alpha * r
