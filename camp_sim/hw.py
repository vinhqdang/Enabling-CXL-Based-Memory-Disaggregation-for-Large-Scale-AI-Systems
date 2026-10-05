"""
Hardware descriptions for the CAMP simulator.

Two things are modelled separately and never mixed:

* ``GPU``   -- what executes a layer once its weights are resident in HBM
               (roofline compute model; see ``cost.py``).
* ``Link``  -- the *host-mediated* path that moves a weight tensor from a CXL
               Type-3 memory expander into GPU HBM.

Path semantics (see Section 3 of the paper)
-------------------------------------------
A CXL Type-3 expander is a CXL.mem endpoint behind the host root complex; the GPU
cannot address it directly.  The runtime pins the CXL-backed NUMA region as page-locked
host memory and issues ``cudaMemcpyAsync`` on a dedicated copy stream.  The GPU copy
engine then reads the data through the root complex, which forwards the reads to the
expander.  Consequently:

* the initiator is a host runtime thread (not the GPU),
* the end-to-end bandwidth is limited by the slower of {CXL device read bandwidth,
  PCIe x16 link} times a protocol efficiency, not by the CXL link alone,
* every copy pays a fixed software cost (API call, driver, completion event), and
* when the region is *not* page-locked the driver stages through a bounce buffer in host
  DRAM ("staged" mode), which adds a second hop.
"""
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class GPU:
    name: str = "H100-SXM-80GB"
    peak_flops: float = 989e12       # dense FP16/BF16 tensor-core FLOP/s (vendor datasheet)
    hbm_bw: float = 3.35e12          # bytes/s
    hbm_bytes: float = 80e9
    # "True" efficiencies used by the simulated device (the ground truth that the
    # runtime cost model in cost.py must *estimate*, it never reads these).
    eff_gemm: float = 0.62
    eff_mem: float = 0.86
    eff_attn: float = 0.42
    launch_us: float = 4.0           # per kernel launch
    sampling_gap_us: float = 150.0   # host-side sampling/scheduling between iterations


@dataclass(frozen=True)
class Link:
    name: str = "CXL-x8-measured"
    bw_gbs: float = 18.0             # sustained device->GPU bandwidth for large transfers (GB/s)
    latency_us: float = 12.0         # per-copy fixed cost: API + driver + CXL first-byte (~0.6 us) + completion
    mode: str = "direct"             # "direct" (pinned) or "staged" (bounce buffer through host DRAM)
    staged_bw_gbs: float = 8.0       # CPU copy rate of the staging hop (serial with the DMA hop)
    chunk_mb: float = 32.0           # DMA is split into chunks of this size
    per_chunk_us: float = 3.0        # per-chunk descriptor/doorbell cost
    sync_us: float = 6.0             # event signalling + stream wait before the consumer kernel
    jitter_sigma: float = 0.06       # per-copy lognormal multiplicative noise
    drift_sigma: float = 0.05        # slow AR(1) bandwidth drift (other traffic, thermals)
    drift_rho: float = 0.97

    def effective_bw(self) -> float:
        if self.mode == "staged":
            # bounce buffer: CPU copy into a staging buffer, then DMA; the two hops are serial
            return 1e9 / (1.0 / self.bw_gbs + 1.0 / self.staged_bw_gbs)
        return self.bw_gbs * 1e9

    def nominal_time(self, nbytes: float) -> float:
        """Noise-free service time of one copy of ``nbytes`` bytes."""
        chunks = max(1.0, nbytes / (self.chunk_mb * 1e6))
        return (self.latency_us + self.sync_us + chunks * self.per_chunk_us) * 1e-6 + nbytes / self.effective_bw()


# Presets.  Numeric ranges are anchored on published device measurements:
#   * ~10 GB/s sustained for a 2nd-generation Type-3 expander (Yoon et al., TraCT, 2025),
#   * ~18 GB/s DMA to the GPU for a CXL-attached prototype and 1.3 GB per-layer transfers
#     (Jang et al., ITME, 2026),
#   * PCIe Gen5 x16 pinned host->GPU copies of ~50 GB/s as the upper bound of the GPU side.
LINKS = {
    "cxl_gen1_measured": Link("cxl_gen1_measured", bw_gbs=10.0),
    "cxl_x8_measured": Link("cxl_x8_measured", bw_gbs=18.0),
    "cxl_x16_expected": Link("cxl_x16_expected", bw_gbs=45.0),
    "host_dram_pinned_gen5": Link("host_dram_pinned_gen5", bw_gbs=52.0, latency_us=10.0),
    "cxl_x8_staged": Link("cxl_x8_staged", bw_gbs=18.0, mode="staged"),
}

H100 = GPU()


def with_bw(link: Link, gbs: float) -> Link:
    return replace(link, bw_gbs=gbs, name=f"{link.name}@{gbs:g}")
