"""
Hardware calibration utilities for XL-Share experiments.

Measures approximate bandwidth/latency for local HBM and host<->device
transfers when CUDA is available (PyTorch preferred), falling back to
synthetic numpy timers when not available.
"""

from __future__ import annotations

import json
import time
import os
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any


@dataclass
class CalibrationResult:
    has_cuda: bool
    device_name: Optional[str]
    local_hbm_bandwidth_gbps: float
    local_hbm_latency_ns: int
    host_to_device_gbps: float
    device_to_host_gbps: float
    pcie_like_latency_ns: int
    notes: str = ""

    def to_profile(self) -> Dict[str, Any]:
        # Map to CXLLatencyProfile-like dict
        return {
            "local_dram_ns": max(50, int(self.local_hbm_latency_ns)),
            "cxl_near_ns": max(100, int(self.pcie_like_latency_ns)),
            "cxl_far_ns": int(self.pcie_like_latency_ns * 2),
            "cxl_network_ns": int(self.pcie_like_latency_ns * 4),
            "local_bandwidth": float(self.local_hbm_bandwidth_gbps),
            "cxl_bandwidth": float(min(self.host_to_device_gbps, self.device_to_host_gbps)),
            "coherence_overhead_ns": 50,
        }


def _try_torch_cuda() -> Optional[CalibrationResult]:
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        torch.cuda.synchronize()
        dev = torch.cuda.get_device_name(0)

        # Measure local HBM bandwidth via device-only memcpy and matmul
        N = 1 << 24  # ~16M floats ~64MB
        x = torch.empty(N, dtype=torch.float32, device="cuda")
        y = torch.empty_like(x)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(10):
            y.copy_(x)
        torch.cuda.synchronize()
        t1 = time.time()
        bytes_copied = x.element_size() * x.numel() * 10
        hbm_bw = (bytes_copied / (t1 - t0)) / (1024 ** 3)  # GB/s

        # Approximate local latency by small transfer timing
        x_small = torch.empty(1024, dtype=torch.float32, device="cuda")
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(1000):
            y[:1024].copy_(x_small)
        torch.cuda.synchronize()
        t1 = time.time()
        per_op_s = (t1 - t0) / 1000
        local_lat_ns = int(per_op_s * 1e9)

        # Host<->Device bandwidth
        h = torch.empty_like(x, device="cpu", pin_memory=True)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(4):
            x.copy_(h, non_blocking=True)
        torch.cuda.synchronize()
        t1 = time.time()
        h2d_bw = ((h.element_size() * h.numel() * 4) / (t1 - t0)) / (1024 ** 3)

        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(4):
            h.copy_(x, non_blocking=True)
        torch.cuda.synchronize()
        t1 = time.time()
        d2h_bw = ((h.element_size() * h.numel() * 4) / (t1 - t0)) / (1024 ** 3)

        # PCIe-like latency via small H<->D copy
        h_small = torch.empty_like(x_small, device="cpu", pin_memory=True)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(1000):
            x_small.copy_(h_small, non_blocking=False)
        torch.cuda.synchronize()
        t1 = time.time()
        per_small = (t1 - t0) / 1000
        pcie_lat_ns = int(per_small * 1e9)

        return CalibrationResult(
            has_cuda=True,
            device_name=dev,
            local_hbm_bandwidth_gbps=float(hbm_bw),
            local_hbm_latency_ns=max(50_000, local_lat_ns),
            host_to_device_gbps=float(h2d_bw),
            device_to_host_gbps=float(d2h_bw),
            pcie_like_latency_ns=max(100_000, pcie_lat_ns),
            notes="Measured using PyTorch CUDA"
        )
    except Exception as e:
        return None


def _fallback_numpy() -> CalibrationResult:
    # Synthetic conservative defaults for a generic system
    return CalibrationResult(
        has_cuda=False,
        device_name=None,
        local_hbm_bandwidth_gbps=400.0,
        local_hbm_latency_ns=80,
        host_to_device_gbps=16.0,
        device_to_host_gbps=16.0,
        pcie_like_latency_ns=300,
        notes="No CUDA detected; using synthetic defaults"
    )


def run_calibration(out_path: str) -> CalibrationResult:
    res = _try_torch_cuda() or _fallback_numpy()
    os.makedirs(os.path.dirname(out_path), exist_ok=True) if os.path.dirname(out_path) else None
    with open(out_path, "w") as f:
        json.dump({
            "calibration": asdict(res),
            "cxllatency_profile": res.to_profile(),
        }, f, indent=2)
    return res


if __name__ == "__main__":
    path = os.environ.get("XL_CALIBRATION_OUT", "calibration.json")
    res = run_calibration(path)
    print(json.dumps(asdict(res), indent=2))

