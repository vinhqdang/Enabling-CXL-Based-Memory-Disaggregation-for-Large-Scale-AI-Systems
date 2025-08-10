"""
XL-Share: CXL-Based Memory Disaggregation System for Large-Scale AI Models

This package implements a memory disaggregation system that enables efficient sharing
and utilization of large AI model parameters via CXL-attached memory pools.
"""

__version__ = "1.0.0"
__author__ = "AI Systems Research Lab"

from .memory_manager import CXLMemoryManager, LocalCache
from .prefetcher import ModelAwarePrefetcher
from .inference_engine import XLShareInferenceEngine, InferenceRequest, InferenceResult
from .emulator import CXLEmulator

__all__ = [
    "CXLMemoryManager",
    "LocalCache", 
    "ModelAwarePrefetcher",
    "XLShareInferenceEngine",
    "InferenceRequest",
    "InferenceResult",
    "CXLEmulator"
]