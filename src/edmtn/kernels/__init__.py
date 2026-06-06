"""Layer 3: combined kernel-tensor (MPO) construction."""

from __future__ import annotations

from .base import KernelMPO, KernelProvider, picking_tensor
from .gaussian_mpo import GaussianKernelEngine

__all__ = [
    "KernelMPO",
    "KernelProvider",
    "picking_tensor",
    "GaussianKernelEngine",
]
