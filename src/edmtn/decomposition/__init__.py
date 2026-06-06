"""Layer 4a: decomposition / compression strategies."""

from __future__ import annotations

from .base import DecompositionResult, DecompositionStrategy, truncation_rank
from .standard_svd import StandardSVD

__all__ = [
    "DecompositionStrategy",
    "DecompositionResult",
    "truncation_rank",
    "StandardSVD",
]
