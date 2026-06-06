"""Layer 6: observable extraction from the EDM-MPS.

Reduced density matrices, single-time expectations, and the coupling-channel
polarization history (Eq. F2) read from a single final EDM, plus convergence
diagnostics for the driver.
"""

from __future__ import annotations

from .convergence import (
    align_on_coarse,
    is_converged,
    max_history_deviation,
    saturated,
)
from .extractor import ObservableExtractor

__all__ = [
    "ObservableExtractor",
    "align_on_coarse",
    "max_history_deviation",
    "is_converged",
    "saturated",
]
