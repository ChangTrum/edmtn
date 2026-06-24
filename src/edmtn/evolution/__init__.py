"""Layer 5: MPS evolution engine for the EDM tensor network.

Drives the forward recursive construction of the extended-density-matrix MPS
(Fig. 3d / Fig. 9b): each step applies the combined-kernel MPO and the new
system superoperator, then recompresses the bonds.
"""

from __future__ import annotations

from .canonicalize import CanonicalizationStrategy, CholeskyQR, HouseholderQR
from .mps_utils import (
    EDMMPS,
    apply_step,
    compress,
    dense_open_armed_correlation,
    dense_reduced_density_matrix,
    left_canonicalize,
    truncate,
)
from .separable_bath import SeparableBathEvolution, SeparableEvolutionResult
from .single_bath import EvolutionResult, SingleBathEvolution

__all__ = [
    "EDMMPS",
    "SingleBathEvolution",
    "EvolutionResult",
    "SeparableBathEvolution",
    "SeparableEvolutionResult",
    "CanonicalizationStrategy",
    "HouseholderQR",
    "CholeskyQR",
    "apply_step",
    "compress",
    "left_canonicalize",
    "truncate",
    "dense_open_armed_correlation",
    "dense_reduced_density_matrix",
]
