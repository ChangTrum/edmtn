"""Layer 5: MPS evolution engine for the EDM tensor network.

Drives the forward recursive construction of the extended-density-matrix MPS
(Fig. 3d / Fig. 9b): each step applies the combined-kernel MPO and the new
system superoperator, then recompresses the bonds.
"""

from __future__ import annotations

from .mps_utils import (
    EDMMPS,
    apply_step,
    dense_open_armed_correlation,
    dense_reduced_density_matrix,
)
from .quimb_edm import QuimbEDM
from .separable_bath import SeparableBathEvolution, SeparableEvolutionResult
from .single_bath import EvolutionResult, SingleBathEvolution

__all__ = [
    "EDMMPS",
    "QuimbEDM",
    "SingleBathEvolution",
    "EvolutionResult",
    "SeparableBathEvolution",
    "SeparableEvolutionResult",
    "apply_step",
    "dense_open_armed_correlation",
    "dense_reduced_density_matrix",
]
