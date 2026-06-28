"""edmtn: Extended Density Matrix (EDM) tensor-network solver for open quantum systems.

A quimb + CuPy implementation of the polynomial-complexity EDM formalism for
non-Markovian open-quantum-system dynamics.

The package is organised in layers:

    backend/       array + linalg backend abstraction (Layer 0)
    models/        physical model definitions          (Layer 1)
    cumulants/     bath cumulant / correlation engines  (Layer 2)
    kernels/       kernel-tensor (MPO) construction      (Layer 3)
    expansion/     Trotter time-step expansion           (Layer 4)
    evolution/     quimb-backed MPS evolution + compression (Layer 5)
    observables/   observable extraction                 (Layer 6)
    driver/        orchestration                          (Layer 7)

Importing this package makes the backend abstraction available and registers
the decomposition backends.
"""

from __future__ import annotations

from . import (
    backend,
    cumulants,
    driver,
    evolution,
    expansion,
    kernels,
    models,
    observables,
)

__version__ = "0.0.1"

__all__ = [
    "backend",
    "models",
    "cumulants",
    "kernels",
    "expansion",
    "evolution",
    "observables",
    "driver",
    "__version__",
]
