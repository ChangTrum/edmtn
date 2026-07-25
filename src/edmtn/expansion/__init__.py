"""Layer 4b: time-step (Trotter) expansion of the small-step propagator."""

from __future__ import annotations

from .base import (
    StepSuperoperators,
    TimeStepExpander,
    anticommutator_superoperator,
    apply_superoperator,
    commutator_superoperator,
    first_order_superoperators,
)
from .dissipative import (
    DissipativeExpander,
    amplitude_damping_matrix,
    cavity_damping_channel,
)
from .first_order import FirstOrderExpander
from .second_order import SecondOrderExpander

__all__ = [
    "TimeStepExpander",
    "StepSuperoperators",
    "FirstOrderExpander",
    "SecondOrderExpander",
    "DissipativeExpander",
    "amplitude_damping_matrix",
    "cavity_damping_channel",
    "first_order_superoperators",
    "commutator_superoperator",
    "anticommutator_superoperator",
    "apply_superoperator",
]
