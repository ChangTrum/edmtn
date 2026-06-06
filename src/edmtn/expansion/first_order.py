"""First-order time-step expansion (Layer 4b).

``e^{eps H^-(t)} ~ I + eps sum_phi S^phi(t) (x) B_phi(t)`` with local error
``O(eps^2)``.  A single family of system superoperators per step.
"""

from __future__ import annotations

import numpy as np

from .base import StepSuperoperators, TimeStepExpander, first_order_superoperators


class FirstOrderExpander(TimeStepExpander):
    """First-order (single-sub-step) expansion."""

    order = 1

    def build(self, coupling_ops: list[np.ndarray], eps: float) -> StepSuperoperators:
        S = first_order_superoperators(coupling_ops, eps)
        d = coupling_ops[0].shape[0]
        return StepSuperoperators(phys_dim=S.shape[0], d=d, families=[S], order=1)
