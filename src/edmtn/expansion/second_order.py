"""Second-order time-step expansion (Layer 4b).

``e^{eps H^-(t)} = [I + (1+i)/2 eps H^-][I + (1-i)/2 eps H^-] + O(eps^3)``,
which in tensor-product form replaces each ``B^phi S^phi`` factor with
``B^phi B^psi S^phi_2 S^psi_1`` where

    S^0_1 = S^0_2 = I,
    S^{phi!=0}_1 = (1 - i)/2 S^phi,
    S^{phi!=0}_2 = (1 + i)/2 S^phi.

Each step thus has two sub-steps; the system superoperators are the first-order
ones rescaled by the two complex coefficients.  Local error ``O(eps^3)``.
"""

from __future__ import annotations

import numpy as np

from .base import StepSuperoperators, TimeStepExpander, first_order_superoperators

_C1 = (1.0 - 1.0j) / 2.0  # first sub-step coefficient
_C2 = (1.0 + 1.0j) / 2.0  # second sub-step coefficient


class SecondOrderExpander(TimeStepExpander):
    """Second-order (two-sub-step) expansion."""

    order = 2

    def build(self, coupling_ops: list[np.ndarray], eps: float) -> StepSuperoperators:
        S = first_order_superoperators(coupling_ops, eps)
        d = coupling_ops[0].shape[0]

        # rescale the non-identity superoperators; keep S^0 = I in both families
        S1 = S.copy()
        S2 = S.copy()
        S1[1:] *= _C1
        S2[1:] *= _C2
        return StepSuperoperators(phys_dim=S.shape[0], d=d, families=[S1, S2], order=2)
