"""Convergence diagnostics for EDM observable histories (Layer 6).

Helpers for the driver (Layer 7) to judge whether a result is converged with
respect to the time step, bond-dimension cutoff, or (for separable baths) the
number of sub-baths.  All operate on observable *histories* ``(times, values)``
produced by :class:`~edmtn.observables.extractor.ObservableExtractor`.
"""

from __future__ import annotations

import numpy as np


def align_on_coarse(times_coarse, values_coarse, times_fine, values_fine, *, atol=1e-9):
    """Sample the fine history at the coarse times; return aligned value pairs.

    The fine grid is assumed to contain (approximately) every coarse time -- the
    typical ``eps`` vs ``eps/2`` refinement.  Coarse times without a fine match
    are dropped.
    """
    tf = np.asarray(times_fine)
    vf = np.asarray(values_fine)
    vc_out, vf_out, t_out = [], [], []
    for tc, vc in zip(times_coarse, values_coarse):
        j = int(np.argmin(np.abs(tf - tc)))
        if abs(tf[j] - tc) <= atol + 1e-6 * abs(tc):
            t_out.append(tc)
            vc_out.append(vc)
            vf_out.append(vf[j])
    return (
        np.asarray(t_out),
        np.asarray(vc_out),
        np.asarray(vf_out),
    )


def max_history_deviation(times_coarse, values_coarse, times_fine, values_fine, *, atol=1e-9):
    """Maximum absolute deviation between two histories on the common coarse grid."""
    _, vc, vf = align_on_coarse(times_coarse, values_coarse, times_fine, values_fine, atol=atol)
    if vc.size == 0:
        raise ValueError("histories share no comparable time points")
    return float(np.max(np.abs(vc - vf)))


def is_converged(times_coarse, values_coarse, times_fine, values_fine, *, tol, atol=1e-9):
    """Whether two histories agree to within ``tol`` on the common grid."""
    return max_history_deviation(
        times_coarse, values_coarse, times_fine, values_fine, atol=atol
    ) <= tol


def saturated(bond_dims, *, window: int = 3, rel: float = 0.0):
    """Whether the bond dimension has stopped growing over the last ``window`` steps.

    Returns ``True`` when the maximum increase across the trailing window is at
    most ``rel`` times the current bond dimension (``rel = 0`` -> exactly flat).
    """
    bd = np.asarray(bond_dims)
    if bd.size <= window:
        return False
    tail = bd[-window - 1:]
    grow = np.max(np.diff(tail))
    return bool(grow <= rel * bd[-1])
