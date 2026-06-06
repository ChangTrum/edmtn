"""Decomposition / compression strategy interface (Layer 4a).

A decomposition strategy performs a truncated factorisation of a matrix (the
reshaped tensor produced during an MPS sweep): it computes an SVD, selects how
many singular values to keep according to a cutoff rule and an optional hard
bond cap, and returns the (optionally singular-value-absorbed) factors plus
diagnostics.

Strategies are backend-agnostic: they operate on NumPy or CuPy 2-D arrays and
keep the factors on the same device.  The rank-selection arithmetic is performed
on the host (the singular-value vector is small and the choice must be made in
double precision regardless of the contraction precision).

Supported ``cutoff_mode`` values:

* ``'abs'``    keep ``s_i > cutoff``
* ``'rel'``    keep ``s_i > cutoff * s_0`` (relative to the largest)
* ``'rel_ref'``keep ``s_i > cutoff * s_ref`` where ``s_ref = s[ref_index]``
               (0-based).  With ``ref_index = d**2`` this is the EDM paper's
               rule ``discard  s_i / s_{d^2+1} <= cutoff``.
* ``'sum2'``   keep the largest values so the discarded sum of squares
               ``<= cutoff**2``
* ``'rsum2'``  as ``'sum2'`` but relative to the total sum of squares
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

_CUTOFF_MODES = ("abs", "rel", "rel_ref", "sum2", "rsum2")


@dataclass
class DecompositionResult:
    """Result of a truncated factorisation.

    Attributes
    ----------
    left, right : ndarray
        The factors, with singular values absorbed according to ``absorb``
        (see :meth:`DecompositionStrategy.compress`).  With ``absorb=None`` they
        are ``U`` and ``Vh``; otherwise ``left @ right`` reconstructs the
        (truncated) matrix.
    s : ndarray
        The kept singular values (real, descending), on the input device.
    info : dict
        Diagnostics: ``bond`` (kept rank), ``error`` (Frobenius norm of the
        discarded singular values), ``discarded_weight`` (their sum of squares),
        ``n_singular`` (rank before truncation), ``cutoff_mode``,
        ``max_bond_hit`` (whether the hard cap was the binding constraint).
    """

    left: object
    s: object
    right: object
    info: dict = field(default_factory=dict)

    @property
    def bond(self) -> int:
        return self.info["bond"]

    @property
    def error(self) -> float:
        return self.info["error"]


def _to_host(a) -> np.ndarray:
    """Return a NumPy copy of ``a`` regardless of whether it lives on the GPU."""
    if type(a).__module__.split(".")[0] == "cupy":
        import cupy as cp

        return cp.asnumpy(a)
    return np.asarray(a)


def truncation_rank(
    s_host: np.ndarray,
    *,
    max_bond: int | None = None,
    cutoff: float = 0.0,
    cutoff_mode: str = "rel",
    ref_index: int | None = None,
) -> int:
    """Number of singular values to keep, given the (descending) host vector ``s_host``.

    Always keeps at least one value (unless the input is empty).
    """
    n = int(s_host.shape[0])
    if n == 0:
        return 0
    keep = n
    if cutoff and cutoff > 0.0:
        if cutoff_mode == "abs":
            keep = int(np.count_nonzero(s_host > cutoff))
        elif cutoff_mode == "rel":
            keep = int(np.count_nonzero(s_host > cutoff * s_host[0]))
        elif cutoff_mode == "rel_ref":
            ri = 0 if ref_index is None else min(int(ref_index), n - 1)
            keep = int(np.count_nonzero(s_host > cutoff * s_host[ri]))
        elif cutoff_mode in ("sum2", "rsum2"):
            sq = s_host.astype(np.float64) ** 2
            tail = np.concatenate([np.cumsum(sq[::-1])[::-1], [0.0]])  # tail[k]=sum_{j>=k}
            tol = cutoff**2
            if cutoff_mode == "rsum2":
                tol *= float(sq.sum())
            keep = int(np.nonzero(tail <= tol)[0][0])
        else:
            raise ValueError(f"unknown cutoff_mode {cutoff_mode!r}; choose from {_CUTOFF_MODES}")
    keep = max(keep, 1)
    if max_bond is not None:
        keep = min(keep, int(max_bond))
    return keep


class DecompositionStrategy(ABC):
    """Base class for truncated-factorisation strategies."""

    @abstractmethod
    def compress(
        self,
        matrix,
        *,
        max_bond: int | None = None,
        cutoff: float = 0.0,
        cutoff_mode: str = "rel",
        ref_index: int | None = None,
        absorb: str | None = "both",
        renorm: bool = False,
        **params,
    ) -> DecompositionResult:
        """Truncated factorisation of ``matrix``.

        Parameters
        ----------
        matrix : ndarray
            2-D array (NumPy or CuPy).
        max_bond : int, optional
            Hard cap on the kept rank.
        cutoff : float
            Truncation precision; ``0`` disables cutoff-based truncation.
        cutoff_mode : str
            One of the modes documented in the module docstring.
        ref_index : int, optional
            Reference index for ``'rel_ref'`` (0-based).
        absorb : {'both', 'left', 'right', None}
            How to absorb the singular values into the returned factors.
        renorm : bool
            Rescale the kept singular values to preserve the Frobenius norm.
        """
