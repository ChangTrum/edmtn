"""Randomized SVD truncation strategy (Layer 4a, Strategy B).

A GEMM-dominated, GPU-friendly alternative to the full :class:`StandardSVD`.  The
range of the matrix is captured with a random sketch (Halko-Martinsson-Tropp); the
small projected matrix is then factorised exactly.  Two operating points:

* ``n_iter = 0`` -- **single-pass** rSVD: the validated "balanced" compression.
  Accuracy below the cutoff and seed-stable, fastest; at very tight cutoff it may
  over-retain the bond by a few percent.
* ``n_iter = 2`` -- **cold** rSVD (power iterations): reproduces the exact
  full-SVD bond dimensions at ~1e-12 accuracy; the "robust" compression.

The target rank is found adaptively with a **spectral resolution guard**: the
sketch is grown (x2) until the cutoff rule selects strictly fewer singular values
than were computed -- i.e. the truncation threshold falls *inside* the resolved
spectrum, so no kept direction can hide in an un-computed tail.  This is what makes
the strategy reliable with no reference run (see ``docs/recommended-config.md`` and
``docs/incremental-update-research.md`` section 13).

Raw factorisations (QR of the sketch, SVD of the small projected matrix) are
delegated to the same Layer-0 decomposition backend as :class:`StandardSVD`, so the
strategy runs on NumPy (CPU) or CuPy (GPU) transparently; the random sketch and the
GEMMs use the array's own namespace.
"""

from __future__ import annotations

import numpy as np

from .base import DecompositionResult, DecompositionStrategy, _to_host, truncation_rank
from .standard_svd import StandardSVD


def _xp(a):
    """Return the array module (``numpy`` or ``cupy``) backing ``a``."""
    if type(a).__module__.split(".")[0] == "cupy":
        import cupy  # noqa: PLC0415

        return cupy
    return np


class RandomizedSVD(DecompositionStrategy):
    """Randomized truncated SVD.

    Parameters
    ----------
    n_iter : int
        Number of power iterations.  ``0`` = single-pass (balanced), ``2`` = cold
        (robust).  Higher values sharpen the small singular values at extra cost.
    n_oversamples : int
        Sketch oversampling (extra random columns beyond the target rank) for
        subspace accuracy.
    seed : int
        Seed for the random sketch; the per-backend generator is created once and
        advanced across calls, so results are reproducible for a fixed call order.
    backend : DecompositionBackend, optional
        Explicit factorisation backend.  If ``None``, a backend is selected from
        the input array type on each call (``'cupy'`` for CuPy arrays, otherwise
        ``'numpy'``) and cached.
    """

    def __init__(self, n_iter=0, n_oversamples=10, seed=0, backend=None):
        if int(n_iter) < 0:
            raise ValueError(f"n_iter must be >= 0, got {n_iter}")
        if int(n_oversamples) < 0:
            raise ValueError(f"n_oversamples must be >= 0, got {n_oversamples}")
        self.n_iter = int(n_iter)
        self.n_oversamples = int(n_oversamples)
        self._seed = int(seed)
        self._backend = backend
        self._cache: dict[str, object] = {}
        self._rng: dict[str, object] = {}
        self._rank_hint: dict[int, int] = {}

    # -- backend / rng helpers ---------------------------------------------

    def _get_backend(self, matrix):
        if self._backend is not None:
            return self._backend
        name = "cupy" if type(matrix).__module__.split(".")[0] == "cupy" else "numpy"
        if name not in self._cache:
            from ..backend import create  # noqa: PLC0415

            self._cache[name] = create(name)
        return self._cache[name]

    def _get_rng(self, xp):
        name = xp.__name__
        if name not in self._rng:
            self._rng[name] = xp.random.default_rng(self._seed)
        return self._rng[name]

    def _gaussian(self, shape, dtype, xp, rng):
        """Random sketch matching the matrix dtype (complex if the matrix is)."""
        g = rng.standard_normal(shape)
        if np.issubdtype(dtype, np.complexfloating):
            g = g + 1j * rng.standard_normal(shape)
        return g.astype(dtype, copy=False)

    # -- core sketch -------------------------------------------------------

    def _sketch_svd(self, matrix, rank, backend, xp, rng):
        """Top-``rank`` randomized SVD ``(U, s, Vh)`` of ``matrix``."""
        m, n = matrix.shape
        r = min(int(rank) + self.n_oversamples, m, n)
        if r <= 0:
            empty_u = matrix[:, :0]
            return empty_u, xp.zeros(0, dtype=_to_host(matrix).real.dtype), matrix[:0, :]
        omega = self._gaussian((n, r), matrix.dtype, xp, rng)
        Q, _ = backend.qr(matrix @ omega)
        for _ in range(self.n_iter):
            Q, _ = backend.qr(matrix.conj().T @ Q)
            Q, _ = backend.qr(matrix @ Q)
        B = Q.conj().T @ matrix
        Ub, s, Vh = backend.svd(B, full_matrices=False)
        U = Q @ Ub
        keep = min(int(rank), s.shape[0])
        return U[:, :keep], s[:keep], Vh[:keep, :]

    # -- strategy interface ------------------------------------------------

    def compress(
        self,
        matrix,
        *,
        max_bond=None,
        cutoff=0.0,
        cutoff_mode="rel",
        ref_index=None,
        absorb="both",
        renorm=False,
        **params,
    ) -> DecompositionResult:
        if matrix.ndim != 2:
            raise ValueError(f"compress expects a 2-D matrix, got shape {matrix.shape}")
        if absorb not in (None, "left", "right", "both"):
            raise ValueError(f"invalid absorb={absorb!r}")

        backend = self._get_backend(matrix)
        xp = _xp(matrix)
        rng = self._get_rng(xp)
        full = int(min(matrix.shape))

        # adaptive rank with the spectral resolution guard
        R = int(min(max(self._rank_hint.get(matrix.shape[1], 0) + 16, 8), full)) if full else 0
        U = s = Vh = None
        s_host = np.zeros(0)
        keep = 0
        while True:
            U, s, Vh = self._sketch_svd(matrix, R, backend, xp, rng)
            s_host = _to_host(s).real.astype(np.float64)
            keep = truncation_rank(
                s_host, max_bond=max_bond, cutoff=cutoff,
                cutoff_mode=cutoff_mode, ref_index=ref_index,
            )
            # resolved once the cutoff bites inside the computed spectrum (or full rank)
            if keep < s_host.shape[0] or R >= full:
                break
            R = min(2 * R, full)
        self._rank_hint[matrix.shape[1]] = keep

        n_singular = int(s_host.shape[0])
        discarded = s_host[keep:]                       # resolved tail below the cutoff
        discarded_weight = float(np.sum(discarded ** 2))
        info = {
            "bond": keep,
            "error": float(np.sqrt(discarded_weight)),
            "discarded_weight": discarded_weight,
            "n_singular": n_singular,
            "cutoff_mode": cutoff_mode,
            "max_bond_hit": max_bond is not None and keep == max_bond and keep < n_singular,
            "n_iter": self.n_iter,
        }

        U_k = U[:, :keep]
        Vh_k = Vh[:keep, :]
        s_k = s[:keep]

        if renorm and discarded_weight > 0.0:
            kept_weight = float(np.sum(s_host[:keep] ** 2))
            if kept_weight > 0.0:
                s_k = s_k * np.sqrt((kept_weight + discarded_weight) / kept_weight)

        left, right = StandardSVD._absorb(U_k, s_k, Vh_k, absorb)
        return DecompositionResult(left=left, s=s_k, right=right, info=info)
