"""Standard truncated SVD strategy (Layer 4a, Strategy A).

The baseline compression used throughout the EDM paper's numerical results: a
full (economy) SVD followed by rank selection from a cutoff rule and an optional
hard bond cap.  Raw factorisation is delegated to a Layer-0 decomposition
backend, auto-selected from the array type (NumPy on the CPU, CuPy on the GPU)
unless an explicit backend is provided.
"""

from __future__ import annotations

import numpy as np

from .base import DecompositionResult, DecompositionStrategy, _to_host, truncation_rank


class StandardSVD(DecompositionStrategy):
    """Truncated SVD via a Layer-0 decomposition backend.

    Parameters
    ----------
    backend : DecompositionBackend, optional
        Explicit factorisation backend.  If ``None``, a backend is selected from
        the input array type on each call (``'cupy'`` for CuPy arrays, otherwise
        ``'numpy'``) and cached.
    """

    def __init__(self, backend=None):
        self._backend = backend
        self._cache: dict[str, object] = {}

    def _get_backend(self, matrix):
        if self._backend is not None:
            return self._backend
        name = "cupy" if type(matrix).__module__.split(".")[0] == "cupy" else "numpy"
        if name not in self._cache:
            from ..backend import create

            self._cache[name] = create(name)
        return self._cache[name]

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
        U, s, Vh = backend.svd(matrix, full_matrices=False)

        s_host = _to_host(s).real.astype(np.float64)
        n_singular = int(s_host.shape[0])
        keep = truncation_rank(
            s_host,
            max_bond=max_bond,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            ref_index=ref_index,
        )

        discarded = s_host[keep:]
        discarded_weight = float(np.sum(discarded**2))
        info = {
            "bond": keep,
            "error": float(np.sqrt(discarded_weight)),
            "discarded_weight": discarded_weight,
            "n_singular": n_singular,
            "cutoff_mode": cutoff_mode,
            "max_bond_hit": max_bond is not None and keep == max_bond and keep < n_singular,
        }

        U_k = U[:, :keep]
        Vh_k = Vh[:keep, :]
        s_k = s[:keep]

        if renorm and discarded_weight > 0.0:
            kept_weight = float(np.sum(s_host[:keep] ** 2))
            if kept_weight > 0.0:
                s_k = s_k * np.sqrt((kept_weight + discarded_weight) / kept_weight)

        left, right = self._absorb(U_k, s_k, Vh_k, absorb)
        return DecompositionResult(left=left, s=s_k, right=right, info=info)

    @staticmethod
    def _absorb(U, s, Vh, absorb):
        """Distribute singular values into the returned factors."""
        if absorb is None:
            return U, Vh
        if absorb == "left":
            return U * s, Vh
        if absorb == "right":
            return U, s[:, None] * Vh
        # both: split as sqrt(s)
        root = s ** 0.5
        return U * root, root[:, None] * Vh
