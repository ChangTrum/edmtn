"""Canonicalisation strategies (Layer 5 seam).

Left-orthonormalises sites ``0 .. n-2`` of an EDM-MPS in place, pushing the
triangular factor into the next site, so the truncation sweep that follows sees a
canonical environment.  Two strategies:

* :class:`HouseholderQR` -- the reference (orthogonal QR; immune to conditioning).
  Byte-for-byte identical to the historical ``mps_utils.left_canonicalize``.
* :class:`CholeskyQR` -- BLAS-3 / GEMM-dominated (Gram + Cholesky + triangular
  solve), GPU-friendly.  ``passes=2`` (CholeskyQR2) reaches machine-precision
  orthogonality; ``passes=1`` is faster but its orthogonality degrades as the bond
  conditioning grows.  Ill-conditioned bonds (where the shifted Cholesky cannot
  orthonormalise to tolerance) **fall back per-bond to Householder QR**, so the
  result is never numerically unsafe; the fallback count is recorded.

Why this exists: on GPU the orthogonalisation (not the compression) is the
bottleneck, and Householder QR parallelises poorly there while Cholesky-QR is pure
BLAS-3.  See ``docs/recommended-config.md`` and
``docs/incremental-update-research.md`` (sections 14, 16).

All strategies are backend-agnostic: they use the array's own namespace, so they
run on NumPy (CPU) or CuPy (GPU).  The default pipeline uses Householder QR, so
behaviour is unchanged unless a strategy is explicitly selected.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

# orthogonality below this is "good enough"; above it a CholeskyQR bond falls back
_FALLBACK_OK = 1e-6


def _xp(a):
    """Return the array module (``numpy`` or ``cupy``) backing ``a``."""
    if type(a).__module__.split(".")[0] == "cupy":
        import cupy  # noqa: PLC0415

        return cupy
    return np


def _ortho_err(Q, xp):
    k = Q.shape[1]
    return float(xp.max(xp.abs(Q.conj().T @ Q - xp.eye(k, dtype=Q.dtype))))


class _CanonFail(Exception):
    """Raised when a fast orthogonaliser fails; the caller falls back to QR."""


class CanonicalizationStrategy(ABC):
    """Left-orthonormalise sites ``0..n-2`` of ``mps`` in place; return ``mps``."""

    @abstractmethod
    def left_canonicalize(self, mps):
        ...


def _push_right(mps, p, R, xp):
    """Absorb factor ``R`` (k x chir) into site ``p+1`` along its left leg."""
    nxt = mps.tensors[p + 1]
    mps.tensors[p + 1] = xp.transpose(xp.tensordot(R, nxt, axes=([1], [1])), (1, 0, 2))


class HouseholderQR(CanonicalizationStrategy):
    """Householder QR sweep -- the conditioning-immune reference."""

    def left_canonicalize(self, mps):
        xp = _xp(mps.tensors[0])
        for p in range(mps.num_sites - 1):
            G = mps.tensors[p]
            dp, chil, chir = G.shape
            Q, R = xp.linalg.qr(G.reshape(dp * chil, chir))
            mps.tensors[p] = Q.reshape(dp, chil, Q.shape[1])
            _push_right(mps, p, R, xp)
        return mps


def _shifted_cholesky(G, xp):
    """Cholesky of (symmetrised) ``G`` with an escalating diagonal shift.

    Returns the upper factor ``R`` with ``R^H R = G + s I``.  Robust to both NumPy
    (raises on non-PD) and CuPy (may return NaNs): it checks finiteness and
    escalates the shift until the factor is finite, or fails over to QR."""
    G = 0.5 * (G + G.conj().T)
    if not bool(xp.all(xp.isfinite(G))):
        raise _CanonFail("non-finite Gram")
    nrm = float(xp.real(xp.trace(G))) or 1.0
    eps = float(np.finfo(G.dtype).eps)
    eye = xp.eye(G.shape[0], dtype=G.dtype)
    s = 0.0
    for _ in range(60):
        try:
            L = xp.linalg.cholesky(G + s * eye)
        except Exception:  # noqa: BLE001  (NumPy raises LinAlgError; be backend-agnostic)
            L = None
        if L is not None and bool(xp.all(xp.isfinite(L))):
            return L.conj().T
        s = max(s * 10.0, 11.0 * eps * nrm)
    raise _CanonFail("Cholesky did not become positive definite")


def _factor_cholqr(A, passes, xp):
    """(Shifted) Cholesky-QR.  ``Q = Q_old Rp^{-1}`` via ``solve(Rp^H, Q_old^H)^H``."""
    Q = A
    R = xp.eye(A.shape[1], dtype=A.dtype)
    for _ in range(passes):
        Rp = _shifted_cholesky(Q.conj().T @ Q, xp)
        Q = xp.linalg.solve(Rp.conj().T, Q.conj().T).conj().T
        R = Rp @ R
        if not bool(xp.all(xp.isfinite(Q))):
            raise _CanonFail("CholeskyQR blew up")
    if _ortho_err(Q, xp) > _FALLBACK_OK:
        raise _CanonFail("CholeskyQR not orthonormal (ill-conditioned)")
    return Q, R


class CholeskyQR(CanonicalizationStrategy):
    """BLAS-3 Cholesky-QR with ``passes`` reorthogonalisations.

    ``passes=2`` (default) is the robust "CholeskyQR2" (machine-precision
    orthogonality); ``passes=1`` is faster but marginal.  Wide/short bonds
    (``m < n``) and any bond the shifted Cholesky cannot orthonormalise fall back
    to Householder QR; ``last_fallback`` / ``last_ortho_err`` record diagnostics
    from the most recent sweep.
    """

    def __init__(self, passes: int = 2):
        if int(passes) < 1:
            raise ValueError(f"passes must be >= 1, got {passes}")
        self.passes = int(passes)
        self.last_fallback = 0
        self.last_ortho_err = 0.0

    def left_canonicalize(self, mps):
        xp = _xp(mps.tensors[0])
        n_fallback = 0
        err = 0.0
        for p in range(mps.num_sites - 1):
            G = mps.tensors[p]
            dp, chil, chir = G.shape
            A = G.reshape(dp * chil, chir)
            if A.shape[0] < A.shape[1]:                  # rank-deficient Gram -> QR
                Q, R = xp.linalg.qr(A)
                n_fallback += 1
            else:
                try:
                    Q, R = _factor_cholqr(A, self.passes, xp)
                except _CanonFail:
                    Q, R = xp.linalg.qr(A)
                    n_fallback += 1
            err = max(err, _ortho_err(Q, xp))
            mps.tensors[p] = Q.reshape(dp, chil, Q.shape[1])
            _push_right(mps, p, R, xp)
        self.last_fallback = n_fallback
        self.last_ortho_err = err
        return mps
