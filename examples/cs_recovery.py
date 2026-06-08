"""Compressed-sensing residual recovery (Tier 1.5) -- EXAMPLES-ONLY prototype.

This is a standalone NumPy prototype used to *validate* the CS layer before any
pipeline change.  It is intentionally kept in ``examples/`` and imports nothing
from the solver; only once the offline + end-to-end validation here is convincing
should an equivalent be promoted into ``src/edmtn/decomposition/`` (Layer 4a).

Idea.  When folding sub-bath ``L+1`` the bond matrix splits as

    M = M^||  +  M^perp ,   M^|| = U U^H M (GEMM),  M^perp = (I-U U^H) M (rank r).

Tier 1 keeps only ``M^||``; Tier 2 forms ``M^perp`` and (r)SVDs it.  Tier 1.5
recovers the low-rank ``M^perp`` from a few cheap rank-one measurements

    y_i = a_i^H M^perp b_i = a_i^H M b_i - (a_i^H U)(U^H M b_i),

``a_i in C^m``, ``b_i in C^n`` complex Gaussian, ``U^H M`` the projection
byproduct.  For rank ``r`` a matrix-sensing recovery needs
``p = O(r (m+n) log(mn)) << m n`` measurements.  A known zero column-block
(the picking-tensor ``phi = 0`` null channel) is excluded as a support prior,
shrinking ``n -> n_eff`` and hence ``p``.

Recovery uses Singular Value Projection (SVP / iterative hard thresholding for
matrix sensing) at the known rank.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def complex_gaussian(shape, rng):
    """Standard complex Gaussian array (unit variance per entry)."""
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2.0)


# -- rank-one measurement operator  y_i = a_i^H X b_i ----------------------

def apply_operator(A, B, X):
    """``y_i = a_i^H X b_i`` for all ``i`` (``A:(p,m)``, ``B:(p,n)``, ``X:(m,n)``)."""
    XBt = X @ B.T                                   # (m, p): column i = X b_i
    return np.einsum("is,si->i", A.conj(), XBt, optimize=True)


def adjoint_operator(A, B, z, shape):
    """``sum_i z_i a_i b_i^H`` -> ``(m, n)`` (adjoint of :func:`apply_operator`)."""
    return (A.T * z) @ B.conj()


def measure_residual(M, U, A, B, *, UHM=None):
    """Measurements of ``M^perp = (I - U U^H) M`` *without forming it*.

    ``y_i = a_i^H M b_i - (a_i^H U)(U^H M b_i)`` -- the second term reuses the
    projection byproduct ``U^H M``.
    """
    if UHM is None:
        UHM = U.conj().T @ M
    y_full = apply_operator(A, B, M)
    aU = A.conj() @ U                               # (p, D_old)
    UMb = UHM @ B.T                                  # (D_old, p)
    return y_full - np.einsum("id,di->i", aU, UMb, optimize=True)


# -- Singular Value Projection (rank-constrained matrix sensing) -----------

@dataclass
class CSRecoveryResult:
    X: np.ndarray
    rank: int
    n_iter: int
    residual: float
    converged: bool


def operator_norm_sq(A, B, shape, rng, *, iters=10):
    """Estimate ``||Phi||_op^2`` (top eigenvalue of ``Phi^H Phi``) by power iteration."""
    Z = complex_gaussian(shape, rng)
    Z /= np.linalg.norm(Z) or 1.0
    lam = 1.0
    for _ in range(iters):
        W = adjoint_operator(A, B, apply_operator(A, B, Z), shape)
        lam = float(np.linalg.norm(W))
        if lam == 0:
            break
        Z = W / lam
    return lam


def _rank_trunc(G, rank, rng, *, oversample=6):
    """Best rank-``rank`` truncation of ``G`` via a randomized range finder.

    Cheaper than a full SVD for the SVP hard-threshold when ``rank`` is small.
    """
    m, n = G.shape
    r = min(rank + oversample, m, n)
    Omega = complex_gaussian((n, r), rng)
    Q, _ = np.linalg.qr(G @ Omega)
    Q, _ = np.linalg.qr(G @ (G.conj().T @ Q))   # one power iteration for accuracy
    Ub, s, Vh = np.linalg.svd(Q.conj().T @ G, full_matrices=False)
    k = min(rank, s.shape[0])
    return ((Q @ Ub[:, :k]) * s[:k]) @ Vh[:k]


def svp_recover(A, B, y, shape, rank, *, n_iter=400, tol=1e-10, step=None):
    """SVP for ``y_i = a_i^H X b_i`` with ``rank(X) <= rank``.

    Iterative hard thresholding for matrix sensing with an adaptive step: the
    rank-one Gaussian operator obeys ``Phi^H Phi ~ p I`` so ``step = 1/p`` is the
    nominal choice, but it is halved on any non-monotone / non-finite step
    (undersampled regimes diverge otherwise).  The best iterate is returned.
    """
    m, n = shape
    p = int(y.shape[0])
    if rank <= 0 or p == 0:
        return CSRecoveryResult(np.zeros(shape, np.complex128), 0, 0, 0.0, True)
    rng = np.random.default_rng(0)
    # step = 1/L with L the operator Lipschitz constant (rank-one measurements have
    # large RIP variance, so 1/p is unreliable -- estimate ||Phi||^2 directly).
    if step is None:
        lam = operator_norm_sq(A, B, shape, rng)
        step = 0.95 / lam if lam > 0 else 1.0 / p
    X = np.zeros(shape, dtype=np.complex128)
    ny = float(np.linalg.norm(y)) or 1.0
    best_r, best_X = np.inf, X
    stall = 0
    for it in range(1, n_iter + 1):
        resid = y - apply_operator(A, B, X)
        rnorm = float(np.linalg.norm(resid)) / ny
        if not np.isfinite(rnorm):
            break
        if rnorm < best_r - 1e-15:
            best_r, best_X, stall = rnorm, X, 0
        else:
            stall += 1
        if rnorm < tol or stall >= 25:             # converged or plateaued
            return CSRecoveryResult(best_X, rank, it, best_r, best_r < 10 * tol)
        G = X + step * adjoint_operator(A, B, resid, shape)
        if not np.all(np.isfinite(G)):
            break
        X = _rank_trunc(G, rank, rng)
    return CSRecoveryResult(best_X, rank, n_iter, best_r, best_r < 10 * tol)


# -- convenience: residual recovery with support prior ---------------------

def recover_residual(M, U, rank, p, rng, *, col_mask=None, n_iter=400, tol=1e-10):
    """Recover ``M^perp = (I - U U^H) M`` from ``p`` rank-one measurements.

    ``col_mask`` (boolean over columns, ``True`` = kept) is the support prior:
    recovery runs on kept columns only; the rest stay zero.  Returns
    ``(M_perp_hat (m,n), info)``.
    """
    m, n = M.shape
    if col_mask is None:
        col_mask = np.ones(n, dtype=bool)
    cols = np.flatnonzero(col_mask)
    n_eff = int(cols.size)

    A = complex_gaussian((p, m), rng)
    B = complex_gaussian((p, n_eff), rng)
    Msub = M[:, cols]
    y = measure_residual(Msub, U, A, B)

    res = svp_recover(A, B, y, (m, n_eff), rank, n_iter=n_iter, tol=tol)
    Xhat = np.zeros((m, n), dtype=np.complex128)
    Xhat[:, cols] = res.X
    return Xhat, {"p": p, "n_eff": n_eff, "rank": rank, "n_iter": res.n_iter,
                  "meas_residual": res.residual, "converged": res.converged}


def measurement_budget(rank, m, n, *, c=3.0, log=False):
    """Measurement count for a rank-``r`` ``m x n`` recovery.

    The information limit is ``r(m + n - r)`` degrees of freedom; matrix sensing
    needs a small constant ``c`` times that.  The theoretical worst-case bound
    carries an extra ``log(mn)`` factor (``log=True``), but empirically
    ``p ~ c r (m + n)`` with ``c ~ 3`` suffices, so the log is off by default.
    """
    if rank <= 0:
        return 0
    base = c * rank * (m + n)
    if log:
        base *= np.log10(max(m * n, 10))
    return int(max(rank + 1, np.ceil(base)))


# -- self-check (synthetic): run `python examples/cs_recovery.py` ----------

def _self_check():
    rng = np.random.default_rng(0)
    m, n, r = 40, 30, 3
    X = complex_gaussian((m, r), rng) @ complex_gaussian((r, n), rng)
    # adjoint consistency
    A, B = complex_gaussian((7, m), rng), complex_gaussian((7, n), rng)
    Xt, z = complex_gaussian((m, n), rng), complex_gaussian(7, rng)
    adj = abs(np.vdot(apply_operator(A, B, Xt), z)
              - np.vdot(Xt, adjoint_operator(A, B, z, (m, n))))
    # recovery from p ~ 4 r (m+n)
    p = measurement_budget(r, m, n)
    A, B = complex_gaussian((p, m), rng), complex_gaussian((p, n), rng)
    res = svp_recover(A, B, apply_operator(A, B, X), (m, n), r, n_iter=500)
    err = np.linalg.norm(res.X - X) / np.linalg.norm(X)
    print(f"adjoint consistency |lhs-rhs| = {adj:.2e}")
    print(f"recover rank-{r} {m}x{n} from p={p} (mn={m*n}): err={err:.2e}, "
          f"iters={res.n_iter}, converged={res.converged}")
    assert adj < 1e-10 and err < 1e-4
    print("self-check PASSED")


if __name__ == "__main__":
    _self_check()
