"""Canonicalisation alternatives for the EDM fold -- attacking the new bottleneck.

EXAMPLES-ONLY; the solver pipeline (src/) is untouched and used verbatim as the
baseline (``EDMSolver`` -> the paper's Fig. 6 algorithm).

After section 13 (uniform single-pass rSVD), compression is cheap and the
wall-clock bottleneck is the per-fold **left-canonicalisation** -- a Householder
QR sweep over the *uncompressed* folded bonds (~D_a*D_old wide). This script
tests three replacements for that QR sweep, in the user's priority order:

  1. ``none``    -- skip canonicalisation entirely; run the rSVD truncation sweep
                    directly on the non-canonical folded MPS.  (No orthogonal
                    gauge -> local SVD no longer sees the true Schmidt spectrum;
                    accuracy must be checked.)
  2. ``cholqr`` / ``cholqr2`` -- Cholesky-QR (Gram + Cholesky + triangular solve,
                    all GEMM/BLAS-3) in place of Householder QR.  ``cholqr2``
                    reorthogonalises (two passes) for conditioning.  Wide/short
                    boundary bonds (m < n) fall back to Householder QR.
  3. ``nspolar`` -- Newton-Schulz polar factor (inverse-free, GEMM-only) as the
                    orthonormaliser.

Each canonicaliser is crossed with both truncation modes (single-pass rSVD
``single`` and cold rSVD ``cold``; ``svd`` full-SVD as an exactness anchor), and
measured against the pipeline to the same bar as section 13: accuracy must sit
below the cutoff and be seed-stable, so the result is trustworthy with no
reference run.  We isolate **canonicalisation wall-clock** (the quantity we are
trying to cut) from truncation wall-clock, and report the canonicaliser's
**orthogonality error** ``max_p ||Q_p^H Q_p - I||`` (flags CholQR instability /
NS non-convergence directly).

Pure CPU / NumPy.  Defaults sized for a 16 GB laptop.

Usage
-----
    python examples/canon_alternatives_e2e.py                       # all canon x {single,cold}
    python examples/canon_alternatives_e2e.py --canon none,cholqr2 --trunc single
    python examples/canon_alternatives_e2e.py --cutoff 1e-8 --seeds 0,1,2
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from adaptive_tiers_e2e import run_baseline  # noqa: E402
from edm_incremental import fold_uncompressed, make_context  # noqa: E402
from uniform_rsvd_e2e import _rsvd_truncate, _svd_truncate  # noqa: E402

from edmtn.evolution import mps_utils  # noqa: E402
from edmtn.observables.extractor import _scalar, _vec_identity  # noqa: E402
from edmtn.models import GaudinModel  # noqa: E402

_DIR_DATA = Path(__file__).resolve().parent / "data"


def coupling_polarization_tolerant(mps, eps, *, channel=3, order=2):
    """Order-2 coupling polarization that does NOT raise on a complex result.

    Mirrors ``ObservableExtractor.coupling_polarization_history`` but returns
    ``(values_real, imag_rel)`` where ``imag_rel = max|Im| / (max|Re|+eps)`` -- so a
    canonicalisation that corrupts the state (non-Hermitian -> complex observable)
    is *quantified* rather than crashing the sweep.
    """
    n = mps.num_sites
    sel = 2 * channel - 1
    zero_mats = [t[0] for t in mps.tensors]
    sel_mats = [t[sel] for t in mps.tensors]
    left = [None] * n
    left[0] = _vec_identity(mps.d, mps.tensors[0])
    for p in range(1, n):
        left[p] = left[p - 1] @ zero_mats[p - 1]
    right = [None] * n
    right[n - 1] = mps.rho0_vec
    for p in range(n - 2, -1, -1):
        right[p] = zero_mats[p + 1] @ right[p + 1]
    n_phys = n // 2
    coeff = (1.0 + 1.0j) / eps
    values = np.empty(n_phys, dtype=np.complex128)
    for p in range(n):
        g = n - p
        if g % 2 == 1:
            m = (g + 1) // 2
            values[m - 1] = coeff * complex(_scalar(left[p] @ (sel_mats[p] @ right[p])))
    imag_rel = float(np.max(np.abs(values.imag)) / (np.max(np.abs(values.real)) + 1e-12))
    return values.real, imag_rel


# --------------------------------------------------------------------------
# canonicalisers -- each left-orthogonalises sites 0..n-2 in place, pushing the
# triangular/Hermitian factor into the next site (same contract as
# mps_utils.left_canonicalize), and returns the max orthogonality error.
# --------------------------------------------------------------------------

def _push_right(mps, p, R):
    """Absorb factor ``R`` (k x chir) into site ``p+1`` along its left leg."""
    nxt = mps.tensors[p + 1]                       # (dp, chil=chir, chir2)
    mps.tensors[p + 1] = np.transpose(np.tensordot(R, nxt, axes=([1], [1])), (1, 0, 2))


def _set_left(mps, p, Q, dp, chil):
    mps.tensors[p] = Q.reshape(dp, chil, Q.shape[1])


def _ortho_err(Q):
    k = Q.shape[1]
    return float(np.max(np.abs(Q.conj().T @ Q - np.eye(k, dtype=Q.dtype))))


# Each canonicaliser returns (Q, R) with Q R = A, Q left-orthonormal, OR raises
# ``_CanonFail`` if the fast method blew up / failed to orthonormalise -- the
# caller then falls back to Householder QR and counts it.

class _CanonFail(Exception):
    pass


_FALLBACK_OK = 1e-6        # accept a factorisation only if ortho err is below this


def canon_qr(mps):
    """Householder QR sweep -- identical to mps_utils.left_canonicalize (reference)."""
    return _sweep(mps, lambda A: np.linalg.qr(A))


def canon_none(mps):
    """No canonicalisation -- the rSVD truncation sweep runs on the raw fold."""
    return mps, float("nan"), 0


def _shifted_cholesky(G):
    """Cholesky of (symmetrised) ``G`` with an escalating diagonal shift; uses
    ``trace(G)`` (= ||A||_F^2, robust, no SVD) as the shift scale.  Returns the
    upper factor ``R`` with ``R^H R = G + s I``, or raises if G is non-finite."""
    G = 0.5 * (G + G.conj().T)
    if not np.isfinite(G).all():
        raise _CanonFail("non-finite Gram")
    nrm = float(np.real(np.trace(G))) or 1.0
    eps = np.finfo(G.real.dtype).eps
    s = 0.0
    for _ in range(60):
        try:
            return np.linalg.cholesky(G + s * np.eye(G.shape[0], dtype=G.dtype)).conj().T
        except np.linalg.LinAlgError:
            s = max(s * 10.0, 11.0 * eps * nrm)
    raise _CanonFail("cholesky did not become PD")


def _factor_cholqr(A, passes):
    """CholeskyQR with ``passes`` passes.  ``Q Rp = Q_old`` via ``Q = Q_old Rp^{-1}``
    where ``Rp`` is the upper Cholesky factor of the Gram (``Rp^H Rp = Q_old^H Q_old``).
    """
    Q = A
    R = np.eye(A.shape[1], dtype=A.dtype)
    for _ in range(passes):
        Rp = _shifted_cholesky(Q.conj().T @ Q)
        Q = np.linalg.solve(Rp.conj().T, Q.conj().T).conj().T   # Q <- Q Rp^{-1}
        R = Rp @ R
        if not np.isfinite(Q).all():
            raise _CanonFail("cholqr blew up")
    if _ortho_err(Q) > _FALLBACK_OK:
        raise _CanonFail("cholqr not orthonormal")
    return Q, R


def _spec_norm(B, iters=8):
    """Largest eigenvalue of Hermitian PSD ``B`` (= sigma_max(A)^2) by power iteration."""
    v = np.ones(B.shape[0], dtype=B.dtype)
    lam = 0.0
    for _ in range(iters):
        w = B @ v
        nw = np.linalg.norm(w)
        if nw == 0:
            return 0.0
        v = w / nw
        lam = float(np.real(v.conj() @ (B @ v)))
    return lam


def _factor_nspolar(A, max_iter, tol):
    """Newton-Schulz polar factor.  Scale by a *tight* sigma_max estimate (power
    iteration on the Gram) so all singular values of X0 are in (0, 1] -> fast
    convergence; loose 1-/inf-norm scaling would stall the small singular values."""
    G0 = A.conj().T @ A
    smax2 = _spec_norm(G0)
    b = float(np.sqrt(smax2)) * 1.01 or 1.0          # >= sigma_max(A), tight
    X = A / b
    n = A.shape[1]
    I = np.eye(n, dtype=A.dtype)
    for _ in range(max_iter):
        G = X.conj().T @ X
        if not np.isfinite(G).all():
            raise _CanonFail("nspolar blew up")
        X = 1.5 * X - 0.5 * (X @ G)
        if np.max(np.abs(G - I)) < tol:
            break
    if _ortho_err(X) > _FALLBACK_OK:
        raise _CanonFail("nspolar not converged")
    return X, X.conj().T @ A


def _sweep(mps, factor):
    """Left-orthonormalise sites 0..n-2 with ``factor(A)->(Q,R)``; Householder QR
    fallback (counted) for wide bonds and ``_CanonFail`` cases."""
    err = 0.0
    n_fallback = 0
    for p in range(mps.num_sites - 1):
        dp, chil, chir = mps.tensors[p].shape
        A = mps.tensors[p].reshape(dp * chil, chir)
        if A.shape[0] < A.shape[1]:                           # wide/short -> Householder
            Q, R = np.linalg.qr(A)
            n_fallback += 1
        else:
            try:
                Q, R = factor(A)
            except _CanonFail:
                Q, R = np.linalg.qr(A)
                n_fallback += 1
        err = max(err, _ortho_err(Q))
        _set_left(mps, p, Q, dp, chil)
        _push_right(mps, p, R)
    return mps, err, n_fallback


def canon_cholqr(mps, passes=1):
    """(Shifted) Cholesky-QR with ``passes`` reorthogonalisations."""
    return _sweep(mps, lambda A: _factor_cholqr(A, passes))


def canon_nspolar(mps, max_iter=24, tol=1e-13):
    """Newton-Schulz polar factor as the orthonormaliser (inverse-free, GEMM-only)."""
    return _sweep(mps, lambda A: _factor_nspolar(A, max_iter, tol))


_CANON = {"qr": canon_qr, "none": canon_none, "cholqr": lambda m: canon_cholqr(m, 1),
          "cholqr2": lambda m: canon_cholqr(m, 2), "nspolar": canon_nspolar}


# --------------------------------------------------------------------------
# truncation sweep (right-to-left), pluggable rSVD mode
# --------------------------------------------------------------------------

def trunc_sweep(B, ctx, trunc_mode, rng, k_memory):
    xi, d2, mb = ctx["cutoff"], ctx["ref_index"], ctx["max_bond"]
    for p in range(B.num_sites - 1, 0, -1):
        G = B.tensors[p]
        dp, chil, chir = G.shape
        M = G.transpose(1, 0, 2).reshape(chil, dp * chir)
        if trunc_mode == "svd":
            US, Vh, k, _ = _svd_truncate(M, xi=xi, ref_index=d2, max_bond=mb)
        else:
            n_iter = 2 if trunc_mode == "cold" else 0
            guess = k_memory.get(p, 16) + 16
            US, Vh, k, _ = _rsvd_truncate(M, xi=xi, ref_index=d2, max_bond=mb,
                                          n_iter=n_iter, rng=rng, guess=guess)
        k_memory[p] = k
        B.tensors[p] = Vh.reshape(k, dp, chir).transpose(1, 0, 2)
        B.tensors[p - 1] = np.tensordot(B.tensors[p - 1], US, axes=([2], [0]))
    return B


# --------------------------------------------------------------------------
# end-to-end fold with a given (canon, trunc) pair
# --------------------------------------------------------------------------

def run_variant(ctx, K, canon_mode, trunc_mode, seed=0):
    eps, order = ctx["eps"], ctx["order"]
    mps = ctx["mps0"]
    rng = np.random.default_rng(seed)
    canon_fn = _CANON[canon_mode]
    k_memory: dict[int, int] = {}
    t_canon = t_trunc = 0.0
    ortho = 0.0
    n_fallback = 0
    traj = []
    t0 = time.perf_counter()
    for k in range(K):
        unc = fold_uncompressed(ctx, mps, k)
        if unc.num_sites <= 1:
            mps = unc
            traj.append(mps.max_bond)
            continue
        B = unc.copy()
        tc = time.perf_counter()
        B, oe, nfb = canon_fn(B)
        t_canon += time.perf_counter() - tc
        n_fallback += nfb
        if oe == oe:                       # not NaN
            ortho = max(ortho, oe)
        tt = time.perf_counter()
        B = trunc_sweep(B, ctx, trunc_mode, rng, k_memory)
        t_trunc += time.perf_counter() - tt
        mps = B
        traj.append(mps.max_bond)
    wall = time.perf_counter() - t0
    pol, imag_rel = coupling_polarization_tolerant(mps, eps, channel=3, order=order)
    return dict(pol=pol, wall=wall, t_canon=t_canon, t_trunc=t_trunc,
                ortho=ortho, dmax=mps.max_bond, traj=traj, imag_rel=imag_rel,
                n_fallback=n_fallback)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--g", type=float, default=1.0)
    ap.add_argument("--K", type=int, default=24)
    ap.add_argument("--T", type=float, default=3.0)
    ap.add_argument("--eps", type=float, default=0.2)
    ap.add_argument("--cutoff", type=float, default=1e-6)
    ap.add_argument("--max-bond", type=int, default=400)
    ap.add_argument("--order", type=int, default=2, choices=(1, 2))
    ap.add_argument("--canon", default="qr,none,cholqr,cholqr2,nspolar")
    ap.add_argument("--trunc", default="single,cold")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--name", default="canon_alt")
    args = ap.parse_args()

    model = GaudinModel(g=args.g, K=args.K)
    ctx = make_context(model, T=args.T, eps=args.eps, order=args.order,
                       cutoff=args.cutoff, max_bond=args.max_bond)
    print(f"Canon alternatives (Gaudin): K={args.K}, T={args.T} g^-1, eps={args.eps} g^-1, "
          f"order={args.order}, xi={args.cutoff:g}")

    print("  baseline (unmodified pipeline)...")
    sz_base, wall_base = run_baseline(model, T=args.T, eps=args.eps, order=args.order,
                                      cutoff=args.cutoff, max_bond=args.max_bond)
    print(f"    baseline wall = {wall_base:.1f}s")

    canons = [c.strip() for c in args.canon.split(",") if c.strip()]
    truncs = [t.strip() for t in args.trunc.split(",") if t.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    print(f"\n  {'canon':>8} {'trunc':>7} | {'wall':>6} {'spd':>5} | "
          f"{'canon_s':>7} {'trunc_s':>7} | {'max|dSz|':>9} {'imag/re':>8} {'Dmax':>5} "
          f"{'ortho':>9} {'QRfb':>5}")
    print("  " + "-" * 96)
    results = {}
    for canon_mode, trunc_mode in product(canons, truncs):
        errs, walls, tcs, tts, dmaxes, orthos, imags, fbs = [], [], [], [], [], [], [], []
        traj_last = None
        for seed in seeds:
            r = run_variant(ctx, args.K, canon_mode, trunc_mode, seed=seed)
            n = min(len(r["pol"]), len(sz_base))
            errs.append(float(np.max(np.abs(np.asarray(r["pol"][:n]) - np.asarray(sz_base[:n])))))
            walls.append(r["wall"]); tcs.append(r["t_canon"]); tts.append(r["t_trunc"])
            dmaxes.append(r["dmax"]); orthos.append(r["ortho"]); imags.append(r["imag_rel"])
            fbs.append(r["n_fallback"]); traj_last = r["traj"]
        err = max(errs); wall = float(np.mean(walls))
        tc = float(np.mean(tcs)); tt = float(np.mean(tts))
        dmax = max(dmaxes); ortho = max(orthos); imag_rel = max(imags); fb = max(fbs)
        results[(canon_mode, trunc_mode)] = dict(
            err=err, wall=wall, t_canon=tc, t_trunc=tt, dmax=dmax, ortho=ortho,
            imag_rel=imag_rel, n_fallback=fb, traj=traj_last)
        if imag_rel > 1e-3:
            flag = "  <-- UNPHYSICAL (complex obs)"
        elif err > 1e-4:
            flag = "  <-- INACCURATE"
        else:
            flag = ""
        print(f"  {canon_mode:>8} {trunc_mode:>7} | {wall:6.1f} {wall_base/wall:4.2f}x | "
              f"{tc:7.2f} {tt:7.2f} | {err:9.2e} {imag_rel:8.1e} {dmax:5d} "
              f"{ortho:9.1e} {fb:5d}{flag}")

    # canon-cost focus: relative to the qr reference at the same trunc mode
    print(f"\n  canonicalisation wall-clock vs the QR reference (the bottleneck we target):")
    for trunc_mode in truncs:
        ref = results.get(("qr", trunc_mode))
        if ref is None:
            continue
        for canon_mode in canons:
            r = results.get((canon_mode, trunc_mode))
            if r is None:
                continue
            spd = ref["t_canon"] / r["t_canon"] if r["t_canon"] > 0 else float("inf")
            tag = "(skips canon)" if canon_mode == "none" else f"{spd:.2f}x vs QR canon"
            print(f"    trunc={trunc_mode:>6}  {canon_mode:>8}: canon {r['t_canon']:6.2f}s  {tag}")

    _DIR_DATA.mkdir(parents=True, exist_ok=True)
    npz = _DIR_DATA / f"{args.name}.npz"
    save = {"K": args.K, "xi": args.cutoff, "wall_base": wall_base,
            "sz_base": np.asarray(sz_base)}
    for (cm, tm), r in results.items():
        key = f"{cm}_{tm}"
        save[f"err_{key}"] = r["err"]; save[f"wall_{key}"] = r["wall"]
        save[f"tcanon_{key}"] = r["t_canon"]; save[f"ttrunc_{key}"] = r["t_trunc"]
        save[f"dmax_{key}"] = r["dmax"]; save[f"ortho_{key}"] = r["ortho"]
        save[f"imag_{key}"] = r["imag_rel"]; save[f"qrfb_{key}"] = r["n_fallback"]
    np.savez(npz, **save)
    print(f"\nsaved {npz}")


if __name__ == "__main__":
    main()
