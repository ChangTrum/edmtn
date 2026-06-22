"""Subspace-rotation tracking (Procrustes) as a cheaper alternative to rSVD in
the transition zone (L ~ 16-23).  EXAMPLES-ONLY study; the pipeline is untouched.

Motivation.  In the transition zone the bond dimension has saturated (no new
*strong* directions: n_new(sqrt(xi)) = 0), yet there is still a "weak rotation"
of the existing subspace (n_new(xi) > 0 at the ~xi level).  Running rSVD there --
a method designed to *find new low-rank directions to add* -- is a poor fit when
the change is a rotation of a fixed-dimension subspace.

Claim under test.  The new left subspace differs from the old by a rotation:
``U_{L+1} ~ U_L R`` for a D x D unitary R.  After the projection ``B = U_L^H M``
(the GEMM byproduct), a small SVD of the *reduced* D x n matrix B yields

    B = R Lambda V^H ,    U_new = U_L R ,

i.e. the rotation R and the full Schmidt spectrum Lambda -- on a D-sized matrix,
not the m x n one.  No rSVD, no random projection / power iteration.

What this script measures, per transition-zone bond:

* premise: principal angles between V^(L) and V^(L+1); the optimal Procrustes
  rotation R aligning them and its residual ``||U_true - U_L R|| = sqrt(sum
  sin^2 theta)`` (the chordal distance).
* rotation tracking (in-span SVD of B): the recovered spectrum vs the true one,
  the recovered rotation, the reconstruction error (= eta, the out-of-span
  residual it cannot represent), and wall-clock.
* baselines: full SVD of M, and rSVD of M^perp (the current Tier-2), with
  reconstruction error and wall-clock.

Honest expectation (see the printed verdict): rotation tracking recovers the
Schmidt spectrum and the bond rotation near-exactly and cheaply (the decomposition
shrinks from m x n to D x n / D x D), but its *reconstruction* error is eta -- it
drops the out-of-span tilts.  So it replaces rSVD wherever eta <= xi (and is the
efficient way to read off the spectrum / rotation anywhere); where eta > xi the
many small tilts are real content that still needs capturing.

Pure CPU / NumPy.

Usage
-----
    python examples/rotation_tracking_poc.py
    python examples/rotation_tracking_poc.py --K 28 --Ls 16,19,22 --stride 3
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from edm_incremental import (  # noqa: E402
    bond_matrix_and_old_subspace, fold_all_L, make_context, randomized_svd,
)

from edmtn.decomposition.base import truncation_rank  # noqa: E402
from edmtn.models import GaudinModel  # noqa: E402

_DIR_DATA = Path(__file__).resolve().parent / "data"


def _median_time(fn, n=5):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        out = fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), out


# --------------------------------------------------------------------------
# rotation tracking primitives
# --------------------------------------------------------------------------

def procrustes_align(U_old, U_new):
    """Optimal unitary ``R`` minimising ``||U_old R - U_new||_F``.

    Returns ``(R, residual, cos_angles)`` with ``R`` the unitary polar factor of
    ``U_old^H U_new``, ``residual = ||U_new - U_old R||_F = sqrt(sum 2(1-cos))``,
    and ``cos_angles`` the cosines of the principal angles.
    """
    E = U_old.conj().T @ U_new
    W, s, Zh = np.linalg.svd(E, full_matrices=False)
    R = W @ Zh
    residual = float(np.linalg.norm(U_new - U_old @ R))
    return R, residual, np.clip(s, 0.0, 1.0)


def inspan_track(M, U_L):
    """In-span SVD ("rotation tracking"): project then SVD the reduced matrix.

    ``B = U_L^H M`` (D x n); ``B = R Lambda V^H``.  The new singular basis is
    ``U_L R`` and the spectrum is ``Lambda`` -- obtained from a D-sized SVD, with
    no random projection.  Returns ``(R, s, Vh, B)``.
    """
    B = U_L.conj().T @ M
    R, s, Vh = np.linalg.svd(B, full_matrices=False)
    return R, s, Vh, B


# --------------------------------------------------------------------------
# per-bond comparison
# --------------------------------------------------------------------------

def analyse_bond(ctx, mps_L, k, tau, *, n_rep=5):
    d2, xi = ctx["ref_index"], ctx["cutoff"]
    M, U = bond_matrix_and_old_subspace(ctx, mps_L, k, tau)
    m, n = M.shape
    D_old = U.shape[1]
    normM = float(np.linalg.norm(M)) or 1.0

    # ground truth full SVD
    t_full, (Uf, sf, Vhf) = _median_time(lambda: np.linalg.svd(M, full_matrices=False), n=n_rep)
    keep = truncation_rank(sf, cutoff=xi, cutoff_mode="rel_ref", ref_index=d2)
    err_base = float(np.linalg.norm(sf[keep:]) / normM) if keep < sf.size else 0.0
    thresh = xi * float(sf[min(d2, sf.size - 1)])

    # residual energy + rank
    UHM = U.conj().T @ M
    Mperp = M - U @ UHM
    eta = float(np.linalg.norm(Mperp) / normM)
    r_eff = int(np.count_nonzero(np.linalg.svd(Mperp, compute_uv=False) > thresh))

    # premise: rotation between old subspace and the true new left basis
    Utrue = Uf[:, :keep]
    _, procr_resid, cos = procrustes_align(U, Utrue)
    chordal = float(np.sqrt(np.clip(1.0 - cos**2, 0.0, None).sum()))

    # Fair timing: the projection B = U^H M and the residual M^perp both come from
    # the SAME shared GEMM (the Tier-1 byproduct), so exclude it from all three and
    # time only the *decomposition* each method runs on its input matrix.
    B = UHM                                                        # = U^H M (D x n)
    t_track, (R, st, Vt) = _median_time(
        lambda: np.linalg.svd(B, full_matrices=False), n=n_rep)   # reduced D x n SVD
    err_track = float(np.linalg.norm(M - U @ B) / normM)          # = eta
    kk = min(keep, st.size)
    sv_err = float(np.linalg.norm(st[:kk] - sf[:kk]) / (np.linalg.norm(sf[:kk]) or 1.0))

    rng = np.random.default_rng(0)
    t_rsvd, (Ur, sr, Vr) = _median_time(
        lambda: randomized_svd(Mperp, r_eff, rng=rng), n=n_rep)
    recon_rsvd = U @ UHM + (Ur * sr) @ Vr if r_eff > 0 else U @ UHM
    err_rsvd = float(np.linalg.norm(M - recon_rsvd) / normM)

    return {
        "tau": tau, "m": m, "n": n, "D_old": D_old, "keep": keep,
        "eta": eta, "r_eff": r_eff, "chordal": chordal, "procr_resid": procr_resid,
        "sv_err": sv_err, "err_track": err_track, "err_rsvd": err_rsvd,
        "err_base": err_base, "eta_le_xi": eta <= xi,
        "t_full": t_full, "t_track": t_track, "t_rsvd": t_rsvd,
    }


def analyse_fold(ctx, mps, L, *, stride):
    k = L
    taus = list(range(1, ctx["n_sites"], stride))
    print(f"\n=== fold L={L} -> {L+1} ===")
    print(f"{'tau':>4} {'m':>4} {'n':>4} {'D':>4} {'eta':>9} {'r_eff':>6} "
          f"{'chordal':>9} {'procr':>9} {'sv_err':>9} {'errTrk':>9} {'errrSVD':>9} "
          f"{'SVD':>7} {'Trk':>7} {'rSVD':>7}")
    print("-" * 118)
    rows = []
    for tau in taus:
        r = analyse_bond(ctx, mps[L], k, tau)
        rows.append(r)
        print(f"{r['tau']:>4} {r['m']:>4} {r['n']:>4} {r['D_old']:>4} {r['eta']:>9.2e} "
              f"{r['r_eff']:>6} {r['chordal']:>9.2e} {r['procr_resid']:>9.2e} "
              f"{r['sv_err']:>9.2e} {r['err_track']:>9.2e} {r['err_rsvd']:>9.2e} "
              f"{r['t_full']*1e3:>6.1f} {r['t_track']*1e3:>6.2f} {r['t_rsvd']*1e3:>6.2f}")
    return rows


# --------------------------------------------------------------------------
# synthetic self-check
# --------------------------------------------------------------------------

def self_check():
    """Rotation tracking recovers the spectrum exactly; error = out-of-span fraction."""
    rng = np.random.default_rng(0)
    m, n, D = 200, 300, 40

    def orth(a, b):
        Q, _ = np.linalg.qr(rng.standard_normal((a, b)) + 1j * rng.standard_normal((a, b)))
        return Q[:, :b]

    U_L = orth(m, D)
    Rtrue, _ = np.linalg.qr(rng.standard_normal((D, D)) + 1j * rng.standard_normal((D, D)))
    s_true = np.sort(rng.uniform(0.1, 1.0, D))[::-1]
    V = orth(n, D)
    M0 = (U_L @ Rtrue * s_true) @ V.conj().T               # in-span, rotated
    Qp = orth(m, D)                                         # out-of-span tilt
    Qp = Qp - U_L @ (U_L.conj().T @ Qp)
    Qp, _ = np.linalg.qr(Qp)
    eps = 1e-4
    M = M0 + eps * (Qp[:, :D] * s_true) @ V.conj().T
    R, st, Vt, B = inspan_track(M, U_L)
    sv_err = np.linalg.norm(np.sort(st)[::-1] - s_true) / np.linalg.norm(s_true)
    recon = np.linalg.norm(M - U_L @ B) / np.linalg.norm(M)
    print("=== synthetic self-check ===")
    print(f"  spectrum recovery err = {sv_err:.2e}   (rotation tracking sees the in-span part)")
    print(f"  reconstruction err    = {recon:.2e}   (~ out-of-span fraction eps~{eps:g})")
    assert sv_err < 1e-8 and recon < 5e-3
    print("  self-check PASSED\n")


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--g", type=float, default=1.0)
    ap.add_argument("--K", type=int, default=24)
    ap.add_argument("--Ls", default="16,19,22", help="transition-zone folds L (-> L+1)")
    ap.add_argument("--T", type=float, default=3.0)
    ap.add_argument("--eps", type=float, default=0.2)
    ap.add_argument("--cutoff", type=float, default=1e-6)
    ap.add_argument("--max-bond", type=int, default=400)
    ap.add_argument("--order", type=int, default=2, choices=(1, 2))
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--name", default="rotation_tracking")
    args = ap.parse_args()

    self_check()

    Ls = [int(s) for s in args.Ls.split(",") if int(s) < args.K]
    model = GaudinModel(g=args.g, K=args.K)
    ctx = make_context(model, T=args.T, eps=args.eps, order=args.order,
                       cutoff=args.cutoff, max_bond=args.max_bond)
    print(f"Rotation-tracking study (Gaudin): K={args.K}, T={args.T} g^-1, eps={args.eps} g^-1, "
          f"order={args.order}, xi={args.cutoff:g}, D_a={ctx['D_a']}")
    t0 = time.perf_counter()
    mps = fold_all_L(ctx, K=max(Ls) + 1)
    print(f"  (fold-all wall {time.perf_counter()-t0:.1f}s)")
    print("  columns: eta=||Mperp||/||M||; sv_err=spectrum vs full SVD; errTrk=rotation-track "
          "recon(=eta); errrSVD=Tier-2 recon; times in ms")

    all_rows = {}
    for L in Ls:
        all_rows[L] = analyse_fold(ctx, mps, L, stride=args.stride)

    # aggregate verdict
    flat = [r for rows in all_rows.values() for r in rows if r["eta"] > 0]
    if flat:
        sv = np.array([r["sv_err"] for r in flat])
        spd_full = np.array([r["t_full"] / r["t_track"] for r in flat])
        spd_rsvd = np.array([r["t_rsvd"] / r["t_track"] for r in flat])
        frac_le = np.mean([r["eta_le_xi"] for r in flat])
        match = np.array([abs(r["procr_resid"] - r["chordal"]) for r in flat]).max()
        print("\n=== verdict (transition zone) ===")
        print(f"  premise: max |procrustes_resid - chordal| = {match:.1e}  "
              "-> U_{L+1} = U_L R + (out-of-span); the change is a rotation R + tilt.")
        print(f"  spectrum recovery (rotation tracking): median sv_err = {np.median(sv):.1e}  "
              "-> Schmidt values + rotation R recovered near-exactly, no rSVD.")
        print(f"  decomposition speedup of rotation tracking: median {np.median(spd_full):.1f}x "
              f"vs full SVD, {np.median(spd_rsvd):.1f}x vs rSVD (reduced D x n SVD).")
        print(f"  reconstruction: rotation-track error = eta; fraction of bonds with eta<=xi "
              f"(rotation tracking already cutoff-accurate) = {100*frac_le:.0f}%.")
        print("  => rotation tracking is the cheap, deterministic way to read the bond's "
              "spectrum + rotation in this zone; it fully replaces rSVD where eta<=xi, and\n"
              "  where eta>xi the residual is many small out-of-span tilts (r_eff) that still "
              "need capturing if strict cutoff reconstruction is required.")

    _DIR_DATA.mkdir(parents=True, exist_ok=True)
    npz = _DIR_DATA / f"{args.name}.npz"
    save = {"K": args.K, "xi": args.cutoff, "Ls": np.asarray(Ls)}
    for L, rows in all_rows.items():
        for key in ("tau", "eta", "r_eff", "chordal", "procr_resid", "sv_err",
                    "err_track", "err_rsvd", "t_full", "t_track", "t_rsvd"):
            save[f"{L}_{key}"] = np.array([r[key] for r in rows])
    np.savez(npz, **save)
    print(f"\nsaved {npz}")


if __name__ == "__main__":
    main()
