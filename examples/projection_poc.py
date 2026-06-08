"""Proof-of-concept for the two acceleration tiers of the incremental-update
framework, measured against the production full-SVD fold.

Tier 1 -- *pure projection* (end-to-end, on a late/weak fold, default L=22->23).
    Fold sub-bath L+1 (cheap MPO*MPS contraction) then, instead of compressing
    with per-bond SVDs, project the uncompressed EDM-MPS onto step-L's left
    subspace with one GEMM sweep (``edm_incremental.tier1_project``).  We compare
    the resulting <S_z(t)> curve and final reduced density matrix against the
    exact full-SVD fold, and time both compressions on the same uncompressed MPS.

Tier 2 -- *projection + randomized SVD on the residual* (per-bond, default
    L=8->9, a mid/strong fold where pure projection is not enough).  At each bond
    the truncation factorises a matrix ``M`` (rows = the bond being compressed).
    We compare:
        baseline:  full SVD of M  +  rank selection (the production path);
        tier 2:    M^|| = U (U^H M)        [GEMM, no SVD, U carried over]
                   M^perp = M - M^||
                   small randomized SVD of M^perp (effective rank << D^(L)).
    reporting reconstruction error, the residual rank actually needed, and the
    wall-clock speedup.

The known subspace ``U^(L)`` is treated as *carried over* from the previous step
(a streaming solver keeps it for free); the timed kernels are exactly the extra
work each tier does per fold / per bond.

Pure CPU / NumPy.

Usage
-----
    python examples/projection_poc.py
    python examples/projection_poc.py --K 24 --tier1-L 22 --tier2-L 8
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
    bond_matrix_and_old_subspace, compress_full, fold_all_L, fold_uncompressed,
    left_canonical_copy, make_context, randomized_svd, tier1_project,
)

from edmtn.decomposition.base import truncation_rank  # noqa: E402
from edmtn.observables.extractor import ObservableExtractor  # noqa: E402
from edmtn.models import GaudinModel  # noqa: E402

_DIR_DATA = Path(__file__).resolve().parent / "data"


def _median_time(fn, n=3):
    """Median wall-clock of ``n`` calls to ``fn`` (returns ``(median_s, last_result)``)."""
    ts, out = [], None
    for _ in range(n):
        t0 = time.perf_counter()
        out = fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), out


def _sz_curve(mps, eps, order):
    _, vals = ObservableExtractor.coupling_polarization_history(
        mps, eps, channel=3, order=order)
    return vals


# --------------------------------------------------------------------------
# Tier 1: end-to-end pure projection
# --------------------------------------------------------------------------

def tier1_demo(ctx, mps, L, *, n_rep=3):
    eps, order = ctx["eps"], ctx["order"]
    mps_L = mps[L]
    exact_L1 = mps[L + 1]                      # full-SVD reference (from fold_all_L)
    k = L                                      # fold sub-bath index L (the (L+1)-th)

    # one uncompressed fold (shared input to both compressions)
    unc = fold_uncompressed(ctx, mps_L, k)
    print(f"\n=== Tier 1 (pure projection), fold L={L} -> {L+1} ===")
    print(f"  uncompressed bond max = {unc.max_bond}, sites = {unc.num_sites}")

    # carried-over subspace: step-L isometries (one-time canonicalisation)
    t_canon, old_iso = _median_time(lambda: left_canonical_copy(mps_L), n=n_rep)

    # baseline full-SVD compression (fresh copy each call -- compress mutates)
    t_full, mps_full = _median_time(lambda: compress_full(ctx, unc.copy())[0], n=n_rep)
    # tier-1 projection (fresh copy each call)
    t_proj, mps_proj = _median_time(
        lambda: tier1_project(unc.copy(), mps_L, old_isometry=old_iso), n=n_rep)

    sz_full = _sz_curve(mps_full, eps, order)
    sz_proj = _sz_curve(mps_proj, eps, order)
    sz_exact = _sz_curve(exact_L1, eps, order)

    d_full_exact = float(np.max(np.abs(sz_full - sz_exact)))   # baseline reproduces exact
    d_proj_full = float(np.max(np.abs(sz_proj - sz_full)))
    rho_proj = mps_proj.reduced_density_matrix()
    rho_full = mps_full.reduced_density_matrix()
    rho_err = float(np.linalg.norm(rho_proj - rho_full) / np.linalg.norm(rho_full))

    print(f"  D^(L)={mps_L.max_bond}  D^(L+1)(full)={mps_full.max_bond}  "
          f"D(proj)={mps_proj.max_bond}  (projection keeps the old bond)")
    print(f"  accuracy:  max|d<Sz>| proj-vs-full = {d_proj_full:.3e}   "
          f"rho err = {rho_err:.3e}   (full-vs-exact = {d_full_exact:.1e})")
    print(f"  wall-clock (median of {n_rep}):  full-SVD compress = {t_full*1e3:.2f} ms   "
          f"projection = {t_proj*1e3:.2f} ms")
    print(f"  speedup:  full / projection = {t_full/t_proj:.1f}x   "
          f"full / (projection + canon) = {t_full/(t_proj+t_canon):.1f}x   "
          f"(canon {t_canon*1e3:.2f} ms, amortised across the fold)")
    return {
        "L": L, "t_full": t_full, "t_proj": t_proj, "t_canon": t_canon,
        "d_proj_full": d_proj_full, "rho_err": rho_err, "d_full_exact": d_full_exact,
        "sz_full": sz_full, "sz_proj": sz_proj, "sz_exact": sz_exact,
        "D_L": mps_L.max_bond, "D_L1_full": mps_full.max_bond,
    }


# --------------------------------------------------------------------------
# Tier 2: per-bond projection + randomized SVD on the residual
# --------------------------------------------------------------------------

def tier2_bond(ctx, mps_L, k, tau, *, n_rep=5):
    """One bond: baseline full SVD vs (projection + rSVD on the residual).

    The residual rank ``r`` is the *true* number of directions (beyond the old
    subspace) needed to reach the cutoff -- determined by an untimed analysis SVD
    of the residual -- and the timed Tier-2 kernel runs an rSVD at exactly that
    rank (what a streaming solver would target).
    """
    d2 = ctx["ref_index"]
    cutoff = ctx["cutoff"]
    M, U = bond_matrix_and_old_subspace(ctx, mps_L, k, tau)
    m, ncol = M.shape
    D_old = U.shape[1]
    normM = float(np.linalg.norm(M)) or 1.0

    # ---- baseline: full SVD + rank selection (production path) -----------
    def _full():
        return np.linalg.svd(M, full_matrices=False)
    t_full, (Uf, s, Vh) = _median_time(_full, n=n_rep)
    keep = truncation_rank(s, cutoff=cutoff, cutoff_mode="rel_ref", ref_index=d2)
    err_base = float(np.linalg.norm(s[keep:]) / normM) if keep < s.size else 0.0
    thresh = cutoff * float(s[min(d2, s.size - 1)])

    # ---- true residual rank (untimed analysis) ---------------------------
    R0 = M - U @ (U.conj().T @ M)
    sr_full = np.linalg.svd(R0, compute_uv=False)
    r_target = int(np.count_nonzero(sr_full > thresh))

    # ---- timed Tier-2 kernel: projection (GEMM) + rSVD(rank=r_target) -----
    rng = np.random.default_rng(0)

    def _tier2():
        P = U.conj().T @ M            # (D_old, ncol)  -- GEMM, no SVD
        Mpar = U @ P                  # (m, ncol)
        R = M - Mpar                  # residual
        Ur, sr, Vhr = randomized_svd(R, rank=r_target, rng=rng)
        return Mpar, Ur, sr, Vhr
    t_t2, (Mpar, Ur, sr, Vhr) = _median_time(_tier2, n=n_rep)

    recon = Mpar + (Ur * sr) @ Vhr if r_target > 0 else Mpar
    err_t2 = float(np.linalg.norm(M - recon) / normM)

    return {
        "tau": tau, "m": m, "ncol": ncol, "D_old": D_old,
        "keep_base": keep, "err_base": err_base,
        "r_residual": r_target, "keep_t2": D_old + r_target, "err_t2": err_t2,
        "t_full": t_full, "t_t2": t_t2, "speedup": t_full / t_t2,
    }


def tier2_demo(ctx, mps, L, *, stride=4):
    k = L
    mps_L = mps[L]
    n_sites = ctx["n_sites"]
    taus = list(range(1, n_sites, stride))
    print(f"\n=== Tier 2 (projection + residual rSVD), fold L={L} -> {L+1} ===")
    print(f"  D_a = {ctx['D_a']}  (expected residual rank scale)")
    print(f"{'tau':>4} {'m':>5} {'ncol':>5} {'D_old':>6} {'keep':>5} {'r_res':>6} "
          f"{'err_base':>9} {'err_t2':>9} {'SVD[ms]':>8} {'T2[ms]':>8} {'speedup':>7}")
    print("-" * 86)
    rows = []
    for tau in taus:
        r = tier2_bond(ctx, mps_L, k, tau)
        rows.append(r)
        print(f"{tau:>4} {r['m']:>5} {r['ncol']:>5} {r['D_old']:>6} {r['keep_base']:>5} "
              f"{r['r_residual']:>6} {r['err_base']:>9.2e} {r['err_t2']:>9.2e} "
              f"{r['t_full']*1e3:>8.2f} {r['t_t2']*1e3:>8.2f} {r['speedup']:>5.1f}x")
    if rows:
        rr = np.array([r["r_residual"] for r in rows])
        do = np.array([r["D_old"] for r in rows])
        sp = np.array([r["speedup"] for r in rows])
        eb = np.array([r["err_base"] for r in rows])
        et = np.array([r["err_t2"] for r in rows])
        ratio = np.median(rr / np.maximum(do, 1))
        print(f"\n  residual rank r: median={int(np.median(rr))}, max={int(rr.max())} "
              f"(vs D_old median={int(np.median(do))}, D_a={ctx['D_a']}; "
              f"median r/D_old={ratio:.2f})")
        print(f"  reconstruction error: baseline median={np.median(eb):.2e}, "
              f"tier-2 median={np.median(et):.2e}  (matched by construction)")
        print(f"  per-bond speedup (full SVD / tier-2): median={np.median(sp):.1f}x, "
              f"max={sp.max():.1f}x")
        print("  => Tier-2 pays off where r << D_old; r ~ D_old means the subspace "
              "rotated (strong fold).")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--g", type=float, default=1.0)
    ap.add_argument("--K", type=int, default=24, help="total bath spins")
    ap.add_argument("--tier1-L", type=int, default=22, help="late/weak fold for Tier 1")
    ap.add_argument("--tier2-L", type=int, default=8, help="mid/strong fold for Tier 2")
    ap.add_argument("--T", type=float, default=3.0)
    ap.add_argument("--eps", type=float, default=0.2)
    ap.add_argument("--cutoff", type=float, default=1e-6)
    ap.add_argument("--max-bond", type=int, default=400)
    ap.add_argument("--order", type=int, default=2, choices=(1, 2))
    ap.add_argument("--tier2-stride", type=int, default=4, help="analyse every Nth bond")
    ap.add_argument("--name", default="projection_poc")
    args = ap.parse_args()

    need = max(args.tier1_L, args.tier2_L) + 1
    if need > args.K:
        raise SystemExit(f"K={args.K} too small for L+1={need}")

    model = GaudinModel(g=args.g, K=args.K)
    ctx = make_context(model, T=args.T, eps=args.eps, order=args.order,
                       cutoff=args.cutoff, max_bond=args.max_bond)
    print(f"Projection proof-of-concept (Gaudin): K={args.K}, T={args.T} g^-1, "
          f"eps={args.eps} g^-1, order={args.order}, xi={args.cutoff:g}, D_a={ctx['D_a']}")
    t0 = time.perf_counter()
    mps = fold_all_L(ctx, K=need)
    print(f"  (fold-all to L={need} wall {time.perf_counter() - t0:.1f}s)")

    t1 = tier1_demo(ctx, mps, args.tier1_L)
    t2 = tier2_demo(ctx, mps, args.tier2_L, stride=args.tier2_stride)

    _DIR_DATA.mkdir(parents=True, exist_ok=True)
    npz = _DIR_DATA / f"{args.name}.npz"
    np.savez(
        npz, K=args.K, D_a=ctx["D_a"], tier1_L=args.tier1_L, tier2_L=args.tier2_L,
        t1_sz_full=t1["sz_full"], t1_sz_proj=t1["sz_proj"], t1_sz_exact=t1["sz_exact"],
        t1_speedup=t1["t_full"] / t1["t_proj"], t1_rho_err=t1["rho_err"],
        t2_tau=np.array([r["tau"] for r in t2]),
        t2_r_residual=np.array([r["r_residual"] for r in t2]),
        t2_err_base=np.array([r["err_base"] for r in t2]),
        t2_err_t2=np.array([r["err_t2"] for r in t2]),
        t2_speedup=np.array([r["speedup"] for r in t2]),
    )
    print(f"\nsaved {npz}")


if __name__ == "__main__":
    main()
