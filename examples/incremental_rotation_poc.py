"""Incremental rotation tracking vs rSVD, at *cutoff* accuracy (transition zone).

Follow-up to ``rotation_tracking_poc.py``.  There, pure rotation tracking (in-span
SVD of ``B = U_L^H M``) recovered the rotation R and Schmidt spectrum near-exactly
but its *reconstruction* error was eta (it drops the out-of-span tilts).  To reach
the cutoff it must also capture the residual ``M^perp`` (the ``r_eff`` small-angle
tilts).  This script asks: can a rotation-tracking *incremental* update reach the
cutoff while beating cold rSVD on wall-clock?

The cost structure (all methods share the projection ``B = U_L^H M`` and the
residual ``M^perp = M - U_L B`` -- the Tier-1 byproduct -- so those are excluded
from timing; only the decomposition is timed):

  full SVD of M (m x n)                          exact baseline
  Tier-2 cold rSVD(M^perp, r_eff, n_iter=2)      current Tier-2
  single-pass rSVD(M^perp, r_eff, n_iter=0)      drop the power iterations
  incremental = in-span SVD(B) + single-pass     rotation R + spectrum + residual
               rSVD(M^perp, r_eff, n_iter=0)

Key empirical lever (Step A): the residual spectrum decays fast, so a *single
pass* (no power iteration) already reaches the cutoff -- the power iterations that
cold rSVD spends are unnecessary here.  Incremental tracking then = a tiny in-span
SVD + a single-pass residual.

Step B tracks the rotation R across consecutive folds at a fixed bond (its
per-step angle and the residual rank), to gauge whether R could be *composed*
incrementally rather than recomputed.

Honest verdict is printed: for cutoff *reconstruction* the win over cold rSVD is
the dropped power iterations (which a plain single-pass rSVD also gets); the
unique value of incremental rotation tracking is that it additionally yields the
clean rotation R and Schmidt spectrum at little extra cost.

Pure CPU / NumPy.

Usage
-----
    python examples/incremental_rotation_poc.py
    python examples/incremental_rotation_poc.py --K 24 --Ls 16,19,22 --track-tau 13
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


def _median_time(fn, n=7):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        out = fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), out


def _resid_err(Mperp, Q, sr, Vr, normM):
    if Q is None or sr.size == 0:
        return float(np.linalg.norm(Mperp) / normM)
    return float(np.linalg.norm(Mperp - (Q * sr) @ Vr) / normM)


# --------------------------------------------------------------------------
# Step A: per-bond method comparison at cutoff
# --------------------------------------------------------------------------

def analyse_bond(ctx, mps_L, k, tau, *, n_rep=7):
    d2, xi = ctx["ref_index"], ctx["cutoff"]
    M, U = bond_matrix_and_old_subspace(ctx, mps_L, k, tau)
    m, n = M.shape
    D = U.shape[1]
    normM = float(np.linalg.norm(M)) or 1.0

    sf = np.linalg.svd(M, compute_uv=False)
    keep = truncation_rank(sf, cutoff=xi, cutoff_mode="rel_ref", ref_index=d2)
    err_base = float(np.linalg.norm(sf[keep:]) / normM) if keep < sf.size else 0.0
    thresh = xi * float(sf[min(d2, sf.size - 1)])

    # shared setup (excluded from timing): projection + residual
    B = U.conj().T @ M
    Mperp = M - U @ B
    eta = float(np.linalg.norm(Mperp) / normM)
    r_eff = int(np.count_nonzero(np.linalg.svd(Mperp, compute_uv=False) > thresh))
    rng = np.random.default_rng(0)

    # timed decompositions
    t_full, _ = _median_time(lambda: np.linalg.svd(M, full_matrices=False), n=n_rep)
    t_cold, (Uc, sc, Vc) = _median_time(
        lambda: randomized_svd(Mperp, r_eff, n_iter=2, rng=rng), n=n_rep)
    t_single, (U0, s0, V0) = _median_time(
        lambda: randomized_svd(Mperp, r_eff, n_iter=0, rng=rng), n=n_rep)

    def _inc():
        R, s_in, V_in = np.linalg.svd(B, full_matrices=False)   # rotation + spectrum
        Q, sr, Vr = randomized_svd(Mperp, r_eff, n_iter=0, rng=rng)
        return R, s_in, Q, sr, Vr
    t_inc, (R, s_in, Qi, sri, Vri) = _median_time(_inc, n=n_rep)

    err_cold = _resid_err(Mperp, Uc, sc, Vc, normM)
    err_single = _resid_err(Mperp, U0, s0, V0, normM)
    err_inc = _resid_err(Mperp, Qi, sri, Vri, normM)        # same residual as single
    sv_err = float(np.linalg.norm(s_in[:keep] - sf[:keep]) / (np.linalg.norm(sf[:keep]) or 1.0))

    return {
        "tau": tau, "m": m, "n": n, "D": D, "keep": keep, "eta": eta, "r_eff": r_eff,
        "err_base": err_base, "err_cold": err_cold, "err_single": err_single,
        "err_inc": err_inc, "sv_err": sv_err,
        "t_full": t_full, "t_cold": t_cold, "t_single": t_single, "t_inc": t_inc,
    }


def analyse_fold(ctx, mps, L, *, stride):
    taus = list(range(1, ctx["n_sites"], stride))
    print(f"\n=== fold L={L} -> {L+1} ===")
    print(f"{'tau':>4} {'D':>4} {'r_eff':>6} {'eta':>9} {'errCold':>9} {'errSing':>9} "
          f"{'errInc':>9} {'svErr':>9} | {'SVD':>6} {'cold':>6} {'sing':>6} {'inc':>6} "
          f"{'inc/cold':>8}")
    print("-" * 108)
    rows = []
    for tau in taus:
        r = analyse_bond(ctx, mps[L], L, tau)
        rows.append(r)
        spd = r["t_cold"] / r["t_inc"] if r["t_inc"] > 0 else 0.0
        print(f"{r['tau']:>4} {r['D']:>4} {r['r_eff']:>6} {r['eta']:>9.2e} "
              f"{r['err_cold']:>9.2e} {r['err_single']:>9.2e} {r['err_inc']:>9.2e} "
              f"{r['sv_err']:>9.2e} | {r['t_full']*1e3:>5.1f} {r['t_cold']*1e3:>5.2f} "
              f"{r['t_single']*1e3:>5.2f} {r['t_inc']*1e3:>5.2f} {spd:>7.2f}x")
    return rows


# --------------------------------------------------------------------------
# Step B: rotation evolution across folds at a fixed bond
# --------------------------------------------------------------------------

def track_rotation(ctx, mps, Ls, tau):
    print(f"\n=== Step B: rotation across folds at bond tau={tau} ===")
    print(f"{'L->L+1':>8} {'D':>4} {'max_angle(rad)':>14} {'chordal':>9} {'r_eff':>6}")
    print("-" * 48)
    d2, xi = ctx["ref_index"], ctx["cutoff"]
    out = []
    for L in Ls:
        M, U = bond_matrix_and_old_subspace(ctx, mps[L], L, tau)
        sf = np.linalg.svd(M, compute_uv=False)
        keep = truncation_rank(sf, cutoff=xi, cutoff_mode="rel_ref", ref_index=d2)
        thresh = xi * float(sf[min(d2, sf.size - 1)])
        Uf, _, _ = np.linalg.svd(M, full_matrices=False)
        cos = np.clip(np.linalg.svd(U.conj().T @ Uf[:, :keep], compute_uv=False), 0, 1)
        max_angle = float(np.arccos(cos.min()))
        chordal = float(np.sqrt(np.clip(1 - cos**2, 0, None).sum()))
        Mperp = M - U @ (U.conj().T @ M)
        r_eff = int(np.count_nonzero(np.linalg.svd(Mperp, compute_uv=False) > thresh))
        out.append({"L": L, "D": U.shape[1], "max_angle": max_angle,
                    "chordal": chordal, "r_eff": r_eff})
        print(f"{L:>3}->{L+1:<3} {U.shape[1]:>4} {max_angle:>14.3e} {chordal:>9.2e} {r_eff:>6}")
    return out


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--g", type=float, default=1.0)
    ap.add_argument("--K", type=int, default=24)
    ap.add_argument("--Ls", default="16,19,22")
    ap.add_argument("--T", type=float, default=3.0)
    ap.add_argument("--eps", type=float, default=0.2)
    ap.add_argument("--cutoff", type=float, default=1e-6)
    ap.add_argument("--max-bond", type=int, default=400)
    ap.add_argument("--order", type=int, default=2, choices=(1, 2))
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--track-tau", type=int, default=13)
    ap.add_argument("--name", default="incremental_rotation")
    args = ap.parse_args()

    Ls = [int(s) for s in args.Ls.split(",") if int(s) < args.K]
    model = GaudinModel(g=args.g, K=args.K)
    ctx = make_context(model, T=args.T, eps=args.eps, order=args.order,
                       cutoff=args.cutoff, max_bond=args.max_bond)
    print(f"Incremental rotation tracking (Gaudin): K={args.K}, T={args.T} g^-1, "
          f"eps={args.eps} g^-1, order={args.order}, xi={args.cutoff:g}")
    t0 = time.perf_counter()
    mps = fold_all_L(ctx, K=max(Ls) + 1)
    print(f"  (fold-all wall {time.perf_counter()-t0:.1f}s)")
    print("  errCold/errSing/errInc = residual reconstruction error; svErr = spectrum "
          "vs full SVD; times in ms; inc/cold = speedup of incremental over cold rSVD")

    all_rows = {L: analyse_fold(ctx, mps, L, stride=args.stride) for L in Ls}
    track = track_rotation(ctx, mps, Ls, args.track_tau)

    flat = [r for rows in all_rows.values() for r in rows if r["r_eff"] > 0]
    if flat:
        spd_cold = np.array([r["t_cold"] / r["t_inc"] for r in flat])
        spd_single = np.array([r["t_single"] / r["t_inc"] for r in flat])
        cold = np.array([r["err_cold"] for r in flat])
        sing = np.array([r["err_single"] for r in flat])
        sv = np.array([r["sv_err"] for r in flat])
        ang = np.array([t["max_angle"] for t in track])
        print("\n=== verdict (transition zone) ===")
        print(f"  power iterations unnecessary: single-pass vs cold rSVD residual error "
              f"median {np.median(sing):.1e} vs {np.median(cold):.1e} (both ~cutoff).")
        print(f"  incremental (in-span SVD + single-pass residual) reaches cutoff AND "
              f"yields the rotation + spectrum (sv_err median {np.median(sv):.1e}).")
        print(f"  wall-clock: incremental vs COLD rSVD(2it) = {np.median(spd_cold):.2f}x; "
              f"incremental vs SINGLE-pass rSVD = {np.median(spd_single):.2f}x.")
        print(f"  rotation per fold is small (max angle median {np.median(ang):.2e} rad) "
              "-> R could be composed/tracked across folds rather than recomputed.")
        verdict = "FASTER than cold rSVD" if np.median(spd_cold) > 1.05 else (
            "~PARITY with cold rSVD" if np.median(spd_cold) > 0.9 else "slower than cold rSVD")
        print(f"  => incremental rotation tracking is {verdict} at cutoff accuracy, and the "
              "speedup is the dropped power iterations (a plain single-pass rSVD shares it).\n"
              "  Its distinctive payoff is the clean rotation R + Schmidt spectrum it returns "
              "for ~free on top of the residual capture.")

    _DIR_DATA.mkdir(parents=True, exist_ok=True)
    npz = _DIR_DATA / f"{args.name}.npz"
    save = {"K": args.K, "xi": args.cutoff, "Ls": np.asarray(Ls)}
    for L, rows in all_rows.items():
        for key in ("tau", "D", "r_eff", "eta", "err_cold", "err_single", "err_inc",
                    "sv_err", "t_full", "t_cold", "t_single", "t_inc"):
            save[f"{L}_{key}"] = np.array([r[key] for r in rows])
    save["track_L"] = np.array([t["L"] for t in track])
    save["track_angle"] = np.array([t["max_angle"] for t in track])
    save["track_reff"] = np.array([t["r_eff"] for t in track])
    np.savez(npz, **save)
    print(f"\nsaved {npz}")


if __name__ == "__main__":
    main()
