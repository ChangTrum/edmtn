"""Zip-up / fused fold -- avoid ever forming the full D*D_a-width MPS.

EXAMPLES-ONLY; the solver pipeline (src/) is untouched and used verbatim as the
baseline (``EDMSolver`` -> the paper's Fig. 6 algorithm).

Idea (Stoudenmire-White zip-up, adapted to the EDM fold).  The pipeline folds the
sub-bath MPO into the whole MPS first (every bond -> D_a*D), then left-canonicalises
that full-width object (the QR sweep -- the bottleneck of section 14) and truncates.
Instead, fuse the two:

  * **Forward sweep (left->right):** at each site, contract the MPO tensor into the
    running (already-compressed) left bond and the MPS tensor, then immediately do a
    *loose* SVD truncation and push ``S V^H`` to the next site.  The left bond handed
    on is kept at ~D_loose, so the working matrix is ``(d_phys*D_loose) x (D_a*D_r)``
    -- it never reaches the full ``(d_phys*D_a*D) x (D_a*D)`` the QR sweep faces.  The
    sweep also leaves the MPS left-canonical (the U factors are isometries).
  * **Backward sweep (right->left):** strict-xi truncation with the validated
    single-pass rSVD (section 13).

Two costs the user flagged are measured head-on:
  1. the forward sweep does an *SVD* per site (pricier than a QR), and
  2. the two sweeps' truncation errors *add* -- and a forward cutoff *looser than the
     final* xi (e.g. sqrt(xi) = 1e-3 >> 1e-6 under rel_ref) discards directions the
     strict backward pass cannot recover.

So we sweep the forward cutoff across xi^2 (conservative superset) ... sqrt(xi)
(aggressive) and measure accuracy vs the pipeline and wall-clock, against the
section-14 QR+single-pass reference.  Same bar as section 13: accuracy below the
cutoff, seed-stable.

Metrics: <S_z(t)> max abs error vs pipeline; imag/real ratio; forward vs backward
sweep wall-clock; max bond reached in the forward sweep (confirms it stays
compressed); final Dmax; total wall and speedup.

Pure CPU / NumPy.  Defaults sized for a 16 GB laptop.

Usage
-----
    python examples/zipup_fused_e2e.py
    python examples/zipup_fused_e2e.py --K 24 --cutoff 1e-6 --fwd sqrt,1,1.5
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from adaptive_tiers_e2e import run_baseline  # noqa: E402
from canon_alternatives_e2e import coupling_polarization_tolerant  # noqa: E402
from edm_incremental import make_context, randomized_svd  # noqa: E402
from uniform_rsvd_e2e import _rsvd_truncate, _svd_truncate  # noqa: E402

from edmtn.decomposition.base import truncation_rank  # noqa: E402
from edmtn.evolution.mps_utils import EDMMPS  # noqa: E402
from edmtn.models import GaudinModel  # noqa: E402

_DIR_DATA = Path(__file__).resolve().parent / "data"


# --------------------------------------------------------------------------
# fused forward sweep: fold the MPO in, loose-truncate inline, keep left-canonical
# --------------------------------------------------------------------------

def _rsvd_factor(mat, cutoff, d2, mb, rng, guess):
    """Single-pass rSVD with resolution guard; returns isometric ``U``, ``s``, ``Vh``
    truncated at ``cutoff`` (rel_ref).  Cheaper than a full SVD for the forward
    loose pass (caveat #1) since single-pass rSVD is validated reliable (section 13)."""
    m, n = mat.shape
    full = min(m, n)
    R = int(np.clip(guess, d2 + 8, full))
    while True:
        U, s, Vh = randomized_svd(mat, R, n_iter=0, rng=rng)
        if s.size == 0:
            return U, s, Vh, 0
        sref = s[min(d2, s.size - 1)]
        if s[-1] <= cutoff * sref or R >= full:
            break
        R = min(2 * R, full)
    kk = truncation_rank(s, cutoff=cutoff, cutoff_mode="rel_ref", ref_index=d2, max_bond=mb)
    return U[:, :kk], s[:kk], Vh[:kk], kk


def forward_fold(ctx, mps_L, k, fwd_cutoff, fwd_svd, rng, fwd_kmem):
    """Contract sub-bath ``k``'s MPO into ``mps_L`` left->right with inline loose
    truncation (``fwd_svd`` = 'full' SVD or 'rsvd' single-pass rSVD).

    Returns ``(fwd_mps, max_fwd_bond)``; ``fwd_mps`` is left-canonical (sites
    0..n-2 isometric, centre at n-1), bonds truncated at ``fwd_cutoff``.
    """
    mpo = ctx["ke"].for_sub_bath(k).get_kernel_mpo(ctx["n_sites"]).site_tensors
    d, d_phys, d2, mb = ctx["d"], ctx["d_phys"], ctx["ref_index"], ctx["max_bond"]
    nsite = mps_L.num_sites
    A = [None] * nsite
    carry = None                       # (k_prev, incoming = a_l*chi_l)
    max_bond = 0
    for p in range(nsite):
        T = mpo[p]                                  # (u, dn, a_l, a_r)
        G = mps_L.tensors[p]                         # (dn, chi_l, chi_r)
        out = np.tensordot(T, G, axes=([1], [0]))    # (u, a_l, a_r, chi_l, chi_r)
        out = out.transpose(0, 1, 3, 2, 4)           # (u, a_l, chi_l, a_r, chi_r)
        u, al, cl, ar, cr = out.shape
        inc, outg = al * cl, ar * cr
        out2 = out.reshape(u, inc, outg)
        if carry is None:
            left = inc
            mat = out2.reshape(u * inc, outg)        # rows (u, inc), cols outg
        else:
            left = carry.shape[0]
            M = np.tensordot(carry, out2, axes=([1], [1]))   # (k_prev, u, outg)
            mat = M.transpose(1, 0, 2).reshape(u * left, outg)
        if p == nsite - 1:                            # centre: keep, no truncation
            A[p] = mat.reshape(u, left, outg)
            break
        if fwd_svd == "rsvd":
            U, s, Vh, kk = _rsvd_factor(mat, fwd_cutoff, d2, mb, rng,
                                        fwd_kmem.get(p, 16) + 16)
            fwd_kmem[p] = kk
        else:
            U, s, Vh = np.linalg.svd(mat, full_matrices=False)
            kk = truncation_rank(s, cutoff=fwd_cutoff, cutoff_mode="rel_ref",
                                 ref_index=d2, max_bond=mb)
            U, s, Vh = U[:, :kk], s[:kk], Vh[:kk]
        A[p] = U.reshape(u, left, kk)
        carry = s[:, None] * Vh                       # (kk, outg)
        max_bond = max(max_bond, kk)
    return EDMMPS(tensors=A, d=d, d_phys=d_phys, rho0_vec=mps_L.rho0_vec), max_bond


def backward_truncate(B, ctx, mode, rng, k_memory):
    """Strict-xi right->left truncation sweep (single-pass rSVD / cold / full SVD)."""
    xi, d2, mb = ctx["cutoff"], ctx["ref_index"], ctx["max_bond"]
    for p in range(B.num_sites - 1, 0, -1):
        G = B.tensors[p]
        dp, chil, chir = G.shape
        M = G.transpose(1, 0, 2).reshape(chil, dp * chir)
        if mode == "svd":
            US, Vh, kk, _ = _svd_truncate(M, xi=xi, ref_index=d2, max_bond=mb)
        else:
            n_iter = 2 if mode == "cold" else 0
            guess = k_memory.get(p, 16) + 16
            US, Vh, kk, _ = _rsvd_truncate(M, xi=xi, ref_index=d2, max_bond=mb,
                                           n_iter=n_iter, rng=rng, guess=guess)
        k_memory[p] = kk
        B.tensors[p] = Vh.reshape(kk, dp, chir).transpose(1, 0, 2)
        B.tensors[p - 1] = np.tensordot(B.tensors[p - 1], US, axes=([2], [0]))
    return B


def run_zipup(ctx, K, fwd_cutoff, bwd_mode, fwd_svd="full", seed=0):
    eps, order = ctx["eps"], ctx["order"]
    mps = ctx["mps0"]
    rng = np.random.default_rng(seed)
    k_memory: dict[int, int] = {}
    fwd_kmem: dict[int, int] = {}
    t_fwd = t_bwd = 0.0
    max_fwd = 0
    traj = []
    t0 = time.perf_counter()
    for k in range(K):
        tf = time.perf_counter()
        fwd, mfb = forward_fold(ctx, mps, k, fwd_cutoff, fwd_svd, rng, fwd_kmem)
        t_fwd += time.perf_counter() - tf
        max_fwd = max(max_fwd, mfb)
        if fwd.num_sites <= 1:
            mps = fwd
            traj.append(mps.max_bond)
            continue
        tb = time.perf_counter()
        mps = backward_truncate(fwd, ctx, bwd_mode, rng, k_memory)
        t_bwd += time.perf_counter() - tb
        traj.append(mps.max_bond)
    wall = time.perf_counter() - t0
    pol, imag_rel = coupling_polarization_tolerant(mps, eps, channel=3, order=order)
    return dict(pol=pol, wall=wall, t_fwd=t_fwd, t_bwd=t_bwd, max_fwd=max_fwd,
                dmax=mps.max_bond, imag_rel=imag_rel, traj=traj)


def _fwd_cutoffs(spec, xi):
    """Parse a forward-cutoff spec token into an actual cutoff value."""
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if tok == "sqrt":
            out.append(("sqrt(xi)", np.sqrt(xi)))
        elif tok in ("1", "xi"):
            out.append(("xi", xi))
        else:
            p = float(tok)
            out.append((f"xi^{tok}", xi ** p))
    return out


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
    ap.add_argument("--fwd", default="sqrt,1,1.5,2",
                    help="forward cutoffs as xi powers: 'sqrt'=xi^0.5, '1'=xi, '2'=xi^2")
    ap.add_argument("--bwd", default="single", choices=("single", "cold", "svd"))
    ap.add_argument("--fwd-svd", default="full", choices=("full", "rsvd"),
                    help="forward loose-truncation kernel: full SVD or single-pass rSVD")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--name", default="zipup")
    args = ap.parse_args()

    model = GaudinModel(g=args.g, K=args.K)
    ctx = make_context(model, T=args.T, eps=args.eps, order=args.order,
                       cutoff=args.cutoff, max_bond=args.max_bond)
    xi = args.cutoff
    print(f"Zip-up fused fold (Gaudin): K={args.K}, T={args.T} g^-1, eps={args.eps} g^-1, "
          f"order={args.order}, xi={xi:g}, fwd_svd={args.fwd_svd}, bwd={args.bwd}")

    print("  baseline (unmodified pipeline)...")
    sz_base, wall_base = run_baseline(model, T=args.T, eps=args.eps, order=args.order,
                                      cutoff=xi, max_bond=args.max_bond)
    print(f"    baseline wall = {wall_base:.1f}s")

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    print(f"\n  {'fwd cut':>10} | {'wall':>6} {'spd':>5} | {'fwd_s':>6} {'bwd_s':>6} | "
          f"{'max|dSz|':>9} {'imag/re':>8} | {'fwdD':>5} {'Dmax':>5}")
    print("  " + "-" * 78)
    results = {}
    for label, fcut in _fwd_cutoffs(args.fwd, xi):
        errs, walls, tfs, tbs, mfs, dmaxes, imags = [], [], [], [], [], [], []
        traj_last = None
        for seed in seeds:
            r = run_zipup(ctx, args.K, fcut, args.bwd, fwd_svd=args.fwd_svd, seed=seed)
            n = min(len(r["pol"]), len(sz_base))
            errs.append(float(np.max(np.abs(np.asarray(r["pol"][:n]) - np.asarray(sz_base[:n])))))
            walls.append(r["wall"]); tfs.append(r["t_fwd"]); tbs.append(r["t_bwd"])
            mfs.append(r["max_fwd"]); dmaxes.append(r["dmax"]); imags.append(r["imag_rel"])
            traj_last = r["traj"]
        err = max(errs); wall = float(np.mean(walls))
        tf = float(np.mean(tfs)); tb = float(np.mean(tbs))
        mf = max(mfs); dmax = max(dmaxes); imag_rel = max(imags)
        results[label] = dict(err=err, wall=wall, t_fwd=tf, t_bwd=tb, max_fwd=mf,
                              dmax=dmax, imag_rel=imag_rel, fcut=fcut, traj=traj_last)
        flag = "  <-- UNPHYSICAL" if imag_rel > 1e-3 else (
            "  <-- INACCURATE" if err > 1e-4 else "")
        print(f"  {label:>10} | {wall:6.1f} {wall_base/wall:4.2f}x | {tf:6.1f} {tb:6.1f} | "
              f"{err:9.2e} {imag_rel:8.1e} | {mf:5d} {dmax:5d}{flag}")

    print(f"\n  baseline {wall_base:.1f}s.  Zip-up keeps the forward bond near the strict")
    print(f"  bond instead of D_a*D; accuracy holds only while the forward cutoff is a")
    print(f"  superset of the final xi (look for the first INACCURATE row going aggressive).")

    _DIR_DATA.mkdir(parents=True, exist_ok=True)
    npz = _DIR_DATA / f"{args.name}.npz"
    save = {"K": args.K, "xi": xi, "wall_base": wall_base, "sz_base": np.asarray(sz_base)}
    for label, r in results.items():
        key = label.replace("(", "").replace(")", "").replace("^", "")
        save[f"err_{key}"] = r["err"]; save[f"wall_{key}"] = r["wall"]
        save[f"tfwd_{key}"] = r["t_fwd"]; save[f"tbwd_{key}"] = r["t_bwd"]
        save[f"dmax_{key}"] = r["dmax"]; save[f"maxfwd_{key}"] = r["max_fwd"]
    np.savez(npz, **save)
    print(f"\nsaved {npz}")


if __name__ == "__main__":
    main()
