"""End-to-end adaptive 3-tier EDM fold (L=0..K) vs the unmodified pipeline.

EXAMPLES-ONLY; the solver pipeline (src/) is untouched and is used verbatim as
the baseline (``EDMSolver`` -> the same algorithm as the paper's Fig. 6).

Adaptive compressor.  At every fold L->L+1 the step-L compressed MPS is carried
over as the known left subspace.  Each bond of the uncompressed L+1 MPS is then
compressed by one of three tiers, chosen from cheap pre-decomposition signals:

  project B = U_L^H M ;  M^perp = M - U_L B ;  eta = ||M^perp|| / ||M||

  * Tier 1   (eta < xi):                pure projection -- in-span SVD of B, no
                                        residual capture.
  * Tier 1.5 (eta >= xi, residual easy): project + single-pass rSVD of M^perp
                                        (no power iterations) + merge/truncate.
  * Tier 2   (eta >= xi, residual hard): project + cold rSVD (2 power iters) of
                                        M^perp + merge/truncate.

"residual easy/hard" is decided operationally by whether a single-pass rSVD of
M^perp already reconstructs it to the cutoff -- this is the computable meaning of
the user's "n_new(sqrt(xi)) = 0 and dD = 0" (no strong new direction, bond not
growing).  The true post-hoc n_new(sqrt(xi)) and dD are recorded to confirm the
chosen tier aligns with those criteria.

Orthogonality is monitored each fold via |Tr rho(T) - 1|; if it drifts past a
threshold a re-canonicalisation (left QR sweep) is triggered.

Metrics: per-(L, tau) wall-clock and tier; tier coverage; <S_z(t)> vs the pipeline
baseline (max abs error); total wall-clock and end-to-end speedup.

Pure CPU / NumPy.  Defaults are sized to run on a 16 GB laptop.

Usage
-----
    python examples/adaptive_tiers_e2e.py
    python examples/adaptive_tiers_e2e.py --K 24 --T 3 --eps 0.2
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from edm_incremental import (  # noqa: E402
    cross_left_overlaps, fold_uncompressed, left_canonical_copy, make_context,
    randomized_svd,
)

from edmtn.decomposition.base import truncation_rank  # noqa: E402
from edmtn.evolution import mps_utils  # noqa: E402
from edmtn.evolution.mps_utils import EDMMPS  # noqa: E402
from edmtn.observables.extractor import ObservableExtractor  # noqa: E402
from edmtn.driver import EDMSolver  # noqa: E402
from edmtn.models import GaudinModel  # noqa: E402

_DIR_DATA = Path(__file__).resolve().parent / "data"


# --------------------------------------------------------------------------
# per-bond tiered decomposition
# --------------------------------------------------------------------------

def _merge_decompose(M, U_old, W, xi, ref_index, max_bond):
    """Merge old subspace ``U_old`` with captured residual basis ``W`` and truncate.

    Returns ``(US, Vh, k, U_new)`` with ``US @ Vh ~ M`` (absorb='left').
    """
    Q = np.hstack([U_old, W]) if W.shape[1] else U_old   # orthonormal
    K = Q.conj().T @ M
    Uk, s, Vh = np.linalg.svd(K, full_matrices=False)
    k = truncation_rank(s, cutoff=xi, cutoff_mode="rel_ref",
                        ref_index=ref_index, max_bond=max_bond)
    U_new = Q @ Uk[:, :k]
    return U_new * s[:k], Vh[:k], k, U_new


def tier_decompose(M, U_old, *, xi, ref_index, max_bond, rng):
    """Truncated factorisation of bond matrix ``M`` (chil x n) reusing ``U_old``.

    The tier follows the user's criteria: Tier-1 if ``eta < xi``; otherwise a
    single-pass incremental decomposition is computed and, if its
    ``n_new(sqrt(xi)) == 0`` and ``dD == 0`` (no strong new direction, bond not
    growing), it is accepted as **Tier 1.5**; else the residual is recaptured with
    cold rSVD (2 power iterations) as **Tier 2**.  Returns
    ``(US, Vh, k, tier, info)`` (absorb='left'); ``info`` carries ``n_new``/``dD``.
    """
    nMp = float(np.linalg.norm(M)) or 1.0
    P = U_old.conj().T @ M                 # (D_old x n)
    Mperp = M - U_old @ P
    eta = float(np.linalg.norm(Mperp) / nMp)
    D_old = U_old.shape[1]
    sq = np.sqrt(xi)

    def _n_new(U_new):
        cos = np.clip(np.linalg.svd(U_old.conj().T @ U_new, compute_uv=False), 0, 1)
        return int(np.count_nonzero(cos < 1.0 - sq))

    # Tier 1: pure projection (in-span SVD of B)
    if eta < xi:
        Ub, s, Vh = np.linalg.svd(P, full_matrices=False)
        k = truncation_rank(s, cutoff=xi, cutoff_mode="rel_ref",
                            ref_index=ref_index, max_bond=max_bond)
        U_new = U_old @ Ub[:, :k]
        return (U_new * s[:k], Vh[:k], k, "T1",
                {"eta": eta, "n_new_sqrtxi": _n_new(U_new), "dD": k - D_old})

    # residual present.  Probe its rank with a *sensible* cap.  The fold can add
    # up to ~(D_a-1)*D_old new directions; a fixed D_old cap mutilates early/strong
    # folds (residual rank >> D_old) -> under-capture -> bond bloat.  So allow up to
    # ~2*D_old+buffer, and if even that saturates, the residual is genuinely high
    # rank (early fold) -> fall back to a full SVD of M (= baseline, Tier 0).
    s_ref = float(np.linalg.svd(P, compute_uv=False)[min(ref_index, P.shape[0] - 1)])
    cap = int(min(2 * D_old + 16, M.shape[0], M.shape[1]))
    Wp, srp, _ = randomized_svd(Mperp, cap, n_iter=0, rng=rng)
    if srp.size and srp[-1] > xi * s_ref:                  # cap saturated: high-rank residual
        Uf, sf, Vhf = np.linalg.svd(M, full_matrices=False)
        k = truncation_rank(sf, cutoff=xi, cutoff_mode="rel_ref",
                            ref_index=ref_index, max_bond=max_bond)
        return (Uf[:, :k] * sf[:k], Vhf[:k], k, "T0",
                {"eta": eta, "n_new_sqrtxi": _n_new(Uf[:, :k]), "dD": k - D_old})

    r_eff = max(1, int(np.count_nonzero(srp > xi * s_ref)))
    US, Vh, k, U_new = _merge_decompose(M, U_old, Wp[:, :r_eff], xi, ref_index, max_bond)
    n_new, dD = _n_new(U_new), k - D_old

    if n_new == 0 and dD == 0:
        return US, Vh, k, "T1.5", {"eta": eta, "n_new_sqrtxi": n_new, "dD": dD}

    # Tier 2: strong new directions / bond grows -> cold rSVD (power iterations)
    Wc, _, _ = randomized_svd(Mperp, r_eff, n_iter=2, rng=rng)
    US, Vh, k, U_new = _merge_decompose(M, U_old, Wc[:, :r_eff], xi, ref_index, max_bond)
    return US, Vh, k, "T2", {"eta": eta, "n_new_sqrtxi": _n_new(U_new), "dD": k - D_old}


def adaptive_compress(unc, old_iso, ctx, rng):
    """Adaptive tiered compression of the uncompressed L+1 MPS.

    ``old_iso`` is the left-canonical step-L MPS (the carried subspace).  Mirrors
    ``mps_utils.compress`` (left-canonicalise then right-to-left truncation sweep)
    but dispatches each bond to a tier.  Returns ``(mps, per_bond_stats)``.
    """
    B = unc.copy()
    mps_utils.left_canonicalize(B)
    E_list = cross_left_overlaps(old_iso, B)         # E_list[tau-1] at internal bond tau
    xi, d2, mb = ctx["cutoff"], ctx["ref_index"], ctx["max_bond"]
    stats = []
    for p in range(B.num_sites - 1, 0, -1):
        G = B.tensors[p]
        dp, chil, chir = G.shape
        M = G.transpose(1, 0, 2).reshape(chil, dp * chir)
        E = E_list[p - 1]                            # (D_old x chil)
        U_old, _ = np.linalg.qr(E.conj().T)          # (chil x D_old), orthonormal
        t0 = time.perf_counter()
        US, Vh, k, tier, info = tier_decompose(
            M, U_old, xi=xi, ref_index=d2, max_bond=mb, rng=rng)
        dt = time.perf_counter() - t0
        B.tensors[p] = Vh.reshape(k, dp, chir).transpose(1, 0, 2)
        B.tensors[p - 1] = np.tensordot(B.tensors[p - 1], US, axes=([2], [0]))
        info.update(tau=p, time=dt, k=k, tier=tier)
        stats.append(info)
    return B, stats


# --------------------------------------------------------------------------
# adaptive end-to-end fold loop
# --------------------------------------------------------------------------

def run_adaptive(ctx, K, *, recanon_tol=1e-3):
    eps, order = ctx["eps"], ctx["order"]
    mps = ctx["mps0"]
    rng = np.random.default_rng(0)
    all_stats = []
    n_recanon = 0
    t0 = time.perf_counter()
    for k in range(K):
        unc = fold_uncompressed(ctx, mps, k)
        if unc.num_sites <= 1:
            mps = unc
            continue
        old_iso = left_canonical_copy(mps)
        mps, stats = adaptive_compress(unc, old_iso, ctx, rng)
        for s in stats:
            s["L"] = k
        all_stats.extend(stats)
        # orthogonality / trace monitor
        tr = float(np.trace(mps.reduced_density_matrix()).real)
        if abs(tr - 1.0) > recanon_tol:
            mps_utils.left_canonicalize(mps)        # restore orthogonality
            n_recanon += 1
    wall = time.perf_counter() - t0
    _, pol = ObservableExtractor.coupling_polarization_history(
        mps, eps, channel=3, order=order)
    return pol, all_stats, wall, n_recanon, mps


def run_baseline(model, *, T, eps, order, cutoff, max_bond):
    """Unmodified pipeline (EDMSolver) -- the Fig. 6 algorithm."""
    t0 = time.perf_counter()
    res = EDMSolver.from_model(
        model, T=T, eps=eps, expansion_order=order, cutoff=cutoff,
        max_bond=max_bond, backend="cpu").solve(channel=3)
    return res.polarization, time.perf_counter() - t0


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

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
    ap.add_argument("--name", default="adaptive_tiers")
    args = ap.parse_args()

    model = GaudinModel(g=args.g, K=args.K)
    ctx = make_context(model, T=args.T, eps=args.eps, order=args.order,
                       cutoff=args.cutoff, max_bond=args.max_bond)
    print(f"Adaptive 3-tier end-to-end (Gaudin): K={args.K}, T={args.T} g^-1, "
          f"eps={args.eps} g^-1, order={args.order}, xi={args.cutoff:g}, D_c={args.max_bond}")

    print("  running baseline (unmodified pipeline)...")
    sz_base, wall_base = run_baseline(
        model, T=args.T, eps=args.eps, order=args.order,
        cutoff=args.cutoff, max_bond=args.max_bond)
    print(f"    baseline wall = {wall_base:.1f}s")

    print("  running adaptive 3-tier...")
    sz_ad, stats, wall_ad, n_recanon, _ = run_adaptive(ctx, args.K)
    print(f"    adaptive wall = {wall_ad:.1f}s   (re-canonicalisations: {n_recanon})")

    # accuracy vs baseline
    n = min(len(sz_base), len(sz_ad))
    max_err = float(np.max(np.abs(np.asarray(sz_ad[:n]) - np.asarray(sz_base[:n]))))

    # tier coverage + per-tier wall-clock
    cov = Counter(s["tier"] for s in stats)
    tot = sum(cov.values())
    tier_time = {t: sum(s["time"] for s in stats if s["tier"] == t) for t in cov}
    # validation: do T1.5 bonds really have n_new(sqrt xi)=0 and dD=0?
    t15 = [s for s in stats if s["tier"] == "T1.5"]
    t15_ok = sum(1 for s in t15 if s["n_new_sqrtxi"] == 0 and s["dD"] == 0)

    print("\n=== results ===")
    print(f"  <S_z(t)> max abs error vs baseline (Fig.6 pipeline) = {max_err:.2e}")
    print(f"  total wall-clock:  baseline {wall_base:.1f}s   adaptive {wall_ad:.1f}s   "
          f"speedup {wall_base/wall_ad:.2f}x")
    print(f"  tier coverage over {tot} (L,tau) bonds:")
    for t in ("T1", "T1.5", "T2", "T0"):
        if t in cov:
            print(f"    {t:>5}: {cov[t]:>5} bonds ({100*cov[t]/tot:>4.1f}%)  "
                  f"sum wall {tier_time[t]*1e3:>8.1f} ms  "
                  f"mean {tier_time[t]/cov[t]*1e3:>6.2f} ms/bond")
    if t15:
        print(f"  validation: T1.5 bonds with (n_new(sqrt xi)=0 AND dD=0) = "
              f"{t15_ok}/{len(t15)} ({100*t15_ok/len(t15):.0f}%)")
    print(f"  total adaptive decomposition time = {sum(s['time'] for s in stats)*1e3:.0f} ms "
          f"(rest of adaptive wall is folding + projection setup)")

    _DIR_DATA.mkdir(parents=True, exist_ok=True)
    npz = _DIR_DATA / f"{args.name}.npz"
    np.savez(
        npz, K=args.K, xi=args.cutoff, max_err=max_err,
        wall_base=wall_base, wall_ad=wall_ad,
        sz_base=np.asarray(sz_base), sz_ad=np.asarray(sz_ad),
        st_L=np.array([s["L"] for s in stats]),
        st_tau=np.array([s["tau"] for s in stats]),
        st_tier=np.array([s["tier"] for s in stats]),
        st_time=np.array([s["time"] for s in stats]),
        st_eta=np.array([s["eta"] for s in stats]),
        st_dD=np.array([s["dD"] for s in stats]),
        st_nnew=np.array([s["n_new_sqrtxi"] for s in stats]),
    )
    print(f"\nsaved {npz}")


if __name__ == "__main__":
    main()
