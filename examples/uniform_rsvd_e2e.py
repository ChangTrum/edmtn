"""Self-contained uniform rSVD fold (L=0..K) -- is single-pass rSVD reliable
enough to drop *all* routing?

EXAMPLES-ONLY; the solver pipeline (src/) is untouched and used verbatim as the
baseline (``EDMSolver`` -> the paper's Fig. 6 algorithm).

Motivation (the question this script answers).  The 3-tier study left us with:
Tier-2 = cold rSVD (2 power iterations, the accuracy guarantor); Tier-1/1.5 =
single-pass rSVD; projection removed everywhere.  If a *single-pass* rSVD also
suffices for the Tier-2 bonds, then single-pass rSVD is reliable on its own and
we need **no tier routing, no hard-coded schedule, no probe, no residual / eta /
n_new computation at all** -- and crucially no baseline to lean on.  The scaling
law then loses its pre-emptive routing role and is demoted to an information-
theoretic statement.

So this script runs a *uniform* per-bond decomposition over the whole fold with
NO oracle and NO subspace transport, choosing the rank purely from the rSVD
spectrum with a resolution guard (grow the sketch until the computed singular
tail drops below the rel_ref cutoff -- this is what makes it deployable without a
reference run).  We compare three uniform strategies against the pipeline:

  * ``svd``  -- full SVD per bond (sanity: should reproduce the baseline exactly).
  * ``rsvd0``-- single-pass rSVD (n_iter=0)   <- the candidate.
  * ``rsvd2``-- cold rSVD       (n_iter=2)     <- the accuracy reference.

Metrics: <S_z(t)> max abs error vs pipeline; per-bond mean retries (sketch
re-grows); final bond dimensions; wall-clock and speedup.

Pure CPU / NumPy.  Defaults sized for a 16 GB laptop.

Usage
-----
    python examples/uniform_rsvd_e2e.py
    python examples/uniform_rsvd_e2e.py --K 24 --T 3 --eps 0.2 --modes rsvd0,rsvd2
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
from edm_incremental import fold_uncompressed, make_context, randomized_svd  # noqa: E402

from edmtn.decomposition.base import truncation_rank  # noqa: E402
from edmtn.evolution import mps_utils  # noqa: E402
from edmtn.observables.extractor import ObservableExtractor  # noqa: E402
from edmtn.models import GaudinModel  # noqa: E402

_DIR_DATA = Path(__file__).resolve().parent / "data"


# --------------------------------------------------------------------------
# per-bond decomposition with self-contained (oracle-free) rank selection
# --------------------------------------------------------------------------

def _rsvd_truncate(M, *, xi, ref_index, max_bond, n_iter, rng, guess):
    """Single decomposition of bond matrix ``M`` with rank chosen from the spectrum.

    Grows the rSVD sketch until the smallest *computed* singular value falls below
    the rel_ref threshold ``xi * s[ref_index]`` -- i.e. the spectrum is resolved
    past the cutoff, so no kept direction can be hiding in an un-computed tail.
    Returns ``(US, Vh, k, n_tries)`` with ``US @ Vh`` the truncated factorisation
    (absorb='left'); ``k`` is the kept rank.
    """
    m, n = M.shape
    full = min(m, n)
    R = int(np.clip(guess, ref_index + 8, full))
    n_tries = 0
    while True:
        n_tries += 1
        U, s, Vh = randomized_svd(M, R, n_iter=n_iter, rng=rng)
        if s.size == 0:
            return M[:, :0], Vh, 0, n_tries
        s_ref = s[min(ref_index, s.size - 1)]
        thresh = xi * s_ref
        # resolved if the computed tail dropped below threshold, or we hit full rank
        if s[-1] <= thresh or R >= full:
            break
        R = min(2 * R, full)
    k = truncation_rank(s, cutoff=xi, cutoff_mode="rel_ref",
                        ref_index=ref_index, max_bond=max_bond)
    return (U[:, :k] * s[:k], Vh[:k], k, n_tries)


def _svd_truncate(M, *, xi, ref_index, max_bond):
    U, s, Vh = np.linalg.svd(M, full_matrices=False)
    k = truncation_rank(s, cutoff=xi, cutoff_mode="rel_ref",
                        ref_index=ref_index, max_bond=max_bond)
    return (U[:, :k] * s[:k], Vh[:k], k, 1)


def uniform_compress(unc, ctx, mode, rng, k_memory):
    """Right-to-left truncation sweep with a *uniform* decomposition at every bond.

    Mirrors ``mps_utils.compress`` (left-canonicalise then right-to-left sweep,
    absorb='left') but swaps the full SVD for ``mode`` and selects rank from the
    spectrum.  No subspace transport, no projection, no tiers.  ``k_memory`` maps
    ``tau -> last kept rank`` to warm-start the sketch size across folds.
    """
    B = unc.copy()
    mps_utils.left_canonicalize(B)
    xi, d2, mb = ctx["cutoff"], ctx["ref_index"], ctx["max_bond"]
    n = B.num_sites
    tries = []
    for p in range(n - 1, 0, -1):
        G = B.tensors[p]
        dp, chil, chir = G.shape
        M = G.transpose(1, 0, 2).reshape(chil, dp * chir)
        if mode == "svd":
            US, Vh, k, nt = _svd_truncate(M, xi=xi, ref_index=d2, max_bond=mb)
        else:
            n_iter = 2 if mode == "rsvd2" else 0
            guess = k_memory.get(p, 16) + 16
            US, Vh, k, nt = _rsvd_truncate(M, xi=xi, ref_index=d2, max_bond=mb,
                                           n_iter=n_iter, rng=rng, guess=guess)
        k_memory[p] = k
        tries.append(nt)
        B.tensors[p] = Vh.reshape(k, dp, chir).transpose(1, 0, 2)
        B.tensors[p - 1] = np.tensordot(B.tensors[p - 1], US, axes=([2], [0]))
    return B, tries


def run_uniform(ctx, K, mode, seed=0):
    eps, order = ctx["eps"], ctx["order"]
    mps = ctx["mps0"]
    rng = np.random.default_rng(seed)
    k_memory: dict[int, int] = {}
    all_tries = []
    dmax_per_L = []                      # max bond after compressing each fold L
    t0 = time.perf_counter()
    for k in range(K):
        unc = fold_uncompressed(ctx, mps, k)
        if unc.num_sites <= 1:
            mps = unc
            dmax_per_L.append(mps.max_bond)
            continue
        mps, tries = uniform_compress(unc, ctx, mode, rng, k_memory)
        all_tries.extend(tries)
        dmax_per_L.append(mps.max_bond)
    wall = time.perf_counter() - t0
    _, pol = ObservableExtractor.coupling_polarization_history(
        mps, eps, channel=3, order=order)
    return pol, wall, all_tries, mps.max_bond, dmax_per_L


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
    ap.add_argument("--modes", default="svd,rsvd0,rsvd2")
    ap.add_argument("--seeds", default="0",
                    help="comma-separated rSVD seeds; reports worst-case error over seeds")
    ap.add_argument("--name", default="uniform_rsvd")
    args = ap.parse_args()

    model = GaudinModel(g=args.g, K=args.K)
    ctx = make_context(model, T=args.T, eps=args.eps, order=args.order,
                       cutoff=args.cutoff, max_bond=args.max_bond)
    print(f"Uniform rSVD fold (Gaudin): K={args.K}, T={args.T} g^-1, "
          f"eps={args.eps} g^-1, order={args.order}, xi={args.cutoff:g}")

    print("  baseline (unmodified pipeline)...")
    sz_base, wall_base = run_baseline(model, T=args.T, eps=args.eps, order=args.order,
                                      cutoff=args.cutoff, max_bond=args.max_bond)
    print(f"    baseline wall = {wall_base:.1f}s")

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    results = {}
    for mode in modes:
        print(f"  uniform mode = {mode} ...")
        errs, walls, dmaxes, tries_all = [], [], [], []
        sz_last = None
        traj_last = None
        for seed in seeds:
            sz, wall, tries, dmax, traj = run_uniform(ctx, args.K, mode, seed=seed)
            n = min(len(sz), len(sz_base))
            errs.append(float(np.max(np.abs(np.asarray(sz[:n]) - np.asarray(sz_base[:n])))))
            walls.append(wall)
            dmaxes.append(dmax)
            tries_all.extend(tries)
            sz_last = sz
            traj_last = traj
        err = max(errs)                      # worst case over seeds
        wall = float(np.mean(walls))
        dmax = max(dmaxes)
        mean_tries = float(np.mean(tries_all)) if tries_all else 1.0
        results[mode] = dict(sz=sz_last, wall=wall, err=err, dmax=dmax,
                             mean_tries=mean_tries, errs=errs, traj=traj_last)
        tail = f"  (over {len(seeds)} seeds, worst)" if len(seeds) > 1 else ""
        print(f"    {mode:>6}: wall {wall:5.1f}s  ({wall_base/wall:4.2f}x)  "
              f"max|d<Sz>|={err:.2e}  Dmax={dmax}  mean sketch tries={mean_tries:.2f}{tail}")
        print(f"           Dmax per L = {traj_last}")

    def _plateau_L(traj):
        """First fold index L (1-based) after which the bond never grows again."""
        for i in range(len(traj) - 1):
            if all(traj[j + 1] <= traj[j] for j in range(i, len(traj) - 1)):
                return i + 1
        return len(traj)

    print("\n=== summary (vs unmodified pipeline) ===")
    print(f"  baseline: {wall_base:.1f}s")
    for mode, r in results.items():
        print(f"  {mode:>6}: {r['wall']:5.1f}s ({wall_base/r['wall']:4.2f}x)  "
              f"max|d<Sz>|={r['err']:.2e}  Dmax={r['dmax']}  "
              f"growth stops at L={_plateau_L(r['traj'])}")
    if "svd" in results:
        base_traj = results["svd"]["traj"]
        print(f"\n  bond-growth trajectories (Dmax per L):")
        print(f"    {'svd':>6}: {base_traj}")
        for mode, r in results.items():
            if mode == "svd":
                continue
            delta = [a - b for a, b in zip(r["traj"], base_traj)]
            print(f"    {mode:>6}: {r['traj']}")
            print(f"    {'Δ':>6}: {delta}   (rSVD minus full-SVD; >0 = over-retain)")
    if "rsvd0" in results:
        e = results["rsvd0"]["err"]
        verdict = ("RELIABLE: single-pass alone holds accuracy -> no routing needed"
                   if e < 1e-4 else
                   "NOT reliable on its own at this xi -- power iterations still pay")
        print(f"\n  verdict (single-pass rSVD, max err {e:.2e}): {verdict}")

    _DIR_DATA.mkdir(parents=True, exist_ok=True)
    npz = _DIR_DATA / f"{args.name}.npz"
    save = {"K": args.K, "xi": args.cutoff, "wall_base": wall_base,
            "sz_base": np.asarray(sz_base)}
    for mode, r in results.items():
        save[f"sz_{mode}"] = np.asarray(r["sz"])
        save[f"wall_{mode}"] = r["wall"]
        save[f"err_{mode}"] = r["err"]
        save[f"dmax_{mode}"] = r["dmax"]
        save[f"traj_{mode}"] = np.asarray(r["traj"])
    np.savez(npz, **save)
    print(f"\nsaved {npz}")


if __name__ == "__main__":
    main()
