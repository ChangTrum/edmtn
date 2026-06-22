"""Adaptive 3-tier fold with a *hard-coded* tier schedule and projection-free
Tier-1.5/2 -- testing the two improvements flagged in docs section 12.

EXAMPLES-ONLY; pipeline (src/) untouched and used verbatim as the baseline.

Changes vs ``adaptive_tiers_e2e.py``:

1. **Hard-coded tier (cheap-predictor assumption).**  We already know, for this
   parameter setting, which tier each ``(L, tau)`` bond takes -- so we drop the
   per-bond decision (the probe + n_new/dD computation, which cost as much as the
   compression) and look the tier up from an oracle built once, offline.
2. **Projection removed from Tier 1.5 / Tier 2.**  Instead of project + rSVD of
   the residual, run rSVD directly on the bond matrix ``M`` (single-pass for 1.5,
   cold for 2).  These bonds then need no ``U_L`` and hence **no per-fold subspace
   transport** -- the dominant overhead in section 12.  Per-bond FLOPs may rise a
   little, but the end-to-end wall-clock should drop.
3. **Tier 1 stays pure projection** (no bond growth, gauge unchanged).  It is the
   only tier that needs ``U_L``; ``--t1 project`` computes it via the transport,
   ``--t1 rsvd`` instead uses a single-pass rSVD (no transport) -- so we can test
   experimentally whether Tier-1's projection / streaming carry is even necessary.

Metrics (vs the unmodified pipeline): <S_z(t)> max abs error, total wall-clock and
speedup, per-tier wall-clock, transport time, tier coverage.

Pure CPU / NumPy.  Sized for a 16 GB laptop.

Usage
-----
    python examples/adaptive_tiers_hardcoded.py                 # both T1 modes
    python examples/adaptive_tiers_hardcoded.py --K 24 --T 3 --eps 0.2
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

from adaptive_tiers_e2e import run_adaptive, run_baseline  # noqa: E402
from edm_incremental import (  # noqa: E402
    cross_left_overlaps, fold_uncompressed, left_canonical_copy, make_context,
    randomized_svd,
)

from edmtn.decomposition.base import truncation_rank  # noqa: E402
from edmtn.evolution import mps_utils  # noqa: E402
from edmtn.observables.extractor import ObservableExtractor  # noqa: E402
from edmtn.models import GaudinModel  # noqa: E402

_DIR_DATA = Path(__file__).resolve().parent / "data"


def build_oracle(ctx, K):
    """Run the faithful adaptive fold once (offline) to record (L,tau)->(tier,k)."""
    _, stats, _, _, _ = run_adaptive(ctx, K)
    return {(s["L"], s["tau"]): (s["tier"], int(s["k"])) for s in stats}


def hardcoded_compress(unc, ctx, lookup, L, t1_mode, old_iso, rng):
    """Compress with a hard-coded per-bond tier; projection-free 1.5/2.

    Mirrors the baseline sweep (left-canonicalise, then right-to-left truncation)
    but replaces each bond's full SVD with the hard-coded tier's decomposition.
    ``transport_time`` (cross-overlap + old canonicalisation) is returned for
    accounting; it is incurred only for ``--t1 project`` folds that contain a
    Tier-1 bond.
    """
    B = unc.copy()
    mps_utils.left_canonicalize(B)
    n = B.num_sites
    transport_time = 0.0
    E_list = None
    if t1_mode == "project" and any(lookup.get((L, p), ("", 0))[0] == "T1"
                                    for p in range(1, n)):
        t0 = time.perf_counter()
        E_list = cross_left_overlaps(old_iso, B)
        transport_time = time.perf_counter() - t0

    xi, d2, mb = ctx["cutoff"], ctx["ref_index"], ctx["max_bond"]
    stats = []
    for p in range(n - 1, 0, -1):
        G = B.tensors[p]
        dp, chil, chir = G.shape
        M = G.transpose(1, 0, 2).reshape(chil, dp * chir)
        tier, k0 = lookup.get((L, p), ("T2", min(chil, dp * chir)))
        k = max(1, min(k0, chil, dp * chir))
        t0 = time.perf_counter()
        if tier == "T0":                                        # full-SVD fallback (early fold)
            U, s, Vh = np.linalg.svd(M, full_matrices=False)
            kk = min(k, s.size)
            US, Vhk = U[:, :kk] * s[:kk], Vh[:kk]
        elif tier == "T1" and t1_mode == "project":
            U_old, _ = np.linalg.qr(E_list[p - 1].conj().T)     # (chil x D_old)
            Bp = U_old.conj().T @ M
            Ub, s, Vh = np.linalg.svd(Bp, full_matrices=False)
            kk = min(k, s.size)
            US, Vhk = (U_old @ Ub[:, :kk]) * s[:kk], Vh[:kk]
        else:
            n_iter = 2 if tier == "T2" else 0                   # T1/T1.5 single-pass
            U, s, Vh = randomized_svd(M, k, n_iter=n_iter, rng=rng)
            kk = U.shape[1]
            US, Vhk = U * s, Vh
        dt = time.perf_counter() - t0
        B.tensors[p] = Vhk.reshape(kk, dp, chir).transpose(1, 0, 2)
        B.tensors[p - 1] = np.tensordot(B.tensors[p - 1], US, axes=([2], [0]))
        stats.append({"L": L, "tau": p, "tier": tier, "time": dt})
    return B, stats, transport_time


def run_hardcoded(ctx, K, lookup, t1_mode):
    eps, order = ctx["eps"], ctx["order"]
    mps = ctx["mps0"]
    rng = np.random.default_rng(0)
    all_stats = []
    transport_total = 0.0
    t0 = time.perf_counter()
    for k in range(K):
        unc = fold_uncompressed(ctx, mps, k)
        if unc.num_sites <= 1:
            mps = unc
            continue
        old_iso = left_canonical_copy(mps) if t1_mode == "project" else None
        mps, stats, tt = hardcoded_compress(unc, ctx, lookup, k, t1_mode, old_iso, rng)
        transport_total += tt
        all_stats.extend(stats)
    wall = time.perf_counter() - t0
    _, pol = ObservableExtractor.coupling_polarization_history(
        mps, eps, channel=3, order=order)
    return pol, all_stats, wall, transport_total


def _report(name, sz, stats, wall, transport, sz_base, wall_base):
    n = min(len(sz), len(sz_base))
    max_err = float(np.max(np.abs(np.asarray(sz[:n]) - np.asarray(sz_base[:n]))))
    cov = Counter(s["tier"] for s in stats)
    tot = sum(cov.values())
    tier_time = {t: sum(s["time"] for s in stats if s["tier"] == t) for t in cov}
    print(f"\n=== {name} ===")
    print(f"  <S_z(t)> max abs error vs baseline = {max_err:.2e}")
    print(f"  wall-clock {wall:.1f}s   speedup vs baseline {wall_base/wall:.2f}x   "
          f"(transport {transport:.1f}s)")
    for t in ("T1", "T1.5", "T2"):
        if t in cov:
            print(f"    {t:>5}: {cov[t]:>5} bonds ({100*cov[t]/tot:>4.1f}%)  "
                  f"decomp {tier_time[t]*1e3:>8.1f} ms  "
                  f"mean {tier_time[t]/cov[t]*1e3:>6.2f} ms/bond")
    return max_err


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
    ap.add_argument("--t1", default="both", choices=("project", "rsvd", "both"))
    ap.add_argument("--name", default="adaptive_hardcoded")
    args = ap.parse_args()

    model = GaudinModel(g=args.g, K=args.K)
    ctx = make_context(model, T=args.T, eps=args.eps, order=args.order,
                       cutoff=args.cutoff, max_bond=args.max_bond)
    print(f"Hard-coded adaptive 3-tier (Gaudin): K={args.K}, T={args.T} g^-1, "
          f"eps={args.eps} g^-1, order={args.order}, xi={args.cutoff:g}")

    print("  baseline (unmodified pipeline)...")
    sz_base, wall_base = run_baseline(model, T=args.T, eps=args.eps, order=args.order,
                                      cutoff=args.cutoff, max_bond=args.max_bond)
    print(f"    baseline wall = {wall_base:.1f}s")

    print("  building tier oracle (offline; the cheap-predictor assumption)...")
    t0 = time.perf_counter()
    lookup = build_oracle(ctx, args.K)
    print(f"    oracle built in {time.perf_counter()-t0:.1f}s (not part of deployment cost)")

    modes = ["project", "rsvd"] if args.t1 == "both" else [args.t1]
    results = {}
    for mode in modes:
        print(f"  running hard-coded variant (T1 mode = {mode})...")
        sz, stats, wall, tt = run_hardcoded(ctx, args.K, lookup, mode)
        err = _report(f"hard-coded, T1={mode}", sz, stats, wall, tt, sz_base, wall_base)
        results[mode] = {"sz": sz, "wall": wall, "transport": tt, "max_err": err}

    print("\n=== summary ===")
    print(f"  baseline (pipeline): {wall_base:.1f}s")
    for mode, r in results.items():
        print(f"  hard-coded T1={mode:>7}: {r['wall']:.1f}s  "
              f"({wall_base/r['wall']:.2f}x)  max|d<Sz>|={r['max_err']:.2e}")

    _DIR_DATA.mkdir(parents=True, exist_ok=True)
    npz = _DIR_DATA / f"{args.name}.npz"
    save = {"K": args.K, "xi": args.cutoff, "wall_base": wall_base,
            "sz_base": np.asarray(sz_base)}
    for mode, r in results.items():
        save[f"sz_{mode}"] = np.asarray(r["sz"])
        save[f"wall_{mode}"] = r["wall"]
        save[f"transport_{mode}"] = r["transport"]
        save[f"max_err_{mode}"] = r["max_err"]
    np.savez(npz, **save)
    print(f"\nsaved {npz}")


if __name__ == "__main__":
    main()
