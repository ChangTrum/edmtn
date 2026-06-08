"""Validate the compressed-sensing layer (Tier 1.5) on real EDM residuals, and
demonstrate the adaptive Tier-1 / Tier-1.5 / Tier-2 dispatcher -- all offline,
with NO change to the solver pipeline.

Step 1 -- offline CS validation.
    On the true bond residual ``M^perp = (I - U U^H) M`` (ground truth from the
    full SVD), simulate rank-one measurements and recover ``M^perp`` by Singular
    Value Projection.  Sweep the measurement count ``p`` (as multiples of the
    information limit ``r(m+n)``) and report the recovery error and the resulting
    overall bond error -- with and without the picking-tensor ``phi=0`` support
    prior.  Confirms how few measurements ``p << mn`` reach the cutoff ``xi``.

Step 2 -- adaptive dispatcher (per-bond analysis).
    For each bond of a fold, estimate ``eta = ||M^perp|| / ||M||`` and the
    residual rank ``r_eff``, then dispatch:
        eta < xi/(T L)         -> Tier 1   (pure projection, zero extra cost)
        xi/(T L) < eta < delta -> Tier 1.5 (CS recovery of top-r_eff)
        eta > delta            -> Tier 2   (rSVD)
    Report the tier distribution, per-bond error, and a cost model.

FINDINGS (why this stays in examples/ and is NOT promoted to the pipeline).
    The CS recovery is correct -- it recovers ``M^perp`` to the cutoff from
    ``p ~ 2-3 r(m+n)`` rank-one measurements -- but for the Gaudin EDM it is not
    a worthwhile middle tier:
    * the picking-tensor ``phi=0`` support prior does NOT hold for the
      post-contraction bond residual (that block carries ~95% of ||M^perp||);
    * ``r_eff`` is bimodal -- ~0 at late folds (Tier 1 already suffices) or
      moderate (tens) where it matters, so ``p ~ r(m+n)`` is a large fraction of
      ``mn``, not ``<< mn``;
    * SVP recovery costs ``O(mn p)`` per iteration -> ~10^3x slower than the
      one-shot rSVD it would replace.
    Conclusion: keep a TWO-tier pipeline (Tier 1 projection / Tier 2 rSVD).  CS
    would only pay off for a bath whose residual is genuinely low-rank *and*
    non-negligible, or if the measurements were computed without forming ``M``.

Pure CPU / NumPy.

Usage
-----
    python examples/cs_recovery_poc.py
    python examples/cs_recovery_poc.py --K 24 --good-L 22 --hard-L 8 --bond -1
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cs_recovery import recover_residual  # noqa: E402
from edm_incremental import (  # noqa: E402
    bond_matrix_and_old_subspace, fold_all_L, make_context, randomized_svd,
)

from edmtn.decomposition.base import truncation_rank  # noqa: E402
from edmtn.models import GaudinModel  # noqa: E402

_DIR_DATA = Path(__file__).resolve().parent / "data"


def _median_time(fn, n=3):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), None


# --------------------------------------------------------------------------
# per-bond residual ground truth + phi=0 support prior
# --------------------------------------------------------------------------

def residual_ground_truth(ctx, mps_L, k, tau):
    """Return everything needed to study one bond's residual M^perp."""
    M, U = bond_matrix_and_old_subspace(ctx, mps_L, k, tau)
    m, ncol = M.shape
    d_phys = ctx["d_phys"]
    chir = ncol // d_phys                       # columns per phi block
    UHM = U.conj().T @ M
    Mperp = M - U @ UHM
    s = np.linalg.svd(M, compute_uv=False)
    keep = truncation_rank(s, cutoff=ctx["cutoff"], cutoff_mode="rel_ref",
                           ref_index=ctx["ref_index"])
    thresh = ctx["cutoff"] * float(s[min(ctx["ref_index"], s.size - 1)])
    sp = np.linalg.svd(Mperp, compute_uv=False)
    r_eff = int(np.count_nonzero(sp > thresh))
    normM = float(np.linalg.norm(M))
    eta = float(np.linalg.norm(Mperp)) / normM
    err_base = float(np.linalg.norm(s[keep:]) / normM) if keep < s.size else 0.0

    # picking-tensor support prior: phi=0 columns (first `chir`) should be ~zero
    mask = np.ones(ncol, dtype=bool)
    mask[:chir] = False                          # exclude the phi=0 null channel
    phi0_weight = (float(np.linalg.norm(Mperp[:, :chir]))
                   / (np.linalg.norm(Mperp) + 1e-300))
    return {
        "M": M, "U": U, "Mperp": Mperp, "normM": normM, "thresh": thresh,
        "r_eff": r_eff, "eta": eta, "err_base": err_base, "keep": keep,
        "m": m, "ncol": ncol, "chir": chir, "mask": mask, "phi0_weight": phi0_weight,
        "D_old": U.shape[1],
    }


# --------------------------------------------------------------------------
# Step 1: offline CS validation on one fold
# --------------------------------------------------------------------------

def _scan_bonds(ctx, mps_L, k, taus):
    """Per-bond ground truth (r_eff, eta, phi=0 weight) for a fold."""
    return [(tau, residual_ground_truth(ctx, mps_L, k, tau)) for tau in taus]


def step1_validate(ctx, mps, L, *, stride=3, factors=(2, 3), seed=0, cs_sweep=True):
    k = L
    taus = list(range(1, ctx["n_sites"], stride))
    scan = _scan_bonds(ctx, mps[L], k, taus)
    r_effs = np.array([g["r_eff"] for _, g in scan])
    etas = np.array([g["eta"] for _, g in scan])
    phi0 = np.array([g["phi0_weight"] for _, g in scan])

    print(f"\n=== Step 1: CS validation, fold L={L}->{L+1} ===")
    print(f"  bonds scanned: {len(scan)}   eta=||Mperp||/||M|| in "
          f"[{etas.min():.1e}, {etas.max():.1e}]   "
          f"r_eff in [{int(r_effs.min())}, {int(r_effs.max())}] (median {int(np.median(r_effs))})")
    print(f"  picking-tensor phi=0 support prior: ||Mperp[phi=0]||/||Mperp|| "
          f"median = {np.median(phi0):.2e}  ->  "
          f"{'usable' if np.median(phi0) < 1e-8 else 'NOT a zero block (prior unavailable)'}")

    nz = [(tau, g) for tau, g in scan if g["r_eff"] > 0]
    if not nz:
        print("  every bond has r_eff = 0: pure projection (Tier 1) already reaches "
              "cutoff here -- CS not needed at this fold.")
        return {"L": L, "rows": [], "r_med": 0}

    # representative bond: median r_eff among the residual-bearing bonds
    nz.sort(key=lambda tg: tg[1]["r_eff"])
    tau, gt = nz[len(nz) // 2]
    m, ncol = gt["m"], gt["ncol"]
    r = gt["r_eff"]
    info_lim = r * (m + ncol)
    if not cs_sweep:
        print(f"  (CS recovery sweep skipped; r_eff median {int(np.median(r_effs))} -> "
              f"info limit r(m+n) ~ {100*np.median(r_effs)*(m+ncol)/(m*ncol):.0f}% of mn -> "
              f"{'CS-favourable' if np.median(r_effs)*(m+ncol) < 0.15*m*ncol else 'Tier-2 territory'})")
        return {"L": L, "rows": [], "r_med": int(np.median(r_effs))}

    Mperp, M, U, normM = gt["Mperp"], gt["M"], gt["U"], gt["normM"]
    normMp = float(np.linalg.norm(Mperp))
    print(f"  representative bond tau={tau}: M {m}x{ncol} (mn={m*ncol}), D_old={gt['D_old']}")
    print(f"  Tier-1 (drop residual) bond error = eta = {gt['eta']:.2e};  "
          f"r_eff={r};  full-SVD bond error = {gt['err_base']:.2e}")
    print(f"  information limit r(m+n) = {info_lim} = {100*info_lim/(m*ncol):.0f}% of mn")

    # reference wall-clocks on this bond (what Tier-2 / baseline cost)
    t_svd, _ = _median_time(lambda: np.linalg.svd(M, full_matrices=False), n=3)
    t_rsvd, _ = _median_time(lambda: randomized_svd(Mperp, r), n=3)

    print(f"  {'p':>8} {'p/mn':>6} {'rec_err':>9} {'bond_err':>9} {'t_CS[ms]':>9} "
          f"{'t_rSVD[ms]':>10} {'t_SVD[ms]':>9}")
    print("  " + "-" * 66)
    rng = np.random.default_rng(seed)
    rows = []
    for f in factors:
        p = int(max(r + 1, np.ceil(f * info_lim)))
        t0 = time.perf_counter()
        Xh, info = recover_residual(M, U, r, p, rng, col_mask=None, n_iter=150)
        t_cs = time.perf_counter() - t0
        rec = float(np.linalg.norm(Xh - Mperp) / normMp)
        bond = float(np.linalg.norm(Mperp - Xh) / normM)
        rows.append({"p": p, "rec": rec, "bond": bond, "t_cs": t_cs})
        print(f"  {p:>8} {p/(m*ncol):>6.2f} {rec:>9.2e} {bond:>9.2e} "
              f"{t_cs*1e3:>9.0f} {t_rsvd*1e3:>10.2f} {t_svd*1e3:>9.2f}")
    print(f"  => CS reaches ~cutoff at p ~ {100*info_lim*2/(m*ncol):.0f}% of mn, but its "
          f"O(mn*p) recovery is >>{int(rows[-1]['t_cs']/max(t_rsvd,1e-9))}x slower than "
          f"one-shot rSVD (Tier 2).")
    return {"L": L, "tau": tau, "r_eff": r, "m": m, "ncol": ncol,
            "info_lim": info_lim, "t_rsvd": t_rsvd, "t_svd": t_svd, "rows": rows}


# --------------------------------------------------------------------------
# Step 2: adaptive dispatcher (per-bond analysis)
# --------------------------------------------------------------------------

def step2_dispatch(ctx, mps, L, *, stride=3, delta=5e-3):
    k = L
    n_sites = ctx["n_sites"]
    xi = ctx["cutoff"]
    eta_tier1 = xi / (ctx["n_sites"] * max(L, 1))     # xi / (T L) budget
    taus = list(range(1, n_sites, stride))
    print(f"\n=== Step 2: adaptive dispatcher, fold L={L}->{L+1} ===")
    print(f"  thresholds: Tier1 if eta < {eta_tier1:.1e} (= xi/(T L)); "
          f"Tier1.5 if < {delta:g}; else Tier2")
    print(f"{'tau':>4} {'eta':>9} {'r_eff':>6} {'D_old':>6} {'tier':>8} "
          f"{'p_cs':>7} {'p_cs/mn':>8}")
    print("-" * 56)
    counts = {"Tier1": 0, "Tier1.5": 0, "Tier2": 0}
    rows = []
    for tau in taus:
        gt = residual_ground_truth(ctx, mps[L], k, tau)
        eta, r = gt["eta"], gt["r_eff"]
        if eta < eta_tier1 or r == 0:
            tier = "Tier1"
            p_cs = 0
        elif eta < delta:
            tier = "Tier1.5"
            p_cs = int(np.ceil(3 * r * (gt["m"] + gt["ncol"])))  # no phi=0 prior
        else:
            tier = "Tier2"
            p_cs = 0
        counts[tier] += 1
        frac = (p_cs / (gt["m"] * gt["ncol"])) if p_cs else 0.0
        rows.append({"tau": tau, "eta": eta, "r_eff": r, "tier": tier, "p_cs": p_cs})
        print(f"{tau:>4} {eta:>9.2e} {r:>6} {gt['D_old']:>6} {tier:>8} "
              f"{p_cs:>7} {frac:>8.3f}")
    tot = sum(counts.values())
    print(f"\n  tier distribution over {tot} bonds: {counts}")
    print("  NOTE: the dispatcher's control logic (scaling law -> predict r_eff -> "
          "route by eta) is sound, but for this model the CS (Tier 1.5) band is NOT a\n"
          "  cheap shortcut: where eta is non-negligible, r_eff is moderate so the CS\n"
          "  measurement budget p ~ r(m+n) is a large fraction of mn AND its O(mn*p)\n"
          "  recovery is ~10^3x slower than the one-shot rSVD it would replace (Step 1).\n"
          "  => practical scheme stays TWO tiers: Tier-1 projection where eta<=xi-budget,\n"
          "  else Tier-2 rSVD.  CS only wins if measurements avoid forming M (future work).")
    return rows, counts


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--g", type=float, default=1.0)
    ap.add_argument("--K", type=int, default=24)
    ap.add_argument("--good-L", type=int, default=22, help="late/weak fold (small r)")
    ap.add_argument("--hard-L", type=int, default=8, help="mid/strong fold (larger r)")
    ap.add_argument("--T", type=float, default=3.0)
    ap.add_argument("--eps", type=float, default=0.2)
    ap.add_argument("--cutoff", type=float, default=1e-6)
    ap.add_argument("--max-bond", type=int, default=400)
    ap.add_argument("--order", type=int, default=2, choices=(1, 2))
    ap.add_argument("--name", default="cs_recovery_poc")
    args = ap.parse_args()

    need = max(args.good_L, args.hard_L) + 1
    if need > args.K:
        raise SystemExit(f"K={args.K} too small for L+1={need}")

    model = GaudinModel(g=args.g, K=args.K)
    ctx = make_context(model, T=args.T, eps=args.eps, order=args.order,
                       cutoff=args.cutoff, max_bond=args.max_bond)
    print(f"CS / Tier-1.5 validation (Gaudin): K={args.K}, T={args.T} g^-1, "
          f"eps={args.eps} g^-1, order={args.order}, xi={args.cutoff:g}, D_a={ctx['D_a']}")
    t0 = time.perf_counter()
    mps = fold_all_L(ctx, K=need)
    print(f"  (fold-all to L={need} wall {time.perf_counter() - t0:.1f}s)")

    s1 = {}
    s1[args.good_L] = step1_validate(ctx, mps, args.good_L, cs_sweep=True)
    s1[args.hard_L] = step1_validate(ctx, mps, args.hard_L, cs_sweep=False)

    rows2, counts2 = step2_dispatch(ctx, mps, args.hard_L)

    _DIR_DATA.mkdir(parents=True, exist_ok=True)
    npz = _DIR_DATA / f"{args.name}.npz"
    save = {"K": args.K, "good_L": args.good_L, "hard_L": args.hard_L}
    for L, s in s1.items():
        if s["rows"]:
            save[f"p_{L}"] = np.array([r["p"] for r in s["rows"]])
            save[f"rec_{L}"] = np.array([r["rec"] for r in s["rows"]])
            save[f"bonderr_{L}"] = np.array([r["bond"] for r in s["rows"]])
    save["disp_tau"] = np.array([r["tau"] for r in rows2])
    save["disp_eta"] = np.array([r["eta"] for r in rows2])
    save["disp_reff"] = np.array([r["r_eff"] for r in rows2])
    np.savez(npz, **save)
    print(f"\nsaved {npz}")


if __name__ == "__main__":
    main()
