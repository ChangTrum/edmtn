"""Locate the critical sub-bath count L* (where pure projection becomes viable)
and fit the scaling of the subspace increment against ``x = g_{L+1}^2 / g_L^2``.

Builds on ``validate_subspace.py``: it streams the separable fold once
(snapshotting every ``L``) and runs the per-bond subspace diagnostics on *every*
consecutive transition ``L -> L+1`` -- so the 16-21 "transition zone" is sampled
densely rather than at a few hand-picked points.

Two questions:

1. **Critical L\\*.**  The framework's Tier-1 (pure projection, no SVD) is valid
   once folding a sub-bath stops adding genuinely new directions.  We report the
   smallest ``L`` meeting several increasingly strict criteria
   (``max_resid < 1e-3``, ``< 1e-4``, ``n_new(sqrt(xi)) = 0``,
   ``n_new(xi) <= D_a``).

2. **Scaling law.**  Sec. 3.3 of the framework predicts the perturbation scales
   as ``Delta g / g_L ~ g_{L+1}^2 / (2 g_L^2)``.  We test whether the measured
   residual energy ratio and chordal distance follow a clean power law in
   ``x = g_{L+1}^2 / g_L^2`` (``g_{L+1}`` the new coupling, ``g_L`` the effective
   coupling of the first ``L`` spins), and report the fitted exponents.

Pure CPU / NumPy.

Usage
-----
    python examples/critical_L_and_scaling.py                 # K=28 default
    python examples/critical_L_and_scaling.py --K 49 --T 4 --eps 0.1 --L0 4
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from edm_incremental import analyse_transition, fold_all_L, make_context  # noqa: E402

from edmtn.models import GaudinModel  # noqa: E402

_DIR_DATA = Path(__file__).resolve().parent / "data"
_DIR_PICS = Path(__file__).resolve().parent / "pictures"


def _aggregate(rec):
    """Reduce per-bond diagnostics of one transition to scalar summaries."""
    resid = rec["resid_ratio"]
    return {
        "max_dD": int(rec["dD"].max()),
        "max_resid": float(resid.max()),
        "rms_resid": float(np.sqrt(np.mean(resid**2))),
        "max_chord": float(rec["chordal_norm"].max()),
        "max_nnew_xi2": int(rec["n_new[xi^2]"].max()),
        "max_nnew_xi": int(rec["n_new[xi]"].max()),
        "max_nnew_rtxi": int(rec["n_new[sqrt(xi)]"].max()),
    }


def _powerlaw_fit(x, y):
    """Fit ``y ~ C x^p`` on strictly-positive points; return ``(p, C, R2, n)``."""
    m = (x > 0) & (y > 0)
    if m.sum() < 3:
        return (np.nan, np.nan, np.nan, int(m.sum()))
    lx, ly = np.log(x[m]), np.log(y[m])
    p, lc = np.polyfit(lx, ly, 1)
    resid = ly - (p * lx + lc)
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return (float(p), float(np.exp(lc)), float(r2), int(m.sum()))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--g", type=float, default=1.0)
    ap.add_argument("--K", type=int, default=28, help="total bath spins")
    ap.add_argument("--L0", type=int, default=2, help="first transition L (L0 -> L0+1)")
    ap.add_argument("--T", type=float, default=3.0)
    ap.add_argument("--eps", type=float, default=0.2)
    ap.add_argument("--cutoff", type=float, default=1e-6, help="truncation precision xi")
    ap.add_argument("--max-bond", type=int, default=400)
    ap.add_argument("--order", type=int, default=2, choices=(1, 2))
    ap.add_argument("--name", default="critical_L")
    args = ap.parse_args()

    model = GaudinModel(g=args.g, K=args.K)
    xi = args.cutoff
    ctx = make_context(model, T=args.T, eps=args.eps, order=args.order,
                       cutoff=args.cutoff, max_bond=args.max_bond)
    D_a = ctx["D_a"]
    g = model.couplings  # g[k] = coupling of sub-bath k+1 (0-indexed, descending)

    print(f"Critical-L / scaling study (Gaudin): K={args.K}, T={args.T} g^-1, "
          f"eps={args.eps} g^-1, order={args.order}, xi={xi:g}, D_a={D_a}")
    t0 = time.perf_counter()
    mps = fold_all_L(ctx, verbose=True)
    print(f"  (fold-all wall {time.perf_counter() - t0:.1f}s)\n")

    Ls = list(range(max(1, args.L0), args.K))  # transition L -> L+1
    rows = []
    print(f"{'L->L+1':>8} {'x=gL1^2/gL^2':>13} {'maxdD':>6} {'max_resid':>10} "
          f"{'rms_resid':>10} {'nnew(xi)':>9} {'nnew(rtxi)':>11} {'chord/rtD':>10}")
    print("-" * 92)
    for L in Ls:
        rec = analyse_transition(mps[L], mps[L + 1], xi)
        agg = _aggregate(rec)
        gbarL = model.effective_coupling(L)
        x = float(g[L] ** 2 / gbarL**2)  # g_{L+1}=g[L] (0-indexed), g_L=gbarL
        agg.update(L=L, x=x)
        rows.append(agg)
        print(f"{L:>3}->{L+1:<3} {x:>13.3e} {agg['max_dD']:>6} {agg['max_resid']:>10.2e} "
              f"{agg['rms_resid']:>10.2e} {agg['max_nnew_xi']:>9} "
              f"{agg['max_nnew_rtxi']:>11} {agg['max_chord']:>10.2e}")

    # ---- critical L* under several criteria ------------------------------
    def first(pred):
        for r in rows:
            if pred(r):
                return r["L"]
        return None

    crits = {
        "max_resid < 1e-3": first(lambda r: r["max_resid"] < 1e-3),
        "max_resid < 1e-4": first(lambda r: r["max_resid"] < 1e-4),
        "n_new(sqrt(xi)) = 0": first(lambda r: r["max_nnew_rtxi"] == 0),
        f"n_new(xi) <= D_a={D_a}": first(lambda r: r["max_nnew_xi"] <= D_a),
        "max dD = 0": first(lambda r: r["max_dD"] == 0),
    }
    print("\nCritical L* (smallest L with the fold L->L+1 satisfying):")
    for name, Lstar in crits.items():
        print(f"  {name:<26} L* = {Lstar if Lstar is not None else 'never (in range)'}")

    # ---- scaling law: diagnostic ~ x^p -----------------------------------
    x = np.array([r["x"] for r in rows])
    print("\nScaling vs x = g_{L+1}^2 / g_L^2  (framework Sec. 3.3: Delta g/g_L ~ x/2):")
    for key, label in [("max_resid", "max residual ratio"),
                       ("rms_resid", "rms residual ratio"),
                       ("max_chord", "max chordal / sqrt(D)")]:
        y = np.array([r[key] for r in rows])
        p, C, r2, npts = _powerlaw_fit(x, y)
        print(f"  {label:<22} ~ {C:.3g} * x^{p:.3f}   (R^2={r2:.3f}, n={npts})")
    # n_new is discrete: report Pearson r of n_new(sqrt xi) with x
    y = np.array([r["max_nnew_rtxi"] for r in rows], float)
    if y.std() > 0 and x.std() > 0:
        r_pearson = float(np.corrcoef(np.log(x), np.log(y + 1))[0, 1])
        print(f"  n_new(sqrt(xi)) vs x:   corr(log(x), log(1+n_new)) = {r_pearson:.3f}")

    # ---- save + plot -----------------------------------------------------
    _DIR_DATA.mkdir(parents=True, exist_ok=True)
    keys = rows[0].keys()
    data = {k: np.array([r[k] for r in rows]) for k in keys}
    npz = _DIR_DATA / f"{args.name}.npz"
    np.savez(npz, K=args.K, xi=xi, D_a=D_a, **data)
    print(f"\nsaved {npz}")
    _plot(rows, xi, D_a, args.name)


def _plot(rows, xi, D_a, name):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plots")
        return
    L = np.array([r["L"] for r in rows])
    x = np.array([r["x"] for r in rows])
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(15, 4))

    a.semilogy(L, [r["max_resid"] for r in rows], "o-", ms=3, label="max residual")
    a.semilogy(L, [r["rms_resid"] for r in rows], "s-", ms=3, label="rms residual")
    a.semilogy(L, [r["max_chord"] for r in rows], "^-", ms=3, label="chordal/sqrt(D)")
    a.axhline(xi, color="k", ls="--", lw=1, label=r"$\xi$")
    a.set_xlabel("L"); a.set_ylabel("diagnostic"); a.set_title("subspace increment vs L")
    a.legend(fontsize=8)

    b2 = b.twinx()
    b.plot(L, [r["max_nnew_xi"] for r in rows], "o-", ms=3, color="C0",
           label=r"$n_{\rm new}(\xi)$")
    b.plot(L, [r["max_nnew_rtxi"] for r in rows], "s-", ms=3, color="C1",
           label=r"$n_{\rm new}(\sqrt{\xi})$")
    b.axhline(D_a, color="k", ls="--", lw=1, label=f"$D_a={D_a}$")
    b2.semilogy(L, x, ":", color="C3", label=r"$x=g_{L+1}^2/g_L^2$")
    b.set_xlabel("L"); b.set_ylabel("new directions"); b.set_title("new directions vs L")
    b2.set_ylabel("x"); b.legend(fontsize=8, loc="upper right")

    order = np.argsort(x)
    c.loglog(x[order], np.array([r["max_resid"] for r in rows])[order], "o", ms=4,
             label="max residual")
    c.loglog(x[order], np.array([r["max_chord"] for r in rows])[order], "^", ms=4,
             label="chordal/sqrt(D)")
    # reference slope-1 line through the cloud
    xs = x[x > 0]
    if xs.size:
        xr = np.array([xs.min(), xs.max()])
        yref = xr / xr.max() * max(r["max_resid"] for r in rows)
        c.loglog(xr, yref, "k--", lw=1, label="slope 1")
    c.set_xlabel(r"$x = g_{L+1}^2 / g_L^2$"); c.set_ylabel("diagnostic")
    c.set_title("scaling law"); c.legend(fontsize=8)

    try:
        fig.tight_layout()
    except Exception:  # noqa: BLE001  (layout is cosmetic; never fail the run on it)
        pass
    _DIR_PICS.mkdir(parents=True, exist_ok=True)
    png = _DIR_PICS / f"{name}.png"
    fig.savefig(png, dpi=130)
    print(f"saved {png}")


if __name__ == "__main__":
    main()
