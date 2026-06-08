"""Validate the *subspace-increment* hypothesis behind the proposed
"已知子空间 + 参数增量分解 + CS 恢复" framework, directly on the running EDM-MPS.

Background
----------
For a separable bath the EDM is built by folding sub-baths in one at a time
(paper Eq. 21 / Fig. 5c).  Folding sub-bath ``L+1`` is a single MPO * MPS
contraction along the time axis followed by an SVD recompression -- exactly the
``O((D D_a)^3)`` step the framework wants to avoid.

The framework's central claim ("假设1") is that folding sub-bath ``L+1`` barely
rotates the left singular subspace already spanned at step ``L``: at every bond
``tau`` the genuinely *new* directions number at most ``D_a`` (the sub-bath MPO
lateral bond, ``= 4`` for a spin-1/2 Gaudin bath).  If true, the bond matrix of
step ``L+1`` decomposes as

    M_tau  =  M_tau^||  +  M_tau^perp ,
              (in V^(L)_tau)   (residual, rank <= D_a)

so ``M^||`` needs no SVD (just a projection) and only the small residual must be
recovered -- the opening for compressed sensing.

This script measures that claim with **no CS / ML machinery whatsoever** -- only
the projection + principal-angle analysis the framework itself prescribes for
verifying 假设1.  For each ``L -> L+1`` transition it takes the two *already
compressed* EDM-MPS (the real output of the solver, same truncation ``xi`` the
paper uses) and compares them bond by bond.

For each internal bond ``tau`` it reports three diagnostics, all built from the
principal angles ``theta_i`` between the left singular subspaces
``V^(L)_tau`` (dim ``D^(L)_tau``) and ``V^(L+1)_tau`` (dim ``D^(L+1)_tau``):

1. **Residual energy ratio** -- ``||M^perp||_F / ||M||_F`` for the step-``L+1``
   bond matrix, i.e. the fraction of the new signal's weight that lies *outside*
   the old subspace.  Computed exactly (energy-weighted) as

       ratio^2 = 1 - Tr[E rho_left E^H] / Tr[rho_left],

   with ``E = Q_L^H Q_{L+1}`` the cross-overlap of the two left isometries and
   ``rho_left`` the step-``L+1`` bond density matrix.  Compared against ``xi``.

2. **New-direction count** ``n_new(delta) = #{ new directions with cos theta_i
   < 1 - delta }`` for ``delta in {xi^2, xi, sqrt(xi)}``.  The framework predicts
   ``n_new(xi) <= D_a = 4`` -- the headline test.

3. **Global chordal distance** ``sqrt(sum_i sin^2 theta_i) / sqrt(D^(L)_tau)``,
   the normalised Grassmann distance between the old and new left subspaces,
   compared against ``xi``.

A built-in ``--self-check`` compares a step with *itself* (identical MPS); all
three diagnostics must then be ~0 (n_new = 0), confirming the estimator.

What to expect
--------------
The hypothesis is **regime-dependent**, exactly as the framework's own Sec. 3.3
argues (``Delta g / g_L ~ g_{L+1}^2 / (2 g_L^2) -> 0`` as ``L -> K``).  Folding a
*strong* early sub-bath (small ``L``) rotates the left subspace substantially --
the literal ``n_new(xi) <= D_a`` test fails and the residual is a few percent.
As ``L`` grows the added sub-baths weaken, and the residual energy, the chordal
distance and the new-direction count all collapse: by ``L ~ K`` the residual
falls below ~1e-3, ``n_new(sqrt(xi))`` reaches 0, and ``n_new(xi)`` approaches
``D_a``.  Run a spread of transitions (early/mid/late) to see the crossover --
that crossover, not a single pass/fail, is the real finding.

Everything is pure CPU / NumPy -- no CuPy, no MKL needed.

Usage
-----
    python examples/validate_subspace.py                 # quick CPU default
    python examples/validate_subspace.py --self-check
    python examples/validate_subspace.py --K 49 --transitions 10,20,30 \
        --T 8 --eps 0.05 --cutoff 1e-6 --max-bond 400
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from edm_incremental import analyse_transition  # noqa: E402  (sibling example module)

from edmtn.driver import EDMSolver  # noqa: E402
from edmtn.kernels.separable_mpo import SeparableKernelEngine  # noqa: E402
from edmtn.models import GaudinModel  # noqa: E402

_DIR_DATA = Path(__file__).resolve().parent / "data"
_DIR_PICS = Path(__file__).resolve().parent / "pictures"


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def _solve_mps(model, L, *, T, eps, order, cutoff, max_bond):
    """Compressed EDM-MPS after folding the first ``L`` sub-baths."""
    res = EDMSolver.from_model(
        model, T=T, eps=eps, expansion_order=order,
        cutoff=cutoff, max_bond=max_bond, sub_baths=L, backend="cpu",
    ).solve(channel=3)
    return res.mps


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--g", type=float, default=1.0)
    ap.add_argument("--K", type=int, default=24, help="total bath spins")
    ap.add_argument("--transitions", default="3,8,15,22",
                    help="comma list of L; each measures the fold L -> L+1 "
                         "(span early/mid/late to see the crossover)")
    ap.add_argument("--T", type=float, default=4.0, help="total time in g^-1")
    ap.add_argument("--eps", type=float, default=0.2, help="time step in g^-1")
    ap.add_argument("--cutoff", type=float, default=1e-6,
                    help="SVD truncation precision xi (drives the delta thresholds)")
    ap.add_argument("--max-bond", type=int, default=400, help="hard bond-dim cap D_c")
    ap.add_argument("--order", type=int, default=2, choices=(1, 2))
    ap.add_argument("--self-check", action="store_true",
                    help="compare each step with itself (must give n_new=0, resid~0)")
    ap.add_argument("--name", default="subspace", help="output .npz base name")
    ap.add_argument("--quick", action="store_true",
                    help="tiny preview: K=8, transitions=2,4, T=3, eps=0.2")
    args = ap.parse_args()

    if args.quick:
        K, transitions, T, eps = 8, [2, 4], 3.0, 0.2
    else:
        K, T, eps = args.K, args.T, args.eps
        transitions = [int(s) for s in args.transitions.split(",")]
    cutoff, max_bond, order = args.cutoff, args.max_bond, args.order

    transitions = [L for L in transitions if 1 <= L < K]
    if not transitions:
        raise SystemExit("no valid transitions (need 1 <= L < K)")

    model = GaudinModel(g=args.g, K=K)
    D_a = SeparableKernelEngine.from_model(model, T=T, eps=eps).corr.bond_dim  # = 4
    xi = cutoff

    print(f"Subspace-increment validation (Gaudin): K={K}, T={T} g^-1, eps={eps} g^-1, "
          f"order={order}, xi={xi:g}, D_c={max_bond}, D_a={D_a}")
    print(f"thresholds delta: xi^2={xi*xi:g}, xi={xi:g}, sqrt(xi)={np.sqrt(xi):g}")
    if args.self_check:
        print("** SELF-CHECK MODE: comparing each step with itself **")
    print()

    # solve each needed sub-bath count once
    need = set(transitions)
    if not args.self_check:
        need |= {L + 1 for L in transitions}
    need = sorted(need)
    cache, t0 = {}, time.perf_counter()
    for L in need:
        cache[L] = _solve_mps(model, L, T=T, eps=eps, order=order,
                              cutoff=cutoff, max_bond=max_bond)
        print(f"  solved L={L:>3}: sites={cache[L].num_sites}, "
              f"Dmax={cache[L].max_bond}")
    print(f"  (solve wall {time.perf_counter() - t0:.1f}s)\n")

    hdr = (f"{'L->L+1':>8} {'gL':>6} {'D^L':>5} {'D^L+1':>6} {'maxdD':>6} "
           f"{'max_resid':>10} {'n_new(xi^2)':>11} {'n_new(xi)':>9} "
           f"{'n_new(rtxi)':>11} {'max_chord/rtD':>13}")
    print(hdr)
    print("-" * len(hdr))

    all_data = {}
    for L in transitions:
        mps_L = cache[L]
        mps_L1 = cache[L] if args.self_check else cache[L + 1]
        rec = analyse_transition(mps_L, mps_L1, xi)

        gL = model.effective_coupling(min(L + 1, K))
        scaled_t = (mps_L.num_sites - rec["tau"]) * (eps / order) * gL
        rec["scaled_t"] = scaled_t

        nn_xi = rec["n_new[xi]"]
        verdict_nnew = int(nn_xi.max()) <= D_a
        verdict_chord = float(rec["chordal_norm"].max()) < xi
        verdict_resid = float(rec["resid_ratio"].max()) < xi

        print(f"{L:>3}->{L+1:<3} {gL:>6.3f} {int(rec['DL'].max()):>5} "
              f"{int(rec['DL1'].max()):>6} {int(rec['dD'].max()):>6} "
              f"{rec['resid_ratio'].max():>10.2e} "
              f"{int(rec['n_new[xi^2]'].max()):>11} {int(nn_xi.max()):>9} "
              f"{int(rec['n_new[sqrt(xi)]'].max()):>11} "
              f"{rec['chordal_norm'].max():>13.2e}"
              f"  [{'PASS' if verdict_nnew else 'FAIL'} n_new "
              f"{'<=' if verdict_nnew else '>'} D_a]")

        for key, arr in rec.items():
            all_data[f"{L}_{key}"] = arr

    print()
    print("Interpretation:")
    print(f"  * n_new(xi) <= D_a={D_a}  -> per-fold new directions are bounded by the "
          "sub-bath MPO bond (假设1 holds): projection + low-rank recovery is valid.")
    print("  * max_resid << 1          -> the new sub-bath's signal lives almost "
          "entirely in the old subspace (residual is weak -> good CS regime).")
    print("  * max_chord/sqrt(D) small -> the old left subspace is barely rotated "
          "(high overlap V^(L) ~ V^(L+1)).")

    _DIR_DATA.mkdir(parents=True, exist_ok=True)
    npz = _DIR_DATA / f"{args.name}.npz"
    np.savez(npz, transitions=np.asarray(transitions), K=K, xi=xi, D_a=D_a, **all_data)
    print(f"\nsaved {npz}")

    _plot(transitions, all_data, xi, D_a, args.name)


def _plot(transitions, data, xi, D_a, name):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plots")
        return
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(15, 4))
    for L in transitions:
        t = data[f"{L}_scaled_t"]
        order_idx = np.argsort(t)
        ts = t[order_idx]
        a.plot(ts, data[f"{L}_n_new[xi]"][order_idx], marker=".", ms=3, label=f"L={L}")
        b.plot(ts, data[f"{L}_resid_ratio"][order_idx], marker=".", ms=3, label=f"L={L}")
        c.plot(ts, data[f"{L}_chordal_norm"][order_idx], marker=".", ms=3, label=f"L={L}")
    a.axhline(D_a, color="k", ls="--", lw=1, label=f"$D_a={D_a}$")
    a.set_ylabel(r"$n_{\rm new}(\xi)$"); a.set_title("new directions per fold")
    b.axhline(xi, color="k", ls="--", lw=1, label=r"$\xi$")
    b.set_yscale("log"); b.set_ylabel(r"$\|M^\perp\|/\|M\|$")
    b.set_title("residual energy ratio")
    c.axhline(xi, color="k", ls="--", lw=1, label=r"$\xi$")
    c.set_yscale("log"); c.set_ylabel(r"$\sqrt{\sum\sin^2\theta_i}/\sqrt{D^{(L)}}$")
    c.set_title("chordal distance (normalised)")
    for ax in (a, b, c):
        ax.set_xlabel(r"$\bar g_{L+1}\, t$"); ax.legend(fontsize=8)
    fig.tight_layout()
    _DIR_PICS.mkdir(parents=True, exist_ok=True)
    png = _DIR_PICS / f"{name}.png"
    fig.savefig(png, dpi=130)
    print(f"saved {png}")


if __name__ == "__main__":
    main()
