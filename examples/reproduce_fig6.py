"""Reproduce Fig. 6 of Chen & Liu, *Polynomial complexity of open quantum system
problems* (arXiv:2509.00424): the Gaudin (central-spin) model.

Fig. 6a: central-spin polarization <S_z> vs the scaled time g_L * t, with the
         first L (strongest) bath spins included, for several L.
Fig. 6b: EDM bond dimension D_t vs g_L * t.

The striking result is that both curves nearly collapse onto a *universal* curve
when time is scaled by the effective coupling g_L = sqrt(sum_{k<=L} g_k^2): adding
more (weaker) bath spins barely changes the dynamics or the bond growth.

Parameters follow the paper: N = 49 spins, linearly decreasing couplings,
infinite-temperature bath, eps = 0.03 g^-1, T = 15 g^-1, truncation xi = 1e-6,
hard cutoff D_c = 400, second-order expansion.

This is intentionally *not* a pytest -- at publication settings it is a heavy
O(K * N^2) computation.  Use ``--quick`` for a fast preview, and ``--K`` / ``--T``
/ ``--eps`` / ``--max-bond`` to trade cost against fidelity.  Results are written
to a .npz and, if matplotlib is available, plotted to PNGs.

Usage
-----
    python examples/reproduce_fig6.py                 # publication-ish (slow)
    python examples/reproduce_fig6.py --quick         # fast preview
    python examples/reproduce_fig6.py --K 30 --Ls 10,20,30 --T 10 --eps 0.05
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from edmtn.driver import EDMSolver
from edmtn.models import GaudinModel

_DIR_DATA = Path(__file__).resolve().parent / "data"
_DIR_PICS = Path(__file__).resolve().parent / "pictures"


def run(model, L, *, T, eps, cutoff, max_bond, order, backend):
    """Solve with the first ``L`` sub-baths folded in; return (result, wall)."""
    t0 = time.perf_counter()
    res = EDMSolver.from_model(
        model, T=T, eps=eps, expansion_order=order,
        cutoff=cutoff, max_bond=max_bond, sub_baths=L, backend=backend,
    ).solve(channel=3)  # channel 3 = S_z
    return res, time.perf_counter() - t0


def _bond_trajectory(res, eps, order, gL):
    """(scaled time, D) for the per-time EDM bond dimension D_t."""
    D = np.asarray(res.mps.bond_dims)[::-1]            # earliest-time first
    t = (np.arange(1, len(D) + 1) * eps / order) * gL  # sub-step -> scaled time
    return t, D


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--g", type=float, default=1.0, help="base coupling constant")
    ap.add_argument("--K", type=int, default=49, help="total number of bath spins")
    ap.add_argument("--Ls", default="10,20,30,40,49", help="comma list of L (first-L spins)")
    ap.add_argument("--T", type=float, default=15.0, help="total time in g^-1")
    ap.add_argument("--eps", type=float, default=0.03, help="time step in g^-1")
    ap.add_argument("--cutoff", type=float, default=1e-6, help="SVD truncation precision")
    ap.add_argument("--max-bond", type=int, default=400, help="hard bond-dim cutoff D_c")
    ap.add_argument("--order", type=int, default=2, choices=(1, 2))
    ap.add_argument("--backend", default="auto", choices=("auto", "cpu", "gpu"),
                    help="compute backend ('auto' uses the GPU for Gaudin)")
    ap.add_argument("--quick", action="store_true",
                    help="fast preview (K=20, Ls=5,10,20, T=6, eps=0.1, D_c=100)")
    ap.add_argument("--name", default="fig6", help="output file base name")
    args = ap.parse_args()

    _DIR_DATA.mkdir(parents=True, exist_ok=True)
    _DIR_PICS.mkdir(parents=True, exist_ok=True)

    if args.quick:
        K, Ls, T, eps, cutoff, max_bond = 20, [5, 10, 20], 6.0, 0.1, 1e-5, 100
    else:
        K, T, eps, cutoff, max_bond = args.K, args.T, args.eps, args.cutoff, args.max_bond
        Ls = [int(s) for s in args.Ls.split(",")]
    Ls = [L for L in Ls if L <= K]

    model = GaudinModel(g=args.g, K=K)
    print(f"Fig. 6 reproduction: K={K}, Ls={Ls}, T={T} g^-1, eps={eps} g^-1, "
          f"order={args.order}, cutoff={cutoff}, D_c={max_bond}, backend={args.backend}")
    print(f"{'L':>4} {'g_L':>7} {'steps':>6} {'<Sz(end)>':>10} {'Dmax':>6} "
          f"{'wall[s]':>8} {'backend':>14}")

    data = {}
    for L in Ls:
        res, wall = run(model, L, T=T, eps=eps, cutoff=cutoff,
                        max_bond=max_bond, order=args.order, backend=args.backend)
        gL = model.effective_coupling(L)
        tD, D = _bond_trajectory(res, eps, args.order, gL)
        data[f"t_{L}"] = res.times * gL          # Fig. 6a x-axis (scaled time)
        data[f"sz_{L}"] = res.polarization
        data[f"tD_{L}"] = tD                      # Fig. 6b x-axis
        data[f"D_{L}"] = D
        print(f"{L:>4} {gL:>7.3f} {res.times.size:>6} {res.polarization[-1]:>10.4f} "
              f"{int(D.max()):>6} {wall:>8.1f} {res.backend:>14}")

    npz_path = _DIR_DATA / f"{args.name}.npz"
    np.savez(npz_path, Ls=np.asarray(Ls), **data)
    print(f"saved {npz_path}")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plots")
        return

    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
    for L in Ls:
        a.plot(data[f"t_{L}"], data[f"sz_{L}"], label=f"L={L}")
        b.plot(data[f"tD_{L}"], data[f"D_{L}"], label=f"L={L}")
    a.set_xlabel(r"$\bar g_L\, t$")
    a.set_ylabel(r"$\langle S_z\rangle$")
    a.legend(fontsize=8)
    a.set_title("Fig. 6a")
    b.set_xlabel(r"$\bar g_L\, t$")
    b.set_ylabel(r"$D_t$")
    b.legend(fontsize=8)
    b.set_title("Fig. 6b")
    fig.tight_layout()

    png_path = _DIR_PICS / f"{args.name}.png"
    fig.savefig(png_path, dpi=130)
    print(f"saved {png_path}")


if __name__ == "__main__":
    main()
