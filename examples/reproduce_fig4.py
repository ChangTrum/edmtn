"""Reproduce Fig. 4 of Chen & Liu, *Polynomial complexity of open quantum system
problems* (arXiv:2509.00424), at (near) publication settings.

Fig. 4a: spin polarization <S_z(t)> vs mu*t for several spin-bath couplings J0.
Fig. 4b: maximum EDM bond dimension vs mu*t.

Parameters follow the paper: Ohmic bath, omega_c = 5 mu, eps = 0.01 mu^-1,
truncation precision xi = 1e-5, second-order time-step expansion.

This is intentionally *not* a pytest -- the EDM algorithm is O(N^2) in the number
of steps, so the full T = 17 mu^-1 run at eps = 0.01 takes a long time
(thousands of steps).  Use ``--quick`` for a coarse, fast preview, and ``--T`` /
``--eps`` to trade cost against fidelity.  Results are written to a .npz and, if
matplotlib is available, plotted to PNGs.

Usage
-----
    python examples/reproduce_fig4.py                 # publication-ish (slow)
    python examples/reproduce_fig4.py --quick         # fast preview
    python examples/reproduce_fig4.py --T 8 --eps 0.02
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from edmtn.driver import EDMSolver
from edmtn.models import SpinBosonModel

COUPLINGS = (0.1, 0.3, 0.5, 0.7, 1.0, 1.2)

_DIR_DATA = Path(__file__).resolve().parent / "data"
_DIR_PICS = Path(__file__).resolve().parent / "pictures"


def run(J0, *, T, eps, omega_c=5.0, mu=1.0, cutoff=1e-5, order=2):
    model = SpinBosonModel(J0=J0, omega_c=omega_c, mu=mu)
    t0 = time.perf_counter()
    res = EDMSolver.from_model(
        model, T=T, eps=eps, expansion_order=order, cutoff=cutoff
    ).solve()
    return res, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--T", type=float, default=17.0, help="total time in mu^-1")
    ap.add_argument("--eps", type=float, default=0.01, help="time step in mu^-1")
    ap.add_argument(
        "--cutoff", type=float, default=1e-5, help="SVD truncation precision"
    )
    ap.add_argument("--order", type=int, default=2, choices=(1, 2))
    ap.add_argument(
        "--quick", action="store_true", help="coarse fast preview (T=8, eps=0.05)"
    )
    ap.add_argument("--name", default="fig4", help="output file base name")
    args = ap.parse_args()

    _DIR_DATA.mkdir(parents=True, exist_ok=True)
    _DIR_PICS.mkdir(parents=True, exist_ok=True)

    T, eps = (8.0, 0.05) if args.quick else (args.T, args.eps)
    print(
        f"Fig. 4 reproduction: T={T} mu^-1, eps={eps} mu^-1, "
        f"order={args.order}, cutoff={args.cutoff}"
    )
    print(f"{'J0':>5} {'steps':>6} {'<Sz(end)>':>10} {'Dmax':>6} {'wall[s]':>8}")

    data = {}
    for J0 in COUPLINGS:
        res, wall = run(J0, T=T, eps=eps, cutoff=args.cutoff, order=args.order)
        data[f"t_{J0}"] = res.times
        data[f"sz_{J0}"] = res.polarization
        data[f"D_{J0}"] = np.asarray(res.bond_dims)
        print(
            f"{J0:>5} {len(res.times):>6} {res.polarization[-1]:>10.4f} "
            f"{res.max_bond:>6} {wall:>8.1f}"
        )

    npz_path = _DIR_DATA / f"{args.name}.npz"
    np.savez(npz_path, **data)
    print(f"saved {npz_path}")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plots")
        return

    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
    for J0 in COUPLINGS:
        a.plot(data[f"t_{J0}"], data[f"sz_{J0}"], label=f"J0={J0}")
        b.plot(data[f"t_{J0}"], data[f"D_{J0}"], label=f"J0={J0}")
    a.set_xlabel(r"$\mu t$")
    a.set_ylabel(r"$\langle S_z(t)\rangle$")
    a.legend(fontsize=8)
    a.set_title("Fig. 4a")
    b.set_xlabel(r"$\mu t$")
    b.set_ylabel(r"$D_{\max}$")
    b.legend(fontsize=8)
    b.set_title("Fig. 4b")
    fig.tight_layout()

    png_path = _DIR_PICS / f"{args.name}.png"
    fig.savefig(png_path, dpi=130)
    print(f"saved {png_path}")


if __name__ == "__main__":
    main()
