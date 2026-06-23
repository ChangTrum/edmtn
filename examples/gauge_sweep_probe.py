"""Cheap go/no-go probe: can a single R->L truncation sweep with a carried gauge
matrix replace the L->R canonicalisation?

EXAMPLES-ONLY; pipeline (src/) untouched.

Proposed scheme (the one under test).  Drop the L->R canonicalisation entirely.
Sweep R->L on the *raw folded* MPS.  At bond tau take the bond matrix M_tau, apply
the carried gauge G_tau (which encodes the accumulated gauge offset of all sites to
the right of tau), SVD ``M_tau . G_tau``, truncate, keep the right factor as the new
right-canonical site, and pass the un-absorbed left factor on as G_{tau-1}.

Whether this *truncates correctly* hinges on conditioning: the local singular
values are Schmidt coefficients only if the left block is canonical, which this
sweep never establishes, so the carried gauge has to compound the non-canonicality.
If the gauge becomes ill-conditioned before mid-chain the scheme is dead -- a
correct gauge correction would have to invert / factor an ill-conditioned matrix
(beyond e.g. CholQR's ~1e3 working range), which is exactly the instability §14's
skip-QR exhibited.  So we just measure two condition numbers per bond and decide;
no end-to-end run needed.

  * ``cond(G_tau)`` -- the carried gauge actually used by the scheme (= ratio of the
    kept singular values at bond tau+1, compounded leftward).
  * ``cond(L_tau)`` -- the left environment ``L_tau = sum_d A_d^H L A_d`` of the raw
    folded MPS (intrinsic, truncation-free): how far from left-canonical the fold is,
    i.e. the conditioning any correct single-sweep gauge correction must handle.
    Reported over the numerically non-negligible eigenvalues.

Pure CPU / NumPy.

Usage
-----
    python examples/gauge_sweep_probe.py                 # L=12, xi=1e-6
    python examples/gauge_sweep_probe.py --L 6,12,18 --cutoff 1e-6,1e-8
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from edm_incremental import fold_all_L, fold_uncompressed, make_context  # noqa: E402

from edmtn.decomposition.base import truncation_rank  # noqa: E402
from edmtn.models import GaudinModel  # noqa: E402


def raw_folded(ctx, L):
    """Raw (uncompressed, non-canonical) folded MPS at step ``L`` (1-based)."""
    comp = fold_all_L(ctx, L - 1) if L > 1 else None
    mps_prev = comp[L - 1] if L > 1 else ctx["mps0"]
    return fold_uncompressed(ctx, mps_prev, L - 1)


def left_env_cond(mps, floor=1e-12):
    """``cond(L_tau)`` of the left environment at every internal bond.

    ``L_0 = I`` (boundary); ``L_{p+1} = sum_d A_p[d]^H L_p A_p[d]``.  Condition
    number is taken over eigenvalues above ``floor * lambda_max`` (the physically
    populated subspace; the rest is exact null space the truncation discards).
    """
    n = mps.num_sites
    L = np.eye(mps.tensors[0].shape[1], dtype=np.complex128)
    conds = []
    for p in range(n - 1):
        A = mps.tensors[p]                                  # (d, chi_l, chi_r)
        L = np.einsum("dlc,lm,dmr->cr", np.conj(A), L, A, optimize=True)
        w = np.linalg.eigvalsh(0.5 * (L + L.conj().T)).real
        w = w[w > floor * w.max()]
        conds.append(float(w.max() / w.min()) if w.size else np.inf)
    return conds


def gauge_sweep_cond(mps, xi, d2, mb):
    """Carried-gauge R->L truncation sweep; return ``cond(G_tau)`` per bond (tau=1..n-1).

    ``G`` starts at the right boundary; at each site the bond matrix absorbs ``G`` on
    its right bond, is SVD-truncated at strict ``xi``, the right factor becomes the new
    right-canonical site and ``G <- U S`` is carried left.  ``cond(G)`` is the ratio of
    its (kept) singular values, recorded in fold order (tau ascending)."""
    n = mps.num_sites
    G = np.eye(mps.tensors[-1].shape[2], dtype=np.complex128)   # right boundary
    conds = {}
    for p in range(n - 1, 0, -1):
        A = mps.tensors[p]                                  # (d, chi_l, chi_r)
        B = np.tensordot(A, G, axes=([2], [0]))             # (d, chi_l, g)
        d, chil, g = B.shape
        M = B.transpose(1, 0, 2).reshape(chil, d * g)       # (chi_l, d*g)
        U, s, Vh = np.linalg.svd(M, full_matrices=False)
        k = truncation_rank(s, cutoff=xi, cutoff_mode="rel_ref", ref_index=d2, max_bond=mb)
        s = s[:k]
        conds[p] = float(s[0] / s[-1]) if s.size and s[-1] > 0 else np.inf
        G = U[:, :k] * s                                    # carried gauge (chi_l, k)
    return [conds[p] for p in range(1, n)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--g", type=float, default=1.0)
    ap.add_argument("--K", type=int, default=24)
    ap.add_argument("--T", type=float, default=3.0)
    ap.add_argument("--eps", type=float, default=0.2)
    ap.add_argument("--order", type=int, default=2, choices=(1, 2))
    ap.add_argument("--max-bond", type=int, default=400)
    ap.add_argument("--L", default="12")
    ap.add_argument("--cutoff", default="1e-6")
    ap.add_argument("--kill", type=float, default=1e3, help="cond threshold to declare dead")
    args = ap.parse_args()

    Ls = [int(x) for x in args.L.split(",") if x.strip()]
    xis = [float(x) for x in args.cutoff.split(",") if x.strip()]
    model = GaudinModel(g=args.g, K=args.K)

    for xi in xis:
        ctx = make_context(model, T=args.T, eps=args.eps, order=args.order,
                           cutoff=xi, max_bond=args.max_bond)
        d2, mb = ctx["ref_index"], ctx["max_bond"]
        print(f"\n=== xi={xi:g}  (kill threshold cond > {args.kill:g}) ===")
        for L in Ls:
            mps = raw_folded(ctx, L)
            n = mps.num_sites
            gc = gauge_sweep_cond(mps, xi, d2, mb)
            le = left_env_cond(mps)
            mid = n // 2
            # first bond (from the right, i.e. largest tau) where cond(G) crosses kill
            cross = next((t for t in range(len(gc) - 1, -1, -1) if gc[t] > args.kill), None)
            cross_tau = (cross + 1) if cross is not None else None
            print(f"\n  L={L}: {n} sites, raw folded Dmax={mps.max_bond}")
            print(f"    cond(G_tau)  : min={min(gc):.1e}  median={np.median(gc):.1e}  "
                  f"max={max(gc):.1e}   at mid-chain(tau={mid})={gc[mid-1]:.1e}")
            print(f"    cond(L_tau)  : min={min(le):.1e}  median={np.median(le):.1e}  "
                  f"max={max(le):.1e}   at mid-chain(tau={mid})={le[mid-1]:.1e}")
            if cross_tau is not None:
                frac = cross_tau / n
                print(f"    -> cond(G) first exceeds {args.kill:g} at tau={cross_tau} "
                      f"({frac:.0%} along the chain from the right)")
            verdict = ("DEAD: gauge ill-conditioned before mid-chain"
                       if gc[mid - 1] > args.kill else
                       "survives mid-chain -- would need an end-to-end accuracy test")
            print(f"    verdict: {verdict}")


if __name__ == "__main__":
    main()
