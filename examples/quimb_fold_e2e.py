"""Phase 0 (e2e): full Gaudin solve compressing every fold with a quimb-NATIVE
cutoff, re-validating <S_z(t)> against the StandardSVD (rel_ref) reference.

This de-risks dropping rel_ref (user decision) before any src rewrite: with a
quimb-native cutoff_mode the truncation rule -- and hence the bond dims -- change,
so the invariant we check is the *observable* (the physics), not the bond.  The
fold contraction is reused from the existing pipeline; only the compression step
is swapped to quimb's `tensor_network_1d_compress` (cotengra/autoray).

EXAMPLES-ONLY; src untouched.

    PYTHONPATH=src python examples/quimb_fold_e2e.py --K 24 --modes rsum2,rel
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from edm_incremental import compress_full, fold_uncompressed, make_context  # noqa: E402
from quimb_tn_feasibility import edmmps_to_quimb, quimb_to_edmmps  # noqa: E402
from canon_alternatives_e2e import coupling_polarization_tolerant  # noqa: E402

from edmtn.models import GaudinModel  # noqa: E402

import quimb.tensor as qtn  # noqa: E402


def quimb_compress(unc, ctx, cutoff_mode, cutoff, method):
    tn = edmmps_to_quimb(unc)
    site_tags = [f"I{p}" for p in range(unc.num_sites)]
    cq = qtn.tensor_network_1d_compress(
        tn, max_bond=ctx["max_bond"], cutoff=cutoff, cutoff_mode=cutoff_mode,
        method=method, site_tags=site_tags, permute_arrays=False)
    return quimb_to_edmmps(cq, unc.num_sites, unc)


def run_fold(ctx, K, compress_fn):
    eps, order = ctx["eps"], ctx["order"]
    mps = ctx["mps0"]
    t0 = time.perf_counter()
    for k in range(K):
        unc = fold_uncompressed(ctx, mps, k)
        mps = unc if unc.num_sites <= 1 else compress_fn(unc)
    wall = time.perf_counter() - t0
    pol, imag_rel = coupling_polarization_tolerant(mps, eps, channel=3, order=order)
    return np.asarray(pol), wall, mps.max_bond, imag_rel


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--g", type=float, default=1.0)
    ap.add_argument("--K", type=int, default=24)
    ap.add_argument("--T", type=float, default=3.0)
    ap.add_argument("--eps", type=float, default=0.2)
    ap.add_argument("--cutoff", type=float, default=1e-6, help="reference (rel_ref) cutoff")
    ap.add_argument("--max-bond", type=int, default=400)
    ap.add_argument("--modes", default="rsum2,rel", help="quimb cutoff modes to sweep")
    ap.add_argument("--qcutoffs", default="1e-7,1e-9,1e-11",
                    help="quimb cutoff values to sweep (mode-dependent meaning)")
    ap.add_argument("--method", default="zipup", choices=("zipup", "dm", "svd"))
    args = ap.parse_args()

    model = GaudinModel(g=args.g, K=args.K)
    ctx = make_context(model, T=args.T, eps=args.eps, order=2,
                       cutoff=args.cutoff, max_bond=args.max_bond)
    print(f"quimb e2e fold (Gaudin): K={args.K}, T={args.T}, eps={args.eps}, "
          f"ref rel_ref xi={args.cutoff:g}, method={args.method}")

    # reference: the production StandardSVD (rel_ref) solve
    pol_ref, wall_ref, dmax_ref, _ = run_fold(ctx, args.K, lambda u: compress_full(ctx, u)[0])
    print(f"  reference StandardSVD(rel_ref): Dmax={dmax_ref}, {wall_ref:.1f}s")

    print(f"\n  {'mode':>7} {'qcutoff':>9} | {'max|dSz| vs ref':>15} {'imag/re':>8} {'Dmax':>5} {'wall':>7}")
    print("  " + "-" * 60)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    qcuts = [float(c) for c in args.qcutoffs.split(",") if c.strip()]
    for mode in modes:
        for qc in qcuts:
            pol, wall, dmax, imag = run_fold(
                ctx, args.K, lambda u, m=mode, c=qc: quimb_compress(u, ctx, m, c, args.method))
            n = min(len(pol), len(pol_ref))
            err = float(np.max(np.abs(pol[:n] - pol_ref[:n])))
            flag = "  <- matches" if err < 1e-4 and imag < 1e-3 else ""
            print(f"  {mode:>7} {qc:9.0e} | {err:15.2e} {imag:8.1e} {dmax:5d} {wall:6.1f}s{flag}")

    print("\n  invariant = the observable <S_z(t)> (physics); bond dims change with the rule.")
    print("  goal: a quimb-native cutoff that matches the reference < 1e-4 at a comparable Dmax.")


if __name__ == "__main__":
    main()
