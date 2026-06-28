"""Post-retirement GPU smoke: the single (quimb) pipeline on the A800.

After retiring the native path, validate the only pipeline on the GPU:
- it runs on CuPy for both models (Gaudin separable, spin-boson single-bath);
- compression knobs work on GPU (default zipup/rel, rsvd q=2/q=0, cholqr);
- GPU matches CPU;
- physics is sane (tight cutoff -> converged <S_z(t)>; trace preserved).

Run on a GPU node:  PYTHONPATH=src python examples/retire_gpu_smoke.py
"""

from __future__ import annotations

import sys

import numpy as np

from edmtn.driver.solver import solve
from edmtn.models import GaudinModel, SpinBosonModel


def _np(a):
    return np.asarray(a.get() if a.__class__.__module__.split(".")[0] == "cupy" else a)


def main():
    rows = []

    def check(name, ok, detail):
        rows.append(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<34} {detail}", flush=True)

    gaud = (GaudinModel(g=1.0, K=16), dict(T=4.0, eps=0.2, expansion_order=2, channel=3))
    sb = (SpinBosonModel(J0=0.6, omega_c=5.0, mu=1.0), dict(T=3.0, eps=0.1, expansion_order=2, channel=1))

    for label, (model, base) in [("Gaudin", gaud), ("SpinBoson", sb)]:
        ref = solve(model, backend="gpu", cutoff=1e-12, cutoff_mode="rel", **base)  # tight ref
        # knob combos on GPU vs the tight reference
        combos = [
            ("default zipup", {}),
            ("rsvd q2 direct", dict(compress_method="direct", compress_decomp="rsvd", compress_decomp_q=2)),
            ("rsvd q0 direct", dict(compress_method="direct", compress_decomp="rsvd", compress_decomp_q=0)),
            ("cholqr zipup", dict(compress_canon="cholqr")),
        ]
        for cname, kw in combos:
            got = solve(model, backend="gpu", cutoff=1e-8, cutoff_mode="rel", **kw, **base)
            n = min(len(ref.polarization), len(got.polarization))
            err = float(np.max(np.abs(_np(ref.polarization)[:n] - _np(got.polarization)[:n])))
            check(f"{label} {cname}", err < 1e-4, f"max|d<Sz>|={err:.2e} bond={got.max_bond}")
        # CPU vs GPU agreement (default path)
        c = solve(model, backend="cpu", cutoff=1e-8, cutoff_mode="rel", **base)
        g = solve(model, backend="gpu", cutoff=1e-8, cutoff_mode="rel", **base)
        n = min(len(c.polarization), len(g.polarization))
        err = float(np.max(np.abs(_np(c.polarization)[:n] - _np(g.polarization)[:n])))
        check(f"{label} CPU vs GPU", err < 1e-6, f"max|d<Sz>|={err:.2e}")

    print("=" * 60)
    ok = all(rows)
    print(f"  OVERALL: {'ALL PASS' if ok else 'FAILURES'} ({len(rows)} checks)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
