"""HPC track (cuQuantum 2D contraction) GPU validation — run on c1 (A800).

    PYTHONPATH=src python examples/cutensornet_sanity.py

Validates the ``backend="hpc"`` path end-to-end on the GPU: the exact mode
(`compress_decomp="exact"`) reproduces Track 1's exact fold to machine precision,
and the approximate mode (`compress_decomp="approx"`) truncates with err≈cutoff —
both reporting the reference error metrics. Also exercises the public
``solve(backend="hpc")`` driver entry. Non-zero exit ⇒ a gate failed.
"""

from __future__ import annotations

import numpy as np

from edmtn.driver import solve
from edmtn.driver.auto_config import SolverConfig
from edmtn.evolution.cutensornet import _make_expander, solve_cutensornet
from edmtn.evolution.separable_bath import SeparableBathEvolution
from edmtn.kernels.separable_mpo import SeparableKernelEngine
from edmtn.models import GaudinModel


def _section(t):
    print(f"\n=== {t} ===", flush=True)


def track1_final_rho(model, order, eps, N):
    """Track 1 exact (uncompressed) ρ(N) — the correctness anchor (small N only)."""
    ke = SeparableKernelEngine.from_model(model, N * eps, eps)
    res = SeparableBathEvolution(_make_expander(order)).run(model, ke, eps, N, compress=False)
    return res.mps.reduced_density_matrix()


def versions():
    _section("versions")
    import cupy as cp
    import cuquantum
    print(f"cuquantum {getattr(cuquantum, '__version__', '?')}  cupy {cp.__version__}")
    props = cp.cuda.runtime.getDeviceProperties(0)
    name = props["name"]
    print(f"device: {name.decode() if isinstance(name, bytes) else name}")


def exact_vs_track1():
    _section("exact mode vs Track 1 (machine precision)")
    worst = 0.0
    for order, K, N, pf in [(1, 3, 4, "cuquantum"), (2, 2, 3, "cuquantum"),
                            (1, 4, 3, "cuquantum"), (1, 3, 4, "cotengra")]:
        model = GaudinModel(g=1.0, K=K, time_step_order=order)
        cfg = SolverConfig(eps=0.1, T=N * 0.1, expansion_order=order, backend="hpc",
                           compress_decomp="exact", pathfinder=pf)
        out = solve_cutensornet(model, cfg, channel=3, executor="cuquantum")
        ref = track1_final_rho(model, order, 0.1, N)
        err = float(np.max(np.abs(out["final_rho"] - ref)))
        worst = max(worst, err)
        m = out["error_metrics"]
        print(f"  order={order} K={K} N={N} pf={pf}: max|Δ|={err:.2e} "
              f"herm={m['hermiticity']:.1e} trdev={m['trace_dev']:.1e} "
              f"slices={m.get('num_slices')} {'PASS' if err < 1e-10 else 'FAIL'}")
    assert worst < 1e-10, f"exact mismatch {worst:.2e}"
    print(f"  exact: all PASS (worst {worst:.2e})")


def approx_vs_exact():
    _section("approx mode: err vs cutoff (truncation engaged), vs exact-hpc")
    model = GaudinModel(g=1.0, K=6, time_step_order=1)
    N = 9  # longer evolution so the EDM bond grows enough for the cutoff to bite
    cfg_ex = SolverConfig(eps=0.1, T=N * 0.1, expansion_order=1, backend="hpc",
                          compress_decomp="exact")
    rho_ex = solve_cutensornet(model, cfg_ex, channel=None, executor="cuquantum")["final_rho"]
    for cutoff in (1e-2, 1e-4, 1e-6):
        cfg = SolverConfig(eps=0.1, T=N * 0.1, expansion_order=1, backend="hpc",
                           compress_decomp="approx", cutoff=cutoff, cutoff_mode="rel",
                           max_bond=256)
        out = solve_cutensornet(model, cfg, channel=None, executor="cuquantum")
        err = float(np.max(np.abs(out["final_rho"] - rho_ex)))
        print(f"  cutoff={cutoff:.0e}: max|Δ vs exact-hpc|={err:.2e} "
              f"herm={out['error_metrics']['hermiticity']:.1e}")


def driver_end_to_end():
    _section("public solve(backend='hpc') driver path")
    model = GaudinModel(g=1.0, K=3, time_step_order=2)
    res = solve(model, T=0.3, eps=0.1, channel=3, backend="hpc", compress_decomp="exact")
    print(f"  backend label: {res.backend}")
    print(f"  times: {np.asarray(res.times)}")
    print(f"  <S_z(t)>: {np.asarray(res.polarization)}")
    print(f"  rho(t) returned: {res.density_matrices is not None} "
          f"(n={len(res.density_matrices)})")
    print(f"  error_metrics: {res.error_metrics}")
    assert res.density_matrices is not None and res.error_metrics is not None


def main() -> int:
    versions()
    exact_vs_track1()
    approx_vs_exact()
    driver_end_to_end()
    _section("RESULT")
    print("HPC track GPU validation: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
