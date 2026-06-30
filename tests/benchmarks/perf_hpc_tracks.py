"""HPC benchmark runner — ONE (scale, config) case per call; writes a JSON result.

Gaudin only. Compares the three execution paths on hard metrics (wall time, memory,
accuracy/error), at two scales. Every config goes through the unified `solve()`, so
**all of them return the full rho(t) trajectory** (Track 2 included — `solve(backend=
'hpc')` builds density_matrices over t; it is just O(N) contractions today, the known
optimization target, vs Track 1's one-fold O(N) sweep).

Scales (both order 2, eps=0.1, T=8 -> N=80, 160 sub-steps):
  s1: K=10   s2: K=15      (s0: K=4, T=1 -- tiny smoke, not a benchmark point)
Configs:
  cpu     -- Track 1 (NumPy, a8), compressed fold.
  gpu1t1  -- Track 1 (CuPy, c1 1xA800), compressed fold.
  hpc     -- Track 2 (cuTensorNet, c1) exact 2D contraction; 1 GPU (single process)
             or N GPUs (under `srun --ntasks=N`, auto-detected from the launcher).

Track-1 recipe (user-specified): direct compression + cold rSVD, rel cutoff 1e-6,
max_bond 1024, canonicalisation left at default. Track 2 uses the default cuQuantum
path-finder. Under MPI only rank 0 writes the result file.
"""

from __future__ import annotations

import argparse
import json
import os
import time

SCALES = {
    "s0": dict(K=4, order=2, eps=0.1, T=1.0),
    "s1": dict(K=10, order=2, eps=0.1, T=8.0),
    "s2": dict(K=15, order=2, eps=0.1, T=8.0),
}
T1_KNOBS = dict(cutoff=1e-6, cutoff_mode="rel", max_bond=1024,
                compress_method="direct", compress_decomp="rsvd", compress_decomp_q=2)
BACKEND = {"cpu": "cpu", "gpu1t1": "gpu", "hpc": "hpc"}


def _rank():
    for v in ("OMPI_COMM_WORLD_RANK", "PMI_RANK", "SLURM_PROCID"):
        if v in os.environ:
            try:
                return int(os.environ[v])
            except ValueError:
                pass
    return 0


def _peak_rss_mb():
    import resource  # noqa: PLC0415
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0  # KB->MB (Linux)


def _gpu_mem():
    try:
        import cupy as cp  # noqa: PLC0415
        free, total = cp.cuda.runtime.memGetInfo()
        return {"gpu_dev_used_mb": (total - free) / 1e6,
                "gpu_pool_used_mb": cp.get_default_memory_pool().used_bytes() / 1e6}
    except Exception:  # noqa: BLE001
        return {}


def _to_np(a):
    import numpy as np  # noqa: PLC0415
    return a.get() if hasattr(a, "get") else np.asarray(a)


def main():
    import numpy as np  # noqa: PLC0415
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", required=True, choices=list(SCALES))
    ap.add_argument("--config", required=True, choices=list(BACKEND))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    p = SCALES[a.scale]
    K, order, eps, T = p["K"], p["order"], p["eps"], p["T"]
    N = int(round(T / eps))

    from edmtn.driver import solve  # noqa: PLC0415
    from edmtn.models import GaudinModel  # noqa: PLC0415
    model = GaudinModel(g=1.0, K=K, time_step_order=order)
    knobs = dict(expansion_order=order)
    if a.config in ("cpu", "gpu1t1"):
        knobs.update(T1_KNOBS)

    t0 = time.perf_counter()
    res = solve(model, T=T, eps=eps, channel=3, backend=BACKEND[a.config], **knobs)
    wall = time.perf_counter() - t0

    if _rank() != 0:           # under MPI only rank 0 records
        return

    rho = _to_np(res.density_matrices[-1]) if res.density_matrices is not None \
        else _to_np(res.mps.reduced_density_matrix())
    Sz = _to_np(model.coupling_operators_at(T)[2])      # channel 3 = S_z
    rec = dict(scale=a.scale, config=a.config, model="gaudin", g=1.0, K=K, order=order,
               eps=eps, T=T, N=N, backend_label=res.backend, wall_s=wall,
               sz_traj=[float(x) for x in np.asarray(res.polarization)],
               sz_T=float(np.trace(Sz @ rho).real),
               hermiticity=float(np.max(np.abs(rho - rho.conj().T))),
               trace_dev=float(abs(complex(np.trace(rho)) - 1.0)),
               rho_re=rho.real.tolist(), rho_im=rho.imag.tolist(),
               peak_rss_mb=_peak_rss_mb(), status="ok")
    if res.error_metrics:                       # hpc: hermiticity/trace/num_slices/flops
        rec["error_metrics"] = res.error_metrics
    if res.bond_dims:                           # Track 1: per-step max bond
        rec["max_bond"] = int(res.max_bond)
    rec.update(_gpu_mem())
    with open(a.out, "w") as f:
        json.dump(rec, f, indent=2)
    print(f"WROTE {a.out}: backend={res.backend} wall={wall:.1f}s sz_T={rec['sz_T']:.6f}",
          flush=True)


if __name__ == "__main__":
    main()
