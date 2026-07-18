"""Multi-GPU distributed cuTensorNet worker (P1-14) -- launched by cluster/test_gpu_hpc.sbatch.

This is NOT a pytest test (its name is deliberately not ``test_*`` so pytest never collects it).
The sbatch runs it as ``mpirun -n <ngpu> python tests/hardware/_multigpu_worker.py <result.json>``
via OpenMPI's ``/usr/bin/mpirun`` (c1's MPICH is launcher-less and its srun/pmi2 path regressed, so
OpenMPI's own launcher is used): one MPI rank per GPU, each rank runs the real distributed
``solve_cutensornet`` (cuTensorNet owns the slice distribution), and rank 0 writes a single
structured JSON result the pytest process then checks -- so the verdict never depends on parsing
interleaved multi-rank stdout.

Verdict (JSON): every rank completed (``all_ranks_ok``); ``size`` ranks / ``ngpu`` GPUs; each
rank's ``result.rank``/``ngpu`` matched MPI; the ranks used ``size`` DISTINCT physical GPUs
(``device_ids`` UUID/PCI, ``devices_unique`` -- NOT the logical device number, which can be 0 for
every rank); every rank's rho(T) agrees (``ranks_agree``, via gather) and matches the
single-process NumPy exact reference (``max_err_vs_numpy``); Hermiticity and trace deviation are
within tolerance.  Any rank failing -- or the ranks sharing a GPU -- makes the launcher return
non-zero.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np


def _isolate_numba_cache() -> None:
    """Per-(job, rank) NUMBA cache dir, set BEFORE importing anything that JITs -- four ranks
    sharing one cache dir has caused real JIT-cache corruption (see local-perf-debug-lessons)."""
    job = os.environ.get("SLURM_JOB_ID", "nojob")
    proc = os.environ.get("SLURM_PROCID", "0")
    d = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"edmtn-numba-{job}-{proc}")
    os.makedirs(d, exist_ok=True)
    os.environ["NUMBA_CACHE_DIR"] = d


def _physical_gpu_id() -> str:
    """Stable PHYSICAL identity of the current CUDA device (UUID if exposed, else PCI
    domain:bus:device).  NOT the logical device number: under ``--gpus-per-task=1`` every rank's
    logical device is 0, so only the physical id proves the ranks used distinct GPUs."""
    import cupy as cp  # noqa: PLC0415

    dev = cp.cuda.runtime.getDevice()
    props = cp.cuda.runtime.getDeviceProperties(dev)
    uuid = props.get("uuid")
    if uuid:
        return "uuid:" + (uuid.hex() if isinstance(uuid, (bytes, bytearray)) else str(uuid))
    return "pci:%04x:%02x:%02x" % (
        props.get("pciDomainID", 0), props.get("pciBusID", 0), props.get("pciDeviceID", 0))


def main(result_path: str) -> int:
    _isolate_numba_cache()  # must precede the edmtn (numba-JIT) imports

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()

    from edmtn.driver.auto_config import SolverConfig
    from edmtn.evolution.cutensornet import solve_cutensornet
    from edmtn.models import GaudinModel

    model = GaudinModel(g=1.0, K=2, time_step_order=1)
    cfg = SolverConfig(eps=0.1, T=0.3, expansion_order=1, backend="hpc")

    # -- every rank runs the distributed solve (cuTensorNet distributes the slices) --
    ok, err, rho, gpu_id = True, None, None, None
    try:
        out = solve_cutensornet(model, cfg, channel=3, executor="cuquantum")
        rho = np.asarray(out["final_rho"])
        gpu_id = _physical_gpu_id()   # the device THIS rank actually contracted on
        if out["rank"] != rank:
            raise AssertionError(f"result.rank {out['rank']} != MPI rank {rank}")
        if out["ngpu"] != size:
            raise AssertionError(f"result.ngpu {out['ngpu']} != MPI size {size}")
    except Exception as exc:  # noqa: BLE001 - report, never abort a collective
        ok, err = False, f"rank {rank}: {exc!r}"

    # -- collectives are called by ALL ranks unconditionally (no divergence -> no deadlock) --
    all_ok = bool(comm.allreduce(ok, op=MPI.LAND))
    errs = comm.gather(err, root=0)
    rhos = comm.gather(rho, root=0)
    gpu_ids = comm.gather(gpu_id, root=0)
    visible = comm.gather(os.environ.get("CUDA_VISIBLE_DEVICES"), root=0)

    if rank == 0:
        result = {
            "size": int(size), "all_ranks_ok": all_ok, "ngpu": None, "ranks_agree": False,
            "max_err_vs_numpy": float("inf"), "hermiticity": float("inf"),
            "trace_dev": float("inf"), "device_ids": [], "devices_unique": False,
            "cuda_visible_devices": None, "errors": [e for e in errs if e],
        }
        if all_ok:
            try:
                ref = solve_cutensornet(model, cfg, channel=3, executor="numpy")
                rref = np.asarray(ref["final_rho"])
                r0 = np.asarray(rhos[0])
                result["ngpu"] = int(size)
                result["ranks_agree"] = all(
                    np.allclose(np.asarray(r), r0, atol=1e-12) for r in rhos)
                result["max_err_vs_numpy"] = max(
                    float(np.max(np.abs(np.asarray(r) - rref))) for r in rhos)
                result["hermiticity"] = float(np.max(np.abs(r0 - r0.conj().T)))
                result["trace_dev"] = float(abs(np.trace(r0) - 1.0))
                # physical-GPU distinctness: size ids, all present, all different
                result["device_ids"] = list(gpu_ids)
                result["cuda_visible_devices"] = list(visible)
                result["devices_unique"] = (
                    all(g for g in gpu_ids) and len(set(gpu_ids)) == size)
                if not result["devices_unique"]:
                    result["all_ranks_ok"] = False
                    result["errors"].append(
                        f"ranks did not use {size} distinct physical GPUs: ids={gpu_ids} "
                        f"CUDA_VISIBLE_DEVICES={visible}")
            except Exception as exc:  # noqa: BLE001
                result["all_ranks_ok"] = False
                result["errors"].append(f"rank 0 reference/compare: {exc!r}")
        with open(result_path, "w") as fh:
            json.dump(result, fh)
        all_ok = result["all_ranks_ok"]

    comm.Barrier()
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
