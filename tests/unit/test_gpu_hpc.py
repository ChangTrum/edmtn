"""Real GPU / HPC execution tests (P1-14).

These are the tests that MUST actually run on hardware -- CPU-passing never proves that CuPy
arrays stay on the device, that quimb's GPU compression is wired, that CPU and GPU agree, or
that the cuTensorNet channel / time-axis / MPI device mapping is right.  They are gated by real
hardware detection in ``tests/conftest.py`` (markers ``gpu`` / ``cuquantum`` / ``multigpu``), so
a CPU box collects them and skips them with a specific reason, while a GPU/HPC runner executes
them.  On a GPU runner, pass ``--require-gpu`` / ``--require-cuquantum`` / ``--require-multigpu=N``
so a detection failure fails the command instead of silently skipping.

  * single GPU, Track 1:  spin-boson & Gaudin CPU/CuPy parity, results stay CuPy on device.
  * single GPU, Track 2:  cuTensorNet vs the NumPy exact reference + error metrics + ngpu/rank.
  * multi GPU,  Track 2:  a separate MPI worker (``tests/hardware/_multigpu_worker.py``) launched by
                          ``cluster/test_gpu_hpc.sbatch`` (OpenMPI ``/usr/bin/mpirun -n <ngpu>``); this
                          test only reads + validates the worker's JSON result (path via
                          ``EDMTN_MULTIGPU_RESULT``).  (MPICH on c1 is launcher-less and its
                          srun/pmi2 path regressed, so OpenMPI's mpirun is used -- see
                          cluster/test_gpu_hpc.sbatch.)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from edmtn.driver import EDMSolver
from edmtn.driver.auto_config import SolverConfig
from edmtn.evolution.cutensornet import _mpi_context, solve_cutensornet
from edmtn.models import GaudinModel, SpinBosonModel


def _module(x) -> str:
    return type(x).__module__.split(".")[0]


# -- single GPU, Track 1: CPU/CuPy parity, results stay on device ----------------------------

@pytest.mark.gpu
def test_spinboson_cpu_gpu_parity():
    import cupy as cp  # noqa: PLC0415

    model = SpinBosonModel(J0=0.5, omega_c=5.0, mu=1.0)
    kw = dict(T=0.4, eps=0.1, expansion_order=2, cutoff=0.0)  # small, f64, exact -> no truncation gap
    cpu = EDMSolver.from_model(model, backend="cpu", **kw).solve(channel=1)
    gpu = EDMSolver.from_model(model, backend="gpu", **kw).solve(channel=1)

    assert cpu.backend.startswith("cpu") and gpu.backend.startswith("gpu")  # no silent GPU->CPU fallback
    assert _module(gpu.mps.tensors[0]) == "cupy"                             # MPS really on device
    rho_gpu = gpu.mps.reduced_density_matrix()
    assert _module(rho_gpu) == "cupy"                                        # reduced state stays CuPy

    np.testing.assert_allclose(gpu.times, cpu.times, atol=1e-12)
    np.testing.assert_allclose(gpu.polarization, cpu.polarization, atol=1e-8)
    np.testing.assert_allclose(cp.asnumpy(rho_gpu), cpu.mps.reduced_density_matrix(), atol=1e-8)
    assert gpu.expansion_order == cpu.expansion_order


@pytest.mark.gpu
def test_gaudin_cpu_gpu_parity():
    import cupy as cp  # noqa: PLC0415

    model = GaudinModel(g=0.8, K=3)
    kw = dict(T=0.4, eps=0.1, expansion_order=2, cutoff=0.0)
    cpu = EDMSolver.from_model(model, backend="cpu", **kw).solve(channel=3)
    gpu = EDMSolver.from_model(model, backend="gpu", **kw).solve(channel=3)

    assert cpu.backend.startswith("cpu") and gpu.backend.startswith("gpu")
    assert _module(gpu.mps.tensors[0]) == "cupy"
    rho_gpu = gpu.mps.reduced_density_matrix()
    assert _module(rho_gpu) == "cupy"

    np.testing.assert_allclose(gpu.times, cpu.times, atol=1e-12)
    np.testing.assert_allclose(gpu.polarization, cpu.polarization, atol=1e-8)
    np.testing.assert_allclose(cp.asnumpy(rho_gpu), cpu.mps.reduced_density_matrix(), atol=1e-8)
    assert gpu.expansion_order == cpu.expansion_order
    assert gpu.sub_baths_used == cpu.sub_baths_used == 3


# -- single GPU, Track 2: cuTensorNet vs NumPy exact ----------------------------------------

@pytest.mark.gpu
@pytest.mark.cuquantum
def test_track2_single_gpu_cutensornet_matches_numpy():
    import cupy as cp  # noqa: PLC0415

    model = GaudinModel(g=1.0, K=2, time_step_order=1)
    cfg = SolverConfig(eps=0.1, T=0.3, expansion_order=1, backend="hpc")
    # This must exercise the TRUE single-process (non-distributed) cuTensorNet path, not the
    # size=1 distributed branch that a stray SLURM_NTASKS>1 would trigger. ngpu==1 alone can't
    # tell them apart, so assert the launcher context is genuinely absent (run under
    # `srun --ntasks=1`, per cluster/test_gpu_hpc.sbatch).
    assert _mpi_context() is None
    # single-GPU hpc emits the "running on a single GPU" efficiency hint -- expected, assert it
    # here so it never leaks as an undeclared warning into the suite
    with pytest.warns(UserWarning, match="single GPU"):
        gpu = solve_cutensornet(model, cfg, channel=3, executor="cuquantum")
    ref = solve_cutensornet(model, cfg, channel=3, executor="numpy")

    np.testing.assert_allclose(gpu["times"], ref["times"], atol=1e-12)
    np.testing.assert_allclose(gpu["polarization"], ref["polarization"], atol=1e-9)
    np.testing.assert_allclose(gpu["final_rho"], ref["final_rho"], atol=1e-10)
    assert gpu["error_metrics"]["hermiticity"] < 1e-12
    assert gpu["error_metrics"]["trace_dev"] < 1e-12
    assert gpu["ngpu"] == 1 and gpu["rank"] == 0
    # Track 2 returns a host (cp.asnumpy'd) array by contract -- final_rho is NumPy, not CuPy
    assert _module(gpu["final_rho"]) == "numpy"
    assert cp.cuda.runtime.getDevice() == 0            # single-GPU run stays on device 0


# -- multi GPU, Track 2: distributed contraction via a separate MPI worker -------------------

@pytest.mark.gpu
@pytest.mark.cuquantum
@pytest.mark.multigpu(4)
def test_track2_multigpu_distributed_matches_numpy():
    # The distributed worker is launched by cluster/test_gpu_hpc.sbatch via OpenMPI's
    # `/usr/bin/mpirun -n <ngpu> python tests/hardware/_multigpu_worker.py <json>` (c1's MPICH is
    # launcher-less and its srun/pmi2 path regressed, so OpenMPI's mpirun is used).  This test only
    # validates the worker's JSON result; the sbatch passes its path via EDMTN_MULTIGPU_RESULT.
    # NOTE: `-m multigpu --require-multigpu=N` makes a missing result JSON a HARD FAILURE (conftest),
    # so a worker that never ran can't slip through as an all-skip exit 0.
    result_path = os.environ.get("EDMTN_MULTIGPU_RESULT")
    if not result_path or not os.path.exists(result_path):
        pytest.skip("no multi-GPU worker result JSON "
                    "(set EDMTN_MULTIGPU_RESULT; run via cluster/test_gpu_hpc.sbatch)")

    data = json.loads(Path(result_path).read_text())
    assert data["all_ranks_ok"] is True, f"worker errors: {data['errors']}"  # every rank completed
    assert data["size"] == 4                    # one rank per GPU
    assert data["ngpu"] == 4                     # cuTensorNet used all 4 GPUs
    # the ranks used 4 DISTINCT physical GPUs (UUID/PCI, not the always-0 logical device number)
    assert len(data["device_ids"]) == 4 and data["devices_unique"] is True
    assert data["ranks_agree"] is True          # every rank's rho(T) agrees (MPI gather)
    assert data["max_err_vs_numpy"] < 1e-10      # distributed rho(T) == NumPy exact reference
    assert data["hermiticity"] < 1e-12 and data["trace_dev"] < 1e-12


# -- gating regression: --require-multigpu must reject a MISSING worker result JSON ----------
# (runs on any box -- no GPU/MPI needed -- exercising the conftest rule directly. This is the
# "hardware satisfied but result JSON absent" case that must NOT pass as an all-skip exit 0.)

def _load_root_conftest():
    import importlib.util  # noqa: PLC0415
    path = Path(__file__).resolve().parent.parent / "conftest.py"
    spec = importlib.util.spec_from_file_location("edmtn_root_conftest", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_require_multigpu_needs_readable_result_json(tmp_path):
    ct = _load_root_conftest()
    # unset / missing / invalid -> a problem (which the conftest turns into a non-zero exit under
    # --require-multigpu); a readable JSON -> None (no problem)
    assert ct._multigpu_result_problem(None) is not None
    assert ct._multigpu_result_problem(str(tmp_path / "does_not_exist.json")) is not None
    bad = tmp_path / "bad.json"
    bad.write_text("{ not valid json")
    assert ct._multigpu_result_problem(str(bad)) is not None
    good = tmp_path / "ok.json"
    good.write_text('{"all_ranks_ok": true, "size": 4}')
    assert ct._multigpu_result_problem(str(good)) is None
