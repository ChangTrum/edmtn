"""Shared hardware detection + GPU/cuQuantum/multi-GPU test gating (P1-14).

Real hardware is auto-detected here -- lru-cached, importing CuPy / cuQuantum / mpi4py lazily
and at most once, so the CUDA runtime is never initialised more than once during collection.
Tests only carry markers::

    @pytest.mark.gpu
    @pytest.mark.cuquantum          # also mark gpu
    @pytest.mark.multigpu(4)        # also mark gpu + cuquantum

``pytest_collection_modifyitems`` skips a marked test -- with a SPECIFIC reason -- when the
hardware/stack it needs is absent, BEFORE its body runs (so a CPU box never imports CuPy in a
test body).  A normal CPU run therefore collects those tests and skips them; no test uses an
unconditional always-skip.

To stop a broken GPU runner from "passing" by skipping everything, pass ``--require-gpu`` /
``--require-cuquantum`` / ``--require-multigpu=N``: the whole command then exits non-zero if the
required hardware/stack is missing, instead of silently skipping.

Each detector returns ``(ok, reason)`` with a SPECIFIC reason (CuPy missing vs CUDA init failed
vs no device vs cuQuantum missing vs distributed API missing vs no SLURM allocation ...), so a
skip or a forced failure says exactly what was wrong.
"""

from __future__ import annotations

import functools
import json
import os
import os.path as osp
import shutil

import pytest


# -- detection: (ok, reason); lru-cached so CUDA initialises at most once --------------------

@functools.lru_cache(maxsize=None)
def gpu_status() -> tuple[bool, str]:
    try:
        import cupy as cp  # noqa: PLC0415
    except Exception as exc:  # ModuleNotFoundError on a CPU box
        return False, f"CuPy not importable ({type(exc).__name__})"
    try:
        n = int(cp.cuda.runtime.getDeviceCount())
    except Exception as exc:  # CUDA driver / runtime init failure
        return False, f"CUDA runtime init failed ({type(exc).__name__}: {exc})"
    if n < 1:
        return False, "no visible CUDA device"
    return True, f"{n} CUDA GPU(s)"


@functools.lru_cache(maxsize=None)
def gpu_count() -> int:
    ok, _ = gpu_status()
    if not ok:
        return 0
    import cupy as cp  # noqa: PLC0415
    return int(cp.cuda.runtime.getDeviceCount())


@functools.lru_cache(maxsize=None)
def cuquantum_status() -> tuple[bool, str]:
    ok, reason = gpu_status()
    if not ok:
        return False, reason
    try:  # the tensornet API the hpc track actually calls
        from cuquantum.tensornet import contract  # noqa: F401,PLC0415
    except Exception as exc:
        return False, f"cuQuantum tensornet API not importable ({type(exc).__name__})"
    return True, "cuQuantum tensornet available"


def _env_int(name: str):
    """Return ``int(os.environ[name])`` or ``None`` (unset / not an int)."""
    raw = os.environ.get(name)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


@functools.lru_cache(maxsize=None)
def multigpu_status(n: int) -> tuple[bool, str]:
    # the multi-GPU acceptance is a single-node, >=2-rank distributed run
    if not isinstance(n, int) or n < 2:
        return False, f"multigpu needs an integer n >= 2 (got {n!r})"
    ok, reason = cuquantum_status()
    if not ok:
        return False, reason
    have = gpu_count()
    if have < n:
        return False, f"fewer than {n} visible GPUs ({have})"
    try:  # the distributed stack the multi-GPU worker needs
        from cuquantum.bindings import cutensornet  # noqa: F401,PLC0415
        from cuquantum.tensornet import get_mpi_comm_pointer  # noqa: F401,PLC0415
    except Exception as exc:
        return False, f"cuQuantum distributed API not importable ({type(exc).__name__})"
    # check mpi4py is importable WITHOUT initialising MPI: `from mpi4py import MPI` runs MPI_Init,
    # which aborts (PMI_KVS_Commit) for a launcher-less process under the MPICH ABI -- the worker
    # (under mpiexec) is where MPI is actually initialised, never this gate.
    import importlib.util  # noqa: PLC0415
    if importlib.util.find_spec("mpi4py") is None:
        return False, "mpi4py not importable"
    if not os.environ.get("SLURM_JOB_ID"):
        return False, "no SLURM allocation (SLURM_JOB_ID unset)"
    if shutil.which("srun") is None:
        return False, "srun not found on PATH"
    # the allocation must actually provide >= n task slots on a SINGLE node (the worker srun
    # requests --ntasks=n --nodes=1), else this only fails much later deep inside srun
    ntasks = _env_int("SLURM_NTASKS")
    if ntasks is None:
        ntasks = _env_int("SLURM_NPROCS")
    if ntasks is None or ntasks < n:
        return False, f"SLURM allocation has < {n} task slots (SLURM_NTASKS={os.environ.get('SLURM_NTASKS')!r})"
    nnodes = _env_int("SLURM_NNODES")
    if nnodes is None:
        nnodes = _env_int("SLURM_JOB_NUM_NODES")
    if nnodes is not None and nnodes != 1:
        return False, f"multi-GPU acceptance is single-node only (SLURM_NNODES={nnodes})"
    comm_lib = os.environ.get("CUTENSORNET_COMM_LIB")
    if not comm_lib or not osp.isfile(comm_lib):
        return False, "CUTENSORNET_COMM_LIB unset or points to a missing file"
    return True, f"{n}-GPU single-node SLURM allocation ready"


# -- options / gating ------------------------------------------------------------------------

def pytest_addoption(parser):
    parser.addoption("--require-gpu", action="store_true", default=False,
                     help="exit non-zero (not skip) if no CUDA GPU / CuPy is available")
    parser.addoption("--require-cuquantum", action="store_true", default=False,
                     help="exit non-zero (not skip) if cuQuantum/cuTensorNet is unavailable")
    parser.addoption("--require-multigpu", type=int, default=0, metavar="N",
                     help="exit non-zero (not skip) if no usable N-GPU MPI/SLURM allocation")


def _multigpu_n(item) -> int:
    m = item.get_closest_marker("multigpu")
    if m is None:
        return 0
    return int(m.args[0]) if m.args else 2


def _multigpu_result_problem(result_path):
    """Return a reason string if the multi-GPU worker RESULT JSON is missing/unreadable/invalid,
    else ``None``.  Under ``--require-multigpu`` a usable allocation is NOT enough: the worker
    (launched by cluster/test_gpu_hpc.sbatch) must have actually produced a readable JSON result,
    else the multigpu test ``pytest.skip``s and ``-m multigpu --require-multigpu=N`` could exit 0
    on all-skips -- masquerading as a multi-GPU pass.  Kept a pure function so it is unit-testable
    without any GPU/MPI hardware."""
    if not result_path:
        return "EDMTN_MULTIGPU_RESULT is unset (the worker produced no result)"
    if not osp.isfile(result_path) or not os.access(result_path, os.R_OK):
        return f"EDMTN_MULTIGPU_RESULT is not a readable file: {result_path!r}"
    try:
        with open(result_path) as fh:
            json.loads(fh.read())
    except Exception as exc:  # noqa: BLE001
        return f"EDMTN_MULTIGPU_RESULT is not valid JSON ({type(exc).__name__}): {result_path!r}"
    return None


def pytest_collection_modifyitems(config, items):
    # 1) hard requirements: a --require-* the runner can't satisfy FAILS the whole command, so
    #    "everything skipped" can never masquerade as a pass on a broken GPU/HPC runner.
    problems = []
    if config.getoption("--require-gpu"):
        ok, reason = gpu_status()
        if not ok:
            problems.append(f"--require-gpu: {reason}")
    if config.getoption("--require-cuquantum"):
        ok, reason = cuquantum_status()
        if not ok:
            problems.append(f"--require-cuquantum: {reason}")
    req_mgpu = config.getoption("--require-multigpu")
    if req_mgpu:
        ok, reason = multigpu_status(req_mgpu)
        if not ok:
            problems.append(f"--require-multigpu={req_mgpu}: {reason}")
        # a usable allocation alone is not proof: the worker must have produced a readable JSON
        # result, else all-skip would exit 0 and masquerade as a multi-GPU pass
        rprob = _multigpu_result_problem(os.environ.get("EDMTN_MULTIGPU_RESULT"))
        if rprob:
            problems.append(f"--require-multigpu={req_mgpu}: {rprob}")
    if problems:
        pytest.exit("required hardware/stack unavailable:\n  " + "\n  ".join(problems),
                    returncode=1)

    # 2) otherwise skip each hardware-marked test whose stack is absent, with a SPECIFIC reason
    #    (most-specific marker governs: multigpu implies cuquantum implies gpu).
    for item in items:
        if item.get_closest_marker("multigpu") is not None:
            n = _multigpu_n(item)
            ok, reason = multigpu_status(n)
            if not ok:
                item.add_marker(pytest.mark.skip(reason=f"multigpu({n}): {reason}"))
        elif item.get_closest_marker("cuquantum") is not None:
            ok, reason = cuquantum_status()
            if not ok:
                item.add_marker(pytest.mark.skip(reason=f"cuquantum: {reason}"))
        elif item.get_closest_marker("gpu") is not None:
            ok, reason = gpu_status()
            if not ok:
                item.add_marker(pytest.mark.skip(reason=f"gpu: {reason}"))


@pytest.fixture
def gpu_available() -> bool:
    """True when a real CUDA GPU + CuPy is present.

    For *adaptive* tests that run on any machine and branch on GPU presence (e.g. the
    backend-resolution fallback), NOT for GPU-only tests -- those use ``@pytest.mark.gpu``.
    """
    ok, _ = gpu_status()
    return ok
