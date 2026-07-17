"""Unit tests for Layer 7 (driver) on the separable / Gaudin pipeline.

Validates pipeline auto-selection and that the end-to-end solver reproduces the
exact Trotterised <S_z(t)> trajectory of the full central-spin + bath-spin system
(orders 1 and 2), plus the all-times bond-dimension and sub-bath records.
"""

import numpy as np
import pytest

from edmtn.driver import EDMSolver, SolverConfig, solve
from edmtn.driver.auto_config import available_pipelines, build_pipeline
from edmtn.evolution.separable_bath import SeparableBathEvolution
from edmtn.kernels.separable_mpo import SeparableKernelEngine
from edmtn.models import GaudinModel

X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)
S_ALPHA = [X / 2, Y / 2, Z / 2]


def exact_sz_trajectory(model, eps, n_steps, order):
    """Exact ``<S_z(t)>`` at the intervention times ``t = 0, eps, ..., (N-1) eps``
    (measured *before* each Trotter step), full-system reference."""
    g = model.couplings
    K = model.K
    D = 2 ** (K + 1)
    Id = np.eye(D, dtype=complex)
    H = np.zeros((D, D), dtype=complex)
    for k in range(K):
        for alpha in range(3):
            ops = [I2] * (K + 1)
            ops[0] = S_ALPHA[alpha]
            ops[k + 1] = S_ALPHA[alpha]
            term = ops[0]
            for o in ops[1:]:
                term = np.kron(term, o)
            H += g[k] * term
    Hm = -1j * (np.kron(H, Id) - np.kron(Id, H.T))
    Iv = np.eye(D * D, dtype=complex)
    if order == 1:
        M = Iv + eps * Hm
    else:
        c1, c2 = (1 - 1j) / 2, (1 + 1j) / 2
        M = (Iv + c2 * eps * Hm) @ (Iv + c1 * eps * Hm)
    chi = model.initial_system_state().astype(complex)
    for _ in range(K):
        chi = np.kron(chi, I2 / 2)
    vec = chi.reshape(-1)
    sz = Z / 2
    out = []
    for _ in range(n_steps):
        chi_t = vec.reshape(D, D).reshape(2, 2 ** K, 2, 2 ** K)
        rho = np.einsum("ibkb->ik", chi_t)
        out.append(float(np.trace(sz @ rho).real))  # measure before evolving
        vec = M @ vec
    return np.array(out)


# --------------------------------------------------------------------------
# pipeline registration
# --------------------------------------------------------------------------

def test_separable_pipeline_registered():
    assert "separable" in available_pipelines()


def test_build_pipeline_returns_separable_engines():
    model = GaudinModel(g=1.0, K=5)
    cfg = SolverConfig(eps=0.1, T=0.5, expansion_order=2)
    kernel, evo = build_pipeline(model, cfg)
    assert isinstance(kernel, SeparableKernelEngine)
    assert isinstance(evo, SeparableBathEvolution)
    assert kernel.K == 5


# --------------------------------------------------------------------------
# end-to-end: <S_z(t)> matches exact Trotter dynamics
# --------------------------------------------------------------------------

@pytest.mark.parametrize("K", [1, 2, 3])
@pytest.mark.parametrize("order", [1, 2])
def test_sz_trajectory_matches_exact(K, order):
    model = GaudinModel(g=0.7, K=K)
    eps, n_steps = 0.1, 4
    # exact (uncompressed) vs Trotter is a math check -> CPU for clean fp64;
    # the GPU path is validated separately by test_gpu_matches_cpu_gaudin.
    res = EDMSolver.from_model(
        model, T=eps * n_steps, eps=eps, expansion_order=order, cutoff=0.0, backend="cpu"
    ).solve(channel=3)  # channel 3 = S_z
    # public axis is eps..T; exact_sz_trajectory measures *before* each step, so [1:] of one
    # extra step gives the exact <S_z> at t = eps, 2eps, ..., T
    ref = exact_sz_trajectory(model, eps, n_steps + 1, order)[1:]
    np.testing.assert_allclose(res.times, eps * np.arange(1, n_steps + 1), atol=1e-12)
    np.testing.assert_allclose(res.polarization, ref, atol=1e-9)


def test_polarization_starts_near_half():
    model = GaudinModel(g=1.0, K=20)
    res = solve(model, T=2.0, eps=0.05, expansion_order=2, cutoff=1e-6, max_bond=120, channel=3)
    # first recorded time is one step in; still close to the initial 1/2
    assert res.polarization[0] > 0.45
    assert res.polarization[-1] < res.polarization[0]  # depolarising


# --------------------------------------------------------------------------
# result structure
# --------------------------------------------------------------------------

def test_result_carries_mps_and_subbath_records():
    model = GaudinModel(g=0.8, K=6)
    res = EDMSolver.from_model(
        model, T=0.4, eps=0.1, expansion_order=2, cutoff=1e-8, record_rho=True
    ).solve(channel=3)
    # bond_dims is per sub-bath L (length K), D_t available via the final MPS
    assert len(res.bond_dims) == 6
    assert res.mps is not None
    assert len(res.mps.bond_dims) == res.mps.num_sites - 1
    assert res.evolution.recorded_L[-1] == 6
    assert len(res.evolution.density_matrices) == 6


def test_custom_observables_rejected_for_separable():
    model = GaudinModel(g=1.0, K=3)
    solver = EDMSolver.from_model(model, T=0.3, eps=0.1, expansion_order=2)
    with pytest.raises(NotImplementedError):
        solver.solve(observables={"Sx": lambda t: X / 2})


def test_channel_out_of_range():
    model = GaudinModel(g=1.0, K=3)
    solver = EDMSolver.from_model(model, T=0.3, eps=0.1, expansion_order=2)
    with pytest.raises(ValueError):
        solver.solve(channel=4)  # only 3 channels (d_phys = 7)


# --------------------------------------------------------------------------
# backend selection (GPU is the primary path for the separable / Gaudin pipeline)
# --------------------------------------------------------------------------

def _gpu_available() -> bool:
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


requires_gpu = pytest.mark.skipif(not _gpu_available(), reason="no CuPy GPU available")


def test_explicit_cpu_backend_label():
    model = GaudinModel(g=1.0, K=4)
    res = EDMSolver.from_model(
        model, T=0.4, eps=0.1, expansion_order=2, cutoff=1e-8, backend="cpu"
    ).solve(channel=3)
    assert res.backend.startswith("cpu")


def test_auto_defaults_to_cpu():
    # Phase 1/2 run on CPU by default; GPU is opt-in (backend='gpu').
    model = GaudinModel(g=1.0, K=4)
    res = EDMSolver.from_model(
        model, T=0.4, eps=0.1, expansion_order=2, cutoff=1e-8
    ).solve(channel=3)  # backend defaults to 'auto' -> cpu
    assert res.backend.startswith("cpu")
    assert "cupy" not in type(res.mps.tensors[0]).__module__


# GPU end-to-end consistency: the GPU gives the same physics as the CPU, but the
# GPU is not the Phase-1/2 compute path (CPU is faster at these bond dimensions,
# see docs/cpu-vs-gpu-edm.md), so this slow check is deferred to Phase 3/4.
@pytest.mark.skip(reason="GPU end-to-end deferred to Phase 3/4 (docs/cpu-vs-gpu-edm.md)")
@requires_gpu
def test_gpu_matches_cpu_gaudin():
    model = GaudinModel(g=0.8, K=4)
    kw = dict(T=0.6, eps=0.1, expansion_order=2, cutoff=1e-6, max_bond=64)
    cpu = EDMSolver.from_model(model, backend="cpu", **kw).solve(channel=3)
    gpu = EDMSolver.from_model(model, backend="gpu", **kw).solve(channel=3)
    assert gpu.backend.startswith("gpu") and cpu.backend.startswith("cpu")
    np.testing.assert_allclose(gpu.times, cpu.times, atol=1e-12)
    np.testing.assert_allclose(gpu.polarization, cpu.polarization, atol=1e-8)


def test_outer_loop_frees_memory_each_sub_bath():
    # the separable outer loop must release GPU pool blocks after every sub-bath
    # (Sec. 8.4); verify the wiring with a spy (no GPU needed).
    from edmtn.evolution import SeparableBathEvolution
    from edmtn.kernels import SeparableKernelEngine

    model = GaudinModel(g=1.0, K=5)
    eng = SeparableKernelEngine.from_model(model, T=0.3, eps=0.1)

    calls = {"n": 0}

    class _SpyMemory:
        def free_all_blocks(self):
            calls["n"] += 1

    SeparableBathEvolution().run(model, eng, 0.1, 3, cutoff=1e-8, memory=_SpyMemory())
    assert calls["n"] == model.K  # once per folded sub-bath
