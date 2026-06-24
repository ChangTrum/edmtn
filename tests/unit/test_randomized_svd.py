"""Regression tests for the RandomizedSVD strategy (Layer 4a, Strategy B).

Locks in the validated behaviour before it is load-bearing (P6):

* single-pass (``n_iter=0``) and cold (``n_iter=2``) reconstruct and truncate,
* the spectral resolution guard recovers a rank larger than the initial sketch,
* single-pass is seed-stable (same kept bond, matching reconstruction),
* cold rSVD reproduces the *exact* StandardSVD bond on a slowly decaying spectrum,
* the rel_ref (paper) cutoff selects the same rank as StandardSVD,
* end-to-end: dropping RandomizedSVD into the solver matches the StandardSVD
  pipeline's <S_z(t)> to below the cutoff.
"""

import numpy as np
import pytest

from edmtn.decomposition import DecompositionResult, RandomizedSVD, StandardSVD


def _gpu_available() -> bool:
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


requires_gpu = pytest.mark.skipif(not _gpu_available(), reason="no CuPy GPU available")


def matrix_with_singulars(svals, seed=0, complex_=True):
    """Build a (complex) matrix with prescribed (descending) singular values."""
    rng = np.random.default_rng(seed)
    svals = np.asarray(svals, dtype=np.float64)
    n = len(svals)
    m = n + 3
    if complex_:
        U, _ = np.linalg.qr(rng.standard_normal((m, n)) + 1j * rng.standard_normal((m, n)))
        V, _ = np.linalg.qr(rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))
        return (U * svals) @ V.conj().T
    U, _ = np.linalg.qr(rng.standard_normal((m, n)))
    V, _ = np.linalg.qr(rng.standard_normal((n, n)))
    return (U * svals) @ V.T


def _host(a):
    return np.asarray(a)


# --------------------------------------------------------------------------
# reconstruction / absorb
# --------------------------------------------------------------------------

@pytest.mark.parametrize("n_iter", [0, 2])
def test_reconstruction_no_cutoff(n_iter):
    M = matrix_with_singulars([1.0, 0.5, 0.25, 0.1])
    r = RandomizedSVD(n_iter=n_iter).compress(M, cutoff=0.0, absorb="both")
    assert r.bond == 4
    np.testing.assert_allclose(_host(r.left @ r.right), M, atol=1e-10)


@pytest.mark.parametrize("absorb", [None, "left", "right", "both"])
def test_absorb_modes_reconstruct(absorb):
    M = matrix_with_singulars([2.0, 1.0, 0.4])
    r = RandomizedSVD(n_iter=2).compress(M, cutoff=0.0, absorb=absorb)
    recon = (r.left * r.s) @ r.right if absorb is None else r.left @ r.right
    np.testing.assert_allclose(_host(recon), M, atol=1e-10)


def test_real_matrix_reconstructs():
    M = matrix_with_singulars([1.0, 0.6, 0.3, 0.1], complex_=False)
    r = RandomizedSVD(n_iter=2).compress(M, cutoff=0.0, absorb="both")
    np.testing.assert_allclose(_host(r.left @ r.right), M, atol=1e-10)


def test_singular_values_descending():
    M = matrix_with_singulars([3.0, 1.5, 0.7, 0.2])
    r = RandomizedSVD(n_iter=2).compress(M, cutoff=0.0, absorb=None)
    s = _host(r.s).real
    np.testing.assert_allclose(s, [3.0, 1.5, 0.7, 0.2], atol=1e-9)


# --------------------------------------------------------------------------
# the resolution guard
# --------------------------------------------------------------------------

def test_resolution_guard_recovers_large_rank():
    # 40 significant values then a cliff; a fresh strategy starts its sketch at
    # ~16 and must grow until the cutoff bites inside the resolved spectrum.
    svals = list(np.linspace(1.0, 0.2, 40)) + [1e-12] * 5
    M = matrix_with_singulars(svals)
    r = RandomizedSVD(n_iter=0).compress(M, cutoff=1e-6, cutoff_mode="rel")
    assert r.bond == 40
    assert np.linalg.norm(_host(r.left @ r.right) - M) < 1e-6


def test_max_bond_cap_respected():
    M = matrix_with_singulars([1.0, 0.5, 0.3, 0.2, 0.1])
    r = RandomizedSVD(n_iter=2).compress(M, max_bond=3, cutoff=0.0, absorb="both")
    assert r.bond == 3
    assert r.info["max_bond_hit"] is True


# --------------------------------------------------------------------------
# agreement with StandardSVD
# --------------------------------------------------------------------------

def test_rel_ref_rank_matches_standard():
    # the production cutoff rule (paper rel_ref) must select the same rank
    svals = [1.0, 0.5, 0.3, 0.2, 0.1, 1e-2, 1e-4, 1e-7, 1e-9]
    M = matrix_with_singulars(svals)
    kw = dict(cutoff=1e-5, cutoff_mode="rel_ref", ref_index=4, absorb="left")
    std = StandardSVD().compress(M, **kw)
    cold = RandomizedSVD(n_iter=2).compress(M, **kw)
    assert cold.bond == std.bond
    np.testing.assert_allclose(_host(cold.left @ cold.right),
                               _host(std.left @ std.right), atol=1e-9)


def test_cold_matches_standard_bond_slow_decay():
    # geometric (slowly decaying) spectrum: cold rSVD reproduces the exact bond
    svals = list(0.75 ** np.arange(20))
    M = matrix_with_singulars(svals)
    kw = dict(cutoff=1e-6, cutoff_mode="rel")
    std = StandardSVD().compress(M, **kw)
    cold = RandomizedSVD(n_iter=2).compress(M, **kw)
    assert cold.bond == std.bond


def test_single_pass_never_under_retains():
    # single-pass may over-retain a little but must never keep fewer than the
    # exact rank (that would mean a dropped direction)
    svals = list(0.75 ** np.arange(20))
    M = matrix_with_singulars(svals)
    kw = dict(cutoff=1e-6, cutoff_mode="rel")
    std = StandardSVD().compress(M, **kw)
    single = RandomizedSVD(n_iter=0).compress(M, **kw)
    assert single.bond >= std.bond
    assert np.linalg.norm(_host(single.left @ single.right) - M) < 1e-5


# --------------------------------------------------------------------------
# seed stability
# --------------------------------------------------------------------------

def test_single_pass_seed_stable():
    M = matrix_with_singulars([1.0, 0.5, 0.3, 0.2, 0.1, 1e-2, 1e-4, 1e-8])
    kw = dict(cutoff=1e-5, cutoff_mode="rel_ref", ref_index=4, absorb="left")
    r0 = RandomizedSVD(n_iter=0, seed=0).compress(M, **kw)
    r1 = RandomizedSVD(n_iter=0, seed=12345).compress(M, **kw)
    assert r0.bond == r1.bond
    np.testing.assert_allclose(_host(r0.left @ r0.right),
                               _host(r1.left @ r1.right), atol=1e-7)


# --------------------------------------------------------------------------
# interface parity with StandardSVD
# --------------------------------------------------------------------------

def test_info_fields_present():
    M = matrix_with_singulars([1.0, 0.5, 0.1])
    r = RandomizedSVD().compress(M, cutoff=0.0)
    assert isinstance(r, DecompositionResult)
    for key in ("bond", "error", "discarded_weight", "n_singular", "cutoff_mode",
                "max_bond_hit", "n_iter"):
        assert key in r.info


def test_renorm_preserves_frobenius_norm():
    M = matrix_with_singulars([1.0, 0.5, 0.3, 0.2])
    fro = np.linalg.norm(M)
    r = RandomizedSVD(n_iter=2).compress(M, max_bond=2, cutoff=0.0, absorb="both", renorm=True)
    assert np.isclose(np.linalg.norm(_host(r.left @ r.right)), fro, atol=1e-8)


def test_non_2d_raises():
    with pytest.raises(ValueError):
        RandomizedSVD().compress(np.zeros((2, 2, 2)))


def test_invalid_absorb_raises():
    with pytest.raises(ValueError):
        RandomizedSVD().compress(matrix_with_singulars([1.0, 0.5]), absorb="sideways")


def test_negative_n_iter_raises():
    with pytest.raises(ValueError):
        RandomizedSVD(n_iter=-1)


def test_backend_autoselect_numpy():
    r = RandomizedSVD()
    r.compress(matrix_with_singulars([1.0, 0.5]), cutoff=0.0)
    assert "numpy" in r._cache


# --------------------------------------------------------------------------
# end-to-end: drop-in for StandardSVD in the solver
# --------------------------------------------------------------------------

@pytest.mark.parametrize("n_iter", [0, 2])
def test_solver_matches_pipeline_below_cutoff(n_iter):
    from edmtn.driver import EDMSolver
    from edmtn.models import GaudinModel

    model = GaudinModel(g=1.0, K=3)
    cutoff = 1e-6
    common = dict(T=2.0, eps=0.5, expansion_order=2, cutoff=cutoff,
                  max_bond=200, backend="cpu")
    base = EDMSolver.from_model(model, decomposition=StandardSVD(), **common).solve(channel=3)
    rsvd = EDMSolver.from_model(
        model, decomposition=RandomizedSVD(n_iter=n_iter), **common).solve(channel=3)
    n = min(len(base.polarization), len(rsvd.polarization))
    err = np.max(np.abs(np.asarray(base.polarization[:n]) - np.asarray(rsvd.polarization[:n])))
    assert err < cutoff


@requires_gpu
def test_gpu_reconstruction_and_device():
    import cupy as cp

    M = matrix_with_singulars([1.0, 0.5, 0.25, 0.1])
    Mg = cp.asarray(M)
    r = RandomizedSVD(n_iter=2).compress(Mg, cutoff=0.0, absorb="both")
    assert type(r.left).__module__.split(".")[0] == "cupy"
    np.testing.assert_allclose(cp.asnumpy(r.left @ r.right), M, atol=1e-9)
