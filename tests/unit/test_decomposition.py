"""Unit tests for Layer 4a (decomposition strategies)."""

import numpy as np
import pytest

from edmtn.decomposition import DecompositionResult, StandardSVD, truncation_rank


def _gpu_available() -> bool:
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


requires_gpu = pytest.mark.skipif(not _gpu_available(), reason="no CuPy GPU available")


def matrix_with_singulars(svals, seed=0):
    """Build a complex matrix with prescribed (descending) singular values."""
    rng = np.random.default_rng(seed)
    svals = np.asarray(svals, dtype=np.float64)
    n = len(svals)
    m = n + 3
    U, _ = np.linalg.qr(rng.standard_normal((m, n)) + 1j * rng.standard_normal((m, n)))
    V, _ = np.linalg.qr(rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))
    return (U * svals) @ V.conj().T


@pytest.fixture
def svd():
    return StandardSVD()


# --------------------------------------------------------------------------
# truncation_rank (pure host logic)
# --------------------------------------------------------------------------

S = np.array([1.0, 0.5, 0.3, 0.2, 0.1, 1e-3, 1e-7])


def test_rank_no_cutoff_keeps_all():
    assert truncation_rank(S, cutoff=0.0) == len(S)


def test_rank_abs():
    assert truncation_rank(S, cutoff=0.15, cutoff_mode="abs") == 4  # >0.15: 1,.5,.3,.2


def test_rank_rel():
    # relative to s[0]=1, cutoff 1e-2 -> keep s>0.01: 1,.5,.3,.2,.1
    assert truncation_rank(S, cutoff=1e-2, cutoff_mode="rel") == 5


def test_rank_rel_ref_paper_rule():
    # ref_index=4 -> s_ref=0.1, cutoff 1e-5 -> keep s>1e-6: drops only 1e-7
    assert truncation_rank(S, cutoff=1e-5, cutoff_mode="rel_ref", ref_index=4) == 6


def test_rank_rel_ref_keeps_reference_block():
    # the paper rule always keeps at least ref_index+1 values
    keep = truncation_rank(S, cutoff=0.9, cutoff_mode="rel_ref", ref_index=4)
    assert keep >= 5


def test_rank_sum2():
    # discard tail sum of squares <= cutoff^2; tail of [1e-3,1e-7]^2 ~ 1e-6
    assert truncation_rank(S, cutoff=2e-3, cutoff_mode="sum2") == 5


def test_rank_rsum2_relative():
    keep = truncation_rank(S, cutoff=1e-3, cutoff_mode="rsum2")
    assert 1 <= keep <= len(S)


def test_rank_max_bond_cap():
    assert truncation_rank(S, cutoff=0.0, max_bond=3) == 3


def test_rank_keeps_at_least_one():
    assert truncation_rank(S, cutoff=1e3, cutoff_mode="abs") == 1


def test_rank_unknown_mode_raises():
    with pytest.raises(ValueError):
        truncation_rank(S, cutoff=1e-3, cutoff_mode="bogus")


# --------------------------------------------------------------------------
# StandardSVD reconstruction / absorb
# --------------------------------------------------------------------------

def test_exact_reconstruction_no_cutoff(svd):
    M = matrix_with_singulars([1.0, 0.5, 0.25, 0.1])
    r = svd.compress(M, cutoff=0.0, absorb="both")
    np.testing.assert_allclose(r.left @ r.right, M, atol=1e-12)
    assert r.bond == 4


@pytest.mark.parametrize("absorb", [None, "left", "right", "both"])
def test_absorb_modes_reconstruct(svd, absorb):
    M = matrix_with_singulars([2.0, 1.0, 0.4])
    r = svd.compress(M, cutoff=0.0, absorb=absorb)
    if absorb is None:
        recon = (r.left * r.s) @ r.right
    else:
        recon = r.left @ r.right
    np.testing.assert_allclose(recon, M, atol=1e-12)


def test_singular_values_descending_and_correct(svd):
    M = matrix_with_singulars([3.0, 1.5, 0.7, 0.2])
    r = svd.compress(M, cutoff=0.0, absorb=None)
    np.testing.assert_allclose(_host(r.s), [3.0, 1.5, 0.7, 0.2], atol=1e-12)


def test_low_rank_truncation(svd):
    M = matrix_with_singulars([1.0, 0.5, 0.3, 1e-9, 1e-12])
    r = svd.compress(M, cutoff=1e-6, cutoff_mode="rel", absorb="both")
    assert r.bond == 3
    # truncated reconstruction is close to the original (small discarded weight)
    assert np.linalg.norm(_host(r.left @ r.right) - M) < 1e-6


def test_truncation_error_matches_discarded_norm(svd):
    M = matrix_with_singulars([1.0, 0.5, 0.3, 0.05, 0.01])
    r = svd.compress(M, max_bond=3, cutoff=0.0, absorb="both")
    expected = np.sqrt(0.05**2 + 0.01**2)
    assert np.isclose(r.error, expected, atol=1e-9)
    assert r.info["max_bond_hit"] is True


def test_renorm_preserves_frobenius_norm(svd):
    M = matrix_with_singulars([1.0, 0.5, 0.3, 0.2])
    fro = np.linalg.norm(M)
    r = svd.compress(M, max_bond=2, cutoff=0.0, absorb="both", renorm=True)
    assert np.isclose(np.linalg.norm(_host(r.left @ r.right)), fro, atol=1e-9)


def test_info_fields(svd):
    M = matrix_with_singulars([1.0, 0.5, 0.1])
    r = svd.compress(M, cutoff=0.0)
    assert isinstance(r, DecompositionResult)
    for key in ("bond", "error", "discarded_weight", "n_singular", "cutoff_mode", "max_bond_hit"):
        assert key in r.info


def test_non_2d_raises(svd):
    with pytest.raises(ValueError):
        svd.compress(np.zeros((2, 2, 2)))


def test_backend_autoselect_numpy(svd):
    M = matrix_with_singulars([1.0, 0.5])
    svd.compress(M, cutoff=0.0)
    assert "numpy" in svd._cache


# --------------------------------------------------------------------------
# GPU path
# --------------------------------------------------------------------------

@requires_gpu
def test_gpu_reconstruction_and_device():
    import cupy as cp

    M = matrix_with_singulars([1.0, 0.5, 0.25, 0.1])
    Mg = cp.asarray(M)
    svd = StandardSVD()
    r = svd.compress(Mg, cutoff=0.0, absorb="both")
    assert type(r.left).__module__.split(".")[0] == "cupy"
    np.testing.assert_allclose(cp.asnumpy(r.left @ r.right), M, atol=1e-10)
    assert "cupy" in svd._cache


@requires_gpu
def test_gpu_matches_cpu_truncation():
    import cupy as cp

    M = matrix_with_singulars([1.0, 0.5, 0.3, 0.2, 0.1, 1e-3, 1e-7])
    cpu = StandardSVD().compress(M, cutoff=1e-5, cutoff_mode="rel_ref", ref_index=4)
    gpu = StandardSVD().compress(cp.asarray(M), cutoff=1e-5, cutoff_mode="rel_ref", ref_index=4)
    assert cpu.bond == gpu.bond


def _host(a):
    if type(a).__module__.split(".")[0] == "cupy":
        import cupy as cp

        return cp.asnumpy(a)
    return np.asarray(a)
