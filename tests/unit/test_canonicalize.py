"""Canonicalisation-strategy tests (P2): CholeskyQR vs the Householder reference.

Locks the behaviour before it is load-bearing: CholeskyQR2 must (a) produce a
left-canonical MPS to machine precision, (b) leave the represented state unchanged,
(c) drive the solver to the same ``<S_z(t)>`` as the default Householder path
(`< xi`), and (d) fall back to Householder on rank-deficient bonds.  Mirrors the
P6 RandomizedSVD tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from edmtn.evolution import CholeskyQR, HouseholderQR, left_canonicalize
from edmtn.evolution.mps_utils import EDMMPS
from edmtn.driver.solver import solve
from edmtn.models import GaudinModel


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _random_edmmps(n, d, d_phys, chi, rng, *, wide_first=False):
    """A random EDM-MPS chain (open boundaries dim 1) for canonicalisation tests."""
    tensors = []
    left = 1
    for p in range(n):
        right = 1 if p == n - 1 else chi
        # optionally make the first internal bond wide (m < n) to exercise fallback
        if wide_first and p == 0:
            right = d_phys * left + 5
        t = (rng.standard_normal((d_phys, left, right))
             + 1j * rng.standard_normal((d_phys, left, right))).astype(np.complex128)
        tensors.append(t)
        left = right
    return EDMMPS(tensors=tensors, d=d, d_phys=d_phys, rho0_vec=np.ones(1, np.complex128))


def _dense(mps):
    """Contract the MPS to its full dense vector (for state-equality checks)."""
    v = mps.tensors[0]                       # (phi, 1, chi)
    out = v.reshape(v.shape[0], v.shape[2])  # (phi, chi)
    for t in mps.tensors[1:]:
        out = np.einsum("...a,pab->...pb", out, t)
        out = out.reshape(-1, out.shape[-1])
    return out.reshape(-1)


def _max_left_ortho_err(mps):
    """Max ||Q^H Q - I|| over the left-canonical sites 0..n-2."""
    err = 0.0
    for p in range(mps.num_sites - 1):
        dp, chil, chir = mps.tensors[p].shape
        Q = mps.tensors[p].reshape(dp * chil, chir)
        err = max(err, float(np.max(np.abs(Q.conj().T @ Q - np.eye(chir)))))
    return err


# --------------------------------------------------------------------------
# orthogonality + state preservation
# --------------------------------------------------------------------------

@pytest.mark.parametrize("passes", [1, 2])
def test_cholqr_left_canonical_and_state_preserved(passes):
    rng = np.random.default_rng(0)
    mps = _random_edmmps(6, 2, 7, 12, rng)
    before = _dense(mps)
    CholeskyQR(passes=passes).left_canonicalize(mps)
    # left-canonical to (near) machine precision for passes=2
    tol = 1e-10 if passes == 2 else 1e-4
    assert _max_left_ortho_err(mps) < tol
    after = _dense(mps)
    # the represented state is unchanged up to global scale/phase
    ov = np.vdot(before, after) / (np.linalg.norm(before) * np.linalg.norm(after))
    assert abs(abs(ov) - 1.0) < 1e-10


def test_cholqr2_matches_householder_orthogonality():
    rng = np.random.default_rng(1)
    base = _random_edmmps(7, 2, 7, 16, rng)
    a, b = base.copy(), base.copy()
    HouseholderQR().left_canonicalize(a)
    cq = CholeskyQR(passes=2)
    cq.left_canonicalize(b)
    assert _max_left_ortho_err(a) < 1e-10
    assert _max_left_ortho_err(b) < 1e-10
    assert cq.last_ortho_err < 1e-10


def test_cholqr_fallback_on_wide_bond():
    """A wide/short bond (m < n) must fall back to Householder QR and be counted."""
    rng = np.random.default_rng(2)
    mps = _random_edmmps(5, 2, 7, 10, rng, wide_first=True)
    cq = CholeskyQR(passes=2)
    cq.left_canonicalize(mps)
    assert cq.last_fallback >= 1
    assert _max_left_ortho_err(mps) < 1e-10        # still correctly orthonormal


def test_left_canonicalize_dispatch_default_is_householder():
    """left_canonicalize(canon=None) and canon=HouseholderQR() agree exactly."""
    rng = np.random.default_rng(3)
    base = _random_edmmps(6, 2, 7, 12, rng)
    a, b = base.copy(), base.copy()
    left_canonicalize(a)                            # default path
    left_canonicalize(b, canon=HouseholderQR())     # explicit strategy
    for ta, tb in zip(a.tensors, b.tensors):
        assert ta.shape == tb.shape
        np.testing.assert_allclose(ta, tb, rtol=1e-12, atol=1e-12)


# --------------------------------------------------------------------------
# end-to-end through the solver
# --------------------------------------------------------------------------

@pytest.mark.parametrize("passes", [1, 2])
def test_cholqr_end_to_end_matches_householder(passes):
    """Gaudin <S_z(t)> with CholeskyQR canonicalisation matches the default path < xi."""
    model = GaudinModel(g=1.0, K=12)
    common = dict(T=3.0, eps=0.2, expansion_order=2, cutoff=1e-6, max_bond=400, channel=3)
    ref = solve(model, **common)
    got = solve(model, canonicalization=CholeskyQR(passes=passes), **common)
    n = min(len(ref.polarization), len(got.polarization))
    err = float(np.max(np.abs(np.asarray(ref.polarization[:n])
                              - np.asarray(got.polarization[:n]))))
    assert err < 1e-6
    assert max(got.bond_dims) == max(ref.bond_dims)   # canonicalisation does not change bonds


@pytest.mark.skipif(True, reason="no CuPy GPU available")
def test_cholqr_gpu():  # pragma: no cover - exercised on the GPU node
    import cupy as cp  # noqa: PLC0415

    rng = cp.random.default_rng(0)
    n, d_phys, chi = 6, 7, 12
    tensors, left = [], 1
    for p in range(n):
        right = 1 if p == n - 1 else chi
        tensors.append((rng.standard_normal((d_phys, left, right))
                        + 1j * rng.standard_normal((d_phys, left, right))).astype(cp.complex128))
        left = right
    mps = EDMMPS(tensors=tensors, d=2, d_phys=d_phys, rho0_vec=cp.ones(1, cp.complex128))
    cq = CholeskyQR(passes=2)
    cq.left_canonicalize(mps)
    assert cq.last_ortho_err < 1e-10
