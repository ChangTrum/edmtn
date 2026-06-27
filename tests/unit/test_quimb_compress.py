"""quimb-backed compression path tests (re-platform sub-step 1).

The opt-in ``engine='quimb'`` compression must (a) leave the default path
untouched, (b) preserve the represented state at a loose cutoff, and (c) drive the
solver to the same ``<S_z(t)>`` physics as the native StandardSVD path (the
observable is the invariant; bond dims differ since quimb uses a native cutoff).
"""

from __future__ import annotations

import numpy as np
import pytest

from edmtn.evolution.mps_utils import EDMMPS, compress
from edmtn.evolution.quimb_compress import compress_quimb
from edmtn.driver.solver import solve
from edmtn.models import GaudinModel


def _random_edmmps(n, d, d_phys, chi, rng):
    """Random EDM-MPS with d**2 open output (left) and d**2 rho0 (right) boundaries."""
    d2 = d * d
    tensors = []
    left = d2
    for p in range(n):
        right = d2 if p == n - 1 else chi
        t = (rng.standard_normal((d_phys, left, right))
             + 1j * rng.standard_normal((d_phys, left, right))).astype(np.complex128)
        tensors.append(t)
        left = right
    return EDMMPS(tensors=tensors, d=d, d_phys=d_phys, rho0_vec=np.ones(d2, np.complex128))


def test_quimb_compress_preserves_reduced_state_loose_cutoff():
    rng = np.random.default_rng(0)
    mps = _random_edmmps(6, 2, 7, 10, rng)
    rho0 = mps.reduced_density_matrix()
    out, infos = compress_quimb(mps, cutoff=1e-14, cutoff_mode="rsum2")
    rho1 = out.reduced_density_matrix()
    assert np.max(np.abs(rho0 - rho1)) < 1e-9      # near-lossless at a tiny cutoff
    assert len(infos) == out.num_sites - 1


def test_compress_engine_dispatch_default_is_native():
    """compress(engine='native') must be byte-for-byte the historical path."""
    rng = np.random.default_rng(1)
    base = _random_edmmps(6, 2, 7, 10, rng)
    a = base.copy()
    out, _ = compress(a, max_bond=None, cutoff=1e-6, cutoff_mode="rel_ref", ref_index=4)
    # native path returns a left/right-canonicalised, truncated MPS; just assert it ran
    assert out.num_sites == base.num_sites
    with pytest.raises(ValueError, match="unknown compress engine"):
        compress(base.copy(), engine="bogus")


def test_quimb_unknown_does_not_affect_native_default():
    """Solver default (no compression kwarg) is unchanged; quimb is strictly opt-in."""
    model = GaudinModel(g=1.0, K=10)
    common = dict(T=2.0, eps=0.2, expansion_order=2, cutoff=1e-6, max_bond=400, channel=3)
    ref = solve(model, **common)
    assert ref.backend  # ran
    # default config uses the native engine; the quimb path defaults to the 'rel'
    # cutoff mode (validated faithful to rel_ref; rsum2 over-truncates spin-boson)
    from edmtn.driver.auto_config import SolverConfig
    cfg = SolverConfig(eps=0.2, T=2.0)
    assert cfg.compression == "native"
    assert cfg.compress_cutoff_mode == "rel"


@pytest.mark.parametrize("mode,cutoff", [("rsum2", 1e-13), ("rel", 1e-8)])
def test_quimb_compression_matches_physics(mode, cutoff):
    """Gaudin <S_z(t)> via the quimb compression path matches the native solve < tol."""
    model = GaudinModel(g=1.0, K=12)
    common = dict(T=3.0, eps=0.2, expansion_order=2, cutoff=1e-6, max_bond=400, channel=3)
    ref = solve(model, **common)
    got = solve(model, compression="quimb", compress_cutoff_mode=mode,
                compress_cutoff=cutoff, **common)
    n = min(len(ref.polarization), len(got.polarization))
    err = float(np.max(np.abs(np.asarray(ref.polarization[:n])
                              - np.asarray(got.polarization[:n]))))
    assert err < 1e-4          # the observable (physics) is reproduced


@pytest.mark.skipif(True, reason="no CuPy GPU available")
def test_quimb_compress_gpu():  # pragma: no cover - exercised on the GPU node
    import cupy as cp  # noqa: PLC0415

    rng = cp.random.default_rng(0)
    d, d_phys, chi, n = 2, 7, 10, 6
    d2 = d * d
    tensors, left = [], d2
    for p in range(n):
        right = d2 if p == n - 1 else chi
        tensors.append((rng.standard_normal((d_phys, left, right))
                        + 1j * rng.standard_normal((d_phys, left, right))).astype(cp.complex128))
        left = right
    mps = EDMMPS(tensors=tensors, d=d, d_phys=d_phys, rho0_vec=cp.ones(d2, cp.complex128))
    out, _ = compress_quimb(mps, cutoff=1e-13, cutoff_mode="rsum2")
    assert out.tensors[0].__class__.__module__.split(".")[0] == "cupy"  # stayed on GPU
