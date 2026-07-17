"""Centralized SolverConfig validation (P0-1).

Every knob is validated at construction in ``SolverConfig.__post_init__`` so an
illegal value fails loudly and immediately at the config entry point -- not as a
divide-by-zero / int-cast / silent round-down deep inside a pipeline, and with the
SAME error on both tracks (cpu/gpu Track 1 and the hpc Track 2). The config is a
frozen dataclass; ``n_steps`` is validated once and cached.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from edmtn.driver.auto_config import SolverConfig


# -- time parameters -------------------------------------------------------

@pytest.mark.parametrize("eps", [0, -0.1, np.nan, np.inf])
def test_bad_eps_rejected(eps):
    with pytest.raises(ValueError):
        SolverConfig(eps=eps, T=1.0)


@pytest.mark.parametrize("T", [0, -1, np.nan, np.inf])
def test_bad_T_rejected(T):
    with pytest.raises(ValueError):
        SolverConfig(eps=0.1, T=T)


def test_non_integer_ratio_rejected():
    with pytest.raises(ValueError):
        SolverConfig(eps=0.3, T=1.0)


@pytest.mark.parametrize("eps,T", [(1e-308, 1e308), (1e308, 1e-308)])
def test_ratio_overflow_underflow_rejected(eps, T):
    # T/eps overflows to inf or underflows to 0 -> our ValueError, never a leaked
    # OverflowError / ZeroDivisionError / int-conversion error
    with pytest.raises(ValueError):
        SolverConfig(eps=eps, T=T)


@pytest.mark.parametrize("field", ["eps", "T", "cutoff"])
def test_huge_int_float_overflow_rejected(field):
    # a huge Python int (10**400) is a real number but overflows float(); the float
    # conversion must raise our ValueError, never leak the raw OverflowError (P1-11 /
    # the direct-call vs driver-path consistent-exception contract).
    kwargs = {"eps": 0.1, "T": 0.5, field: 10 ** 400}
    with pytest.raises(ValueError):
        SolverConfig(**kwargs)


# -- truncation / bond parameters ------------------------------------------

@pytest.mark.parametrize("cutoff", [-1, np.nan, np.inf])
def test_bad_cutoff_rejected(cutoff):
    with pytest.raises(ValueError):
        SolverConfig(eps=0.1, T=1.0, cutoff=cutoff)


@pytest.mark.parametrize("max_bond", [0, -1, 1.5, True])
def test_bad_max_bond_rejected(max_bond):
    with pytest.raises(ValueError):
        SolverConfig(eps=0.1, T=1.0, max_bond=max_bond)


@pytest.mark.parametrize("q", [-1, 1.5, True])
def test_bad_compress_decomp_q_rejected(q):
    with pytest.raises(ValueError):
        SolverConfig(eps=0.1, T=1.0, compress_decomp_q=q)


# -- order and boolean parameters ------------------------------------------

@pytest.mark.parametrize("order", [0, 3, 1.5, True])
def test_bad_expansion_order_rejected(order):
    with pytest.raises(ValueError):
        SolverConfig(eps=0.1, T=1.0, expansion_order=order)


@pytest.mark.parametrize("record_rho", [1, "yes", np.bool_(True)])
def test_bad_record_rho_rejected(record_rho):
    with pytest.raises(ValueError):
        SolverConfig(eps=0.1, T=1.0, record_rho=record_rho)


# -- enum parameters -------------------------------------------------------

@pytest.mark.parametrize("kwargs", [
    dict(backend="auto"),
    dict(backend="cuda"),
    dict(precision="f64x"),
    dict(cutoff_mode="rel_ref"),
    dict(compress_method="bogus"),
    dict(compress_decomp="bogus"),
    dict(compress_canon="bogus"),
    dict(pathfinder="bogus"),
    dict(preset="bogus"),
])
def test_bad_enum_rejected(kwargs):
    with pytest.raises(ValueError):
        SolverConfig(eps=0.1, T=1.0, **kwargs)


# -- sub_baths (type + positivity only; K-awareness is P0-10) --------------

@pytest.mark.parametrize("sub_baths", [0, -1, 1.5, True, "3"])
def test_bad_sub_baths_rejected(sub_baths):
    with pytest.raises(ValueError):
        SolverConfig(eps=0.1, T=1.0, sub_baths=sub_baths)


# -- time_windows: NotImplementedError (concept valid, feature not shipped) -

@pytest.mark.parametrize("time_windows", [0, 1, -1, "2"])
def test_time_windows_not_implemented(time_windows):
    with pytest.raises(NotImplementedError):
        SolverConfig(eps=0.1, T=1.0, time_windows=time_windows)


# -- preset legality is checked on the hpc track too (but not applied) ------

def test_hpc_preset_validated_but_not_applied():
    for preset in ("balanced", "robust"):
        cfg = SolverConfig(eps=0.1, T=1.0, backend="hpc", preset=preset)
        assert cfg.compress_decomp == "exact"        # not applied on hpc
        assert cfg.compress_decomp_q == 2
    assert SolverConfig(eps=0.1, T=1.0, backend="hpc", preset=None).preset is None
    with pytest.raises(ValueError):                    # unknown name still rejected
        SolverConfig(eps=0.1, T=1.0, backend="hpc", preset="turbo")


# -- legal NumPy scalars pass and normalize to Python scalars --------------

def test_numpy_scalars_pass_and_normalize():
    cfg = SolverConfig(
        eps=np.float64(0.1),
        T=np.float64(1.0),
        cutoff=np.float64(1e-8),
        max_bond=np.int64(64),
        expansion_order=np.int64(2),
        compress_decomp_q=np.int64(0),
        sub_baths=np.int64(2),
    )
    assert type(cfg.eps) is float
    assert type(cfg.T) is float
    assert type(cfg.cutoff) is float
    assert type(cfg.max_bond) is int
    assert type(cfg.expansion_order) is int
    assert type(cfg.compress_decomp_q) is int
    assert type(cfg.sub_baths) is int
    assert cfg.n_steps == 10


# -- immutability + replace() ----------------------------------------------

def test_frozen_and_replace():
    cfg = SolverConfig(eps=0.1, T=1.0)
    with pytest.raises(FrozenInstanceError):
        cfg.T = 2.0
    with pytest.raises(FrozenInstanceError):
        cfg.backend = "auto"
    fine = replace(cfg, eps=0.05)          # re-runs validation, recomputes n_steps
    assert fine.n_steps == 20
    assert cfg.n_steps == 10               # original unchanged


# -- legal boundary values pass --------------------------------------------

def test_legal_values_pass():
    assert SolverConfig(eps=0.1, T=1.0).n_steps == 10
    assert SolverConfig(eps=0.1, T=1.0, cutoff=0.0).cutoff == 0.0     # exact keeps every sv
    assert SolverConfig(eps=0.1, T=1.0, max_bond=None).max_bond is None
    assert SolverConfig(eps=0.1, T=1.0, expansion_order=1).expansion_order == 1
    assert SolverConfig(eps=0.1, T=1.0, record_rho=True).record_rho is True
