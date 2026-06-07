"""Unit tests for the Layer-0 mixed-precision policy (PrecisionPolicy).

Pure CPU — no GPU needed.
"""

import numpy as np
import pytest

import edmtn.backend as bk
from edmtn.backend.precision import PrecisionPolicy


# --------------------------------------------------------------------------
# construction / validation
# --------------------------------------------------------------------------

def test_default_is_full_f64():
    p = PrecisionPolicy()
    assert (p.build, p.contract, p.decompose) == ("f64", "f64", "f64")


def test_presets():
    assert PrecisionPolicy.full_f64() == PrecisionPolicy("f64", "f64", "f64")
    mixed = PrecisionPolicy.mixed()
    assert mixed.contract == "f32"
    assert mixed.build == "f64" and mixed.decompose == "f64"


def test_decompose_must_be_f64():
    with pytest.raises(ValueError, match="decompose"):
        PrecisionPolicy(decompose="f32")


def test_invalid_label_rejected():
    with pytest.raises(ValueError):
        PrecisionPolicy(contract="f16")


def test_frozen():
    p = PrecisionPolicy()
    with pytest.raises(Exception):
        p.contract = "f32"


# --------------------------------------------------------------------------
# dtype lookup
# --------------------------------------------------------------------------

def test_complex_dtype_mapping():
    p = PrecisionPolicy.mixed()
    assert p.complex_dtype("build") == np.dtype(np.complex128)
    assert p.complex_dtype("contract") == np.dtype(np.complex64)
    assert p.complex_dtype("decompose") == np.dtype(np.complex128)


def test_real_dtype_mapping():
    p = PrecisionPolicy.mixed()
    assert p.real_dtype("contract") == np.dtype(np.float32)
    assert p.real_dtype("build") == np.dtype(np.float64)


def test_unknown_stage_rejected():
    with pytest.raises(ValueError):
        PrecisionPolicy().complex_dtype("warmup")


# --------------------------------------------------------------------------
# caster
# --------------------------------------------------------------------------

def test_caster_casts_to_stage_dtype():
    p = PrecisionPolicy.mixed()
    cast_contract = p.caster("contract", np)
    out = cast_contract(np.ones(3, dtype=np.complex128))
    assert out.dtype == np.complex64

    cast_build = p.caster("build", np)
    out2 = cast_build(np.ones(3, dtype=np.complex64))
    assert out2.dtype == np.complex128


def test_factory_cast_and_caster_use_policy():
    f = bk.ArrayFactory("numpy", precision=PrecisionPolicy.mixed())
    a = np.ones(4, dtype=np.complex128)
    assert f.cast(a, "contract").dtype == np.complex64
    assert f.caster("decompose")(a).dtype == np.complex128
