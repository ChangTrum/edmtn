"""Basic observable extraction on the Dicke pipeline.

The contract under test is ``docs/design/dicke-observable-extraction.md``: the cavity
moments are post-processing of the reduced state, while the collective-spin moments come
from a **modified bath-side closing** folded as a first-order jet alongside the value
channel.

The reference is built from scratch here -- a dense Liouvillian on the full
cavity-and-spins space, propagated with the same discretisation, reading only the model's
*physical* inputs -- and imports nothing from the pipeline it checks.  It returns the
**full** ``rho_CB(T)``, so the ``d x d`` tangent matrices are compared against independent
partial traces rather than only their three scalars: a wrong closing that happened to give
the right trace would still be caught.

As in ``test_dicke_evolution.py``, agreement says the contraction reproduces *that
discretisation* exactly; it says nothing about how close the discretisation is to the true
dynamics.  The last section is the other half of the story -- each mutation there is a way
of getting the closing wrong, run through the same machinery to show it produces a
different answer, so the checks above are not passing vacuously.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.linalg import expm

from edmtn.driver import solve
from edmtn.driver.auto_config import SolverConfig, build_pipeline
from edmtn.driver.solver import MOMENT_NAMES, EDMSolver, _extract_moments
from edmtn.evolution.mps_utils import EDMMPS
from edmtn.evolution.quimb_edm import QuimbEDM
from edmtn.models import DickeModel, GaudinModel
from edmtn.observables.extractor import finite_complex_expectation, real_scalar_expectation

_I2 = np.eye(2, dtype=np.complex128)
_SX = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
_SY = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
_SZ = np.diag([1.0, -1.0]).astype(np.complex128)
_SP = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128)
_SM = _SP.conj().T
_C1, _C2 = (1.0 - 1.0j) / 2.0, (1.0 + 1.0j) / 2.0

#: the contract the pipeline must meet (the measured agreement is ~1e-14)
ANCHOR_TOL = 1e-10

#: nothing degenerate: distinct couplings with a sign, distinct frequencies, a complex
#: coherent cavity and off-axis Bloch vectors -- so parity is broken and every transverse
#: component is non-zero.  Both are needed by the phase / conjugation mutations, which
#: carry no information on a symmetric configuration.
ASYMMETRIC = dict(
    K=2, n_fock=3,
    coupling=[0.35, -0.22],
    omega=[0.8, 1.3],
    omega_c=1.1,
    cavity_state="coherent", cavity_params={"alpha": 0.3 + 0.2j},
    bath_state=[[0.2, -0.3, 0.5], [0.0, 0.4, -0.4]],
)
RATES = dict(kappa=0.23, pump=[0.05, 0.02], emission=[0.11, 0.07], dephasing=[0.04, 0.09])


def _kron_all(ops):
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out


def _spin_op(d, K, k, op):
    return _kron_all([np.eye(d, dtype=np.complex128)]
                     + [op if j == k else _I2 for j in range(K)])


def reference_full_state(model, T, eps, order):
    """Dense full-Hilbert propagation of the same discretisation: returns ``rho^I_CB(T)``."""
    d, K = model.n_fock, model.K
    bp = model.bath_params()
    dim = d * 2 ** K
    Id = np.eye(dim, dtype=np.complex128)

    a1 = np.diag(np.sqrt(np.arange(1, d, dtype=np.float64)), 1).astype(np.complex128)
    a = _kron_all([a1] + [_I2] * K)

    def dissipator(L):
        M = L.conj().T @ L
        return np.kron(L, L.conj()) - 0.5 * (np.kron(M, Id) + np.kron(Id, M.T))

    LD = model.kappa * dissipator(a)
    for k in range(K):
        LD += (bp.pump[k] * dissipator(_spin_op(d, K, k, _SP))
               + bp.emission[k] * dissipator(_spin_op(d, K, k, _SM))
               + 0.5 * bp.dephasing[k] * dissipator(_spin_op(d, K, k, _SZ)))

    def commutator_superop(t):
        S = a * np.exp(-1j * model.omega_c * t) + a.conj().T * np.exp(1j * model.omega_c * t)
        B = sum(bp.couplings[k] * (np.cos(bp.omegas[k] * t) * _spin_op(d, K, k, _SX)
                                   - np.sin(bp.omegas[k] * t) * _spin_op(d, K, k, _SY))
                for k in range(K))
        H = S @ B
        return -1j * (np.kron(H, Id) - np.kron(Id, H.T))

    rho0 = _kron_all([model.initial_system_state()]
                     + [0.5 * (_I2 + r[0] * _SX + r[1] * _SY + r[2] * _SZ)
                        for r in model.bath_bloch_vectors()])
    v = rho0.reshape(-1)
    E_h = expm(0.5 * eps * LD)
    Idl = np.eye(dim * dim, dtype=np.complex128)
    for n in range(1, int(round(T / eps)) + 1):
        A = commutator_superop((n - 0.5) * eps)
        v = E_h @ v
        if order == 2:
            v = (Idl + _C2 * eps * A) @ ((Idl + _C1 * eps * A) @ v)
        else:
            v = (Idl + eps * A) @ v
        v = E_h @ v
    return v.reshape(dim, dim)


def reference_tangents(model, T, eps, order):
    """``{channel: (d, d)}`` partial traces ``Tr_B[(1 (x) O) rho_CB]`` of the reference.

    ``O`` is the Schroedinger-picture collective operator written in the interaction
    picture -- ``(1/2) sum_k sigma_{k,z}`` needs no phase, ``sum_k e^{i omega_k T}
    sigma_k^+`` does.
    """
    d, K = model.n_fock, model.K
    full = reference_full_state(model, T, eps, order).reshape(d, 2 ** K, d, 2 ** K)
    spin = lambda k, op: _kron_all([op if j == k else _I2 for j in range(K)])  # noqa: E731
    jz = sum(0.5 * spin(k, _SZ) for k in range(K))
    jp = sum(np.exp(1j * model.omegas[k] * T) * spin(k, _SP) for k in range(K))
    # Tr_B[(1 (x) O) rho]_{ij} = sum_{m,n} O[m, n] rho[i, n, j, m]
    return {"Jz": np.einsum("mn,injm->ij", jz, full),
            "Jplus": np.einsum("mn,injm->ij", jp, full)}


def _tangents(model, T, eps, order, channels=("Jplus", "Jz"), **kw):
    """Run the evolution directly with the model's own closings, uncompressed by default."""
    cfg = SolverConfig(eps=eps, T=T, expansion_order=order)
    kernel, evolution = build_pipeline(model, cfg)
    closures = model.collective_spin_closures(cfg.n_steps * cfg.eps)
    ev = evolution.run(model, kernel, eps, cfg.n_steps,
                       tangent_closings={ch: closures[ch] for ch in channels},
                       **{"compress": False, **kw})
    return ev


# -- 1. the anchor: full d x d tangent matrices vs independent partial traces ------------

@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("rates", [False, True], ids=["closed", "dissipative"])
def test_uncompressed_tangent_matrices_match_the_full_hilbert_reference(order, rates):
    model = DickeModel(**ASYMMETRIC, **(RATES if rates else {}))
    ev = _tangents(model, T=0.3, eps=0.1, order=order)
    ref = reference_tangents(model, T=0.3, eps=0.1, order=order)
    for ch in ("Jplus", "Jz"):
        got = np.asarray(ev.tangent_density_matrices[ch])
        assert got.shape == ref[ch].shape
        assert np.max(np.abs(got - ref[ch])) < ANCHOR_TOL


def test_the_reference_is_sensitive_to_the_couplings():
    """A perturbed model must NOT match, so the anchor above is not degenerate."""
    model = DickeModel(**ASYMMETRIC)
    ev = _tangents(model, T=0.3, eps=0.1, order=2)
    perturbed = dict(ASYMMETRIC, coupling=[0.35, -0.23])
    ref = reference_tangents(DickeModel(**perturbed), T=0.3, eps=0.1, order=2)
    assert np.max(np.abs(np.asarray(ev.tangent_density_matrices["Jz"]) - ref["Jz"])) > 1e-8


def test_tangent_matrices_are_hermitian_and_carry_more_than_their_trace():
    """``tilde_rho^(alpha)`` is Hermitian, and reading it against a cavity operator gives
    the joint correlator -- which is why the whole matrix is kept, not only its trace."""
    model = DickeModel(**ASYMMETRIC)
    ev = _tangents(model, T=0.3, eps=0.1, order=2)
    full = reference_full_state(model, T=0.3, eps=0.1, order=2)
    d, K = model.n_fock, model.K
    tilde = np.asarray(ev.tangent_density_matrices["Jz"])
    assert np.max(np.abs(tilde - tilde.conj().T)) < ANCHOR_TOL
    n_op = np.diag(np.arange(d).astype(np.complex128))
    jz = sum(0.5 * _spin_op(d, K, k, _SZ) for k in range(K))
    joint = np.trace(_kron_all([n_op] + [_I2] * K) @ jz @ full)
    assert abs(np.trace(n_op @ tilde) - joint) < ANCHOR_TOL


def test_sub_baths_sums_only_the_folded_spins():
    """A partial fold is the model holding only those spins -- both in the dynamics and in
    the sum, which is why the returned moments describe the first ``L`` spins alone."""
    ev = _tangents(DickeModel(**ASYMMETRIC), T=0.3, eps=0.1, order=2, sub_baths=1)
    first = DickeModel(K=1, n_fock=ASYMMETRIC["n_fock"],
                       coupling=ASYMMETRIC["coupling"][:1], omega=ASYMMETRIC["omega"][:1],
                       omega_c=ASYMMETRIC["omega_c"], cavity_state="coherent",
                       cavity_params=ASYMMETRIC["cavity_params"],
                       bath_state=ASYMMETRIC["bath_state"][:1])
    ref = reference_tangents(first, T=0.3, eps=0.1, order=2)
    for ch in ("Jplus", "Jz"):
        got = np.asarray(ev.tangent_density_matrices[ch])
        assert np.max(np.abs(got - ref[ch])) < ANCHOR_TOL


# -- 2. single-site coverage: both boundaries land on the same tensor --------------------

@pytest.mark.parametrize("channel", ["Jplus", "Jz"])
def test_single_step_grid_agrees(channel):
    """``order=1, n_steps=1``: the newest and oldest sites are the same tensor."""
    model = DickeModel(**ASYMMETRIC, **RATES)
    ev = _tangents(model, T=0.1, eps=0.1, order=1)
    ref = reference_tangents(model, T=0.1, eps=0.1, order=1)
    got = np.asarray(ev.tangent_density_matrices[channel])
    assert np.max(np.abs(got - ref[channel])) < ANCHOR_TOL


def test_single_site_kernel_contracts_both_boundaries():
    """Directly on the kernel: the one site must become ``v_a A[.., a, b] r_b``.

    The default closing is the index-0 slice, i.e. ``e_0`` -- "not measuring".
    """
    model = DickeModel(**ASYMMETRIC, **RATES)
    cfg = SolverConfig(eps=0.1, T=0.1, expansion_order=1)
    kernel, _ = build_pipeline(model, cfg)
    provider = kernel.for_sub_bath(1)
    raw = provider._op[0]                       # (up, down, a_left, a_right), the only site
    r = kernel.corr.boundary_vector(1)
    for v in (None, np.array([0.0, 0.0, 0.0, 0.5], dtype=np.complex128),
              0.5 * np.array([0.0, 1.0, 1.0j, 0.0], dtype=np.complex128)):
        site = provider.get_kernel_mpo(1, v).site_tensors[0]
        closing = np.eye(4)[0] if v is None else v
        expected = np.einsum("a,udab,b->ud", closing, raw, r)
        assert site.shape == (raw.shape[0], raw.shape[1], 1, 1)
        assert np.max(np.abs(site[:, :, 0, 0] - expected)) < 1e-14


def test_kernel_rejects_a_malformed_closing():
    model = DickeModel(**ASYMMETRIC)
    kernel, _ = build_pipeline(model, SolverConfig(eps=0.1, T=0.2, expansion_order=1))
    provider = kernel.for_sub_bath(0)
    with pytest.raises(ValueError, match="length-4"):
        provider.get_kernel_mpo(2, np.ones(3, dtype=np.complex128))
    with pytest.raises(ValueError, match="finite"):
        provider.get_kernel_mpo(2, np.array([np.nan, 0, 0, 0], dtype=np.complex128))


# -- 3. add_exact ------------------------------------------------------------------------

def _random_edm(n, d, d_phys, chi, rng, rho0=None):
    d2 = d * d
    tensors, left = [], d2
    for p in range(n):
        right = d2 if p == n - 1 else chi
        tensors.append((rng.standard_normal((d_phys, left, right))
                        + 1j * rng.standard_normal((d_phys, left, right))).astype(complex))
        left = right
    if rho0 is None:
        rho0 = np.ones(d2, dtype=np.complex128)
    return QuimbEDM.from_edmmps(EDMMPS(tensors=tensors, d=d, d_phys=d_phys, rho0_vec=rho0))


@pytest.mark.parametrize("n,chi_a,chi_b", [(1, 0, 0), (2, 3, 5), (4, 3, 6), (4, 5, 5)])
def test_add_exact_is_additive_on_every_open_arm(n, chi_a, chi_b):
    """The whole EDM adds, not merely its reduced state: compare the open-arm tensor."""
    rng = np.random.default_rng(7)
    a = _random_edm(n, 2, 3, max(chi_a, 1), rng)
    b = _random_edm(n, 2, 3, max(chi_b, 1), rng)
    total = a.add_exact(b)
    got = total.to_edmmps().open_arm_tensor()
    want = a.to_edmmps().open_arm_tensor() + b.to_edmmps().open_arm_tensor()
    assert got.shape == want.shape
    assert np.max(np.abs(got - want)) < 1e-10


def test_add_exact_leaves_both_inputs_untouched_and_shares_no_buffer():
    """Scribble on EVERY buffer the result carries -- the site tensors *and* ``rho0_vec``.

    Only touching the tensors would miss a shared boundary vector, which is a buffer the
    result owns just as much as the sites.
    """
    rng = np.random.default_rng(11)
    a, b = _random_edm(3, 2, 3, 4, rng), _random_edm(3, 2, 3, 6, rng)
    before_a = a.to_edmmps().open_arm_tensor().copy()
    before_b = b.to_edmmps().open_arm_tensor().copy()
    rho_a, rho_b = a.rho0_vec.copy(), b.rho0_vec.copy()
    total = a.add_exact(b)
    for site in total.to_edmmps().tensors:
        site += 1.0                                    # scribble on the result
    total.rho0_vec += 1.0
    assert np.max(np.abs(a.to_edmmps().open_arm_tensor() - before_a)) == 0.0
    assert np.max(np.abs(b.to_edmmps().open_arm_tensor() - before_b)) == 0.0
    assert np.array_equal(a.rho0_vec, rho_a)
    assert np.array_equal(b.rho0_vec, rho_b)
    assert not np.shares_memory(total.rho0_vec, a.rho0_vec)


def test_add_exact_direct_sums_internal_bonds_only():
    """OUT and RHO0 are shared external legs; direct-summing one would not be a sum."""
    rng = np.random.default_rng(3)
    a, b = _random_edm(3, 2, 3, 4, rng), _random_edm(3, 2, 3, 6, rng)
    total = a.add_exact(b).to_edmmps()
    assert total.tensors[0].shape[1] == 4                      # OUT stays d**2
    assert total.tensors[-1].shape[2] == 4                     # RHO0 stays d**2
    assert total.bond_dims == [x + y for x, y in
                               zip(a.to_edmmps().bond_dims, b.to_edmmps().bond_dims)]
    # Direct-summing OUT as well would double that leg; the open-arm additivity test above
    # is what would then fail, because the result would not even have the EDM's shape.
    assert total.open_arm_tensor().shape == (3, 3, 3, 2, 2)


@pytest.mark.parametrize("kw,match", [
    (dict(n=4), "structurally identical"),
    (dict(d_phys=4), "structurally identical"),
    (dict(rho0=np.arange(4).astype(complex)), "different rho0_vec"),
])
def test_add_exact_rejects_structural_mismatch(kw, match):
    rng = np.random.default_rng(5)
    a = _random_edm(3, 2, 3, 4, rng)
    b = _random_edm(kw.get("n", 3), 2, kw.get("d_phys", 3), 4, rng, rho0=kw.get("rho0"))
    with pytest.raises(ValueError, match=match):
        a.add_exact(b)


# -- 4. the value channel must be untouched ---------------------------------------------

@pytest.mark.parametrize("moments", [("Jz",), ("Jabs",), ("n",)])
def test_requesting_moments_does_not_perturb_the_value_channel(moments):
    """Array by array: the main chain, its bond history and its truncation record."""
    model = DickeModel(**ASYMMETRIC, **RATES)
    kw = dict(T=0.3, eps=0.1, cutoff=1e-6, expansion_order=2)
    plain = solve(model, **kw)
    with_m = solve(model, **kw, moments=moments)
    assert with_m.bond_dims == plain.bond_dims
    assert with_m.truncation_errors == plain.truncation_errors
    assert with_m.compression_method_used == plain.compression_method_used
    assert with_m.final_time_bond_dims == plain.final_time_bond_dims
    assert np.max(np.abs(np.asarray(with_m.final_density_matrix)
                         - np.asarray(plain.final_density_matrix))) == 0.0
    for x, y in zip(with_m.mps.tensors, plain.mps.tensors):
        assert np.array_equal(x, y)


# -- 5. the vocabulary contract ----------------------------------------------------------

def _solver(**cfg):
    model = DickeModel(**ASYMMETRIC)
    return EDMSolver.from_model(model, T=0.2, eps=0.1, **cfg)


def test_moments_default_to_nothing():
    res = solve(DickeModel(**ASYMMETRIC), T=0.2, eps=0.1)
    assert res.moments is None
    assert res.moment_truncation_errors is None


@pytest.mark.parametrize("request_,expected", [
    (("n",), {"n", "trace"}),
    (("n", "n_factorial2"), {"n", "n_factorial2", "trace"}),
    (("Jx",), {"Jx", "Jy", "trace"}),                 # one chain gives both components
    (("Jy",), {"Jx", "Jy", "trace"}),
    (("Jz",), {"Jz", "trace"}),
    (("Jabs",), {"Jabs", "Jx", "Jy", "Jz", "trace"}),
    (("Jz", "Jz", "n"), {"Jz", "n", "trace"}),        # duplicates collapse
])
def test_only_what_was_asked_for_plus_by_products(request_, expected):
    res = solve(DickeModel(**ASYMMETRIC), T=0.2, eps=0.1, moments=request_)
    assert set(res.moments) == expected


def test_requesting_a_component_never_triggers_the_other_channel():
    res = solve(DickeModel(**ASYMMETRIC), T=0.2, eps=0.1, moments=("Jx",))
    assert "Jz" not in res.moments and "Jabs" not in res.moments
    assert set(res.moment_truncation_errors) == {"Jplus"}
    res = solve(DickeModel(**ASYMMETRIC), T=0.2, eps=0.1, moments=("Jz",))
    assert set(res.moment_truncation_errors) == {"Jz"}


def test_cavity_moments_alone_record_no_channel_truncation():
    res = solve(DickeModel(**ASYMMETRIC), T=0.2, eps=0.1, moments=("n", "n_factorial2"))
    assert res.moment_truncation_errors is None


@pytest.mark.parametrize("bad", ["Jz", b"Jz", ["jz"], ["g2"], [1], [None], 5, [b"Jz"]])
def test_illegal_moment_requests_raise_value_error(bad):
    with pytest.raises(ValueError):
        _solver().solve(moments=bad)


@pytest.mark.parametrize("unordered", [{"Jz"}, frozenset({"Jz"}), (m for m in ["Jz"])],
                         ids=["set", "frozenset", "generator"])
def test_an_unordered_or_lazy_request_is_refused(unordered):
    """The contract is ``Sequence[str]``: "duplicates collapse, first order preserved" has
    no meaning for a set, and a generator would be consumed by whoever looked at it first."""
    with pytest.raises(ValueError, match="ordered sequence"):
        _solver().solve(moments=unordered)


def test_an_unknown_name_is_a_value_error_on_a_model_without_the_capability():
    """Vocabulary first, capability second -- a typo is a ValueError on every model."""
    gaudin = GaudinModel(K=2, g=0.5)
    with pytest.raises(ValueError, match="unknown moment"):
        EDMSolver.from_model(gaudin, T=0.2, eps=0.1).solve(moments=["nope"])
    with pytest.raises(NotImplementedError, match="collective_spin_closures"):
        EDMSolver.from_model(gaudin, T=0.2, eps=0.1).solve(moments=["Jz"])


def test_an_empty_request_is_not_an_error():
    for empty in ([], ()):
        assert solve(DickeModel(**ASYMMETRIC), T=0.2, eps=0.1, moments=empty).moments is None


def test_scalar_types():
    res = solve(DickeModel(**ASYMMETRIC), T=0.2, eps=0.1,
                moments=("n", "n_factorial2", "Jabs"))
    assert type(res.moments["trace"]) is complex
    for name in ("n", "n_factorial2", "Jx", "Jy", "Jz", "Jabs"):
        assert type(res.moments[name]) is float


@pytest.mark.parametrize("bad", [complex(float("nan"), 0.0), complex(float("inf"), 0.0),
                                 complex(0.0, float("nan"))])
def test_a_non_finite_scalar_raises_instead_of_being_returned(bad):
    """A ``nan`` slides through any tolerance comparison -- ``nan > x`` is ``False`` -- so
    the finiteness check has to come first, on the complex-valued quantities too."""
    with pytest.raises(FloatingPointError, match="not finite"):
        real_scalar_expectation("Jz", bad)
    with pytest.raises(FloatingPointError, match="not finite"):
        finite_complex_expectation("Jplus", bad)


def test_a_non_finite_reduced_state_is_refused_by_the_packing():
    """End to end through the packing, so the guard is wired in and not merely defined."""
    rho = np.full((3, 3), np.nan, dtype=np.complex128)
    tangents = {"Jz": np.eye(3, dtype=np.complex128),
                "Jplus": np.eye(3, dtype=np.complex128)}
    with pytest.raises(FloatingPointError):
        _extract_moments(("n",), rho, tangents)
    with pytest.raises(FloatingPointError):
        _extract_moments(("Jz",), np.eye(3, dtype=np.complex128),
                         {"Jz": np.full((3, 3), np.inf, dtype=np.complex128)})


def test_jabs_is_checked_after_it_is_derived_not_only_before():
    """Three finite components can still overflow when combined.

    ``sqrt(Jx**2 + Jy**2 + Jz**2)`` on values near the float64 ceiling squares to ``inf``
    while every input passes its own finiteness check -- so the derived quantity needs its
    own check, and ``hypot`` is what keeps the honest cases from overflowing at all.
    """
    rho = np.eye(3, dtype=np.complex128) / 3.0
    trace_of = lambda x: np.diag([x, 0.0, 0.0]).astype(np.complex128)   # noqa: E731

    # (a) hypot is what keeps an honest large case from overflowing at all: the naive
    #     formula cannot even be evaluated here -- Python raises rather than returning inf
    big = 1e200
    with pytest.raises(OverflowError):
        math.sqrt(big ** 2 + big ** 2)
    out = _extract_moments(("Jabs",), rho, {"Jplus": trace_of(big), "Jz": trace_of(big)})
    assert math.isfinite(out["Jabs"])
    assert out["Jabs"] == pytest.approx(math.hypot(big, 0.0, big))

    # (b) ... and when the norm genuinely exceeds float64, hypot returns `inf` rather than
    #     raising, so the SECOND finiteness check -- on the derived value -- is the only
    #     thing standing between the caller and an `inf`.  Delete that check and this fails.
    over = 1.3e308                                  # each component finite; the norm is not
    assert math.isfinite(over) and math.isinf(math.hypot(over, 0.0, over))
    with pytest.raises(FloatingPointError, match="Jabs"):
        _extract_moments(("Jabs",), rho, {"Jplus": trace_of(over), "Jz": trace_of(over)})

    # (c) a component that is itself non-finite is refused before any of this
    with pytest.raises(FloatingPointError):
        _extract_moments(("Jabs",), rho,
                         {"Jplus": trace_of(np.inf), "Jz": trace_of(big)})


def test_a_provider_returning_the_wrong_mapping_does_not_leak_a_key_error():
    """``collective_spin_closures`` is duck-typed; a wrong return must be a clear error."""
    model = DickeModel(**ASYMMETRIC)
    model.collective_spin_closures = lambda t: {"Jz": np.zeros((2, 4), dtype=complex)}
    with pytest.raises(ValueError, match="missing the channel"):
        solve(model, T=0.2, eps=0.1, moments=("Jx",))
    model.collective_spin_closures = lambda t: [np.zeros((2, 4), dtype=complex)]
    with pytest.raises(ValueError, match="must return a mapping"):
        solve(model, T=0.2, eps=0.1, moments=("Jz",))


def test_the_vocabulary_is_exactly_the_documented_one():
    assert MOMENT_NAMES == ("n", "n_factorial2", "Jx", "Jy", "Jz", "Jabs")
    for name in MOMENT_NAMES:
        assert name in solve(DickeModel(**ASYMMETRIC), T=0.2, eps=0.1,
                             moments=[name]).moments


def test_spin_moments_and_time_reads_are_rejected_together():
    with pytest.raises(ValueError, match="record_time_reads"):
        _solver(record_time_reads=True, compress_method="dm_tracking").solve(moments=["Jz"])
    # ... but the cavity moments, which fold nothing extra, coexist
    res = _solver(record_time_reads=True, compress_method="dm_tracking").solve(moments=["n"])
    assert set(res.moments) == {"n", "trace"}
    assert res.density_matrices is not None


def test_direct_run_allows_the_same_combination_without_compression():
    model = DickeModel(**ASYMMETRIC)
    ev = _tangents(model, T=0.2, eps=0.1, order=2, record_time_reads=True)
    assert ev.time_density_matrices is not None
    assert ev.tangent_density_matrices is not None


def test_a_kernel_without_closings_is_refused():
    """Capability, not bath type: the Gaudin kernel takes no lateral closing."""
    gaudin = GaudinModel(K=2, g=0.5)
    cfg = SolverConfig(eps=0.1, T=0.2)
    kernel, evolution = build_pipeline(gaudin, cfg)
    with pytest.raises(ValueError, match="supports_closings"):
        evolution.run(gaudin, kernel, 0.1, 2, compress=False,
                      tangent_closings={"Jz": np.zeros((2, 4), dtype=complex)})


@pytest.mark.parametrize("bad", [{}, {"Jz": np.zeros((3, 4), complex)},
                                 {"Jz": np.zeros(4, complex)}, [("Jz", None)]])
def test_run_validates_the_closing_mapping(bad):
    model = DickeModel(**ASYMMETRIC)
    kernel, evolution = build_pipeline(model, SolverConfig(eps=0.1, T=0.2))
    with pytest.raises(ValueError):
        evolution.run(model, kernel, 0.1, 2, compress=False, tangent_closings=bad)


# -- 6. bounds, parity and picture invariance -------------------------------------------

@pytest.mark.parametrize("sub_baths", [1, 2])
def test_the_modulus_is_bounded_by_half_the_folded_spin_count(sub_baths):
    res = solve(DickeModel(**ASYMMETRIC), T=0.3, eps=0.1, moments=("Jabs",),
                sub_baths=sub_baths, cutoff=0.0)
    assert res.sub_baths_used == sub_baths
    assert res.moments["Jabs"] <= sub_baths / 2 + 1e-12


def test_a_parity_symmetric_configuration_has_no_transverse_moment():
    """Vacuum cavity + ground spins: ``<J_x> = <J_y> = 0`` for any ``g_k``, ``omega_k``."""
    model = DickeModel(K=2, n_fock=4, coupling=[0.35, -0.22], omega=[0.8, 1.3],
                       omega_c=1.1, cavity_state="vacuum", bath_state="ground")
    res = solve(model, T=0.4, eps=0.1, moments=("Jabs",), cutoff=0.0)
    assert abs(res.moments["Jx"]) < 1e-12
    assert abs(res.moments["Jy"]) < 1e-12
    assert abs(res.moments["Jz"]) > 0.5            # not everything is zero


@pytest.mark.parametrize("breaker", [
    dict(cavity_state="coherent", cavity_params={"alpha": 0.4}),
    dict(bath_state=[[0.6, 0.0, -0.8], [0.0, 0.0, -1.0]]),
])
def test_breaking_parity_in_the_initial_state_makes_the_transverse_moment_non_zero(breaker):
    """Otherwise the zero above would be a property of the test, not of the physics."""
    kw = dict(K=2, n_fock=4, coupling=[0.35, -0.22], omega=[0.8, 1.3], omega_c=1.1,
              cavity_state="vacuum", bath_state="ground")
    kw.update(breaker)
    res = solve(DickeModel(**kw), T=0.4, eps=0.1, moments=("Jabs",), cutoff=0.0)
    assert max(abs(res.moments["Jx"]), abs(res.moments["Jy"])) > 1e-6


@pytest.mark.parametrize("omega,invariant", [(1.1, True), ([0.8, 1.3], False)])
def test_the_modulus_is_picture_invariant_only_for_a_homogeneous_splitting(omega, invariant):
    """A common rotation about ``z`` preserves ``|<J>|``; per-spin rotations do not."""
    model = DickeModel(**dict(ASYMMETRIC, omega=omega))
    T, eps = 0.3, 0.1
    with_phase = _tangents(model, T=T, eps=eps, order=2)
    closures = model.collective_spin_closures(T)
    unphased = 0.5 * np.array([0.0, 1.0, 1.0j, 0.0], dtype=np.complex128)
    interaction = {"Jplus": np.tile(unphased, (model.K, 1)), "Jz": closures["Jz"]}
    cfg = SolverConfig(eps=eps, T=T, expansion_order=2)
    kernel, evolution = build_pipeline(model, cfg)
    no_phase = evolution.run(model, kernel, eps, cfg.n_steps, compress=False,
                             tangent_closings=interaction)

    def modulus(ev):
        jp = complex(np.trace(np.asarray(ev.tangent_density_matrices["Jplus"])))
        jz = complex(np.trace(np.asarray(ev.tangent_density_matrices["Jz"]))).real
        return float(np.sqrt(jp.real ** 2 + jp.imag ** 2 + jz ** 2))

    same = abs(modulus(with_phase) - modulus(no_phase)) < 1e-10
    assert same is invariant


# -- 7. the compressed paths (the public solver always compresses) -----------------------

def _moments_with(model, T=0.3, **cfg):
    return solve(model, T=T, eps=0.1, moments=("Jabs", "n"), **cfg).moments


@pytest.mark.parametrize("method", ["zipup", "dm", "direct"])
def test_a_lossless_compression_reproduces_the_uncompressed_result(method):
    """``cutoff=0`` with no bond cap discards nothing, on every compression path.

    ``dm`` -- and **only** ``dm`` -- runs on a shorter grid.  This is the one configuration
    in which quimb's density-matrix method does not truncate at all, so it keeps every
    eigenvector and the bond expands toward the full environment rank
    (``9 * 3^(n_sites-1)``: 2187 at three steps, 243 at two) instead of shrinking; the cost
    follows the cube of that bond.  Two steps still exercise two bath folds, the tangent
    addition and this compression path, so the claim under test is unchanged -- and
    ``zipup``/``direct``, for which the extra step is free, keep the longer grid.
    """
    T = 0.2 if method == "dm" else 0.3
    model = DickeModel(**ASYMMETRIC, **RATES)
    ev = _tangents(model, T=T, eps=0.1, order=2)
    ref_jz = complex(np.trace(np.asarray(ev.tangent_density_matrices["Jz"]))).real
    got = _moments_with(model, T=T, cutoff=0.0, max_bond=None, compress_method=method)
    assert abs(got["Jz"] - ref_jz) < 1e-10


def test_tightening_the_cutoff_moves_the_moments_toward_the_uncompressed_reference():
    model = DickeModel(**ASYMMETRIC, **RATES)
    ev = _tangents(model, T=0.3, eps=0.1, order=2)
    ref = {ch: complex(np.trace(np.asarray(m)))
           for ch, m in ev.tangent_density_matrices.items()}
    errs = []
    for cutoff in (1e-2, 1e-6, 1e-12):
        got = _moments_with(model, cutoff=cutoff)
        errs.append(max(abs(got["Jz"] - ref["Jz"].real),
                        abs(got["Jx"] - ref["Jplus"].real),
                        abs(got["Jy"] - ref["Jplus"].imag)))
    assert errs[0] >= errs[1] >= errs[2]
    assert errs[2] < 1e-9


def test_a_rank_cap_also_converges_as_it_is_raised():
    model = DickeModel(**ASYMMETRIC, **RATES)
    ev = _tangents(model, T=0.3, eps=0.1, order=2)
    ref = complex(np.trace(np.asarray(ev.tangent_density_matrices["Jz"]))).real
    errs = [abs(_moments_with(model, cutoff=0.0, max_bond=b)["Jz"] - ref)
            for b in (16, 64)]
    assert errs[0] > errs[1]
    assert errs[1] < 1e-12                       # 64 is the full bond here: no truncation


def test_a_brutal_rank_cap_raises_rather_than_returning_a_meaningless_number():
    """The imaginary-part guard is live, and it is what stands between a destroyed tangent
    and a plausible-looking float.

    Measured on this configuration: a cap of 2/4/8 leaves ``<J_z>`` wrong by a factor of
    four (and at ``4``, of the wrong sign) with a 7-10% relative imaginary part, while a
    cap of 16 already agrees to six digits with a relative imaginary part of 1e-6.  The
    compressed tangent is no longer Hermitian, so its trace stops being real -- and that
    is exactly the signal the guard reads.
    """
    model = DickeModel(**ASYMMETRIC, **RATES)
    with pytest.raises(ValueError, match="imaginary part"):
        _moments_with(model, cutoff=0.0, max_bond=2)


def test_rsvd_records_an_unmeasurable_truncation():
    """Every fold compresses the extra chain, so every entry is the honest ``None``."""
    model = DickeModel(**ASYMMETRIC)
    res = solve(model, T=0.3, eps=0.1, moments=("Jz",), compress_decomp="rsvd")
    record = res.moment_truncation_errors["Jz"]
    assert len(record) == len(res.sub_bath_counts)
    assert record == [None] * len(record)


def test_the_first_fold_is_compressed_like_every_other():
    """``dM_0 = 0`` saves the zero chain's fold and addition -- never the caller's
    compression.

    At ``K = 1`` there is only that first fold, so skipping it would make ``cutoff`` and
    ``max_bond`` permanently inert on this channel while the record still claimed ``0.0``.
    A rank cap of 8 truncates this chain (its uncompressed bonds run to 36); the moment
    must move with it, and must converge back as the cap is raised.  A harder cap is not
    used because the imaginary-part guard -- correctly -- refuses the destroyed tangent
    that a cap of 1 or 2 produces.
    """
    model = DickeModel(K=1, n_fock=3, coupling=[0.35], omega=[0.8], omega_c=1.1,
                       cavity_state="coherent", cavity_params={"alpha": 0.3 + 0.2j},
                       bath_state=[[0.2, -0.3, 0.5]])
    kw = dict(T=0.3, eps=0.1, moments=("Jz",), cutoff=0.0)
    exact = solve(model, **kw, max_bond=None)
    capped = solve(model, **kw, max_bond=8)
    loose = solve(model, **kw, max_bond=16)
    assert capped.final_time_bond_dims == [8] * len(capped.final_time_bond_dims)
    # before the fix every one of these was bit-identical to `exact` with a recorded 0.0
    assert abs(capped.moments["Jz"] - exact.moments["Jz"]) > 1e-3
    assert abs(loose.moments["Jz"] - exact.moments["Jz"]) < 1e-6      # and it converges back
    # ... and the record reports the truncation that actually happened, not a stand-in 0.0
    assert capped.moment_truncation_errors["Jz"][0] > 0.0
    assert exact.moment_truncation_errors["Jz"] == [0.0]      # nothing could be discarded


def test_without_compression_the_first_fold_still_reports_a_true_zero():
    model = DickeModel(**ASYMMETRIC)
    ev = _tangents(model, T=0.3, eps=0.1, order=2, channels=("Jz",))
    assert ev.tangent_truncation_errors["Jz"] == [0.0, 0.0]


def test_the_channel_truncation_record_is_keyed_by_channel_and_aligned_with_L():
    model = DickeModel(**dict(ASYMMETRIC, K=4, coupling=[0.35, -0.22, 0.3, -0.15],
                              omega=[0.8, 1.3, 0.55, 1.0],
                              bath_state=[[0.2, -0.3, 0.5], [0.0, 0.4, -0.4],
                                          [0.1, 0.1, 0.9], [0.0, 0.0, -1.0]]))
    res = solve(model, T=0.3, eps=0.1, moments=("Jabs",), cutoff=1e-3)
    assert set(res.moment_truncation_errors) == {"Jplus", "Jz"}
    for record in res.moment_truncation_errors.values():
        assert len(record) == len(res.sub_bath_counts)
        assert all(w is None or w >= 0.0 for w in record)


def test_the_channel_record_covers_the_folds_it_did_not_record():
    """``record_every > 1`` must take the maximum over the skipped folds, not drop them."""
    model = DickeModel(**ASYMMETRIC, **RATES)
    kw = dict(T=0.3, eps=0.1, order=2, compress=True, cutoff=1e-4, channels=("Jz",))
    every = _tangents(model, **kw, record_every=1).tangent_truncation_errors["Jz"]
    coarse = _tangents(model, **kw, record_every=2).tangent_truncation_errors["Jz"]
    assert len(every) == 2 and len(coarse) == 1        # K = 2: L = 1, 2 vs L = 2 only
    assert coarse[0] == max(every)


def test_the_tangent_truncation_is_recorded_separately_from_the_value_channel():
    """A jet approximation's error is its own; the value channel's record is not evidence."""
    model = DickeModel(**ASYMMETRIC, **RATES)
    res = solve(model, T=0.3, eps=0.1, moments=("Jz",), cutoff=1e-3)
    assert len(res.moment_truncation_errors["Jz"]) == len(res.truncation_errors)
    assert res.moment_truncation_errors["Jz"] != res.truncation_errors


@pytest.mark.gpu
def test_moments_end_to_end_on_the_gpu():
    model = DickeModel(**ASYMMETRIC, **RATES)
    cpu = solve(model, T=0.3, eps=0.1, moments=("Jabs", "n"), cutoff=1e-10)
    gpu = solve(model, T=0.3, eps=0.1, moments=("Jabs", "n"), cutoff=1e-10, backend="gpu")
    assert gpu.backend.startswith("gpu")
    for name in ("Jx", "Jy", "Jz", "Jabs", "n"):
        assert abs(gpu.moments[name] - cpu.moments[name]) < 1e-8
    assert type(gpu.moments["n"]) is float


@pytest.mark.gpu
def test_add_exact_keeps_cupy_on_the_device():
    import cupy as cp

    rng = np.random.default_rng(1)
    a, b = _random_edm(3, 2, 3, 4, rng), _random_edm(3, 2, 3, 5, rng)
    to_gpu = lambda edm: QuimbEDM.from_edmmps(EDMMPS(  # noqa: E731
        tensors=[cp.asarray(t) for t in edm.to_edmmps().tensors], d=edm.d,
        d_phys=edm.d_phys, rho0_vec=cp.asarray(edm.rho0_vec)))
    total = to_gpu(a).add_exact(to_gpu(b))
    assert all(isinstance(t, cp.ndarray) for t in total.to_edmmps().tensors)
    want = a.to_edmmps().open_arm_tensor() + b.to_edmmps().open_arm_tensor()
    assert np.max(np.abs(cp.asnumpy(total.to_edmmps().open_arm_tensor()) - want)) < 1e-10


# -- 8. the mutations that must turn this file red ---------------------------------------

def _jet_by_hand(model, T, eps, order, channel, closing=None, source="old"):
    """The jet recursion written out with the public container API.

    ``source='new'`` builds the tangent's source term from the *updated* value chain
    instead of the one this fold started from -- the mutation of the recursion itself.
    """
    cfg = SolverConfig(eps=eps, T=T, expansion_order=order)
    kernel, evolution = build_pipeline(model, cfg)
    n_sites, d = order * cfg.n_steps, model.system_dim
    rho0 = model.initial_system_state().reshape(-1).astype(np.complex128)
    identity = lambda a: a                                                  # noqa: E731
    M = QuimbEDM.from_edmmps(evolution._build_system_mps(
        model, eps, cfg.n_steps, order, d, kernel.d_phys, rho0, identity))
    v = (model.collective_spin_closures(cfg.n_steps * eps)[channel]
         if closing is None else closing)
    dM = None
    for k in range(model.K):
        provider = kernel.for_sub_bath(k)
        mpo_0 = provider.get_kernel_mpo(n_sites).site_tensors
        mpo_v = provider.get_kernel_mpo(n_sites, v[k]).site_tensors
        M_old = M
        M = M.fold_raw(mpo_0)
        src = (M if source == "new" else M_old).fold_raw(mpo_v)
        dM = src if dM is None else dM.fold_raw(mpo_0).add_exact(src)
    return np.asarray(dM.reduced_density_matrix())


def test_the_hand_written_jet_reproduces_the_shipped_loop():
    """So the mutations below differ from the shipped result for one reason only."""
    model = DickeModel(**ASYMMETRIC, **RATES)
    ev = _tangents(model, T=0.3, eps=0.1, order=2)
    for ch in ("Jplus", "Jz"):
        got = _jet_by_hand(model, 0.3, 0.1, 2, ch)
        assert np.max(np.abs(got - np.asarray(ev.tangent_density_matrices[ch]))) < 1e-12


def test_mutation_taking_the_source_term_from_the_updated_chain():
    model = DickeModel(**ASYMMETRIC, **RATES)
    ref = reference_tangents(model, T=0.3, eps=0.1, order=2)["Jz"]
    correct = _jet_by_hand(model, 0.3, 0.1, 2, "Jz", source="old")
    mutated = _jet_by_hand(model, 0.3, 0.1, 2, "Jz", source="new")
    assert np.max(np.abs(correct - ref)) < ANCHOR_TOL
    assert np.max(np.abs(mutated - ref)) > 1e-6


@pytest.mark.parametrize("mutate,name", [
    (lambda v: v.conj(), "conjugated"),
    (lambda v: 2.0 * v, "half factor dropped"),
])
def test_mutation_of_the_transverse_closing(mutate, name):
    model = DickeModel(**ASYMMETRIC, **RATES)
    ref = reference_tangents(model, T=0.3, eps=0.1, order=2)["Jplus"]
    closing = model.collective_spin_closures(0.3)["Jplus"]
    correct = _jet_by_hand(model, 0.3, 0.1, 2, "Jplus")
    mutated = _jet_by_hand(model, 0.3, 0.1, 2, "Jplus", closing=mutate(closing))
    assert np.max(np.abs(correct - ref)) < ANCHOR_TOL
    assert np.max(np.abs(mutated - ref)) > 1e-6, name


def test_mutation_of_the_picture_phase_sign():
    """``e^{+i omega_k T}`` written as ``e^{-i omega_k T}``: needs inhomogeneous omega."""
    model = DickeModel(**ASYMMETRIC, **RATES)
    ref = reference_tangents(model, T=0.3, eps=0.1, order=2)["Jplus"]
    wrong = np.zeros((model.K, 4), dtype=np.complex128)
    for k in range(model.K):
        wrong[k] = 0.5 * np.exp(-1j * model.omegas[k] * 0.3) * np.array([0, 1, 1j, 0])
    mutated = _jet_by_hand(model, 0.3, 0.1, 2, "Jplus", closing=wrong)
    assert np.max(np.abs(mutated - ref)) > 1e-6


def test_mutation_of_the_jz_half_factor():
    model = DickeModel(**ASYMMETRIC, **RATES)
    ref = reference_tangents(model, T=0.3, eps=0.1, order=2)["Jz"]
    doubled = model.collective_spin_closures(0.3)["Jz"] * 2.0
    mutated = _jet_by_hand(model, 0.3, 0.1, 2, "Jz", closing=doubled)
    assert np.max(np.abs(mutated - ref)) > 1e-6
