"""Abstract open-quantum-system model interface (Layer 1).

A model defines the *physics*: the system Hilbert-space dimension, the system
Hamiltonian that fixes the interaction picture, the system operators that couple
to the bath, the initial system state, and a description of the bath.

Models are backend-agnostic.  They return small (``d x d``) dense NumPy arrays;
the higher layers are responsible for moving operators onto the compute backend.
The interaction-picture coupling operators ``S_alpha(t) = e^{i H_S t} S_alpha
e^{-i H_S t}`` are provided through a default implementation that subclasses may
override with a closed form.
"""

from __future__ import annotations

import numbers
from abc import ABC, abstractmethod

import numpy as np


def validate_channel(channel, n_channels: int) -> int:
    """Validate a 1-based coupling-channel index and normalise it to a Python ``int``.

    Shared by every public entry point that takes a ``channel`` -- the solver
    (:class:`~edmtn.driver.solver.EDMSolver`), the HPC solve
    (:func:`~edmtn.evolution.cutensornet.solve_cutensornet`), and the observable extractor
    (:meth:`~edmtn.observables.extractor.ObservableExtractor.coupling_polarization_history`)
    -- so an illegal channel fails identically everywhere (``ValueError``) instead of, e.g.,
    ``channel=0`` silently selecting the last channel via a negative index, or a float/string
    leaking ``IndexError``/``TypeError`` deep in an array index.  ``n_channels`` is the number
    of coupling channels the caller exposes (``len(model.coupling_operators())`` for a model,
    ``(mps.d_phys - 1) // 2`` for an EDM-MPS).
    """
    if isinstance(channel, bool) or not isinstance(channel, numbers.Integral):
        raise ValueError(f"channel must be an integer in 1..{n_channels}, got {channel!r}")
    c = int(channel)
    if not 1 <= c <= n_channels:
        raise ValueError(f"channel must be in 1..{n_channels}, got {channel!r}")
    return c


def validate_sub_baths(sub_baths, K: int) -> int:
    """Resolve/validate the number of separable sub-baths to fold, against the model's ``K``.

    ``None`` means "all ``K``".  Otherwise it must be a non-bool integer in ``1..K``, returned
    as a Python ``int``.  There is NO silent clamp/truncation: an out-of-range value (``K+1``),
    a float (``2.9``), a bool, or a string raises ``ValueError`` instead of quietly changing how
    many bath spins are actually included.  Shared by Track 1, Track 2 and any direct evolution
    entry point so all of them reject an illegal value identically.
    """
    if sub_baths is None:
        return K
    if isinstance(sub_baths, bool) or not isinstance(sub_baths, numbers.Integral):
        raise ValueError(f"sub_baths must be None or an integer in 1..{K}, got {sub_baths!r}")
    value = int(sub_baths)
    if not 1 <= value <= K:
        raise ValueError(f"sub_baths must be in 1..{K}, got {sub_baths!r}")
    return value


#: numerical tolerance for :meth:`AbstractOQSModel.validate` (Hermiticity, trace, PSD floor).
#: Passed explicitly everywhere -- NumPy's ``allclose``/``isclose`` default ``rtol=1e-5`` is too
#: loose for a fixed normalisation like the unit trace.
_VALIDATE_TOL = 1e-8


def _as_finite_operator(name: str, value, d: int) -> np.ndarray:
    """Return ``value`` as a finite numeric ``(d, d)`` array, else raise ``ValueError``.

    Model methods are user-supplied, so a malformed return (a Python list without ``.shape``,
    a string/object array that would make ``np.isfinite`` raise ``TypeError``, a wrong shape)
    must surface as a descriptive ``ValueError`` naming the field -- never an ``AttributeError``
    / ``TypeError`` / ``LinAlgError`` leaking out of :meth:`AbstractOQSModel.validate`.  A
    non-numeric dtype is rejected outright (``"1"`` is NOT coerced to a complex ``1``).
    """
    try:
        arr = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} could not be read as a numeric array: {exc}") from exc
    if arr.shape != (d, d):
        raise ValueError(f"{name} shape {arr.shape} != ({d}, {d})")
    if arr.dtype == object or not np.issubdtype(arr.dtype, np.number):
        raise ValueError(f"{name} must be a numeric array, got dtype {arr.dtype!r}")
    try:
        finite = bool(np.all(np.isfinite(arr)))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite numeric array: {exc}") from exc
    if not finite:
        raise ValueError(f"{name} has non-finite entries (NaN/Inf)")
    return arr


class AbstractOQSModel(ABC):
    """Base class for open-quantum-system models.

    Subclasses must define the system operators, the initial state and a bath
    description.  Class attributes :attr:`bath_type` and :attr:`time_step_order`
    select the downstream pipeline and the Trotter expansion order.
    """

    #: 'gaussian' | 'separable' | 'chain' | 'generic'
    bath_type: str = "generic"

    #: order of the small-step expansion used by the evolution engine (1 or 2)
    time_step_order: int = 2

    # -- system ------------------------------------------------------------

    @property
    @abstractmethod
    def system_dim(self) -> int:
        """Dimension ``d`` of the system Hilbert space."""

    @abstractmethod
    def system_hamiltonian(self) -> np.ndarray:
        """The system Hamiltonian ``H_S`` (``d x d`` Hermitian).

        Defines the interaction picture in which the bath coupling is expressed.
        """

    @abstractmethod
    def coupling_operators(self) -> list[np.ndarray]:
        """Schroedinger-picture system operators ``{S_alpha}`` that couple to the bath."""

    @abstractmethod
    def system_operators(self) -> dict[str, np.ndarray]:
        """Named static system operators, used for observables.

        Should include at least the operators needed to evaluate the
        observables of interest (and conventionally the identity).
        """

    @abstractmethod
    def initial_system_state(self) -> np.ndarray:
        """Initial system density matrix ``rho(0)`` (``d x d``, Hermitian, trace 1)."""

    # -- bath --------------------------------------------------------------

    @abstractmethod
    def bath_params(self):
        """Model-specific bath parameters (typically a dataclass)."""

    def memory_time(self) -> float | None:
        """Finite bath memory time, or ``None`` if not imposing a cutoff."""
        return None

    # -- interaction picture (default implementations) --------------------

    def _propagator(self, t: float) -> np.ndarray:
        """``U(t) = e^{i H_S t}`` via eigendecomposition of the Hermitian ``H_S``."""
        w, V = np.linalg.eigh(self.system_hamiltonian())
        return (V * np.exp(1j * w * t)) @ V.conj().T

    def interaction_picture_operator(self, op: np.ndarray, t: float) -> np.ndarray:
        """Return ``e^{i H_S t} op e^{-i H_S t}``."""
        U = self._propagator(t)
        return U @ op @ U.conj().T

    def coupling_operators_at(self, t: float) -> list[np.ndarray]:
        """Interaction-picture coupling operators ``{S_alpha(t)}``."""
        return [self.interaction_picture_operator(S, t) for S in self.coupling_operators()]

    # -- validation --------------------------------------------------------

    def validate(self) -> None:
        """Sanity-check the model's dimension, operators and initial state; raise on any
        inconsistency (always ``ValueError`` for a malformed model, never a leaked
        ``AttributeError``/``TypeError``/``LinAlgError``).

        Checks, in order: ``system_dim`` is a positive integer; the Hamiltonian is a finite
        Hermitian ``(d, d)`` numeric array; the initial state is a finite ``(d, d)`` numeric
        array with unit trace (imag part within tolerance), Hermitian and positive
        semidefinite (min eigenvalue ``>= -tol``); there is at least one coupling operator and
        each is a finite ``(d, d)`` numeric array.  Coupling operators are NOT required to be
        Hermitian (future channels may use a non-Hermitian basis).  The solver calls this
        automatically in :class:`~edmtn.driver.solver.EDMSolver` before building any pipeline.
        """
        # 1. system dimension: a genuine positive integer (bool / float / <=0 rejected)
        d_raw = self.system_dim
        if isinstance(d_raw, bool) or not isinstance(d_raw, numbers.Integral) or int(d_raw) < 1:
            raise ValueError(f"system_dim must be a positive integer, got {d_raw!r}")
        d = int(d_raw)

        # 2. read every model method exactly once (a stateful/dynamic model must not be asked
        #    twice within one validation and possibly answer differently)
        H = self.system_hamiltonian()
        rho = self.initial_system_state()
        try:
            coupling_ops = list(self.coupling_operators())
        except TypeError as exc:
            raise ValueError(
                f"coupling_operators() must return an iterable of operators: {exc}") from exc

        # 3. Hamiltonian: numeric, (d, d), finite, Hermitian
        H = _as_finite_operator("system_hamiltonian", H, d)
        if not np.allclose(H, H.conj().T, rtol=_VALIDATE_TOL, atol=_VALIDATE_TOL):
            raise ValueError("system_hamiltonian is not Hermitian")

        # 4. initial state: numeric, (d, d), finite
        rho = _as_finite_operator("initial_system_state", rho, d)

        # 5. trace normalisation (absolute tol for a fixed unit trace; imag part near zero)
        tr = complex(np.trace(rho))
        if abs(tr.imag) > _VALIDATE_TOL:
            raise ValueError(
                f"initial_system_state trace has nonzero imaginary part {tr.imag:.3g} "
                f"(tol {_VALIDATE_TOL:g})")
        if not np.isclose(tr.real, 1.0, rtol=0.0, atol=_VALIDATE_TOL):
            raise ValueError(
                f"initial_system_state trace {tr.real:.6g} != 1 (tol {_VALIDATE_TOL:g})")

        # 6. initial state Hermitian
        if not np.allclose(rho, rho.conj().T, rtol=_VALIDATE_TOL, atol=_VALIDATE_TOL):
            raise ValueError("initial_system_state is not Hermitian")

        # 7. initial state positive semidefinite (eigvalsh on an explicitly Hermitised copy so a
        #    tiny non-Hermitian residue can't make it read only one triangle)
        rho_h = 0.5 * (rho + rho.conj().T)
        try:
            eig_min = float(np.linalg.eigvalsh(rho_h).min())
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                f"initial_system_state eigenvalue computation failed: {exc}") from exc
        if eig_min < -_VALIDATE_TOL:
            raise ValueError(
                f"initial_system_state is not positive semidefinite: minimum eigenvalue "
                f"{eig_min:.3g} < -{_VALIDATE_TOL:g}")

        # 8. at least one coupling operator
        if len(coupling_ops) == 0:
            raise ValueError("coupling_operators() is empty; the model has no coupling channels")

        # 9. each coupling operator: numeric, (d, d), finite (Hermiticity NOT required)
        for i, S in enumerate(coupling_ops):
            _as_finite_operator(f"coupling operator {i}", S, d)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(d={self.system_dim}, bath_type={self.bath_type!r})"
