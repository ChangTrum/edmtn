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
        """Sanity-check the operators and initial state; raise on inconsistency."""
        d = self.system_dim
        H = self.system_hamiltonian()
        if H.shape != (d, d):
            raise ValueError(f"system_hamiltonian shape {H.shape} != ({d}, {d})")
        if not np.allclose(H, H.conj().T):
            raise ValueError("system_hamiltonian is not Hermitian")
        rho = self.initial_system_state()
        if rho.shape != (d, d):
            raise ValueError(f"initial_system_state shape {rho.shape} != ({d}, {d})")
        if not np.allclose(rho, rho.conj().T):
            raise ValueError("initial_system_state is not Hermitian")
        if not np.isclose(np.trace(rho), 1.0):
            raise ValueError(f"initial_system_state trace {np.trace(rho):.3g} != 1")
        for i, S in enumerate(self.coupling_operators()):
            if S.shape != (d, d):
                raise ValueError(f"coupling operator {i} shape {S.shape} != ({d}, {d})")

    def __repr__(self) -> str:
        return f"{type(self).__name__}(d={self.system_dim}, bath_type={self.bath_type!r})"
