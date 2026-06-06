"""Top-level EDM solver (Layer 7).

``EDMSolver`` orchestrates the full stack: it auto-selects the engine pipeline
from the model's ``bath_type`` (Layer 7 :mod:`auto_config`), evolves the EDM-MPS
(Layer 5), and extracts observables (Layer 6).  The coupling-channel
polarization ``<S_a(t)>`` is always returned (the cheap Eq.-F2 sweep); custom
operators are evaluated from the recorded reduced states when ``record_rho`` is
set.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..observables.convergence import is_converged, max_history_deviation
from ..observables.extractor import ObservableExtractor
from .auto_config import SolverConfig, build_pipeline


@dataclass
class SolverResult:
    """Result of :meth:`EDMSolver.solve`.

    Attributes
    ----------
    times : ndarray
        Step times ``[eps, 2 eps, ..., T]`` (ascending).
    polarization : ndarray
        Coupling-channel ``<S_a(t)>`` over ``times`` (Eq. F2).
    bond_dims : list[int]
        Maximum bond dimension after each step.
    truncation_errors : list[float]
        Largest discarded weight per step.
    observables : dict[str, ndarray]
        Custom observable histories (empty unless requested + ``record_rho``).
    mps : EDMMPS
        Final EDM-MPS.
    evolution : EvolutionResult
        Raw Layer-5 output.
    """

    times: object
    polarization: object
    bond_dims: list
    truncation_errors: list
    observables: dict = field(default_factory=dict)
    mps: object = None
    evolution: object = None

    @property
    def max_bond(self) -> int:
        return max(self.bond_dims) if self.bond_dims else 1


class EDMSolver:
    """Driver that solves the reduced dynamics of an open quantum system.

    Parameters
    ----------
    model : AbstractOQSModel
        The physical model (Layer 1).
    config : SolverConfig
        Time grid and truncation controls.
    """

    def __init__(self, model, config: SolverConfig):
        self.model = model
        self.config = config
        self.kernel_engine, self.evolution = build_pipeline(model, config)

    @classmethod
    def from_model(cls, model, *, T: float, eps: float, **kwargs) -> "EDMSolver":
        """Build a solver with a fresh :class:`SolverConfig`."""
        return cls(model, SolverConfig(eps=eps, T=T, **kwargs))

    # -- main entry point --------------------------------------------------

    def solve(self, observables: dict | None = None, *, channel: int = 1) -> SolverResult:
        """Evolve and extract observables.

        Parameters
        ----------
        observables : dict[str, callable], optional
            Mapping ``name -> operator_fn(t)`` evaluated as ``Tr[O(t) rho(t)]``
            from the recorded reduced states (requires ``config.record_rho``).
        channel : int
            Coupling channel whose polarization history is returned.
        """
        cfg = self.config
        # the efficient Eq.-F2 sweep is first-order specific; second order reads
        # the coupling polarization from the recorded reduced states instead.
        second_order = cfg.expansion_order == 2
        need_rho = cfg.record_rho or bool(observables) or second_order
        ev = self.evolution.run(
            self.model,
            self.kernel_engine,
            cfg.eps,
            cfg.n_steps,
            max_bond=cfg.max_bond,
            cutoff=cfg.cutoff,
            cutoff_mode=cfg.cutoff_mode,
            ref_index=cfg.ref_index,
            record_rho=need_rho,
        )

        if second_order:
            ci = channel - 1
            times, vals = ObservableExtractor.expectation_history(
                ev.density_matrices,
                ev.times,
                lambda t: self.model.coupling_operators_at(t)[ci],
            )
            pol = vals.real
        else:
            times, pol = ObservableExtractor.coupling_polarization_history(
                ev.mps, cfg.eps, channel=channel
            )

        extra: dict = {}
        if observables:
            for name, op_fn in observables.items():
                _, vals = ObservableExtractor.expectation_history(
                    ev.density_matrices, ev.times, op_fn
                )
                extra[name] = vals

        return SolverResult(
            times=times,
            polarization=pol,
            bond_dims=ev.bond_dims,
            truncation_errors=ev.truncation_errors,
            observables=extra,
            mps=ev.mps,
            evolution=ev,
        )

    # -- convergence helpers ----------------------------------------------

    def timestep_convergence(self, *, tol: float | None = None, channel: int = 1):
        """Compare the polarization at ``eps`` and ``eps/2``.

        Returns ``(deviation, converged_or_None)``: the maximum deviation on the
        common time grid and, if ``tol`` is given, whether it is within ``tol``.
        """
        coarse = self.solve(channel=channel)
        fine_cfg = SolverConfig(
            eps=self.config.eps / 2,
            T=self.config.T,
            cutoff=self.config.cutoff,
            cutoff_mode=self.config.cutoff_mode,
            max_bond=self.config.max_bond,
            ref_index=self.config.ref_index,
            expansion_order=self.config.expansion_order,
            decomposition=self.config.decomposition,
        )
        fine = EDMSolver(self.model, fine_cfg).solve(channel=channel)
        dev = max_history_deviation(
            coarse.times, coarse.polarization, fine.times, fine.polarization
        )
        ok = None if tol is None else (dev <= tol)
        return dev, ok


def solve(model, *, T: float, eps: float, observables: dict | None = None, **kwargs) -> SolverResult:
    """Convenience one-shot solve."""
    return EDMSolver.from_model(model, T=T, eps=eps, **kwargs).solve(observables)
