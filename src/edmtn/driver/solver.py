"""Top-level EDM solver (Layer 7).

``EDMSolver`` orchestrates the full stack: it auto-selects the engine pipeline
from the model's ``bath_type`` (Layer 7 :mod:`auto_config`), evolves the EDM-MPS
(Layer 5), and extracts observables (Layer 6).  The coupling-channel
polarization ``<S_a(t)>`` is always returned (the cheap Eq.-F2 sweep); custom
operators are evaluated from the recorded reduced states when ``record_rho`` is
set.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from ..models.base import validate_channel
from ..observables.convergence import is_converged, max_history_deviation
from ..observables.extractor import ObservableExtractor
from .auto_config import SolverConfig, build_pipeline, resolve_config_for_model


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
    expansion_order : int
        The resolved Trotter order actually used (``1`` or ``2``) -- the model's
        ``time_step_order`` unless the config overrode it.  Recorded so the result's
        metadata matches the algorithm that produced it.
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
    expansion_order: int
    observables: dict = field(default_factory=dict)
    mps: object = None
    evolution: object = None
    backend: str = ""
    density_matrices: object = None  # ρ(t) history (the hpc track returns this first-class)
    error_metrics: dict | None = None  # hpc only: ‖ρ−ρ†‖ / |Tr ρ−1| (+ slices/flops or discarded weight)

    @property
    def max_bond(self) -> int:
        return max(self.bond_dims) if self.bond_dims else 1


@dataclass(frozen=True)
class TimestepConvergence:
    """Result of :meth:`EDMSolver.timestep_convergence`.

    Attributes
    ----------
    deviation : float
        Max ``|Δ<S_a(t)>|`` between the ``eps`` and ``eps/2`` runs on the common time grid.
    converged : bool or None
        ``deviation <= tol`` (or ``None`` when no ``tol`` was given).
    metadata : dict
        Self-describing record of the comparison: the FULL ``coarse_config`` / ``fine_config``
        (:class:`SolverConfig`, so no field can be silently dropped as new knobs are added),
        the normalised ``channel``, the ``tolerance``, and the ACTUAL executed
        ``coarse_backend`` / ``fine_backend`` labels (which reveal e.g. a GPU→CPU fallback).

    Backward compatible with the legacy 2-tuple contract: ``dev, ok = result``,
    ``result[0]`` / ``result[1]`` and ``len(result) == 2`` all still work.
    """

    deviation: float
    converged: bool | None
    metadata: dict

    def __iter__(self):
        yield self.deviation
        yield self.converged

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index):
        return (self.deviation, self.converged)[index]


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
        # resolve the effective order ONCE (config default None -> model.time_step_order),
        # store the resolved config so every layer reads the same order; original untouched
        self.config = resolve_config_for_model(model, config)
        # the hpc (cuQuantum 2D) track builds nothing from the Track-1 pipeline
        if self.config.backend == "hpc":
            self.kernel_engine = self.evolution = None
        else:
            self.kernel_engine, self.evolution = build_pipeline(model, self.config)

    @classmethod
    def from_model(cls, model, *, T: float, eps: float, **kwargs) -> "EDMSolver":
        """Build a solver with a fresh :class:`SolverConfig`."""
        return cls(model, SolverConfig(eps=eps, T=T, **kwargs))

    # -- backend selection -------------------------------------------------

    def _resolve_backend(self):
        """Return ``(convert, memory, label)`` for the configured backend.

        ``backend='cpu'`` (the default) runs on NumPy: the EDM hot path is many
        sequential medium SVD/QR calls with Python orchestration between them,
        where the CPU beats the GPU at the bond dimensions these phases reach
        (benchmarks in ``tests/benchmarks/perf_*`` and the analysis
        in ``docs/benchmarks/cpu-vs-gpu-edm.md``).  The GPU stays a first-class, validated
        option via explicit ``backend='gpu'`` -- it becomes the faster path once
        the Phase-3 decomposition layer (randomized / single-pass SVD) shifts the
        work onto large GEMM-bound operations.  Falls back to CPU (never raises)
        if a GPU is requested but unavailable.
        """
        from ..backend import ArrayFactory, PrecisionPolicy

        cfg = self.config
        pref = cfg.backend
        if pref not in ("cpu", "numpy", "gpu", "cupy"):
            raise ValueError(f"unknown backend {cfg.backend!r}")
        precision = (
            PrecisionPolicy.mixed() if cfg.precision == "mixed" else PrecisionPolicy.full_f64()
        )
        if pref in ("gpu", "cupy"):
            factory = ArrayFactory.auto(prefer="cupy", precision=precision)
        else:
            factory = ArrayFactory("numpy", precision=precision)
        # CPU/complex128 needs no cast -- keep the Phase-1 path byte-for-byte.
        if not factory.is_gpu and precision.contract == "f64":
            convert = None
        else:
            convert = factory.caster("contract")
        label = ("gpu" if factory.is_gpu else "cpu") + "/" + precision.contract
        if factory.fallback_reason:
            label += f" (fallback: {factory.fallback_reason})"
        return convert, factory.memory, label

    # -- main entry point --------------------------------------------------

    def solve(self, observables: dict | None = None, *, channel: int = 1) -> SolverResult:
        """Evolve and extract observables.

        Parameters
        ----------
        observables : dict[str, callable], optional
            Mapping ``name -> operator_fn(t)`` evaluated as ``Tr[O(t) rho(t)]``
            from the recorded reduced states (requires ``config.record_rho``).
        channel : int
            Coupling channel (1-based) whose polarization history is returned.
        """
        # validate once, before any backend/bath dispatch or evolution -- every inner path
        # then receives a normalised Python int (no negative-index channel selection)
        channel = validate_channel(channel, len(self.model.coupling_operators()))
        if self.config.backend == "hpc":
            return self._solve_hpc(observables, channel=channel)
        if self.model.bath_type == "separable":
            return self._solve_separable(observables, channel=channel)

        cfg = self.config
        convert, _memory, backend_label = self._resolve_backend()
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
            record_rho=need_rho,
            convert=convert,
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
            expansion_order=cfg.expansion_order,
            observables=extra,
            mps=ev.mps,
            evolution=ev,
            backend=backend_label,
        )

    # -- hpc track (cuQuantum 2D one-shot contraction) --------------------

    def _solve_hpc(self, observables: dict | None, *, channel: int) -> SolverResult:
        """Solve on the HPC track: lay the EDM out as a 2D space×time network and
        contract it with cuQuantum (cuTensorNet). Returns ρ(t) first-class plus the
        channel polarization and the reference error metrics."""
        if observables:
            raise NotImplementedError(
                "custom per-time observables are not supported on the hpc track; "
                "read result.density_matrices (ρ(t)) or use the channel polarization")
        from ..evolution.cutensornet import solve_cutensornet  # noqa: PLC0415

        out = solve_cutensornet(self.model, self.config, channel=channel,
                                executor="cuquantum")
        label = f"hpc/{out['mode']}/{out['pathfinder']}"
        if out.get("ngpu", 1) > 1:
            label += f"/{out['ngpu']}gpu"
        return SolverResult(
            times=out["times"],
            polarization=out["polarization"],
            bond_dims=[],            # cuTensorNet manages bonds internally (one-shot)
            truncation_errors=[],
            expansion_order=self.config.expansion_order,
            observables={},
            mps=None,
            evolution=None,
            backend=label,
            density_matrices=out["density_matrices"],
            error_metrics=out["error_metrics"],
        )

    # -- separable bath (outer-loop recursion) ----------------------------

    def _solve_separable(self, observables: dict | None, *, channel: int) -> SolverResult:
        """Solve a separable-bath model (Eq. 21 outer loop over sub-baths).

        Returns the all-times coupling-channel polarization for the full bath
        (``<S_a(t)>`` vs ``t``; channel ``3`` is ``<S_z>`` for the Gaudin model).
        ``bond_dims`` reports the maximum bond after folding in each sub-bath;
        the per-time bond dimension ``D_t`` (Fig. 6b) is ``result.mps.bond_dims``,
        and per-sub-bath final states / bonds are on ``result.evolution``.
        """
        if observables:
            raise NotImplementedError(
                "custom per-time observables are not supported for separable baths; "
                "use channel polarization, or read result.evolution.density_matrices"
            )
        cfg = self.config
        convert, memory, backend_label = self._resolve_backend()
        ev = self.evolution.run(
            self.model,
            self.kernel_engine,
            cfg.eps,
            cfg.n_steps,
            max_bond=cfg.max_bond,
            cutoff=cfg.cutoff,
            cutoff_mode=cfg.cutoff_mode,
            record_rho=cfg.record_rho,
            sub_baths=cfg.sub_baths,
            convert=convert,
            memory=memory,
        )
        _, raw_pol = ObservableExtractor.coupling_polarization_history(
            ev.mps, cfg.eps, channel=channel, order=cfg.expansion_order
        )
        # The Eq.-F2 sweep yields <S_a(t)> at t = 0, eps, ..., (N-1) eps (measured *before*
        # each Trotter step).  Put it on the PUBLIC axis eps, 2eps, ..., T -- the axis
        # spin-boson, Track 2 and the SolverResult docstring already use -- by dropping the
        # t=0 point and appending the final-time value Tr[S_a(T) rho(T)] read from the final
        # MPS (backend-safe via ObservableExtractor.expectation; no record_rho needed).
        N = cfg.n_steps
        Sop = self.model.coupling_operators_at(N * cfg.eps)[channel - 1]
        p_T = float(ObservableExtractor.expectation(ev.mps, Sop).real)
        pol = np.concatenate((raw_pol[1:], np.asarray([p_T], dtype=np.float64)))
        times = cfg.eps * np.arange(1, N + 1, dtype=np.float64)
        return SolverResult(
            times=times,
            polarization=pol,
            bond_dims=ev.bond_dims,
            truncation_errors=ev.truncation_errors,
            expansion_order=cfg.expansion_order,
            observables={},
            mps=ev.mps,
            evolution=ev,
            backend=backend_label,
        )

    # -- convergence helpers ----------------------------------------------

    def timestep_convergence(self, *, tol: float | None = None,
                             channel: int = 1) -> TimestepConvergence:
        """Compare the coupling polarization at ``eps`` and ``eps/2``.

        The fine run is built with ``dataclasses.replace(self.config, eps=eps/2)``, so it
        inherits EVERY resolved config field except ``eps`` (``sub_baths``, ``backend``,
        ``precision``, ``preset``, ``record_rho``, ``pathfinder``, cutoff/bond/compression,
        and any future knob).  Coarse and fine are therefore the SAME physical model, differing
        only in the time step (``n_steps`` doubles) -- fixing the old hand-copied config that
        dropped fields and silently compared a different model (e.g. ``sub_baths=1`` reverting
        to the full bath, or a requested GPU/HPC fine run being silently replaced by the
        default CPU backend).

        Returns a :class:`TimestepConvergence` (``.deviation`` / ``.converged`` / ``.metadata``);
        it still unpacks as the legacy ``dev, ok = ...`` 2-tuple.
        """
        channel = validate_channel(channel, len(self.model.coupling_operators()))
        coarse = self.solve(channel=channel)
        fine_cfg = replace(self.config, eps=self.config.eps / 2)
        fine = EDMSolver(self.model, fine_cfg).solve(channel=channel)
        dev = max_history_deviation(
            coarse.times, coarse.polarization, fine.times, fine.polarization
        )
        ok = None if tol is None else (dev <= tol)
        metadata = {
            "coarse_config": self.config,        # full SolverConfig -> no field can be dropped
            "fine_config": fine_cfg,
            "channel": channel,                  # normalised Python int
            "tolerance": tol,
            "coarse_backend": coarse.backend,    # ACTUAL executed labels (reveal GPU->CPU fallback)
            "fine_backend": fine.backend,
        }
        return TimestepConvergence(dev, ok, metadata)


def solve(
    model, *, T: float, eps: float, observables: dict | None = None, channel: int = 1, **kwargs
) -> SolverResult:
    """Convenience one-shot solve."""
    return EDMSolver.from_model(model, T=T, eps=eps, **kwargs).solve(observables, channel=channel)
