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

    Every array below names its own horizontal axis, so callers never have to inspect the
    internal ``evolution`` object to know what an index means.

    Attributes
    ----------
    times : ndarray
        Physical time grid ``[eps, 2 eps, ..., T]`` (ascending).
    polarization : ndarray
        Coupling-channel ``<S_a(t)>`` over ``times`` (Eq. F2).
    density_matrices : list[ndarray] or None
        ``rho(t)`` aligned 1:1 with ``times`` (so ``len == len(times)``), or ``None`` when the
        pipeline produces no time-axis reduced-state history.  Single-bath: present whenever the
        evolution recorded reduced states (``record_rho``, custom observables, OR second order --
        the ρ(t) is exposed as-is, not re-hidden).  Track 2: always present (first-class output).
        **Separable/Gaudin Track 1: always None** -- its per-``L`` states are ``rho_L(T)`` (axis =
        sub-bath count, not time) and live in ``sub_bath_final_density_matrices``, never here.
    time_bond_dims : list[int] or None
        Max bond dimension after each *physical time step* (``len == len(times)``), or ``None`` if
        the pipeline has no per-time-step bond history (Track 1 separable, Track 2).
    sub_bath_counts : list[int] or None
        Separable Track 1: the recorded sub-bath counts ``L`` (``evolution.recorded_L``).
    sub_bath_bond_dims : list[int] or None
        Separable Track 1: ``D_L`` after folding in ``L`` sub-baths, aligned with ``sub_bath_counts``.
    sub_bath_final_density_matrices : list[ndarray] or None
        Separable Track 1: ``rho_L(T)`` per recorded ``L`` (aligned with ``sub_bath_counts``); only
        present when ``record_rho``.  ``None`` otherwise / on other pipelines.
    final_time_bond_dims : list[int] or None
        The final EDM-MPS's internal bond dimensions along the *time* chain (``mps.bond_dims``).
        NOT aligned with ``times``: length is ``mps.num_sites - 1`` (``num_sites == order*n_steps``).
        ``None`` when there is no MPS (Track 2).
    sub_baths_used : int or None
        Separable Track 1 / Track 2: the actual number of sub-baths ``L`` folded (the resolved
        ``sub_baths``; ``K`` when ``sub_baths=None``).  ``None`` for non-separable models -- so the
        caller can tell how many bath spins were really included, without guessing from the request.
    bond_dims : list[int]
        **Legacy, pipeline-specific bond history** (kept for back-compat; powers ``max_bond``):
        single-bath Track 1 = alias of ``time_bond_dims``; separable Track 1 = alias of
        ``sub_bath_bond_dims``; Track 2 = ``[]`` (no boundary-MPS bond history).  Prefer the
        axis-explicit fields above for new code.
    truncation_errors : list[float | None]
        Largest per-bond **discarded weight** -- ``max_b sum_{i discarded at bond b} sigma_i**2``
        -- of each recorded compression, on the pipeline's own axis: per physical time step for
        single-bath Track 1 (order 2 takes the max over both sub-steps, so it stays aligned with
        ``times``), and per recorded sub-bath count ``L`` for separable Track 1 (the max over every
        fold since the previous recorded ``L``, so a ``record_every > 1`` drops nothing).  This is
        the discarded WEIGHT, NOT quimb's discarded 2-norm (``sqrt(sum sigma**2)``), and it is a
        LOCAL per-record quantity -- not a cumulative or global error bound for the trajectory.
        ``0.0`` = a compression ran and discarded nothing (or none ran, e.g. ``compress=False``);
        ``None`` = the chosen decomposition cannot measure it exactly (``compress_decomp='rsvd'``,
        whose randomized sketch never sees the tail of the spectrum it omitted).  Track 2 returns
        ``[]`` -- it is exact-only and performs no truncation.
    expansion_order : int
        The resolved Trotter order actually used (``1`` or ``2``).
    observables : dict[str, ndarray]
        Custom observable histories (empty unless requested + reduced states recorded).
        Not supported on separable Track 1 or on Track 2 (both raise ``NotImplementedError``).
    backend : str
        The device/track that ACTUALLY ran, e.g. ``'cpu/f64'``, ``'gpu/f64'``,
        ``'hpc/exact/cuquantum'`` (``.../<n>gpu`` when distributed).  A requested GPU that
        was unavailable shows as CPU with a ``(fallback: ...)`` suffix, so this is the
        honest record of what executed -- not what was asked for.
    mps : EDMMPS or None
        Final EDM-MPS; ``None`` on Track 2 (the 2D contraction builds no MPS).
    evolution : EvolutionResult or None
        Raw Layer-5 output (internal; the top-level fields above are the public contract).
        ``None`` on Track 2, which has no Layer-5 evolution object.
    error_metrics : dict or None
        Track 2 only: reference error metrics (``‖ρ−ρ†‖`` / ``|Tr ρ−1|`` + optimizer stats).
    """

    times: object
    polarization: object
    bond_dims: list
    truncation_errors: list[float | None]
    expansion_order: int
    observables: dict = field(default_factory=dict)
    mps: object = None
    evolution: object = None
    backend: str = ""
    density_matrices: object = None  # rho(t) aligned with times, else None (see docstring)
    error_metrics: dict | None = None  # hpc only: ‖ρ−ρ†‖ / |Tr ρ−1| (+ slices/flops or discarded weight)
    # -- P0-8 axis-explicit fields (all default None; keep manual SolverResult(...) working) --
    time_bond_dims: object = None                    # max bond per physical time step (∥ times)
    sub_bath_counts: object = None                   # separable T1: recorded L values
    sub_bath_bond_dims: object = None                # separable T1: D_L (∥ sub_bath_counts)
    sub_bath_final_density_matrices: object = None   # separable T1: rho_L(T) (∥ sub_bath_counts, if record_rho)
    final_time_bond_dims: object = None              # final EDM-MPS internal bonds along the time chain
    sub_baths_used: int | None = None                # actual number of sub-baths folded (None if N/A)

    @property
    def max_bond(self) -> int:
        """Largest entry of the legacy ``bond_dims`` -- so its axis follows that alias:
        the per-time-step maximum on single-bath Track 1, the per-fold (``L``) maximum on
        separable Track 1, and ``1`` on Track 2 (whose ``bond_dims`` is ``[]``)."""
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
        Self-describing record of the comparison, with keys:

        * ``coarse_config`` / ``fine_config`` -- the FULL :class:`SolverConfig` of each run
          (so no field can be silently dropped as new knobs are added);
        * ``channel`` -- the normalised 1-based channel;
        * ``tolerance`` -- the ``tol`` passed in (``None`` if omitted);
        * ``coarse_backend`` / ``fine_backend`` -- the ACTUAL executed backend labels
          (revealing e.g. a GPU->CPU fallback);
        * ``coarse_sub_baths_used`` / ``fine_sub_baths_used`` -- the number of sub-baths
          each run really folded, read back from the results rather than the request.

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
        # reject a malformed model up front -- BEFORE resolving the config or building any
        # pipeline/kernel -- so a bad Hamiltonian / initial state / coupling set fails loudly at
        # construction (both tracks) instead of surfacing deep in the cumulant/kernel build
        self.model.validate()
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
            bond_dims=ev.bond_dims,                        # legacy alias of time_bond_dims here
            truncation_errors=ev.truncation_errors,
            expansion_order=cfg.expansion_order,
            observables=extra,
            mps=ev.mps,
            evolution=ev,
            backend=backend_label,
            density_matrices=ev.density_matrices,          # rho(t) if the evolution recorded it, else None
            time_bond_dims=ev.bond_dims,                   # max bond per physical time step
            final_time_bond_dims=ev.mps.bond_dims,         # final MPS internal bonds along time
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
            sub_baths_used=out["sub_baths_used"],
        )

    # -- separable bath (outer-loop recursion) ----------------------------

    def _solve_separable(self, observables: dict | None, *, channel: int) -> SolverResult:
        """Solve a separable-bath model (Eq. 21 outer loop over sub-baths).

        Returns the all-times coupling-channel polarization for the full bath
        (``<S_a(t)>`` vs ``t``; channel ``3`` is ``<S_z>`` for the Gaudin model).  The
        per-``L`` fold records are published at the top level (no need to read
        ``result.evolution``): the sub-bath counts ``L`` on ``result.sub_bath_counts``,
        ``D_L`` on ``result.sub_bath_bond_dims``, ``rho_L(T)`` on
        ``result.sub_bath_final_density_matrices`` (when ``record_rho``), and the final
        EDM-MPS's per-time internal bonds ``D_t`` (Fig. 6b) on ``result.final_time_bond_dims``.
        There is no time-axis ``rho(t)`` history, so ``result.density_matrices`` is ``None``.
        """
        if observables:
            raise NotImplementedError(
                "custom per-time observables are not supported for separable baths; "
                "use the channel polarization, or run with record_rho=True and read "
                "result.sub_bath_final_density_matrices (the per-L rho_L(T))"
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
            bond_dims=ev.bond_dims,                            # legacy alias of sub_bath_bond_dims here
            truncation_errors=ev.truncation_errors,
            expansion_order=cfg.expansion_order,
            observables={},
            mps=ev.mps,
            evolution=ev,
            backend=backend_label,
            # density_matrices stays None: the per-L states below are rho_L(T), not rho(t)
            sub_bath_counts=ev.recorded_L,
            sub_bath_bond_dims=ev.bond_dims,
            sub_bath_final_density_matrices=ev.density_matrices,   # rho_L(T) if record_rho, else None
            final_time_bond_dims=ev.mps.bond_dims,
            sub_baths_used=ev.n_sub_baths,                         # actual L folded (== validate_sub_baths)
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
            "coarse_sub_baths_used": coarse.sub_baths_used,  # ACTUAL L folded (None-request -> K),
            "fine_sub_baths_used": fine.sub_baths_used,       # taken from the results, not the request
        }
        return TimestepConvergence(dev, ok, metadata)


def solve(
    model, *, T: float, eps: float, observables: dict | None = None, channel: int = 1, **kwargs
) -> SolverResult:
    """Convenience one-shot solve."""
    return EDMSolver.from_model(model, T=T, eps=eps, **kwargs).solve(observables, channel=channel)
