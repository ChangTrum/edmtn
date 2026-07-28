"""Top-level EDM solver (Layer 7).

``EDMSolver`` orchestrates the full stack: it auto-selects the engine pipeline
from the model's ``bath_type`` (Layer 7 :mod:`auto_config`), evolves the EDM-MPS
(Layer 5), and extracts observables (Layer 6).  The coupling-channel
polarization ``<S_a(t)>`` is returned by the pipelines that define it (the cheap
Eq.-F2 sweep) and is ``None`` on ``separable_td``, which does not; custom
operators are evaluated from the recorded reduced states when ``record_rho`` is
set.  ``final_density_matrix`` is filled on every pipeline.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace

import numpy as np

from ..evolution.mps_utils import _xp
from ..models.base import validate_channel
from ..observables.convergence import is_converged, max_history_deviation
from ..observables.extractor import (
    ObservableExtractor,
    finite_complex_expectation,
    real_scalar_expectation,
)
from .auto_config import SolverConfig, build_pipeline, resolve_config_for_model

#: The closed vocabulary of :meth:`EDMSolver.solve`'s ``moments`` argument.  Deliberately
#: small and fixed: an unknown name is a ``ValueError`` on every model, so a typo can never
#: be read as "compute nothing".  ``g2(0)`` is **not** here -- it is the ratio
#: ``n_factorial2 / n**2``, whose denominator passes through zero on a run started from the
#: vacuum, and choosing the interval on which that ratio is meaningful is the caller's
#: physics decision, not the pipeline's.
MOMENT_NAMES = ("n", "n_factorial2", "Jx", "Jy", "Jz", "Jabs")

#: Moments that need a bath-side measurement insertion, i.e. an extra folded chain per
#: channel.  The rest are post-processing of the reduced cavity state and cost nothing.
_TANGENT_MOMENTS = frozenset({"Jx", "Jy", "Jz", "Jabs"})
_JPLUS_MOMENTS = frozenset({"Jx", "Jy", "Jabs"})
_JZ_MOMENTS = frozenset({"Jz", "Jabs"})


@dataclass
class SolverResult:
    """Result of :meth:`EDMSolver.solve`.

    Every array below names its own horizontal axis, so callers never have to inspect the
    internal ``evolution`` object to know what an index means.

    Attributes
    ----------
    times : ndarray
        Physical time grid ``[eps, 2 eps, ..., T]`` (ascending).
    polarization : ndarray or None
        Coupling-channel ``<S_a(t)>`` over ``times`` (Eq. F2) -- **not available on every
        pipeline**.  ``None`` on the ``separable_td`` (Dicke) pipeline, which publishes no
        time-resolved coupling-channel history; read ``final_density_matrix`` (or the
        per-``L`` states) there instead.
    final_density_matrix : ndarray or None
        The reduced density matrix at the end of the solve, on **every** pipeline and
        regardless of ``record_rho`` -- so a successful solve always returns a physical
        state, not just bond dimensions.  Reuses a state the pipeline already computed
        where one exists, so it costs no extra contraction.  In the **backend-native**
        array type (a CuPy array after a GPU run), like the other reduced-state fields.
        **Separable pipelines:** this is ``rho_L(T)`` for ``L = sub_baths_used`` folded
        sub-baths -- with ``sub_baths < K`` it is *not* the full-``K`` result.
    density_matrices : list[ndarray] or None
        ``rho(t)`` aligned 1:1 with ``times`` (so ``len == len(times)``), or ``None`` when the
        pipeline produces no time-axis reduced-state history.  Single-bath: present whenever the
        evolution recorded reduced states (``record_rho``, custom observables, OR second order --
        the ρ(t) is exposed as-is, not re-hidden).  Track 2: always present (first-class output).
        **Separable Track 1 (Gaudin AND Dicke):** ``None`` by default; populated by
        ``record_time_reads=True``, which reads every physical step off ONE fold via the
        causal-prefix terminators (:mod:`edmtn.evolution.prefix_reads`) and requires
        ``compress_method='dm_tracking'`` when compressing.  Their per-``L`` states are
        ``rho_L(T)`` (axis = sub-bath count, not time) and live in
        ``sub_bath_final_density_matrices``, never here.
    time_bond_dims : list[int] or None
        Max bond dimension after each *physical time step* (``len == len(times)``), or ``None`` if
        the pipeline has no per-time-step bond history (Track 1 separable, Track 2).
    sub_bath_counts : list[int] or None
        Separable Track 1 (Gaudin and Dicke): the recorded sub-bath counts ``L``.
    sub_bath_bond_dims : list[int] or None
        Separable Track 1 (Gaudin and Dicke): ``D_L`` after folding in ``L`` sub-baths,
        aligned with ``sub_bath_counts``.
    sub_bath_final_density_matrices : list[ndarray] or None
        Separable Track 1 (Gaudin and Dicke): ``rho_L(T)`` per recorded ``L`` (aligned with
        ``sub_bath_counts``); only present when ``record_rho``.  ``None`` otherwise / on
        other pipelines.
    final_time_bond_dims : list[int] or None
        The final EDM-MPS's internal bond dimensions along the *time* chain (``mps.bond_dims``).
        NOT aligned with ``times``: length is ``mps.num_sites - 1`` (``num_sites == order*n_steps``).
        ``None`` when there is no MPS (Track 2).
    sub_baths_used : int or None
        Separable Track 1 (Gaudin and Dicke) / Track 2: the actual number of sub-baths ``L`` folded (the resolved
        ``sub_baths``; ``K`` when ``sub_baths=None``).  ``None`` for non-separable models -- so the
        caller can tell how many bath spins were really included, without guessing from the request.
    bond_dims : list[int]
        **Legacy, pipeline-specific bond history** (kept for back-compat; powers ``max_bond``):
        single-bath Track 1 = alias of ``time_bond_dims``; separable Track 1 = alias of
        ``sub_bath_bond_dims``; Track 2 = ``[]`` (no boundary-MPS bond history).  Prefer the
        axis-explicit fields above for new code.
    truncation_errors : list[float | None]
        Largest per-bond **discarded weight** of each public record -- measured as
        ``max_b sum_{i discarded at bond b} sigma_i**2`` on the ``zipup``/``direct`` exact
        paths, and as ``max_b sum_{i discarded at bond b} lambda_i`` of the discarded
        density-matrix eigenvalues (``lambda_i = sigma_i**2``, summed directly, NOT
        ``lambda_i**2``) on the ``dm`` path -- on the pipeline's own axis: per physical time step for
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
    moments : dict or None
        Final-time basic observables, ``None`` unless ``moments`` was requested (nothing
        is computed by default).  Contains exactly the requested names, plus the
        by-products of a channel that had to run anyway (``Jx`` and ``Jy`` come from one
        chain, so each brings the other; ``Jabs`` brings all three components), plus
        ``trace`` -- ``Tr rho(T)``, raw and un-normalised, so a truncation or step-size
        trace deviation stays visible instead of being silently divided out.  Requesting a
        single component never triggers the other channel and never computes ``Jabs``.
        Scalars are Python numbers: ``trace`` is ``complex``, everything else ``float``
        (``Jx``/``Jy`` are the real and imaginary parts of ``<J_+>``, which is complex by
        construction; ``n``, ``n_factorial2`` and ``Jz`` pass an imaginary-part guard).
        The collective-spin values are **sums over the sub-baths actually folded**, so with
        ``sub_baths = L < K`` they describe the first ``L`` spins and are bounded by
        ``|<J>| <= L/2`` -- a bound that holds for a *normalised* physical state, so
        judge it together with the raw ``trace`` returned alongside (nothing is
        normalised on the way out).
    moment_truncation_errors : dict or None
        ``{channel: list[float | None]}`` for the collective-spin channels that ran
        (``'Jplus'``, ``'Jz'``), aligned with ``sub_bath_counts`` and carrying the same
        per-interval semantics as ``truncation_errors``.  Keyed by **channel**, so ``Jx``
        and ``Jy`` share one record rather than appearing as two independent copies.
        ``None`` when no spin moment was requested.  Recorded separately from
        ``truncation_errors`` on purpose: once the chains are compressed the tangent is a
        jet *approximation*, and the value channel's record is not evidence about it.
    compression_method_used : str or None
        The outer 1D-compress path actually entered -- ``'zipup'``, ``'direct'``, ``'dm'``
        or ``'dm_tracking'``.  ``None`` when no compression ran, or on Track 2.  It does
        **not** report the per-bond decomposition: ``compress_decomp='rsvd'`` carries a
        silent per-bond guard that falls back to the exact full SVD, so one run can mix the
        two and a single string could not describe it.
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
    final_density_matrix: object = None              # reduced state at the end of the solve (all pipelines)
    compression_method_used: str | None = None       # outer 1D-compress path entered (see docstring)
    moments: dict | None = None                      # requested final-time moments (see docstring)
    moment_truncation_errors: dict | None = None     # per spin CHANNEL, ∥ sub_bath_counts

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

    def _resolve_channel(self, channel):
        """Normalise the requested coupling channel, or ``None`` for "no polarization".

        ``None`` (the default) means "unspecified".  On the pipelines that publish a
        coupling-channel polarization it resolves to the historical default ``1``; on
        ``separable_td`` it means the caller did not ask for one, and the solve proceeds
        without it.

        An **explicitly given** channel is always type- and range-checked first, so the
        strict contract holds everywhere: a bool, a float, a string, ``0``, a negative
        index or an out-of-range value raises ``ValueError`` on every model.  Only a legal
        channel on ``separable_td`` then raises ``NotImplementedError`` -- the Dicke model
        does have a coupling operator (``d_phys = 3``); what this pipeline lacks is a
        validated time-resolved Eq.-F2/F3 extraction for a *rotating* coupling operator.
        """
        n_ch = len(self.model.coupling_operators())
        if self.model.bath_type == "separable_td":
            if channel is None:
                return None
            validate_channel(channel, n_ch)     # type/range first, then the capability limit
            raise NotImplementedError(
                f"bath_type='separable_td' has {n_ch} coupling operator(s), but this "
                f"pipeline publishes no time-resolved coupling-channel history: "
                f"the Eq.-F2/F3 arm selector's time mapping is established only for a time-INDEPENDENT coupling operator, and it has not been defined or validated for a rotating "
                f"S(t). Call without `channel` and read result.final_density_matrix, or "
                f"result.sub_bath_final_density_matrices with record_rho=True")
        return validate_channel(1 if channel is None else channel, n_ch)

    def _resolve_moments(self, moments):
        """Normalise the requested ``moments``, or return ``None`` for "compute nothing".

        Nothing extra is computed unless the caller asks for it by name, and only what
        was asked for is computed -- the collective-spin moments each cost a whole extra
        folded chain, so a default-on or "compute the neighbours too" policy would charge
        for work nobody requested.

        The rules, all applied here rather than deep in a pipeline:

        * ``None`` means no request; so does an empty sequence, which is a caller
          building a list programmatically and finding nothing to add;
        * the argument must be an **ordered** ``Sequence`` -- a list or tuple.  A ``set``
          or a generator is refused: "duplicates are dropped preserving first appearance"
          is meaningless for an unordered container, and a generator would be consumed by
          the first thing that looked at it;
        * a **bare string** (or ``bytes``) is refused rather than iterated -- both *are*
          sequences, so ``moments='Jz'`` would otherwise silently become the characters
          ``'J'`` and ``'z'``;
        * every item must be a strict ``str`` from :data:`MOMENT_NAMES`; an unknown name
          is a ``ValueError`` **on every model**, before any capability question;
        * duplicates are dropped, preserving first appearance.

        Capability is then decided by a **Dicke-specific provider on the model**, not by
        ``bath_type == 'separable_td'``: the bath type is a pipeline class and does not
        guarantee a Fock-truncated cavity or Pauli spins, which is what makes both the
        photon-number moments and ``J = sigma/2`` meaningful.  A model that cannot supply
        ``collective_spin_closures(t)`` raises ``NotImplementedError``.
        """
        if moments is None:
            return None
        if isinstance(moments, (str, bytes)):
            raise ValueError(
                f"moments must be a sequence of names, not the bare string {moments!r}: a "
                f"string iterates as single characters.  Pass ({moments!r},) instead")
        if not isinstance(moments, Sequence):
            raise ValueError(
                f"moments must be an ordered sequence (list or tuple) of names from "
                f"{MOMENT_NAMES}, or None, got {type(moments).__name__}")
        names: list[str] = []
        for item in moments:
            if type(item) is not str:
                raise ValueError(
                    f"every entry of moments must be a string from {MOMENT_NAMES}, got "
                    f"{item!r}")
            if item not in MOMENT_NAMES:
                raise ValueError(
                    f"unknown moment {item!r}; choose from {MOMENT_NAMES}")
            if item not in names:
                names.append(item)
        if not names:
            return None                     # an empty request is not an error, just nothing
        if not callable(getattr(self.model, "collective_spin_closures", None)):
            raise NotImplementedError(
                f"moments={tuple(names)} needs a model exposing collective_spin_closures(t) "
                f"-- a Fock-truncated cavity coupled to Pauli spins (DickeModel); "
                f"{type(self.model).__name__} does not provide one")
        return tuple(names)

    def solve(self, observables: dict | None = None, *,
              channel: int | None = None,
              moments: Sequence[str] | None = None) -> SolverResult:
        """Evolve and extract observables.

        Parameters
        ----------
        observables : dict[str, callable], optional
            Mapping ``name -> operator_fn(t)`` evaluated as ``Tr[O(t) rho(t)]``
            from the recorded reduced states (requires ``config.record_rho``).
        channel : int, optional
            Coupling channel (1-based) whose polarization history is returned.  ``None``
            (the default) selects channel ``1`` on the pipelines that provide a
            polarization history, and means "no polarization requested" on
            ``separable_td``, which provides none -- see :meth:`_resolve_channel`.
        moments : sequence of str, optional
            Basic final-time observables to extract, from :data:`MOMENT_NAMES`; ``None``
            (the default) computes none.  ``'n'`` and ``'n_factorial2'`` are
            post-processing of the reduced cavity state and essentially free; each
            collective-spin channel costs an additional folded chain.  Lands in
            :attr:`SolverResult.moments` -- see :meth:`_resolve_moments` for the naming
            rules and :attr:`SolverResult.moments` for what comes back.
        """
        # validate once, before any backend/bath dispatch or evolution -- every inner path
        # then receives a normalised Python int (no negative-index channel selection),
        # or None where the pipeline publishes no polarization
        moments = self._resolve_moments(moments)
        channel = self._resolve_channel(channel)
        if moments is not None:
            if self.config.backend == "hpc":
                raise NotImplementedError(
                    "moments are not available on backend='hpc': the 2D contraction is "
                    "Gaudin/`separable`-only and builds no sub-bath fold to insert a "
                    "bath-side measurement into")
            if self.config.record_time_reads and _TANGENT_MOMENTS.intersection(moments):
                raise ValueError(
                    f"moments={moments} with record_time_reads=True is not supported: the "
                    f"collective-spin moments fold an extra chain per channel, while time "
                    f"reads force compress_method='dm_tracking' to transport the "
                    f"causal-prefix terminators -- which those chains do not carry.  "
                    f"Request them in two separate solves; 'n' and 'n_factorial2' do "
                    f"coexist with record_time_reads")
        if self.config.backend == "hpc":
            return self._solve_hpc(observables, channel=channel)
        if self.model.bath_type == "separable_td":
            return self._solve_separable_td(observables, moments=moments)
        if moments is not None:
            raise NotImplementedError(
                f"moments={moments} are implemented on the separable_td (Dicke) pipeline "
                f"only, but this model's bath_type is {self.model.bath_type!r}")
        if self.model.bath_type == "separable":
            return self._solve_separable(observables, channel=channel)

        cfg = self.config
        convert, _memory, backend_label = self._resolve_backend()
        # the efficient Eq.-F2 sweep is first-order specific; second order reads
        # the coupling polarization from the recorded reduced states instead.
        second_order = cfg.expansion_order == 2
        # record_time_reads is a generic "give me rho(t)" request; this pipeline already
        # has a per-step recorder, so it needs no prefix machinery -- just turn it on.
        need_rho = (cfg.record_rho or bool(observables) or second_order
                    or cfg.record_time_reads)
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
            final_density_matrix=_final_reduced_state(ev.mps, ev.density_matrices),
            compression_method_used=ev.compression_method_used,
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
            final_density_matrix=out["final_rho"],   # already produced by the 2D contraction
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
        ``result.density_matrices`` is ``None`` unless ``record_time_reads=True``, which fills the
        time axis from this same fold (:mod:`edmtn.evolution.prefix_reads`).
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
            record_time_reads=cfg.record_time_reads,
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
        # ONE reduced-state contraction, shared by the final polarization point and the
        # public final_density_matrix field (previously each would have contracted the MPS
        # separately).  Reuses the recorded rho_L(T) when record_rho already produced it.
        final_rho = _final_reduced_state(ev.mps, ev.density_matrices)
        _, vals = ObservableExtractor.expectation_history(
            [final_rho], [N * cfg.eps], lambda t: Sop)
        p_T = float(vals[0].real)
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
            # the TIME axis (None unless record_time_reads); the per-L states below are
            # rho_L(T), a different axis, and stay in sub_bath_final_density_matrices
            density_matrices=ev.time_density_matrices,
            sub_bath_counts=ev.recorded_L,
            sub_bath_bond_dims=ev.bond_dims,
            sub_bath_final_density_matrices=ev.density_matrices,   # rho_L(T) if record_rho, else None
            final_time_bond_dims=ev.mps.bond_dims,
            sub_baths_used=ev.n_sub_baths,                         # actual L folded (== validate_sub_baths)
            final_density_matrix=final_rho,                        # rho_L(T) for L = sub_baths_used
            compression_method_used=ev.compression_method_used,
        )

    # -- time-dependent separable bath (Dicke) ----------------------------

    def _solve_separable_td(self, observables: dict | None, *, moments=None) -> SolverResult:
        """Solve a time-dependent separable-bath model (Dicke): same Eq.-21 outer loop as
        :meth:`_solve_separable`, but **without** a coupling-channel polarization history.

The Eq.-F2/F3 sweep selects a coupling-operator arm at one time site and closes
        the rest.  Its time mapping is established only for a **time-independent**
        coupling operator: the arm at a site carries ``S`` at that site's sample time,
        while the environment to its right is the state *before* that step, so with a
        rotating ``S(t)`` the operator and the state sit at different times.  Defining and
        validating that alignment for a time-dependent coupling is separate work, not done
        here -- so no history is published rather than one whose time axis is not
        established.  ``polarization`` is therefore ``None``.  ``density_matrices`` is ``None``
        **by default** and is filled by ``record_time_reads=True``, which reads the time
        axis off this same fold (:mod:`edmtn.evolution.prefix_reads`); the per-``L`` states
        are a different axis (``rho_L(T)``, indexed by sub-bath count) and stay in
        ``sub_bath_final_density_matrices``.
        Everything else is the standard separable contract, plus
        ``final_density_matrix`` -- so a default solve still returns a physical state.

        ``moments`` (already resolved by :meth:`_resolve_moments`) selects the basic
        observables of ``docs/design/dicke-observable-extraction.md``.  The cavity moments
        are read off the final reduced state; each requested collective-spin channel adds
        one tangent chain to the fold, closed on the bath side at the final grid time.
        """
        if observables:
            raise NotImplementedError(
                "custom per-time observables are not supported for time-dependent "
                "separable baths; run with record_rho=True and read "
                "result.sub_bath_final_density_matrices (the per-L rho_L(T)), or read "
                "result.final_density_matrix")
        cfg = self.config
        convert, memory, backend_label = self._resolve_backend()
        tangent_closings = None
        if moments is not None and _TANGENT_MOMENTS.intersection(moments):
            # the grid's own final time, n_steps*eps, not cfg.T -- the same value the
            # evolution samples, so the picture phases cannot sit a round-off apart
            closures = self.model.collective_spin_closures(cfg.n_steps * cfg.eps)
            wanted = []
            if _JPLUS_MOMENTS.intersection(moments):
                wanted.append("Jplus")
            if _JZ_MOMENTS.intersection(moments):
                wanted.append("Jz")
            # the provider is duck-typed, so what it returns is checked here rather than
            # left to leak a bare KeyError from the indexing below; the shapes and
            # finiteness of the vectors stay with validate_tangent_closings, which the
            # evolution applies to every entry
            if not isinstance(closures, Mapping):
                raise ValueError(
                    f"{type(self.model).__name__}.collective_spin_closures(t) must return a "
                    f"mapping of channel name -> closing array, got "
                    f"{type(closures).__name__}")
            missing = [ch for ch in wanted if ch not in closures]
            if missing:
                raise ValueError(
                    f"{type(self.model).__name__}.collective_spin_closures(t) is missing the "
                    f"channel(s) {missing} needed by moments={moments}; it returned "
                    f"{sorted(closures)}")
            tangent_closings = {ch: closures[ch] for ch in wanted}
        ev = self.evolution.run(
            self.model,
            self.kernel_engine,
            cfg.eps,
            cfg.n_steps,
            max_bond=cfg.max_bond,
            cutoff=cfg.cutoff,
            cutoff_mode=cfg.cutoff_mode,
            record_rho=cfg.record_rho,
            record_time_reads=cfg.record_time_reads,
            sub_baths=cfg.sub_baths,
            convert=convert,
            memory=memory,
            tangent_closings=tangent_closings,
        )
        final_rho = _final_reduced_state(ev.mps, ev.density_matrices)
        moment_values = (None if moments is None
                         else _extract_moments(moments, final_rho, ev.tangent_density_matrices))
        return SolverResult(
            times=cfg.eps * np.arange(1, cfg.n_steps + 1, dtype=np.float64),
            polarization=None,                                     # no channel history here
            bond_dims=ev.bond_dims,                                # legacy alias of sub_bath_bond_dims
            truncation_errors=ev.truncation_errors,
            expansion_order=cfg.expansion_order,
            observables={},
            mps=ev.mps,
            evolution=ev,
            backend=backend_label,
            # the TIME axis (None unless record_time_reads); the per-L states below are
            # rho_L(T), a different axis, and stay in sub_bath_final_density_matrices
            density_matrices=ev.time_density_matrices,
            sub_bath_counts=ev.recorded_L,
            sub_bath_bond_dims=ev.bond_dims,
            sub_bath_final_density_matrices=ev.density_matrices,   # rho_L(T) if record_rho, else None
            final_time_bond_dims=ev.mps.bond_dims,
            sub_baths_used=ev.n_sub_baths,
            final_density_matrix=final_rho,
            compression_method_used=ev.compression_method_used,
            moments=moment_values,
            moment_truncation_errors=ev.tangent_truncation_errors,
        )

    # -- convergence helpers ----------------------------------------------

    def timestep_convergence(self, *, tol: float | None = None,
                             channel: int | None = None) -> TimestepConvergence:
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

        Not available on ``separable_td`` (Dicke): the comparison is defined on the
        coupling-channel polarization, which that pipeline does not publish.  It raises
        ``NotImplementedError`` up front rather than letting a ``None`` polarization leak
        into a deep ``TypeError``; compare ``final_density_matrix`` from two solves at
        ``eps`` and ``eps/2`` instead.
        """
        if self.model.bath_type == "separable_td":
            # keep the strict channel contract even though the capability is missing:
            # an illegal channel is still a ValueError, only a LEGAL one reaches the gate
            if channel is not None:
                validate_channel(channel, len(self.model.coupling_operators()))
            raise NotImplementedError(
                "timestep_convergence compares the coupling-channel polarization, which "
                "bath_type='separable_td' does not publish; instead solve twice (eps and "
                "eps/2) and compare result.final_density_matrix, e.g. "
                "abs(a.final_density_matrix - b.final_density_matrix).max()")
        channel = self._resolve_channel(channel)
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


def _extract_moments(names, rho, tangents) -> dict:
    """Pack the requested final-time moments (plus by-products and ``trace``).

    ``rho`` is the reduced **cavity** state at ``T`` and ``tangents`` the per-channel
    ``d x d`` matrices ``tilde_rho^(alpha) = Tr_B[(1 (x) sigma_alpha) rho_CB(T)]`` (or
    ``None`` when no spin moment was requested).

    Every reduction runs on the array's own backend and exactly one scalar per quantity
    crosses back with ``.item()``, so a GPU run neither transfers the state to the host nor
    trips CuPy's ban on implicit conversion.

    The cavity moments use the **diagonal only**, which is exact here rather than merely
    convenient: ``a^dag a`` and the normal-ordered ``a^dag a^dag a a = n(n-1)`` are both
    diagonal in the Fock basis and boundary-safe under the truncation (the two
    annihilations act first and never reach the top level).  That is a property of these
    two moments, not a general licence -- in a space of dimension ``d`` the commutator is
    ``[a, a^dag] = 1 - d P_top``, so any *rearranged* expression must be built as an
    explicit matrix in the truncated space instead.  Both are also picture independent,
    commuting with ``H_0``, so no rotation is applied to them.
    """
    xp = _xp(rho)
    diag = rho.diagonal()                        # backend-native, complex
    d = int(diag.shape[0])
    out: dict = {}
    if "n" in names or "n_factorial2" in names:
        levels = xp.arange(d)
        if "n" in names:
            out["n"] = real_scalar_expectation("n", (levels * diag).sum().item())
        if "n_factorial2" in names:
            out["n_factorial2"] = real_scalar_expectation(
                "n_factorial2", (levels * (levels - 1) * diag).sum().item())
    if _JPLUS_MOMENTS.intersection(names):
        # <J_+> is complex by construction: its real and imaginary parts are two different
        # observables, so it takes no imaginary-part guard -- but both parts must still be
        # finite.  Both come from one chain, so each is the other's by-product, and both
        # are kept.
        j_plus = finite_complex_expectation("Jplus", tangents["Jplus"].trace().item())
        out["Jx"] = float(j_plus.real)
        out["Jy"] = float(j_plus.imag)
    if _JZ_MOMENTS.intersection(names):
        out["Jz"] = real_scalar_expectation("Jz", tangents["Jz"].trace().item())
    if "Jabs" in names:
        # hypot, not sqrt of a sum of squares: three individually finite components can
        # still overflow when squared and added, which would return `inf` from a check
        # that already passed on every input.  The result is re-checked, because the
        # finiteness contract covers what is RETURNED, not only what was read.
        out["Jabs"] = real_scalar_expectation(
            "Jabs", math.hypot(out["Jx"], out["Jy"], out["Jz"]))
    # raw and un-normalised: a trace deviation is evidence about the run and must reach
    # the caller rather than being divided out of the moments above.  Un-normalised is not
    # the same as unchecked -- a non-finite trace is a broken run, not evidence.
    out["trace"] = finite_complex_expectation("trace", diag.sum().item())
    return out


def _final_reduced_state(mps, recorded):
    """The final reduced density matrix, reusing a recorded one when the pipeline has it.

    ``recorded`` is the pipeline's list of reduced states (``rho(t)`` for single-bath,
    ``rho_L(T)`` for separable) or ``None``.  Its last entry is already the final state, so
    taking it avoids a second closing contraction of the whole MPS; only when nothing was
    recorded is one contraction performed.  Returns the array in its native backend type.
    """
    if recorded:
        return recorded[-1]
    return mps.reduced_density_matrix()


def solve(
    model, *, T: float, eps: float, observables: dict | None = None,
    channel: int | None = None, moments: Sequence[str] | None = None, **kwargs
) -> SolverResult:
    """Convenience one-shot solve.

    ``channel=None`` (the default) selects coupling channel ``1`` on the pipelines that
    publish a polarization history, and means "none requested" on ``separable_td``.

    ``moments`` is declared **explicitly** rather than left to ``**kwargs``: everything
    unnamed here is forwarded to :class:`~edmtn.driver.auto_config.SolverConfig`, where an
    unknown field is a ``TypeError`` -- so a solve-time argument that only the solver
    understands has to be named, or it would never reach :meth:`EDMSolver.solve`.
    """
    return EDMSolver.from_model(model, T=T, eps=eps, **kwargs).solve(
        observables, channel=channel, moments=moments)
