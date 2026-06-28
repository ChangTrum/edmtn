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
    backend: str = ""
    density_matrices: object = None  # ρ(t) history (the hpc track returns this first-class)
    error_metrics: dict | None = None  # hpc only: ‖ρ−ρ†‖ / |Tr ρ−1| (+ slices/flops or discarded weight)

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
        # the hpc (cuQuantum 2D) track builds nothing from the Track-1 pipeline
        if config.backend == "hpc":
            self.kernel_engine = self.evolution = None
        else:
            self.kernel_engine, self.evolution = build_pipeline(model, config)

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
        in ``docs/cpu-vs-gpu-edm.md``).  The GPU stays a first-class, validated
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
            Coupling channel whose polarization history is returned.
        """
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
        _, pol = ObservableExtractor.coupling_polarization_history(
            ev.mps, cfg.eps, channel=channel, order=cfg.expansion_order
        )
        # The separable EDM's open-arm interventions sit one step earlier than the
        # single-bath grid: the sweep yields <S_a(t)> at t = 0, eps, ..., (N-1) eps
        # (matching Fig. 6, which starts at t = 0 with <S_z> = 1/2).  The final-time
        # state is available as result.evolution.density_matrices[-1].
        times = cfg.eps * np.arange(len(pol))
        return SolverResult(
            times=times,
            polarization=pol,
            bond_dims=ev.bond_dims,
            truncation_errors=ev.truncation_errors,
            observables={},
            mps=ev.mps,
            evolution=ev,
            backend=backend_label,
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
            expansion_order=self.config.expansion_order,
            compress_method=self.config.compress_method,
            compress_decomp=self.config.compress_decomp,
            compress_decomp_q=self.config.compress_decomp_q,
            compress_canon=self.config.compress_canon,
        )
        fine = EDMSolver(self.model, fine_cfg).solve(channel=channel)
        dev = max_history_deviation(
            coarse.times, coarse.polarization, fine.times, fine.polarization
        )
        ok = None if tol is None else (dev <= tol)
        return dev, ok


def solve(
    model, *, T: float, eps: float, observables: dict | None = None, channel: int = 1, **kwargs
) -> SolverResult:
    """Convenience one-shot solve."""
    return EDMSolver.from_model(model, T=T, eps=eps, **kwargs).solve(observables, channel=channel)
