"""Pipeline auto-configuration (Layer 7).

Selects and constructs the engine stack (cumulant -> kernel -> expansion ->
decomposition -> evolution) from a model's ``bath_type``.  Pipelines are kept in
a small registry so future bath types (``separable``, ``chain``) slot in without
touching the driver.

Phase 1 ships the ``gaussian`` pipeline only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..expansion.first_order import FirstOrderExpander
from ..expansion.second_order import SecondOrderExpander
from ..kernels.gaussian_mpo import GaussianKernelEngine
from ..kernels.separable_mpo import SeparableKernelEngine
from ..evolution.separable_bath import SeparableBathEvolution
from ..evolution.single_bath import SingleBathEvolution


@dataclass
class SolverConfig:
    """Configuration for :class:`~edmtn.driver.solver.EDMSolver`.

    Attributes
    ----------
    eps : float
        Time step.
    T : float
        Total evolution time; ``n_steps = round(T / eps)``.
    cutoff : float
        SVD truncation precision (``0`` keeps every singular value -- exact but
        not scalable).
    cutoff_mode : str
        Truncation rule (default ``'rel_ref'``, the paper's ``s_i / s_{d^2+1}``).
    max_bond : int, optional
        Hard bond-dimension cap.
    ref_index : int, optional
        Reference index for ``'rel_ref'`` (defaults to ``d**2``).
    expansion_order : int
        Trotter order.  Phase 1 supports ``1``; ``2`` is accepted here but the
        single-bath engine currently rejects it (doubled sub-step grid pending).
    record_rho : bool
        Store ``rho(t)`` at every step (needed for custom observables).
    cutoff, cutoff_mode : float, str
        quimb truncation controls (default ``rel`` -- the built-in closest to the
        retired ``rel_ref``).
    compress_method, compress_decomp, compress_decomp_q, compress_canon :
        quimb compression controls (see :mod:`edmtn.evolution.quimb_decomp`).
    """

    eps: float
    T: float
    cutoff: float = 1e-8
    cutoff_mode: str = "rel"       # quimb-native cutoff (rel: faithful to the retired rel_ref)
    max_bond: int | None = None
    expansion_order: int = 1
    record_rho: bool = False
    compress_method: str = "zipup"        # 'zipup'|'dm'|'direct' (quimb 1D-compress; N/A under backend='hpc')
    compress_decomp: str = "exact"        # cpu/gpu: 'exact'|'rsvd'.  hpc: 'exact'(no knobs)|'approx'
    compress_decomp_q: int = 2            # rsvd power iterations (2=cold, 0=single-pass; N/A under 'hpc')
    compress_canon: str = "quimb"         # 'quimb'|'householder'|'cholqr' (canon QR; N/A under 'hpc')
    preset: str | None = None  # None|'balanced'|'robust' (cpu/gpu only; see docs/recommended-config.md)
    sub_baths: int | None = None  # separable bath only: fold/contract just the first L sub-baths (Fig. 6)
    backend: str = "cpu"   # 'cpu'|'gpu' -> numpy/cupy (Track 1); 'hpc' -> cuQuantum 2D contraction
    precision: str = "f64"  # 'f64' | 'mixed' (mixed: f32 contraction, f64 decompose -- Phase 3/4)
    # -- backend='hpc' only; ignored otherwise --
    pathfinder: str = "cuquantum"   # 'cuquantum' (default, cuTensorNet owns path) | 'cotengra'
    time_windows: int | None = None  # None = one-shot whole-spacetime; int = manual window blocking

    def __post_init__(self):
        # Presets are Track-1 rСVD recipes; they don't apply to the hpc track
        # (where compress_decomp is exact/approx) -- never silently flip its mode.
        if self.preset is None or self.backend == "hpc":
            return
        if self.preset not in _PRESETS:
            raise ValueError(
                f"unknown preset {self.preset!r}; choose from {sorted(_PRESETS)} or None"
            )
        spec = _PRESETS[self.preset]
        if self.compress_decomp == "exact":  # only fill if left at the default
            self.compress_decomp = spec["compress_decomp"]
            self.compress_decomp_q = spec["compress_decomp_q"]

    @property
    def n_steps(self) -> int:
        return int(round(self.T / self.eps))


# Recommended presets (docs/recommended-config.md): both use quimb rSVD, differing
# only in power iterations -- balanced = single-pass (q=0, fastest), robust = cold
# (q=2, exact-baseline accuracy).  The silent guard falls back to full SVD either way.
_PRESETS: dict = {
    "balanced": {"compress_decomp": "rsvd", "compress_decomp_q": 0},
    "robust": {"compress_decomp": "rsvd", "compress_decomp_q": 2},
}


def _make_expander(order: int):
    if order == 1:
        return FirstOrderExpander()
    if order == 2:
        return SecondOrderExpander()
    raise ValueError(f"unsupported expansion_order {order!r}")


# -- pipeline registry -----------------------------------------------------

_PIPELINES: dict = {}


def register_pipeline(bath_type: str, builder) -> None:
    """Register a ``builder(model, config) -> (kernel_engine, evolution)``."""
    _PIPELINES[bath_type] = builder


def available_pipelines() -> tuple:
    return tuple(sorted(_PIPELINES))


def build_pipeline(model, config: SolverConfig):
    """Construct ``(kernel_engine, evolution_engine)`` for ``model``."""
    bt = model.bath_type
    if bt not in _PIPELINES:
        raise NotImplementedError(
            f"no EDM pipeline registered for bath_type={bt!r}; "
            f"available: {available_pipelines()}"
        )
    return _PIPELINES[bt](model, config)


def _build_gaussian(model, config: SolverConfig):
    kernel_engine = GaussianKernelEngine.from_model(
        model, T=config.T, eps=config.eps, order=config.expansion_order
    )
    evolution = SingleBathEvolution(
        expander=_make_expander(config.expansion_order),
        compress_method=config.compress_method,
        compress_decomp=config.compress_decomp,
        compress_decomp_q=config.compress_decomp_q,
        compress_canon=config.compress_canon,
    )
    return kernel_engine, evolution


def _build_separable(model, config: SolverConfig):
    kernel_engine = SeparableKernelEngine.from_model(model, T=config.T, eps=config.eps)
    evolution = SeparableBathEvolution(
        expander=_make_expander(config.expansion_order),
        compress_method=config.compress_method,
        compress_decomp=config.compress_decomp,
        compress_decomp_q=config.compress_decomp_q,
        compress_canon=config.compress_canon,
    )
    return kernel_engine, evolution


register_pipeline("gaussian", _build_gaussian)
register_pipeline("separable", _build_separable)
