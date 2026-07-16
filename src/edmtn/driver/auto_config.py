"""Pipeline auto-configuration (Layer 7).

Selects and constructs the engine stack (cumulant -> kernel -> expansion ->
decomposition -> evolution) from a model's ``bath_type``.  Pipelines are kept in
a small registry so future bath types (``separable``, ``chain``) slot in without
touching the driver.

Phase 1 ships the ``gaussian`` pipeline only.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass, field

from ..expansion.first_order import FirstOrderExpander
from ..expansion.second_order import SecondOrderExpander
from ..kernels.gaussian_mpo import GaussianKernelEngine
from ..kernels.separable_mpo import SeparableKernelEngine
from ..evolution.separable_bath import SeparableBathEvolution
from ..evolution.single_bath import SingleBathEvolution

# -- config validation -----------------------------------------------------
# Every SolverConfig knob is validated at construction (in __post_init__) so a
# bad value fails loudly and immediately, at the config entry point, rather than
# surfacing as a divide-by-zero / int-cast / silent-round deep inside a pipeline
# -- and so BOTH tracks (cpu/gpu Track 1 and the hpc Track 2) reject an illegal
# time grid with the SAME error instead of Track 2 silently rounding T/eps down.

_NSTEPS_RTOL = 1e-9  # T/eps must equal an integer within this relative tolerance

# Public allowed enum values -- EDMTN's supported/tested config contract, read
# from the code that consumes each knob (a backend/library may accept more; these
# are the values EDMTN exposes and tests):
_BACKENDS = ("cpu", "numpy", "gpu", "cupy", "hpc")   # solver._resolve_backend + hpc path
_PRECISIONS = ("f64", "mixed")                        # solver._resolve_backend
_CUTOFF_MODES = ("abs", "rel", "sum2", "rsum2", "sum1", "rsum1")  # quimb-native string modes
_COMPRESS_METHODS = ("zipup", "dm", "direct")         # EDMTN's tested subset of quimb 1D-compress
_COMPRESS_DECOMPS = ("exact", "rsvd")                 # quimb_decomp.compress_opts_for
_COMPRESS_CANONS = ("quimb", "householder", "cholqr")  # quimb_decomp._CANON_METHOD
_PATHFINDERS = ("cuquantum", "cotengra")              # cutensornet path-finder select


def _is_int(value) -> bool:
    """True for a genuine integer (Python ``int`` or NumPy integer), excluding ``bool``."""
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def _is_real(value) -> bool:
    """True for a real number usable as a float, excluding ``bool``."""
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _positive_finite_float(name: str, value) -> float:
    if not _is_real(value):
        raise ValueError(f"{name} must be a real number, got {value!r}")
    v = float(value)
    if not math.isfinite(v) or v <= 0.0:
        raise ValueError(f"{name} must be finite and > 0, got {value!r}")
    return v


def _nonnegative_finite_float(name: str, value) -> float:
    if not _is_real(value):
        raise ValueError(f"{name} must be a real number, got {value!r}")
    v = float(value)
    if not math.isfinite(v) or v < 0.0:
        raise ValueError(f"{name} must be finite and >= 0, got {value!r}")
    return v


def _nonnegative_int(name: str, value) -> int:
    if not _is_int(value):
        raise ValueError(f"{name} must be an integer (not bool), got {value!r}")
    v = int(value)
    if v < 0:
        raise ValueError(f"{name} must be >= 0, got {value!r}")
    return v


def _optional_positive_int(name: str, value):
    if value is None:
        return None
    if not _is_int(value):
        raise ValueError(f"{name} must be a positive integer or None (not bool), got {value!r}")
    v = int(value)
    if v < 1:
        raise ValueError(f"{name} must be >= 1 or None, got {value!r}")
    return v


def _boolean(name: str, value) -> bool:
    if not isinstance(value, bool):  # strict: reject 1 / "yes" / np.bool_
        raise ValueError(f"{name} must be a bool, got {value!r}")
    return value


def _choice(name: str, value, allowed):
    if value not in allowed:
        raise ValueError(f"{name} must be one of {tuple(allowed)}, got {value!r}")
    return value


def _validated_n_steps(T: float, eps: float) -> int:
    """Number of steps ``T / eps``, requiring it to be a positive integer within
    ``_NSTEPS_RTOL`` so the grid lands exactly on ``T`` (no silent round-down)."""
    ratio = T / eps
    if not math.isfinite(ratio):  # e.g. huge/tiny inputs overflow or underflow the quotient
        raise ValueError(f"T/eps must be finite; got T={T!r}, eps={eps!r} -> T/eps={ratio!r}")
    nearest = round(ratio)
    if nearest < 1 or abs(ratio - nearest) > _NSTEPS_RTOL * max(1.0, abs(ratio)):
        raise ValueError(
            f"T/eps must be a positive integer within relative tol {_NSTEPS_RTOL:g} "
            f"(so the time grid lands exactly on T); got T={T!r}, eps={eps!r} -> "
            f"T/eps={ratio!r}")
    return int(nearest)


@dataclass(frozen=True)
class SolverConfig:
    """Configuration for :class:`~edmtn.driver.solver.EDMSolver`.

    Attributes
    ----------
    eps : float
        Time step.
    T : float
        Total evolution time; ``T / eps`` must be a positive integer (validated at
        construction) so the grid lands exactly on ``T``.  ``n_steps`` caches it.
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
    compress_decomp: str = "exact"        # cpu/gpu: 'exact'|'rsvd' (N/A under 'hpc': Track 2 is exact-only, no truncation)
    compress_decomp_q: int = 2            # rsvd power iterations (2=cold, 0=single-pass; N/A under 'hpc')
    compress_canon: str = "quimb"         # 'quimb'|'householder'|'cholqr' (canon QR; N/A under 'hpc')
    preset: str | None = None  # None|'balanced'|'robust' (cpu/gpu only; see docs/guides/recommended-config.md)
    sub_baths: int | None = None  # separable bath only: fold/contract just the first L sub-baths (Fig. 6)
    backend: str = "cpu"   # 'cpu'|'gpu' -> numpy/cupy (Track 1); 'hpc' -> cuQuantum 2D contraction
    precision: str = "f64"  # 'f64' | 'mixed' (mixed: f32 contraction, f64 decompose -- Phase 3/4)
    # -- backend='hpc' only; ignored otherwise --
    pathfinder: str = "cuquantum"   # 'cuquantum' (default, cuTensorNet owns path) | 'cotengra'
    time_windows: int | None = None  # None = one-shot whole-spacetime; int = manual window blocking
    # NB: hpc has no GPU-count knob -- it uses every GPU it is launched across (cuTensorNet
    # is one rank per GPU). Launch one rank per GPU with your own workflow, e.g.
    # `srun --mpi=pmi2 --ntasks=<#GPUs> --gres=gpu:<#GPUs>` (see cluster/); edmtn
    # itself does not submit/srun/ssh -- that is the user's job.

    # derived + cached at construction (frozen: set via object.__setattr__ below)
    _n_steps: int = field(init=False, repr=False, compare=False, default=0)

    def __post_init__(self):
        # frozen dataclass: bypass immutability to store validated / normalised values
        def _set(name, value):
            object.__setattr__(self, name, value)

        # -- time grid (validated identically for both tracks) --
        _set("eps", _positive_finite_float("eps", self.eps))
        _set("T", _positive_finite_float("T", self.T))
        _set("_n_steps", _validated_n_steps(self.T, self.eps))

        # -- truncation / compression knobs --
        _set("cutoff", _nonnegative_finite_float("cutoff", self.cutoff))
        _set("cutoff_mode", _choice("cutoff_mode", self.cutoff_mode, _CUTOFF_MODES))
        _set("max_bond", _optional_positive_int("max_bond", self.max_bond))
        if not _is_int(self.expansion_order) or int(self.expansion_order) not in (1, 2):
            raise ValueError(
                f"expansion_order must be the integer 1 or 2, got {self.expansion_order!r}")
        _set("expansion_order", int(self.expansion_order))
        _set("record_rho", _boolean("record_rho", self.record_rho))
        _set("compress_method", _choice("compress_method", self.compress_method, _COMPRESS_METHODS))
        _set("compress_decomp", _choice("compress_decomp", self.compress_decomp, _COMPRESS_DECOMPS))
        _set("compress_decomp_q", _nonnegative_int("compress_decomp_q", self.compress_decomp_q))
        _set("compress_canon", _choice("compress_canon", self.compress_canon, _COMPRESS_CANONS))
        _set("sub_baths", _optional_positive_int("sub_baths", self.sub_baths))

        # -- backend / precision / hpc-only knobs --
        _set("backend", _choice("backend", self.backend, _BACKENDS))
        _set("precision", _choice("precision", self.precision, _PRECISIONS))
        _set("pathfinder", _choice("pathfinder", self.pathfinder, _PATHFINDERS))
        if self.time_windows is not None:  # concept is valid but not implemented -> NotImplementedError
            raise NotImplementedError(
                "manual time-window blocking (time_windows) is wired but not yet "
                "implemented; the hpc track ships one-shot whole-spacetime only. "
                "Use time_windows=None.")

        # -- presets: validate legality for ALL backends; APPLY only on Track 1 --
        # (Track-1 rSVD recipes don't apply to the hpc track, but an unknown preset
        #  name is still rejected there rather than silently ignored.)
        if self.preset is not None and self.preset not in _PRESETS:
            raise ValueError(
                f"unknown preset {self.preset!r}; choose from {sorted(_PRESETS)} or None")
        if self.preset is not None and self.backend != "hpc":
            spec = _PRESETS[self.preset]
            if self.compress_decomp == "exact":  # only fill if left at the default
                _set("compress_decomp", spec["compress_decomp"])
                _set("compress_decomp_q", spec["compress_decomp_q"])

    @property
    def n_steps(self) -> int:
        return self._n_steps


# Recommended presets (docs/guides/recommended-config.md): both use quimb rSVD, differing
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
