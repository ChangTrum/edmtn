"""Mixed-precision policy for Layer 0.

The EDM pipeline has three numerically distinct stages, each with its own
precision sweet spot:

* **build** — cumulant / kernel-tensor construction.  Precision-sensitive; the
  combined-kernel MPO is assembled from products of correlation values, so this
  stays ``f64`` (``complex128``).
* **contract** — the MPS / MPO contraction intermediates.  On the GPU these are
  the throughput-dominant tensors; running them in ``f32`` (``complex64``) can
  roughly double Tensor-Core throughput at large bond dimension, which is the
  whole point of moving Gaudin onto the GPU.
* **decompose** — the SVD whose singular values drive bond-dimension
  truncation.  This is *always* ``f64``: in ``f32`` the small singular values
  that decide the kept rank are swamped by round-off and the bond dimension is
  chosen wrongly (technical plan §12.1).

:class:`PrecisionPolicy` makes that choice explicit and a single source of
truth, replacing the ad-hoc ``convert=lambda a: cupy.asarray(a, cupy.complex64)``
lambdas that the Phase-1 evolution engine threaded through by hand.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_STAGES = ("build", "contract", "decompose")

# stage-precision label -> (complex dtype, real dtype)
_DTYPES = {
    "f64": (np.complex128, np.float64),
    "f32": (np.complex64, np.float32),
}


@dataclass(frozen=True)
class PrecisionPolicy:
    """Per-stage floating-point precision for the EDM pipeline.

    Parameters
    ----------
    build, contract, decompose : {'f64', 'f32'}
        Precision label for each stage.  ``decompose`` must be ``'f64'`` — the
        truncation rank selection is unreliable in single precision.
    """

    build: str = "f64"
    contract: str = "f64"
    decompose: str = "f64"

    def __post_init__(self):
        for stage in _STAGES:
            label = getattr(self, stage)
            if label not in _DTYPES:
                raise ValueError(
                    f"{stage} precision must be one of {sorted(_DTYPES)}, got {label!r}"
                )
        if self.decompose != "f64":
            raise ValueError(
                "decompose precision must be 'f64': single-precision SVD selects "
                "the truncation rank incorrectly (technical plan §12.1)"
            )

    # -- presets -----------------------------------------------------------

    @classmethod
    def full_f64(cls) -> "PrecisionPolicy":
        """All stages in double precision (the safe, Phase-1 default)."""
        return cls(build="f64", contract="f64", decompose="f64")

    @classmethod
    def mixed(cls) -> "PrecisionPolicy":
        """Single-precision contraction, double-precision build/decompose.

        The GPU-throughput trade-off for large bond dimension (Gaudin).
        """
        return cls(build="f64", contract="f32", decompose="f64")

    # -- dtype lookup ------------------------------------------------------

    def _label(self, stage: str) -> str:
        if stage not in _STAGES:
            raise ValueError(f"stage must be one of {_STAGES}, got {stage!r}")
        return getattr(self, stage)

    def complex_dtype(self, stage: str):
        """The complex dtype for ``stage`` (``complex128`` or ``complex64``)."""
        return np.dtype(_DTYPES[self._label(stage)][0])

    def real_dtype(self, stage: str):
        """The real dtype for ``stage`` (``float64`` or ``float32``)."""
        return np.dtype(_DTYPES[self._label(stage)][1])

    # -- casting -----------------------------------------------------------

    def caster(self, stage: str, xp):
        """Return a callable that places an array on ``xp`` at ``stage`` precision.

        ``xp`` is the array module (``numpy`` or ``cupy``).  The returned
        callable is the modern replacement for the hand-written ``convert``
        lambda: ``factory.precision.caster('contract', cupy)(arr)``.
        """
        dtype = self.complex_dtype(stage)
        return lambda a: xp.asarray(a, dtype=dtype)
