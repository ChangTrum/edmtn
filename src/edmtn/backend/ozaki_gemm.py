"""Ozaki/ADP GEMM acceleration backend — Layer 0 seam (skeleton).

CUDA 13.0 Update 2 added an Ozaki-scheme FP64 GEMM emulation to cuBLAS: FP64
operands are split into fixed-point mantissa slices multiplied on the Blackwell
Tensor Cores, with Automatic Dynamic Precision (ADP) choosing the slice count
from the matrix condition number so accuracy is no worse than native FP64.  It
accelerates **GEMM** (tensor contractions, randomized-SVD projections) — not the
cuSOLVER SVD itself (technical plan §2.5).

CuPy does not expose this interface; reaching it needs ctypes / nvmath-python,
and the ZGEMM (complex128) emulation story is unverified.  That real
implementation is **deferred to Phase 4**.  This module exists now only so the
injection seam is present and hardware detection is testable: higher layers can
ask :meth:`OzakiGEMMBackend.detect_hardware` whether the fast path *could* exist,
while :meth:`gemm` stays numerically identical to a plain matmul.

CuPy is imported lazily so detection returns ``False`` (rather than raising) on
machines without a GPU.
"""

from __future__ import annotations

# Blackwell is compute capability major >= 10 (GB200 = 10.0, RTX 5090 = 12.0).
_BLACKWELL_MAJOR = 10
# CUDA runtime 13.0 == 13000 in cudaRuntimeGetVersion encoding.
_MIN_CUDA_RUNTIME = 13000


class OzakiGEMMBackend:
    """Feature-flagged Ozaki/ADP GEMM backend.

    Parameters
    ----------
    strategy : {'auto', 'fixed_mantissa'}
        ADP mode for the eventual Phase-4 implementation.  Unused while the fast
        path is a stub.
    mantissa_bits : int or None
        Fixed mantissa-slice count; ``None`` lets ADP choose.  Unused for now.
    device_id : int
        GPU device index used for hardware detection.

    Attributes
    ----------
    enabled : bool
        ``True`` only when the hardware *could* run the Ozaki path (Blackwell +
        CUDA >= 13.0).  Even then :meth:`gemm` currently uses the plain path;
        ``enabled`` reports capability, not that acceleration is active.
    """

    def __init__(self, strategy: str = "auto", mantissa_bits: int | None = None, device_id: int = 0):
        if strategy not in ("auto", "fixed_mantissa"):
            raise ValueError(f"strategy must be 'auto' or 'fixed_mantissa', got {strategy!r}")
        self.strategy = strategy
        self.mantissa_bits = mantissa_bits
        self.device_id = device_id
        self.enabled = self.detect_hardware(device_id)

    @staticmethod
    def detect_hardware(device_id: int = 0) -> bool:
        """Return whether the device supports the Ozaki/ADP cuBLAS path.

        Requires CuPy, a Blackwell-class GPU (compute capability major >= 10),
        and CUDA runtime >= 13.0.  Any probe failure (no CuPy, no GPU, older
        toolkit) returns ``False`` — so this is safe to call on any machine.
        """
        try:
            import cupy as cp

            if cp.cuda.runtime.getDeviceCount() <= 0:
                return False
            with cp.cuda.Device(device_id):
                cc = cp.cuda.Device(device_id).compute_capability  # e.g. '120'
                major = int(cc[:-1]) if len(cc) > 1 else int(cc)
            runtime = cp.cuda.runtime.runtimeGetVersion()
            return major >= _BLACKWELL_MAJOR and runtime >= _MIN_CUDA_RUNTIME
        except Exception:
            return False

    def gemm(self, A, B, out=None):
        """Matrix product ``A @ B``.

        Currently always the plain backend matmul, regardless of ``enabled`` —
        the Ozaki/ADP fast path is a Phase-4 stub, so results are numerically
        identical to ``A @ B`` everywhere.
        """
        result = A @ B
        if out is not None:
            out[...] = result
            return out
        return result

    def _accelerated_gemm(self, A, B, out=None):  # pragma: no cover - Phase 4
        """Ozaki/ADP-emulated GEMM via ctypes/nvmath-python (Phase 4)."""
        raise NotImplementedError(
            "Ozaki/ADP GEMM is deferred to Phase 4 (needs ctypes/nvmath-python "
            "cuBLAS bindings; ZGEMM emulation support unverified)"
        )
