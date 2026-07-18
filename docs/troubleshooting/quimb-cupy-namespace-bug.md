---
name: quimb-cupy-namespace-bug
description: quimb 1.14 + autoray 0.8.10 + cupy 14.1 GPU split/contract crashes; fix via namespace-cache shim
metadata:
  type: project
---

# quimb + CuPy namespace-cache bug (GPU split/contract)

In the `quimb` conda env (`D:\Productivity\Anaconda3\envs\quimb`, Python 3.14, quimb 1.14.0, autoray 0.8.10, cupy 14.1.0, CUDA 13.2, RTX 5090 sm_120), quimb tensor ops on CuPy-backed tensors crash with `TypeError: cannot use 'tuple' as a dict key (unhashable type: 'cupy.cuda.device.Device')`.

**Root cause:** autoray's `get_namespace(like)` reads `like.device` (a `cupy.cuda.device.Device`) into the cache key `(cls, device, dtype, submodule)`. In cupy 14.1.0 that Device class is unhashable AND immutable (cannot subclass or set `__hash__`).

**Fix (contained, in `src/edmtn/backend/quimb_linalg.py`, `apply_quimb_cupy_compat()`):** replace `autoray.autoray._NAMESPACE_CACHE` with a dict subclass that coerces unhashable keys (swap the device element for `('unhashable', type(device).__name__, repr(device))`). Functions read the module global at call time, so rebinding works. Verified: GPU `tensor.split(method='svd')` and contraction both succeed after the shim. Phase-1 hot path (`CuPySVDBackend`) calls `cupy.linalg.svd` directly and avoids this entirely.
