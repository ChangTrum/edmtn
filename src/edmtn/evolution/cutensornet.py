"""Track 2 (HPC) — 2D space×time EDM contraction via cuQuantum (cuTensorNet).

``backend="hpc"`` routes :func:`edmtn.driver.solve` here instead of Track 1's
sequential fold. The whole separable-bath EDM is laid out as a **2D space×time
tensor network** (paper Sec. V) and contracted **in one shot by cuTensorNet**,
which owns path search, slicing, hardware scheduling, and execution. Two modes,
selected by ``compress_decomp`` (reinterpreted under ``hpc``):

* ``"exact"`` — genuinely **no truncation**, **no knobs**; ``cuquantum.tensornet
  .contract`` (slicing manages memory only). Reports reference error metrics.
* ``"approx"`` — truncation via the ``cutoff`` / ``cutoff_mode`` / ``max_bond``
  knobs (original Track-1 rules), routed **through quimb** ``contract_compressed``.

The contraction-**path-finder** is selectable (``pathfinder``): ``"cuquantum"``
(default — cuTensorNet's own optimizer) or ``"cotengra"`` (cotengra finds the
path, cuQuantum executes). Time layout: one-shot whole-spacetime (default) or
manual time-window blocking (``time_windows``). Both modes return error metrics.

``cupy`` / ``cuquantum`` / ``quimb`` are imported **lazily inside the functions**
so Track 1 (CPU / Windows / macOS) never imports them.

Geometry (validated to ≤2.4e-15 vs Track 1's exact fold, locally and on c1): a
``(1 system + K sub-bath) × T`` grid with d²=4 system bond, ``D_a``=4 lateral bath
bonds, d_phys=7 vertical legs, top arms closed by δ⁰. GPU validation lives in
``examples/cutensornet_sanity.py``.
"""

from __future__ import annotations

import itertools

import numpy as np


class CuTensorNetContractionError(RuntimeError):
    """Raised when an `hpc` one-shot contraction cannot proceed.

    Per the no-silent-guard rule, the message points the user at the manual
    time-window blocking recourse; Track 2 never silently falls back or accepts
    worse-than-requested precision.
    """


# --------------------------------------------------------------------------
# 2D network assembly (backend-agnostic description) — separable bath
# --------------------------------------------------------------------------

def build_2d_network(model, expander, eps: float, n_steps: int, sub_baths=None):
    """Build the 2D space×time EDM network → ``(operands, modes, out_modes, meta)``.

    ``operands[i]`` is contracted with integer mode labels ``modes[i]``;
    ``out_modes`` is the open d² ``vec(ρ(T))`` leg. This is the canonical
    einsum-interleaved description consumed by every backend.
    """
    from ..kernels.separable_mpo import SeparableKernelEngine  # noqa: PLC0415

    order = expander.order
    n_sites = order * n_steps
    d = model.system_dim

    kernel_engine = SeparableKernelEngine.from_model(model, n_steps * eps, eps)
    K = kernel_engine.K
    n_fold = K if sub_baths is None else min(int(sub_baths), K)
    if n_fold < 1:
        raise ValueError(f"sub_baths must be >= 1, got {sub_baths}")
    d_phys = kernel_engine.d_phys

    # system families per column (newest-first), mirroring SeparableBathEvolution
    fam_cache: dict[int, list] = {}
    sys_tensors = []
    for p in range(n_sites):
        g = n_sites - p
        n = (g - 1) // order + 1
        sub = (g - 1) % order
        if n not in fam_cache:
            fam_cache[n] = expander.build_at(model, n * eps, eps).families
        sys_tensors.append(np.asarray(fam_cache[n][sub], dtype=np.complex128))

    bath_sites = [
        [np.asarray(s, dtype=np.complex128)
         for s in kernel_engine.for_sub_bath(k).get_kernel_mpo(n_sites).site_tensors]
        for k in range(n_fold)
    ]

    ids = itertools.count()
    o = [next(ids) for _ in range(n_sites + 1)]
    lat = [[next(ids) for _ in range(n_sites + 1)] for _ in range(n_fold)]
    v = [[next(ids) for _ in range(n_fold + 1)] for _ in range(n_sites)]

    operands: list = []
    modes: list = []

    for p in range(n_sites):
        operands.append(sys_tensors[p])
        modes.append([v[p][0], o[p], o[p + 1]])

    for k in range(n_fold):
        for p in range(n_sites):
            site = bath_sites[k][p]                 # (u, dn, l, r)
            u_, dn_, l_, r_ = site.shape
            md = [v[p][k + 1], v[p][k]]
            shape = [u_, dn_]
            if l_ > 1:
                md.append(lat[k][p]); shape.append(l_)
            if r_ > 1:
                md.append(lat[k][p + 1]); shape.append(r_)
            operands.append(site.reshape(shape))
            modes.append(md)

    e0 = np.zeros(d_phys, dtype=np.complex128)
    e0[0] = 1.0
    for p in range(n_sites):
        operands.append(e0)
        modes.append([v[p][n_fold]])

    rho0 = model.initial_system_state().reshape(-1).astype(np.complex128)
    operands.append(rho0)
    modes.append([o[n_sites]])

    out_modes = [o[0]]
    meta = dict(d=d, d_phys=d_phys, n_sites=n_sites, K=K, n_fold=n_fold)
    return operands, modes, out_modes, meta


def _interleaved(operands, modes, out_modes):
    args = []
    for op, md in zip(operands, modes):
        args += [op, list(md)]
    args.append(list(out_modes))
    return args


# --------------------------------------------------------------------------
# error metrics (returned with every hpc result, exact and approx)
# --------------------------------------------------------------------------

def error_metrics(rho, *, optimizer_info=None, truncation=None) -> dict:
    """Reference error metrics for an `hpc` reduced density matrix.

    Always: ``hermiticity`` = ‖ρ−ρ†‖∞, ``trace_dev`` = |Tr ρ − 1|. Plus, when
    available, the cuTensorNet optimizer info (``num_slices``, ``flops``) for the
    exact mode and the discarded weight for the approximate mode.
    """
    m = {
        "hermiticity": float(np.max(np.abs(rho - rho.conj().T))),
        "trace_dev": float(abs(complex(np.trace(rho)) - 1.0)),
    }
    if optimizer_info is not None:
        oi = optimizer_info[1] if isinstance(optimizer_info, tuple) else optimizer_info
        m["num_slices"] = getattr(oi, "num_slices", None)
        m["flops"] = getattr(oi, "opt_cost", getattr(oi, "flop_count", None))
    if truncation is not None:
        m["discarded_weight"] = float(truncation)
    return m


# --------------------------------------------------------------------------
# contraction modes
# --------------------------------------------------------------------------

def _contract_exact_numpy(operands, modes, out_modes):
    """Exact contraction on CPU — local reference / tests only.

    Uses **opt_einsum**'s path optimizer, NOT ``np.einsum(optimize=True)``: numpy's
    greedy picks a pathological order for the 2D grid and explodes the intermediate
    at modest size — it *hangs* by K=3,N=4 (a C-level einsum loop, uninterruptible),
    whereas opt_einsum ``'auto'`` finds the grid/boundary path in ~ms (verified to
    K=4,N=10). The GPU path uses cuTensorNet's own optimizer, which is unaffected.
    """
    import opt_einsum as oe  # noqa: PLC0415 - quimb/cotengra dependency, always present
    args = []
    for op, md in zip(operands, modes):
        args += [op, list(md)]
    args.append(list(out_modes))
    return oe.contract(*args, optimize="auto"), None


# --------------------------------------------------------------------------
# multi-GPU (cuTensorNet distributed = one MPI rank per GPU, the only multi-GPU model)
# --------------------------------------------------------------------------

def _mpi_context():
    """Return ``(comm, rank, size)`` if launched multi-rank under MPI, else ``None``.

    Detected from the launcher's env (PMI/SLURM/OMPI) so a normal single-process
    call never imports mpi4py or initializes MPI. cuTensorNet's only multi-GPU mode
    is one rank per GPU; launch e.g. ``srun --mpi=pmi2 --ntasks=4 python script.py``.
    """
    import os  # noqa: PLC0415
    n = 1
    for v in ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "SLURM_NTASKS", "MPI_LOCALNRANKS"):
        try:
            n = max(n, int(os.environ.get(v, "1")))
        except ValueError:
            pass
    if n <= 1:
        return None
    from mpi4py import MPI  # noqa: PLC0415 - importing initializes MPI
    comm = MPI.COMM_WORLD
    return comm, comm.Get_rank(), comm.Get_size()


def _resolve_comm_lib() -> str:
    """Resolve + set ``CUTENSORNET_COMM_LIB`` (cuTensorNet's MPI wrapper) across the
    different install layouts; never silently fail (decision: explicit errors).

    Order: an existing valid env var → a prebuilt ``.so`` beside cuquantum's
    ``distributed_interfaces`` (conda-forge ships it) → else a clear build-it error
    (pip wheels ship only the ``.c`` source).
    """
    import os  # noqa: PLC0415
    import os.path as osp  # noqa: PLC0415

    p = os.environ.get("CUTENSORNET_COMM_LIB")
    if p and osp.isfile(p):
        return p
    import cuquantum  # noqa: PLC0415
    di = osp.join(osp.dirname(cuquantum.__file__), "distributed_interfaces")
    prebuilt = osp.join(di, "libcutensornet_distributed_interface_mpi.so")
    if osp.isfile(prebuilt):
        os.environ["CUTENSORNET_COMM_LIB"] = prebuilt
        return prebuilt
    src = osp.join(di, "cutensornet_distributed_interface_mpi.c")
    inc = osp.join(osp.dirname(cuquantum.__file__), "include")
    raise CuTensorNetContractionError(
        "multi-GPU needs the cuTensorNet MPI wrapper, but CUTENSORNET_COMM_LIB is "
        f"unset and no prebuilt library was found in {di}.\nconda-forge cuquantum "
        "ships it prebuilt; with the pip wheel, build it once:\n"
        f"  mpicc -shared -std=c99 -fPIC -I<CUDA>/include -I{inc} \\\n"
        f"    {src} -o {prebuilt}\nthen `export CUTENSORNET_COMM_LIB=<that .so>` "
        "(and on MPICH also `export LD_PRELOAD=<mpi>/libmpi.so`).")


def _make_dist(mpi_ctx):
    """Bind a distributed cuTensorNet handle to the MPI communicator (one GPU/rank).

    Returns ``{comm, rank, size, handle, device_id}`` or ``None`` for single-GPU.
    """
    if mpi_ctx is None:
        return None
    comm, rank, size = mpi_ctx
    _resolve_comm_lib()  # sets CUTENSORNET_COMM_LIB or raises with build instructions
    import cupy as cp  # noqa: PLC0415
    from cupy.cuda.runtime import getDeviceCount  # noqa: PLC0415
    from cuquantum.bindings import cutensornet as cutn  # noqa: PLC0415
    from cuquantum.tensornet import get_mpi_comm_pointer  # noqa: PLC0415

    device_id = rank % getDeviceCount()
    cp.cuda.Device(device_id).use()
    handle = cutn.create()
    cutn.distributed_reset_configuration(handle, *get_mpi_comm_pointer(comm))
    return {"comm": comm, "rank": rank, "size": size, "handle": handle,
            "device_id": device_id}


def _contract_exact_cuquantum(operands, modes, out_modes, *, pathfinder, dist=None):
    """Exact one-shot on GPU. ``pathfinder='cuquantum'`` lets cuTensorNet own the
    whole-network path + slicing; ``'cotengra'`` uses cotengra's path with cuQuantum
    as the executor (no whole-network slicing). With ``dist`` set, cuTensorNet
    distributes the slices across the MPI ranks (one GPU each)."""
    import cupy as cp  # noqa: PLC0415

    if dist is not None:
        cp.cuda.Device(dist["device_id"]).use()
    gpu_ops = [cp.asarray(o) for o in operands]
    out_ix = out_modes[0]

    if dist is not None:
        # multi-GPU: cuTensorNet owns the distributed path + slicing across ranks
        if pathfinder != "cuquantum":
            raise CuTensorNetContractionError(
                "multi-GPU distributed contraction requires pathfinder='cuquantum' "
                "(cuTensorNet owns the distributed path); cotengra is single-GPU only.")
        from cuquantum.tensornet import contract  # noqa: PLC0415
        opts = {"device_id": dist["device_id"], "handle": dist["handle"]}
        info = None
        try:
            res, info = contract(*_interleaved(gpu_ops, modes, out_modes),
                                 options=opts, return_info=True)
        except TypeError:
            res = contract(*_interleaved(gpu_ops, modes, out_modes), options=opts)
        return cp.asnumpy(res), info

    if pathfinder == "cuquantum":
        from cuquantum.tensornet import contract  # noqa: PLC0415
        info = None
        try:
            res, info = contract(*_interleaved(gpu_ops, modes, out_modes), return_info=True)
        except TypeError:
            res = contract(*_interleaved(gpu_ops, modes, out_modes))
        return cp.asnumpy(res), info
    if pathfinder == "cotengra":
        import quimb.tensor as qtn  # noqa: PLC0415
        tn = qtn.TensorNetwork([qtn.Tensor(o, inds=tuple(f"i{m}" for m in md))
                                for o, md in zip(gpu_ops, modes)])
        r = tn.contract(output_inds=(f"i{out_ix}",), optimize="auto", backend="cuquantum")
        return cp.asnumpy(r.data if hasattr(r, "data") else r), None
    raise ValueError(f"unknown pathfinder {pathfinder!r}; choose 'cuquantum' or 'cotengra'")


def _contract_approx_cuquantum(operands, modes, out_modes, *, max_bond, cutoff,
                               cutoff_mode, pathfinder):
    """Approximate (truncated) one-shot through quimb ``contract_compressed`` on GPU.

    The cuTensorNet-native bounded-bond MPS route (``NetworkState``/``MPSConfig``)
    is the multi-GPU-scalable approximate path and is deferred to Phase C; B1's
    approximate mode is the through-quimb ``contract_compressed`` (cupy SVDs).
    """
    import cupy as cp  # noqa: PLC0415
    import quimb.tensor as qtn  # noqa: PLC0415

    gpu_ops = [cp.asarray(o) for o in operands]
    out_ix = out_modes[0]
    tn = qtn.TensorNetwork([qtn.Tensor(o, inds=tuple(f"i{m}" for m in md))
                            for o, md in zip(gpu_ops, modes)])
    # max_bond / cutoff are direct kwargs; cutoff_mode threads via the per-bond
    # compress_opts (contract_compressed rejects it as a direct kwarg).
    opts = dict(max_bond=max_bond, cutoff=cutoff)
    if cutoff_mode is not None:
        opts["compress_opts"] = {"cutoff_mode": cutoff_mode}
    try:
        # TODO(B-perf): a compression-aware contraction tree (cotengra
        # HyperCompressedOptimizer) would avoid the "tree not compressed" notice;
        # 'auto' is correct (B0-validated) but not the most efficient order.
        r = tn.contract_compressed("auto", output_inds=(f"i{out_ix}",), **opts)
    except Exception as e:  # noqa: BLE001
        raise CuTensorNetContractionError(
            f"approximate one-shot contraction failed ({type(e).__name__}: {e}). "
            "Consider manual time-window blocking (time_windows=N) to bound memory, "
            "or relax cutoff/max_bond."
        ) from e
    return cp.asnumpy(r.data if hasattr(r, "data") else r), None


def reduced_density_matrix(model, expander, eps, n_steps, *, mode, pathfinder,
                           max_bond, cutoff, cutoff_mode, sub_baths, executor, dist=None):
    """Assemble + contract the 2D net once → ``(rho, metrics)`` at time ``n_steps``.

    ``dist`` (set only for multi-GPU exact runs) distributes the contraction across
    the MPI ranks via cuTensorNet.
    """
    operands, modes, out_modes, meta = build_2d_network(
        model, expander, eps, n_steps, sub_baths=sub_baths)
    d = meta["d"]
    if mode == "exact":
        if executor == "numpy":
            vec, info = _contract_exact_numpy(operands, modes, out_modes)
        else:
            vec, info = _contract_exact_cuquantum(operands, modes, out_modes,
                                                  pathfinder=pathfinder, dist=dist)
        rho = np.asarray(vec).reshape(d, d)
        return rho, error_metrics(rho, optimizer_info=info)
    if mode == "approx":
        if executor == "numpy":
            raise CuTensorNetContractionError(
                "approximate mode requires the cuQuantum executor (backend='hpc'); "
                "the numpy executor is exact-only (local tests).")
        vec, trunc = _contract_approx_cuquantum(
            operands, modes, out_modes, max_bond=max_bond, cutoff=cutoff,
            cutoff_mode=cutoff_mode, pathfinder=pathfinder)
        rho = np.asarray(vec).reshape(d, d)
        return rho, error_metrics(rho, truncation=trunc)
    raise ValueError(f"unknown hpc mode {mode!r}; under backend='hpc' compress_decomp "
                     "is 'exact' (no knobs) or 'approx' (cutoff knobs)")


# --------------------------------------------------------------------------
# top-level Track-2 solve (invoked from the driver when backend='hpc')
# --------------------------------------------------------------------------

def solve_cutensornet(model, config, *, channel: int | None = None,
                      executor: str = "cuquantum") -> dict:
    """Solve a separable-bath model on the HPC track (2D one-shot contraction).

    The **density operator** is the primary output: ``density_matrices`` holds
    ρ(t) for t = eps..T (and ``final_rho`` = ρ(T)). The channel expectation
    ``polarization`` = ⟨S_channel(t)⟩ is derived only if ``channel`` is given
    (mirroring Track 1, where you pick a channel). ``error_metrics`` (always, exact
    and approx) reports ‖ρ−ρ†‖ / |Tr ρ−1| (+ optimizer slices/flops or discarded
    weight) for the final state. The history is built by contracting the net per
    step (m=1..N) — O(N) contractions, an HPC optimization target; ρ(T) is the
    validated quantity.
    """
    if model.bath_type != "separable":
        raise NotImplementedError(
            "the HPC (cuQuantum 2D) track currently supports separable baths "
            "(e.g. Gaudin); single-bath 2D is a follow-up.")
    if getattr(config, "time_windows", None):
        raise NotImplementedError(
            "manual time-window blocking (time_windows) is wired but not yet "
            "implemented; B1 ships one-shot whole-spacetime. Use time_windows=None.")

    expander = _make_expander(config.expansion_order)

    mode = _resolve_mode(config.compress_decomp)
    pathfinder = getattr(config, "pathfinder", "cuquantum")
    N = config.n_steps

    # multi-GPU: cuTensorNet distributes the EXACT contraction across MPI ranks
    # (one GPU each), auto-detected from the launcher; None for a single process.
    dist = None
    mpi_ctx = _mpi_context() if executor != "numpy" else None
    if mpi_ctx is not None:
        if mode != "exact":
            raise NotImplementedError(
                "multi-GPU is currently only for compress_decomp='exact' (cuTensorNet "
                "distributed contraction); 'approx' (quimb contract_compressed) is "
                "single-GPU — run exact, or launch a single process.")
        dist = _make_dist(mpi_ctx)

    try:
        rhos = []
        metrics_last = None
        for m in range(1, N + 1):
            rho, metrics = reduced_density_matrix(
                model, expander, config.eps, m, mode=mode, pathfinder=pathfinder,
                max_bond=config.max_bond, cutoff=config.cutoff,
                cutoff_mode=config.cutoff_mode, sub_baths=config.sub_baths,
                executor=executor, dist=dist)
            rhos.append(rho)
            metrics_last = metrics
    finally:
        if dist is not None:
            from cuquantum.bindings import cutensornet as cutn  # noqa: PLC0415
            cutn.destroy(dist["handle"])

    times = config.eps * np.arange(1, N + 1)
    pol = None
    if channel is not None:
        Sop = model.coupling_operators_at(N * config.eps)[channel - 1]
        pol = np.array([float(np.trace(Sop @ r).real) for r in rhos])
    return dict(
        times=times, density_matrices=rhos, final_rho=rhos[-1],
        polarization=pol, error_metrics=metrics_last, mode=mode, pathfinder=pathfinder,
        ngpu=(dist["size"] if dist is not None else 1),
        rank=(dist["rank"] if dist is not None else 0),
    )


def _resolve_mode(compress_decomp: str) -> str:
    """Map the `hpc` ``compress_decomp`` value to a contraction mode."""
    if compress_decomp in ("exact", "approx"):
        return compress_decomp
    raise ValueError(
        f"under backend='hpc', compress_decomp must be 'exact' (no knobs) or "
        f"'approx' (cutoff knobs), got {compress_decomp!r}")


def _make_expander(order: int):
    from ..expansion.first_order import FirstOrderExpander  # noqa: PLC0415
    from ..expansion.second_order import SecondOrderExpander  # noqa: PLC0415
    if order == 1:
        return FirstOrderExpander()
    if order == 2:
        return SecondOrderExpander()
    raise ValueError(f"unsupported expansion_order {order!r}")
