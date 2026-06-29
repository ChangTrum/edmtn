"""Cross-node distributed seam for the hpc track — detect-only (Phase D).

cuTensorNet's distributed contraction is MPI-based and **node-count-agnostic**: a
single-node 4-GPU run and a multi-node run are the *same* code path with a larger
rank geometry (more ranks, spanning more hosts). Single-node multi-GPU is validated;
**multi-node is not** (no multi-node test hardware). So rather than silently run an
unvalidated geometry, this seam *detects* the node layout of an hpc run and gates
multi-node off by default with a clear message — set ``EDMTN_ALLOW_MULTINODE=1`` to
opt in (the code path is identical, just unvalidated).

This is a reservation, not an implementation: no multi-node-specific execution logic
lives here. It mirrors the feature-flagged stub of :mod:`edmtn.backend.ozaki_gemm`.
``mpi4py`` is only touched in :meth:`ProcessGroup.from_comm`, so the module imports
fine with no MPI.
"""

from __future__ import annotations

import os

_OPT_IN_ENV = "EDMTN_ALLOW_MULTINODE"


class ProcessGroup:
    """The MPI rank geometry of an hpc run, with a multi-node gate.

    Parameters
    ----------
    n_nodes : int
        Distinct hosts the ranks span.
    n_ranks : int
        Total ranks = total GPUs (one rank per GPU).

    Attributes
    ----------
    multi_node : bool
        ``True`` when the ranks span more than one host.
    opted_in : bool
        Whether ``EDMTN_ALLOW_MULTINODE=1`` is set (read at construction).
    """

    def __init__(self, n_nodes: int, n_ranks: int):
        self.n_nodes = int(n_nodes)
        self.n_ranks = int(n_ranks)
        self.multi_node = self.n_nodes > 1
        self.opted_in = os.environ.get(_OPT_IN_ENV) == "1"

    @classmethod
    def from_comm(cls, comm) -> "ProcessGroup":
        """Build from an mpi4py communicator: unique hostnames → node count.

        Collective (an ``allgather`` of short hostnames); every rank must call it.
        """
        import socket  # noqa: PLC0415
        hosts = comm.allgather(socket.gethostname())
        return cls(len(set(hosts)), len(hosts))

    def require_supported(self) -> "ProcessGroup":
        """Pass on a single node. On multiple nodes, raise unless opted in (then warn).

        Returns ``self`` so callers can chain. Honest gate (no silent run of the
        unvalidated multi-node geometry); the opt-in keeps the seam usable the moment
        multi-node hardware is available to validate.
        """
        if not self.multi_node:
            return self
        if self.opted_in:
            import warnings  # noqa: PLC0415
            warnings.warn(
                f"hpc is running across {self.n_nodes} nodes ({self.n_ranks} ranks). "
                "Multi-node uses the same cuTensorNet distributed code path as "
                "single-node but is NOT validated (no multi-node test hardware). "
                f"Proceeding because {_OPT_IN_ENV}=1.", stacklevel=2)
            return self
        raise NotImplementedError(
            f"hpc was launched across {self.n_nodes} nodes ({self.n_ranks} ranks). "
            "Multi-node distributed contraction is reserved but not yet validated — "
            "single-node multi-GPU is the tested path (no multi-node test hardware). "
            "cuTensorNet's distributed path is node-count-agnostic, so the same code "
            f"should work: to run it unvalidated set {_OPT_IN_ENV}=1, otherwise launch "
            "within one node (e.g. `srun --nodes=1 --ntasks=<#GPUs>`).")
