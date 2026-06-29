"""Cross-node distributed seam (Phase D) — detect-only gate, no MPI/GPU needed."""

from __future__ import annotations

import pytest

from edmtn.backend.process_group import ProcessGroup


def test_single_node_passes():
    pg = ProcessGroup(n_nodes=1, n_ranks=4)
    assert not pg.multi_node
    assert pg.require_supported() is pg          # validated path: no gate


def test_multi_node_gated_by_default():
    pg = ProcessGroup(n_nodes=2, n_ranks=8)
    assert pg.multi_node
    with pytest.raises(NotImplementedError, match="[Mm]ulti-node"):
        pg.require_supported()


def test_multi_node_opt_in_warns_and_passes(monkeypatch):
    monkeypatch.setenv("EDMTN_ALLOW_MULTINODE", "1")
    pg = ProcessGroup(n_nodes=2, n_ranks=8)          # reads the env at construction
    assert pg.opted_in
    with pytest.warns(UserWarning, match="not.*validated|NOT validated"):
        assert pg.require_supported() is pg


def test_from_comm_counts_unique_hosts():
    class _FakeComm:
        def __init__(self, hosts):
            self._hosts = hosts

        def allgather(self, _local):
            return self._hosts                       # the gathered per-rank hostnames

    pg = ProcessGroup.from_comm(_FakeComm(["c1", "c1", "c2", "c2"]))
    assert (pg.n_nodes, pg.n_ranks) == (2, 4)
    pg1 = ProcessGroup.from_comm(_FakeComm(["c1", "c1", "c1", "c1"]))
    assert (pg1.n_nodes, pg1.n_ranks) == (1, 4) and not pg1.multi_node
