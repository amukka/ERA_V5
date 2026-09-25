"""Thirty-two virtual GPUs, and the three collectives they talk with.

A virtual GPU here is a Python thread that owns a rank, a slice of the data and
its own tensors, and that can only reach another rank's tensors by calling a
collective.  That restriction is the whole point: it is what makes the byte
counts below real rather than asserted.  Nothing in ``engines.py`` is allowed to
index another rank's state directly, so every byte that moves between ranks
moves through one of ``all_reduce``, ``reduce_scatter`` or ``all_gather`` and is
counted on the way past.

Two things are worth being explicit about.

*The arithmetic is real; the wire is modelled.*  The values these collectives
return are computed exactly, by summing the contributing ranks in rank order, so
two arrangements that should agree bit for bit do agree bit for bit.  The bytes
attributed to each call are the bytes a ring implementation would put on the
wire -- ``N*(W-1)/W`` per rank for a reduce-scatter or an all-gather, twice that
for an all-reduce.  ``ring_all_reduce`` below is the honest version: it performs
the 2(W-1) neighbour-to-neighbour sends one at a time and counts each one, and
experiment 1 checks that its total lands on the formula and its result lands on
the direct answer.

*Summation order is fixed.*  Every reduction stacks the ranks' contributions in
rank order and sums along that axis, for the full buffer and for a shard alike.
Element by element the sequence of additions is therefore the same whether the
caller asked for an all-reduce or for a reduce-scatter of one slice, which is
what lets experiment 2 ask for bitwise equality between the four arrangements
and get it.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field

import torch


# ---------------------------------------------------------------------------
# the wire
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Link:
    """A bandwidth and a per-call launch cost, used to price the byte counts."""
    name: str
    gb_per_s: float
    latency_us: float = 5.0

    def seconds(self, bytes_per_rank: float, calls: int = 1) -> float:
        return bytes_per_rank / (self.gb_per_s * 1e9) + calls * self.latency_us * 1e-6


NVLINK = Link("NVLink, inside one node", 450.0, latency_us=5.0)
INFINIBAND = Link("InfiniBand, between nodes", 50.0, latency_us=10.0)
PCIE = Link("PCIe, CPU to GPU", 60.0, latency_us=10.0)


# ---------------------------------------------------------------------------
# the ledger
# ---------------------------------------------------------------------------

@dataclass
class CommLedger:
    """Bytes on the wire, per rank, per collective, since the last reset."""
    world_size: int
    bytes_sent: dict = field(default_factory=lambda: defaultdict(float))
    calls: dict = field(default_factory=lambda: defaultdict(int))
    by_op: dict = field(default_factory=lambda: defaultdict(float))
    by_tag: dict = field(default_factory=lambda: defaultdict(float))
    seconds_in_collectives: dict = field(default_factory=lambda: defaultdict(float))
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record(self, rank: int, op: str, tag: str, payload_bytes: float) -> float:
        """Attribute one collective's ring cost to one rank.  Returns the bytes."""
        w = self.world_size
        factor = {"all_reduce": 2.0, "reduce_scatter": 1.0,
                  "all_gather": 1.0, "broadcast": 1.0}[op]
        moved = factor * payload_bytes * (w - 1) / w if w > 1 else 0.0
        with self._lock:
            self.bytes_sent[rank] += moved
            self.calls[(rank, op)] += 1
            self.by_op[op] += moved / w          # mean over ranks
            self.by_tag[tag] += moved / w
        return moved

    def reset(self) -> None:
        with self._lock:
            self.bytes_sent.clear()
            self.calls.clear()
            self.by_op.clear()
            self.by_tag.clear()
            self.seconds_in_collectives.clear()

    # -- read-out ------------------------------------------------------------
    @property
    def mean_bytes_per_rank(self) -> float:
        if not self.bytes_sent:
            return 0.0
        return sum(self.bytes_sent.values()) / len(self.bytes_sent)

    @property
    def n_calls(self) -> int:
        return sum(self.calls.values())

    def summary(self) -> dict:
        return {
            "mean_bytes_per_rank": self.mean_bytes_per_rank,
            "max_bytes_per_rank": max(self.bytes_sent.values(), default=0.0),
            "calls_per_rank": self.n_calls / max(len(self.bytes_sent), 1),
            "by_op": dict(self.by_op),
            "by_tag": dict(self.by_tag),
            "seconds_in_collectives": dict(self.seconds_in_collectives),
        }


# ---------------------------------------------------------------------------
# the mesh
# ---------------------------------------------------------------------------

class Mesh:
    """``world_size`` ranks, each one a thread, sharing nothing but collectives."""

    def __init__(self, world_size: int, link: Link = NVLINK):
        self.world_size = world_size
        self.link = link
        self.ledger = CommLedger(world_size)
        self._barrier = threading.Barrier(world_size)
        self._slots: list = [None] * world_size
        self._result = [None] * world_size
        self._errors: list = []

    # -- running -------------------------------------------------------------
    def run(self, fn, *args, **kwargs) -> list:
        """Call ``fn(rank, mesh, *args)`` on every rank at once; return the list
        of return values, indexed by rank.  An exception on any rank is
        re-raised here, after the other ranks have been released."""
        self._result = [None] * self.world_size
        self._errors = []
        self._barrier.reset()

        def target(rank):
            torch.set_num_threads(1)     # a virtual GPU gets one core, not ten
            try:
                self._result[rank] = fn(rank, self, *args, **kwargs)
            except BaseException as exc:          # noqa: BLE001 - re-raised below
                self._errors.append((rank, exc))
                self._barrier.abort()

        threads = [threading.Thread(target=target, args=(r,), daemon=True,
                                    name=f"gpu{r:02d}")
                   for r in range(self.world_size)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if self._errors:
            rank, exc = self._errors[0]
            raise RuntimeError(f"rank {rank} failed: {exc!r}") from exc
        return self._result

    # -- primitives ----------------------------------------------------------
    def barrier(self) -> None:
        self._barrier.wait()

    def _exchange(self, rank: int, value):
        """Publish this rank's contribution and wait until all are published."""
        self._slots[rank] = value
        self._barrier.wait()
        return self._slots

    def _release(self) -> None:
        self._barrier.wait()

    @staticmethod
    def _reduce(stack_source, op: str):
        """Sum the ranks in rank order, accumulating in fp32, then apply ``op``."""
        acc = torch.stack([t.float() for t in stack_source]).sum(0)
        if op == "mean":
            acc /= len(stack_source)
        return acc

    # -- collectives ---------------------------------------------------------
    def all_reduce(self, rank: int, flat: torch.Tensor, op: str = "mean",
                   tag: str = "grads") -> torch.Tensor:
        """Every rank gives a full buffer and gets the combined full buffer."""
        t0 = time.perf_counter()
        slots = self._exchange(rank, flat)
        out = self._reduce(slots, op).to(flat.dtype)
        self._release()
        self.ledger.record(rank, "all_reduce", tag,
                           flat.numel() * flat.element_size())
        self.ledger.seconds_in_collectives[rank] = (
            self.ledger.seconds_in_collectives.get(rank, 0.0)
            + time.perf_counter() - t0)
        return out

    def reduce_scatter(self, rank: int, flat: torch.Tensor, op: str = "mean",
                       tag: str = "grads") -> torch.Tensor:
        """Every rank gives a full buffer and keeps only its own slice of the
        combined result.  The slice is ``flat.numel()//world_size`` long, which
        is why every buffer handed to a collective is padded to a multiple of
        the world size before it gets here."""
        t0 = time.perf_counter()
        w = self.world_size
        assert flat.numel() % w == 0, "buffer must be padded to a multiple of W"
        n = flat.numel() // w
        slots = self._exchange(rank, flat)
        lo, hi = rank * n, (rank + 1) * n
        out = self._reduce([s[lo:hi] for s in slots], op).to(flat.dtype)
        self._release()
        self.ledger.record(rank, "reduce_scatter", tag,
                           flat.numel() * flat.element_size())
        self.ledger.seconds_in_collectives[rank] = (
            self.ledger.seconds_in_collectives.get(rank, 0.0)
            + time.perf_counter() - t0)
        return out

    def all_gather(self, rank: int, shard: torch.Tensor,
                   tag: str = "params") -> torch.Tensor:
        """Every rank gives one slice and gets the whole buffer back."""
        t0 = time.perf_counter()
        slots = self._exchange(rank, shard)
        out = torch.cat([s for s in slots])
        self._release()
        self.ledger.record(rank, "all_gather", tag,
                           out.numel() * out.element_size())
        self.ledger.seconds_in_collectives[rank] = (
            self.ledger.seconds_in_collectives.get(rank, 0.0)
            + time.perf_counter() - t0)
        return out

    def broadcast(self, rank: int, flat: torch.Tensor | None,
                  src: int = 0, tag: str = "init") -> torch.Tensor:
        t0 = time.perf_counter()
        slots = self._exchange(rank, flat)
        out = slots[src].clone()
        self._release()
        self.ledger.record(rank, "broadcast", tag,
                           out.numel() * out.element_size())
        self.ledger.seconds_in_collectives[rank] = (
            self.ledger.seconds_in_collectives.get(rank, 0.0)
            + time.perf_counter() - t0)
        return out

    # -- the honest ring -----------------------------------------------------
    def ring_all_reduce(self, rank: int, flat: torch.Tensor,
                        op: str = "mean", trace: list | None = None):
        """A ring all-reduce performed one neighbour-to-neighbour send at a time.

        Each rank only ever sends to ``rank+1`` and receives from ``rank-1``.
        The buffer is cut into ``W`` chunks; the first ``W-1`` sends accumulate
        one chunk per rank (that is the reduce-scatter), and the second ``W-1``
        sends circulate the finished chunks (that is the all-gather).  Every
        send is one chunk, so the total leaving each rank is
        ``2*(W-1)*N/W`` bytes, which is where ``2P`` comes from.

        Used by experiment 1 to check the formula the ledger uses everywhere
        else.  It is much slower than ``all_reduce`` -- 2(W-1) barriers instead
        of two -- so nothing else calls it.
        """
        w = self.world_size
        assert flat.numel() % w == 0
        n = flat.numel() // w
        chunks = [flat[i * n:(i + 1) * n].float().clone() for i in range(w)]
        sent = 0

        for step in range(w - 1):                      # phase 1: reduce-scatter
            send_idx = (rank - step) % w
            recv_idx = (rank - step - 1) % w
            slots = self._exchange(rank, chunks[send_idx])
            chunks[recv_idx] = chunks[recv_idx] + slots[(rank - 1) % w]
            self._release()
            sent += n * 4
            if trace is not None and rank == 0:
                trace.append({"phase": "reduce_scatter", "step": step,
                              "chunk_sent": send_idx, "chunk_received": recv_idx})

        if op == "mean":
            chunks[(rank + 1) % w] = chunks[(rank + 1) % w] / w

        for step in range(w - 1):                      # phase 2: all-gather
            send_idx = (rank - step + 1) % w
            recv_idx = (rank - step) % w
            slots = self._exchange(rank, chunks[send_idx])
            chunks[recv_idx] = slots[(rank - 1) % w].clone()
            self._release()
            sent += n * 4
            if trace is not None and rank == 0:
                trace.append({"phase": "all_gather", "step": step,
                              "chunk_sent": send_idx, "chunk_received": recv_idx})

        return torch.cat(chunks).to(flat.dtype), sent
