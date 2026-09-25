"""A discrete-event model of one pass and the transfers it overlaps with.

The thread mesh cannot answer the timing question -- its "wire" is a memcpy
between two Python lists -- so the timing question is answered here instead,
from three measured or stated inputs and nothing else: the per-layer shape of
the pass (measured), the bytes each transfer carries (counted by the ledger in
experiment 3) and a bandwidth (stated).

One function does both directions, because both directions are the same problem.

    forward pass     layer 0 first.  Stage 3 (and stages 1 and 2, for the
                     weights the optimizer has just updated) must *gather* a
                     layer's weights before computing it, and issues the gather
                     ``prefetch`` layers early.

    backward pass    last layer first.  Each layer's gradients are ready when
                     its own backward finishes, and a bucket of ``k`` layers is
                     handed to the link when the last of them is done.

The link serves one transfer at a time at ``gb_per_s`` with ``latency_s`` of
fixed cost each, which is what stops the smallest bucket from always winning.
A pass ends when both its compute and its link are finished; ``exposed_seconds``
is the part of the traffic that was not hidden -- time the compute spent waiting
for weights, plus any transfer still running after the compute ended.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Schedule:
    bucket_layers: int
    prefetch: int
    compute_seconds: float
    transfer_seconds: float
    pass_seconds: float
    exposed_seconds: float
    stalled_seconds: float
    tail_seconds: float
    hidden_fraction: float
    n_transfers: int

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def simulate_pass(layer_times, gather_bytes=None, grad_bytes=None,
                  bucket_layers: int = 1, gb_per_s: float = 450.0,
                  latency_s: float = 5e-6, prefetch: int = 1,
                  reverse: bool = True) -> Schedule:
    """One pass over the layers, with the transfers it triggers, on one link."""
    n = len(layer_times)
    order = list(range(n - 1, -1, -1)) if reverse else list(range(n))

    link_free = 0.0
    clock = 0.0
    transfer_seconds = 0.0
    stalled = 0.0
    n_transfers = 0
    last_end = 0.0
    arrives_at: dict = {}
    pending: list = []

    def send(payload, not_before):
        nonlocal link_free, transfer_seconds, n_transfers, last_end
        cost = payload / (gb_per_s * 1e9) + latency_s
        start = max(not_before, link_free)
        link_free = start + cost
        transfer_seconds += cost
        n_transfers += 1
        last_end = max(last_end, link_free)
        return link_free

    if gather_bytes is not None:
        for pos in range(min(prefetch, n)):      # requested before the pass starts
            arrives_at[order[pos]] = send(gather_bytes[order[pos]], 0.0)

    for pos, i in enumerate(order):
        if gather_bytes is not None:
            arrival = arrives_at.get(i, 0.0)
            if arrival > clock:
                stalled += arrival - clock
                clock = arrival
        start = clock
        if gather_bytes is not None and pos + prefetch < n:
            j = order[pos + prefetch]            # ask for the next one now
            arrives_at[j] = send(gather_bytes[j], start)
        clock = start + layer_times[i]
        if grad_bytes is not None:
            pending.append(i)
            if len(pending) == bucket_layers or pos == n - 1:
                send(sum(grad_bytes[j] for j in pending), clock)
                pending = []

    tail = max(0.0, last_end - clock)
    exposed = stalled + tail
    return Schedule(
        bucket_layers=bucket_layers,
        prefetch=prefetch,
        compute_seconds=sum(layer_times),
        transfer_seconds=transfer_seconds,
        pass_seconds=max(clock, last_end),
        exposed_seconds=exposed,
        stalled_seconds=stalled,
        tail_seconds=tail,
        hidden_fraction=(1.0 - exposed / transfer_seconds) if transfer_seconds else 1.0,
        n_transfers=n_transfers,
    )


def best_schedule(layer_times, buckets=(1, 2, 3, 4, 6, 8, 12, 16, 24, 48),
                  prefetches=(1, 2, 3), **kw) -> Schedule:
    """The configuration with the shortest pass, searched over bucket and depth."""
    best = None
    for k in buckets:
        if k > len(layer_times):
            continue
        for pf in prefetches:
            s = simulate_pass(layer_times, bucket_layers=k, prefetch=pf, **kw)
            if best is None or s.pass_seconds < best.pass_seconds:
                best = s
    return best
