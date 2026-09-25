"""Four arrangements of the same training step, differing only in what is kept.

Every engine here runs the identical model on the identical data with the
identical optimiser.  What changes between them is one thing: which of the
sixteen bytes per weight a rank stores itself and which it asks the other ranks
for when the moment comes.  That is the whole of ZeRO, and writing the four
engines side by side is the quickest way to see that stages 1 and 2 do not
change the arithmetic at all -- they only change the address of the bytes.

The sixteen bytes, and who holds them at each stage, for world size W:

    what                      bytes  DP    ZeRO-1   ZeRO-2   ZeRO-3
    bf16 weights                  2  full  full     full     1/W
    bf16 gradients                2  full  full     1/W      1/W
    fp32 master copy              4  full  1/W      1/W      1/W
    Adam's two moments            8  full  1/W      1/W      1/W
    -------------------------------------------------------------
    per weight                   16  16    4+12/W   2+14/W   16/W

Nothing in that table is hard-coded anywhere in this file.  Each engine
allocates the tensors it genuinely needs, the meter adds up what it allocated,
and experiment 3 checks the total against the formula afterwards.

The communication is likewise not asserted.  Data parallelism all-reduces the
gradients, which a ring does as a reduce-scatter followed by an all-gather, so
2P crosses the wire.  Stages 1 and 2 perform exactly those two phases -- the
reduce-scatter on the gradients, the all-gather on the updated weights -- and
keep the slice in between instead of throwing it away, so they also pay 2P.
Stage 3 keeps no weights at all, so it must all-gather them in the forward pass
and again in the backward, and pays 3P.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from .model import (Config, forward_backward, init_group, group_numel)

BF16 = torch.bfloat16
FP32 = torch.float32


# ---------------------------------------------------------------------------
# memory meter
# ---------------------------------------------------------------------------

class Meter:
    """Every tensor an engine holds, by name, so the total is measured.

    An engine registers a buffer when it allocates it and unregisters it when it
    drops the reference.  ``peak`` is the largest the sum ever got, which is the
    number that decides whether a rank fits on a card -- resident state alone
    would flatter stage 3, whose whole method is to hold a gathered layer for a
    moment.  What the meter does not see is PyTorch's own scratch inside a
    matmul; every buffer this session reasons about is registered here.
    """

    CATEGORIES = ("weights", "gradients", "optimizer",
                  "gathered", "bucket", "activations")

    def __init__(self):
        self.held: dict[str, int] = {}
        self.peak = 0
        self.peak_breakdown: dict[str, int] = {}

    def hold(self, key: str, tensor) -> None:
        self.held[key] = (tensor.numel() * tensor.element_size()
                          if torch.is_tensor(tensor) else int(tensor))
        self._touch()

    def free(self, key: str) -> None:
        self.held.pop(key, None)

    def _touch(self) -> None:
        total = sum(self.held.values())
        if total > self.peak:
            self.peak = total
            self.peak_breakdown = self.breakdown()

    def breakdown(self) -> dict[str, int]:
        out = {c: 0 for c in self.CATEGORIES}
        for key, n in self.held.items():
            out[key.split(":", 1)[0]] += n
        return out

    @property
    def resident(self) -> int:
        """Persistent training state: everything except transients."""
        return sum(n for k, n in self.held.items()
                   if not k.startswith(("gathered", "bucket", "activations")))


# ---------------------------------------------------------------------------
# parameter providers -- the only route from an engine to a weight
# ---------------------------------------------------------------------------

class LocalProvider:
    """Stages 0-2: the rank already holds every group, so acquiring is free."""

    def __init__(self, engine):
        self.e = engine

    def acquire(self, g):
        return self.e.weights[g]

    def release(self, g):
        pass


class ShardedProvider:
    """Stage 3: a group's weights exist only for as long as they are in use."""

    def __init__(self, engine):
        self.e = engine
        self.live: dict[int, torch.Tensor] = {}

    def acquire(self, g):
        if g not in self.live:
            e = self.e
            full = e.mesh.all_gather(e.rank, e.weights[g], tag="params")
            self.live[g] = full
            e.meter.hold(f"gathered:g{g}", full)
            e.gathers += 1
        return self.live[g]

    def release(self, g):
        self.live.pop(g, None)
        self.e.meter.free(f"gathered:g{g}")


# ---------------------------------------------------------------------------
# the engines
# ---------------------------------------------------------------------------

@dataclass
class StepStats:
    """One step, with the time split so the four parts add up to the whole.

    ``seconds_compute`` is the forward and backward with the time spent inside
    collectives taken out of it, which matters for stages 2 and 3 because that
    is where their reduce-scatters happen.  ``seconds_optimizer`` is the AdamW
    arithmetic alone, timed around the update itself, so it measures the
    duplicated work ZeRO is removing rather than the collectives that happen to
    sit beside it.  ``seconds_other`` is what is left: casting between fp32 and
    bf16, copies and zeroing.
    """
    loss: float = 0.0
    seconds_compute: float = 0.0
    seconds_collective: float = 0.0
    seconds_collective_backward: float = 0.0
    seconds_collective_update: float = 0.0
    seconds_optimizer: float = 0.0
    seconds_other: float = 0.0
    seconds_total: float = 0.0
    bytes_on_wire: float = 0.0
    activation_bytes: int = 0
    gathers: int = 0


class Engine:
    """Common machinery.  Subclasses decide what is sharded and what moves."""

    name = "engine"
    stage = -1
    shards_weights = False
    shards_grads = False
    shards_optimizer = False

    def __init__(self, rank: int, mesh, cfg: Config, lr: float = 3e-4,
                 seed: int = 7, betas=(0.9, 0.95), eps: float = 1e-8,
                 weight_decay: float = 0.0):
        self.rank, self.mesh, self.cfg = rank, mesh, cfg
        self.W = mesh.world_size
        self.lr, self.betas, self.eps, self.wd = lr, betas, eps, weight_decay
        self.meter = Meter()
        self.t = 0
        self.gathers = 0
        self._adam_seconds = 0.0

        self.n_padded = {}
        self.weights: dict[int, torch.Tensor] = {}      # bf16, full or shard
        self.master: dict[int, torch.Tensor] = {}       # fp32, shard if sharded
        self.exp_avg: dict[int, torch.Tensor] = {}
        self.exp_avg_sq: dict[int, torch.Tensor] = {}
        self.grads: dict[int, torch.Tensor] = {}        # bf16, full or shard

        for g in range(cfg.n_groups):
            n = group_numel(cfg, g)
            pad = (-n) % self.W
            self.n_padded[g] = n + pad
            full = torch.cat([init_group(cfg, g, seed), torch.zeros(pad)])

            w = self._own(full) if self.shards_weights else full
            self.weights[g] = w.to(BF16).clone()
            self.meter.hold(f"weights:g{g}", self.weights[g])

            opt_slice = self._own(full) if self.shards_optimizer else full
            self.master[g] = opt_slice.clone()
            self.exp_avg[g] = torch.zeros_like(opt_slice)
            self.exp_avg_sq[g] = torch.zeros_like(opt_slice)
            self.meter.hold(f"optimizer:master:g{g}", self.master[g])
            self.meter.hold(f"optimizer:exp_avg:g{g}", self.exp_avg[g])
            self.meter.hold(f"optimizer:exp_avg_sq:g{g}", self.exp_avg_sq[g])

            # The gradient buffer is persistent in every arrangement, the way
            # ``.grad`` is persistent in PyTorch: it is zeroed between steps, not
            # freed.  Stages 2 and 3 allocate a shard of it; stages 0 and 1
            # allocate the whole thing, and that is the 2 bytes per weight they
            # never get back.
            gn = self._shard_len(g) if self.shards_grads else self.n_padded[g]
            self.grads[g] = torch.zeros(gn, dtype=BF16)
            self.meter.hold(f"gradients:g{g}", self.grads[g])

        self.provider = (ShardedProvider(self) if self.shards_weights
                         else LocalProvider(self))

    # -- sharding helpers ----------------------------------------------------
    def _own(self, flat: torch.Tensor) -> torch.Tensor:
        n = flat.numel() // self.W
        return flat[self.rank * n:(self.rank + 1) * n]

    def _shard_len(self, g: int) -> int:
        return self.n_padded[g] // self.W

    # -- the update ----------------------------------------------------------
    def _adamw(self, g: int, grad_fp32: torch.Tensor) -> None:
        """AdamW on whatever slice of group ``g`` this rank is responsible for.

        Identical arithmetic in all four engines; only the length of the slice
        differs.  Kept in fp32 because that is what the 4-byte master copy and
        the 8 bytes of moments are *for* -- see the last row of the table in the
        module docstring."""
        b1, b2 = self.betas
        m, v, p = self.exp_avg[g], self.exp_avg_sq[g], self.master[g]
        t0 = time.perf_counter()
        m.mul_(b1).add_(grad_fp32, alpha=1 - b1)
        v.mul_(b2).addcmul_(grad_fp32, grad_fp32, value=1 - b2)
        bc1 = 1 - b1 ** self.t
        bc2 = 1 - b2 ** self.t
        if self.wd:
            p.mul_(1 - self.lr * self.wd)
        p.addcdiv_(m / bc1, (v / bc2).sqrt().add_(self.eps), value=-self.lr)
        self._adam_seconds += time.perf_counter() - t0

    # -- one step ------------------------------------------------------------
    def step(self, x, y) -> StepStats:
        self.t += 1
        self.gathers = 0
        before_wire = self.mesh.ledger.bytes_sent.get(self.rank, 0.0)
        before_coll = self.mesh.ledger.seconds_in_collectives.get(self.rank, 0.0)
        st = StepStats()

        self._adam_seconds = 0.0

        def coll_now():
            return self.mesh.ledger.seconds_in_collectives.get(self.rank, 0.0)

        t0 = time.perf_counter()
        loss = self._backward(x, y, st)
        t1, coll_1 = time.perf_counter(), coll_now()
        self._update()
        t2, coll_2 = time.perf_counter(), coll_now()

        st.loss = loss
        st.gathers = self.gathers
        st.bytes_on_wire = self.mesh.ledger.bytes_sent.get(self.rank, 0.0) - before_wire
        # collectives are timed from inside the mesh, so take them out of the
        # phase they happened in rather than counting the wait twice
        st.seconds_collective_backward = max(coll_1 - before_coll, 0.0)
        st.seconds_collective_update = max(coll_2 - coll_1, 0.0)
        st.seconds_collective = (st.seconds_collective_backward
                                 + st.seconds_collective_update)
        st.seconds_total = t2 - t0
        st.seconds_compute = max(t1 - t0 - st.seconds_collective_backward, 0.0)
        st.seconds_optimizer = self._adam_seconds
        st.seconds_other = max(st.seconds_total - st.seconds_compute
                               - st.seconds_collective - st.seconds_optimizer, 0.0)
        return st

    def _backward(self, x, y, st):
        raise NotImplementedError

    def _update(self):
        raise NotImplementedError

    # -- read-out ------------------------------------------------------------
    def flat_weights(self) -> dict[int, torch.Tensor]:
        """The full bf16 weights, gathered if need be, for comparing engines."""
        out = {}
        for g in range(self.cfg.n_groups):
            w = self.weights[g]
            if self.shards_weights:
                w = self.mesh.all_gather(self.rank, w, tag="checkpoint")
            out[g] = w[:group_numel(self.cfg, g)].clone()
        return out

    def bytes_per_weight(self) -> float:
        n = sum(group_numel(self.cfg, g) for g in range(self.cfg.n_groups))
        return self.meter.resident / n


class DataParallel(Engine):
    """Every rank holds everything.  One all-reduce of the gradients per step."""
    name, stage = "data parallel", 0

    def _backward(self, x, y, st):
        def on_grad(g, grad_fp32):
            self.grads[g].copy_(grad_fp32)          # fp32 -> the 2-byte buffer
        loss, act = _run(self, x, y, on_grad)
        st.activation_bytes = act
        return loss

    def _update(self):
        for g in range(self.cfg.n_groups):
            averaged = self.mesh.all_reduce(self.rank, self.grads[g], "mean",
                                            tag="grads")
            self._adamw(g, averaged.float())
            self.weights[g].copy_(self.master[g])   # fp32 master -> bf16 compute
            self.grads[g].zero_()


class Zero1(Engine):
    """Optimizer state sharded.  Reduce-scatter the gradients, all-gather the
    updated weights: the two halves of the all-reduce data parallelism was
    already paying, with the slice in the middle kept instead of discarded."""
    name, stage = "ZeRO-1", 1
    shards_optimizer = True

    _backward = DataParallel._backward

    def _update(self):
        for g in range(self.cfg.n_groups):
            mine = self.mesh.reduce_scatter(self.rank, self.grads[g], "mean",
                                            tag="grads")
            self._adamw(g, mine.float())
            self.weights[g].copy_(self.mesh.all_gather(
                self.rank, self.master[g].to(BF16), tag="params"))
            self.grads[g].zero_()


class Zero2(Engine):
    """Gradients sharded as well.  A group's gradients are reduce-scattered the
    moment the backward pass finishes producing them, and the full-size buffer
    is dropped immediately -- so a rank never holds more than one group's worth
    of full gradients.  That bucket is the transient in the meter's peak, and it
    is the thing DeepSpeed's ``bucket_size`` names."""
    name, stage = "ZeRO-2", 2
    shards_grads = True
    shards_optimizer = True

    def _backward(self, x, y, st):
        def on_grad(g, grad_fp32):
            bucket = grad_fp32.to(BF16)
            self.meter.hold(f"bucket:g{g}", bucket)
            mine = self.mesh.reduce_scatter(self.rank, bucket, "mean", tag="grads")
            self.grads[g].copy_(mine)
            self.meter.free(f"bucket:g{g}")
        loss, act = _run(self, x, y, on_grad)
        st.activation_bytes = act
        return loss

    def _update(self):
        for g in range(self.cfg.n_groups):
            self._adamw(g, self.grads[g].float())
            self.weights[g].copy_(self.mesh.all_gather(
                self.rank, self.master[g].to(BF16), tag="params"))
            self.grads[g].zero_()


class Zero3(Engine):
    """Weights sharded too.  A rank owns 1/W of every group and nothing else;
    a group's weights are all-gathered just before they are used, in the
    forward and again in the backward, and released the instant they are not.
    That second gather is the whole of the difference between 2P and 3P."""
    name, stage = "ZeRO-3", 3
    shards_weights = True
    shards_grads = True
    shards_optimizer = True

    _backward = Zero2._backward

    def _update(self):
        for g in range(self.cfg.n_groups):
            self._adamw(g, self.grads[g].float())
            self.weights[g].copy_(self.master[g])       # stays a shard
            self.grads[g].zero_()


def _run(engine, x, y, on_grad):
    """Forward and backward through ``model.forward_backward``, metered.

    There is one implementation of the model's arithmetic in this repository and
    every engine goes through it.  The only thing an engine varies is the
    provider it hands over and what it does in ``on_grad``.
    """
    from .model import forward_backward

    def metered(g, grad_fp32):
        # autograd hands back one group's gradient in fp32, in every
        # arrangement alike; it is transient, and it is the largest single
        # buffer stage 2 and stage 3 ever touch
        engine.meter.hold(f"bucket:fp32grad{g}", grad_fp32)
        on_grad(g, grad_fp32)
        engine.meter.free(f"bucket:fp32grad{g}")

    loss, _, act_bytes = forward_backward(
        engine.cfg, engine.provider, x, y, on_grad=metered, meter=engine.meter)
    for key in [k for k in engine.meter.held if k.startswith("activations")]:
        engine.meter.free(key)
    return loss, act_bytes


ENGINES = {e.name: e for e in (DataParallel, Zero1, Zero2, Zero3)}
STAGE_ORDER = ["data parallel", "ZeRO-1", "ZeRO-2", "ZeRO-3"]
