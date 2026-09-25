"""One way to start a run, and one way to carry its numbers up to 30B.

``train`` is the harness every experiment uses: it builds a mesh, draws the same
batches for every arrangement, runs the requested number of steps and hands back
what each rank measured.  Because the batches are drawn before the ranks start
and sliced by rank inside, two runs of two different engines see byte-for-byte
the same text in the same order, which is what makes their losses comparable and
their weights bitwise comparable.

``project`` is the arithmetic of section 1 of the session notes, applied to
whatever bytes-per-weight the engines actually measured rather than to a quoted
constant.  Everything about the 30B model in this repository comes through here.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, asdict

import torch

from .data import Batcher
from .engines import ENGINES, STAGE_ORDER
from .mesh import Mesh, NVLINK, INFINIBAND, Link
from .model import Config, total_params

GB = 1e9
GiB = 2 ** 30


# ---------------------------------------------------------------------------
# running
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    engine: str
    stage: int
    world_size: int
    micro_batch: int
    steps: int
    n_params: int
    losses: list
    bytes_per_weight: float
    resident_bytes: int
    peak_bytes: int
    peak_breakdown: dict
    resident_breakdown: dict
    activation_bytes: int
    wire_bytes_per_step: float
    collective_calls_per_step: float
    seconds_per_step: float
    seconds_compute: float
    seconds_collective: float
    seconds_optimizer: float
    seconds_other: float
    wall_seconds: float
    weights: dict = None
    weights_by_rank: list = None

    @property
    def P(self) -> int:
        """One complete copy of the parameters in bf16, in bytes."""
        return 2 * self.n_params

    @property
    def wire_in_P(self) -> float:
        return self.wire_bytes_per_step / self.P

    def as_dict(self, drop_weights: bool = True) -> dict:
        d = asdict(self)
        if drop_weights:
            d.pop("weights", None)
            d.pop("weights_by_rank", None)
        d["wire_in_P"] = self.wire_in_P
        d["P_bytes"] = self.P
        return d


def train(engine_name: str, world_size: int = 32, steps: int = 10,
          micro_batch: int = 2, cfg: Config | None = None, seed: int = 7,
          lr: float = 3e-4, link: Link = NVLINK, keep_weights: bool = False,
          warmup: int = 1) -> RunResult:
    """Run one arrangement and return what the ranks measured."""
    cfg = cfg or Config()
    mesh = Mesh(world_size, link=link)
    sampler = Batcher(seq_len=cfg.max_seq)
    batches = [sampler.global_batch(world_size, micro_batch) for _ in range(steps)]

    def worker(rank, mesh):
        eng = ENGINES[engine_name](rank, mesh, cfg, lr=lr, seed=seed)
        stats = []
        for s in range(steps):
            x, y = Batcher.for_rank(batches[s], rank, world_size, micro_batch)
            stats.append(eng.step(x, y))
        out = {
            "losses": [st.loss for st in stats],
            "bytes_per_weight": eng.bytes_per_weight(),
            "resident": eng.meter.resident,
            "peak": eng.meter.peak,
            "breakdown": eng.meter.peak_breakdown,
            "resident_breakdown": eng.meter.breakdown(),
            "activation_bytes": stats[-1].activation_bytes,
            "wire": statistics.mean(st.bytes_on_wire for st in stats),
            "t_total": [st.seconds_total for st in stats],
            "t_compute": [st.seconds_compute for st in stats],
            "t_coll": [st.seconds_collective for st in stats],
            "t_opt": [st.seconds_optimizer for st in stats],
            "t_other": [st.seconds_other for st in stats],
        }
        if keep_weights:
            out["weights"] = eng.flat_weights()
        return out

    t0 = time.perf_counter()
    per_rank = mesh.run(worker)
    wall = time.perf_counter() - t0

    r0 = per_rank[0]
    calls = mesh.ledger.n_calls / world_size / steps
    keep = slice(warmup, None) if steps > warmup else slice(None)

    def mean_over_ranks(key):
        return statistics.mean(statistics.mean(r[key][keep]) for r in per_rank)

    return RunResult(
        engine=engine_name,
        stage=ENGINES[engine_name].stage,
        world_size=world_size,
        micro_batch=micro_batch,
        steps=steps,
        n_params=total_params(cfg),
        losses=r0["losses"],
        bytes_per_weight=r0["bytes_per_weight"],
        resident_bytes=r0["resident"],
        peak_bytes=max(r["peak"] for r in per_rank),
        peak_breakdown=r0["breakdown"],
        resident_breakdown=r0["resident_breakdown"],
        activation_bytes=r0["activation_bytes"],
        wire_bytes_per_step=statistics.mean(r["wire"] for r in per_rank),
        collective_calls_per_step=calls,
        seconds_per_step=mean_over_ranks("t_total"),
        seconds_compute=mean_over_ranks("t_compute"),
        seconds_collective=mean_over_ranks("t_coll"),
        seconds_optimizer=mean_over_ranks("t_opt"),
        seconds_other=mean_over_ranks("t_other"),
        wall_seconds=wall,
        weights=r0.get("weights"),
        weights_by_rank=([r["weights"] for r in per_rank]
                         if keep_weights else None),
    )


def train_all(world_size: int = 32, **kw) -> dict:
    return {name: train(name, world_size=world_size, **kw) for name in STAGE_ORDER}


def max_weight_divergence(a: dict, b: dict) -> float:
    return max(float((a[g].float() - b[g].float()).abs().max()) for g in a)


# ---------------------------------------------------------------------------
# carrying it up to 30B
# ---------------------------------------------------------------------------

V5_PARAMS = 30_000_000_000
CARD_BYTES = 80 * GB               # an 80 GB card, which is 74.5 GiB


def project_state(bytes_per_weight: float, n_params: int = V5_PARAMS) -> float:
    """Training state per GPU, in bytes, at a measured bytes-per-weight."""
    return bytes_per_weight * n_params


def project(results: dict, n_params: int = V5_PARAMS) -> dict:
    """Take the measured per-weight cost of each arrangement to V5's size."""
    out = {}
    P = 2 * n_params
    for name, r in results.items():
        state = project_state(r.bytes_per_weight, n_params)
        out[name] = {
            "bytes_per_weight": r.bytes_per_weight,
            "state_bytes": state,
            "state_gib": state / GiB,
            "fits_80gb_card": state < CARD_BYTES,
            "wire_in_P": r.wire_in_P,
            "wire_bytes": r.wire_in_P * P,
            "seconds_nvlink": NVLINK.seconds(r.wire_in_P * P),
            "seconds_infiniband": INFINIBAND.seconds(r.wire_in_P * P),
        }
    return out
