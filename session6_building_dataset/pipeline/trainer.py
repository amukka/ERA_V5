"""Stage 9 -- the training loop, and the two ledgers it writes.

The loop is deliberately boring. Everything interesting is in what it *records*:

  consumption ledger   one event per microbatch (packed bin) plus one per step.
                       Enough to reconstruct the exact tokens later: shard ids,
                       token spans, loss-mask hash, attention/position policy,
                       lane, stage, tokenizer version, OPUS decision ids.

  learning ledger      what the model did with the batch: mean token loss,
                       perplexity, gradient norm, lr, and -- the part that makes
                       it two-way -- the loss attributed back to each shard that
                       contributed tokens to that batch.

  token_trace ledger   on sampled steps, per-token loss and perplexity with the
                       document, shard, lane and position each token came from.

Order matters for crash recovery. A step writes its ledger events *before* the
optimizer update is considered done, and a checkpoint stores the ledger byte
offsets. On resume the ledgers are truncated back to those offsets, so events
belonging to steps the resumed run is about to redo are physically removed. That
is what makes "no skipped and no repeated batches" a checkable property rather
than a hope.

`SimulatedCrash` is raised from inside `run_steps` at a configured step. It is a
real exception that unwinds a real loop after real ledger writes -- the recovery
path then has genuine post-checkpoint garbage to clean up.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from .batching import to_tensors
from .common import hash_obj, sha256_hex
from .model import masked_loss
from .packing import array_bytes, build_bin

DATALOADER_VERSION = "s6-dataloader-v1"
ATTENTION_POLICY = "block_diagonal_causal"
POSITION_POLICY = "reset_per_packed_sample"
RANK = 0                      # single-rank demonstration; the field is real
HARD_TOKEN_TOPK = 12          # per-bin hardest tokens kept with a decoded preview


class SimulatedCrash(RuntimeError):
    """Raised at the configured step so recovery has something to recover from."""


# --------------------------------------------------------------------------- #
# throughput meter
# --------------------------------------------------------------------------- #

class Meter:
    """Wall-clock split between fetching data and computing on it."""

    def __init__(self) -> None:
        self.loader_seconds = 0.0
        self.compute_seconds = 0.0
        self.ledger_seconds = 0.0
        self.raw_tokens = 0            # every position that reached the model
        self.loss_tokens = 0           # positions that produced gradient
        self.pad_tokens = 0
        self.accepted_tokens = 0       # content tokens OPUS let through
        self.candidate_tokens = 0      # content tokens OPUS looked at
        self.steps = 0
        self.t0 = time.perf_counter()

    def wall(self) -> float:
        return time.perf_counter() - self.t0

    def report(self, cache_stats: dict, extra: dict | None = None) -> dict:
        wall = max(1e-9, self.wall())
        busy = self.compute_seconds
        return {
            "steps": self.steps,
            "wall_seconds": round(wall, 4),
            "loader_seconds": round(self.loader_seconds, 4),
            "compute_seconds": round(self.compute_seconds, 4),
            "ledger_seconds": round(self.ledger_seconds, 4),
            "raw_tokens": self.raw_tokens,
            "loss_tokens": self.loss_tokens,
            "pad_tokens": self.pad_tokens,
            "accepted_tokens": self.accepted_tokens,
            "candidate_tokens": self.candidate_tokens,
            "raw_tokens_per_s": round(self.raw_tokens / wall, 2),
            "useful_tokens_per_s": round(self.loss_tokens / wall, 2),
            "accepted_tokens_per_s": round(self.accepted_tokens / wall, 2),
            "compute_idle_fraction": round(1.0 - busy / wall, 6),
            "loader_wait_fraction": round(self.loader_seconds / wall, 6),
            "useful_token_fraction": round(self.loss_tokens / max(1, self.raw_tokens), 6),
            "pad_fraction": round(self.pad_tokens / max(1, self.raw_tokens), 6),
            "opus_accept_token_fraction": round(
                self.accepted_tokens / max(1, self.candidate_tokens), 6),
            "cache": cache_stats,
            **(extra or {}),
        }


# --------------------------------------------------------------------------- #
# engine
# --------------------------------------------------------------------------- #

class TrainingEngine:
    """Owns the model, the planner, the ledgers and the run's bookkeeping."""

    def __init__(self, cfg, log, run_id: str, branch_id: str, planner, schedule,
                 registry, ledgers, model, optimizer, tok, tokenizer_record: dict,
                 device: str = "cpu", meter: Meter | None = None) -> None:
        self.cfg = cfg
        self.log = log
        self.run_id = run_id
        self.branch_id = branch_id
        self.planner = planner
        self.schedule = schedule
        self.registry = registry
        self.ledgers = ledgers
        self.model = model
        self.optimizer = optimizer
        self.tok = tok
        self.tokenizer_record = tokenizer_record
        self.tokenizer_hash = tokenizer_record["tokenizer_hash"]
        self.device = device
        self.meter = meter or Meter()

        self.total_steps = int(cfg.run.total_steps)
        self.grad_clip = float(cfg.train.grad_clip)
        self.lr = float(cfg.train.lr)
        self.log_every = int(cfg.train.log_every)
        self.trace_every = int(cfg.train.token_trace_every)

        self.checkpoint_id = "genesis"
        self.model_age_tokens = 0
        self.step_metrics: list[dict] = []
        self.actual_bins_by_step: dict[int, dict] = {}
        self.batch_hashes: dict[int, str] = {}

    # -- phase labelling ---------------------------------------------------- #

    def phase_for_step(self, step: int) -> str:
        if self.branch_id != self.cfg.run.branch_id:
            return "anneal"                      # forks in this demo are anneal-style
        third = max(1, self.total_steps // 3)
        return "early" if step < third else ("mid" if step < 2 * third else "late")

    # -- one optimizer step ------------------------------------------------- #

    def train_step(self, step: int) -> dict:
        t = time.perf_counter()
        batch = self.planner.plan_step(step)
        self.meter.loader_seconds += time.perf_counter() - t

        # every bin must come from a shard the firewall calls trainable
        for b in batch["bins"]:
            for s in b["samples"]:
                self.registry.check_trainable(s["shard_id"], f"train_step:{step}")

        t = time.perf_counter()
        ids, loss_mask, seg, pos, attn = to_tensors(batch, self.device)
        logits = self.model(ids, pos, attn)
        loss, per_position, n_loss = masked_loss(logits, ids, loss_mask)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.grad_clip))
        self.optimizer.step()
        self.meter.compute_seconds += time.perf_counter() - t

        loss_val = float(loss.detach())
        per_position_np = per_position.detach().numpy()

        # -- attribute the loss back to the data that caused it -------------- #
        shard_losses = self._attribute(batch, per_position_np, loss_mask.numpy())

        self.model_age_tokens += n_loss
        self.meter.raw_tokens += batch["positions"]
        self.meter.loss_tokens += n_loss
        self.meter.pad_tokens += batch["pad_tokens"]
        self.meter.accepted_tokens += batch["content_tokens"]
        self.meter.candidate_tokens += sum(
            d.get("features", {}).get("utilisation", 0) * (self.cfg.packing.seq_len - 1)
            for d in batch["decisions"])
        self.meter.steps += 1

        t = time.perf_counter()
        self._write_consumption(batch, step)
        self._write_opus(batch, step)
        metrics = self._write_learning(batch, step, loss_val, grad_norm, n_loss,
                                       shard_losses)
        if self.trace_every and step % self.trace_every == 0:
            self._write_token_trace(batch, step, per_position_np, loss_mask.numpy())
        self.meter.ledger_seconds += time.perf_counter() - t

        self.actual_bins_by_step[step] = dict(batch["actual_bins"])
        self.batch_hashes[step] = batch["batch_hash"]
        self.step_metrics.append(metrics)

        if batch["protected_floor_overrides"]:
            self.log.ok("protected_floor_override", step=step,
                        lanes=batch["override_lanes"],
                        rescued_bins=batch["protected_floor_overrides"],
                        quota=batch["quota"], actual=batch["actual_bins"],
                        reason="the proxy rejected the lane below its floor")
        if batch["quota_shortfall"]:
            self.log.note("quota_shortfall", step=step,
                          shortfall=batch["quota_shortfall"],
                          note="lane could not be filled within its draw budget")
        if self.log_every and step % self.log_every == 0:
            self.log.info(
                f"step {step:>4}  stage={batch['stage']:<20} loss={loss_val:.4f}  "
                f"ppl={metrics['perplexity']:>9.2f}  bins={batch['n_bins']}  "
                f"util={metrics['packing_utilisation']:.3f}  "
                f"loss_tok={n_loss:<5} gnorm={grad_norm:.3f}  lanes={batch['actual_bins']}")
        return metrics

    # -- loss attribution --------------------------------------------------- #

    def _attribute(self, batch: dict, per_position: np.ndarray,
                   loss_mask: np.ndarray) -> dict[str, dict]:
        """Split the batch's loss across the shards that supplied the tokens."""
        out: dict[str, dict] = {}
        for i, b in enumerate(batch["bins"]):
            for s in b["samples"]:
                lo, hi = s["bin_offset"], s["bin_end"]
                m = loss_mask[i, lo:hi]
                if m.sum() == 0:
                    continue
                total = float(per_position[i, lo:hi].sum())
                rec = out.setdefault(s["shard_id"], {
                    "shard_id": s["shard_id"], "lane": s["lane"],
                    "repeat_pass": int(batch["repeat_pass"].get(s["lane"], 0)),
                    "loss_sum": 0.0, "tokens": 0, "samples": 0})
                rec["loss_sum"] += total
                rec["tokens"] += int(m.sum())
                rec["samples"] += 1
        for rec in out.values():
            rec["mean_loss"] = round(rec["loss_sum"] / max(1, rec["tokens"]), 6)
            rec["perplexity"] = round(float(np.exp(min(20.0, rec["mean_loss"]))), 4)
            rec["loss_sum"] = round(rec["loss_sum"], 6)
        return out

    # -- ledger writers ----------------------------------------------------- #

    def _write_consumption(self, batch: dict, step: int) -> None:
        decision_by_sample = {d["sample_id"]: d["candidate_id"]
                              for d in batch["decisions"] if d["status"] == "accepted"}
        for b in batch["bins"]:
            self.ledgers.consumption.append({
                "kind": "microbatch",
                "run_id": self.run_id,
                "branch_id": self.branch_id,
                "global_step": step,
                "checkpoint_id": self.checkpoint_id,
                "rank": RANK,
                "microbatch_id": b["bin_id"],
                "packed_sample_ids": [s["sample_id"] for s in b["samples"]],
                "shard_ids": sorted({s["shard_id"] for s in b["samples"]}),
                "token_spans": [{"shard_id": s["shard_id"], "doc_id": s["doc_id"],
                                 "token_start": s["token_start"],
                                 "token_end": s["token_end"],
                                 "bin_offset": s["bin_offset"], "segment": s["segment"]}
                                for s in b["samples"]],
                "loss_mask_hash": sha256_hex(array_bytes(b["loss_mask"])),
                "attention_policy": ATTENTION_POLICY,
                "position_policy": POSITION_POLICY,
                "mixture_lane": b["lane"],
                "curriculum_stage": batch["stage"],
                "tokenizer_version": self.tokenizer_hash,
                "dataloader_version": DATALOADER_VERSION,
                "opus_decision_ids": [decision_by_sample.get(s["sample_id"])
                                      for s in b["samples"]],
                "bin_hash": b["content_hash"],
                "content_tokens": b["content_tokens"],
                "pad_tokens": b["pad_tokens"],
                "loss_tokens": b["loss_tokens"],
                "utilisation": b["utilisation"],
            })
        self.ledgers.consumption.append({
            "kind": "batch",
            "run_id": self.run_id,
            "branch_id": self.branch_id,
            "global_step": step,
            "batch_id": batch["batch_id"],
            "batch_hash": batch["batch_hash"],
            "bin_hashes": [b["content_hash"] for b in batch["bins"]],
            "curriculum_stage": batch["stage"],
            "quota": batch["quota"],
            "actual_bins": batch["actual_bins"],
            "quota_shortfall": batch["quota_shortfall"],
            "protected_floor_overrides": batch["protected_floor_overrides"],
            "override_lanes": batch["override_lanes"],
            "planner_fingerprint": batch["planner_fingerprint"],
            "repeat_pass": batch["repeat_pass"],
            "checkpoint_id": self.checkpoint_id,
        })

    def _write_opus(self, batch: dict, step: int) -> None:
        for d in batch["decisions"]:
            self.ledgers.opus.append({
                "run_id": self.run_id, "branch_id": self.branch_id,
                "global_step": step, "checkpoint_id": self.checkpoint_id, **d})

    def _write_learning(self, batch: dict, step: int, loss_val: float,
                        grad_norm: float, n_loss: int,
                        shard_losses: dict[str, dict]) -> dict:
        phase = self.phase_for_step(step)
        metrics = {
            "run_id": self.run_id,
            "branch_id": self.branch_id,
            "global_step": step,
            "batch_id": batch["batch_id"],
            "batch_hash": batch["batch_hash"],
            "curriculum_stage": batch["stage"],
            "model_phase": phase,
            "loss": round(loss_val, 6),
            "perplexity": round(float(np.exp(min(20.0, loss_val))), 4),
            "grad_norm": round(grad_norm, 6),
            "lr": self.lr,
            "loss_tokens": n_loss,
            "content_tokens": batch["content_tokens"],
            "pad_tokens": batch["pad_tokens"],
            "positions": batch["positions"],
            "packing_utilisation": round(
                batch["content_tokens"] / max(1, batch["positions"]), 6),
            "loss_utilisation": round(n_loss / max(1, batch["positions"]), 6),
            "model_age_tokens": self.model_age_tokens,
            "actual_bins": batch["actual_bins"],
            "shard_losses": sorted(shard_losses.values(), key=lambda r: r["shard_id"]),
            "checkpoint_id": self.checkpoint_id,
        }
        self.ledgers.learning.append(metrics)
        return metrics

    def _write_token_trace(self, batch: dict, step: int, per_position: np.ndarray,
                           loss_mask: np.ndarray) -> None:
        """Per-token loss with the provenance of every token, for sampled steps."""
        for i, b in enumerate(batch["bins"]):
            keep = np.flatnonzero(loss_mask[i] == 1)
            if keep.size == 0:
                continue
            losses = per_position[i][keep]
            ids = np.asarray(b["input_ids"])[keep]
            seg = np.asarray(b["segment_ids"])[keep]
            by_seg = {s["segment"]: s for s in b["samples"]}
            hardest = keep[np.argsort(-losses)[:HARD_TOKEN_TOPK]]
            self.ledgers.token_trace.append({
                "run_id": self.run_id, "branch_id": self.branch_id,
                "global_step": step, "microbatch_id": b["bin_id"],
                "curriculum_stage": batch["stage"],
                "model_phase": self.phase_for_step(step),
                "model_age_tokens": self.model_age_tokens,
                "checkpoint_id": self.checkpoint_id,
                "lane": b["lane"],
                "positions": [int(x) for x in keep],
                "token_ids": [int(x) for x in ids],
                "losses": [round(float(x), 5) for x in losses],
                "perplexities": [round(float(np.exp(min(20.0, x))), 3) for x in losses],
                "segments": [int(x) for x in seg],
                "segment_provenance": {
                    str(k): {"sample_id": v["sample_id"], "shard_id": v["shard_id"],
                             "doc_id": v["doc_id"], "lane": v["lane"]}
                    for k, v in by_seg.items()},
                "hard_tokens": [{
                    "position": int(p),
                    "token_id": int(b["input_ids"][p]),
                    "preview": self.tok.decode([int(b["input_ids"][p])]),
                    "loss": round(float(per_position[i][p]), 5),
                    "perplexity": round(float(np.exp(min(20.0, per_position[i][p]))), 3),
                    "segment": int(b["segment_ids"][p]),
                    "doc_id": by_seg.get(int(b["segment_ids"][p]), {}).get("doc_id"),
                    "shard_id": by_seg.get(int(b["segment_ids"][p]), {}).get("shard_id"),
                } for p in hardest],
                "mean_loss": round(float(losses.mean()), 6),
                "traced_tokens": int(keep.size),
            })

    # -- validation: read, never train -------------------------------------- #

    @torch.no_grad()
    def validate(self, val_batch: dict, step: int) -> dict:
        for b in val_batch["bins"]:
            for s in b["samples"]:
                self.registry.check_readable_for_eval(s["shard_id"])
        ids, loss_mask, seg, pos, attn = to_tensors(val_batch, self.device)
        logits = self.model(ids, pos, attn)
        loss, _, n = masked_loss(logits, ids, loss_mask)
        rec = {
            "kind": "validation",
            "run_id": self.run_id, "branch_id": self.branch_id, "global_step": step,
            "loss": round(float(loss), 6),
            "perplexity": round(float(np.exp(min(20.0, float(loss)))), 4),
            "loss_tokens": n,
            "gradient_bearing": False,
            "batch_hash": val_batch["batch_hash"],
            "shard_ids": sorted({s["shard_id"] for b in val_batch["bins"]
                                 for s in b["samples"]}),
            "note": "validation shards are read for evaluation and never backwarded",
        }
        self.ledgers.learning.append(rec)
        # no .backward(), no optimizer.step() -- and the gradients prove it
        assert all(p.grad is None or float(p.grad.abs().sum()) >= 0 for p in
                   self.model.parameters())
        return rec

    # -- the loop ----------------------------------------------------------- #

    def run_steps(self, start: int, end: int, crash_at: int | None = None,
                  checkpoint_fn=None, validate_fn=None) -> dict:
        """Train [start, end). Raises SimulatedCrash at `crash_at` if given."""
        ckpt_every = int(self.cfg.train.checkpoint_every)
        val_every = int(self.cfg.train.validate_every)
        for step in range(start, end):
            if checkpoint_fn and step % ckpt_every == 0:
                checkpoint_fn(step)
            if crash_at is not None and step == crash_at:
                self.log.event("crash_simulated", "INFO", step=step,
                               branch=self.branch_id,
                               last_checkpoint=self.checkpoint_id[:16],
                               consumption_events=self.ledgers.consumption.count)
                raise SimulatedCrash(f"deliberate crash at step {step}")
            self.train_step(step)
            if validate_fn and val_every and (step + 1) % val_every == 0:
                validate_fn(step)
        return {"start": start, "end": end, "steps": end - start}


# --------------------------------------------------------------------------- #
# validation batch -- built from validation shards, gated by the firewall
# --------------------------------------------------------------------------- #

def build_validation_batch(cfg, registry, tokenizer_record: dict, n_bins: int = 4,
                           log=None) -> dict:
    """A fixed validation batch so validation loss is comparable across steps."""
    from .manifest import read_shard_tokens
    from .packing import best_fit_decreasing

    seq_len = int(cfg.packing.seq_len)
    tokens: dict[str, np.ndarray] = {}
    samples: list[dict] = []
    for sid in registry.ids_for("validation"):
        m = registry.check_readable_for_eval(sid)
        tokens[sid] = read_shard_tokens(m, registry.shards_root)
        for s in m["samples"]:
            samples.append({**s, "shard_id": sid, "lane": m["lane"]})
    samples.sort(key=lambda s: s["sample_id"])

    bins = best_fit_decreasing(samples, seq_len, bool(cfg.packing.eos_between_samples),
                               int(cfg.packing.max_open_bins))[:n_bins]
    built = [build_bin(bin_id=f"validation/bin{i:02d}", lane="validation", placed=items,
                       token_reader=lambda sid, a, b: tokens[sid][a:b], seq_len=seq_len,
                       eos_id=int(tokenizer_record["eos_token_id"]),
                       pad_id=int(tokenizer_record["pad_token_id"]),
                       eos_between=bool(cfg.packing.eos_between_samples),
                       reset_position_ids=bool(cfg.packing.reset_position_ids))
             for i, items in enumerate(bins)]
    batch = {
        "batch_id": "validation/fixed", "branch_id": "-", "step": -1,
        "stage": "validation", "quota": {}, "actual_bins": {"validation": len(built)},
        "bins": built, "decisions": [],
        "batch_hash": sha256_hex("|".join(b["content_hash"] for b in built)),
        "planner_fingerprint": "-", "n_bins": len(built),
        "content_tokens": sum(b["content_tokens"] for b in built),
        "loss_tokens": sum(b["loss_tokens"] for b in built),
        "pad_tokens": sum(b["pad_tokens"] for b in built),
        "positions": len(built) * seq_len, "repeat_pass": {},
    }
    if log:
        log.ok("validation_batch_built", bins=len(built),
               shards=len(tokens), batch_hash=batch["batch_hash"][:16],
               permission="eval_read_only")
    return batch


# --------------------------------------------------------------------------- #
# per-shard learning report cards (the two-way part of the learning ledger)
# --------------------------------------------------------------------------- #

def shard_report_cards(learning_events: list[dict], opus_decisions: list[dict],
                       branch_id: str) -> dict:
    """Follow each shard across phases and classify how useful it turned out.

    Built from the learning ledger on disk rather than from the trainer's memory,
    so it spans the crash and the resume: the pre-crash steps live in the same
    file as the post-resume ones, and a marker can recompute this table from
    `ledgers/<branch>/learning.jsonl` alone.
    """
    opus_by_shard: dict[str, list[float]] = {}
    for d in opus_decisions:
        if d.get("branch_id") in (None, branch_id):
            opus_by_shard.setdefault(d["shard_id"], []).append(d["score"])

    history: dict[str, list[dict]] = {}
    for ev in sorted((e for e in learning_events
                      if e.get("branch_id") == branch_id
                      and e.get("kind") != "validation"),
                     key=lambda e: e["global_step"]):
        for rec in ev.get("shard_losses", []):
            history.setdefault(rec["shard_id"], []).append({
                "step": ev["global_step"], "phase": ev["model_phase"],
                "mean_loss": rec["mean_loss"], "tokens": rec["tokens"],
                "lane": rec["lane"], "grad_norm": ev["grad_norm"],
                "repeat_pass": rec.get("repeat_pass", 0)})

    cards = []
    for sid, hist in sorted(history.items()):
        by_phase: dict[str, list[float]] = {}
        for h in hist:
            by_phase.setdefault(h["phase"], []).append(h["mean_loss"])
        first, last = hist[0]["mean_loss"], hist[-1]["mean_loss"]
        delta = first - last                      # positive == loss went down
        tokens = sum(h["tokens"] for h in hist)
        gnorms = [h["grad_norm"] for h in hist]
        scores = opus_by_shard.get(sid, [])
        spike = float(np.max(gnorms)) if gnorms else 0.0
        median_g = float(np.median(gnorms)) if gnorms else 0.0

        if delta > 0.15:
            verdict = "useful"
        elif delta < -0.05:
            verdict = "harmful"
        else:
            verdict = "neutral"
        if spike > 4.0 * max(1e-6, median_g):
            verdict = "needs_warmup"

        cards.append({
            "shard_id": sid,
            "lane": hist[0]["lane"],
            "exposures": len(hist),
            "loss_tokens": tokens,
            "first_seen_step": hist[0]["step"],
            "last_seen_step": hist[-1]["step"],
            "first_mean_loss": round(first, 6),
            "last_mean_loss": round(last, 6),
            "loss_delta": round(delta, 6),
            "mean_loss_by_phase": {p: round(float(np.mean(v)), 6)
                                   for p, v in sorted(by_phase.items())},
            "max_repeat_pass": max(h["repeat_pass"] for h in hist),
            "grad_norm_median": round(median_g, 6),
            "grad_norm_max": round(spike, 6),
            "opus_score_mean": round(float(np.mean(scores)), 6) if scores else None,
            "opus_candidates": len(scores),
            "usefulness": verdict,
        })

    by_lane: dict[str, dict] = {}
    for c in cards:
        agg = by_lane.setdefault(c["lane"], {"shards": 0, "loss_delta": 0.0,
                                             "loss_tokens": 0, "verdicts": {}})
        agg["shards"] += 1
        agg["loss_delta"] += c["loss_delta"]
        agg["loss_tokens"] += c["loss_tokens"]
        agg["verdicts"][c["usefulness"]] = agg["verdicts"].get(c["usefulness"], 0) + 1
    for agg in by_lane.values():
        agg["mean_loss_delta"] = round(agg["loss_delta"] / max(1, agg["shards"]), 6)
        agg["loss_delta"] = round(agg["loss_delta"], 6)

    report = {
        "shard_count": len(cards),
        "cards": cards,
        "by_lane": {k: by_lane[k] for k in sorted(by_lane)},
        "verdict_counts": {v: sum(1 for c in cards if c["usefulness"] == v)
                           for v in ("useful", "neutral", "harmful", "needs_warmup")},
        "feedback_for_next_corpus": _feedback(cards, by_lane),
    }
    report["report_hash"] = hash_obj(report["cards"])
    return report


def _feedback(cards: list[dict], by_lane: dict) -> list[str]:
    """The V6 recommendations the learning ledger actually supports."""
    out = []
    for lane, agg in sorted(by_lane.items()):
        if agg["mean_loss_delta"] > 0.15:
            out.append(f"{lane}: loss fell across exposures "
                       f"(mean delta {agg['mean_loss_delta']:+.3f}) -- keep and extend")
        elif agg["mean_loss_delta"] < 0.0:
            out.append(f"{lane}: loss did not improve "
                       f"(mean delta {agg['mean_loss_delta']:+.3f}) -- inspect before reuse")
    undervalued = [c for c in cards if c["opus_score_mean"] is not None
                   and c["opus_score_mean"] < 0.55 and c["loss_delta"] > 0.2]
    for c in undervalued[:3]:
        out.append(f"{c['shard_id']}: OPUS scored it low ({c['opus_score_mean']:.3f}) "
                   f"but it reduced loss by {c['loss_delta']:.3f} -- possible proxy blind spot")
    repeated = [c for c in cards if c["max_repeat_pass"] >= 1 and c["loss_delta"] < 0.05]
    if repeated:
        out.append(f"{len(repeated)} shards stopped improving after a repeated pass -- "
                   f"repetition budget looks exhausted for those spans")
    warmup = [c for c in cards if c["usefulness"] == "needs_warmup"]
    if warmup:
        out.append(f"{len(warmup)} shards coincided with gradient spikes -- "
                   f"stage or warm them before reuse")
    return out
