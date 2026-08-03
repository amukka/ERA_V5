"""Stage 11 -- checkpoints that bind model state to data state.

A checkpoint here is not just weights. It stores, atomically, everything needed
to continue the run as if nothing happened:

    state.pt    model weights, optimizer state, torch/numpy/python RNG states
    meta.json   step, next_batch_id, planner state (cursors, epochs, buffers,
                OPUS state), ledger byte offsets, tokenizer hash, config hash,
                shard index hash, mixture plan hash, branch lineage

The two fields that make recovery exact are `next_batch_id` and `ledger_offsets`.
Without the first, a resumed run has to guess where the stream was. Without the
second, the ledgers keep events for batches the resumed run is about to consume
again, and the history contains duplicates.

A checkpoint without a data position is an incomplete checkpoint.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch

from .common import ensure_dir, file_sha256, hash_obj, now_iso, read_json, write_json

META = "meta.json"
STATE = "state.pt"


def checkpoint_dir(cfg, branch_id: str, step: int) -> str:
    return os.path.join(cfg.resolve("checkpoints_dir"), branch_id, f"step_{step:05d}")


def save_checkpoint(cfg, branch_id: str, step: int, model, optimizer, planner,
                    ledgers, tokenizer_hash: str, plan_hash: str, index_hash: str,
                    extra: dict | None = None, log=None) -> dict:
    out = ensure_dir(checkpoint_dir(cfg, branch_id, step))
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng_torch": torch.get_rng_state(),
        "rng_numpy": np.random.get_state(),
        "rng_python": random.getstate(),
    }, os.path.join(out, STATE))

    meta = {
        "checkpoint_version": 1,
        "branch_id": branch_id,
        "step": step,
        "next_batch_id": f"{branch_id}/step{step:05d}",
        "next_step": step,
        "created_at": now_iso(),
        "tokenizer_hash": tokenizer_hash,
        "config_hash": cfg.config_hash,
        "plan_hash": plan_hash,
        "sample_index_hash": index_hash,
        "planner_state": planner.state_dict(),
        "ledger_offsets": ledgers.offsets(),
        "state_file_sha256": file_sha256(os.path.join(out, STATE)),
        "extra": extra or {},
    }
    meta["checkpoint_id"] = hash_obj({k: v for k, v in meta.items()
                                      if k not in ("created_at", "checkpoint_id")})
    write_json(os.path.join(out, META), meta)
    if log:
        log.ok("checkpoint_saved", step=step, branch=branch_id,
               checkpoint_id=meta["checkpoint_id"][:16],
               next_batch_id=meta["next_batch_id"],
               consumption_offset=meta["ledger_offsets"]["consumption"]["bytes"],
               consumption_events=meta["ledger_offsets"]["consumption"]["count"])
    return meta


def load_meta(cfg, branch_id: str, step: int) -> dict:
    return read_json(os.path.join(checkpoint_dir(cfg, branch_id, step), META))


def verify_checkpoint(cfg, branch_id: str, step: int) -> dict:
    """Recompute the checkpoint's own hashes from the files on disk."""
    d = checkpoint_dir(cfg, branch_id, step)
    meta = read_json(os.path.join(d, META))
    state_ok = file_sha256(os.path.join(d, STATE)) == meta["state_file_sha256"]
    id_ok = hash_obj({k: v for k, v in meta.items()
                      if k not in ("created_at", "checkpoint_id")}) == meta["checkpoint_id"]
    return {"step": step, "branch_id": branch_id, "state_file_ok": state_ok,
            "checkpoint_id_ok": id_ok, "ok": bool(state_ok and id_ok),
            "checkpoint_id": meta["checkpoint_id"]}


def restore_checkpoint(cfg, branch_id: str, step: int, model, optimizer,
                       restore_rng: bool = True) -> dict:
    d = checkpoint_dir(cfg, branch_id, step)
    meta = read_json(os.path.join(d, META))
    blob = torch.load(os.path.join(d, STATE), map_location="cpu", weights_only=False)
    model.load_state_dict(blob["model"])
    if optimizer is not None:
        optimizer.load_state_dict(blob["optimizer"])
    if restore_rng:
        torch.set_rng_state(blob["rng_torch"])
        np.random.set_state(blob["rng_numpy"])
        random.setstate(blob["rng_python"])
    return meta


def list_checkpoints(cfg, branch_id: str) -> list[int]:
    root = os.path.join(cfg.resolve("checkpoints_dir"), branch_id)
    if not os.path.isdir(root):
        return []
    steps = []
    for name in os.listdir(root):
        if name.startswith("step_") and os.path.exists(os.path.join(root, name, META)):
            steps.append(int(name.split("_")[1]))
    return sorted(steps)


def latest_checkpoint_at_or_before(cfg, branch_id: str, step: int) -> int | None:
    steps = [s for s in list_checkpoints(cfg, branch_id) if s <= step]
    return max(steps) if steps else None
