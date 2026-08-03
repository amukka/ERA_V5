#!/usr/bin/env python3
"""Training Data Execution System -- the complete demonstration.

    python run_demo.py

Runs the whole path with no manual intervention:

    documents -> tokenized shards -> manifests -> mixture schedule -> packing ->
    batches -> training -> consumption ledger -> learning ledger -> checkpoint ->
    crash -> resume -> replay -> fork -> audit -> performance -> evidence

and regenerates `submission_artifacts/` from scratch each time.

The run is designed to be adversarial towards itself. It plants an evaluation
leak in the corpus without telling the scanner where it is; it deliberately
crashes mid-training; it forges a manifest to attack its own firewall; and the
evidence bundle at the end is produced by a module that re-reads the artifacts
from disk rather than trusting anything the trainer said.

Useful flags:

    --rebuild-corpus   re-download the Hugging Face sources
    --skip-tests       do not shell out to the unittest suite
    --steps N          override run.total_steps for a quick smoke run
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from pipeline.audit import run_audit                                    # noqa: E402
from pipeline.batching import BatchPlanner, SampleIndex                 # noqa: E402
from pipeline.checkpoint import (latest_checkpoint_at_or_before,        # noqa: E402
                                 list_checkpoints, load_meta,
                                 restore_checkpoint, save_checkpoint,
                                 verify_checkpoint)
from pipeline.common import (RunLog, ensure_dir, load_config,           # noqa: E402
                             set_global_seed, write_json)
from pipeline.corpus_builder import build_corpus, load_split            # noqa: E402
from pipeline.evidence import build_evidence                            # noqa: E402
from pipeline.firewall import (audit_consumption_permissions,           # noqa: E402
                               firewall_drill)
from pipeline.ledger import LedgerSet                                   # noqa: E402
from pipeline.manifest import (ShardRegistry, run_admission,            # noqa: E402
                               scan_contamination, verify_manifest)
from pipeline.mixture import MixtureSchedule, supply_check, write_plan  # noqa: E402
from pipeline.model import build_model                                  # noqa: E402
from pipeline.opus import OpusPolicy                                    # noqa: E402
from pipeline.packing import compare_policies                           # noqa: E402
from pipeline.performance import build_performance_report, write_performance  # noqa: E402
from pipeline.replay import (coverage, ledger_batches_by_step,          # noqa: E402
                             replay_by_replan, replay_from_ledger)
from pipeline.shard_builder import build_shards, persist_manifests      # noqa: E402
from pipeline.trainer import (Meter, SimulatedCrash, TrainingEngine,    # noqa: E402
                              build_validation_batch, shard_report_cards)
from tokenizer.freeze import freeze_tokenizer, load_tokenizer           # noqa: E402

POLICY_COMPARISON_SAMPLE = 2500     # samples used for the packing policy table

PHASES = ["01_corpus", "02_tokenizer", "03_shards", "04_admission", "05_firewall",
          "06_mixture", "07_index", "08_train", "09_resume", "10_replay",
          "11_fork", "12_audit", "13_learning", "14_performance", "15_tests",
          "16_evidence"]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

class Timer:
    def __init__(self) -> None:
        self.phases: dict[str, float] = {}
        self._t0 = time.perf_counter()

    def mark(self, phase: str) -> None:
        now = time.perf_counter()
        self.phases[phase] = round(now - self._t0, 4)
        self._t0 = now

    def total(self) -> float:
        return round(sum(self.phases.values()), 4)


def report_path(cfg, name: str) -> str:
    return os.path.join(ensure_dir(os.path.join(cfg.resolve("artifacts_dir"), "reports")),
                        name)


def write_report(cfg, name: str, obj) -> str:
    path = report_path(cfg, name)
    write_json(path, obj)
    return path


def fresh_artifacts(cfg) -> None:
    """Shards are written read-only, so the tree is removed rather than reused."""
    art = cfg.resolve("artifacts_dir")
    if os.path.isdir(art):
        for dirpath, _dirs, files in os.walk(art):
            for f in files:
                try:
                    os.chmod(os.path.join(dirpath, f), 0o644)
                except OSError:
                    pass
        shutil.rmtree(art)
    ensure_dir(art)


def make_model_and_optimizer(cfg, vocab_size: int, lr: float | None = None):
    model = build_model(cfg, vocab_size, cfg.run.device)
    opt = torch.optim.AdamW(model.parameters(),
                            lr=lr if lr is not None else float(cfg.train.lr),
                            weight_decay=float(cfg.train.weight_decay))
    return model, opt


# --------------------------------------------------------------------------- #
# phases
# --------------------------------------------------------------------------- #

def phase_corpus(cfg, log, args) -> dict:
    log.phase("01_corpus", "documents: three real corpora, one planted eval leak")
    manifest = build_corpus(cfg, log, rebuild=args.rebuild_corpus)
    docs = {"train": load_split(cfg, "train"),
            "validation": load_split(cfg, "validation"),
            "eval": load_split(cfg, "eval")}
    for split, d in docs.items():
        log.note("corpus_split", split=split, documents=len(d),
                 chars=sum(x["char_len"] for x in d))
    return {"corpus_manifest": manifest, "docs": docs}


def phase_tokenizer(cfg, log) -> dict:
    log.phase("02_tokenizer", "the frozen tokenizer: the root of trust for every hash")
    record = freeze_tokenizer(cfg, log)
    tok, tokenizer_hash = load_tokenizer(cfg, log)
    log.info(f"vocab={record['vocab_size']}  eos={record['eos_token_id']}  "
             f"pad={record['pad_token_id']}  files={len(record['files'])}")
    return {"tokenizer_record": record, "tok": tok, "tokenizer_hash": tokenizer_hash}


def phase_shards(cfg, log, ctx) -> dict:
    log.phase("03_shards", "documents become immutable, hashed, tokenized shards")
    manifests = build_shards(cfg, ctx["tok"], ctx["tokenizer_hash"], ctx["docs"], log)
    total_tokens = sum(m["num_tokens"] for m in manifests)
    log.info(f"{len(manifests)} shards, {total_tokens} tokens, "
             f"{sum(m['num_samples'] for m in manifests)} packable samples")
    return {"manifests": manifests}


def phase_admission(cfg, log, ctx) -> dict:
    log.phase("04_admission", "verify bytes, scan for contamination, run the gate")
    shards_root = cfg.resolve("shards_dir")
    manifests = ctx["manifests"]

    verified = [verify_manifest(m, shards_root) for m in manifests]
    bad = [d for ok, d in verified if not ok]
    log.check("manifests_validated", not bad, shards=len(manifests), failing=len(bad))

    train_m = [m for m in manifests if m["split"] == "train"]
    eval_m = [m for m in manifests if m["split"] == "eval"]
    contamination = scan_contamination(train_m, eval_m, shards_root, log)
    write_report(cfg, "contamination.json", contamination)

    admission = run_admission(manifests, ctx["tokenizer_hash"], log)
    write_report(cfg, "admission.json", admission)
    index = persist_manifests(cfg, manifests, log)

    registry = ShardRegistry(manifests, shards_root)
    log.info(f"trainable shards: {len(registry.trainable_ids())} of {len(manifests)}")
    return {"registry": registry, "contamination": contamination,
            "admission": admission, "shard_index": index}


def phase_firewall(cfg, log, ctx) -> dict:
    log.phase("05_firewall", "attack the evaluation firewall before training starts")
    ledgers = LedgerSet(cfg.resolve("ledgers_dir"), cfg.run.branch_id)
    drill = firewall_drill(ctx["registry"], ledgers, ctx["contamination"], log)
    val_batch = build_validation_batch(cfg, ctx["registry"], ctx["tokenizer_record"],
                                       log=log)
    return {"ledgers": ledgers, "firewall_drill": drill, "validation_batch": val_batch}


def phase_mixture(cfg, log, ctx) -> dict:
    log.phase("06_mixture", "compile Session 5's curriculum into per-step quotas")
    schedule = MixtureSchedule(cfg.mixture.stages, int(cfg.train.batch_size),
                               float(cfg.mixture.tolerance))
    plan = schedule.compile(int(cfg.run.total_steps))
    write_plan(cfg, plan, log)
    for stage in cfg.mixture.stages:
        log.info(f"{stage['name']:<22} steps [{stage['step_start']},{stage['step_end']}) "
                 f"shares={stage['shares']} floors={stage.get('protected_floors', {})}")
    supply = supply_check(schedule, ctx["registry"], plan, int(cfg.packing.seq_len), log)
    write_report(cfg, "supply_check.json", supply)
    for row in supply["rows"]:
        if not row["satisfied"]:
            log.note("lane_requires_repetition", lane=row["lane"],
                     repetition_factor=row["repetition_factor"],
                     remedy="repeat existing spans (recorded as repeat_pass in the ledger)")
    return {"schedule": schedule, "plan": plan, "supply": supply}


def phase_index(cfg, log, ctx) -> dict:
    log.phase("07_index", "index every trainable sample and compare packing policies")
    index = SampleIndex(ctx["registry"], ctx["tok"], log)
    write_report(cfg, "sample_index.json", index.summary())

    flat = [r for recs in index.by_lane.values() for r in recs]
    flat.sort(key=lambda r: r["sample_id"])
    step = max(1, len(flat) // POLICY_COMPARISON_SAMPLE)
    subset = flat[::step][:POLICY_COMPARISON_SAMPLE]
    comparison = compare_policies(subset, int(cfg.packing.seq_len),
                                  bool(cfg.packing.eos_between_samples))
    comparison["sampled_from"] = len(flat)
    write_report(cfg, "packing_policies.json", comparison)
    for row in comparison["rows"]:
        log.note("packing_policy", policy=row["policy"], bins=row["bins"],
                 utilisation=row["utilisation"], padding_pct=row["padding_pct"],
                 open_bins=row["open_bins"])
    chosen = next(r for r in comparison["rows"] if r["policy"] == cfg.packing.policy)
    log.ok("packing_policy_selected", policy=cfg.packing.policy,
           utilisation=chosen["utilisation"], bins=chosen["bins"],
           open_bins=f"bounded at {cfg.packing.max_open_bins}",
           best_unbounded=comparison["best_policy"])
    if comparison["best_policy"] != cfg.packing.policy:
        top = comparison["rows"][0]
        log.info(f"note: {top['policy']} reaches {top['utilisation']:.6f} against "
                 f"{chosen['utilisation']:.6f} here, but it keeps every open bin until "
                 f"the dataset ends. The configured policy caps open bins at "
                 f"{cfg.packing.max_open_bins} so batches can be emitted while the "
                 f"stream is still arriving -- "
                 f"{(top['utilisation'] - chosen['utilisation']) * 100:.2f}pp of "
                 f"utilisation is the price of bounded memory.")
    return {"index": index, "policy_comparison": comparison}


def _checkpoint_fn(cfg, log, ctx, engine, saved: set):
    def fn(step: int) -> None:
        if step in saved:
            return
        meta = save_checkpoint(
            cfg, engine.branch_id, step, engine.model, engine.optimizer,
            engine.planner, engine.ledgers, engine.tokenizer_hash,
            ctx["plan"]["plan_hash"], ctx["index"].index_hash,
            extra={"model_age_tokens": engine.model_age_tokens,
                   "run_id": engine.run_id,
                   "lr": engine.lr,
                   "parent_branch": ctx.get("parent_branch"),
                   "shard_index_hash": ctx["shard_index"]["index_hash"]},
            log=log)
        engine.checkpoint_id = meta["checkpoint_id"]
        saved.add(step)
    return fn


def _validate_fn(cfg, log, ctx, engine):
    def fn(step: int) -> None:
        rec = engine.validate(ctx["validation_batch"], step)
        log.ok("validation_pass", step=step, loss=rec["loss"],
               perplexity=rec["perplexity"], gradient_bearing=False,
               shards=len(rec["shard_ids"]))
    return fn


def phase_train(cfg, log, ctx) -> dict:
    log.phase("08_train", "consume the stream, record it, and crash on purpose")
    set_global_seed(int(cfg.run.seed))
    model, opt = make_model_and_optimizer(cfg, ctx["tokenizer_record"]["len_tokenizer"])
    log.info(f"model parameters: {model.num_parameters():,}  "
             f"seq_len={cfg.packing.seq_len}  bins/step={cfg.train.batch_size}")

    planner = BatchPlanner(cfg, ctx["schedule"], ctx["index"], OpusPolicy(cfg),
                           ctx["registry"], ctx["tokenizer_record"],
                           cfg.run.branch_id, int(cfg.run.seed), log=log)
    engine = TrainingEngine(cfg, log, cfg.run.run_id, cfg.run.branch_id, planner,
                            ctx["schedule"], ctx["registry"], ctx["ledgers"], model,
                            opt, ctx["tok"], ctx["tokenizer_record"],
                            cfg.run.device, Meter())

    saved: set[int] = set()
    crash_at = int(cfg.recovery.crash_at_step)
    crashed = False
    try:
        engine.run_steps(0, int(cfg.run.total_steps), crash_at=crash_at,
                         checkpoint_fn=_checkpoint_fn(cfg, log, ctx, engine, saved),
                         validate_fn=_validate_fn(cfg, log, ctx, engine))
    except SimulatedCrash as exc:
        crashed = True
        log.info(f"    !! {exc}")
        log.info("    the process would now be gone; everything below is recovery")

    if not crashed:
        raise RuntimeError("the demonstration requires the deliberate crash to fire")

    pre_crash = ctx["ledgers"].consumption.read_all()
    return {"engine": engine, "model": model, "optimizer": opt, "saved_steps": saved,
            "crash_step": crash_at, "pre_crash_consumption": pre_crash,
            "pre_crash_batches": ledger_batches_by_step(pre_crash, cfg.run.branch_id),
            "main_meter": engine.meter}


def phase_resume(cfg, log, ctx) -> dict:
    log.phase("09_resume", "restore the checkpoint, roll the ledgers back, continue")
    branch = cfg.run.branch_id
    crash_step = ctx["crash_step"]
    restore_step = latest_checkpoint_at_or_before(cfg, branch, crash_step)
    if restore_step is None:
        raise RuntimeError("no checkpoint at or before the crash")

    for step in list_checkpoints(cfg, branch):
        v = verify_checkpoint(cfg, branch, step)
        log.check("checkpoint_verified", v["ok"], step=step,
                  checkpoint_id=v["checkpoint_id"][:16],
                  state_file_ok=v["state_file_ok"])

    expected = ctx["pre_crash_batches"].get(restore_step, {})
    log.info(f"the crashed run consumed steps "
             f"{min(ctx['pre_crash_batches'])}..{max(ctx['pre_crash_batches'])}; "
             f"the last checkpoint is step {restore_step}")
    log.ok("expected_next_batch_identified", step=restore_step,
           batch_id=expected.get("batch_id"),
           batch_hash=str(expected.get("batch_hash"))[:16],
           source="consumption ledger, before truncation")

    # -- restore model and data state ---------------------------------------- #
    model, opt = make_model_and_optimizer(cfg, ctx["tokenizer_record"]["len_tokenizer"])
    meta = restore_checkpoint(cfg, branch, restore_step, model, opt)
    truncations = ctx["ledgers"].truncate_to(meta["ledger_offsets"])
    discarded = sum(t["events_discarded"] for t in truncations)
    for t in truncations:
        log.ok("ledger_rolled_back", ledger=t["ledger"],
               events_discarded=t["events_discarded"],
               bytes_discarded=t["bytes_discarded"],
               events_now=t["after"]["count"])
    log.ok("ledgers_truncated_to_checkpoint", total_events_discarded=discarded,
           note="events for steps the resumed run will redo are physically gone")

    # -- predict the next batch before consuming it -------------------------- #
    peek = BatchPlanner(cfg, ctx["schedule"], ctx["index"], OpusPolicy(cfg),
                        ctx["registry"], ctx["tokenizer_record"], branch,
                        int(cfg.run.seed))
    peek.load_state_dict(meta["planner_state"])
    predicted = peek.plan_step(restore_step)
    id_match = predicted["batch_id"] == expected.get("batch_id")
    hash_match = predicted["batch_hash"] == expected.get("batch_hash")
    log.check("resume_next_batch_matched", id_match and hash_match,
              expected_batch_id=expected.get("batch_id"),
              resumed_batch_id=predicted["batch_id"],
              expected_hash=str(expected.get("batch_hash"))[:16],
              resumed_hash=predicted["batch_hash"][:16],
              bins=predicted["n_bins"])
    log.check("resume_bin_hashes_matched",
              list(expected.get("bin_hashes", [])) ==
              [b["content_hash"] for b in predicted["bins"]],
              bins=predicted["n_bins"])

    # -- continue the run ---------------------------------------------------- #
    planner = BatchPlanner(cfg, ctx["schedule"], ctx["index"], OpusPolicy(cfg),
                           ctx["registry"], ctx["tokenizer_record"], branch,
                           int(cfg.run.seed), log=log)
    planner.load_state_dict(meta["planner_state"])
    engine = TrainingEngine(cfg, log, cfg.run.run_id, branch, planner, ctx["schedule"],
                            ctx["registry"], ctx["ledgers"], model, opt, ctx["tok"],
                            ctx["tokenizer_record"], cfg.run.device, ctx["main_meter"])
    engine.checkpoint_id = meta["checkpoint_id"]
    engine.model_age_tokens = int(meta["extra"].get("model_age_tokens", 0))

    saved = set(ctx["saved_steps"])
    t0 = time.perf_counter()
    engine.run_steps(restore_step, int(cfg.run.total_steps),
                     checkpoint_fn=_checkpoint_fn(cfg, log, ctx, engine, saved),
                     validate_fn=_validate_fn(cfg, log, ctx, engine))
    resume_seconds = round(time.perf_counter() - t0, 4)
    log.ok("run_resumed", from_step=restore_step, to_step=int(cfg.run.total_steps),
           steps=int(cfg.run.total_steps) - restore_step, seconds=resume_seconds)

    consumption = ctx["ledgers"].consumption.read_all()
    cov = coverage(consumption, branch, int(cfg.run.total_steps), log)
    actual = ledger_batches_by_step(consumption, branch)
    log.check("resumed_batch_matches_pre_crash_record",
              actual[restore_step]["batch_hash"] == expected.get("batch_hash"),
              step=restore_step,
              hash=actual[restore_step]["batch_hash"][:16])

    # The steps between the checkpoint and the crash are executed twice: once by
    # the run that died, once by the run that resumed. The model state from the
    # first attempt is gone -- it only ever existed in RAM -- but the *data* is
    # reconstructible, so the second attempt must consume byte-identical batches.
    # Anything else means the resumed run trained on a different stream than the
    # one the ledger says it did.
    redone = []
    for step in range(restore_step, ctx["crash_step"]):
        before = ctx["pre_crash_batches"].get(step, {})
        after = actual.get(step, {})
        redone.append({
            "step": step,
            "pre_crash_batch_hash": before.get("batch_hash"),
            "resumed_batch_hash": after.get("batch_hash"),
            "match": before.get("batch_hash") == after.get("batch_hash"),
            "bin_hashes_match": list(before.get("bin_hashes", [])) ==
                                list(after.get("bin_hashes", [])),
        })
    all_redone_match = bool(redone) and all(r["match"] and r["bin_hashes_match"]
                                            for r in redone)
    log.check("redone_steps_are_byte_identical", all_redone_match,
              steps=f"[{restore_step},{ctx['crash_step']})",
              count=len(redone),
              mismatches=[r["step"] for r in redone if not r["match"]],
              note="the crashed attempt and the resumed attempt consumed the same data")

    chain = ctx["ledgers"].verify_all()
    log.check("ledger_hash_chains_verified", chain["ok"],
              **{k: v["events"] for k, v in chain["ledgers"].items() if v["ok"]})

    return {"resume_engine": engine, "restore_step": restore_step,
            "resume_meta": meta, "coverage": cov, "resume_seconds": resume_seconds,
            "crash_record": {
                "step": ctx["crash_step"],
                "restored_from_step": restore_step,
                "restored_checkpoint_id": meta["checkpoint_id"],
                "expected_next_batch_id": expected.get("batch_id"),
                "resumed_next_batch_id": predicted["batch_id"],
                "expected_next_batch_hash": expected.get("batch_hash"),
                "resumed_next_batch_hash": predicted["batch_hash"],
                "batch_id_match": id_match,
                "batch_hash_match": hash_match,
                "events_discarded": discarded,
                "truncations": truncations,
                "steps_redone": list(range(restore_step, ctx["crash_step"])),
                "redone_steps_detail": redone,
                "redone_steps_byte_identical": all_redone_match,
                "why_not_resume_at_the_crash_step":
                    "the data position at step %d is reconstructible, but the model "
                    "and optimizer state is not -- it existed only in memory and the "
                    "seven updates from steps %d..%d died with the process. Resuming "
                    "the data at %d against weights from %d would leave those steps "
                    "untrained. The last point where model state and data state were "
                    "saved together is step %d, so both roll back there."
                    % (ctx["crash_step"], restore_step, ctx["crash_step"] - 1,
                       ctx["crash_step"], restore_step, restore_step),
            }}


def phase_replay(cfg, log, ctx) -> dict:
    log.phase("10_replay", "rebuild a historical interval two independent ways")
    branch = cfg.run.branch_id
    lo, hi = (int(x) for x in cfg.recovery.replay_interval)
    consumption = ctx["ledgers"].consumption.read_all()

    t0 = time.perf_counter()
    from_ledger = replay_from_ledger(cfg, ctx["registry"], consumption,
                                     ctx["tokenizer_record"], (lo, hi), branch, log)
    ledger_seconds = round(time.perf_counter() - t0, 4)
    write_report(cfg, "replay_from_ledger.json", from_ledger)

    base = latest_checkpoint_at_or_before(cfg, branch, lo)
    meta = load_meta(cfg, branch, base)
    t0 = time.perf_counter()
    replan = replay_by_replan(cfg, ctx["schedule"], ctx["index"], ctx["registry"],
                              ctx["tokenizer_record"], branch, int(cfg.run.seed),
                              meta["planner_state"], base, (lo, hi),
                              ledger_batches_by_step(consumption, branch), log)
    replan_seconds = round(time.perf_counter() - t0, 4)
    write_report(cfg, "replay_replan.json", replan)

    log.info(f"replay [{lo},{hi}): {from_ledger['microbatches_replayed']} microbatches "
             f"rebuilt from ledger spans, {replan['steps_replayed']} steps re-planned "
             f"from the step-{base} checkpoint -- all hashes identical")
    return {"replay_ledger": from_ledger, "replay_replan": replan,
            "replay_seconds": {"from_ledger": ledger_seconds, "replan": replan_seconds}}


def phase_fork(cfg, log, ctx) -> dict:
    log.phase("11_fork", "branch from an old checkpoint into a new data stream")
    parent = cfg.run.branch_id
    fk = dict(cfg.recovery.fork)
    branch = fk["branch_id"]
    from_step = int(cfg.recovery.fork_from_step)
    steps = int(cfg.recovery.fork_steps)

    stage = {"name": "fork_anneal", "step_start": from_step,
             "step_end": from_step + steps, "shares": dict(fk["shares"]),
             "protected_floors": dict(fk.get("protected_floors") or {})}
    schedule = MixtureSchedule([stage], int(cfg.train.batch_size),
                               float(cfg.mixture.tolerance))
    log.info(f"fork mixture: {stage['shares']} floors={stage['protected_floors']} "
             f"lr={fk['lr']} (parent lr {cfg.train.lr})")

    model, opt = make_model_and_optimizer(cfg, ctx["tokenizer_record"]["len_tokenizer"],
                                          lr=float(fk["lr"]))
    meta = restore_checkpoint(cfg, parent, from_step, model, opt)
    # the fork inherits the optimizer moments but not the parent's learning rate
    for group in opt.param_groups:
        group["lr"] = float(fk["lr"])
    parent_ckpt = load_meta(cfg, parent, from_step)

    ledgers = LedgerSet(cfg.resolve("ledgers_dir"), branch)
    ledgers.consumption.append({
        "kind": "branch_created", "branch_id": branch, "parent_branch_id": parent,
        "parent_checkpoint_id": parent_ckpt["checkpoint_id"],
        "fork_from_step": from_step, "reason": "anneal-style mixture experiment",
        "mixture": stage["shares"], "protected_floors": stage["protected_floors"],
        "lr": float(fk["lr"]), "parent_lr": float(cfg.train.lr),
        "config_hash": cfg.config_hash})
    log.ok("branch_forked", branch=branch, parent=parent, from_step=from_step,
           parent_checkpoint=parent_ckpt["checkpoint_id"][:16])

    planner = BatchPlanner(cfg, schedule, ctx["index"], OpusPolicy(cfg), ctx["registry"],
                           ctx["tokenizer_record"], branch, int(cfg.run.seed), log=log)
    planner.load_state_dict(meta["planner_state"])
    fork_ctx = dict(ctx)
    fork_ctx["plan"] = {"plan_hash": schedule.compile(from_step + steps)["plan_hash"]}
    fork_ctx["parent_branch"] = parent

    engine = TrainingEngine(cfg, log, cfg.run.run_id, branch, planner, schedule,
                            ctx["registry"], ledgers, model, opt, ctx["tok"],
                            ctx["tokenizer_record"], cfg.run.device, Meter())
    engine.checkpoint_id = parent_ckpt["checkpoint_id"]
    engine.model_age_tokens = int(parent_ckpt["extra"].get("model_age_tokens", 0))
    engine.lr = float(fk["lr"])
    engine.run_steps(from_step, from_step + steps,
                     checkpoint_fn=_checkpoint_fn(cfg, log, fork_ctx, engine, set()))

    fork_batches = ledger_batches_by_step(ledgers.consumption.read_all(), branch)
    main_batches = ledger_batches_by_step(ctx["ledgers"].consumption.read_all(), parent)
    divergence = None
    same = []
    for s in range(from_step, from_step + steps):
        if s in fork_batches and s in main_batches:
            if fork_batches[s]["batch_hash"] != main_batches[s]["batch_hash"]:
                divergence = divergence if divergence is not None else s
            else:
                same.append(s)
    log.check("fork_stream_diverges_from_parent", divergence == from_step,
              divergence_step=divergence, identical_steps=same,
              note="a fork must not silently receive the parent's data stream")

    record = {
        "branch_id": branch, "parent_branch_id": parent, "from_step": from_step,
        "steps": steps, "parent_checkpoint_id": parent_ckpt["checkpoint_id"],
        "restored_checkpoint_id": meta["checkpoint_id"],
        "divergence_step": divergence, "diverged": divergence is not None,
        "divergence_recorded": True,
        "identical_steps": same,
        "mixture": stage["shares"], "protected_floors": stage["protected_floors"],
        "lr": float(fk["lr"]),
        "fork_batch_hashes": {str(s): fork_batches[s]["batch_hash"]
                              for s in sorted(fork_batches)},
        "parent_batch_hashes": {str(s): main_batches[s]["batch_hash"]
                                for s in range(from_step, from_step + steps)
                                if s in main_batches},
        "final_loss": engine.step_metrics[-1]["loss"] if engine.step_metrics else None,
        "chain_ok": ledgers.verify_all()["ok"],
    }
    write_report(cfg, "fork.json", record)
    return {"fork": record, "fork_ledgers": ledgers, "fork_engine": engine}


def phase_audit(cfg, log, ctx) -> dict:
    log.phase("12_audit", "query the ledgers: what trained this, and what preceded it")
    branch = cfg.run.branch_id
    consumption = ctx["ledgers"].consumption.read_all()
    learning = ctx["ledgers"].learning.read_all()
    opus_events = ctx["ledgers"].opus.read_all()

    lo, hi = (int(x) for x in cfg.recovery.replay_interval)
    audit = run_audit(cfg, consumption, learning, opus_events, branch,
                      (lo, hi), int(cfg.recovery.crash_at_step) // 10 * 10, log)
    write_report(cfg, "audit.json", audit)

    perm = audit_consumption_permissions(consumption, ctx["registry"], log)
    write_report(cfg, "firewall.json", {"drill": ctx["firewall_drill"],
                                        "consumption_audit": perm})
    return {"audit": audit, "consumption_audit": perm,
            "consumption": consumption, "learning": learning, "opus_events": opus_events}


def phase_learning(cfg, log, ctx) -> dict:
    log.phase("13_learning", "turn the run into a signal for the next corpus")
    branch = cfg.run.branch_id
    cards = shard_report_cards(ctx["learning"], ctx["opus_events"], branch)
    write_report(cfg, "shard_report_cards.json", cards)
    log.ok("learning_ledger_summarised", shards=cards["shard_count"],
           verdicts=cards["verdict_counts"],
           lanes=len(cards["by_lane"]))
    for line in cards["feedback_for_next_corpus"]:
        log.info(f"V6 feedback: {line}")

    opus_summary = OpusPolicy(cfg).summary(ctx["opus_events"])
    write_report(cfg, "opus_summary.json", opus_summary)
    log.ok("opus_audit_trail", candidates=opus_summary["candidates"],
           by_status=opus_summary["by_status"],
           overrides=opus_summary["protected_floor_overrides"],
           mean_score=opus_summary["mean_score"])

    actual = [ctx.get("actual_by_step", {}).get(s, {})
              for s in range(int(cfg.run.total_steps))]
    compliance = ctx["schedule"].compliance(actual, ctx["plan"])
    write_report(cfg, "mixture_compliance.json", compliance)
    log.check("mixture_compliance", compliance["within_tolerance"],
              max_deviation=compliance["max_deviation"],
              tolerance=compliance["tolerance"])
    log.check("protected_floors_respected", compliance["floors_respected"])
    for row in compliance["rows"]:
        log.note("lane_share", stage=row["stage"], lane=row["lane"],
                 planned=row["planned_share"], actual=row["actual_share"],
                 floor=row["floor"], ok=row["floor_respected"])

    trace_events = ctx["ledgers"].token_trace.read_all()
    hardest = sorted((h for t in trace_events for h in t["hard_tokens"]),
                     key=lambda h: -h["loss"])[:20]
    write_report(cfg, "token_perplexity.json", {
        "trace_events": len(trace_events),
        "traced_tokens": sum(t["traced_tokens"] for t in trace_events),
        "traced_steps": sorted({t["global_step"] for t in trace_events}),
        "mean_loss_by_step": {str(t["global_step"]): t["mean_loss"]
                              for t in trace_events},
        "hardest_tokens": hardest})
    log.ok("token_trace_written", events=len(trace_events),
           tokens=sum(t["traced_tokens"] for t in trace_events),
           steps=sorted({t["global_step"] for t in trace_events}))
    if hardest:
        log.info("hardest tokens: " + ", ".join(
            f"{h['preview']!r}(ppl={h['perplexity']:.0f}, {h['shard_id']})"
            for h in hardest[:5]))
    return {"cards": cards, "opus_summary": opus_summary, "compliance": compliance}


def phase_performance(cfg, log, ctx, timer: Timer) -> dict:
    log.phase("14_performance", "useful loss-bearing tokens per second, not raw tokens")
    main_meter = ctx["main_meter"].report(ctx["index"].cache.stats())
    fork_meter = ctx["fork_engine"].meter.report(ctx["index"].cache.stats())
    report = build_performance_report(
        cfg, {"main": main_meter, "fork": fork_meter}, ctx["index"].cache.stats(),
        ctx["consumption"], ctx["opus_events"], ctx["policy_comparison"],
        timer.phases,
        {"resume_seconds": ctx["resume_seconds"], **ctx["replay_seconds"]},
        ctx["fork_engine"].model.num_parameters(), log)
    write_performance(cfg, report)

    recovery = {
        "checkpoints": [{**verify_checkpoint(cfg, cfg.run.branch_id, s),
                         "has_ledger_offsets": bool(
                             load_meta(cfg, cfg.run.branch_id, s)["ledger_offsets"]),
                         "has_planner_state": bool(
                             load_meta(cfg, cfg.run.branch_id, s)["planner_state"])}
                        for s in list_checkpoints(cfg, cfg.run.branch_id)],
        "crash": ctx["crash_record"],
        "coverage": ctx["coverage"],
        "replay_ledger": {k: v for k, v in ctx["replay_ledger"].items() if k != "rows"},
        "replay_replan": {k: v for k, v in ctx["replay_replan"].items() if k != "rows"},
        "fork": ctx["fork"],
        "resume_seconds": ctx["resume_seconds"],
    }
    recovery["replay_ledger"]["rows_file"] = "reports/replay_from_ledger.json"
    recovery["replay_replan"]["rows_file"] = "reports/replay_replan.json"
    write_report(cfg, "recovery.json", recovery)
    return {"performance": report, "recovery": recovery}


def phase_tests(cfg, log, args) -> dict:
    log.phase("15_tests", "run the invariant suite against the artifacts just produced")
    if args.skip_tests:
        log.note("tests_skipped", reason="--skip-tests")
        return {"tests": {"total": 0, "failed": 0, "command": None, "skipped": True}}
    cmd = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-t", ".", "-v"]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    tail = proc.stderr.strip().splitlines()
    total = failed = 0
    for line in tail:
        if line.startswith("Ran ") and " test" in line:
            total = int(line.split()[1])
        if line.startswith("FAILED"):
            failed = line.count("failures=") and int(
                line.split("failures=")[1].split(",")[0].rstrip(")")) or 1
    if proc.returncode != 0 and failed == 0:
        failed = max(1, failed)
    with open(os.path.join(cfg.resolve("artifacts_dir"), "tests.log"), "w",
              encoding="utf-8") as fh:
        fh.write(proc.stdout + "\n" + proc.stderr)
    log.check("automated_tests_passed", proc.returncode == 0,
              tests=total, failed=failed, command=" ".join(cmd[1:]))
    for line in tail[-12:]:
        log.info(line)
    return {"tests": {"total": total, "failed": failed, "command": " ".join(cmd),
                      "returncode": proc.returncode}}


def phase_evidence(cfg, log, ctx) -> dict:
    log.phase("16_evidence", "regenerate every claim from the artifacts on disk")
    art = cfg.resolve("artifacts_dir")
    files = sum(len(fs) for _, _, fs in os.walk(art))
    bundle = build_evidence(cfg, log, {
        "expected_phases": PHASES,
        "tokenizer_record": ctx["tokenizer_record"],
        "corpus_manifest": ctx["corpus_manifest"],
        "tests": ctx["tests"],
        "artifact_file_count": files,
    })
    return {"evidence": bundle}


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(ROOT, "config", "config.yaml"))
    ap.add_argument("--rebuild-corpus", action="store_true")
    ap.add_argument("--skip-tests", action="store_true")
    ap.add_argument("--steps", type=int, default=None,
                    help="override run.total_steps (smoke runs)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.steps:
        cfg["run"]["total_steps"] = args.steps
        cfg["mixture"]["stages"][-1]["step_end"] = args.steps
        cfg["recovery"]["crash_at_step"] = min(cfg["recovery"]["crash_at_step"],
                                               args.steps - 2)
    fresh_artifacts(cfg)

    art = cfg.resolve("artifacts_dir")
    log = RunLog(os.path.join(art, "run.log"), os.path.join(art, "run_events.jsonl"),
                 echo=not args.quiet)
    timer = Timer()

    log.line("=" * 78)
    log.line("TRAINING DATA EXECUTION SYSTEM -- Session 6")
    log.line("=" * 78)
    log.line(f"run_id      {cfg.run.run_id}")
    log.line(f"config      {os.path.relpath(cfg.path, ROOT)}  sha256={cfg.config_hash[:16]}")
    log.line(f"seed        {cfg.run.seed}   device={cfg.run.device}   "
             f"steps={cfg.run.total_steps}")
    log.line(f"python      {sys.version.split()[0]}   torch={torch.__version__}   "
             f"numpy={np.__version__}")

    ctx: dict = {}
    ctx.update(phase_corpus(cfg, log, args));          timer.mark("01_corpus")
    ctx.update(phase_tokenizer(cfg, log));              timer.mark("02_tokenizer")
    ctx.update(phase_shards(cfg, log, ctx));            timer.mark("03_shards")
    ctx.update(phase_admission(cfg, log, ctx));         timer.mark("04_admission")
    ctx.update(phase_firewall(cfg, log, ctx));          timer.mark("05_firewall")
    ctx.update(phase_mixture(cfg, log, ctx));           timer.mark("06_mixture")
    ctx.update(phase_index(cfg, log, ctx));             timer.mark("07_index")
    ctx.update(phase_train(cfg, log, ctx));             timer.mark("08_train")
    ctx.update(phase_resume(cfg, log, ctx));            timer.mark("09_resume")
    ctx.update(phase_replay(cfg, log, ctx));            timer.mark("10_replay")
    ctx.update(phase_fork(cfg, log, ctx));              timer.mark("11_fork")
    ctx.update(phase_audit(cfg, log, ctx));             timer.mark("12_audit")

    ctx["actual_by_step"] = {ev["global_step"]: ev["actual_bins"]
                             for ev in ctx["consumption"] if ev.get("kind") == "batch"}
    ctx.update(phase_learning(cfg, log, ctx));          timer.mark("13_learning")
    ctx.update(phase_performance(cfg, log, ctx, timer)); timer.mark("14_performance")
    ctx.update(phase_tests(cfg, log, args));            timer.mark("15_tests")
    ctx.update(phase_evidence(cfg, log, ctx));          timer.mark("16_evidence")

    write_json(os.path.join(art, "phase_timings.json"),
               {"phases": timer.phases, "total_seconds": timer.total()})

    counts = log.counts()
    summary = ctx["evidence"]["summary"]
    log.phase("done", "summary")
    log.line(f"    events        {counts['PASS']} PASS   {counts['FAIL']} FAIL   "
             f"{counts['INFO']} INFO")
    log.line(f"    requirements  {summary['passed']}/{summary['requirements']} passed  "
             f"({summary['checks_passed']}/{summary['checks']} checks)")
    log.line(f"    points        {summary['points_claimed']}/{summary['points_total']}")
    log.line(f"    wall clock    {timer.total()}s")
    log.line(f"    artifacts     {os.path.relpath(art, ROOT)}/")
    log.line("")
    for name in ("run.log", "evidence.json", "evidence.md", "performance.json"):
        log.line(f"      {os.path.relpath(art, ROOT)}/{name}")
    for name in ("manifests", "ledgers", "checkpoints", "shards", "reports"):
        n = sum(len(fs) for _, _, fs in os.walk(os.path.join(art, name)))
        log.line(f"      {os.path.relpath(art, ROOT)}/{name}/  ({n} files)")

    ok = summary["all_pass"] and counts["FAIL"] == 0
    log.line("")
    log.line("RESULT: " + ("ALL REQUIREMENTS PASS" if ok else "FAILURES PRESENT"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
