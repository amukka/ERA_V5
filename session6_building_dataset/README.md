# Session 6 — Training Data Execution System

A small but complete training data system for V5. It takes real documents from
Hugging Face and carries them all the way to an optimizer step, recording enough
along the way to answer four questions afterwards:

- **What did the run consume?** — the consumption ledger, with token spans.
- **Why did it consume that?** — the OPUS audit trail, including the rejections.
- **What did the model learn from it?** — the learning ledger and per-token trace.
- **Can the run be reconstructed?** — checkpoints bound to ledger offsets, plus
  replay and fork.

The goal is not scale. The goal is that every claim above is backed by an
artifact a marker can open, and by a check that recomputes it.

```
documents → tokenized shards → manifests → mixture schedule → packing →
batches → training → consumption ledger → learning ledger → checkpoint →
crash → resume → replay → fork → audit → performance → evidence
```

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_demo.py
```

One command, no manual intervention. It wipes and regenerates
`submission_artifacts/` end to end, including the deliberate crash, the resume,
the replay, the fork and the evidence bundle. First run downloads three Hugging
Face corpora and the GPT-2 tokenizer (~20 s of network); later runs reuse the
cached corpus. The demonstration itself takes about 20 seconds on CPU.

```bash
python run_demo.py --steps 24        # shorter run
python run_demo.py --rebuild-corpus  # re-download the sources
python -m unittest discover -s tests # the invariant suite on its own
```

## What comes out

```
submission_artifacts/
  run.log             every event, in order, with [PASS]/[FAIL] markers
  run_events.jsonl    the same events, machine-readable
  evidence.json       every requirement, recomputed from artifacts
  evidence.md         the human-readable summary table
  performance.json    throughput and packing efficiency
  tests.log           the invariant suite's output
  phase_timings.json  wall clock per phase
  manifests/          one per shard, plus shard_index.json and mixture_plan.json
  shards/             immutable uint16 token blobs, chmod 444
  ledgers/main/       consumption, learning, opus, firewall, token_trace
  ledgers/fork_b/     the forked branch's own ledgers
  checkpoints/        model + optimizer + RNG + planner state + ledger offsets
  reports/            contamination, admission, firewall, replay, recovery,
                      audit, mixture compliance, shard report cards, ...
```

## Architecture

Each module is one stage, and each has a header comment explaining the design
choice it embodies. Read them in this order:

| Module | Stage |
| --- | --- |
| `pipeline/corpus_builder.py` | documents, with provenance, splits and a planted leak |
| `tokenizer/freeze.py` | freeze the tokenizer, hash it, refuse to run if it moves |
| `pipeline/shard_builder.py` | documents → immutable hashed token shards |
| `pipeline/manifest.py` | manifests, contamination scan, admission gate, registry |
| `pipeline/mixture.py` | curriculum stages → integer per-step lane quotas |
| `pipeline/opus.py` | the selection policy inside the data path |
| `pipeline/packing.py` | bins, loss masks, attention masks, position ids |
| `pipeline/batching.py` | the deterministic batch planner |
| `pipeline/ledger.py` | append-only, hash-chained ledgers with byte offsets |
| `pipeline/model.py` | a small LM that honours segment-aware attention |
| `pipeline/trainer.py` | the loop, and the two ledgers it writes |
| `pipeline/checkpoint.py` | checkpoints that bind model state to data state |
| `pipeline/replay.py` | rebuild a historical interval, two independent ways |
| `pipeline/firewall.py` | attack the evaluation firewall |
| `pipeline/audit.py` | query the ledgers after the fact |
| `pipeline/performance.py` | useful loss-bearing tokens per second |
| `pipeline/evidence.py` | re-read everything from disk and grade it |

`config/config.yaml` is the single source of truth. Its SHA256 is stamped into
every checkpoint and into `evidence.json`.

## Design decisions

### Identity is always a hash of bytes actually written

Never a name, never a timestamp, never an in-memory id. A shard is its
`content_hash`; a packed bin is the SHA256 of the four arrays that reach the
model; a manifest hashes itself with its own hash field removed. This is what
lets `evidence.py` recompute every claim from disk rather than believing the
trainer.

There is exactly one byte encoding for hashing a batch array
(`packing.array_bytes`, int64), used by the trainer, the replayer and the
evidence builder alike, so the three cannot silently disagree. It is int64
rather than uint32 because `segment_ids` uses `-1` for padding, and a hash that
cannot represent padding cannot detect it.

### Nothing random is random

Anything that looks like sampling is a keyed hash, not a stateful RNG:
`stable_permutation(n, seed, branch, lane, epoch)` gives the same order in any
process on any machine. There is no `random.shuffle` in the data path. This is
why a resumed run lands on exactly the batch the crashed run was about to take,
and why replanning steps 5–10 six phases later reproduces the same hashes.

### OPUS never reads model state

Every OPUS feature is derived from content — utilisation, MinHash novelty,
length fit, lane affinity. A score that depended on the live model could not be
recomputed during replay without re-running training, and the replay guarantee
would collapse. That constraint is the reason replay-by-replanning works with a
model whose weights differ from the ones that were live at the time.

### The proxy is biased, and the floor is what catches it

The quality screen rejects spans that are mostly punctuation. Measured against
this corpus it rejects **~79% of the code lane and ~0% of prose**, because code
*is* punctuation. That is proxy bias against a capability, left in deliberately:
the `code` lane carries a protected floor, the floor rescues the lane the
selector would otherwise erase, and every rescue is recorded as
`protected_floor_override(was:symbol_heavy_content)` in the OPUS ledger.

For the floor to have anything to do, the draw budget is proportional to what a
lane owes (`draws_per_bin`). A real loader cannot scan the corpus to fill one
batch, and without that budget a lane with an 80% rejection rate would simply
keep drawing until it got lucky, and no floor would ever fire.

### Contamination is a run length, not a count

Counting matching n-grams does not work. A 100k-token shard shares thousands of
12-grams with any corpus in the same language, and a threshold low enough to
catch a leak condemns everything. Verbatim reproduction has a different shape:
one long unbroken run of matching windows. The scanner flags a *document* whose
longest run reaches 48 windows.

Nothing tells the scanner where the planted leak is. It finds it as a
**2801-window verbatim run**, against a longest run of 45 in the shards it
leaves clean. It also finds genuine cross-split duplicate functions in
`code_search_net` that nobody planted.

### Crash recovery is truncation, not hope

A checkpoint stores model, optimizer, RNG, planner state (cursors, epochs,
buffers, OPUS state), *and* the ledger byte offsets. On resume the ledgers are
physically truncated back to those offsets, so events for steps the resumed run
is about to redo are gone. "No skipped and no repeated batches" is then a
checkable property of the file, not a hope: `replay.coverage` asserts every step
0…59 appears exactly once, in order.

### Replay is tested two ways, because they fail differently

`replay_from_ledger` rebuilds each bin from the ledger's token spans plus the
sealed shard bytes — proving the ledger is a *sufficient* description of the
stream. `replay_by_replan` restores a checkpoint and lets the planner produce
the interval again — proving the *planner* is deterministic. A planner that
consults the clock passes the first and fails the second; a ledger that omits a
span fails the first and passes the second. The assignment needs both.

### Packing: bounded memory costs 0.4pp of utilisation

The policy table in `reports/packing_policies.json` ranks `first_fit_decreasing`
slightly above the configured `best_fit_decreasing` (0.9976 vs 0.9936). That is
not an argument for switching. The unbounded policies keep every partially
filled bin in memory until the dataset ends; the configured policy caps open
bins at 8 so batches can be emitted while the stream is still arriving. The
table reports an `open_bins` column precisely so the comparison is not read as
apples-to-apples. Against `pad_only` — the only other streamable policy — the
chosen policy saves 37 bins on the same 2500 samples.

## Results from the committed run

| | |
| --- | --- |
| Requirements passed | **9/9** (58/58 checks), 1000/1000 points claimed |
| Automated tests | **74 passed** (2 skip during the run — they check the evidence bundle, which is written after the tests; re-run `python -m unittest discover -s tests` afterwards and all 74 execute) |
| Log events | 117 PASS, 0 FAIL |
| Shards | 31 built, 17 admitted, 14 blocked |
| Contamination | 8 shards / 15 documents blocked; planted leak found as a 2801-window verbatim run |
| Firewall | 15 attack attempts, 0 leaks, forged manifest blocked by content hash |
| Mixture deviation | 0.0000 against a 0.06 tolerance, floors respected |
| OPUS | 484 accepted / 425 rejected / 66 deferred / 28 protected-floor overrides |
| Packing | 0.9530 utilisation, 0.9464 useful-token fraction, 37 bins saved vs `pad_only` |
| Throughput | ~5700 raw tok/s, ~5400 useful loss-bearing tok/s |
| Checkpoints | 6, each carrying planner state and ledger byte offsets |
| Crash / resume | crashed at step 27, resumed from step 20, 156 ledger events discarded |
| Resume proof | expected `main/step00020` matched by batch id **and** batch hash |
| Replay | 40 microbatches from the ledger, 5 steps by replanning, all hashes identical |
| Fork | `fork_b` from step 10, diverged at step 10, zero identical steps |
| Wall clock | ~17 s on CPU, ~15 MB of committed artifacts |

The full detail is in `submission_artifacts/evidence.md`.

## Checking reproducibility yourself

Run the demonstration twice and compare one field:

```bash
python run_demo.py && jq -r .reproducibility.stream_hash submission_artifacts/evidence.json
python run_demo.py && jq -r .reproducibility.stream_hash submission_artifacts/evidence.json
```

Both print the same hash. It covers the config, the tokenizer, the shard index,
the compiled mixture plan, every OPUS decision, and the full batch stream of
both branches — everything that is content-derived.

`evidence_hash` is *not* stable between runs, on purpose: it covers the measured
throughput, and tokens/sec is a measurement rather than a property of the data.
Conflating the two would let a timing wobble look like a reproducibility
failure, so the bundle reports them separately.

Wall-clock fields are recorded in manifests and checkpoints but excluded from
their hashes, so rebuilding the same shard from the same documents yields the
same `manifest_hash` — otherwise "immutable object" would mean nothing.

## The evidence bundle is adversarial

`pipeline/evidence.py` runs last and trusts nothing. It re-reads the shard
bytes, the manifests, the five ledgers, the checkpoints, the mixture plan and
the performance report, and recomputes every number it reports:

- it re-materialises packed bins from shard bytes and re-derives the mask
  invariants, rather than believing the recorded hashes;
- it recomputes packing utilisation from the consumption ledger and compares it
  to what `performance.json` claims;
- it re-verifies the tokenizer hash against `tokenizer.json` on disk;
- it checks that no shard carrying an evaluation canary ever appears in the
  consumption ledger;
- it verifies every ledger hash chain and every checkpoint's self-hash.

If a claim cannot be reproduced from artifacts, it is recorded as a FAIL rather
than omitted.

## What is deliberately small

The model is a 2-layer, 64-dim causal LM (3.3M parameters) at a 128-token
context, trained for 60 steps on CPU. It exists so that loss, gradient norms and
per-token perplexity are real numbers attached to real data, not so that it
learns anything useful — though it does learn: training loss falls 10.83 → 7.92
and the held-out validation loss falls 9.55 → 7.64 alongside it.

Every property this submission argues for — determinism, auditability, exact
recovery, firewall enforcement — is independent of model size, and the same
ledger and planner code would run against a 120B model unchanged. Single rank,
single worker: the `rank` field is recorded throughout and is always 0.

Width is 64 rather than 128 for a practical reason worth stating: a checkpoint
stores weights *and* both AdamW moments, and the tied 50257-row embedding
dominates all three. At width 128 each checkpoint was 82 MB and the artifact
directory was half a gigabyte. `checkpoints/**/state.pt` is also gitignored —
it is regenerable binary, and the `meta.json` beside it is the part that is
evidence (step, `next_batch_id`, planner state, ledger byte offsets, hashes).
The committed artifacts come to ~15 MB; `python run_demo.py` rebuilds the rest.
