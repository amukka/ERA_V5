# Evidence Bundle

**Run:** `s6-tdes`  
**Generated:** 2026-08-03T14:06:10Z  
**Config hash:** `730aecc6ea8d9d37`  
**Tokenizer hash:** `1fe93b6152957cf9`  
**Shard index hash:** `0c38ab744dfcba2f`  
**Mixture plan hash:** `5629576d3ce278d1`  
**Evidence hash:** `613e44ce3b4f028f`  
**Stream hash:** `f46a7d019c1154ef` — content-derived; identical on a re-run of `python run_demo.py`

**Result: 9/9 requirements passed, 59/59 individual checks passed.**

Every number below was recomputed by `pipeline/evidence.py` from the artifacts on disk (shard bytes, manifests, ledgers, checkpoints), after the run finished. Nothing is copied from the trainer's memory.

## Required summary

| Requirement | Result | Evidence |
| --- | --- | --- |
| Tokenizer integrity | **PASS** | Manifest record — `manifests/shard_index.json` |
| Evaluation firewall | **PASS** | Blocked-shard event — `reports/firewall.json` |
| Packing correctness | **PASS** | Packed-batch report — `ledgers/main/consumption.jsonl` |
| Mixture compliance | **PASS** | Planned versus actual shares — `reports/mixture_compliance.json` |
| OPUS audit trail | **PASS** | Candidate decision records — `ledgers/main/opus.jsonl` |
| Crash recovery | **PASS** | Expected and resumed batch ids — `reports/recovery.json` |
| Replay | **PASS** | Original and replay hashes — `reports/replay_from_ledger.json` |
| Learning trace | **PASS** | Loss linked to source data — `ledgers/main/learning.jsonl` |
| Throughput | **PASS** | Performance report — `performance.json` |

## Scored areas

| Area | Result | Checks | Points |
| --- | --- | --- | --- |
| End-to-end execution | **PASS** | 3/3 | 150/150 |
| Shards, manifests and tokenizer integrity | **PASS** | 7/7 | 100/100 |
| Packing, masks and batch correctness | **PASS** | 7/7 | 150/150 |
| Mixture schedule, protected floors and OPUS | **PASS** | 8/8 | 150/150 |
| Consumption and learning ledgers | **PASS** | 7/7 | 150/150 |
| Checkpoint, crash, resume, replay and fork | **PASS** | 11/11 | 150/150 |
| Evaluation and validation firewall | **PASS** | 7/7 | 50/50 |
| Throughput and packing efficiency | **PASS** | 5/5 | 50/50 |
| Tests, evidence quality and documentation | **PASS** | 4/4 | 50/50 |
| **Total** | **PASS** | **59/59** | **1000/1000** |

## Headline numbers

- Packing utilisation **0.9530** (recomputed from the ledger: 0.9530), 37 fewer bins than `pad_only` for the same samples.
- Useful loss-bearing tokens/s **5398.0** against raw 5703.6 tokens/s (useful fraction 0.9464).
- Mixture deviation **0.0000** against a tolerance of 0.06; protected floors respected: True.
- OPUS saw **975** candidates: {'accepted': 484, 'deferred': 66, 'rejected': 425}, with 28 protected-floor overrides.
- Crash at step **27**, resumed from checkpoint **20**, discarding **156** post-checkpoint ledger events.
- Expected next batch `main/step00020` (`af2787e371cb47a7`) matched the resumed batch `main/step00020` (`af2787e371cb47a7`).
- Replay reproduced **40** microbatches from the ledger and **5** steps by replanning.
- Fork `fork_b` branched at step 10 and diverged at step 10.

## Every check

### End-to-end execution — **PASS**

_One command runs the whole path and every phase completes_

| Check | Result | Detail |
| --- | --- | --- |
| `every_phase_ran` | **PASS** | `{"missing":[]}` |
| `no_failed_events` | **PASS** | `{"failures":0}` |
| `run_log_present` | **PASS** | `` |

<details><summary>metrics</summary>

```json
{"events":138,"failed_event_names":[],"failed_events":0,"phases_executed":["01_corpus","02_tokenizer","03_shards","04_admission","05_firewall","06_mixture","07_index","08_train","09_resume","10_replay","11_fork","12_audit","13_learning","14_performance","15_tests","16_evidence"]}
```

</details>

Evidence: `run.log`, `run_events.jsonl`

### Shards, manifests and tokenizer integrity — **PASS**

_Immutable hashed shards, one frozen tokenizer, verified manifests_

| Check | Result | Detail |
| --- | --- | --- |
| `tokenizer_hash_verified` | **PASS** | `{"recomputed":"1fe93b6152957cf9","recorded":"1fe93b6152957cf9"}` |
| `all_manifests_carry_the_frozen_tokenizer_hash` | **PASS** | `{"found":["1fe93b6152957cf9"]}` |
| `shard_bytes_match_content_hashes` | **PASS** | `{"failing":0}` |
| `manifest_self_hashes_valid` | **PASS** | `` |
| `shard_index_hash_reproducible` | **PASS** | `` |
| `shard_files_are_read_only` | **PASS** | `{"writable":[]}` |
| `every_manifest_has_lineage` | **PASS** | `` |

<details><summary>metrics</summary>

```json
{"cleaning_pipeline_hashes":["7a198c9462ce575d2bb98d913240392d754be76bb3796eab6bc4ae4a623a6170"],"distinct_tokenizer_hashes_in_manifests":1,"index_hash":"0c38ab744dfcba2f3a93ccd46299c0a6df65b8761054e503c1a7b4d0dcbb6815","index_hash_recomputed":"0c38ab744dfcba2f3a93ccd46299c0a6df65b8761054e503c1a7b4d0dcbb6815","manifests_failing_verification":[],"shards":31,"token_total":2971696,"tokenizer_hash":"1fe93b6152957cf9cfd6d89002467f789ce8b3f3e000b3a2edf27c808ddd0b9e","tokenizer_hash_recomputed":"1fe93b6152957cf9cfd6d89002467f789ce8b3f3e000b3a2edf27c808ddd0b9e"}
```

</details>

Evidence: `manifests/shard_index.json`, `manifests/`, `shards/`, `../tokenizer/tokenizer_freeze.json`

### Packing, masks and batch correctness — **PASS**

_Bins rebuild from shard bytes; loss/attention/position rules hold_

| Check | Result | Detail |
| --- | --- | --- |
| `bins_rebuild_to_the_recorded_hash` | **PASS** | `{"mismatches":0,"rebuilt":40}` |
| `loss_never_crosses_a_sample_boundary` | **PASS** | `{"violations":0}` |
| `padding_never_bears_loss` | **PASS** | `{"violations":0}` |
| `position_ids_restart_per_packed_sample` | **PASS** | `{"violations":0}` |
| `attention_is_block_diagonal_and_causal` | **PASS** | `{"leaks":0}` |
| `no_bin_exceeds_the_context_window` | **PASS** | `` |
| `packing_policy_recorded_in_every_event` | **PASS** | `` |

<details><summary>metrics</summary>

```json
{"attention_leaks":0,"bins_rechecked":40,"bins_total":480,"boundary_violations":0,"hash_mismatches":0,"over_length_bins":0,"padded_loss_positions":0,"position_violations":0,"sequence_length":128}
```

</details>

Evidence: `ledgers/main/consumption.jsonl`, `reports/packing_policies.json`

### Mixture schedule, protected floors and OPUS — **PASS**

_Realised lane shares track the plan; floors hold; OPUS is audited_

| Check | Result | Detail |
| --- | --- | --- |
| `mixture_within_tolerance` | **PASS** | `{"max_deviation":0.0,"tolerance":0.06}` |
| `protected_floors_respected` | **PASS** | `` |
| `mixture_plan_hash_reproducible` | **PASS** | `` |
| `opus_recorded_all_four_outcomes` | **PASS** | `{"observed":["accepted","deferred","rejected"]}` |
| `opus_protected_floor_overrides_present` | **PASS** | `{"overrides":28}` |
| `every_rejection_carries_a_reason` | **PASS** | `` |
| `rejected_candidates_are_retained_not_dropped` | **PASS** | `` |
| `every_consumed_sample_has_an_opus_decision` | **PASS** | `` |

<details><summary>metrics</summary>

```json
{"floors_respected":true,"max_deviation":0.0,"opus_by_status":{"accepted":484,"deferred":66,"rejected":425},"opus_candidates":975,"plan_hash":"5629576d3ce278d12892b724b7cc8ac982396495dd8cf5a9d0ea25cd438b4455","plan_hash_recomputed":"5629576d3ce278d12892b724b7cc8ac982396495dd8cf5a9d0ea25cd438b4455","planned_overall_shares":{"code":0.25,"dialog":0.25,"wiki":0.5},"protected_floor_overrides":28,"realised_overall_shares":{"code":0.25,"dialog":0.25,"wiki":0.5},"tolerance":0.06}
```

</details>

Evidence: `reports/mixture_compliance.json`, `reports/opus_summary.json`, `manifests/mixture_plan.json`, `ledgers/main/opus.jsonl`

### Consumption and learning ledgers — **PASS**

_Hash-chained, append-only, and the learning side links loss to data_

| Check | Result | Detail |
| --- | --- | --- |
| `all_ledger_hash_chains_verify` | **PASS** | `{"detail":{"consumption":"ok","firewall":"ok","learning":"ok","opus":"ok","token_trace":"ok"}}` |
| `consumption_records_token_spans` | **PASS** | `` |
| `consumption_records_the_required_fields` | **PASS** | `` |
| `learning_links_loss_back_to_shards` | **PASS** | `{"linked":60,"steps":60}` |
| `token_level_trace_carries_provenance` | **PASS** | `{"traces":32}` |
| `shard_report_cards_cover_consumed_shards` | **PASS** | `` |
| `learning_ledger_produces_feedback` | **PASS** | `` |

<details><summary>metrics</summary>

```json
{"chain_detail":{"consumption":"ok","firewall":"ok","learning":"ok","opus":"ok","token_trace":"ok"},"chain_ok":true,"consumption_events":540,"feedback_lines":3,"firewall_events":19,"learning_events":63,"opus_events":975,"shard_report_cards":17,"steps_with_shard_attribution":60,"token_trace_events":32,"traced_tokens":3830,"verdict_counts":{"harmful":0,"needs_warmup":0,"neutral":1,"useful":16}}
```

</details>

Evidence: `ledgers/main/`, `reports/shard_report_cards.json`

### Checkpoint, crash, resume, replay and fork — **PASS**

_Data state travels with model state_

| Check | Result | Detail |
| --- | --- | --- |
| `checkpoints_verify_against_their_own_hashes` | **PASS** | `{"checkpoints":6}` |
| `checkpoint_binds_a_data_position` | **PASS** | `` |
| `resume_next_batch_matched` | **PASS** | `{"expected":"main/step00020","resumed":"main/step00020"}` |
| `resume_discarded_post_checkpoint_events` | **PASS** | `{"discarded":156}` |
| `redone_steps_are_byte_identical_to_the_crashed_attempt` | **PASS** | `{"mismatches":[],"steps":[20,21,22,23,24,25,26]}` |
| `no_skipped_or_repeated_batches` | **PASS** | `{"duplicated":[],"missing":[]}` |
| `replay_from_ledger_hashes_match` | **PASS** | `` |
| `replay_by_replanning_hashes_match` | **PASS** | `` |
| `fork_creates_a_new_branch_with_its_own_ledger` | **PASS** | `{"fork_events":73}` |
| `fork_divergence_is_explicit_and_recorded` | **PASS** | `` |
| `fork_shares_the_parent_checkpoint` | **PASS** | `` |

<details><summary>metrics</summary>

```json
{"checkpoints":6,"checkpoints_verified":6,"coverage":{"branch_id":"main","duplicated_steps":[],"exactly_once":true,"expected_steps":60,"missing_steps":[],"monotonic":true,"out_of_range_steps":[],"recorded_steps":60,"unique_steps":60},"crash_step":27,"expected_next_batch":"main/step00020","expected_next_batch_hash":"af2787e371cb47a7ee022bea221e80e5a1f646b7611833be97ac8f0a6b1a7bef","fork_branch":"fork_b","fork_diverged_at":10,"fork_from_step":10,"ledger_events_discarded":156,"replay_from_ledger_microbatches":40,"replay_replan_steps":5,"resumed_from_checkpoint":20,"resumed_next_batch":"main/step00020","resumed_next_batch_hash":"af2787e371cb47a7ee022bea221e80e5a1f646b7611833be97ac8f0a6b1a7bef"}
```

</details>

Evidence: `reports/recovery.json`, `checkpoints/`, `ledgers/fork_b/`

### Evaluation and validation firewall — **PASS**

_Test data is registered so it can be kept out, and it stays out_

| Check | Result | Detail |
| --- | --- | --- |
| `eval_shards_are_registered_with_never_train` | **PASS** | `` |
| `eval_shard_blocked` | **PASS** | `{"attempts":14}` |
| `validation_readable_but_never_gradient_bearing` | **PASS** | `` |
| `forged_manifest_blocked` | **PASS** | `{"detected_by":"unknown_shard_id+content_hash_matches:eval-code-0000"}` |
| `planted_leak_found_by_scanning_not_by_being_told` | **PASS** | `{"shards":["train-code-0000","train-code-0001","train-code-0002","train-code-0003","train-code-0004","train-code-0006","train-wiki-0009","train-wiki-0010"]}` |
| `no_eval_or_blocked_data_in_loss_bearing_batches` | **PASS** | `{"violations":0}` |
| `canaries_never_reached_training` | **PASS** | `` |

<details><summary>metrics</summary>

```json
{"blocked_attempts":14,"consumed_blocked_shards":[],"consumed_eval_shards":[],"consumed_validation_shards":[],"contaminated_shards":["train-code-0000","train-code-0001","train-code-0002","train-code-0003","train-code-0004","train-code-0006","train-wiki-0009","train-wiki-0010"],"contamination_fingerprint_ngrams":176319,"eval_shards_registered":3,"leak_detected_in":["train-code-0000","train-code-0001","train-code-0002","train-code-0003","train-code-0004","train-code-0006","train-wiki-0009","train-wiki-0010"],"leaked":[],"planted_leak_doc":"wiki.train.90000","validation_shards_registered":3}
```

</details>

Evidence: `reports/firewall.json`, `reports/contamination.json`, `ledgers/main/firewall.jsonl`

### Throughput and packing efficiency — **PASS**

_Useful loss-bearing tokens per second, reconstructable from ledgers_

| Check | Result | Detail |
| --- | --- | --- |
| `packing_utilisation_reconstructable_from_the_ledger` | **PASS** | `{"recomputed":0.952995,"reported":0.952995}` |
| `useful_tokens_per_second_reported` | **PASS** | `` |
| `useful_tokens_are_a_strict_subset_of_raw_tokens` | **PASS** | `` |
| `chosen_policy_beats_pad_only` | **PASS** | `{"gain":0.014704}` |
| `cache_and_loader_costs_measured` | **PASS** | `` |

<details><summary>metrics</summary>

```json
{"accepted_tokens_per_s":5442.85,"bins_saved_vs_pad_only":37,"cache_hit_rate":0.976023,"loader_wait_fraction":0.34601,"loss_utilisation_recomputed":0.945117,"microbatches":480,"opus_rejection_rate":0.435897,"packing_utilisation":0.952995,"packing_utilisation_recomputed":0.952995,"policy_comparison":[{"bins":2453,"content_tokens":313237,"open_bins":"unbounded","padding_pct":0.2379,"padding_tokens":747,"policy":"first_fit_decreasing","positions":313984,"streamable":false,"utilisation":0.997621},{"bins":2454,"content_tokens":313237,"open_bins":"unbounded","padding_pct":0.2786,"padding_tokens":875,"policy":"best_fit","positions":314112,"streamable":false,"utilisation":0.997214},{"bins":2455,"content_tokens":313237,"open_bins":"unbounded","padding_pct":0.3192,"padding_tokens":1003,"policy":"first_fit","positions":314240,"streamable":false,"utilisation":0.996808},{"bins":2463,"content_tokens":313237,"open_bins":"bounded","padding_pct":0.643,"padding_tokens":2027,"policy":"best_fit_decreasing","positions":315264,"streamable":true,"utilisation":0.99357},{"bins":2500,"content_tokens":313237,"open_bins":"unbounded","padding_pct":2.1134,"padding_tokens":6763,"policy":"pad_only","positions":320000,"streamable":true,"utilisation":0.978866}],"raw_tokens_per_s":5703.64,"samples_per_bin":1.0083,"useful_token_fraction":0.946406,"useful_tokens_per_s":5397.96}
```

</details>

Evidence: `performance.json`, `reports/packing_policies.json`

### Tests, evidence quality and documentation — **PASS**

_Automated invariant tests, a generated bundle, a README_

| Check | Result | Detail |
| --- | --- | --- |
| `automated_tests_ran` | **PASS** | `{"total":74}` |
| `automated_tests_passed` | **PASS** | `{"failed":0}` |
| `readme_present` | **PASS** | `` |
| `evidence_bundle_is_generated_not_hardcoded` | **PASS** | `{"note":"every metric in this file is recomputed from artifacts on disk"}` |

<details><summary>metrics</summary>

```json
{"artifact_files":107,"readme_bytes":13457,"test_command":"/Users/srinivasmukka/SchoolOfAI/ERA_V5/ERA_V5/session6_building_dataset/.venv/bin/python -m unittest discover -s tests -t . -v","tests_failed":0,"tests_run":74}
```

</details>

Evidence: `../tests/`, `../README.md`, `evidence.json`, `evidence.md`

## Artifact inventory

107 files, each hashed in `evidence.json` under `artifact_hashes`.

