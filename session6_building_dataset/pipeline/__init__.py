"""Training Data Execution System -- Session 6.

Stage order, which is also the order the modules should be read in:

    corpus_builder   documents
    shard_builder    immutable tokenized shards
    manifest         manifests, admission gate, evaluation firewall
    mixture          curriculum stages, lane quotas, protected floors
    opus             the selection policy inside the data path
    packing          bins, loss masks, attention masks, position ids
    batching         the deterministic batch planner
    ledger           append-only, hash-chained consumption/learning ledgers
    model            a small causal LM that honours segment-aware attention
    trainer          the training loop that consumes the stream and records it
    replay           reconstruct a historical interval and compare hashes
    audit            answer "what trained this checkpoint?"
    performance      throughput and packing efficiency
    evidence         the machine-readable evidence bundle
"""

__all__ = [
    "common", "corpus_builder", "shard_builder", "manifest", "mixture", "opus",
    "packing", "batching", "ledger", "model", "trainer", "replay", "audit",
    "performance", "evidence",
]
