"""Stages 9/10 -- append-only ledgers.

Five ledgers, one JSONL file each, all with the same shape:

    consumption   every sample served into a loss-bearing batch
    learning      what the model did with each batch (loss, grad norm, lr)
    opus          every candidate decision, accepted or not
    firewall      every block, every attempted access to non-trainable data
    token_trace   token-level loss for sampled steps

Two properties make them usable as evidence rather than as logs:

**Hash chain.** Each event stores `prev_hash` and `event_hash =
SHA256(prev_hash + canonical(body))`. A modified, reordered or deleted event
breaks the chain at that point and `verify_chain` says exactly where. Nobody can
quietly improve the history after the fact.

**Byte offsets.** `offset()` returns (bytes, count). A checkpoint stores those
numbers, and recovery calls `truncate_to` with them, which physically shortens
the file back to the checkpointed state. That is what makes crash recovery exact
rather than approximate: events written after the last checkpoint are removed,
so the resumed run cannot double-count a batch it already recorded.
"""

from __future__ import annotations

import json
import os
from typing import Iterator

from .common import canonical_json, ensure_dir, now_iso, sha256_hex

GENESIS = "0" * 64


class Ledger:
    def __init__(self, path: str, name: str) -> None:
        self.path = path
        self.name = name
        ensure_dir(os.path.dirname(os.path.abspath(path)))
        if not os.path.exists(path):
            open(path, "w", encoding="utf-8").close()
        self._seq, self._tail = self._recover_tail()

    # -- state -------------------------------------------------------------- #

    def _recover_tail(self) -> tuple[int, str]:
        seq, tail = 0, GENESIS
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    ev = json.loads(line)
                    seq = ev["seq"] + 1
                    tail = ev["event_hash"]
        return seq, tail

    @property
    def tail_hash(self) -> str:
        return self._tail

    @property
    def count(self) -> int:
        return self._seq

    def byte_size(self) -> int:
        return os.path.getsize(self.path)

    def offset(self) -> dict:
        return {"bytes": self.byte_size(), "count": self._seq, "tail_hash": self._tail}

    # -- writing ------------------------------------------------------------ #

    def append(self, event: dict) -> dict:
        body = dict(event)
        body["seq"] = self._seq
        body["ts"] = now_iso()
        body["prev_hash"] = self._tail
        body["event_hash"] = sha256_hex(self._tail + canonical_json(
            {k: v for k, v in body.items() if k != "event_hash"}))
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(canonical_json(body) + "\n")
            fh.flush()
            os.fsync(fh.fileno())          # a checkpoint may reference this byte
        self._seq += 1
        self._tail = body["event_hash"]
        return body

    def truncate_to(self, offset: dict) -> dict:
        """Roll the ledger back to a checkpointed offset (crash recovery)."""
        before = self.offset()
        target_bytes, target_count = int(offset["bytes"]), int(offset["count"])
        if before["bytes"] < target_bytes:
            raise RuntimeError(
                f"{self.name}: ledger is shorter than the checkpoint claims "
                f"({before['bytes']} < {target_bytes})")
        with open(self.path, "r+b") as fh:
            fh.truncate(target_bytes)
            fh.flush()
            os.fsync(fh.fileno())
        self._seq, self._tail = self._recover_tail()
        if self._seq != target_count:
            raise RuntimeError(f"{self.name}: truncation landed on {self._seq} events, "
                               f"expected {target_count}")
        if offset.get("tail_hash") and self._tail != offset["tail_hash"]:
            raise RuntimeError(f"{self.name}: tail hash after truncation does not match")
        return {"ledger": self.name, "before": before, "after": self.offset(),
                "events_discarded": before["count"] - self._seq,
                "bytes_discarded": before["bytes"] - target_bytes}

    # -- reading ------------------------------------------------------------ #

    def __iter__(self) -> Iterator[dict]:
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def read_all(self) -> list[dict]:
        return list(self)

    def where(self, **eq) -> list[dict]:
        return [e for e in self if all(e.get(k) == v for k, v in eq.items())]

    # -- integrity ---------------------------------------------------------- #

    def verify_chain(self) -> dict:
        prev, n = GENESIS, 0
        for i, ev in enumerate(self):
            if ev["seq"] != i:
                return {"ok": False, "ledger": self.name, "error": "seq_gap",
                        "at": i, "found": ev["seq"], "events": n}
            if ev["prev_hash"] != prev:
                return {"ok": False, "ledger": self.name, "error": "prev_hash_mismatch",
                        "at": i, "events": n}
            expect = sha256_hex(prev + canonical_json(
                {k: v for k, v in ev.items() if k != "event_hash"}))
            if expect != ev["event_hash"]:
                return {"ok": False, "ledger": self.name, "error": "event_hash_mismatch",
                        "at": i, "events": n}
            prev = ev["event_hash"]
            n += 1
        return {"ok": True, "ledger": self.name, "events": n, "tail_hash": prev}


class LedgerSet:
    """The five ledgers of one branch, checkpointed and rolled back together."""

    NAMES = ("consumption", "learning", "opus", "firewall", "token_trace")

    def __init__(self, ledgers_dir: str, branch_id: str) -> None:
        self.dir = ensure_dir(os.path.join(ledgers_dir, branch_id))
        self.branch_id = branch_id
        self.ledgers = {name: Ledger(os.path.join(self.dir, f"{name}.jsonl"), name)
                        for name in self.NAMES}

    def __getattr__(self, item: str) -> Ledger:
        try:
            return self.__dict__["ledgers"][item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def offsets(self) -> dict:
        return {name: lg.offset() for name, lg in self.ledgers.items()}

    def truncate_to(self, offsets: dict) -> list[dict]:
        return [self.ledgers[name].truncate_to(off) for name, off in sorted(offsets.items())
                if name in self.ledgers]

    def verify_all(self) -> dict:
        results = {name: lg.verify_chain() for name, lg in self.ledgers.items()}
        return {"ok": all(r["ok"] for r in results.values()), "ledgers": results}
