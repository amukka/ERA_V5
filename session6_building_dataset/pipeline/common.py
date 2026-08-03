"""Shared primitives: hashing, canonical serialisation, config, run log.

Two rules hold everywhere in this package:

1. Identity is a hash of bytes we actually wrote (never a name, never a
   timestamp, never an in-memory id).
2. Anything that looks random in the data path is derived from a hash of a key,
   not from a stateful RNG, so the same key yields the same decision in another
   process on another machine.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from typing import Any, Iterable

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------------------- #
# hashing / serialisation
# --------------------------------------------------------------------------- #

def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(data: Any) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    elif isinstance(data, memoryview):
        data = bytes(data)
    return hashlib.sha256(data).hexdigest()


def hash_obj(obj: Any) -> str:
    return sha256_hex(canonical_json(obj))


def hash_tokens(ids: Iterable[int]) -> str:
    """Hash a token-id sequence through a fixed dtype so it is machine stable."""
    import numpy as np
    arr = ids if isinstance(ids, np.ndarray) else np.asarray(list(ids), dtype=np.uint32)
    return sha256_hex(arr.astype(np.uint32).tobytes())


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def short(h: str, n: int = 12) -> str:
    return h[:n]


# --------------------------------------------------------------------------- #
# keyed determinism
# --------------------------------------------------------------------------- #

def stable_hash_int(*parts: Any) -> int:
    return int(sha256_hex("|".join(str(p) for p in parts))[:16], 16)


def stable_unit(*parts: Any) -> float:
    """Deterministic float in [0, 1) keyed by the parts."""
    return stable_hash_int(*parts) / 2.0 ** 64


def stable_permutation(n: int, *parts: Any) -> list[int]:
    """Keyed Fisher-Yates: same key -> same permutation, always."""
    idx = list(range(n))
    for i in range(n - 1, 0, -1):
        j = stable_hash_int(*parts, "swap", i) % (i + 1)
        idx[i], idx[j] = idx[j], idx[i]
    return idx


# --------------------------------------------------------------------------- #
# files
# --------------------------------------------------------------------------- #

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def write_json(path: str, obj: Any, indent: int = 2) -> str:
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=indent, sort_keys=True, ensure_ascii=False)
        fh.write("\n")
    os.replace(tmp, path)
    return hash_obj(obj)


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def write_jsonl(path: str, rows: Iterable[dict]) -> int:
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(canonical_json(r) + "\n")
            n += 1
    return n


def read_jsonl(path: str) -> list[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #

class Config(dict):
    """Dict with attribute access, a config hash, and path resolution."""

    def __init__(self, data: dict, path: str) -> None:
        super().__init__(data)
        self.path = path
        self.config_hash = file_sha256(path)

    def __getattr__(self, item: str) -> Any:
        try:
            v = self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc
        return Config(v, self.path) if isinstance(v, dict) else v

    def resolve(self, key: str) -> str:
        """Absolute path for one of the entries under `paths:`."""
        return os.path.join(ROOT, self["paths"][key])


def load_config(path: str | None = None) -> Config:
    import yaml
    path = path or os.path.join(ROOT, "config", "config.yaml")
    with open(path, "r", encoding="utf-8") as fh:
        return Config(yaml.safe_load(fh), path)


# --------------------------------------------------------------------------- #
# run log -- one writer, two outputs that cannot disagree
# --------------------------------------------------------------------------- #

class RunLog:
    def __init__(self, log_path: str, events_path: str, echo: bool = True) -> None:
        ensure_dir(os.path.dirname(os.path.abspath(log_path)))
        self.log_path, self.events_path, self.echo = log_path, events_path, echo
        self.events: list[dict] = []
        self._phase = "init"
        open(self.log_path, "w", encoding="utf-8").close()
        open(self.events_path, "w", encoding="utf-8").close()
        self.t0 = time.time()

    def line(self, msg: str = "") -> None:
        with open(self.log_path, "a", encoding="utf-8") as fh:
            fh.write(msg + "\n")
        if self.echo:
            sys.stdout.write(msg + "\n")
            sys.stdout.flush()

    def phase(self, name: str, title: str) -> None:
        self._phase = name
        self.line("")
        self.line("=" * 78)
        self.line(f"PHASE {name} :: {title}")
        self.line("=" * 78)
        # recorded as an event, not only as a banner, so the evidence builder can
        # prove from run_events.jsonl alone that every phase actually started
        ev = {"ts": now_iso(), "t": round(time.time() - self.t0, 3), "phase": name,
              "event": "phase_started", "status": "INFO", "fields": {"title": title}}
        self.events.append(ev)
        with open(self.events_path, "a", encoding="utf-8") as fh:
            fh.write(canonical_json(ev) + "\n")

    def info(self, msg: str) -> None:
        self.line(f"    {msg}")

    def event(self, name: str, status: str = "PASS", **fields: Any) -> dict:
        ev = {"ts": now_iso(), "t": round(time.time() - self.t0, 3), "phase": self._phase,
              "event": name, "status": status, "fields": fields}
        self.events.append(ev)
        with open(self.events_path, "a", encoding="utf-8") as fh:
            fh.write(canonical_json(ev) + "\n")
        detail = "  ".join(f"{k}={_fmt(v)}" for k, v in sorted(fields.items()))
        self.line(f"[{status}] {name}" + (f"  {detail}" if detail else ""))
        return ev

    def ok(self, name: str, **fields: Any) -> dict:
        return self.event(name, "PASS", **fields)

    def fail(self, name: str, **fields: Any) -> dict:
        return self.event(name, "FAIL", **fields)

    def note(self, name: str, **fields: Any) -> dict:
        return self.event(name, "INFO", **fields)

    def check(self, name: str, condition: bool, **fields: Any) -> bool:
        self.event(name, "PASS" if condition else "FAIL", **fields)
        return bool(condition)

    def counts(self) -> dict:
        out = {"PASS": 0, "FAIL": 0, "INFO": 0}
        for ev in self.events:
            out[ev["status"]] = out.get(ev["status"], 0) + 1
        return out

    def failures(self) -> list[dict]:
        return [e for e in self.events if e["status"] == "FAIL"]


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.6g}"
    if isinstance(v, (list, tuple)):
        s = json.dumps(list(v)[:6], ensure_ascii=False)
        return s if len(v) <= 6 else s[:-1] + ",...]"
    if isinstance(v, dict):
        return canonical_json(v)
    return str(v)


def set_global_seed(seed: int) -> None:
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
