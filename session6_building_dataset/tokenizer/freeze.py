"""Stage 3 -- the frozen tokenizer.

The tokenizer is downloaded once, written to this directory, and then treated
as immutable. `tokenizer_freeze.json` records

    tokenizer_hash = SHA256(tokenizer.json)

plus the hash of every other file the tokenizer directory contains. That hash
is stamped into every shard manifest and every checkpoint, and it is re-verified
before training is allowed to start:

    current tokenizer_hash != manifest tokenizer_hash  ->  the run refuses

Token ids are meaningless without the tokenizer that produced them, so this is
the root of trust for every downstream hash in the system.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.common import ensure_dir, file_sha256, read_json, write_json  # noqa: E402

FREEZE_FILE = "tokenizer_freeze.json"
PRIMARY_FILE = "tokenizer.json"


class TokenizerIntegrityError(RuntimeError):
    """Raised when the on-disk tokenizer no longer matches the frozen hash."""


def _frozen_dir(cfg) -> str:
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        cfg.tokenizer.frozen_dir)


def _materialise(cfg, out_dir: str):
    """Download the tokenizer and write its files into the frozen directory."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.tokenizer.hf_id)
    if tok.pad_token is None:
        # GPT-2 has no pad token. Padding is excluded by the loss mask and the
        # attention mask; the id only ever fills unused positions.
        tok.pad_token = tok.eos_token
    tok.save_pretrained(out_dir)
    return tok


def freeze_tokenizer(cfg, log=None, force: bool = False) -> dict:
    """Materialise the tokenizer locally and record its hashes."""
    out_dir = ensure_dir(_frozen_dir(cfg))
    freeze_path = os.path.join(out_dir, FREEZE_FILE)

    if os.path.exists(freeze_path) and not force:
        record = read_json(freeze_path)
        # The freeze record is committed but the tokenizer files are not, so a
        # fresh clone has the hash without the thing it describes. Re-fetch and
        # require the re-fetched bytes to hash to the committed value -- which is
        # a stronger check than reuse, not a weaker one: it proves the upstream
        # tokenizer has not moved under us.
        if not os.path.exists(os.path.join(out_dir, PRIMARY_FILE)):
            if log:
                log.note("tokenizer_files_absent", recorded=record["tokenizer_hash"][:16],
                         action="re-fetching and checking against the frozen hash")
            _materialise(cfg, out_dir)
            ok, current, recorded = verify_tokenizer(cfg)
            if not ok:
                raise TokenizerIntegrityError(
                    f"re-fetched tokenizer does not match the frozen hash: "
                    f"{current[:16]} != {recorded[:16]}")
            if log:
                log.ok("tokenizer_refetched_hash_matched", hf_id=record["hf_id"],
                       tokenizer_hash=recorded[:16])
            return record
        ok, current, recorded = verify_tokenizer(cfg)
        if ok:
            if log:
                log.ok("tokenizer_frozen_reused", hf_id=record["hf_id"],
                       vocab_size=record["vocab_size"], tokenizer_hash=recorded[:16])
            return record
        raise TokenizerIntegrityError(
            f"tokenizer.json changed since freeze: {current[:16]} != {recorded[:16]}")

    tok = _materialise(cfg, out_dir)

    files = {}
    for name in sorted(os.listdir(out_dir)):
        p = os.path.join(out_dir, name)
        if os.path.isfile(p) and name not in (FREEZE_FILE, "freeze.py", "__init__.py") \
                and not name.endswith(".pyc"):
            files[name] = file_sha256(p)

    record = {
        "hf_id": cfg.tokenizer.hf_id,
        "tokenizer_hash": files[PRIMARY_FILE],
        "vocab_size": int(tok.vocab_size),
        "len_tokenizer": int(len(tok)),
        "eos_token_id": int(tok.eos_token_id),
        "pad_token_id": int(tok.pad_token_id),
        "model_max_length": int(min(tok.model_max_length, 10 ** 6)),
        "files": files,
        "frozen_by": "tokenizer/freeze.py",
    }
    write_json(freeze_path, record)
    if log:
        log.ok("tokenizer_frozen", hf_id=record["hf_id"], vocab_size=record["vocab_size"],
               tokenizer_hash=record["tokenizer_hash"][:16], files=len(files))
    return record


def freeze_record(cfg) -> dict:
    return read_json(os.path.join(_frozen_dir(cfg), FREEZE_FILE))


def verify_tokenizer(cfg) -> tuple[bool, str, str]:
    """Recompute SHA256(tokenizer.json) and compare with the freeze record."""
    out_dir = _frozen_dir(cfg)
    record = read_json(os.path.join(out_dir, FREEZE_FILE))
    current = file_sha256(os.path.join(out_dir, PRIMARY_FILE))
    all_match = all(
        file_sha256(os.path.join(out_dir, name)) == h
        for name, h in record["files"].items()
        if os.path.exists(os.path.join(out_dir, name)))
    return (current == record["tokenizer_hash"] and all_match, current,
            record["tokenizer_hash"])


def load_tokenizer(cfg, log=None):
    """Load the frozen tokenizer, refusing to proceed if the hash moved."""
    from transformers import AutoTokenizer
    ok, current, recorded = verify_tokenizer(cfg)
    if not ok:
        if log:
            log.fail("tokenizer_hash_verified", current=current[:16], expected=recorded[:16])
        raise TokenizerIntegrityError(
            f"frozen tokenizer mismatch: {current[:16]} != {recorded[:16]}")
    if log:
        log.ok("tokenizer_hash_verified", tokenizer_hash=recorded[:16],
               source=cfg.tokenizer.hf_id)
    tok = AutoTokenizer.from_pretrained(_frozen_dir(cfg))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok, recorded


if __name__ == "__main__":  # `python tokenizer/freeze.py` re-freezes
    from pipeline.common import RunLog, load_config
    cfg = load_config()
    log = RunLog(os.path.join(cfg.resolve("artifacts_dir"), "freeze.log"),
                 os.path.join(cfg.resolve("artifacts_dir"), "freeze_events.jsonl"))
    freeze_tokenizer(cfg, log, force="--force" in sys.argv)
