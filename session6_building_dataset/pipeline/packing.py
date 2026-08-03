"""Stage 6 -- packing, masks and position ids.

Policy: **best-fit decreasing**. Samples are sorted longest-first, then each is
placed in the open bin that leaves the least free space after insertion; if it
fits nowhere, a new bin opens. Sorting first is what lifts utilisation above
plain best-fit: the big awkward samples are placed while the bins are still
empty, and the small ones fill the gaps left behind.

Every packed bin is `seq_len` positions wide and carries five aligned arrays:

    input_ids     tokens, with EOS after each packed sample, PAD to the end
    segment_ids   which packed sample each position belongs to (-1 = padding)
    position_ids  restart at 0 for every sample, so sample two does not look
                  like it lives at position 300 of sample one
    loss_mask     1 where the position is a legitimate prediction target
    attention     block-diagonal AND causal: a sample cannot attend to a
                  neighbour that happens to share its bin

`loss_mask[j] = 1` requires all three of:
    j > 0                          the first position has nothing to predict from
    input_ids[j] is not padding    padding never contributes gradient
    segment[j] == segment[j-1]     no sample is ever predicted from another one

`build_bin` is the single canonical construction path. Planning calls it with
tokens read from shards, and replay calls it again months later with tokens read
from the same shards -- which is why replay hashes match.
"""

from __future__ import annotations

from typing import Callable, Iterable

import numpy as np

from .common import sha256_hex

PAD_SEGMENT = -1


# --------------------------------------------------------------------------- #
# bin packing policies
# --------------------------------------------------------------------------- #

def _slot_cost(sample: dict, eos_between: bool) -> int:
    return int(sample["n_tokens"]) + (1 if eos_between else 0)


def best_fit_decreasing(samples: list[dict], seq_len: int, eos_between: bool = True,
                        max_open_bins: int = 8) -> list[list[dict]]:
    """Sort longest-first, then best-fit. Returns bins of placed sample refs."""
    ordered = sorted(samples, key=lambda s: (-_slot_cost(s, eos_between), s["sample_id"]))
    closed: list[list[dict]] = []
    open_bins: list[dict] = []          # {"items": [...], "used": int}

    for s in ordered:
        cost = _slot_cost(s, eos_between)
        if cost > seq_len:              # cannot happen: shards cut at seq_len-1
            raise ValueError(f"sample {s['sample_id']} longer than seq_len")
        best, best_left = None, None
        for b in open_bins:
            left = seq_len - b["used"] - cost
            if left >= 0 and (best_left is None or left < best_left):
                best, best_left = b, left
        if best is None:
            if len(open_bins) >= max_open_bins:
                # retire the fullest bin to make room; it is the least likely
                # to accept anything smaller than what we already tried
                fullest = max(open_bins, key=lambda b: b["used"])
                open_bins.remove(fullest)
                closed.append(fullest["items"])
            best = {"items": [], "used": 0}
            open_bins.append(best)
        s = dict(s)
        s["bin_offset"] = best["used"]
        best["items"].append(s)
        best["used"] += cost

    for b in open_bins:
        if b["items"]:
            closed.append(b["items"])
    return closed


def _first_fit(samples, seq_len, eos_between, decreasing=False):
    items = sorted(samples, key=lambda s: (-_slot_cost(s, eos_between), s["sample_id"])) \
        if decreasing else list(samples)
    bins: list[dict] = []
    for s in items:
        cost = _slot_cost(s, eos_between)
        for b in bins:
            if b["used"] + cost <= seq_len:
                b["items"].append(s)
                b["used"] += cost
                break
        else:
            bins.append({"items": [s], "used": cost})
    return [b["items"] for b in bins]


def _best_fit_ordered(samples, seq_len, eos_between):
    """Plain best-fit: arrival order, no sorting -- the baseline BFD beats."""
    bins: list[dict] = []
    for s in samples:
        cost = _slot_cost(s, eos_between)
        best, best_left = None, None
        for b in bins:
            left = seq_len - b["used"] - cost
            if left >= 0 and (best_left is None or left < best_left):
                best, best_left = b, left
        if best is None:
            bins.append({"items": [s], "used": cost})
        else:
            best["items"].append(s)
            best["used"] += cost
    return [b["items"] for b in bins]


def _pad_only(samples, seq_len, eos_between):
    return [[s] for s in samples]


POLICIES: dict[str, Callable] = {
    "pad_only": _pad_only,
    "first_fit": lambda s, L, e: _first_fit(s, L, e, decreasing=False),
    "best_fit": _best_fit_ordered,
    "first_fit_decreasing": lambda s, L, e: _first_fit(s, L, e, decreasing=True),
    "best_fit_decreasing": lambda s, L, e: best_fit_decreasing(s, L, e),
}

# Which policies are allowed to keep every bin they have ever opened. Only the
# configured policy pays the streaming constraint, so the table has to say so:
# an unbounded policy will win on utilisation and cannot be run on a real stream.
UNBOUNDED = {"pad_only", "first_fit", "best_fit", "first_fit_decreasing"}


def compare_policies(samples: list[dict], seq_len: int, eos_between: bool = True) -> dict:
    """Utilisation of each policy over the same sample set (for the report).

    Read the `open_bins` column before reading the utilisation column. The
    unbounded policies may hold every partially-filled bin in memory until the
    dataset ends, which is why they can edge out a bounded best-fit-decreasing
    by a fraction of a percent. A streaming loader does not have that option: it
    must emit batches as it goes, so it caps the number of open bins and accepts
    slightly more padding in exchange for bounded memory and steady output.
    """
    rows = []
    for name, fn in POLICIES.items():
        bins = fn(samples, seq_len, eos_between)
        used = sum(sum(_slot_cost(s, eos_between) for s in b) for b in bins)
        total = len(bins) * seq_len
        rows.append({
            "policy": name, "bins": len(bins), "positions": total,
            "content_tokens": used, "padding_tokens": total - used,
            "utilisation": round(used / max(1, total), 6),
            "padding_pct": round(100.0 * (total - used) / max(1, total), 4),
            "open_bins": "unbounded" if name in UNBOUNDED else "bounded",
            "streamable": name not in UNBOUNDED or name == "pad_only",
        })
    rows.sort(key=lambda r: -r["utilisation"])
    streamable = [r for r in rows if r["streamable"]]
    return {"seq_len": seq_len, "sample_count": len(samples),
            "token_total": sum(s["n_tokens"] for s in samples), "rows": rows,
            "best_policy": rows[0]["policy"],
            "best_streamable_policy": streamable[0]["policy"] if streamable else None,
            "note": "unbounded policies may hold every open bin until the dataset "
                    "ends; only the bounded ones can serve a live training stream"}


# --------------------------------------------------------------------------- #
# materialising a bin
# --------------------------------------------------------------------------- #

class PackedBin(dict):
    """A dict so it serialises straight into ledgers and reports."""

    @property
    def arrays(self) -> tuple[np.ndarray, ...]:
        return (np.asarray(self["input_ids"], dtype=np.int64),
                np.asarray(self["loss_mask"], dtype=np.int64),
                np.asarray(self["segment_ids"], dtype=np.int64),
                np.asarray(self["position_ids"], dtype=np.int64))


def build_bin(bin_id: str, lane: str, placed: Iterable[dict],
              token_reader: Callable[[str, int, int], np.ndarray],
              seq_len: int, eos_id: int, pad_id: int,
              eos_between: bool = True, reset_position_ids: bool = True) -> PackedBin:
    """Materialise one packed bin. This is the canonical, replayable path."""
    input_ids = np.full(seq_len, pad_id, dtype=np.int64)
    segment_ids = np.full(seq_len, PAD_SEGMENT, dtype=np.int64)
    position_ids = np.zeros(seq_len, dtype=np.int64)
    samples_out = []

    cursor = 0
    for seg, s in enumerate(placed):
        toks = np.asarray(token_reader(s["shard_id"], s["token_start"], s["token_end"]),
                          dtype=np.int64)
        n = len(toks)
        if n != s["n_tokens"]:
            raise ValueError(f"span mismatch for {s['sample_id']}: {n} != {s['n_tokens']}")
        end = cursor + n
        input_ids[cursor:end] = toks
        segment_ids[cursor:end] = seg
        if eos_between:
            input_ids[end] = eos_id
            segment_ids[end] = seg
            end += 1
        span = end - cursor
        position_ids[cursor:end] = np.arange(span) if reset_position_ids else \
            np.arange(cursor, end)
        samples_out.append({
            "sample_id": s["sample_id"], "shard_id": s["shard_id"], "doc_id": s["doc_id"],
            "lane": s.get("lane", lane), "token_start": int(s["token_start"]),
            "token_end": int(s["token_end"]), "n_tokens": int(n),
            "bin_offset": int(cursor), "bin_end": int(end), "segment": seg,
            "token_hash": s.get("token_hash"),
        })
        cursor = end

    if not reset_position_ids:
        position_ids[cursor:] = np.arange(cursor, seq_len)

    # loss mask: predictable, non-padding, and not across a segment boundary
    loss_mask = np.zeros(seq_len, dtype=np.int64)
    if cursor > 1:
        j = np.arange(1, seq_len)
        ok = (segment_ids[j] != PAD_SEGMENT) & (segment_ids[j] == segment_ids[j - 1])
        loss_mask[1:] = ok.astype(np.int64)

    attention_mask = (segment_ids != PAD_SEGMENT).astype(np.int64)

    binrec = PackedBin({
        "bin_id": bin_id,
        "lane": lane,
        "seq_len": seq_len,
        "input_ids": input_ids.tolist(),
        "loss_mask": loss_mask.tolist(),
        "segment_ids": segment_ids.tolist(),
        "position_ids": position_ids.tolist(),
        "attention_mask": attention_mask.tolist(),
        "samples": samples_out,
        "n_samples": len(samples_out),
        "content_tokens": int(cursor),
        "pad_tokens": int(seq_len - cursor),
        "loss_tokens": int(loss_mask.sum()),
        "utilisation": round(float(cursor) / seq_len, 6),
        "loss_utilisation": round(float(loss_mask.sum()) / seq_len, 6),
        "policy": "best_fit_decreasing",
    })
    binrec["content_hash"] = bin_hash(binrec)
    return binrec


def array_bytes(values) -> bytes:
    """The one canonical byte encoding for hashing a batch array.

    int64 rather than uint32 because `segment_ids` uses -1 for padding, and a
    hash that cannot represent padding is a hash that cannot detect it. Every
    hash of a batch array in this system goes through here, so the trainer, the
    replayer and the evidence builder cannot disagree about encoding.
    """
    return np.asarray(values, dtype=np.int64).tobytes()


def bin_hash(binrec: dict) -> str:
    """Hash over every array that reaches the model."""
    return sha256_hex(b"".join(
        array_bytes(binrec[k])
        for k in ("input_ids", "loss_mask", "segment_ids", "position_ids")))


def attention_4d(segment_ids: np.ndarray, dtype=None):
    """Additive block-diagonal causal mask for one batch of bins.

    segment_ids: (B, T) with PAD_SEGMENT for padding.
    returns:     (B, 1, T, T) additive float mask.
    """
    import torch
    if not isinstance(segment_ids, torch.Tensor):
        segment_ids = torch.as_tensor(np.asarray(segment_ids))
    b, t = segment_ids.shape
    causal = torch.tril(torch.ones(t, t, dtype=torch.bool, device=segment_ids.device))
    same = segment_ids.unsqueeze(2) == segment_ids.unsqueeze(1)      # (B, T, T)
    real = (segment_ids != PAD_SEGMENT).unsqueeze(1)                 # (B, 1, T)
    allow = causal.unsqueeze(0) & same & real
    # a padding row attends to itself so softmax never sees an all-masked row
    diag = torch.eye(t, dtype=torch.bool, device=segment_ids.device).unsqueeze(0)
    allow = allow | diag
    dtype = dtype or torch.float32
    mask = torch.zeros(b, 1, t, t, dtype=dtype, device=segment_ids.device)
    mask.masked_fill_(~allow.unsqueeze(1), torch.finfo(dtype).min)
    return mask
