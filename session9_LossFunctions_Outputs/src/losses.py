"""The harness from the assignment, made correct and observable.

The assignment hands us four lines:

    hidden = model(tokens)
    logits = output_head(hidden)
    loss = cross_entropy(logits[:, :-1].reshape(-1, vocab_size),
                         tokens[:, 1:].reshape(-1))

Everything wrong with those four lines is quiet, so every decision they make
implicitly is made explicitly here and given a name:

* which position predicts which token          -> ``align``
* which pairs are allowed to contribute        -> ``loss_mask``
* what the sum is divided by                   -> ``LossRecord.n_contributing``
* whether the logits have to exist all at once -> ``chunked_cross_entropy``

Nothing here changes what the model learns.  It changes what you can see.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .data import Seqs


# --------------------------------------------------------------------------
# 1. the shift
# --------------------------------------------------------------------------

@dataclass
class Alignment:
    """One (input position, target token) pairing, and how it was made.

    ``offset`` is how far ahead the target sits.  1 is next-token prediction.
    2 is the second head in part 2.  0 and -1 are the two ways to get it wrong,
    and they are here as first-class options rather than as a mistake you have
    to make by accident: both hand the model an answer it has already seen, and
    neither raises anything.
    """

    offset: int
    input_pos: torch.Tensor    # (P,) index into T of the position doing the work
    target_pos: torch.Tensor   # (P,) index into T of the token it must name
    name: str
    story: str


def align(T: int, offset: int = 1, device=None) -> Alignment:
    """Pair every usable position with the token ``offset`` ahead of it."""
    if offset > 0:
        # position i predicts token i+offset; the last `offset` positions have
        # nothing after them, the first `offset` tokens have nothing before.
        ip = torch.arange(0, T - offset, device=device)
        tp = torch.arange(offset, T, device=device)
        name = f"shift +{offset}"
        story = (f"position i predicts token i+{offset}"
                 if offset > 1 else
                 "position i predicts token i+1 -- the text is its own "
                 "supervision")
    elif offset == 0:
        ip = tp = torch.arange(0, T, device=device)
        name = "no shift"
        story = ("position i predicts token i -- the token it was just given. "
                 "This is not a prediction, it is a copy")
    else:
        k = -offset
        ip = torch.arange(k, T, device=device)
        tp = torch.arange(0, T - k, device=device)
        name = f"shift {offset}"
        story = (f"position i predicts token i-{k} -- a token already inside "
                 "its own context window. Also a copy, and it trains beautifully")
    return Alignment(offset, ip, tp, name, story)


def aligned_pairs(seqs: Seqs, a: Alignment):
    """Return ``(input_ids, target_ids, doc_in, doc_tgt, valid_in, valid_tgt)``
    for the pairing ``a``, each of shape (B, P)."""
    return (
        seqs.tokens.index_select(1, a.input_pos),
        seqs.tokens.index_select(1, a.target_pos),
        seqs.doc_id.index_select(1, a.input_pos),
        seqs.doc_id.index_select(1, a.target_pos),
        seqs.valid.index_select(1, a.input_pos),
        seqs.valid.index_select(1, a.target_pos),
    )


# --------------------------------------------------------------------------
# 2. the mask
# --------------------------------------------------------------------------

def loss_mask(seqs: Seqs, a: Alignment, mask_padding=True,
              mask_boundaries=True) -> torch.Tensor:
    """(B, P) bool: True where this pair is allowed to contribute.

    Two independent reasons to drop a pair, and they are separable on purpose
    because deliverables 3 and 4 need to switch them one at a time:

    padding      either end of the pair is a PAD slot.  Training on it teaches
                 the model to predict padding, which is trivially easy, so the
                 loss improves while the model does not.

    boundaries   the two ends came from different documents.  Nothing about
                 the end of one document predicts the start of the next; the
                 pair is real text on both sides and is still a lie.
    """
    _, _, doc_in, doc_tgt, valid_in, valid_tgt = aligned_pairs(seqs, a)
    keep = torch.ones_like(valid_in)
    if mask_padding:
        keep = keep & valid_in & valid_tgt
    if mask_boundaries:
        keep = keep & (doc_in == doc_tgt)
    return keep


# --------------------------------------------------------------------------
# 3. the loss
# --------------------------------------------------------------------------

@dataclass
class LossRecord:
    loss: torch.Tensor              # the scalar the optimiser sees
    n_contributing: int             # what the sum was divided by
    n_pairs: int                    # how many pairs existed before masking
    per_token: torch.Tensor         # (B, P) nats, zero where masked out
    alignment: Alignment
    extra: dict = field(default_factory=dict)

    @property
    def perplexity(self) -> float:
        return float(torch.exp(self.loss.detach()))

    @property
    def loss_over_all_pairs(self) -> float:
        """The same sum divided by B x P instead: the wrong denominator.

        Kept as a property rather than computed in passing because it is the
        third of the session's four quiet bugs, and it is worth being able to
        print the right and the wrong number from one object.
        """
        return float(self.per_token.sum() / self.n_pairs)


def lm_loss(logits, seqs: Seqs, offset: int = 1, mask=None,
            mask_padding=True, mask_boundaries=True, trace=None) -> LossRecord:
    """Cross-entropy over the pairing ``offset``, masked and correctly divided.

    ``logits`` is the full (B, T, V) tensor -- this is the ordinary path, the
    one the assignment writes and the one deliverable 7 measures the memory of.
    """
    from .model import note

    B, T, V = logits.shape
    a = align(T, offset, logits.device)
    if mask is None:
        mask = loss_mask(seqs, a, mask_padding, mask_boundaries)

    sel_logits = note(trace, f"logits[:, :-{max(offset,1)}]",
                      logits.index_select(1, a.input_pos),
                      "B=rows, P=positions that have something to predict, "
                      "V=vocab; the last positions are dropped because nothing "
                      "follows them")
    targets = note(trace, "targets", seqs.tokens.index_select(1, a.target_pos),
                   "B=rows, P=pairs; the token id each position must name. "
                   f"Position i of the row is token i+{offset} of the sequence")
    flat_logits = note(trace, "flat_logits", sel_logits.reshape(-1, V),
                       "N=B*P, one row per prediction the batch will be scored "
                       "on, V=vocab; cross_entropy has no use for the batch "
                       "and position axes, so they are folded into one")
    flat_targets = note(trace, "flat_targets", targets.reshape(-1),
                        "N=B*P; one correct id per row of flat_logits, in the "
                        "same order")

    per_token = F.cross_entropy(flat_logits, flat_targets, reduction="none")
    per_token = note(trace, "per_token_loss", per_token.view(B, -1),
                     "B=rows, P=pairs; -log(probability the model gave the "
                     "true token) at each position, before anything is masked")
    per_token = per_token * mask
    note(trace, "loss_mask", mask,
         "B=rows, P=pairs; True where the pair is allowed to contribute. "
         "Its sum is the denominator, and it is not B*P")

    n = int(mask.sum())
    loss = per_token.sum() / max(n, 1)
    note(trace, "loss", loss.detach(),
         "a scalar in nats: the mean surprise per contributing token. This is "
         "the one number the whole forward pass exists to produce")
    return LossRecord(loss, n, mask.numel(), per_token, a)


def perplexity(loss) -> float:
    return float(torch.exp(torch.as_tensor(loss)))


# --------------------------------------------------------------------------
# 4. the same loss, without ever holding all the logits
# --------------------------------------------------------------------------

def _chunk_loss_sum(h, weight, targets):
    """Logits for one chunk, its summed cross-entropy, and nothing kept.

    Written as a free function so ``checkpoint`` can call it: under
    checkpointing this runs with grad disabled in the forward pass, so the
    (chunk, V) logits are freed as soon as the sum is taken, and recomputed
    from ``h`` and ``weight`` when backward needs them.
    """
    return F.cross_entropy(F.linear(h, weight), targets, reduction="sum")


def chunked_cross_entropy(hidden, weight, seqs: Seqs, offset: int = 1,
                          mask=None, chunk_size: int = 1024,
                          recompute: bool = True, mask_padding=True,
                          mask_boundaries=True) -> LossRecord:
    """Cross-entropy that never materialises more than ``chunk_size`` logits.

    Two things do the work here.

    *Gather first.*  Masked-out positions never get logits at all, rather than
    getting them and having them multiplied by zero.  The naive path pays for
    every padded position in the batch.

    *Sum, then divide once.*  Each chunk contributes a sum of nats, and the
    division by the total contributing count happens exactly once at the end.
    Averaging the chunks' averages would reintroduce session 10's accumulation
    bug at a smaller scale: chunks hold different numbers of tokens whenever
    the count is not a multiple of the chunk size.

    ``recompute=True`` wraps each chunk in ``torch.utils.checkpoint``, which is
    what makes the saving survive into the backward pass: without it the
    autograd graph holds on to every chunk's logits and peak memory is back
    where it started.
    """
    B, T, D = hidden.shape
    a = align(T, offset, hidden.device)
    if mask is None:
        mask = loss_mask(seqs, a, mask_padding, mask_boundaries)

    h_sel = hidden.index_select(1, a.input_pos).reshape(-1, D)
    t_sel = seqs.tokens.index_select(1, a.target_pos).reshape(-1)
    flat_mask = mask.reshape(-1)
    h_sel = h_sel[flat_mask]
    t_sel = t_sel[flat_mask]

    n = int(h_sel.shape[0])
    total = hidden.new_zeros(())
    per_chunk = []
    for i in range(0, n, chunk_size):
        h, t = h_sel[i:i + chunk_size], t_sel[i:i + chunk_size]
        if recompute and torch.is_grad_enabled() and hidden.requires_grad:
            s = checkpoint(_chunk_loss_sum, h, weight, t, use_reentrant=False)
        else:
            s = _chunk_loss_sum(h, weight, t)
        per_chunk.append((int(h.shape[0]), float(s.detach())))
        total = total + s

    loss = total / max(n, 1)
    return LossRecord(loss, n, mask.numel(), torch.empty(0), a,
                      extra={"chunk_size": chunk_size,
                             "n_chunks": len(per_chunk),
                             "recompute": recompute,
                             "chunks": per_chunk})
