"""E2 — Can the code be recovered from the d-dimensional embedding?

This is the load-bearing claim. The forward path compresses an 8,224-dimensional
code into d model dimensions, and d is far smaller. Compression that small is
normally lossy, so the natural expectation is that the code cannot come back.

It can, and the reason is structural rather than lucky. The duplex code is not
an arbitrary vector in R^8224. It is a concatenation of 32 one-hot blocks of
257 classes each, so it carries exactly 32 * log2(257) = 256.4 bits and lives on
a set of size 257^32. Recovering it is not general vector reconstruction, it is
32 independent 257-way classifications, and the count of measurements a random
projection needs for that scales like the sparsity times the log of the block
size, not like the ambient dimension:

    d  >~  2 * P * ln(C)  =  2 * 32 * ln(257)  ~=  355

so a few hundred dimensions should be enough for a code that nominally needs
8,224. This experiment measures whether that holds on a real 250K vocabulary.

Three decoders are compared, none of which stores anything per token:
  pinv    - the algebraic minimum-norm inverse of the analysis matrix
  lstsq   - a least-squares synthesis matrix, the linear duplex head
  tied    - W_syn = W_ana^T, the elegant version with no synthesis parameters

and each is scored two ways: exact recovery of all 32 columns, and whether the
resulting scores identify the correct token out of the whole vocabulary. The
second is the one that matters, because that is what replaces the output head.

Noise is then injected, because at inference the model does not hand the decoder
the exact embedding of the right token. It hands it a prediction. The noise
sweep is what says whether the scheme survives contact with a real hidden state.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from duplex.codec import DUPLEX, KroneckerCodec
from duplex.vocab import load_vocab

RESULTS = Path(__file__).resolve().parent.parent / "results"


def build_codes(codec: KroneckerCodec, byte_seqs, device):
    """Sparse code table plus the two z-normalisation constants.

    For the duplex codec every column is marked, so the nonzero count is
    pos_dim for every token and both constants are scalars.
    """
    idx, mask, n = codec.build_table(byte_seqs, device=device)
    assert bool((n == codec.cfg.pos_dim).all()), "duplex codes must be fully marked"
    mu, sigma = codec.moments(int(codec.cfg.pos_dim))
    return idx, float(mu), float(sigma), float(1.0 / np.sqrt(codec.cfg.pos_dim))


def analysis(idx, W, val, mu, sigma, chunk=20000):
    """X = K_z @ W, computed as a 32-row gather per token."""
    outs = []
    colsum = W.sum(dim=0)
    for s in range(0, idx.shape[0], chunk):
        block = idx[s : s + chunk]
        outs.append((W[block].sum(dim=1) * val - mu * colsum) / sigma)
    return torch.cat(outs)


def synthesis_lstsq(X, idx, N, val, mu, sigma, ridge=1e-3):
    """Least-squares W_syn mapping embeddings back to codes."""
    d = X.shape[1]
    XtX = X.t() @ X + ridge * torch.eye(d, device=X.device, dtype=X.dtype)
    # Xt @ K_z, using the sparse structure of K
    XtK = torch.zeros(d, N, device=X.device, dtype=X.dtype)
    P = idx.shape[1]
    for p in range(P):
        XtK.index_add_(1, idx[:, p], X.t() * val)
    XtK = (XtK - mu * X.sum(dim=0, keepdim=True).t().expand(d, N)) / sigma
    return torch.linalg.solve(XtX, XtK)


def sparse_codes(idx, N, val):
    """The whole vocabulary as one sparse [V, N] matrix of raw codes.

    This is the object that replaces the D x V head: V * P nonzeros recording
    where each token's bytes fall, and not one trainable weight.
    """
    V, P = idx.shape
    rows = torch.arange(V, device=idx.device).unsqueeze(1).expand(V, P).reshape(-1)
    return torch.sparse_coo_tensor(
        torch.stack([rows, idx.reshape(-1)]),
        torch.full((V * P,), val, dtype=torch.float32, device=idx.device),
        (V, N),
    ).coalesce()


def score_vocab(Y, K, mu, sigma, chunk=2048):
    """logits[i, v] = <Y[i], code_z(v)>, as one sparse matmul. No head params.

    z-normalisation is affine, so the dense correction is a single scalar per
    row rather than a dense 8,224-wide subtraction, and the sparsity survives.
    """
    outs = []
    for s in range(0, Y.shape[0], chunk):
        y = Y[s : s + chunk]
        acc = torch.sparse.mm(K, y.t()).t()
        outs.append((acc - mu * y.sum(dim=1, keepdim=True)) / sigma)
    return torch.cat(outs)


def evaluate(Y, idx, K, mu, sigma, eval_ids):
    """Exact column recovery and vocabulary-level identification.

    These come apart, and the gap is the interesting part. Recovering the code
    exactly means winning all 32 columns at once. Identifying the token only
    means beating the other real tokens, and real tokens are a vanishingly
    small, well-separated subset of the 257^32 possible codes. Identification
    is therefore the easier problem, and it is also the only one that matters,
    because it is what the output head is for.
    """
    C, P = DUPLEX.char_dim, idx.shape[1]
    grid = Y.view(Y.shape[0], P, C)
    pred_rows = grid.argmax(dim=2)
    true_rows = idx[eval_ids] % C
    col_acc = (pred_rows == true_rows).float().mean().item()
    exact = (pred_rows == true_rows).all(dim=1).float().mean().item()

    logits = score_vocab(Y, K, mu, sigma)
    ident = (logits.argmax(dim=1) == eval_ids).float().mean().item()
    return col_acc, exact, ident


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", default="xlm-roberta-base")
    ap.add_argument("--dims", type=int, nargs="+", default=[64, 128, 192, 256, 384, 512, 768, 1024])
    ap.add_argument("--eval-n", type=int, default=8000)
    ap.add_argument("--noise", type=float, nargs="+", default=[0.0, 0.05, 0.1, 0.2, 0.4, 0.8])
    ap.add_argument("--noise-dim", type=int, default=512)
    ap.add_argument("--out", default="e2_recovery.json")
    args = ap.parse_args()

    device = "cpu"  # exact linear algebra; MPS float32 solve is less reliable here
    torch.manual_seed(0)

    vocab = load_vocab(args.vocab)
    codec = KroneckerCodec(DUPLEX)
    idx, mu, sigma, val = build_codes(codec, vocab.byte_seqs, device)
    V, N = len(vocab), DUPLEX.code_dim
    print(f"vocab {vocab.name}  V={V}  code_dim={N}  predicted d* ~ {int(2*32*np.log(257))}")

    g = torch.Generator().manual_seed(0)
    eval_ids = torch.randperm(V, generator=g)[: args.eval_n]
    K = sparse_codes(idx, N, val)

    rows = []
    for d in args.dims:
        W = torch.randn(N, d, generator=g) / np.sqrt(N)
        X = analysis(idx, W, val, mu, sigma)

        rec = {"d": d, "compression": N / d}
        # pinv: algebraic inverse of the analysis map
        Wp = torch.linalg.pinv(W)                       # [d, N] -> acts as synthesis
        for name, Wsyn in [
            ("pinv", Wp),
            ("lstsq", synthesis_lstsq(X, idx, N, val, mu, sigma)),
            ("tied", W.t()),
        ]:
            Y = X[eval_ids] @ Wsyn
            col, exact, ident = evaluate(Y, idx, K, mu, sigma, eval_ids)
            rec[name] = {"column_acc": col, "exact_code": exact, "token_identify": ident}
            print(f"  d={d:<5d} {name:6s} col={col*100:6.2f}%  exact={exact*100:6.2f}%  id={ident*100:6.2f}%", flush=True)
        rows.append(rec)

    # noise sweep at a fixed width
    d = args.noise_dim
    W = torch.randn(N, d, generator=g) / np.sqrt(N)
    X = analysis(idx, W, val, mu, sigma)
    Wsyn = synthesis_lstsq(X, idx, N, val, mu, sigma)
    sig_rms = X.pow(2).mean().sqrt().item()
    noise_rows = []
    print(f"\nnoise sweep at d={d} (signal rms {sig_rms:.4f})")
    for nl in args.noise:
        Xe = X[eval_ids] + torch.randn(len(eval_ids), d, generator=g) * (nl * sig_rms)
        col, exact, ident = evaluate(Xe @ Wsyn, idx, K, mu, sigma, eval_ids)
        noise_rows.append({"noise": nl, "column_acc": col, "exact_code": exact, "token_identify": ident})
        print(f"  noise={nl:<5.2f} col={col*100:6.2f}%  exact={exact*100:6.2f}%  id={ident*100:6.2f}%", flush=True)

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / args.out).write_text(
        json.dumps(
            {
                "vocab": vocab.name,
                "vocab_size": V,
                "code_dim": N,
                "predicted_d_star": int(2 * 32 * np.log(257)),
                "eval_n": len(eval_ids),
                "sweep": rows,
                "noise_dim": d,
                "noise": noise_rows,
            },
            indent=2,
        )
    )
    print(f"\nwrote {RESULTS / args.out}")


if __name__ == "__main__":
    main()
