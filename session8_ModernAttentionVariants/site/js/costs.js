/* Shared cost model for the §15 and §16 boards.
   Every formula here is one already built earlier in the session — §10's cache
   bill, §11's head sharing, §12's compression and top-k, §13's schedule. Nothing
   new is introduced; the boards only combine them.

   Extension ceilings are REPORTED figures with a caveat attached wherever they
   are shown. They are evidence from specific runs, not guarantees. */

/* What the sparse-attention (G) layers do. There is deliberately no "linear"
   entry here: linear/DeltaNet state is what the D layers always do, so a purely
   linear network is expressed as gPer8 = 0 rather than as an attention family.
   Mixing them up produced a config with no mixing work at all. */
export const ATTN = {
  full: { label: 'full softmax', storesAll: true, reads: (T) => T },
  topk: { label: 'top-k sparse', storesAll: true, reads: (T, c) => Math.min(c.topK, T) },
  compressed: { label: 'compressed + top-k', storesAll: false, reads: (T, c) => Math.min(c.topK, T) },
};

export const EXTENSION = {
  none: { label: 'none — train natively', ceiling: 1, note: 'no extension: the model must be trained at the target' },
  pi: { label: 'Position Interpolation', ceiling: 16, note: 'reported 2K → 32K' },
  ntk: { label: 'NTK-aware scaling', ceiling: 32, note: 'reported around 32×' },
  yarn: { label: 'YaRN', ceiling: 32, note: 'reported 4K → 128K' },
  drope: { label: 'DroPE (V4)', ceiling: 32, note: 'reported 8K → 256K, applied before annealing' },
};

export const POSITION = {
  learned: { label: 'learned absolute table', hardWall: true },
  sinusoidal: { label: 'sinusoidal', hardWall: false },
  rope: { label: 'RoPE', hardWall: false },
};

/** Layers that carry a KV cache, and layers that carry a fixed state. */
export const layerSplit = (c) => {
  const g = Math.round((c.layers * c.gPer8) / 8);
  return { g, d: c.layers - g };
};

/** Stored K/V positions after any sequence compression (§12). */
export const storedPositions = (c, T) =>
  (ATTN[c.attn].storesAll ? T : Math.ceil(T / c.blockSize));

/** KV cache bytes for one sequence (§10 formula, §11 head sharing, §13 schedule). */
export function cacheBytes(c, T) {
  const { g } = layerSplit(c);
  return 2 * g * c.kvHeads * c.headDim * storedPositions(c, T) * c.bpn;
}

/** The fixed recurrent state, which does not grow with T (§4, §13). */
export function stateBytes(c) {
  const { d } = layerSplit(c);
  return d * c.kvHeads * c.headDim * c.headDim * c.bpn;
}

export const servingBytes = (c, T) => cacheBytes(c, T) + stateBytes(c);

/**
 * Mixing work for the whole sequence, in multiply-accumulate units, against
 * full softmax attention at the same length. Attention-bearing layers pay
 * reads × head_dim per query; fixed-state layers pay head_dim² per token.
 */
export function mixingWork(c, T) {
  const { g, d } = layerSplit(c);
  const reads = ATTN[c.attn].reads(T, c);
  const attnPart = g * c.kvHeads * T * reads * c.headDim;
  const statePart = d * c.kvHeads * T * c.headDim * c.headDim;
  return attnPart + statePart;
}

export function mixingRelativeToFull(c, T) {
  const full = { ...c, attn: 'full', gPer8: 8 };
  const denom = mixingWork(full, T);
  return denom > 0 ? mixingWork(c, T) / denom : 0;
}

/** How far position can be claimed to reach, and why. */
export function reach(c) {
  const p = POSITION[c.position];
  if (p.hardWall) {
    return { limit: c.trainedCtx, why: 'a stored table has no row beyond its training length' };
  }
  const e = EXTENSION[c.extension];
  return {
    limit: c.trainedCtx * e.ceiling,
    why: e.ceiling === 1
      ? 'no extension method: reach is the trained length'
      : `${e.label} — ${e.note}`,
  };
}

/**
 * Very rough relative training cost. Native long training pays the full
 * sequence cost; extension pays the short-context cost plus a recalibration
 * stage. Deliberately coarse and labelled as such wherever shown.
 */
export function trainingCostRel(c) {
  const nativeCtx = c.extension === 'none' ? c.targetCtx : c.trainedCtx;
  const ref = mixingWork(c, 8192);
  if (ref <= 0) return 0;
  const base = mixingWork(c, nativeCtx) / ref;
  const recal = c.extension === 'none' ? 0 : 0.1;      // ~10% for a recalibration + anneal stage
  return base * (1 + recal);
}

/* ---- human-scale document conversions, all rough and labelled ---- */
export const WORDS_PER_TOKEN_EN = 0.75;      // ~1.33 tokens per English word
export const WORDS_PER_PAGE = 500;

export const pages = (tokens, fertility) =>
  (tokens * WORDS_PER_TOKEN_EN) / fertility / WORDS_PER_PAGE;

export const DEFAULT_CONFIG = {
  layers: 32,
  qHeads: 32,
  kvHeads: 8,
  headDim: 128,
  bpn: 2,
  gPer8: 2,
  attn: 'topk',
  blockSize: 8,
  topK: 256,
  position: 'rope',
  extension: 'drope',
  trainedCtx: 8192,
  targetCtx: 262144,
  stream: true,
  chunked: true,
  batch: 8,
  memBudgetGB: 80,
  evalBeyondRetrieval: false,
};
