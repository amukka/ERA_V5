/* §2 — Six tokens, four numbers each, every value computed.
   Five steps: project, score, scale+softmax, causal mask, weighted sum.

   The embeddings are hand-built on four readable feature axes so the
   query-key match the prose promises ("sat" looks for "cat") is a real
   consequence of the arithmetic rather than a drawn picture. */

import {
  register, el, matrixTable as mt, stepper, segmented, toggle,
  readout, matmul, transpose, softmax, dot, fx, heat, signedHeat, clear,
} from '../lib.js';

const TOKENS = ['The', 'cat', 'sat', 'on', 'the', 'mat'];
const FEATS = ['det', 'anim', 'act', 'place'];
const D = 4;
const SQRT_DK = Math.sqrt(D);

// Token embeddings on the four feature axes above.
const X = [
  [1.0, 0.0, 0.0, 0.1], // The
  [0.1, 1.0, 0.1, 0.0], // cat
  [0.0, 0.2, 1.0, 0.1], // sat
  [0.2, 0.0, 0.2, 0.6], // on
  [1.0, 0.0, 0.0, 0.1], // the
  [0.1, 0.1, 0.0, 1.0], // mat
];

// Wq encodes "what does this kind of word look for?" Row i = input feature i.
// An action word (row 2) looks hard for an animate word; a place word (row 3)
// looks for its determiner.
const Wq = [
  [0.0, 0.6, 0.0, 0.6],
  [1.6, 0.0, 0.0, 0.0],
  [0.0, 2.4, 0.0, 0.6],
  [1.4, 0.0, 0.4, 0.0],
];

// Wk is the identity so a key is literally the token's content vector. Nothing
// in the mechanism requires that — it just keeps the key grid readable.
const Wk = [
  [1, 0, 0, 0],
  [0, 1, 0, 0],
  [0, 0, 1, 0],
  [0, 0, 0, 1],
];

// Wv mixes features into what a token hands over when chosen.
const Wv = [
  [0.2, 0.1, 0.0, 0.1],
  [0.1, 0.9, 0.2, 0.0],
  [0.0, 0.2, 0.9, 0.1],
  [0.1, 0.0, 0.1, 0.8],
];

const Q = matmul(X, Wq);
const K = matmul(X, Wk);
const V = matmul(X, Wv);

const KT = transpose(K);
const RAW = matmul(Q, KT);                           // QK^T
const SCALED = RAW.map((r) => r.map((v) => v / SQRT_DK));

// Wv genuinely mixes the four axes, so V's columns are no longer those features.
// Q and K keep the feature names because a query column reads as "how much do I
// want feature j", which is exactly what it is compared against in a key.
const V_COLS = ['v0', 'v1', 'v2', 'v3'];

const PROJ = {
  Q: {
    name: 'Wq', W: Wq, out: Q, cols: FEATS,
    blurb: 'Wq turns a token into what it is looking for. Read a result column as '
      + '"how much do I want this feature?" — which is why a query can be compared '
      + 'against a key at all. The 2.4 in the act→anim slot is the largest weight in '
      + 'the matrix: it is what makes an action word go hunting for something animate.',
  },
  K: {
    name: 'Wk', W: Wk, out: K, cols: FEATS,
    blurb: 'Wk is the identity matrix here, so every key is literally the token\'s own '
      + 'embedding — that is why this result grid looks identical to X. It keeps the score '
      + 'grid readable. Real models learn a full Wk and keys are not the embeddings.',
  },
  V: {
    name: 'Wv', W: Wv, out: V, cols: V_COLS,
    blurb: 'Wv decides what a token hands over once it has been chosen. Unlike Wk it really '
      + 'mixes the four axes, so its output columns are no longer the original features. '
      + 'They are labelled v0–v3 rather than pretending otherwise.',
  },
};

const rowsWith = (masked) =>
  SCALED.map((row, i) =>
    softmax(row.map((v, j) => (masked && j > i ? -Infinity : v))));

const STEPS = [
  { key: 'step 1', title: 'Project Q, K, V' },
  { key: 'step 2', title: 'Score every pair' },
  { key: 'step 3', title: 'Scale, then softmax' },
  { key: 'step 4', title: 'Causal mask' },
  { key: 'step 5', title: 'Weighted sum' },
];

register(
  'w-s02-walkthrough',
  'Attention, one value at a time',
  '§2 · 6 tokens · d_k = 4',
  'Six tokens, four numbers each, with every value computed live. Walk the five steps. '
  + 'On the mask step, turn the mask off — attention weight moves onto tokens that have not '
  + 'happened yet. That is exactly the problem the causal mask solves.',
  (root) => {
    let step = 0;
    let query = 2;      // "sat"
    let masked = true;
    let inspect = 'Q';  // which weight matrix step 1 is showing
    let keyIdx = 1;     // "cat" — which score cell step 2 works out

    const stage = el('div', { class: 'stage' });
    const ros = el('div', { class: 'readouts' });

    const seq = stepper({ steps: STEPS, value: 0, onChange: (i) => { step = i; render(); } });

    const qSel = segmented({
      label: 'Query token',
      options: TOKENS.map((t, i) => ({ value: i, label: t, title: `query from position ${i}` })),
      value: query,
      onChange: (v) => { query = v; render(); },
    });

    const maskSw = toggle({
      label: 'Causal mask on',
      value: true,
      onChange: (v) => { masked = v; render(); },
    });

    root.append(
      seq.node,
      el('div', { class: 'controls' }, qSel.node, el('div', { class: 'ctl' }, maskSw.node)),
      stage,
      ros,
    );

    function render() {
      clear(stage); clear(ros);
      const W = rowsWith(masked);
      const w = W[query];

      // ---- step 1: projections
      if (step === 0) {
        const p = PROJ[inspect];
        const projSel = segmented({
          label: 'Inspect projection',
          options: Object.keys(PROJ).map((k) => ({
            value: k, label: `${k} = X·${PROJ[k].name}`,
          })),
          value: inspect,
          onChange: (v) => { inspect = v; render(); },
        });

        stage.append(
          el('p', { class: 'stage-title', text: 'One vector in, three vectors out' }),
          el('p', {
            class: 'stage-sub',
            text: 'Each row is one token. The same embedding row is multiplied by three '
              + 'different weight matrices, giving that token a query, a key and a value. '
              + 'Pick a projection below to see its weight matrix and the exact arithmetic '
              + `for the highlighted token, "${TOKENS[query]}".`,
          }),
          el('div', { class: 'controls' }, projSel.node),

          // X · W = result, with the actual weight matrix on screen
          el('div', { class: 'gridrow' },
            mt({
              caption: 'X · embeddings', data: X, rowLabels: TOKENS, colLabels: FEATS,
              cell: (v, i) => ({ bg: i === query ? 'rgba(57,135,229,.16)' : null }),
            }),
            el('div', { class: 'op', text: '×' }),
            mt({
              caption: `${p.name} · weights`, data: p.W,
              rowLabels: FEATS.map((f) => `${f} →`), colLabels: p.cols,
              cell: (v) => ({
                bg: Math.abs(v) > 1e-9 ? 'rgba(144,133,233,.14)' : null,
                color: Math.abs(v) < 1e-9 ? '#4a4a52' : null,
              }),
            }),
            el('div', { class: 'op', text: '=' }),
            mt({
              caption: `${inspect} = X·${p.name}`, data: p.out,
              rowLabels: TOKENS, colLabels: p.cols,
              cell: (v, i) => ({ cls: i === query ? 'hl' : '' }),
            }),
          ),

          el('p', { class: 'stage-sub', style: { margin: '20px 0 10px' }, text: p.blurb }),
          workedProjection(query, inspect),
        );
        addReadouts([
          { label: 'Tokens', value: '6', sub: 'sequence length T' },
          { label: 'Width', value: '4', sub: 'head dim d_k' },
          { label: 'Weights in this matrix', value: '16', sub: `${p.name} is 4 × 4` },
          {
            label: 'Non-zero weights', value: String(p.W.flat().filter((v) => Math.abs(v) > 1e-9).length),
            sub: 'the rest contribute nothing', tone: 'violet',
          },
        ]);
        return;
      }

      // ---- step 2: raw scores
      if (step === 1) {
        const maxAbs = Math.max(...RAW.flat().map(Math.abs));
        const kSel = segmented({
          label: 'Key token (the column)',
          options: TOKENS.map((t, i) => ({ value: i, label: t, title: `key at position ${i}` })),
          value: keyIdx,
          onChange: (v) => { keyIdx = v; render(); },
        });

        stage.append(
          el('p', { class: 'stage-title', text: 'Every query meets every key: 6 × 6 = 36 scores' }),
          el('p', {
            class: 'stage-sub',
            text: 'Row = the token asking. Column = the token being looked at. Each cell is one '
              + 'dot product q·k. Nothing has been masked or normalised yet — this is the full '
              + 'all-to-all comparison, and it is the reason attention costs T² work. Pick a '
              + 'query row and a key column to see that one cell worked out in full.',
          }),
          el('div', { class: 'controls' }, kSel.node),

          // Q (6×4) × Kᵀ (4×6) = scores (6×6). Kᵀ is shown transposed so its
          // columns line up with the columns of the score grid.
          el('div', { class: 'gridrow' },
            mt({
              caption: 'Q · one row per query', data: Q, rowLabels: TOKENS, colLabels: FEATS,
              cell: (v, i) => ({
                cls: i === query ? 'hl' : '',
                bg: i === query ? 'rgba(57,135,229,.16)' : null,
              }),
            }),
            el('div', { class: 'op', text: '×' }),
            mt({
              caption: 'Kᵀ · one column per key', data: KT, rowLabels: FEATS, colLabels: TOKENS,
              cell: (v, i, j) => ({
                cls: j === keyIdx ? 'hl' : '',
                bg: j === keyIdx ? 'rgba(27,175,122,.16)' : null,
              }),
            }),
          ),
          el('div', { class: 'gridrow', style: { marginTop: '20px' } },
            mt({
              caption: 'QKᵀ · raw scores', data: RAW, rowLabels: TOKENS, colLabels: TOKENS,
              cell: (v, i, j) => ({
                bg: signedHeat(v, maxAbs),
                cls: (i === query && j === keyIdx) ? 'hl' : '',
              }),
            }),
          ),
          workedDot(query, keyIdx),
        );

        const best = RAW[query].indexOf(Math.max(...RAW[query]));
        addReadouts([
          { label: 'Scores computed', value: '36', sub: 'T × T' },
          {
            label: 'Selected cell', value: fx(RAW[query][keyIdx]), tone: 'blue',
            sub: `q("${TOKENS[query]}") · k("${TOKENS[keyIdx]}")`,
          },
          {
            label: 'Best key for this query', value: TOKENS[best], tone: 'aqua',
            sub: `score ${fx(RAW[query][best])}`,
          },
          { label: 'At T = 10,000', value: '100M', sub: 'the first bill, §3', tone: 'orange' },
        ]);
        return;
      }

      // ---- step 3: scale + softmax, no mask
      if (step === 2) {
        const unmasked = SCALED.map((row) => softmax(row));
        const rowMax = Math.max(...unmasked[query]);
        stage.append(
          el('p', { class: 'stage-title', text: 'Divide by √d_k = 2, then softmax each row' }),
          el('p', {
            class: 'stage-sub',
            text: 'Scaling keeps the numbers in a range where softmax is not saturated — the '
              + 'weights come out flatter than the raw scores suggest. Softmax then turns each '
              + 'row into positive weights that add to exactly 1.',
          }),
          el('div', { class: 'gridrow' },
            mt({
              caption: 'QKᵀ / √d_k', data: SCALED, rowLabels: TOKENS, colLabels: TOKENS,
              cell: (v, i) => ({ cls: i === query ? 'hl' : '' }),
            }),
            el('div', { class: 'op', text: '→' }),
            mt({
              caption: 'softmax rows (no mask yet)', data: unmasked,
              rowLabels: TOKENS, colLabels: TOKENS, digits: 3,
              cell: (v, i) => ({ bg: heat(v), cls: i === query ? 'hl' : '' }),
            }),
          ),
          el('pre', {
            class: 'ascii',
            text: `row for "${TOKENS[query]}" sums to `
              + `${fx(unmasked[query].reduce((a, b) => a + b, 0), 4)}`,
          }),
        );
        addReadouts([
          { label: '√d_k', value: '2.00', sub: 'd_k = 4' },
          { label: 'Row sum', value: '1.0000', sub: 'every row, after softmax' },
          {
            label: 'Max weight in row', value: fx(rowMax, 3),
            sub: `on "${TOKENS[unmasked[query].indexOf(rowMax)]}"`,
          },
          { label: 'Mask applied', value: 'no', sub: 'next step fixes this', tone: 'red' },
        ]);
        return;
      }

      // ---- step 4: mask
      if (step === 3) {
        const open = softmax(SCALED[query]);
        const leak = masked ? 0 : open.reduce((s, v, j) => s + (j > query ? v : 0), 0);
        stage.append(
          el('p', {
            class: 'stage-title',
            text: masked
              ? 'Mask on: the future gets exactly zero'
              : 'Mask off: the model is reading the future',
          }),
          el('p', {
            class: 'stage-sub',
            text: masked
              ? 'M is 0 where attention is allowed and −∞ above the diagonal. It is added before '
                + 'softmax, and softmax(−∞) = 0, so every forbidden cell receives weight zero. '
                + 'This is what lets the whole sequence train in parallel while still generating '
                + 'left to right.'
              : `Every cell above the diagonal now carries real weight. Token "${TOKENS[query]}" `
                + 'is using information from positions that have not been generated yet — the '
                + 'model would look excellent during training and fall apart at generation time.',
          }),
          mt({
            caption: masked ? 'masked softmax weights' : 'UNMASKED — future leaking in',
            data: W, rowLabels: TOKENS, colLabels: TOKENS, digits: 3,
            cell: (v, i, j) => ({
              text: masked && j > i ? '0' : fx(v, 3),
              cls: (masked && j > i) ? 'masked' : (!masked && j > i && v > 0.001 ? 'future' : ''),
              bg: (j <= i || !masked) ? heat(v) : null,
            }),
          }),
        );
        addReadouts([
          {
            label: 'Mask', value: masked ? 'on' : 'off', tone: masked ? 'aqua' : 'red',
            sub: masked ? 'M = −∞ above diagonal' : 'no M added',
          },
          { label: 'Allowed cells', value: masked ? '21' : '36', sub: masked ? 'T(T+1)/2' : 'all pairs' },
          {
            label: 'Weight on the future', value: masked ? '0.000' : fx(leak, 3),
            tone: masked ? 'aqua' : 'red', sub: `row "${TOKENS[query]}"`,
          },
          {
            label: 'Visible to this query', value: String(masked ? query + 1 : 6),
            sub: 'tokens it may read',
          },
        ]);
        return;
      }

      // ---- step 5: weighted sum
      const out = V[0].map((_, d) => w.reduce((s, wi, j) => s + wi * V[j][d], 0));
      const contrib = w.map((wi, j) => V[j].map((v) => wi * v));
      stage.append(
        el('p', { class: 'stage-title', text: `One output vector for "${TOKENS[query]}"` }),
        el('p', {
          class: 'stage-sub',
          text: 'Each value row is multiplied by its weight, then the rows are added down the '
            + 'columns. Masked rows contribute nothing because their weight is zero. The four '
            + 'numbers at the bottom are what leaves the attention layer for this token.',
        }),
        el('div', { class: 'gridrow' },
          mt({
            caption: 'weight', data: w.map((v) => [v]), rowLabels: TOKENS, colLabels: ['w'],
            digits: 3,
            cell: (v) => ({ bg: heat(v), cls: v < 1e-9 ? 'masked' : '' }),
          }),
          el('div', { class: 'op', text: '×' }),
          mt({ caption: 'V', data: V, rowLabels: TOKENS, colLabels: V_COLS }),
          el('div', { class: 'op', text: '=' }),
          // The Σ row lives inside this table on purpose: the output is produced BY
          // adding these columns, so it must not look like another cell alongside them.
          mt({
            caption: 'weighted contribution → add ↓',
            data: [...contrib, out],
            rowLabels: [...TOKENS, 'Σ'],
            colLabels: V_COLS,
            digits: 3,
            cell: (v, i) => (i === TOKENS.length
              ? { bg: 'rgba(144,133,233,.20)', cls: 'sum' }
              : { cls: w[i] < 1e-9 ? 'masked' : '', bg: heat(w[i] * 0.9) }),
          }),
        ),
        columnSums(w, out),
      );
      const top = w.indexOf(Math.max(...w));
      addReadouts([
        { label: 'Values summed', value: String(masked ? query + 1 : 6), sub: 'non-zero weights' },
        { label: 'Dominant source', value: TOKENS[top], sub: `weight ${fx(w[top], 3)}`, tone: 'blue' },
        { label: 'Output width', value: '4', sub: 'same as input' },
        { label: 'Weight total', value: fx(w.reduce((a, b) => a + b, 0), 4), sub: 'softmax guarantee' },
      ]);
    }

    function addReadouts(list) {
      list.forEach((r) => ros.append(readout(r).node));
    }

    /**
     * The vertical addition, spelled out. The output numbers are produced BY
     * summing the contribution columns, so they appear nowhere inside that grid —
     * which is exactly the point people get stuck on.
     */
    function columnSums(w, out) {
      const lines = [
        'the output is not a cell in the grid above — it is the SUM of each column:',
        '',
      ];
      V_COLS.forEach((c, d) => {
        const terms = w.map((wi, i) => (wi < 1e-9 ? '0' : fx(contribAt(w, i, d), 3))).join(' + ');
        lines.push(`  ${c}:  ${terms}  =  ${fx(out[d], 3)}`);
      });
      lines.push(
        '',
        `six rows collapse into one. That row is what leaves the layer for "${TOKENS[query]}".`,
      );
      return el('pre', { class: 'ascii', text: lines.join('\n') });
    }

    const contribAt = (w, i, d) => w[i] * V[i][d];

    /**
     * The four output numbers of one projected row, each written out in full.
     * Output number j is a dot product of the token's embedding with COLUMN j of
     * the weight matrix — showing all four terms, zeros included, is the whole
     * point, so nothing has to be taken on trust.
     */
    function workedProjection(i, key) {
      const p = PROJ[key];
      const x = X[i];
      const pad = (s, n) => String(s).padStart(n);
      const lines = [
        `x("${TOKENS[i]}")  =  [ ${x.map((v) => fx(v)).join('   ')} ]`,
        '',
        `each output number pairs x with one COLUMN of ${p.name}:`,
        '',
      ];
      p.cols.forEach((c, j) => {
        const col = p.W.map((r) => r[j]);
        const terms = x.map((xv, d) => `${fx(xv)}×${fx(col[d])}`).join('  +  ');
        const sum = x.reduce((s, xv, d) => s + xv * col[d], 0);
        const head = `  ${key}[${j}]  ${pad(`(${c})`, 7)}`;
        lines.push(
          `${head}  =  ${terms}`,
          `${' '.repeat(head.length)}  =  ${fx(sum)}`,
          '',
        );
      });
      lines.push(`${key}("${TOKENS[i]}")  =  [ ${p.out[i].map((v) => fx(v)).join('   ')} ]`);
      return el('pre', { class: 'ascii', text: lines.join('\n') });
    }

    /**
     * One cell of QKᵀ written out in full: pair the query row with the key row
     * component by component, multiply, add. i = query position, j = key position.
     */
    function workedDot(i, j) {
      const best = RAW[i].indexOf(Math.max(...RAW[i]));
      const score = dot(Q[i], K[j]);
      const terms = Q[i].map((qv, d) => `${fx(qv)}×${fx(K[j][d])}`).join('  +  ');
      const products = Q[i].map((qv, d) => fx(qv * K[j][d])).join('  +  ');
      const note = j === best
        ? '  ← highest score in this row'
        : `  (highest in this row is "${TOKENS[best]}" at ${fx(RAW[i][best])})`;
      return el('pre', {
        class: 'ascii',
        text: `q("${TOKENS[i]}")  =  [ ${Q[i].map((v) => fx(v)).join('   ')} ]\n`
          + `k("${TOKENS[j]}")  =  [ ${K[j].map((v) => fx(v)).join('   ')} ]\n`
          + '\n'
          + `q · k  =  ${terms}\n`
          + `       =  ${products}\n`
          + `       =  ${fx(score)}${note}`
          + (j > i
            ? `\n\nNote: key "${TOKENS[j]}" is AFTER query "${TOKENS[i]}". This score is `
              + 'computed anyway —\nthe causal mask in step 4 is what removes it.'
            : ''),
      });
    }

    render();
  },
);
