/* §4 — What happens if you remove the softmax.
   One claim, two routes. The direct route visits every old key and value. The
   regrouped route reads one pre-built state. With softmax off they agree exactly;
   with softmax on the regrouping stops working, because softmax's denominator is
   shared across the whole row and cannot be built before the query arrives.

   Scalar mode reproduces the lesson's numbers exactly: q=2, k=[0.5,1.0,1.5],
   v=[10,20,30] -> 140 both ways, 25.75 once softmax is switched on. */

import {
  register, el, matrixTable as mt, stepper, segmented, toggle, slider, readout,
  softmax, dot, outer, addM, matvec, transpose, zeros, maxDiff, fx, clear, rng, compact,
} from '../lib.js';

const D = 4;
const MAX_N = 6;

/* Scalar track: token i has k = 0.5i and v = 10i, so i = 1..3 is the lesson. */
const sK = (i) => 0.5 * (i + 1);
const sV = (i) => 10 * (i + 1);

/* Vector track: deterministic, rounded to 2dp so the printed arithmetic is exact. */
const R = rng(11);
const rnd = () => Math.round((R() * 2 - 1) * 100) / 100;
const VEC = Array.from({ length: MAX_N }, () => ({
  k: Array.from({ length: D }, rnd),
  v: Array.from({ length: D }, rnd),
}));
const VQ = [0.8, -0.4, 0.6, 0.2];

const STEPS = [
  { key: 'step 1', title: 'The past arrives' },
  { key: 'step 2', title: 'Direct route' },
  { key: 'step 3', title: 'Regrouped route' },
  { key: 'step 4', title: 'Compare' },
];

register(
  'w-s04-softmax-off',
  'Two routes to the same answer',
  '§4 · fixed-size state',
  'The left route visits every old key and value. The right route reads one pre-built '
  + 'state that was assembled before the query arrived. With softmax off both give the '
  + 'identical answer. Switch softmax on and the regrouping breaks — that is the trade '
  + 'linear attention makes.',
  (root) => {
    let step = 0;
    let mode = 'scalar';
    let useSoftmax = false;
    let n = 3;
    let tok = 0;      // which token's outer product step 1 works out
    let dim = 0;      // which output dimension step 3 traces through S
    let timer = null;

    const stage = el('div', { class: 'stage' });
    const ros = el('div', { class: 'readouts' });

    const seq = stepper({ steps: STEPS, value: 0, onChange: (i) => { step = i; render(); } });

    const modeSel = segmented({
      label: 'Numbers',
      options: [
        { value: 'scalar', label: 'Scalar (the lesson)', title: 'q=2, k=0.5/1.0/1.5, v=10/20/30' },
        { value: 'vector', label: 'Vectors (4-dim)', title: 'real k, v vectors and a 4×4 state' },
      ],
      value: mode,
      onChange: (v) => { mode = v; render(); },
    });

    const nSlider = slider({
      label: 'Past tokens',
      min: 1, max: MAX_N, step: 1, value: n,
      fmt: (v) => `${v}`,
      onInput: (v) => { n = v; render(); },
    });

    const smSw = toggle({
      label: 'Softmax on',
      value: false,
      onChange: (v) => { useSoftmax = v; render(); },
    });

    const replay = el('button', {
      class: 'btn',
      onclick: () => {
        clearInterval(timer);
        let i = 1;
        const target = n;
        n = 1; nSlider.set(1); step = 0; seq.set(0); render();
        timer = setInterval(() => {
          i += 1;
          if (i > target) { clearInterval(timer); timer = null; return; }
          n = i; nSlider.set(i); render();
        }, 520);
      },
    }, 'Replay the write ▸');

    root.append(
      seq.node,
      el('div', { class: 'controls' },
        modeSel.node, nSlider.node,
        el('div', { class: 'ctl' }, smSw.node),
        el('div', { class: 'ctl' }, replay)),
      stage,
      ros,
    );

    /* ---------- the two routes, computed ---------- */

    function computeScalar() {
      const q = 2;
      const ks = Array.from({ length: n }, (_, i) => sK(i));
      const vs = Array.from({ length: n }, (_, i) => sV(i));
      const scores = ks.map((k) => q * k);
      const w = useSoftmax ? softmax(scores) : scores;
      const direct = w.reduce((s, wi, i) => s + wi * vs[i], 0);
      // The state is built with no query present, so it can only ever be Σ k·v.
      const S = ks.reduce((s, k, i) => s + k * vs[i], 0);
      const state = q * S;
      return { q, ks, vs, scores, w, direct, S, state };
    }

    function computeVector() {
      const q = VQ;
      const items = VEC.slice(0, n);
      const scores = items.map((it) => dot(q, it.k));
      const w = useSoftmax ? softmax(scores) : scores;
      const direct = Array.from({ length: D }, (_, d) =>
        items.reduce((s, it, i) => s + w[i] * it.v[d], 0));
      const S = items.reduce((acc, it) => addM(acc, outer(it.v, it.k)), zeros(D, D));
      const state = matvec(S, q);
      return { q, items, scores, w, direct, S, state };
    }

    const data = () => (mode === 'scalar' ? computeScalar() : computeVector());

    /* ---------- render ---------- */

    function render() {
      clear(stage); clear(ros);
      const d = data();
      const agree = mode === 'scalar'
        ? Math.abs(d.direct - d.state) < 1e-9
        : maxDiff(d.direct, d.state) < 1e-9;

      if (step === 0) stepPast(d);
      else if (step === 1) stepDirect(d);
      else if (step === 2) stepRegrouped(d);
      else stepCompare(d, agree);

      const directNums = mode === 'scalar' ? n * 2 : n * D * 2;
      const stateNums = mode === 'scalar' ? 1 : D * D;
      addReadouts([
        {
          label: 'Softmax', value: useSoftmax ? 'on' : 'off',
          tone: useSoftmax ? 'red' : 'aqua',
          sub: useSoftmax ? 'denominator is shared' : 'each term independent',
        },
        {
          label: 'Direct route keeps', value: compact(directNums), tone: 'orange',
          sub: `${n} token${n > 1 ? 's' : ''} × k and v — grows with T`,
        },
        {
          label: 'State keeps', value: compact(stateNums), tone: 'blue',
          sub: mode === 'scalar' ? 'one number, any T' : `${D} × ${D} matrix, any T`,
        },
        {
          label: 'Routes agree?', value: agree ? 'yes' : 'no',
          tone: agree ? 'aqua' : 'red',
          sub: agree ? 'identical output' : 'regrouping is invalid',
        },
      ]);
    }

    function addReadouts(list) { list.forEach((r) => ros.append(readout(r).node)); }

    /* ---- step 1: the past arrives and the state accumulates ---- */
    function stepPast(d) {
      stage.append(
        el('p', { class: 'stage-title', text: 'The state is built before any query exists' }),
        el('p', {
          class: 'stage-sub',
          text: 'Each old token contributes key × value into one running total. Nothing here '
            + 'needs the query, so all of this can be done as the tokens arrive. Move the '
            + '"past tokens" slider — the state absorbs more information but never changes shape.',
        }),
      );

      if (mode === 'scalar') {
        const lines = ['S = 0'];
        let acc = 0;
        for (let i = 0; i < n; i++) {
          acc += sK(i) * sV(i);
          lines.push(`token ${i + 1}:  k=${fx(sK(i))}  v=${fx(sV(i), 0)}`
            + `   →   S = S + ${fx(sK(i))}×${fx(sV(i), 0)} = ${fx(acc, 1)}`);
        }
        lines.push('', `state after ${n} token${n > 1 ? 's' : ''} = ${fx(d.S, 1)}   (one number)`);
        stage.append(el('pre', { class: 'ascii', text: lines.join('\n') }));
      } else {
        stage.append(
          el('div', { class: 'gridrow' },
            mt({
              caption: `keys k₁…k${n}`, data: d.items.map((it) => it.k),
              rowLabels: d.items.map((_, i) => `k${i + 1}`), colLabels: ['0', '1', '2', '3'],
            }),
            mt({
              caption: `values v₁…v${n}`, data: d.items.map((it) => it.v),
              rowLabels: d.items.map((_, i) => `v${i + 1}`), colLabels: ['0', '1', '2', '3'],
            }),
            el('div', { class: 'op', text: '→' }),
            mt({
              caption: 'S = Σ vᵢkᵢᵀ  ·  always 4×4', data: d.S,
              rowLabels: ['0', '1', '2', '3'], colLabels: ['0', '1', '2', '3'],
              cell: (v) => ({ bg: 'rgba(57,135,229,.10)' }),
            }),
          ),
          el('div', { class: 'controls', style: { marginTop: '20px' } }, tokSel(d).node),
          outerWorked(d.items, Math.min(tok, n - 1)),
          cellWorked(d.items, d.S),
          el('p', {
            class: 'stage-sub',
            style: { margin: '4px 0 12px' },
            text: `Now add another token. Every new token turns into its own ${D}×${D} grid the `
              + 'same way, and that grid is added on top of what is already there. Watch the '
              + 'numbers change while the shape does not.',
          }),
          cumulativeStates(d.items),
        );
      }
    }

    /**
     * The running state after each token, side by side. This is the claim the
     * caption makes, drawn: the numbers inside change with every token, the
     * D×D shape never does — because adding two grids of the same shape gives a
     * grid of that shape, and never a bigger one.
     */
    function cumulativeStates(items) {
      const AX = ['0', '1', '2', '3'];
      const cum = [];
      let acc = zeros(D, D);
      items.forEach((it) => { acc = addM(acc, outer(it.v, it.k)); cum.push(acc); });

      const row = el('div', { class: 'gridrow' });
      cum.forEach((St, i) => {
        if (i > 0) {
          row.append(el('div', { class: 'op', text: '+' }));
          row.append(mt({
            caption: `v${'₁₂₃₄₅₆'[i]}k${'₁₂₃₄₅₆'[i]}ᵀ`, data: outer(items[i].v, items[i].k),
            rowLabels: AX, colLabels: AX, digits: 2,
            cell: () => ({ bg: 'rgba(255,255,255,.035)' }),
          }));
          row.append(el('div', { class: 'op', text: '=' }));
        }
        row.append(mt({
          caption: `S after ${i + 1} token${i > 0 ? 's' : ''}  ·  ${D}×${D}`,
          data: St, rowLabels: AX, colLabels: AX, digits: 2,
          cell: () => ({
            bg: i === cum.length - 1 ? 'rgba(57,135,229,.20)' : 'rgba(57,135,229,.08)',
          }),
        }));
      });

      // One cell tracked across the whole accumulation.
      const trace = cum.map((St, i) => `  after ${i + 1} token${i > 0 ? 's' : ' '}`
        + `   S is ${D}×${D}   S[0][0] = ${fx(St[0][0], 4).padStart(8)}`).join('\n');

      return el('div', {}, row, el('pre', {
        class: 'ascii',
        text: `${trace}\n\n`
          + 'The value in every cell moves. The number OF cells does not — adding two\n'
          + `${D}×${D} grids gives a ${D}×${D} grid, never a bigger one. So:\n\n`
          + `  after ${n} token${n > 1 ? 's' : ''}       →  one ${D}×${D} matrix, ${D * D} numbers\n`
          + `  after 1,000 tokens   →  one ${D}×${D} matrix, ${D * D} numbers\n`
          + `  after 1,000,000      →  one ${D}×${D} matrix, ${D * D} numbers\n\n`
          + 'A softmax KV cache would instead be holding 1,000,000 separate keys and\n'
          + '1,000,000 separate values at that point. That is the whole trade.',
      }));
    }

    const tokSel = (d) => segmented({
      label: 'Inspect token',
      options: d.items.map((_, i) => ({ value: i, label: `token ${i + 1}` })),
      value: Math.min(tok, n - 1),
      onChange: (v) => { tok = v; render(); },
    });

    /**
     * The outer product as a multiplication table: v down the side, k across the
     * top, every cell the product of its row and column. Two vectors of D numbers
     * go in; one D×D grid comes out, and the individual k and v are no longer
     * separable from it. That fusion is the compression.
     */
    function outerWorked(items, idx) {
      const { k, v } = items[idx];
      const O = outer(v, k);
      const AX = ['0', '1', '2', '3'];
      const i1 = idx + 1;

      // One width for the row label drives both the header indent and the rows,
      // so the columns cannot drift apart.
      const rowLabel = (a) => `   v${i1}[${a}] = ${fx(v[a]).padStart(6)}   `;
      const W = rowLabel(0).length;
      const lines = [
        `every cell is (its row's v) × (its column's k):`,
        '',
        `k${i1}ᵀ →`.padStart(W) + k.map((x) => fx(x).padStart(8)).join(''),
        '',
      ];
      v.forEach((vv, a) => {
        lines.push(rowLabel(a) + k.map((kk) => fx(vv * kk, 4).padStart(8)).join(''));
      });

      return el('div', {},
        // vᵢ as a COLUMN times kᵢ TRANSPOSED as a ROW — the shapes are the point.
        el('div', { class: 'gridrow' },
          mt({
            caption: `v${i1} · column (${D}×1)`, data: v.map((x) => [x]),
            rowLabels: AX, colLabels: [`v${i1}`],
            cell: () => ({ bg: 'rgba(235,104,52,.15)' }),
          }),
          el('div', { class: 'op', text: '×' }),
          mt({
            caption: `k${i1}ᵀ · row (1×${D})`, data: [k],
            rowLabels: [`k${i1}ᵀ`], colLabels: AX,
            cell: () => ({ bg: 'rgba(27,175,122,.15)' }),
          }),
          el('div', { class: 'op', text: '=' }),
          mt({
            caption: `v${i1} k${i1}ᵀ  (${D}×${D})`, data: O,
            rowLabels: AX, colLabels: AX, digits: 3,
            cell: () => ({ bg: 'rgba(57,135,229,.12)' }),
          }),
        ),
        el('pre', { class: 'ascii', text: lines.join('\n') }),
        el('pre', {
          class: 'ascii',
          text: `shapes:   (${D} × 1) × (1 × ${D})  =  (${D} × ${D})\n\n`
            + 'Compare with the score in step 2, which uses the SAME two kinds of vector\n'
            + 'the other way round:\n\n'
            + `   q · k    =  (1 × ${D}) × (${D} × 1)  =  (1 × 1)   → one number, a score\n`
            + `   v · kᵀ   =  (${D} × 1) × (1 × ${D})  =  (${D} × ${D})   → a grid, the state\n\n`
            + 'The inner dimensions are what cancel. Put the 4 on the inside and it\n'
            + 'collapses to a scalar; put the 1 on the inside and it opens out to a grid.\n\n'
            + `k${i1} and v${i1} cannot be pulled back out of that grid — that is the compression.`,
        }),
      );
    }

    /** How a single cell of S accumulates across every token seen so far. */
    function cellWorked(items, S) {
      const terms = items.map((it) => `${fx(it.v[0])}×${fx(it.k[0])}`).join('  +  ');
      const prods = items.map((it) => fx(it.v[0] * it.k[0], 4)).join('  +  ');
      return el('pre', {
        class: 'ascii',
        text: `the grids are then added. One cell, top-left, over all ${items.length} `
          + `token${items.length > 1 ? 's' : ''}:\n\n`
          + `   S[0][0] = ${terms}\n`
          + `           = ${prods}\n`
          + `           = ${fx(S[0][0], 4)}\n\n`
          + `All ${D * D} cells accumulate the same way. More tokens change the numbers\n`
          + 'inside S. They never add a cell to it.',
      });
    }

    /* ---- step 2: visit every old key ---- */
    function stepDirect(d) {
      stage.append(
        el('p', { class: 'stage-title', text: 'Direct route — the query visits every old key' }),
        el('p', {
          class: 'stage-sub',
          text: useSoftmax
            ? 'With softmax on, every score is exponentiated and divided by one shared total. '
              + 'Note that the weight for key 1 cannot be worked out without also knowing the '
              + 'scores of keys 2 and 3.'
            : 'Without softmax the raw score is used directly on each value. Every term is '
              + 'independent of the others — that independence is what the next step exploits.',
        }),
      );

      if (mode === 'scalar') {
        const lines = [`q = ${fx(d.q, 0)}`, ''];
        d.ks.forEach((k, i) => lines.push(
          `q × k${i + 1} = ${fx(d.q, 0)} × ${fx(k)} = ${fx(d.scores[i])}`));
        lines.push('');
        if (useSoftmax) {
          const ex = d.scores.map(Math.exp);
          const tot = ex.reduce((a, b) => a + b, 0);
          ex.forEach((e, i) => lines.push(`exp(${fx(d.scores[i])}) = ${fx(e, 4)}`));
          lines.push(`shared denominator = ${fx(tot, 4)}`, '');
          d.w.forEach((wi, i) => lines.push(`weight ${i + 1} = ${fx(wi, 4)}`));
          lines.push('', `output = ${d.w.map((wi, i) => `${fx(wi, 3)}×${fx(d.vs[i], 0)}`).join(' + ')}`);
        } else {
          lines.push(`output = ${d.scores.map((s, i) => `${fx(s)}×${fx(d.vs[i], 0)}`).join(' + ')}`);
          lines.push(`       = ${d.scores.map((s, i) => fx(s * d.vs[i], 1)).join(' + ')}`);
        }
        lines.push(`       = ${fx(d.direct, useSoftmax ? 2 : 1)}`);
        stage.append(el('pre', { class: 'ascii', text: lines.join('\n') }));
      } else {
        stage.append(
          mt({
            caption: useSoftmax ? 'score, then softmax weight' : 'score = q·kᵢ (used directly)',
            data: d.scores.map((s, i) => (useSoftmax ? [s, d.w[i]] : [s])),
            rowLabels: d.items.map((_, i) => `k${i + 1}`),
            colLabels: useSoftmax ? ['q·kᵢ', 'weight'] : ['q·kᵢ'],
            digits: 4,
          }),
          el('pre', {
            class: 'ascii',
            text: `q = [ ${d.q.map((v) => fx(v)).join('   ')} ]\n\n`
              + `y = Σ weightᵢ · vᵢ\n  = [ ${d.direct.map((v) => fx(v, 4)).join('   ')} ]`,
          }),
        );
      }
    }

    /* ---- step 3: read the pre-built state ---- */
    function stepRegrouped(d) {
      stage.append(
        el('p', { class: 'stage-title', text: 'Regrouped route — the query reads one state' }),
        el('p', {
          class: 'stage-sub',
          text: 'Every term in the direct route contained the same q, so q can be factored out. '
            + 'What is left inside the brackets never mentions the query, which is why it could '
            + 'be computed in advance. The query now touches one object instead of every old token.',
        }),
      );

      if (mode === 'scalar') {
        stage.append(el('pre', {
          class: 'ascii',
          text: `direct:     (q×k₁)×v₁ + (q×k₂)×v₂ + …\n`
            + `factor q:   q × (k₁v₁ + k₂v₂ + …)\n`
            + `                  └────── S = ${fx(d.S, 1)} ──────┘\n\n`
            + `output = q × S\n`
            + `       = ${fx(d.q, 0)} × ${fx(d.S, 1)}\n`
            + `       = ${fx(d.state, 1)}`
            + (useSoftmax
              ? '\n\nThe state route cannot apply softmax: it no longer holds the individual\n'
                + 'keys, so there is nothing to build a shared denominator from.'
              : ''),
        }));
      } else {
        const AX = ['0', '1', '2', '3'];
        const sub = (i) => `${'₁₂₃₄₅₆'[i]}`;
        stage.append(
          // The same factoring the scalar step shows, in vector form. The point is
          // that the bracket holds grids instead of numbers, so it sums to a grid.
          el('pre', {
            class: 'ascii',
            text: 'the same factoring, with vectors instead of single numbers:\n\n'
              + '  direct:     '
              + d.items.map((_, i) => `(q·k${sub(i)})v${sub(i)}`).join('  +  ') + '\n'
              + '              '
              + d.items.map((_, i) => `${fx(d.scores[i], 3).padStart(6)} ·v${sub(i)}`).join('  +  ')
              + '\n'
              + '                 ↑ each is one NUMBER scaling a 4-vector\n\n'
              + '  factored:   ( '
              + d.items.map((_, i) => `v${sub(i)}k${sub(i)}ᵀ`).join(' + ') + ' ) q\n'
              + '                └'
              + '─'.repeat(Math.max(4, d.items.length * 6 - 4)) + ' S '
              + '─'.repeat(Math.max(4, d.items.length * 6 - 4)) + '┘\n'
              + `                 ↑ each is one 4×4 GRID, so S is a 4×4 grid\n\n`
              + 'In the scalar version the bracket held 3 numbers and summed to ONE\n'
              + `number (70). Here it holds ${n} grid${n > 1 ? 's' : ''} of 4×4 and sums to ONE grid of 4×4.\n`
              + 'Same factoring. The object inside the bracket is just bigger.',
          }),
          el('div', { class: 'controls' }, dimSel().node),

          // The whole state as ONE matrix product. S = Σ vᵢkᵢᵀ is the same thing as
          // Vᵀ K, so V is the matrix that transposes — K is already rows-as-tokens,
          // the orientation the product needs. The inner dimension that cancels is
          // the token count, which is why S comes out D×D for any number of tokens.
          el('div', { class: 'gridrow' },
            mt({
              caption: `Vᵀ  (${D}×${n})  ·  row ${dim} highlighted`,
              data: transpose(d.items.map((it) => it.v)),
              rowLabels: AX, colLabels: d.items.map((_, i) => `v${i + 1}`),
              cell: (v, i) => ({
                bg: i === dim ? 'rgba(235,104,52,.28)' : 'rgba(235,104,52,.07)',
                cls: i === dim ? 'hl' : '',
              }),
            }),
            el('div', { class: 'op', text: '×' }),
            mt({
              caption: `K  (${n}×${D})`, data: d.items.map((it) => it.k),
              rowLabels: d.items.map((_, i) => `k${i + 1}`), colLabels: AX,
              cell: () => ({ bg: 'rgba(27,175,122,.13)' }),
            }),
            el('div', { class: 'op', text: '=' }),
            mt({
              caption: `S  (${D}×${D})  ·  row ${dim} highlighted`, data: d.S,
              rowLabels: AX, colLabels: AX,
              cell: (v, i) => ({
                bg: i === dim ? 'rgba(57,135,229,.26)' : 'rgba(57,135,229,.08)',
                cls: i === dim ? 'hl' : '',
              }),
            }),
          ),
          el('pre', {
            class: 'ascii',
            text: `S = Σᵢ vᵢkᵢᵀ  is the same thing as one matrix product:\n\n`
              + `   S = Vᵀ K      (${D} × ${n}) × (${n} × ${D})  =  (${D} × ${D})\n\n`
              + 'K needs no transposing — it already stores one token per row, which is\n'
              + 'the orientation the product wants. V is the one that flips.\n\n'
              + `The inner dimension is the TOKEN COUNT (${n}), and inner dimensions cancel.\n`
              + `That is the real reason S is ${D}×${D} for 3 tokens and still ${D}×${D} for a\n`
              + 'million: the token axis is summed away by the multiplication itself.\n\n'
              + `Row ${dim} of Vᵀ is slot ${dim} of every value — it feeds row ${dim} of S and nothing else.`,
          }),
          stateProvenance(d, dim),

          // Reading one row: (1×4) × (4×1) = (1×1), the same shape as a score.
          el('div', { class: 'gridrow' },
            mt({
              caption: `row ${dim} of S  (1×${D})`, data: [d.S[dim]],
              rowLabels: [`S[${dim}]`], colLabels: AX,
              cell: () => ({ bg: 'rgba(57,135,229,.26)' }),
            }),
            el('div', { class: 'op', text: '×' }),
            mt({
              caption: `q  (${D}×1)`, data: d.q.map((v) => [v]),
              rowLabels: AX, colLabels: ['q'],
              cell: () => ({ bg: 'rgba(144,133,233,.14)' }),
            }),
            el('div', { class: 'op', text: '=' }),
            mt({
              caption: `y[${dim}]  (1×1)`, data: [[d.state[dim]]],
              rowLabels: ['Σ'], colLabels: [`y[${dim}]`], digits: 4,
              cell: () => ({ bg: 'rgba(144,133,233,.24)' }),
            }),
          ),
          stateRead(d, dim),

          // The full output, for context — four such reads, one per row of S.
          el('div', { class: 'gridrow' },
            mt({
              caption: 'all four rows read → y = S q', data: d.state.map((v) => [v]),
              rowLabels: AX, colLabels: ['y'], digits: 4,
              cell: (v, i) => ({
                bg: i === dim ? 'rgba(144,133,233,.24)' : 'rgba(144,133,233,.08)',
                cls: i === dim ? 'hl' : '',
              }),
            }),
          ),
        );
      }
    }

    const dimSel = () => segmented({
      label: 'Output dimension',
      options: [0, 1, 2, 3].map((i) => ({ value: i, label: `y[${i}]`, title: `row ${i} of S` })),
      value: dim,
      onChange: (v) => { dim = v; render(); },
    });

    /**
     * Where row `a` of S actually came from. Every cell in that row is a sum over
     * tokens of (component a of the value) × (one component of the key) — so the
     * row only ever sees slot `a` of the values, spread across key directions.
     */
    function stateProvenance(d, a) {
      const lines = [
        `Every cell of S is one sum over tokens:   S[row][col] = Σᵢ vᵢ[row] × kᵢ[col]`,
        '',
      ];
      for (let r = 0; r < D; r++) {
        for (let b = 0; b < D; b++) {
          const terms = d.items.map((it) => `${fx(it.v[r])}×${fx(it.k[b])}`).join(' + ');
          lines.push(`  S[${r}][${b}] = ${terms} = ${fx(d.S[r][b], 4).padStart(8)}`
            + (r === a ? '   ←' : ''));
        }
        if (r < D - 1) lines.push('');
      }
      lines.push(
        '',
        `That is the whole final state — all ${D * D} cells, each one a sum over the ${n} `
        + `token${n > 1 ? 's' : ''}.`,
        `Rows marked ← are row ${a}, the one the query reads below.`,
        '',
        `Notice only vᵢ[row] appears in a given row: row ${a} of S sees slot ${a} of every`,
        'value, spread across the four key directions. No query anywhere in this.',
      );
      return el('pre', { class: 'ascii', text: lines.join('\n') });
    }

    /** Reading row a with q, then the substitution that recovers the direct route. */
    function stateRead(d, a) {
      const readTerms = d.S[a].map((s, b) => `${fx(s, 3)}×${fx(d.q[b])}`).join('  +  ');
      const readProds = d.S[a].map((s, b) => fx(s * d.q[b], 4)).join('  +  ');
      const subTerms = d.items.map((it, i) => `${fx(it.v[a])}\u00d7${fx(d.scores[i], 4)}`).join('  +  ');
      const subProds = d.items.map((it, i) => fx(it.v[a] * d.scores[i], 4)).join('  +  ');
      const viaDirect = d.items.reduce((s, it, i) => s + it.v[a] * d.scores[i], 0);

      return el('pre', {
        class: 'ascii',
        text: `Now the query reads that row:\n\n`
          + `  y[${a}]  =  ${readTerms}\n`
          + `${' '.repeat(6)}  =  ${readProds}\n`
          + `${' '.repeat(6)}  =  ${fx(d.state[a], 4)}\n\n`
          + (useSoftmax
            ? 'The state route cannot apply softmax — the individual keys are gone, so there\n'
              + 'is nothing left to build a shared denominator from. This number is therefore\n'
              + 'the un-normalised answer, and it will not match step 2.'
            : `Why that equals the direct route — substitute S and regroup:\n\n`
              + `  y[${a}] = Σ_b ( Σ_i v${'ᵢ'}[${a}] k${'ᵢ'}[b] ) q[b]\n`
              + `       = Σ_i v${'ᵢ'}[${a}] ( Σ_b k${'ᵢ'}[b] q[b] )      ← the inner sum is just q·k${'ᵢ'}\n`
              + `       = Σ_i v${'ᵢ'}[${a}] × score${'ᵢ'}\n\n`
              + `  scores q·k${'ᵢ'} = ${d.scores.map((s) => fx(s, 4)).join(', ')}\n\n`
              + `  y[${a}]  =  ${subTerms}\n`
              + `${' '.repeat(6)}  =  ${subProds}\n`
              + `${' '.repeat(6)}  =  ${fx(viaDirect, 4)}   ← same number, via the direct route`),
      });
    }

    /* ---- step 4: verdict ---- */
    function stepCompare(d, agree) {
      const dv = mode === 'scalar' ? fx(d.direct, 4) : `[ ${d.direct.map((v) => fx(v, 4)).join('  ')} ]`;
      const sv = mode === 'scalar' ? fx(d.state, 4) : `[ ${d.state.map((v) => fx(v, 4)).join('  ')} ]`;
      const diff = mode === 'scalar'
        ? Math.abs(d.direct - d.state)
        : maxDiff(d.direct, d.state);

      stage.append(
        el('p', {
          class: 'stage-title',
          text: agree
            ? 'The two routes agree exactly'
            : 'The routes no longer agree — softmax broke the regrouping',
        }),
        el('p', {
          class: 'stage-sub',
          text: agree
            ? 'Same answer, two different amounts of work. The direct route touched every old '
              + 'token; the regrouped route touched one state. That is the whole basis of linear '
              + 'attention: sequence work becomes linear instead of quadratic, and the memory '
              + 'stops growing.'
            : 'Softmax divides every score by a total that depends on all the other scores. The '
              + 'weight for one key therefore cannot be known until every key has been seen — so '
              + 'the past cannot be folded up before the query arrives. Exact softmax attention '
              + 'needs the individual old keys kept.',
        }),
        el('pre', {
          class: 'ascii',
          text: `direct route     = ${dv}\n`
            + `regrouped route  = ${sv}\n`
            + `difference       = ${diff < 1e-12 ? '0' : fx(diff, 4)}`
            + `${diff < 1e-12 && diff !== 0 ? '  (float noise below 1e-12)' : ''}\n\n`
            + (agree
              ? 'Softmax OFF → old key-value pairs can be folded into one fixed state'
              : 'Exact softmax ON → each new query must revisit the individual old keys'),
        }),
      );

      if (!agree) {
        stage.append(el('p', {
          class: 'stage-sub',
          text: 'This is the trade, not a bug. Removing softmax buys a fixed-size state and '
            + 'linear work, and gives up positive weights, weights that sum to one, competition '
            + 'between keys, and a fresh per-query distribution over exact old keys. §5 shows the '
            + 'first problem that trade creates.',
        }));
      }
    }

    render();
  },
);
