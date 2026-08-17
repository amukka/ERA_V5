/* §7 — The other lever: do not look at everything.
   Softmax is kept. What changes is how many keys each query is allowed to use.

   The widget's real job is the catch: value work falls with k, but the NAIVE
   selection cost stays at all T scores, because finding the best k requires
   scoring every candidate first. If scoring was the expensive part, naive top-k
   has not solved anything. */

import {
  register, el, matrixTable as mt, stepper, slider, toggle, readout, svg,
  softmax, dot, topKIndices, fx, clear, rng, heat,
} from '../lib.js';

const T = 12;                      // candidate keys, as in the lesson
const D = 4;
const SQRT_DK = Math.sqrt(D);

/* Deterministic keys, values and one query, rounded so the arithmetic is exact. */
const R = rng(23);
const rnd = () => Math.round((R() * 2 - 1) * 100) / 100;
const KEYS = Array.from({ length: T }, () => Array.from({ length: D }, rnd));
const VALS = Array.from({ length: T }, () => Array.from({ length: D }, rnd));
const Q = [0.7, 0.5, -0.6, 0.3];

const SCORES = KEYS.map((k) => dot(Q, k) / SQRT_DK);
const FULL_W = softmax(SCORES);
const FULL_OUT = Array.from({ length: D }, (_, d) =>
  FULL_W.reduce((s, w, i) => s + w * VALS[i][d], 0));

/** Top-k attention: keep the k best scores, softmax over those alone. */
function topk(k) {
  const keep = topKIndices(SCORES, k);
  const masked = SCORES.map((s, i) => (keep.includes(i) ? s : -Infinity));
  const w = softmax(masked);
  const out = Array.from({ length: D }, (_, d) =>
    w.reduce((s, wi, i) => s + wi * VALS[i][d], 0));
  const err = Math.sqrt(out.reduce((s, v, d) => s + (v - FULL_OUT[d]) ** 2, 0));
  const mass = keep.reduce((s, i) => s + FULL_W[i], 0);   // full-attention weight kept
  return { keep, w, out, err, mass };
}

const STEPS = [
  { key: 'step 1', title: 'Score every candidate' },
  { key: 'step 2', title: 'Keep the best k' },
  { key: 'step 3', title: 'Softmax and combine' },
  { key: 'step 4', title: 'The catch' },
];

register(
  'w-s07-topk',
  'Reading fewer keys',
  '§7 · top-k sparse attention',
  'Softmax stays exactly as it was. The only change is that each query uses a small number '
  + 'of keys instead of all of them. Move k and watch the final sum shrink — then look at '
  + 'what the selection step still costs.',
  (root) => {
    let step = 0;
    let k = 4;
    let showDropped = true;

    const stage = el('div', { class: 'stage' });
    const ros = el('div', { class: 'readouts' });

    const seq = stepper({ steps: STEPS, value: 0, onChange: (i) => { step = i; render(); } });

    const kSlider = slider({
      label: 'Budget k',
      min: 1, max: T, step: 1, value: k,
      fmt: (v) => `${v} of ${T} keys`,
      onInput: (v) => { k = v; render(); },
    });

    const dropSw = toggle({
      label: 'Show dropped keys',
      value: true,
      onChange: (v) => { showDropped = v; render(); },
    });

    root.append(
      seq.node,
      el('div', { class: 'controls' }, kSlider.node, el('div', { class: 'ctl' }, dropSw.node)),
      stage,
      ros,
    );

    function render() {
      clear(stage); clear(ros);
      const r = topk(k);

      if (step === 0) stepScore(r);
      else if (step === 1) stepSelect(r);
      else if (step === 2) stepCombine(r);
      else stepCatch(r);

      addReadouts([
        { label: 'Budget k', value: String(k), tone: 'blue', sub: `of ${T} candidates` },
        {
          label: 'Values summed', value: String(k), tone: 'blue',
          sub: `was ${T} — this is the saving`,
        },
        {
          label: 'Scores computed', value: String(T), tone: 'orange',
          sub: 'naive selection — never falls',
        },
        {
          label: 'Attention kept', value: `${fx(r.mass * 100, 1)}%`,
          tone: r.mass > 0.9 ? 'aqua' : 'red',
          sub: 'of full-attention weight',
        },
        {
          label: 'Output error', value: fx(r.err, 4),
          tone: r.err < 0.01 ? 'aqua' : r.err < 0.1 ? 'orange' : 'red',
          sub: 'distance from full attention',
        },
      ]);
    }

    function addReadouts(list) { list.forEach((r) => ros.append(readout(r).node)); }

    const rowLabels = KEYS.map((_, i) => `k${i + 1}`);

    /* ---- step 1 ---- */
    function stepScore(r) {
      stage.append(
        el('p', { class: 'stage-title', text: `One query against all ${T} candidate keys` }),
        el('p', {
          class: 'stage-sub',
          text: 'Nothing is sparse yet. Every candidate gets a full q·k score, exactly as in §2. '
            + 'This is the step that top-k does not avoid — hold that thought for step 4.',
        }),
        el('div', { class: 'gridrow' },
          mt({
            caption: 'candidate keys', data: KEYS, rowLabels, colLabels: ['0', '1', '2', '3'],
            cell: () => ({ bg: 'rgba(27,175,122,.10)' }),
          }),
          el('div', { class: 'op', text: '·q' }),
          mt({
            caption: 'score / √d_k', data: SCORES.map((s) => [s]),
            rowLabels, colLabels: ['s'], digits: 4,
            cell: (v) => ({ bg: v > 0 ? heat(v / 1.2) : null }),
          }),
        ),
        el('pre', {
          class: 'ascii',
          text: `q = [ ${Q.map((v) => fx(v)).join('   ')} ]\n\n`
            + `all ${T} scores computed:  ${SCORES.map((s) => fx(s, 2)).join('  ')}\n`
            + `highest: k${SCORES.indexOf(Math.max(...SCORES)) + 1} at ${fx(Math.max(...SCORES), 4)}`
            + `   lowest: k${SCORES.indexOf(Math.min(...SCORES)) + 1} at ${fx(Math.min(...SCORES), 4)}`,
        }),
      );
    }

    /* ---- step 2 ---- */
    function stepSelect(r) {
      const ranked = SCORES.map((s, i) => ({ s, i }))
        .sort((a, b) => b.s - a.s);
      stage.append(
        el('p', { class: 'stage-title', text: `Sort, then keep the top ${k}` }),
        el('p', {
          class: 'stage-sub',
          text: `The ${k} highest-scoring keys survive; the other ${T - k} are dropped before the `
            + 'output is formed. Dropped keys are not down-weighted — they are removed, so they '
            + 'contribute exactly nothing.',
        }),
        el('pre', {
          class: 'ascii',
          text: 'ranked by score:\n\n'
            + ranked.map((x, rank) => `  ${(rank + 1 + '.').padStart(4)} k${String(x.i + 1).padEnd(3)}`
              + `score ${fx(x.s, 4).padStart(8)}   `
              + (rank < k ? '█ kept' : '· dropped')).join('\n')
            + `\n\n  cut-off after ${k}: score ${fx(ranked[k - 1].s, 4)}`,
        }),
      );
    }

    /* ---- step 3 ---- */
    function stepCombine(r) {
      const rows = showDropped ? KEYS.map((_, i) => i) : r.keep;
      stage.append(
        el('p', { class: 'stage-title', text: `Softmax over ${k}, then the weighted sum` }),
        el('p', {
          class: 'stage-sub',
          text: `Softmax runs over the ${k} survivors only, so the denominator is a total over `
            + `${k} terms instead of ${T}. The weights still add to one — they are simply shared `
            + 'among fewer keys, so each survivor gets a larger share than it had under full '
            + 'attention. Same renormalisation as the causal mask in §2.',
        }),
        el('div', { class: 'gridrow' },
          mt({
            caption: 'top-k weight', data: rows.map((i) => [r.w[i]]),
            rowLabels: rows.map((i) => `k${i + 1}`), colLabels: ['w'], digits: 4,
            cell: (v) => ({ bg: heat(v * 1.4), cls: v < 1e-12 ? 'masked' : '' }),
          }),
          mt({
            caption: 'full-attention weight', data: rows.map((i) => [FULL_W[i]]),
            rowLabels: rows.map((i) => `k${i + 1}`), colLabels: ['w'], digits: 4,
            cell: (v) => ({ bg: heat(v * 1.4) }),
          }),
          el('div', { class: 'op', text: '×' }),
          mt({
            caption: 'values', data: rows.map((i) => VALS[i]),
            rowLabels: rows.map((i) => `v${i + 1}`), colLabels: ['0', '1', '2', '3'],
            cell: (v, i) => ({ cls: r.w[rows[i]] < 1e-12 ? 'masked' : '' }),
          }),
        ),
        el('div', { class: 'gridrow', style: { marginTop: '18px' } },
          mt({
            caption: `top-k output (k=${k})`, data: r.out.map((v) => [v]),
            rowLabels: ['0', '1', '2', '3'], colLabels: ['y'], digits: 4,
            cell: () => ({ bg: 'rgba(144,133,233,.22)' }),
          }),
          mt({
            caption: 'full attention output', data: FULL_OUT.map((v) => [v]),
            rowLabels: ['0', '1', '2', '3'], colLabels: ['y'], digits: 4,
            cell: () => ({ bg: 'rgba(255,255,255,.05)' }),
          }),
          mt({
            caption: 'difference', data: r.out.map((v, d) => [v - FULL_OUT[d]]),
            rowLabels: ['0', '1', '2', '3'], colLabels: ['Δ'], digits: 4,
            cell: (v) => ({
              bg: Math.abs(v) > 0.05 ? 'rgba(224,82,82,.22)' : 'rgba(27,175,122,.14)',
            }),
          }),
        ),
        el('pre', {
          class: 'ascii',
          text: `weights sum to ${fx(r.w.reduce((a, b) => a + b, 0), 4)} over ${k} key`
            + `${k > 1 ? 's' : ''}\n`
            + `these ${k} key${k > 1 ? 's' : ''} held ${fx(r.mass * 100, 1)}% of full attention's weight\n`
            + `output moved ${fx(r.err, 4)} away from the full-attention answer`,
        }),
      );
    }

    /* ---- step 4: the catch ---- */
    function stepCatch(r) {
      stage.append(
        el('p', { class: 'stage-title', text: 'Naive top-k does not reduce the scoring cost' }),
        el('p', {
          class: 'stage-sub',
          text: 'To know which k keys are best, every candidate has to be scored and the scores '
            + 'sorted. The final sum uses k values, but the selection still touched all '
            + `${T}. If scoring was the expensive part, nothing has been saved.`,
        }),
        el('pre', {
          class: 'ascii',
          text: `  score all ${T} keys  →  discover top ${k}  →  use only ${k} values\n`
            + `  └──── still ${T} ────┘                     └── saved ──┘`,
        }),
        costBars(),
        el('p', {
          class: 'stage-sub',
          text: 'Practical sparse-attention systems therefore need a cheaper proposal step — a '
            + 'local window, a learned router, or a compressed index that suggests where to look. '
            + 'Exact attention then runs only on those candidates. The real trade is between a '
            + 'cheaper candidate search and the risk of never proposing a useful key.',
        }),
        el('p', {
          class: 'note warn',
          html: '<b>Architecture note.</b> LightningLM V4 used sparse-attention G-layers alongside '
            + 'DeltaNet layers, and cut its maximum budget from 1024 to 256 because of '
            + 'backward-kernel contention in that particular hardware and software stack. That '
            + 'makes 256 an implementation constraint from one run, not a universal law of sparse '
            + 'attention. Re-measuring it on current kernels is an open V5 question (§16).',
        }),
      );
    }

    /** Two bars: the value work that falls with k, and the scoring cost that does not. */
    function costBars() {
      const W = 660, H = 132, L = 132, R = 56;
      const iw = W - L - R;
      const g = svg('svg', {
        viewBox: `0 0 ${W} ${H}`, width: '100%',
        style: 'display:block;max-width:100%;height:auto', role: 'img',
        'aria-label': `With k=${k}, value work is ${k} of ${T} while naive selection still scores all ${T}`,
      });
      const bar = (y, frac, colour, label, note) => {
        g.append(
          svg('text', {
            x: L - 12, y: y + 17, 'text-anchor': 'end', 'font-size': 11.5,
            fill: '#b8b7b0', text: label,
          }),
          svg('rect', {
            x: L, y, width: iw, height: 24, rx: 5, fill: 'rgba(255,255,255,.055)',
          }),
          svg('rect', {
            x: L, y, width: Math.max(2, iw * frac), height: 24, rx: 5, fill: colour,
          }),
          svg('text', {
            x: L + iw + 10, y: y + 17, 'font-size': 11.5, fill: '#89877f', text: note,
          }),
        );
      };
      bar(14, k / T, '#3987e5', 'value work', `${k} / ${T}`);
      bar(56, 1, '#eb6834', 'naive selection', `${T} / ${T}`);
      g.append(svg('text', {
        x: L, y: 112, 'font-size': 11, fill: '#89877f',
        text: 'the blue bar is what top-k buys. the orange bar is what it still pays.',
      }));
      return g;
    }

    render();
  },
);
