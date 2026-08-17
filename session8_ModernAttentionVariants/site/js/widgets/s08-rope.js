/* §8 — Position, and the part that is usually skipped.
   The lesson's own example, computed: the same word ("bank") at two positions.
   Because the content is identical, q and k are identical, so without RoPE the
   score is the same no matter where the two tokens sit. Rotating each by its
   position makes the score depend on i − j and nothing else.

   Head width 8 → four 2D pairs. With base 10000 the frequencies come out at
   1, 0.1, 0.01, 0.001 exactly, which keeps every printed number legible. */

import {
  register, el, matrixTable as mt, stepper, segmented, toggle, slider, readout, svg,
  dot, fx, clear,
} from '../lib.js';

const D = 8;                       // head width
const PAIRS = D / 2;               // four 2D pairs
const BASE = 10000;
/* θ_p = 1 / base^(2p/D) → exactly 1, 0.1, 0.01, 0.001 for D = 8. */
const THETA = Array.from({ length: PAIRS }, (_, p) => 1 / BASE ** ((2 * p) / D));

/* One word's content, as four 2D pairs. Query and key are the SAME vector,
   because it is the same word appearing twice. */
const CONTENT = [[0.9, 0.3], [0.6, -0.5], [0.8, 0.2], [-0.4, 0.7]];

/* The value vector, shown only to make the point that RoPE never touches it.
   Position belongs in the SCORE, and V does not appear in the score. */
const VALUE = [[0.4, -0.7], [-0.2, 0.9], [0.5, 0.5], [0.3, -0.1]];

const rot = ([x0, x1], a) => [
  x0 * Math.cos(a) - x1 * Math.sin(a),
  x0 * Math.sin(a) + x1 * Math.cos(a),
];

/** Score for one pair, with or without RoPE applied. */
const pairScore = (p, i, j, on) => (on
  ? dot(rot(CONTENT[p], i * THETA[p]), rot(CONTENT[p], j * THETA[p]))
  : dot(CONTENT[p], CONTENT[p]));

const totalScore = (i, j, on) =>
  THETA.reduce((s, _, p) => s + pairScore(p, i, j, on), 0);

/* Two ways a sentence can change. Inserting words BETWEEN cat and sat changes
   their gap, so RoPE represents a different relationship. Adding words BEFORE
   both moves their absolute positions but leaves the gap alone — and the score
   is then bit-for-bit identical. */
const INSERT = [
  { text: 'The cat sat', cat: 1, sat: 2 },
  { text: 'The cat yesterday sat', cat: 1, sat: 3 },
  { text: 'The cat yesterday suddenly sat', cat: 1, sat: 4 },
  { text: 'The cat … 20 more words … sat', cat: 1, sat: 23 },
];
const SHIFT = [
  { text: 'The cat sat', cat: 1, sat: 2 },
  { text: 'Yesterday, the cat sat', cat: 3, sat: 4 },
  { text: 'Yesterday in London, the cat sat', cat: 5, sat: 6 },
  { text: '… 100 words … the cat sat', cat: 101, sat: 102 },
];

const STEPS = [
  { key: 'step 1', title: 'Content cannot see distance' },
  { key: 'step 2', title: 'Rotate by position' },
  { key: 'step 3', title: 'Only the gap matters' },
  { key: 'step 4', title: 'Four pairs, four rates' },
  { key: 'step 5', title: 'Insert vs shift' },
];

register(
  'w-s08-rope',
  'Distance as relative rotation',
  '§8 · RoPE · d = 8, base 10000',
  'The same word at two positions, so the content vectors are identical. Switch RoPE off and '
  + 'the score never changes, however far apart the tokens are. Switch it on and the score '
  + 'tracks the distance — then move both tokens together and watch it hold still.',
  (root) => {
    let step = 0;
    let i = 8;          // query position
    let j = 2;          // key position
    let pair = 0;
    let on = true;

    const stage = el('div', { class: 'stage' });
    const ros = el('div', { class: 'readouts' });

    const seq = stepper({ steps: STEPS, value: 0, onChange: (s) => { step = s; render(); } });

    const iS = slider({
      label: 'Query position i', min: 0, max: 40, step: 1, value: i,
      fmt: (v) => String(v), onInput: (v) => { i = v; render(); },
    });
    const jS = slider({
      label: 'Key position j', min: 0, max: 40, step: 1, value: j,
      fmt: (v) => String(v), onInput: (v) => { j = v; render(); },
    });
    const ropeSw = toggle({
      label: 'RoPE on', value: true, onChange: (v) => { on = v; render(); },
    });
    const pairSel = segmented({
      label: 'Pair',
      options: THETA.map((t, p) => ({ value: p, label: `p${p} · θ=${t}` })),
      value: pair,
      onChange: (v) => { pair = v; render(); },
    });
    const shift = el('button', {
      class: 'btn',
      onclick: () => {
        const room = Math.min(40 - i, 40 - j);
        const by = Math.min(10, room);
        i += by; j += by;
        iS.set(i); jS.set(j); render();
      },
    }, 'Shift both +10 →');

    root.append(
      seq.node,
      el('div', { class: 'controls' },
        iS.node, jS.node, pairSel.node,
        el('div', { class: 'ctl' }, ropeSw.node),
        el('div', { class: 'ctl' }, shift)),
      stage,
      ros,
    );

    function render() {
      clear(stage); clear(ros);
      shift.disabled = i >= 40 || j >= 40;
      const gap = i - j;

      if (step === 0) stepContent();
      else if (step === 1) stepRotate();
      else if (step === 2) stepGap();
      else if (step === 3) stepPairs();
      else stepSentences();

      addReadouts([
        { label: 'Positions', value: `${j} → ${i}`, sub: 'key → query' },
        { label: 'Distance i − j', value: String(gap), tone: 'violet', sub: 'what RoPE encodes' },
        {
          label: `Pair ${pair} angle`, value: `${fx(gap * THETA[pair], 3)} rad`,
          tone: 'blue', sub: `(i−j) × ${THETA[pair]}`,
        },
        {
          label: `Pair ${pair} score`, value: fx(pairScore(pair, i, j, on), 4),
          tone: on ? 'blue' : 'orange',
          sub: on ? 'depends on distance' : 'RoPE off — fixed',
        },
        {
          label: 'Score, all 4 pairs', value: fx(totalScore(i, j, on), 4),
          tone: on ? 'aqua' : 'red',
          sub: on ? 'position is in the score' : 'position is invisible',
        },
      ]);
    }

    function addReadouts(list) { list.forEach((r) => ros.append(readout(r).node)); }

    /* ---- step 1 ---- */
    function stepContent() {
      const raw = CONTENT.map((c) => dot(c, c));
      stage.append(
        el('p', { class: 'stage-title', text: 'The same word twice — and the score cannot tell them apart' }),
        el('p', {
          class: 'stage-sub',
          text: `Imagine "bank" at position ${j} and "bank" at position ${i}. Same word, so the same `
            + 'embedding, so the same query and key. The dot product compares content only, and '
            + 'these contents are identical — so the score is exactly the same whether the tokens '
            + 'are adjacent or hundreds of positions apart. Move both sliders and watch nothing '
            + 'happen to the bottom row.',
        }),
        el('div', { class: 'gridrow' },
          mt({
            caption: `q · "bank" at ${i}`, data: CONTENT,
            rowLabels: THETA.map((_, p) => `pair ${p}`), colLabels: ['x0', 'x1'],
            cell: () => ({ bg: 'rgba(57,135,229,.14)' }),
          }),
          mt({
            caption: `k · "bank" at ${j}`, data: CONTENT,
            rowLabels: THETA.map((_, p) => `pair ${p}`), colLabels: ['x0', 'x1'],
            cell: () => ({ bg: 'rgba(27,175,122,.14)' }),
          }),
          el('div', { class: 'op', text: '→' }),
          mt({
            caption: 'q·k per pair, no RoPE', data: raw.map((v) => [v]),
            rowLabels: THETA.map((_, p) => `pair ${p}`), colLabels: ['s'], digits: 4,
            cell: () => ({ bg: 'rgba(224,82,82,.16)' }),
          }),
        ),
        el('pre', {
          class: 'ascii',
          text: `  pair 0:  ${fx(CONTENT[0][0])}×${fx(CONTENT[0][0])} + ${fx(CONTENT[0][1])}×${fx(CONTENT[0][1])}`
            + `  =  ${fx(raw[0], 4)}\n\n`
            + `  total over all ${PAIRS} pairs  =  ${fx(raw.reduce((a, b) => a + b, 0), 4)}\n\n`
            + `  positions ${j} and ${i}   →  score ${fx(raw.reduce((a, b) => a + b, 0), 4)}\n`
            + `  positions 0 and 1     →  score ${fx(raw.reduce((a, b) => a + b, 0), 4)}\n`
            + `  positions 2 and 900   →  score ${fx(raw.reduce((a, b) => a + b, 0), 4)}\n\n`
            + 'This is Fact 2 from §2: attention itself does not know token order. The causal\n'
            + 'mask stops a token reading the future, but among what it MAY read, the dot\n'
            + 'product still has no idea what is near and what is far.',
        }),
      );
    }

    /* ---- step 2 ---- */
    function stepRotate() {
      const th = THETA[pair];
      const aQ = i * th;
      const aK = j * th;
      const rq = rot(CONTENT[pair], aQ);
      const rk = rot(CONTENT[pair], aK);
      stage.append(
        el('p', { class: 'stage-title', text: 'Treat two dimensions as an arrow, and turn it' }),
        el('p', {
          class: 'stage-sub',
          text: `Pair ${pair} of the head is the 2D arrow (x0, x1). Choose a small angle θ = ${th} `
            + 'per step of the sequence, then rotate each token\'s arrow by its position × θ. The '
            + 'query at position i turns by i·θ; the key at position j turns by j·θ.',
        }),
        // What goes in: q and k are the same vector (same word), v is separate.
        el('p', {
          class: 'stage-sub',
          style: { margin: '4px 0 10px' },
          text: 'Before any rotation, this is what the token has. RoPE rotates the query and the '
            + 'key. It never touches the value — position belongs in the score, and V does not '
            + 'appear in the score.',
        }),
        el('div', { class: 'gridrow' },
          mt({
            caption: `q BEFORE  ·  pair ${pair}`, data: CONTENT[pair].map((v) => [v]),
            rowLabels: ['x0', 'x1'], colLabels: ['q'], digits: 2,
            cell: () => ({ bg: 'rgba(57,135,229,.16)' }),
          }),
          mt({
            caption: `k BEFORE  ·  identical`, data: CONTENT[pair].map((v) => [v]),
            rowLabels: ['x0', 'x1'], colLabels: ['k'], digits: 2,
            cell: () => ({ bg: 'rgba(27,175,122,.16)' }),
          }),
          mt({
            caption: 'v  ·  NEVER rotated', data: VALUE[pair].map((v) => [v]),
            rowLabels: ['x0', 'x1'], colLabels: ['v'], digits: 2,
            cell: () => ({ bg: 'rgba(255,255,255,.05)', color: '#89877f' }),
          }),
        ),

        // The query's rotation, worked.
        el('div', { class: 'gridrow', style: { marginTop: '18px' } },
          mt({
            caption: `R(i·θ) = R(${fx(aQ, 3)})`,
            data: [[Math.cos(aQ), -Math.sin(aQ)], [Math.sin(aQ), Math.cos(aQ)]],
            rowLabels: ['row 0', 'row 1'], colLabels: ['c0', 'c1'], digits: 4,
            cell: () => ({ bg: 'rgba(144,133,233,.14)' }),
          }),
          el('div', { class: 'op', text: '×' }),
          mt({
            caption: 'q before', data: CONTENT[pair].map((v) => [v]),
            rowLabels: ['x0', 'x1'], colLabels: ['q'], digits: 2,
            cell: () => ({ bg: 'rgba(57,135,229,.16)' }),
          }),
          el('div', { class: 'op', text: '=' }),
          mt({
            caption: `q AFTER  ·  position ${i}`, data: rq.map((v) => [v]),
            rowLabels: ['x0', 'x1'], colLabels: ["q'"], digits: 4,
            cell: () => ({ bg: 'rgba(57,135,229,.28)' }),
          }),
        ),

        // The key's rotation, same matrix form but its own angle.
        el('div', { class: 'gridrow', style: { marginTop: '14px' } },
          mt({
            caption: `R(j·θ) = R(${fx(aK, 3)})`,
            data: [[Math.cos(aK), -Math.sin(aK)], [Math.sin(aK), Math.cos(aK)]],
            rowLabels: ['row 0', 'row 1'], colLabels: ['c0', 'c1'], digits: 4,
            cell: () => ({ bg: 'rgba(144,133,233,.14)' }),
          }),
          el('div', { class: 'op', text: '×' }),
          mt({
            caption: 'k before', data: CONTENT[pair].map((v) => [v]),
            rowLabels: ['x0', 'x1'], colLabels: ['k'], digits: 2,
            cell: () => ({ bg: 'rgba(27,175,122,.16)' }),
          }),
          el('div', { class: 'op', text: '=' }),
          mt({
            caption: `k AFTER  ·  position ${j}`, data: rk.map((v) => [v]),
            rowLabels: ['x0', 'x1'], colLabels: ["k'"], digits: 4,
            cell: () => ({ bg: 'rgba(27,175,122,.28)' }),
          }),
        ),

        // V, untouched, alongside for contrast.
        el('div', { class: 'gridrow', style: { marginTop: '14px' } },
          mt({
            caption: 'no rotation applied to v', data: VALUE[pair].map((v) => [v]),
            rowLabels: ['x0', 'x1'], colLabels: ['v'], digits: 2,
            cell: () => ({ bg: 'rgba(255,255,255,.05)', color: '#89877f' }),
          }),
          el('div', { class: 'op', text: '=' }),
          mt({
            caption: 'v, unchanged', data: VALUE[pair].map((v) => [v]),
            rowLabels: ['x0', 'x1'], colLabels: ['v'], digits: 2,
            cell: () => ({ bg: 'rgba(255,255,255,.05)', color: '#89877f' }),
          }),
        ),
        el('pre', {
          class: 'ascii',
          text: `  θ = ${th}      i·θ = ${i}×${th} = ${fx(aQ, 3)} rad`
            + `      j·θ = ${j}×${th} = ${fx(aK, 3)} rad\n\n`
            + `  q'[0] = x0·cos(i·θ) − x1·sin(i·θ)\n`
            + `        = ${fx(CONTENT[pair][0])}×${fx(Math.cos(aQ), 4)} − ${fx(CONTENT[pair][1])}×${fx(Math.sin(aQ), 4)}\n`
            + `        = ${fx(rq[0], 4)}\n\n`
            + `  q'[1] = x0·sin(i·θ) + x1·cos(i·θ)\n`
            + `        = ${fx(CONTENT[pair][0])}×${fx(Math.sin(aQ), 4)} + ${fx(CONTENT[pair][1])}×${fx(Math.cos(aQ), 4)}\n`
            + `        = ${fx(rq[1], 4)}\n\n`
            + `  k'[0] = ${fx(CONTENT[pair][0])}×${fx(Math.cos(aK), 4)} − ${fx(CONTENT[pair][1])}×${fx(Math.sin(aK), 4)}`
            + ` = ${fx(rk[0], 4)}\n`
            + `  k'[1] = ${fx(CONTENT[pair][0])}×${fx(Math.sin(aK), 4)} + ${fx(CONTENT[pair][1])}×${fx(Math.cos(aK), 4)}`
            + ` = ${fx(rk[1], 4)}\n\n`
            + `  rotation changes the DIRECTION, never the length:\n`
            + `    |q| before = ${fx(Math.hypot(...CONTENT[pair]), 4)}`
            + `    |q'| after = ${fx(Math.hypot(...rq), 4)}\n`
            + `    |k| before = ${fx(Math.hypot(...CONTENT[pair]), 4)}`
            + `    |k'| after = ${fx(Math.hypot(...rk), 4)}\n\n`
            + `  q and k started IDENTICAL — same word, same content. They are different now\n`
            + `  only because they sit at different positions (${i} and ${j}).\n\n`
            + arrowStory(aQ, rq, 'q', i)
            + arrowStory(aK, rk, 'k', j)
            + `  v is not in this list. RoPE is applied to Q and K only:\n`
            + `    score = q'·k'   ← position enters here\n`
            + `    output = Σ weight × v   ← v carries content, never position`,
        }),
        dial(aQ, aK, rq, rk),
      );
    }

    /* ---- step 3 ---- */
    function stepGap() {
      const th = THETA[pair];
      const rq = rot(CONTENT[pair], i * th);
      const rk = rot(CONTENT[pair], j * th);
      const s = dot(rq, rk);
      const len2 = dot(CONTENT[pair], CONTENT[pair]);
      stage.append(
        el('p', { class: 'stage-title', text: 'Both absolute rotations cancel — only i − j survives' }),
        el('p', {
          class: 'stage-sub',
          text: 'A dot product depends on the angle between two arrows. Each was turned by its own '
            + 'absolute position, but the angle between them is the difference of those turns. Press '
            + '"Shift both +10" and watch the positions change while the score does not.',
        }),
        el('pre', {
          class: 'ascii',
          text: `  R(i·θ)q · R(j·θ)k   =   q · R((j − i)·θ)k\n\n`
            + `  the positional part of the score depends on i − j, the distance\n\n`
            + `  ${'─'.repeat(56)}\n`
            + `  i            = ${String(i).padStart(4)}\n`
            + `  j            = ${String(j).padStart(4)}\n`
            + `  i − j        = ${String(i - j).padStart(4)}\n`
            + `  (i−j)·θ      = ${fx((i - j) * th, 4).padStart(9)} rad\n`
            + `  cos((i−j)·θ) = ${fx(Math.cos((i - j) * th), 4).padStart(9)}\n`
            + `  |content|²   = ${fx(len2, 4).padStart(9)}\n`
            + `  ${'─'.repeat(56)}\n`
            + `  score        = |content|² × cos((i−j)·θ)\n`
            + `               = ${fx(len2, 4)} × ${fx(Math.cos((i - j) * th), 4)}\n`
            + `               = ${fx(len2 * Math.cos((i - j) * th), 4)}\n\n`
            + `  computed directly from the rotated vectors: ${fx(s, 4)}\n`
            + `  difference: ${fx(Math.abs(s - len2 * Math.cos((i - j) * th)), 8)}`,
        }),
        el('pre', {
          class: 'ascii',
          text: '  same distance, different absolute positions:\n\n'
            + [[2, 8], [12, 18], [22, 28], [32, 38]].map(([jj, ii]) =>
              `    j=${String(jj).padStart(2)}  i=${String(ii).padStart(2)}   gap ${ii - jj}`
              + `   score ${fx(pairScore(pair, ii, jj, true), 6)}`).join('\n')
            + '\n\n  Both arrows turn further as the sequence moves, but they turn TOGETHER,\n'
            + '  so the angle between them — and therefore the score — is unchanged.',
        }),
        dial(i * th, j * th, rq, rk),
      );
    }

    /* ---- step 4 ---- */
    function stepPairs() {
      const gap = i - j;
      stage.append(
        el('p', { class: 'stage-title', text: 'Every pair rotates at its own rate' }),
        el('p', {
          class: 'stage-sub',
          text: 'A real head has more than two dimensions, so RoPE groups them into 2D pairs and '
            + 'rotates each pair at a different speed. Fast pairs resolve nearby distances; slow '
            + 'pairs still change measurably over hundreds of positions. That is how one head '
            + 'represents distance at several scales at once.',
        }),
        mt({
          caption: `all ${PAIRS} pairs at distance ${gap}`,
          data: THETA.map((t, p) => [
            t, gap * t, Math.cos(gap * t), pairScore(p, i, j, true), (2 * Math.PI) / t,
          ]),
          rowLabels: THETA.map((_, p) => `pair ${p}`),
          colLabels: ['θ', '(i−j)·θ', 'cos', 'score', 'wavelength'],
          digits: 4,
          cell: (v, r, c) => ({
            text: c === 4 ? fx(v, 1) : fx(v, 4),
            bg: r === pair ? 'rgba(57,135,229,.20)' : null,
          }),
        }),
        el('pre', {
          class: 'ascii',
          text: `  wavelength = 2π / θ — how many positions before the pair comes full circle\n\n`
            + THETA.map((t, p) => `    pair ${p}   θ = ${String(t).padEnd(7)}`
              + `wavelength ${fx((2 * Math.PI) / t, 1).padStart(9)} positions`).join('\n')
            + '\n\n  pair 0 turns a full circle every ~6 tokens — it can only distinguish very\n'
            + '  local distances. pair 3 takes ~6,283 tokens for one turn — it is what carries\n'
            + '  long-range position. §9 is about what happens to that slow pair when the\n'
            + '  context is stretched far beyond the training length.',
        }),
        el('p', {
          class: 'note',
          html: '<b>Not every dimension has to be rotated.</b> Some implementations rotate only part '
            + 'of the head and leave the rest untouched — DeepSeek-V4 applies RoPE to the last 64 '
            + 'dimensions. That is an implementation choice and does not change the mechanism above.',
        }),
      );
    }

    /* ---- step 5: two ways to change a sentence ---- */
    function stepSentences() {
      const table = (rows, label) => {
        const ref = totalScore(rows[0].sat, rows[0].cat, true);
        return el('pre', {
          class: 'ascii',
          text: `${label}\n\n`
            + '  sentence                              cat  sat   gap   score      vs first\n'
            + `  ${'─'.repeat(76)}\n`
            + rows.map((r) => {
              const s = totalScore(r.sat, r.cat, true);
              const d = Math.abs(s - ref);
              return `  ${r.text.padEnd(36)}${String(r.cat).padStart(4)}`
                + `${String(r.sat).padStart(5)}${String(r.sat - r.cat).padStart(6)}`
                + `${fx(s, 4).padStart(10)}   `
                + (d < 1e-12 ? 'IDENTICAL' : `differs by ${fx(d, 4)}`);
            }).join('\n'),
        });
      };

      stage.append(
        el('p', { class: 'stage-title', text: 'Inserting words changes the relationship. Moving the sentence does not.' }),
        el('p', {
          class: 'stage-sub',
          text: 'Two different things can happen to a sentence, and RoPE treats them very '
            + 'differently. Put words BETWEEN cat and sat and their gap grows, so the score '
            + 'changes — that is RoPE doing its job. Put words BEFORE both and their absolute '
            + 'positions grow while the gap holds, and the score does not move at all.',
        }),
        table(INSERT, 'CASE A — insert words BETWEEN cat and sat  ·  the gap changes'),
        table(SHIFT, 'CASE B — add words BEFORE both  ·  the gap is unchanged'),
        el('pre', {
          class: 'ascii',
          text: '  In case B the absolute positions went from 1,2 up to 101,102 — a hundred\n'
            + '  places along — and the score did not change in the last decimal place.\n'
            + '  Both arrows rotated much further, but they rotated TOGETHER.\n\n'
            + '  So this is the honest statement about long context:\n\n'
            + '    absolute position    →  cancels exactly. never a problem for the score.\n'
            + '    relative distance    →  the only thing the score sees.\n\n'
            + '  Which means the difficulty at 256K is NOT "angles the model never saw at\n'
            + '  position 200,000". It is case A taken to an extreme: a model trained inside\n'
            + '  8K has never seen a GAP of 200,000. §9 measures exactly that.',
        }),
        el('p', {
          class: 'note',
          html: '<b>Worth checking yourself.</b> Set the sliders to i=8, j=2 and note the score. '
            + 'Then press <b>Shift both +10</b> twice — i=28, j=22. The positions have moved twenty '
            + 'places and the score is unchanged. Now move only <em>i</em> and it changes '
            + 'immediately.',
        }),
      );
    }

    /**
     * The same rotation told as an angle rather than as components, because the
     * components hide it: a sign flip in x0 looks like destruction until you see
     * that the arrow simply swung past vertical. Length is invariant throughout.
     */
    function arrowStory(angle, after, name, pos) {
      const deg = (r) => (r * 180) / Math.PI;
      const wrap = (d) => ((d % 360) + 360) % 360;
      const base = deg(Math.atan2(CONTENT[pair][1], CONTENT[pair][0]));
      const turned = deg(angle);
      const end = wrap(base + turned);
      const len = Math.hypot(...CONTENT[pair]);
      const quad = end < 90 ? 'upper-right — both components positive'
        : end < 180 ? 'upper-LEFT — x0 goes NEGATIVE, x1 positive'
          : end < 270 ? 'lower-left — both components negative'
            : 'lower-right — x0 positive, x1 negative';
      return `  ${name} as an arrow, at position ${pos}:\n`
        + `    length ${fx(len, 4)}, starting angle ${fx(base, 1)}°\n`
        + `    turned by ${fx(angle, 3)} rad = ${fx(turned, 1)}°`
        + `  (${fx(angle / (2 * Math.PI), 2)} full turns)\n`
        + `    ends at ${fx(end, 1)}°  →  ${quad}\n`
        + `    cos(${fx(end, 1)}°) × ${fx(len, 4)} = ${fx(after[0], 4)}`
        + `    sin(${fx(end, 1)}°) × ${fx(len, 4)} = ${fx(after[1], 4)}\n\n`;
    }

    /** Two arrows on a dial, with the angle between them marked. */
    function dial(aQ, aK, rq, rk) {
      const S = 190, C = S / 2, R = 68;
      const g = svg('svg', {
        viewBox: `0 0 ${S} ${S}`, width: S, height: S,
        style: 'display:block;margin-top:18px', role: 'img',
        'aria-label': `query arrow at ${fx(aQ, 2)} radians, key arrow at ${fx(aK, 2)} radians`,
      });
      const arrow = (v, colour, label) => {
        const L = Math.hypot(...v) || 1;
        const x = C + (v[0] / L) * R;
        const y = C - (v[1] / L) * R;
        g.append(
          svg('line', {
            x1: C, y1: C, x2: x, y2: y, stroke: colour, 'stroke-width': 2.4,
            'stroke-linecap': 'round',
          }),
          svg('circle', { cx: x, cy: y, r: 4, fill: colour }),
          svg('text', {
            x: x + (x > C ? 7 : -7), y: y + (y > C ? 13 : -6),
            'text-anchor': x > C ? 'start' : 'end',
            'font-size': 11, fill: colour, 'font-weight': 600, text: label,
          }),
        );
      };
      g.append(
        svg('circle', {
          cx: C, cy: C, r: R, fill: 'none', stroke: 'rgba(255,255,255,.10)', 'stroke-width': 1,
        }),
        svg('line', {
          x1: C - R - 8, y1: C, x2: C + R + 8, y2: C,
          stroke: 'rgba(255,255,255,.07)', 'stroke-width': 1,
        }),
        svg('line', {
          x1: C, y1: C - R - 8, x2: C, y2: C + R + 8,
          stroke: 'rgba(255,255,255,.07)', 'stroke-width': 1,
        }),
      );
      arrow(rk, '#5fd2a8', `k @ ${j}`);
      arrow(rq, '#8fbcf5', `q @ ${i}`);
      g.append(svg('text', {
        x: C, y: S - 4, 'text-anchor': 'middle', 'font-size': 10.5, fill: '#89877f',
        text: `angle between = ${fx(Math.abs(aQ - aK), 3)} rad`,
      }));
      return g;
    }

    render();
  },
);
