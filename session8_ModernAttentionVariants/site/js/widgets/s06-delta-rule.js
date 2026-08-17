/* §6 — The delta rule: write only what needs to change.
   Continues §5's state, but with a full 4-component answer rather than a single
   number parked in slot 0, so every slot of the state does real work.

       add-only:   S = S + (wanted        × kᵀ)   →  every slot overshoots
       delta:      S = S + ((wanted − got) × kᵀ)  →  every slot lands exactly

   kA is unit-norm, so a read returns the stored answer vector untouched:
       S kA = (a kAᵀ) kA = a (kA·kA) = a
   Slot 0 carries the lesson's 40 → 55 so the prose numbers still appear, but the
   other three slots show the rule correcting up, down and through zero at once. */

import {
  register, el, matrixTable as mt, stepper, segmented, readout,
  outer, addM, matvec, zeros, fx, clear,
} from '../lib.js';

const D = 4;
const AXES = ['0', '1', '2', '3'];

const OLD = [40, -12, 28, 5];               // what memory holds, from §5
const WANTED = [55, 30, -10, 18];           // what it should hold now
const DELTA = WANTED.map((w, i) => w - OLD[i]);   // [15, 42, -38, 13]

/* The arriving token's key and value. Both come from its embedding via Wk and Wv,
   exactly as in §2 — that projection is not re-derived here. kA is unit-norm so a
   read returns the stored answer untouched. */
const kA = [0.5, 0.5, 0.5, 0.5];
const K_NEW = kA;
const V_NEW = WANTED;

/* A sequence of revisions, for the last step. */
const TARGETS = [
  [40, -12, 28, 5],
  [55, 30, -10, 18],
  [30, 5, 12, -8],
  [70, -20, 40, 22],
];

const read = (S) => matvec(S, kA);
const norm = (v) => Math.hypot(...v);
const gap = (a, b) => norm(a.map((x, i) => x - b[i]));
const vec = (v, d = 1) => `[ ${v.map((x) => fx(x, d).padStart(6)).join(' ')} ]`;

const S1 = outer(OLD, kA);                                  // §5's state
const S_ADD = addM(S1, outer(WANTED, kA));                  // add-only
const S_DELTA = addM(S1, outer(DELTA, kA));                 // delta

/** Replays TARGETS under one write rule. */
function replay(rule) {
  let S = zeros(D, D);
  return TARGETS.map((t) => {
    const before = read(S);
    const write = rule === 'delta' ? t.map((x, i) => x - before[i]) : t;
    S = addM(S, outer(write, kA));
    const after = read(S);
    return { target: t, before, write, after, err: gap(after, t) };
  });
}

const STEPS = [
  { key: 'step 1', title: 'A new token arrives' },
  { key: 'step 2', title: 'Read what is stored' },
  { key: 'step 3', title: 'Measure the gap' },
  { key: 'step 4', title: 'Write only the gap' },
  { key: 'step 5', title: 'Many corrections' },
];

register(
  'w-s06-delta-rule',
  'Write only what needs to change',
  '§6 · the delta rule',
  'The state from §5 holds a four-component answer and every component is now out of date. '
  + 'Adding the whole new answer overshoots in all four. The delta rule reads memory first, '
  + 'works out what is missing in each slot, and writes only that. Same state, same shape, '
  + 'one different write rule.',
  (root) => {
    let step = 0;
    let rule = 'delta';

    const stage = el('div', { class: 'stage' });
    const ros = el('div', { class: 'readouts' });

    const seq = stepper({ steps: STEPS, value: 0, onChange: (i) => { step = i; render(); } });

    const ruleSel = segmented({
      label: 'Write rule',
      options: [
        { value: 'delta', label: 'delta — write the gap', title: 'S += (wanted − got) kᵀ' },
        { value: 'add', label: 'add-only — write it all', title: 'S += wanted kᵀ  (§5)' },
      ],
      value: rule,
      onChange: (v) => { rule = v; render(); },
    });

    root.append(
      seq.node,
      el('div', { class: 'controls' }, ruleSel.node),
      stage,
      ros,
    );

    function render() {
      clear(stage); clear(ros);
      const isDelta = rule === 'delta';
      const after = isDelta ? S_DELTA : S_ADD;
      const got = read(after);
      const err = gap(got, WANTED);

      if (step === 0) stepOrigin();
      else if (step === 1) stepRead();
      else if (step === 2) stepGap();
      else if (step === 3) stepWrite(isDelta, after, got, err);
      else stepMany();

      const rows = replay(rule);
      const last = rows[rows.length - 1];
      addReadouts([
        {
          label: 'Write rule', value: isDelta ? 'delta' : 'add-only',
          tone: isDelta ? 'aqua' : 'red',
          sub: isDelta ? 'S += (wanted − got) kᵀ' : 'S += wanted kᵀ',
        },
        {
          label: 'Slots correct', value: `${got.filter((v, i) => Math.abs(v - WANTED[i]) < 1e-9).length} of ${D}`,
          tone: isDelta ? 'aqua' : 'red',
          sub: isDelta ? 'all four land' : 'all four overshoot',
        },
        {
          label: 'Error ‖got − wanted‖', value: fx(err, 2),
          tone: isDelta ? 'aqua' : 'red',
          sub: isDelta ? 'exact' : 'the stale answer, still there',
        },
        {
          label: 'State shape', value: `${D} × ${D}`,
          sub: `${D * D} numbers — unchanged by either rule`,
        },
        {
          label: `Error after ${TARGETS.length} revisions`, value: fx(last.err, 1),
          tone: isDelta ? 'aqua' : 'red',
          sub: isDelta ? 'still exact' : 'compounding',
        },
      ]);
    }

    function addReadouts(list) { list.forEach((r) => ros.append(readout(r).node)); }

    /* ---- step 1: a new token arrives and its grid is added ---- */
    function stepOrigin() {
      const grid = outer(V_NEW, K_NEW);
      stage.append(
        el('p', { class: 'stage-title', text: 'A new token arrives with its own key and value' }),
        el('p', {
          class: 'stage-sub',
          text: 'The state is a 4×4 grid holding everything so far. The new token brings a key and '
            + 'a value — k = x·Wk and v = x·Wv, the same projections as §2. Turn those two vectors '
            + 'into a grid with v kᵀ, and add it to the old state. That is the whole write.',
        }),
        el('div', { class: 'gridrow' },
          mt({
            caption: 'v  ·  new token (4×1)', data: V_NEW.map((v) => [v]),
            rowLabels: AXES, colLabels: ['v'], digits: 0,
            cell: () => ({ bg: 'rgba(235,104,52,.20)' }),
          }),
          el('div', { class: 'op', text: '×' }),
          mt({
            caption: 'kᵀ  ·  new token (1×4)', data: [K_NEW],
            rowLabels: ['kᵀ'], colLabels: AXES, digits: 2,
            cell: () => ({ bg: 'rgba(27,175,122,.20)' }),
          }),
          el('div', { class: 'op', text: '=' }),
          mt({
            caption: 'v kᵀ  (4×4)', data: grid,
            rowLabels: AXES, colLabels: AXES, digits: 1,
            cell: () => ({ bg: 'rgba(144,133,233,.16)' }),
          }),
        ),
        el('div', { class: 'gridrow', style: { marginTop: '18px' } },
          mt({
            caption: 'S old  (4×4)', data: S1, rowLabels: AXES, colLabels: AXES, digits: 1,
            cell: (v) => ({
              bg: v > 1e-9 ? 'rgba(57,135,229,.16)' : v < -1e-9 ? 'rgba(235,104,52,.14)' : null,
            }),
          }),
          el('div', { class: 'op', text: '+' }),
          mt({
            caption: 'v kᵀ  (4×4)', data: grid, rowLabels: AXES, colLabels: AXES, digits: 1,
            cell: () => ({ bg: 'rgba(144,133,233,.16)' }),
          }),
          el('div', { class: 'op', text: '=' }),
          mt({
            caption: 'S new  (4×4)', data: S_ADD, rowLabels: AXES, colLabels: AXES, digits: 1,
            cell: () => ({ bg: 'rgba(57,135,229,.24)' }),
          }),
        ),
        el('pre', {
          class: 'ascii',
          text: '  S_new  =  S_old  +  v kᵀ\n\n'
            + '  shapes:  (4×4)  +  (4×1)(1×4)  =  (4×4) + (4×4)  =  (4×4)\n\n'
            + `  row 0:  ${S1[0].map((v) => fx(v, 1)).join(', ')}`
            + `  +  ${grid[0].map((v) => fx(v, 1)).join(', ')}`
            + `  =  ${S_ADD[0].map((v) => fx(v, 1)).join(', ')}\n\n`
            + 'That is the add-only write. It is the obvious thing to do, and it is what §5\n'
            + 'did. Whether it gives the right answer depends entirely on what the state\n'
            + 'already holds for this key — which nobody has checked yet. Step 2 checks.',
        }),
      );
    }

    /* ---- step 2: read ---- */
    function stepRead() {
      const got = read(S1);
      stage.append(
        el('p', { class: 'stage-title', text: 'First, ask memory what it currently says' }),
        el('p', {
          class: 'stage-sub',
          text: 'This is the step the add-only rule skips entirely. Probe the state with the same '
            + 'key and see what comes back — one matrix-vector product, the same read as §4. Every '
            + 'row of the state is populated now, so the answer is a genuine four-component vector.',
        }),
        el('div', { class: 'gridrow' },
          mt({
            caption: 'S  ·  from §5', data: S1, rowLabels: AXES, colLabels: AXES, digits: 1,
            cell: (v) => ({ bg: v > 1e-9 ? 'rgba(57,135,229,.16)' : v < -1e-9 ? 'rgba(235,104,52,.16)' : null }),
          }),
          el('div', { class: 'op', text: '×' }),
          mt({
            caption: 'key A', data: kA.map((v) => [v]),
            rowLabels: AXES, colLabels: ['k'], digits: 2,
            cell: () => ({ bg: 'rgba(27,175,122,.14)' }),
          }),
          el('div', { class: 'op', text: '=' }),
          mt({
            caption: 'read  ·  the stored answer', data: got.map((v) => [v]),
            rowLabels: AXES, colLabels: ['y'], digits: 1,
            cell: () => ({ bg: 'rgba(144,133,233,.22)' }),
          }),
        ),
        el('pre', {
          class: 'ascii',
          text: got.map((v, a) => `  y[${a}]  =  `
            + S1[a].map((s, b) => `${fx(s, 1)}×${fx(kA[b], 1)}`).join(' + ')
            + `  =  ${fx(v, 1).padStart(6)}`).join('\n')
            + `\n\n  read     = ${vec(got)}\n`
            + `  wanted   = ${vec(WANTED)}\n\n`
            + 'All four slots are out of date, not just the first one.',
        }),
      );
    }

    /* ---- step 2: the gap ---- */
    function stepGap() {
      stage.append(
        el('p', { class: 'stage-title', text: 'Subtract, component by component' }),
        el('p', {
          class: 'stage-sub',
          text: 'The delta is a vector, and each slot needs a different correction. Some go up, '
            + 'some go down, and one has to cross zero. A single scalar could never express this — '
            + 'which is why the correction has to be worked out per component.',
        }),
        el('div', { class: 'gridrow' },
          mt({
            caption: 'wanted', data: WANTED.map((v) => [v]),
            rowLabels: AXES, colLabels: ['w'], digits: 0,
            cell: () => ({ bg: 'rgba(27,175,122,.16)' }),
          }),
          el('div', { class: 'op', text: '−' }),
          mt({
            caption: 'memory says', data: OLD.map((v) => [v]),
            rowLabels: AXES, colLabels: ['g'], digits: 0,
            cell: () => ({ bg: 'rgba(57,135,229,.16)' }),
          }),
          el('div', { class: 'op', text: '=' }),
          mt({
            caption: 'delta  ·  what to write', data: DELTA.map((v) => [v]),
            rowLabels: AXES, colLabels: ['Δ'], digits: 0,
            cell: (v) => ({
              bg: v >= 0 ? 'rgba(144,133,233,.24)' : 'rgba(235,104,52,.24)',
            }),
          }),
        ),
        el('pre', {
          class: 'ascii',
          text: DELTA.map((d, a) => `  Δ[${a}]  =  ${fx(WANTED[a], 0).padStart(4)}`
            + ` − ${fx(OLD[a], 0).padStart(4)}  =  ${fx(d, 0).padStart(5)}`
            + (d < 0 ? '   ← negative: information must be REMOVED' : '')).join('\n')
            + `\n\n  delta = ${vec(DELTA, 0)}\n\n`
            + `The add-only rule would instead write the whole ${vec(WANTED, 0)},\n`
            + 'ignoring everything already in the state.',
        }),
      );
    }

    /* ---- step 3: the write ---- */
    function stepWrite(isDelta, after, got, err) {
      const written = isDelta ? DELTA : WANTED;
      const added = outer(written, kA);
      stage.append(
        el('p', {
          class: 'stage-title',
          text: isDelta
            ? 'Write the delta — every slot lands exactly'
            : 'Write the whole answer — every slot overshoots',
        }),
        el('p', {
          class: 'stage-sub',
          text: isDelta
            ? 'The grid added carries only the corrections, including negative ones, so the state '
              + 'ends up holding precisely the wanted answer. Nothing stale survives.'
            : 'The grid added carries the complete new answer, so whatever each slot already held '
              + 'is still underneath it. Switch the write rule above to compare.',
        }),
        el('div', { class: 'gridrow' },
          mt({
            caption: 'S before', data: S1, rowLabels: AXES, colLabels: AXES, digits: 1,
            cell: (v) => ({ bg: v > 1e-9 ? 'rgba(57,135,229,.14)' : v < -1e-9 ? 'rgba(235,104,52,.12)' : null }),
          }),
          el('div', { class: 'op', text: '+' }),
          mt({
            caption: `${isDelta ? 'delta' : 'wanted'} × kAᵀ`, data: added,
            rowLabels: AXES, colLabels: AXES, digits: 1,
            cell: () => ({ bg: isDelta ? 'rgba(27,175,122,.20)' : 'rgba(235,104,52,.20)' }),
          }),
          el('div', { class: 'op', text: '=' }),
          mt({
            caption: 'S after', data: after, rowLabels: AXES, colLabels: AXES, digits: 1,
            cell: () => ({ bg: isDelta ? 'rgba(27,175,122,.24)' : 'rgba(235,104,52,.24)' }),
          }),
        ),
        el('div', { class: 'gridrow', style: { marginTop: '18px' } },
          mt({
            caption: 'reads back as', data: got.map((v) => [v]),
            rowLabels: AXES, colLabels: ['y'], digits: 1,
            cell: () => ({ bg: 'rgba(144,133,233,.22)' }),
          }),
          mt({
            caption: 'wanted', data: WANTED.map((v) => [v]),
            rowLabels: AXES, colLabels: ['w'], digits: 1,
            cell: () => ({ bg: 'rgba(255,255,255,.05)' }),
          }),
          mt({
            caption: 'difference', data: got.map((v, i) => [v - WANTED[i]]),
            rowLabels: AXES, colLabels: ['Δ'], digits: 1,
            cell: (v) => ({
              bg: Math.abs(v) < 1e-9 ? 'rgba(27,175,122,.20)' : 'rgba(224,82,82,.24)',
            }),
          }),
        ),
        el('pre', {
          class: 'ascii',
          text: `  wrote    ${vec(written, 0)}\n`
            + `  reads    ${vec(got)}\n`
            + `  wanted   ${vec(WANTED)}\n`
            + `  error    ${fx(err, 2)}   ${isDelta ? '✓ every slot exact' : '✗ every slot wrong'}\n\n`
            + `The state is still ${D}×${D}. Only the grid added to it differs.`,
        }),
      );
    }

    /* ---- step 4: a sequence of revisions ---- */
    function stepMany() {
      const rows = replay(rule);
      stage.append(
        el('p', {
          class: 'stage-title',
          text: rule === 'delta'
            ? 'Four revisions — the state tracks every one'
            : 'Four revisions — the error compounds',
        }),
        el('p', {
          class: 'stage-sub',
          text: 'One wrong correction is recoverable. The real problem is a long sequence where the '
            + 'same association is revised again and again. Switch the write rule and compare the '
            + 'error column.',
        }),
        el('pre', {
          class: 'ascii',
          text: `  target                     memory now                  error\n`
            + `  ${'─'.repeat(62)}\n`
            + rows.map((r) => `  ${vec(r.target, 0)}   ${vec(r.after, 0)}   `
              + `${fx(r.err, 2).padStart(7)}  ${r.err < 1e-9 ? '✓' : '✗'}`).join('\n')
            + '\n\n'
            + `  writes issued:\n`
            + rows.map((r, i) => `    ${i + 1}. ${vec(r.write, 0)}`).join('\n')
            + '\n\n'
            + (rule === 'delta'
              ? 'Every write is measured against what memory actually holds, so the state stays on\n'
                + 'target no matter how many revisions arrive. Several of those writes are negative\n'
                + 'in some slots — the delta rule removes information as readily as it adds it.'
              : 'Nothing is ever removed, so every stale answer stays in the sum. The final state is\n'
                + 'the total of all four targets rather than the last one, and the error grows with\n'
                + 'every revision.'),
        }),
      );
    }

    render();
  },
);
