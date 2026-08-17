/* §5 — A new state can still carry the old contribution.
   Two writes to one fixed-size state. The first makes key A return 40. The answer
   then changes to 55, and an add-only write produces 95 instead — because the new
   state was computed as (old state + the complete new answer).

   The arithmetic is exact, not illustrative. kA is a unit vector, so reading the
   state with it returns the written answer vector unchanged:
       read = S kA = (a kAᵀ) kA = a (kA·kA) = a       when |kA| = 1
   Answers point along [1,0,0,0], so "the answer" is the first component.

   §5 deliberately stops at the problem. The delta rule is §6. */

import {
  register, el, matrixTable as mt, stepper, readout, segmented,
  outer, addM, matvec, zeros, dot, fx, clear,
} from '../lib.js';

const D = 4;
const AXES = ['0', '1', '2', '3'];

const kA = [0.5, 0.5, 0.5, 0.5];                     // |kA| = 1
const kB = [Math.SQRT1_2, Math.SQRT1_2, 0, 0];       // |kB| = 1, not orthogonal to kA

const OLD = 40;
const WANTED = 55;

const answer = (m) => [m, 0, 0, 0];
const readMag = (S, k) => matvec(S, k)[0];

/* The three states this widget ever shows. Only one exists at a time. */
const S0 = zeros(D, D);
const S1 = outer(answer(OLD), kA);
const S2 = addM(S1, outer(answer(WANTED), kA));      // add-only write

const STEPS = [
  { key: 'step 1', title: 'Empty scratchpad' },
  { key: 'step 2', title: 'Write: A → 40' },
  { key: 'step 3', title: 'The answer changes' },
  { key: 'step 4', title: 'Add-only write' },
];

register(
  'w-s05-add-only-state',
  'One state, two writes, the wrong answer',
  '§5 · add-only interference',
  'Step through the two writes in order. The first makes key A return 40. The second is '
  + 'computed as old state plus the complete new answer, so key A returns 95 instead of 55. '
  + 'There are never two state matrices — only one, whose numbers still contain the old '
  + 'answer.',
  (root) => {
    let step = 0;
    let probe = 'A';

    const stage = el('div', { class: 'stage' });
    const ros = el('div', { class: 'readouts' });

    const seq = stepper({ steps: STEPS, value: 0, onChange: (i) => { step = i; render(); } });

    const probeSel = segmented({
      label: 'Read the state with',
      options: [
        { value: 'A', label: 'key A', title: 'the key that was written' },
        { value: 'B', label: 'key B', title: 'a key that was never written' },
      ],
      value: probe,
      onChange: (v) => { probe = v; render(); },
    });

    root.append(
      seq.node,
      el('div', { class: 'controls' }, probeSel.node),
      stage,
      ros,
    );

    const stateAt = (s) => (s === 0 ? S0 : s === 1 ? S1 : s === 2 ? S1 : S2);
    const k = () => (probe === 'A' ? kA : kB);

    function render() {
      clear(stage); clear(ros);
      const S = stateAt(step);
      const readA = readMag(S, kA);
      const readB = readMag(S, kB);
      const shown = probe === 'A' ? readA : readB;

      const titles = [
        'Nothing written yet',
        'One association is stored',
        'The stored answer is now out of date',
        'One new state — and it still carries the old answer',
      ];
      const subs = [
        'The state is a fixed-size grid of numbers, all zero. Reading it with any key returns '
        + 'nothing. Think of it as a scratchpad with two jobs: WRITE connects a clue to an '
        + 'answer, READ uses the clue to recover it.',
        'The write was S = S + (answer × keyᵀ). Reading with key A now returns 40, so the state '
        + 'is genuinely working as a memory. Note the state did not need to grow to hold this.',
        'Suppose the answer for key A should now be 55 instead of 40. The new answer should '
        + 'replace the old one. Watch what an add-only write actually does.',
        'S = old state + complete new answer. There is only one matrix now — but it was '
        + 'calculated starting from the old state, so the old answer is still inside the sum. '
        + 'Key A returns 40 + 55 = 95.',
      ];

      stage.append(
        el('p', { class: 'stage-title', text: titles[step] }),
        el('p', { class: 'stage-sub', text: subs[step] }),
      );

      /* the write being performed, as arithmetic */
      if (step === 1 || step === 3) {
        const add = step === 1 ? OLD : WANTED;
        stage.append(el('pre', {
          class: 'ascii',
          text: `write:  S = S + (answer × keyᵀ)\n`
            + `           answer = [ ${answer(add).map((v) => fx(v, 0)).join('  ')} ]`
            + `   key A = [ ${kA.map((v) => fx(v)).join('  ')} ]\n\n`
            + (step === 3
              ? `        the OLD contribution is still in S, because the calculation\n`
                + `        started from the old state — nothing cancelled it\n`
              : ''),
        }));
        stage.append(writeChain(step));
      }

      /* the state, and the read */
      stage.append(
        el('div', { class: 'gridrow' },
          mt({
            caption: step === 0 ? 'S · empty' : `S · after ${step === 3 ? 'two writes' : 'one write'}`,
            data: S, rowLabels: AXES, colLabels: AXES, digits: 1,
            cell: (v) => ({
              bg: Math.abs(v) > 1e-9 ? 'rgba(57,135,229,.14)' : null,
              color: Math.abs(v) < 1e-9 ? '#4a4a52' : null,
            }),
          }),
          el('div', { class: 'op', text: '×' }),
          mt({
            caption: `key ${probe}`, data: k().map((v) => [v]),
            rowLabels: AXES, colLabels: ['k'], digits: 3,
            cell: () => ({ bg: 'rgba(27,175,122,.14)' }),
          }),
          el('div', { class: 'op', text: '=' }),
          mt({
            caption: `read → ${fx(shown, 2)}`, data: matvec(S, k()).map((v) => [v]),
            rowLabels: AXES, colLabels: ['y'], digits: 2,
            cell: (v, i) => ({ bg: i === 0 ? 'rgba(144,133,233,.20)' : null }),
          }),
        ),
      );

      if (step === 3) {
        stage.append(el('pre', {
          class: 'ascii',
          text: `wanted:   key A → ${WANTED}\n`
            + `got:      key A → ${fx(readA, 0)}     ${fx(OLD, 0)} + ${fx(WANTED, 0)}\n\n`
            + `The write added the whole new answer when it should have added only the\n`
            + `part that was missing: ${WANTED} − ${OLD} = ${WANTED - OLD}. That correction is §6.`,
        }));
      }

      if (probe === 'B' && step > 0) {
        stage.append(el('p', {
          class: 'stage-sub',
          text: `Key B was never written to, yet it reads ${fx(readB, 2)}. Because key B is not `
            + `orthogonal to key A (kA·kB = ${fx(dot(kA, kB), 3)}), it partly overlaps the same `
            + 'directions in the state, so it picks up some of what was written for A. This is '
            + 'interference: one fixed-size state holding many associations that cannot all be '
            + 'kept perfectly separate.',
        }));
      }

      const err = readA - WANTED;
      addReadouts([
        {
          label: 'State shape', value: `${D} × ${D}`, sub: `${D * D} numbers — never grows`,
        },
        {
          label: 'Matrices in memory', value: '1',
          sub: 'one current state, not a history', tone: 'blue',
        },
        {
          label: 'Key A reads', value: fx(readA, 1),
          tone: step === 3 ? 'red' : 'aqua',
          sub: step === 3 ? `wanted ${WANTED}` : 'as written',
        },
        {
          label: 'Key B reads', value: fx(readB, 1), tone: 'orange',
          sub: 'never written — interference',
        },
        {
          label: 'Error', value: step === 3 ? `+${fx(err, 0)}` : '0',
          tone: step === 3 ? 'red' : 'aqua',
          sub: step === 3 ? 'the old answer, carried forward' : 'nothing stale yet',
        },
      ]);
    }

    function addReadouts(list) { list.forEach((r) => ros.append(readout(r).node)); }

    /**
     * The write drawn as a grid addition, the same way §4 step 1 accumulates
     * tokens. The shape is unchanged by the write — which is exactly the trouble:
     * there is no new cell for the new answer to live in, so it lands on top of
     * the old one in the same sixteen cells.
     */
    function writeChain(s) {
      const before = s === 1 ? S0 : S1;
      const added = outer(answer(s === 1 ? OLD : WANTED), kA);
      const after = s === 1 ? S1 : S2;
      const label = s === 1 ? 'first' : 'second';

      const row = el('div', { class: 'gridrow' },
        mt({
          caption: s === 1 ? 'S before  ·  empty' : `S before  ·  holds ${OLD}`,
          data: before, rowLabels: AXES, colLabels: AXES, digits: 1,
          cell: (v) => ({
            bg: Math.abs(v) > 1e-9 ? 'rgba(57,135,229,.14)' : null,
            color: Math.abs(v) < 1e-9 ? '#4a4a52' : null,
          }),
        }),
        el('div', { class: 'op', text: '+' }),
        mt({
          caption: `answer ${s === 1 ? OLD : WANTED} × kAᵀ`,
          data: added, rowLabels: AXES, colLabels: AXES, digits: 1,
          cell: (v) => ({
            bg: Math.abs(v) > 1e-9 ? 'rgba(235,104,52,.20)' : null,
            color: Math.abs(v) < 1e-9 ? '#4a4a52' : null,
          }),
        }),
        el('div', { class: 'op', text: '=' }),
        mt({
          caption: `S after the ${label} write`,
          data: after, rowLabels: AXES, colLabels: AXES, digits: 1,
          cell: (v, i) => ({
            bg: i === 0 ? 'rgba(144,133,233,.24)' : 'rgba(255,255,255,.03)',
            color: Math.abs(v) < 1e-9 ? '#4a4a52' : null,
          }),
        }),
      );

      const cell = (M) => fx(M[0][0], 1).padStart(6);
      const trace = s === 1
        ? `  before        S is ${D}×${D}   S[0][0] = ${cell(S0)}   key A reads  ${fx(readMag(S0, kA), 1).padStart(5)}\n`
          + `  after write 1 S is ${D}×${D}   S[0][0] = ${cell(S1)}   key A reads  ${fx(readMag(S1, kA), 1).padStart(5)}`
        : `  after write 1 S is ${D}×${D}   S[0][0] = ${cell(S1)}   key A reads  ${fx(readMag(S1, kA), 1).padStart(5)}\n`
          + `  after write 2 S is ${D}×${D}   S[0][0] = ${cell(S2)}   key A reads  ${fx(readMag(S2, kA), 1).padStart(5)}`
          + `   ← wanted ${WANTED}`;

      return el('div', {}, row, el('pre', {
        class: 'ascii',
        text: `${trace}\n\n`
          + `Row 0, cell by cell:  ${before[0].map((v) => fx(v, 1)).join(', ')}`
          + `  +  ${added[0].map((v) => fx(v, 1)).join(', ')}`
          + `  =  ${after[0].map((v) => fx(v, 1)).join(', ')}\n\n`
          + (s === 1
            ? `The state is still ${D}×${D}, exactly as in §4 — a write never changes its shape.`
            : `The state is still ${D}×${D}. And that is the problem: the write had nowhere new\n`
              + `to put ${WANTED}, so it landed on top of ${OLD} in the same sixteen cells. Both are\n`
              + `now inside every number in row 0, and ${OLD} + ${WANTED} = ${OLD + WANTED} is what comes back out.`),
      }));
    }

    render();
  },
);
