/* §14 — State that outlives the window.
   A document is split into chunks. Chunk 1 ends, its internal attention state is
   discarded, chunk 2 begins. Without an explicit mechanism, nothing crosses.

   V4's Memory Stream carries exactly one thing forward: the final hidden state of
   the chunk, as one model-width vector. Two rules make it practical —
       memory        = stop_gradient(final hidden state)
       updated token = current token + scale × gate × memory

   Reported injection scale rises from ~0.078 to ~0.391 by the 2B stage, with the
   average injected memory about 6% of the current embedding magnitude. With the
   vectors below and scale 0.391, a gate near 0.18 reproduces that 6% — so the
   default lands on the reported operating point rather than an invented one. */

import {
  register, el, matrixTable as mt, stepper, slider, toggle, readout, svg,
  fx, clear,
} from '../lib.js';

const D = 4;
const AXES = ['0', '1', '2', '3'];

/* One token in chunk 2, and the summary vector arriving from chunk 1. */
const CURRENT = [0.8, -0.5, 0.6, 0.3];
const MEMORY = [0.4, 0.7, -0.3, 0.5];

const SCALES = [0.078, 0.195, 0.391];        // reported: seed → mid → 2B stage
const norm = (v) => Math.hypot(...v);

const CHUNK1 = 'Chunk 1 … the review meeting was moved to Osaka on the 14th … ';
const CHUNK2 = '… Chunk 2 — where is the review meeting being held?';

const STEPS = [
  { key: 'step 1', title: 'The boundary' },
  { key: 'step 2', title: 'One vector crosses' },
  { key: 'step 3', title: 'The learned gate' },
  { key: 'step 4', title: 'Stop-gradient' },
];

register(
  'w-s14-memory-stream',
  'A nudge across the chunk boundary',
  '§14 · O(1) cross-chunk state',
  'Turn the stream off and nothing survives the boundary. Turn it on and one model-width '
  + 'vector crosses — no more, however long the document. Then move the gate: closed ignores '
  + 'the summary, and the default setting is the reported ~6% nudge.',
  (root) => {
    let step = 0;
    let on = true;
    let gate = 0.18;
    let scale = 0.391;

    const stage = el('div', { class: 'stage' });
    const ros = el('div', { class: 'readouts' });

    const seq = stepper({ steps: STEPS, value: 0, onChange: (s) => { step = s; render(); } });

    const onSw = toggle({
      label: 'Memory stream on', value: true, onChange: (v) => { on = v; render(); },
    });
    const gS = slider({
      label: 'Learned gate', min: 0, max: 1, step: 0.01, value: gate,
      fmt: (v) => fx(v, 2), onInput: (v) => { gate = v; render(); },
    });
    const sS = slider({
      label: 'Injection scale', values: SCALES, value: scale,
      fmt: (v) => `${v}${v === 0.391 ? ' · 2B stage' : v === 0.078 ? ' · seed' : ''}`,
      onInput: (v) => { scale = v; render(); },
    });

    root.append(
      seq.node,
      el('div', { class: 'controls' },
        el('div', { class: 'ctl' }, onSw.node), gS.node, sS.node),
      stage, ros,
    );

    const factor = () => (on ? scale * gate : 0);
    const injected = () => MEMORY.map((v) => v * factor());
    const updated = () => CURRENT.map((v, i) => v + injected()[i]);
    const pct = () => (norm(injected()) / norm(CURRENT)) * 100;

    function render() {
      clear(stage); clear(ros);
      gS.node.style.opacity = on ? '1' : '.4';
      sS.node.style.opacity = on ? '1' : '.4';

      if (step === 0) stepBoundary();
      else if (step === 1) stepCross();
      else if (step === 2) stepGate();
      else stepStopGrad();

      addReadouts([
        {
          label: 'Stream', value: on ? 'on' : 'off', tone: on ? 'aqua' : 'red',
          sub: on ? 'one vector crosses' : 'nothing crosses',
        },
        {
          label: 'Injection factor', value: fx(factor(), 4), tone: 'blue',
          sub: `scale ${scale} × gate ${fx(gate, 2)}`,
        },
        {
          label: 'Injected magnitude', value: `${fx(pct(), 1)}%`,
          tone: pct() < 1 ? 'red' : pct() < 15 ? 'aqua' : 'orange',
          sub: 'of the current token — reported ≈6%',
        },
        {
          label: 'Carried forward', value: `${D} numbers`, tone: 'violet',
          sub: 'model width — never more',
        },
        {
          label: 'After 2,000 chunks', value: `${D} numbers`, tone: 'violet',
          sub: 'this is what O(1) means',
        },
      ]);
    }

    function addReadouts(list) { list.forEach((r) => ros.append(readout(r).node)); }

    /* ---- step 1 ---- */
    function stepBoundary() {
      stage.append(
        el('p', { class: 'stage-title', text: 'Chunk 1 ends. What survives?' }),
        el('p', {
          class: 'stage-sub',
          text: 'Everything so far has described memory inside the current window. Split a long '
            + 'document into chunks and chunk 1\'s internal attention state is discarded when chunk '
            + '2 begins. Without an explicit mechanism, the answer to "what crosses" may be: none '
            + 'of it.',
        }),
        boundaryDiagram(),
        el('pre', {
          class: 'ascii',
          text: `  ${CHUNK1}\n`
            + `  ${'─'.repeat(62)}   ← chunk boundary: KV cache and state discarded\n`
            + `  ${CHUNK2}\n\n`
            + (on
              ? '  With the stream ON, one model-width vector is handed forward.'
              : '  With the stream OFF, chunk 2 begins with no trace of chunk 1. The question\n'
                + '  in chunk 2 refers to something the model can no longer see at all.'),
        }),
      );
    }

    /* ---- step 2 ---- */
    function stepCross() {
      stage.append(
        el('p', { class: 'stage-title', text: 'The final hidden state becomes one summary vector' }),
        el('p', {
          class: 'stage-sub',
          text: 'At the end of a chunk, take its final hidden state and use it as the summary for '
            + 'the next chunk. The vector has the same width as the model. It does not get longer '
            + 'when the document does — after two chunks or two thousand, the stream is still one '
            + 'model-width vector. That is fixed-size, or O(1), cross-chunk state.',
        }),
        el('pre', {
          class: 'ascii',
          text: '  chunk 1 ──writes──> one memory vector ──nudges──> chunk 2\n\n'
            + `  memory = ${on ? `[ ${MEMORY.map((v) => fx(v)).join('   ')} ]` : 'nothing — stream is off'}\n\n`
            + `  after 2 chunks      → ${D} numbers\n`
            + `  after 200 chunks    → ${D} numbers\n`
            + `  after 2,000 chunks  → ${D} numbers\n\n`
            + '  It is also a severe compression. One vector cannot preserve every sentence\n'
            + '  from every earlier chunk. It can only carry a useful summary signal.',
        }),
        el('div', { class: 'gridrow' },
          mt({
            caption: 'chunk 1 · final hidden state', data: MEMORY.map((v) => [v]),
            rowLabels: AXES, colLabels: ['h'], digits: 2,
            cell: () => ({ bg: on ? 'rgba(27,175,122,.20)' : 'rgba(255,255,255,.04)' }),
          }),
          el('div', { class: 'op', text: '→' }),
          mt({
            caption: on ? 'memory, crossing' : 'nothing crosses',
            data: (on ? MEMORY : [0, 0, 0, 0]).map((v) => [v]),
            rowLabels: AXES, colLabels: ['m'], digits: 2,
            cell: (v) => ({
              bg: on ? 'rgba(144,133,233,.22)' : null,
              color: on ? null : '#4a4a52',
            }),
          }),
        ),
      );
    }

    /* ---- step 3 ---- */
    function stepGate() {
      const inj = injected();
      const upd = updated();
      stage.append(
        el('p', { class: 'stage-title', text: 'Each token gets a small learned gate' }),
        el('p', {
          class: 'stage-sub',
          text: 'The gate is between zero and one, and it is learned because different tokens need '
            + 'different amounts of old information. A question referring to the previous chunk may '
            + 'use the stream; an unrelated token may largely ignore it.',
        }),
        el('div', { class: 'eq', text: 'updated token = current token + scale × gate × memory' }),
        el('div', { class: 'gridrow' },
          mt({
            caption: 'current token', data: CURRENT.map((v) => [v]),
            rowLabels: AXES, colLabels: ['x'], digits: 2,
            cell: () => ({ bg: 'rgba(57,135,229,.18)' }),
          }),
          el('div', { class: 'op', text: '+' }),
          mt({
            caption: `${fx(factor(), 4)} × memory`, data: inj.map((v) => [v]),
            rowLabels: AXES, colLabels: ['Δ'], digits: 4,
            cell: (v) => ({
              bg: Math.abs(v) > 1e-9 ? 'rgba(144,133,233,.22)' : null,
              color: Math.abs(v) < 1e-9 ? '#4a4a52' : null,
            }),
          }),
          el('div', { class: 'op', text: '=' }),
          mt({
            caption: 'updated token', data: upd.map((v) => [v]),
            rowLabels: AXES, colLabels: ["x'"], digits: 4,
            cell: () => ({ bg: 'rgba(27,175,122,.20)' }),
          }),
        ),
        el('pre', {
          class: 'ascii',
          text: `  scale × gate = ${scale} × ${fx(gate, 2)} = ${fx(factor(), 4)}\n\n`
            + inj.map((v, i) => `  x'[${i}] = ${fx(CURRENT[i]).padStart(5)}`
              + ` + ${fx(factor(), 4)}×${fx(MEMORY[i]).padStart(5)}`
              + ` = ${fx(CURRENT[i], 4).padStart(7)} + ${fx(v, 4).padStart(7)}`
              + ` = ${fx(upd[i], 4)}`).join('\n')
            + `\n\n  |injected| / |current| = ${fx(norm(inj), 4)} / ${fx(norm(CURRENT), 4)}`
            + ` = ${fx(pct(), 1)}%\n\n`
            + (gate < 0.02
              ? '  gate near 0 → this token mostly ignores the old summary.'
              : pct() > 25
                ? '  large gate → the old summary begins to dominate the present. This is past\n'
                  + '  the reported operating point; the stream is meant to nudge, not overwrite.'
                : '  small gate → the old summary gives this token a nudge. The reported average\n'
                  + '  is about 6% of the current embedding magnitude, which is roughly here.'),
        }),
        magnitudeBar(),
      );
    }

    /* ---- step 4 ---- */
    function stepStopGrad() {
      stage.append(
        el('p', { class: 'stage-title', text: 'Information moves forward. Gradients do not move back.' }),
        el('p', {
          class: 'stage-sub',
          text: 'The vector is written with stop-gradient. The next chunk can read that memory during '
            + 'the forward pass, but its training loss cannot send gradients backward through the '
            + 'boundary into all the previous chunks — which is what stops the training graph growing '
            + 'to the size of the whole document.',
        }),
        el('div', { class: 'eq', text: 'memory = stop_gradient(final hidden state)' }),
        stopGradDiagram(),
        el('pre', {
          class: 'ascii',
          text: '  forward   chunk 1 ──────────────────────>  chunk 2      allowed\n'
            + '  backward  chunk 1 <──────╳──────────────── chunk 2      blocked\n\n'
            + '  Without this, training one chunk would require holding the activations of\n'
            + '  every earlier chunk. The graph would grow with document length, which is\n'
            + '  exactly the cost the fixed-size stream exists to avoid.\n\n'
            + `  reported injection scale:  0.078 at the seed stage → ${SCALES[2]} by 2B\n`
            + '  reported average injection: about 6% of the current embedding magnitude\n\n'
            + '  The right mental model: the Memory Stream is a small nudge from the previous\n'
            + '  chunk. Not a copy of the old chunk, and not a replacement for the present token.',
        }),
        el('p', {
          class: 'note warn',
          html: '<b>What this widget shows and does not show.</b> The arithmetic above is exactly the '
            + 'reported update rule, at the reported scale. Whether a 6% nudge is <em>enough</em> to '
            + 'answer a question about the previous chunk is an empirical property of a trained '
            + 'model, and nothing here demonstrates it. §16 lists as an open question whether the '
            + 'Memory Stream still earns its place once the ordinary context window is already very '
            + 'long.',
        }),
      );
    }

    /* ---- diagrams ---- */
    function boundaryDiagram() {
      const W = 660, H = 120;
      const g = svg('svg', {
        viewBox: `0 0 ${W} ${H}`, width: '100%',
        style: 'display:block;max-width:100%;height:auto;margin:4px 0 14px', role: 'img',
        'aria-label': on ? 'One vector crosses the chunk boundary' : 'Nothing crosses the chunk boundary',
      });
      g.append(
        svg('rect', { x: 0, y: 26, width: 270, height: 54, rx: 8, fill: 'rgba(57,135,229,.14)', stroke: 'rgba(57,135,229,.4)' }),
        svg('text', { x: 135, y: 48, 'text-anchor': 'middle', 'font-size': 12, fill: '#8fbcf5', 'font-weight': 600, text: 'chunk 1' }),
        svg('text', { x: 135, y: 66, 'text-anchor': 'middle', 'font-size': 10, fill: '#89877f', text: 'attention state, then discarded' }),
        svg('line', { x1: 330, y1: 12, x2: 330, y2: 96, stroke: 'rgba(224,82,82,.55)', 'stroke-width': 2, 'stroke-dasharray': '5 4' }),
        svg('text', { x: 330, y: 108, 'text-anchor': 'middle', 'font-size': 10, fill: '#ef8a8a', text: 'boundary' }),
        svg('rect', { x: 390, y: 26, width: 270, height: 54, rx: 8, fill: 'rgba(27,175,122,.12)', stroke: 'rgba(27,175,122,.4)' }),
        svg('text', { x: 525, y: 48, 'text-anchor': 'middle', 'font-size': 12, fill: '#5fd2a8', 'font-weight': 600, text: 'chunk 2' }),
        svg('text', { x: 525, y: 66, 'text-anchor': 'middle', 'font-size': 10, fill: '#89877f', text: 'starts fresh' }),
      );
      if (on) {
        g.append(
          svg('path', { d: 'M 272 40 C 300 40, 360 40, 388 40', fill: 'none', stroke: '#9085e9', 'stroke-width': 2 }),
          svg('polygon', { points: '388,40 380,36 380,44', fill: '#9085e9' }),
          svg('text', { x: 330, y: 32, 'text-anchor': 'middle', 'font-size': 9.5, fill: '#b5aef2', text: '1 vector' }),
        );
      } else {
        g.append(
          svg('path', { d: 'M 272 40 C 300 40, 310 40, 316 40', fill: 'none', stroke: 'rgba(224,82,82,.6)', 'stroke-width': 2 }),
          svg('text', { x: 300, y: 32, 'text-anchor': 'middle', 'font-size': 9.5, fill: '#ef8a8a', text: 'nothing' }),
        );
      }
      return g;
    }

    function stopGradDiagram() {
      const W = 660, H = 108;
      const g = svg('svg', {
        viewBox: `0 0 ${W} ${H}`, width: '100%',
        style: 'display:block;max-width:100%;height:auto;margin:4px 0 12px', role: 'img',
        'aria-label': 'Forward pass crosses the boundary; gradients are blocked',
      });
      g.append(
        svg('rect', { x: 0, y: 30, width: 250, height: 48, rx: 8, fill: 'rgba(57,135,229,.12)', stroke: 'rgba(57,135,229,.35)' }),
        svg('text', { x: 125, y: 59, 'text-anchor': 'middle', 'font-size': 12, fill: '#8fbcf5', text: 'chunk 1' }),
        svg('rect', { x: 410, y: 30, width: 250, height: 48, rx: 8, fill: 'rgba(27,175,122,.12)', stroke: 'rgba(27,175,122,.35)' }),
        svg('text', { x: 535, y: 59, 'text-anchor': 'middle', 'font-size': 12, fill: '#5fd2a8', text: 'chunk 2' }),
        // forward
        svg('path', { d: 'M 252 44 L 406 44', stroke: '#5fd2a8', 'stroke-width': 2 }),
        svg('polygon', { points: '408,44 400,40 400,48', fill: '#5fd2a8' }),
        svg('text', { x: 330, y: 36, 'text-anchor': 'middle', 'font-size': 10, fill: '#5fd2a8', text: 'forward: memory' }),
        // backward, blocked
        svg('path', { d: 'M 406 66 L 252 66', stroke: 'rgba(224,82,82,.75)', 'stroke-width': 2 }),
        svg('polygon', { points: '250,66 258,62 258,70', fill: 'rgba(224,82,82,.75)' }),
        svg('line', { x1: 322, y1: 56, x2: 338, y2: 76, stroke: '#e05252', 'stroke-width': 2.6 }),
        svg('line', { x1: 338, y1: 56, x2: 322, y2: 76, stroke: '#e05252', 'stroke-width': 2.6 }),
        svg('text', { x: 330, y: 92, 'text-anchor': 'middle', 'font-size': 10, fill: '#ef8a8a', text: 'backward: stop_gradient' }),
      );
      return g;
    }

    /** Current token magnitude against the injected nudge. */
    function magnitudeBar() {
      const W = 640, H = 84, L = 96, R = 96;
      const iw = W - L - R;
      const g = svg('svg', {
        viewBox: `0 0 ${W} ${H}`, width: '100%',
        style: 'display:block;max-width:100%;height:auto;margin:8px 0 2px', role: 'img',
        'aria-label': `Injected memory is ${fx(pct(), 1)} percent of the current token magnitude`,
      });
      const bar = (y, frac, colour, label, note) => g.append(
        svg('text', { x: L - 12, y: y + 16, 'text-anchor': 'end', 'font-size': 11, fill: '#b8b7b0', text: label }),
        svg('rect', { x: L, y, width: iw, height: 22, rx: 4, fill: 'rgba(255,255,255,.05)' }),
        svg('rect', { x: L, y, width: Math.max(1, iw * Math.min(1, frac)), height: 22, rx: 4, fill: colour }),
        svg('text', { x: L + iw + 8, y: y + 16, 'font-size': 10.5, fill: '#89877f', text: note }),
      );
      bar(6, 1, '#3987e5', 'current token', fx(norm(CURRENT), 4));
      bar(38, norm(injected()) / norm(CURRENT), '#9085e9', 'injected memory',
        `${fx(norm(injected()), 4)}  (${fx(pct(), 1)}%)`);
      g.append(svg('text', {
        x: L, y: 76, 'font-size': 10, fill: '#89877f',
        text: 'the reported operating point is a nudge of roughly 6%, not a replacement',
      }));
      return g;
    }

    render();
  },
);
