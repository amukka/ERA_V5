/* §9 — DroPE, and how 8K became 256K.
   The one thing this widget must not do is animate a mechanism the record does
   not specify. So everything here is either (a) arithmetic that follows from
   RoPE itself, or (b) a quoted figure from the V4 record, clearly labelled.

   The honest live demonstration is the turn count. Note carefully WHAT is out of
   distribution: not absolute position — §8 measures that as cancelling exactly,
   to 1e-16 even at a million. It is the relative GAP. A model trained at 8K has
   seen gaps up to 8192; serving 256K requires gaps up to 262144, and across those
   the slow pairs sweep angle ranges they never met. That is computable, it is not
   a hypothesis, and it is why every published fix rescales the low frequencies. */

import {
  register, el, matrixTable as mt, stepper, slider, readout, svg,
  fx, tokens, clear,
} from '../lib.js';

const D = 8;
const PAIRS = D / 2;
const BASE = 10000;
const THETA = Array.from({ length: PAIRS }, (_, p) => 1 / BASE ** ((2 * p) / D));

const TRAINED = 8192;                 // V4's reported training context
const REPORTED = 262144;              // V4's reported serving context
const TARGETS = [8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576];

const turns = (pos, th) => (pos * th) / (2 * Math.PI);

const STEPS = [
  { key: 'step 1', title: 'The rotation is always defined' },
  { key: 'step 2', title: 'The gap is what is new' },
  { key: 'step 3', title: 'What the record says' },
];

register(
  'w-s09-drope',
  'Defined is not proven',
  '§9 · context extension',
  'RoPE can be evaluated at any position — the formula never runs out. That is not the same '
  + 'as the model being able to use that position. Move the target context and watch what the '
  + 'slow pairs are being asked to do compared with what they met in training.',
  (root) => {
    let step = 0;
    let target = REPORTED;

    const stage = el('div', { class: 'stage' });
    const ros = el('div', { class: 'readouts' });

    const seq = stepper({ steps: STEPS, value: 0, onChange: (s) => { step = s; render(); } });

    const tS = slider({
      label: 'Target context', values: TARGETS, value: target,
      fmt: (v) => `${tokens(v)} tokens`,
      onInput: (v) => { target = v; render(); },
    });

    root.append(seq.node, el('div', { class: 'controls' }, tS.node), stage, ros);

    function render() {
      clear(stage); clear(ros);
      const factor = target / TRAINED;
      const slow = THETA[PAIRS - 1];

      if (step === 0) stepDefined();
      else if (step === 1) stepTrained();
      else stepRecord();

      addReadouts([
        { label: 'Trained context', value: tokens(TRAINED), sub: 'V4, reported' },
        {
          label: 'Target context', value: tokens(target), tone: 'violet',
          sub: 'what we want to serve',
        },
        {
          label: 'Extension factor', value: `${fx(factor, factor % 1 ? 1 : 0)}×`,
          tone: factor <= 32 ? 'aqua' : 'red',
          sub: factor <= 32 ? 'within V4\'s reported 32×' : 'beyond any reported result',
        },
        {
          label: 'Slow pair, trained gaps', value: `${fx(turns(TRAINED, slow), 2)} turns`,
          tone: 'blue', sub: `widest gap ${tokens(TRAINED)}, θ = ${slow}`,
        },
        {
          label: 'Slow pair, target gaps', value: `${fx(turns(target, slow), 2)} turns`,
          tone: turns(target, slow) > turns(TRAINED, slow) * 2 ? 'red' : 'orange',
          sub: 'distances never trained on',
        },
      ]);
    }

    function addReadouts(list) { list.forEach((r) => ros.append(readout(r).node)); }

    /* ---- step 1: the formula never runs out ---- */
    function stepDefined() {
      const rows = THETA.map((th) => {
        const a = target * th;
        return [th, a, ((a % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI), Math.cos(a), Math.sin(a)];
      });
      stage.append(
        el('p', { class: 'stage-title', text: `Ask RoPE for the rotation at position ${tokens(target)}` }),
        el('p', {
          class: 'stage-sub',
          text: 'It answers. There is no lookup table to fall off the end of — the angle is computed '
            + 'from a function, so it exists at every position, including positions no model has '
            + 'ever been trained on. Every number below is a real evaluation.',
        }),
        mt({
          caption: `rotation at position ${tokens(target)}`, data: rows,
          rowLabels: THETA.map((_, p) => `pair ${p}`),
          colLabels: ['θ', 'angle (rad)', 'angle mod 2π', 'cos', 'sin'],
          digits: 4,
          cell: (v, r, c) => ({
            text: c === 1 ? fx(v, 1) : fx(v, 4),
            bg: c >= 3 ? 'rgba(57,135,229,.14)' : null,
          }),
        }),
        el('pre', {
          class: 'ascii',
          text: `  pair 0 at position ${tokens(target)}:\n`
            + `    angle = ${target} × ${THETA[0]} = ${fx(target * THETA[0], 1)} rad\n`
            + `    cos   = ${fx(Math.cos(target * THETA[0]), 6)}\n`
            + `    sin   = ${fx(Math.sin(target * THETA[0]), 6)}\n\n`
            + '  So this is TRUE:      the positional rule is defined at '
            + `${tokens(target)}\n`
            + '  And this is NOT shown: the model can use a '
            + `${tokens(target)} context reliably\n\n`
            + '  Session 7 showed the hard wall a stored absolute position table creates —\n'
            + '  ask for row 262144 of an 8192-row table and there is nothing there. RoPE\n'
            + '  removes that wall. It does not follow that the model behaves well past the\n'
            + '  training length; that is a claim about the whole trained network, not the\n'
            + '  positional function.',
        }),
      );
    }

    /* ---- step 2: what training actually covered ---- */
    function stepTrained() {
      const rows = THETA.map((th) => [
        th, turns(TRAINED, th), turns(target, th), turns(target, th) / turns(TRAINED, th),
      ]);
      stage.append(
        el('p', { class: 'stage-title', text: 'The unfamiliar thing is the GAP, not the position' }),
        el('p', {
          class: 'stage-sub',
          text: 'It is tempting to say the model has never seen "position 200,000". But §8 measured '
            + 'the opposite: absolute position cancels exactly, so shifting a whole sentence a '
            + 'hundred places leaves every score identical to the last decimal. What genuinely never '
            + 'occurred in training is a large relative DISTANCE — and the longest possible distance '
            + 'is the context length itself.',
        }),
        el('pre', {
          class: 'ascii',
          text: `  trained at ${tokens(TRAINED)}   →  gaps from 1 up to ${TRAINED} were seen\n`
            + `  serving at ${tokens(target)}  →  gaps from 1 up to ${target} must work\n`
            + `                        →  ${commaify(target - TRAINED)} distances never trained on`,
        }),
        el('p', {
          class: 'stage-sub',
          text: 'Each pair sweeps a number of turns across the widest gap it has to represent. The '
            + 'fast pairs already wrapped thousands of times inside 8K, so nothing new happens to '
            + 'them. The damage is confined to the slow pairs — which are exactly the ones carrying '
            + 'long-range distance.',
        }),
        mt({
          caption: `turns swept across the widest gap  ·  ${tokens(TRAINED)} trained vs ${tokens(target)} target`,
          data: rows,
          rowLabels: THETA.map((_, p) => `pair ${p}`),
          colLabels: ['θ', `gap ${tokens(TRAINED)}`, `gap ${tokens(target)}`, 'ratio'],
          digits: 2,
          cell: (v, r, c) => ({
            text: c === 0 ? String(THETA[r]) : fx(v, 2),
            bg: c === 3 && v > 1
              ? (r === PAIRS - 1 ? 'rgba(224,82,82,.24)' : 'rgba(235,104,52,.14)')
              : null,
          }),
        }),
        turnBars(),
        el('pre', {
          class: 'ascii',
          text: `  pair ${PAIRS - 1} is the long-range one, θ = ${THETA[PAIRS - 1]}`
            + `  (wavelength ${fx((2 * Math.PI) / THETA[PAIRS - 1], 0)} tokens)\n\n`
            + `    widest gap in training, ${TRAINED}:  ${fx(turns(TRAINED, THETA[PAIRS - 1]), 2)} turns\n`
            + `    widest gap at target,  ${target}:  ${fx(turns(target, THETA[PAIRS - 1]), 2)} turns\n\n`
            + `  Everything past ${fx(turns(TRAINED, THETA[PAIRS - 1]), 2)} turns is an angle regime this pair never\n`
            + '  encountered while learning. The function still returns a number; the trained\n'
            + '  weights have no experience of what it means.\n\n'
            + '  This is why the published extension methods — Position Interpolation,\n'
            + '  NTK-aware scaling, YaRN — all work by rescaling the LOW frequencies. They are\n'
            + '  fixing the gap range, not the absolute position, because the absolute position\n'
            + '  was never the problem.',
        }),
      );
    }

    const commaify = (n) => Math.round(n).toLocaleString('en-US');

    /* ---- step 3: the record, and its boundary ---- */
    function stepRecord() {
      stage.append(
        el('p', { class: 'stage-title', text: 'What the V4 record establishes, and what it does not' }),
        el('pre', {
          class: 'ascii',
          text: `  trained context:    8K\n`
            + `  reported context:   256K\n`
            + `  extension:          32×          256K ÷ 8K = ${REPORTED / TRAINED}\n\n`
            + `  positional recalibration: DroPE, applied BEFORE annealing`,
        }),
        el('p', {
          class: 'stage-sub',
          text: '"Before annealing" is the load-bearing detail. It means the positional behaviour was '
            + 'changed while the model still had training steps and learning rate left to adapt to '
            + 'it. DroPE should not be presented as a switch that turns an 8K model into a 256K '
            + 'model at inference time.',
        }),
        el('div', { class: 'gridrow' },
          el('div', { style: { flex: '1 1 300px', minWidth: '280px' } },
            el('p', {
              class: 'stage-title',
              style: { color: '#5fd2a8' },
              text: '✓ what the record establishes',
            }),
            el('ul', { class: 'plain', style: { fontSize: '13.5px' } },
              el('li', { text: 'V4 trained at 8K.' }),
              el('li', { text: 'V4 reportedly reached 256K.' }),
              el('li', { text: 'A step called DroPE was used before annealing.' }),
              el('li', { text: 'The extension factor was therefore 32×.' }))),
          el('div', { style: { flex: '1 1 300px', minWidth: '280px' } },
            el('p', {
              class: 'stage-title',
              style: { color: '#ef8a8a' },
              text: '✗ what it does not establish',
            }),
            el('ul', { class: 'plain', style: { fontSize: '13.5px' } },
              el('li', { text: 'The exact DroPE algorithm.' }),
              el('li', { text: 'Which rotary dimensions it changes.' }),
              el('li', { text: 'That 32× transfers to any other model or run.' }),
              el('li', { text: 'That a larger factor — 320×, 3,200× — would work.' }))),
        ),
        el('p', {
          class: 'note warn',
          html: 'This widget deliberately does not simulate a DroPE mechanism. Anything more '
            + 'detailed than the four bullets on the left is a <b>hypothesis until checked against '
            + 'the reference implementation</b>. The arithmetic in steps 1 and 2 follows from RoPE '
            + 'itself and stands on its own; the 8K → 256K figures are quoted from the V4 record.',
        }),
        el('pre', {
          class: 'ascii',
          text: '  A position function can exist beyond training length.\n'
            + '  Model capability at that length must still be earned — and demonstrated.\n\n'
            + `  current slider: ${tokens(target)}  = ${fx(target / TRAINED, target / TRAINED % 1 ? 1 : 0)}× the training length\n`
            + (target / TRAINED > 32
              ? '  → beyond the 32× that V4 reported. No evidence here either way.'
              : target / TRAINED === 32
                ? '  → exactly the factor V4 reported.'
                : '  → within the range V4 reported.'),
        }),
      );
    }

    /** Turns completed per pair, trained window against target, on a log scale. */
    function turnBars() {
      const W = 680, H = 30 + PAIRS * 34, L = 88, R = 92;
      const iw = W - L - R;
      const maxT = Math.max(...THETA.map((th) => turns(TARGETS[TARGETS.length - 1], th)));
      const px = (t) => (Math.log10(Math.max(0.01, t)) - Math.log10(0.01))
        / (Math.log10(maxT) - Math.log10(0.01)) * iw;

      const g = svg('svg', {
        viewBox: `0 0 ${W} ${H}`, width: '100%',
        style: 'display:block;max-width:100%;height:auto;margin:6px 0 4px', role: 'img',
        'aria-label': 'Turns completed by each rotary pair, training window versus target context',
      });
      THETA.forEach((th, p) => {
        const y = 8 + p * 34;
        const wTrained = px(turns(TRAINED, th));
        const wTarget = px(turns(target, th));
        g.append(
          svg('text', {
            x: L - 10, y: y + 15, 'text-anchor': 'end', 'font-size': 11,
            fill: p === PAIRS - 1 ? '#ef8a8a' : '#b8b7b0', text: `pair ${p}`,
          }),
          svg('rect', { x: L, y, width: iw, height: 20, rx: 4, fill: 'rgba(255,255,255,.04)' }),
          svg('rect', {
            x: L, y, width: Math.max(1, wTarget), height: 20, rx: 4,
            fill: p === PAIRS - 1 ? 'rgba(224,82,82,.45)' : 'rgba(235,104,52,.30)',
          }),
          svg('rect', {
            x: L, y, width: Math.max(1, wTrained), height: 20, rx: 4, fill: '#3987e5',
          }),
          svg('text', {
            x: L + iw + 8, y: y + 15, 'font-size': 10.5, fill: '#89877f',
            text: `${fx(turns(TRAINED, th), 1)} → ${fx(turns(target, th), 1)}`,
          }),
        );
      });
      g.append(svg('text', {
        x: L, y: H - 3, 'font-size': 10.5, fill: '#89877f',
        text: 'blue = turns across the widest TRAINED gap · red = across the target gap · log scale',
      }));
      return g;
    }

    render();
  },
);
