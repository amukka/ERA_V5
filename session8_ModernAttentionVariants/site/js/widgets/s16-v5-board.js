/* §16 — What V5 has to decide.
   A candidate builder. Every reported figure it uses comes from a formula already
   established in this session, and the output is a specification block — which is
   what an architecture proposal should eventually become.

   The board separates three kinds of statement, and labels them on screen:
     settled   — the current evidence already rules this in or out
     evidence  — V4 did this and it worked; that is not proof it is optimal
     open      — needs a measurement nobody has made yet */

import {
  register, el, segmented, slider, toggle, readout, stepper,
  bytes, tokens, fx, clear,
} from '../lib.js';
import {
  ATTN, EXTENSION, POSITION, DEFAULT_CONFIG, layerSplit,
  cacheBytes, stateBytes, servingBytes, mixingRelativeToFull, reach, trainingCostRel,
} from '../costs.js';

const T_VALUES = [32768, 131072, 262144, 524288, 1048576];

const SETTLED = [
  ['A stored absolute position table is out.',
    'Session 7 showed the hard length wall it creates — there is no row beyond the training length.'],
  ['GQA alone is not enough for the target.',
    'It reduces KV-cache growth by a constant factor but does not remove linear growth with context (§11).'],
  ['Some stronger long-context mechanism is required.',
    'Across the three reference architectures, every one uses at least one of linear state, sparsity or sequence compression. None uses none.'],
];

const OPEN = [
  ['Extend or build native',
    'V4 stretched 32× with DroPE. DeepSeek built for a million natively. Extension is much cheaper but has a ceiling.',
    'An extension-factor study at V5\'s target, plus a real compute and memory estimate for training natively at that length.'],
  ['The schedule and its ratio',
    'DDDGDDDG worked from 1.78B to 120B, but the ratio was never varied systematically.',
    'A schedule ablation at a scale large enough that the result is likely to transfer.'],
  ['Linear state vs sequence compression',
    'V4 and Qwen3.6 chose the first family; DeepSeek chose the second. Both work.',
    'A matched-budget comparison at genuinely long context, not a short-context benchmark.'],
  ['The sparsity budget',
    'V4\'s cap of 256 came partly from backward-kernel contention on the previous hardware/software stack (§7).',
    'Re-measurement on the current kernels and current hardware rather than inheritance.'],
  ['Whether the Memory Stream still earns its place',
    'Once the ordinary context window is already very long, the cross-chunk carry may be redundant.',
    'An ablation at the actual target context length.'],
  ['Every head count and head dimension',
    'Those numbers were derived around a width of 4,096, and V5 has not committed to that width.',
    'Decide the model width first, then derive the attention geometry from it.'],
];

const STEPS = [
  { key: 'step 1', title: 'Build a candidate' },
  { key: 'step 2', title: 'What is already settled' },
  { key: 'step 3', title: 'What is still open' },
];

register(
  'w-s16-v5-board',
  'V5 attention decision board',
  '§16 · candidate specification',
  'Build a candidate from the choices in this session. The board uses the same formulas '
  + 'introduced earlier to report cache, compute, reach and training cost, and names any failed '
  + 'constraint numerically. The output is a specification — which is what an architecture '
  + 'proposal should eventually become.',
  (root) => {
    let step = 0;
    const c = { ...DEFAULT_CONFIG, targetCtx: 262144 };

    const stage = el('div', { class: 'stage' });
    const ros = el('div', { class: 'readouts' });

    const seq = stepper({ steps: STEPS, value: 0, onChange: (s) => { step = s; render(); } });

    const controls = el('div', { class: 'controls' });
    const build = [
      slider({
        label: 'Target context', values: T_VALUES, value: c.targetCtx,
        fmt: (v) => tokens(v), onInput: (v) => { c.targetCtx = v; render(); },
      }),
      slider({
        label: 'Trained context', values: [4096, 8192, 32768, 131072, 1048576], value: c.trainedCtx,
        fmt: (v) => tokens(v), onInput: (v) => { c.trainedCtx = v; render(); },
      }),
      slider({
        label: 'Layers', min: 16, max: 96, step: 8, value: c.layers,
        fmt: (v) => String(v), onInput: (v) => { c.layers = v; render(); },
      }),
      slider({
        label: 'Sparse layers per 8', min: 0, max: 8, step: 1, value: c.gPer8,
        fmt: (v) => `${8 - v} D / ${v} G`, onInput: (v) => { c.gPer8 = v; render(); },
      }),
      segmented({
        label: 'Position', options: Object.entries(POSITION).map(([k, p]) => ({ value: k, label: p.label })),
        value: c.position, onChange: (v) => { c.position = v; render(); },
      }),
      segmented({
        label: 'Extension policy',
        options: Object.entries(EXTENSION).map(([k, e]) => ({ value: k, label: k === 'none' ? 'native' : e.label })),
        value: c.extension, onChange: (v) => { c.extension = v; render(); },
      }),
      segmented({
        label: 'Attention family', options: Object.entries(ATTN).map(([k, a]) => ({ value: k, label: a.label })),
        value: c.attn, onChange: (v) => { c.attn = v; render(); },
      }),
      segmented({
        label: 'K/V heads', options: [{ value: 32, label: '32' }, { value: 8, label: '8' }, { value: 1, label: '1' }],
        value: c.kvHeads, onChange: (v) => { c.kvHeads = v; render(); },
      }),
      segmented({
        label: 'Cache precision',
        options: [{ value: 4, label: 'fp32' }, { value: 2, label: 'bf16' }, { value: 1, label: 'fp8' }],
        value: c.bpn, onChange: (v) => { c.bpn = v; render(); },
      }),
      slider({
        label: 'Block size m', values: [1, 4, 8, 16, 32], value: c.blockSize,
        fmt: (v) => (v === 1 ? 'none' : `${v}`), onInput: (v) => { c.blockSize = v; render(); },
      }),
      slider({
        label: 'Read budget k', values: [64, 128, 256, 512, 1024], value: c.topK,
        fmt: (v) => String(v), onInput: (v) => { c.topK = v; render(); },
      }),
      slider({
        label: 'Memory budget', min: 40, max: 640, step: 40, value: c.memBudgetGB,
        fmt: (v) => `${v} GB`, onInput: (v) => { c.memBudgetGB = v; render(); },
      }),
    ];
    const streamSw = toggle({
      label: 'Cross-chunk stream', value: c.stream, onChange: (v) => { c.stream = v; render(); },
    });
    controls.append(...build.map((b) => b.node), el('div', { class: 'ctl' }, streamSw.node));

    root.append(seq.node, controls, stage, ros);

    function constraints() {
      const T = c.targetCtx;
      const r = reach(c);
      const mem = servingBytes(c, T) * c.batch;
      const budget = c.memBudgetGB * 1e9;
      const rel = mixingRelativeToFull(c, T);
      const out = [];
      if (r.limit < T) {
        out.push(`positional reach is ${tokens(r.limit)}, short of the ${tokens(T)} target by `
          + `${fx(T / r.limit, 1)}× — ${r.why}`);
      }
      if (mem > budget) {
        out.push(`serving memory is ${bytes(mem)} for ${c.batch} users, over the `
          + `${c.memBudgetGB} GB budget by ${bytes(mem - budget)}`);
      }
      if (rel > 0.25) {
        out.push(`mixing compute is ${fx(rel * 100, 0)}% of full quadratic attention — the `
          + 'sparsity or state is not doing enough work');
      }
      if (c.chunked && !c.stream) {
        out.push('documents are chunked but nothing crosses the boundary — chunk 1 is discarded '
          + 'entirely');
      }
      if (c.attn === 'compressed' && c.blockSize === 1) {
        out.push('compressed attention selected but block size is 1, so nothing is actually '
          + 'compressed');
      }
      if (c.gPer8 === 0) {
        out.push('no sparse-attention layers at all — this is a purely linear/DeltaNet network, so '
          + `the ${ATTN[c.attn].label} choice has no layer to run in and every query reads only the `
          + 'compressed state (§5 interference at every depth)');
      }
      return out;
    }

    function render() {
      clear(stage); clear(ros);
      controls.style.display = step === 0 ? 'flex' : 'none';
      if (step === 0) stepBuild();
      else if (step === 1) stepSettled();
      else stepOpen();

      const T = c.targetCtx;
      const mem = servingBytes(c, T) * c.batch;
      const bad = constraints();
      addReadouts([
        {
          label: 'Constraints failed', value: String(bad.length),
          tone: bad.length ? 'red' : 'aqua',
          sub: bad.length ? 'named on the board' : 'candidate is internally consistent',
        },
        {
          label: `Serving memory · ${c.batch} users`, value: bytes(mem),
          tone: mem <= c.memBudgetGB * 1e9 ? 'aqua' : 'red',
          sub: `budget ${c.memBudgetGB} GB`,
        },
        {
          label: 'Mixing compute', value: `${fx(mixingRelativeToFull(c, T) * 100, 1)}%`,
          tone: 'blue', sub: 'of full attention at target',
        },
        {
          label: 'Positional reach', value: tokens(reach(c).limit),
          tone: reach(c).limit >= T ? 'aqua' : 'red',
          sub: `${fx(reach(c).limit / c.trainedCtx, 0)}× the trained length`,
        },
        {
          label: 'Training cost', value: `${fx(trainingCostRel(c), 1)}×`,
          tone: 'violet', sub: 'rough, vs 8K-context training',
        },
      ]);
    }

    function addReadouts(list) { list.forEach((r) => ros.append(readout(r).node)); }

    /* ---- step 1: the candidate spec ---- */
    function stepBuild() {
      const T = c.targetCtx;
      const { g, d } = layerSplit(c);
      const bad = constraints();
      const motif = Array.from({ length: 8 }, (_, i) =>
        (c.gPer8 === 0 ? 'D' : (i + 1) % Math.max(1, Math.round(8 / c.gPer8)) === 0 ? 'G' : 'D')).join('');

      stage.append(
        el('p', {
          class: 'stage-title',
          text: bad.length
            ? `Candidate has ${bad.length} failed constraint${bad.length > 1 ? 's' : ''}`
            : 'Candidate is internally consistent',
        }),
        el('p', {
          class: 'stage-sub',
          text: 'We should not end this session by pretending V5\'s attention architecture has been '
            + 'chosen. V4 is finished, and its configuration gives us evidence — it does not give us '
            + 'the V5 configuration. The frontier moved, the hardware changed, the kernels changed, '
            + 'and some V4 choices were never ablated.',
        }),
        el('pre', {
          class: 'ascii',
          text: '  CANDIDATE SPECIFICATION\n'
            + `  ${'═'.repeat(66)}\n`
            + `  target context        ${tokens(T)}\n`
            + `  trained context       ${tokens(c.trainedCtx)}\n`
            + `  position              ${POSITION[c.position].label}\n`
            + `  extension policy      ${EXTENSION[c.extension].label}\n`
            + `  layers                ${c.layers}  (${d} fixed-state, ${g} sparse-attention)\n`
            + `  depth schedule        ${motif} × ${c.layers / 8}\n`
            + `  attention family      ${ATTN[c.attn].label}\n`
            + `  K/V heads             ${c.kvHeads}\n`
            + `  head dim              ${c.headDim}\n`
            + `  cache precision       ${c.bpn === 4 ? 'fp32' : c.bpn === 2 ? 'bf16' : 'fp8'} (${c.bpn} B)\n`
            + `  block size m          ${c.blockSize === 1 ? 'no compression' : c.blockSize}`
            + `${c.attn === 'compressed' ? '' : '   (unused — this family stores every position)'}\n`
            + `  read budget k         ${c.topK}\n`
            + `  cross-chunk state     ${c.stream ? 'Memory Stream, O(1)' : 'none'}\n`
            + `  ${'─'.repeat(66)}\n`
            + `  KV cache / sequence   ${bytes(cacheBytes(c, T))}\n`
            + `  fixed state / seq     ${bytes(stateBytes(c))}\n`
            + `  serving, ${String(c.batch).padStart(2)} users     ${bytes(servingBytes(c, T) * c.batch)}`
            + `   (budget ${c.memBudgetGB} GB)\n`
            + `  mixing compute        ${fx(mixingRelativeToFull(c, T) * 100, 2)}% of full attention\n`
            + `  positional reach      ${tokens(reach(c).limit)}\n`
            + `  training cost         ${fx(trainingCostRel(c), 1)}× an 8K-context run (rough)\n`
            + `  ${'═'.repeat(66)}`,
        }),
        bad.length
          ? el('pre', {
            class: 'ascii',
            text: '  FAILED CONSTRAINTS\n\n'
              + bad.map((b, i) => `  ${i + 1}. ${b}`).join('\n\n'),
          })
          : el('pre', {
            class: 'ascii',
            text: '  No constraint fails at these settings.\n\n'
              + '  That means the candidate is arithmetically coherent — not that it would\n'
              + '  train well. Nothing on this board measures model quality.',
          }),
        el('p', {
          class: 'note warn',
          html: '<b>Every number above comes from a formula in this session, not from a measurement.</b> '
            + 'Cache is §10\'s bill with §11\'s head sharing, §12\'s compression and §13\'s schedule. '
            + 'Reach uses reported extension ceilings. Training cost is deliberately coarse. A board '
            + 'like this rules candidates out on arithmetic; it cannot rule one in.',
        }),
      );
    }

    /* ---- step 2 ---- */
    function stepSettled() {
      stage.append(
        el('p', { class: 'stage-title', text: 'What the current evidence already settles' }),
        el('p', {
          class: 'stage-sub',
          text: 'We are not searching the whole architecture space from zero. Some conclusions are '
            + 'already strong enough to carry forward.',
        }),
        el('pre', {
          class: 'ascii',
          text: SETTLED.map(([claim, why], i) => `  ${i + 1}. ${claim}\n     ${why}`).join('\n\n'),
        }),
        el('pre', {
          class: 'ascii',
          text: '  And we know the broad design pressures:\n\n'
            + '    position must extrapolate or be trained long\n'
            + '    cache cannot remain huge\n'
            + '    compute cannot remain quadratic everywhere\n'
            + '    exact memory is still useful somewhere\n'
            + '    fixed state is cheap but lossy\n'
            + '    compression is cheap but destroys detail unless repaired',
        }),
        el('p', {
          class: 'stage-sub',
          text: 'Getting to this shape is the part that took the field years. The remaining questions '
            + 'are narrow enough that well-designed proxy runs can actually settle them.',
        }),
      );
    }

    /* ---- step 3 ---- */
    function stepOpen() {
      stage.append(
        el('p', { class: 'stage-title', text: 'What must be re-tested for V5' }),
        el('p', {
          class: 'stage-sub',
          text: 'The useful question is not "what did V4 use?" It is: which V4 decisions are now '
            + 'evidence, which are merely defaults, and what must be re-measured? This is also where '
            + 'the cohort contributes — candidate techniques and ablations from these sessions are '
            + 'inputs into the V5 architecture review.',
        }),
        el('pre', {
          class: 'ascii',
          text: OPEN.map(([q, ctx, needs], i) =>
            `  ${i + 1}. ${q}\n     ${ctx}\n     NEEDS: ${needs}`).join('\n\n'),
        }),
        el('pre', {
          class: 'ascii',
          text: '  We know the shape of the decision. We do not yet know the winning\n'
            + '  configuration.',
        }),
        el('p', {
          class: 'note',
          html: '<b>Section 17 of the source material is missing.</b> The capture was truncated inside '
            + '"Road 2: build long, then train long", so the two-roads summary is not represented on '
            + 'this page. Road 1 — train short, then stretch — is V4\'s 8K → 256K with DroPE, and is '
            + 'the cheaper road with a practical ceiling. Road 2 designs the architecture so long '
            + 'sequences are affordable natively. Those are the two candidates step 3\'s first open '
            + 'question is asking you to choose between.',
        }),
      );
    }

    render();
  },
);
