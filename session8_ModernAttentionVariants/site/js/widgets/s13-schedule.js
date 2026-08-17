/* §13 — Schedules across depth.
   D = DeltaNet fixed-state layer, G = sparse-attention layer. A deep model does
   not have to pick one memory system for the whole network.

   The compute model is calibrated to the reported V4 figure rather than invented.
   The record says going from 1 G per 8 layers to 8 per 8 makes KV state 8.0×
   larger while estimated mixing compute rises only ~1.41×. Solving

       8r / (7 + r) = 1.41   →   r ≈ 1.5

   gives the cost of one G layer relative to one D layer, and that single constant
   reproduces the quoted numbers. It is a back-derived estimate, labelled as such. */

import {
  register, el, segmented, slider, readout, svg,
  bytes, commas, tokens, fx, clear,
} from '../lib.js';

const MOTIF = 8;                 // layers per repeating motif
const REPEATS = 4;               // 32-layer model
const LAYERS = MOTIF * REPEATS;

const KV_HEADS = 8;
const HEAD_DIM = 128;
const BPN = 2;
const G_COST = 1.5;             // back-derived from the reported 1.41× (see header)

/* Bytes a single G layer adds per token, and the fixed state a single D layer holds. */
const G_BYTES_PER_TOKEN = 2 * KV_HEADS * HEAD_DIM * BPN;
const D_STATE_BYTES = KV_HEADS * HEAD_DIM * HEAD_DIM * BPN;

const SCHEDULES = [
  { id: 'DDDDDDDD', label: 'all D', motif: 'DDDDDDDD', note: 'illustration' },
  { id: 'DDDDDDDG', label: '1 G per 8', motif: 'DDDDDDDG', note: 'illustration · the V4 baseline for the quoted ratios' },
  { id: 'DDDGDDDG', label: 'DDDGDDDG', motif: 'DDDGDDDG', note: 'reported V4 motif' },
  { id: 'DGDGDGDG', label: '4 G per 8', motif: 'DGDGDGDG', note: 'illustration' },
  { id: 'GGGGGGGG', label: 'all G', motif: 'GGGGGGGG', note: 'illustration' },
];

const T_VALUES = [8192, 32768, 131072, 262144, 1048576];

register(
  'w-s13-schedule',
  'Not every layer needs the same memory',
  '§13 · depth schedules · 32 layers',
  'Each D block compresses history into a fixed-size state; each G block gives the model '
  + 'another chance to read selected earlier tokens directly. Switch schedules to see the trade '
  + 'between frequent exact access and the KV cache the attention layers carry.',
  (root) => {
    let sched = 'DDDGDDDG';
    let T = 262144;

    const stage = el('div', { class: 'stage' });
    const ros = el('div', { class: 'readouts' });

    const sSel = segmented({
      label: 'Depth schedule',
      options: SCHEDULES.map((s) => ({ value: s.id, label: s.label, title: s.motif })),
      value: sched,
      onChange: (v) => { sched = v; render(); },
    });
    const tS = slider({
      label: 'Context length T', values: T_VALUES, value: T,
      fmt: (v) => `${tokens(v)} tokens`,
      onInput: (v) => { T = v; render(); },
    });

    root.append(el('div', { class: 'controls' }, sSel.node, tS.node), stage, ros);

    const stats = (motif, ctx) => {
      const g = [...motif].filter((c) => c === 'G').length;
      const d = MOTIF - g;
      const gTotal = g * REPEATS;
      const dTotal = d * REPEATS;
      return {
        g, d, gTotal, dTotal,
        kv: gTotal * G_BYTES_PER_TOKEN * ctx,
        state: dTotal * D_STATE_BYTES,
        compute: (d * 1 + g * G_COST) / (7 * 1 + 1 * G_COST),   // vs the 1 G per 8 baseline
      };
    };

    function render() {
      clear(stage); clear(ros);
      const S = SCHEDULES.find((x) => x.id === sched);
      const st = stats(S.motif, T);
      const base = stats('DDDDDDDG', T);
      const isV4 = S.id === 'DDDGDDDG';

      stage.append(
        el('p', {
          class: 'stage-title',
          text: `${S.motif}  ·  ${st.d} fixed-state and ${st.g} sparse-attention layer`
            + `${st.g === 1 ? '' : 's'} per 8`,
        }),
        el('p', {
          class: 'stage-sub',
          text: 'A fixed-state layer compresses the past into a small running state, so it is cheap '
            + 'to serve — its memory does not grow with the sequence. The tradeoff is that the old '
            + 'sequence is no longer available token by token. A sparse-attention layer keeps keys '
            + 'and values and lets a query read selected earlier tokens directly, restoring exact '
            + 'access — but every such layer adds a KV cache that grows with context.',
        }),
        stack(S.motif),
        el('pre', {
          class: 'ascii',
          text: `  ${S.motif} repeated ${REPEATS}× = ${LAYERS} layers\n\n`
            + `  ${String(st.dTotal).padStart(2)} D layers  ·  fixed state, ${bytes(D_STATE_BYTES)} each`
            + `  →  ${bytes(st.state)} total, CONSTANT in T\n`
            + `  ${String(st.gTotal).padStart(2)} G layers  ·  KV cache, grows with T`
            + `      →  ${bytes(st.kv)} at ${tokens(T)}\n\n`
            + `  serving memory for one sequence: ${bytes(st.state + st.kv)}\n`
            + (st.gTotal === 0
              ? '  no KV cache at all — nothing in this schedule grows with context.'
              : `  of which ${fx((st.kv / (st.kv + st.state)) * 100, 1)}% is the KV cache.`),
        }),
        comparison(T),
        el('pre', {
          class: 'ascii',
          text: '  Why not use G everywhere?\n'
            + '    More G layers mean more frequent chances to read exact earlier tokens — but\n'
            + '    more per-sequence KV cache. Against the 1-G-per-8 baseline, going to 8 per 8\n'
            + `    makes the KV state ${fx(stats('GGGGGGGG', T).kv / base.kv, 1)}× larger`
            + ` while estimated mixing compute rises only ${fx(stats('GGGGGGGG', T).compute, 2)}×.\n\n`
            + '    For this configuration the price of more G layers is SERVING MEMORY,\n'
            + '    not FLOPs. That is an unusual shape of tradeoff and worth noticing.\n\n'
            + '  Why not use D everywhere?\n'
            + '    A fixed recurrent state is cheap, but it is a compressed summary. If every\n'
            + '    layer uses only that state, the model never gets another direct look at the\n'
            + '    exact old token representations — §5\'s interference, at every depth.',
        }),
        el('p', {
          class: isV4 ? 'note' : 'note warn',
          html: isV4
            ? '<b>DDDGDDDG is reported evidence.</b> V4 kept this motif from the 1.78B seed model all '
              + 'the way to the 120B run, which is meaningful evidence that the mixture works across '
              + 'scale. It is <b>not</b> evidence that six D and two G is the best possible ratio — '
              + 'the neighbouring schedules were never cleanly ablated. A successful design choice, '
              + 'not a proven optimum.'
            : '<b>This schedule is an illustration, not an experiment anyone ran.</b> Only DDDGDDDG '
              + 'is in the V4 record. The numbers here follow from the formulas in §10–§12 and the '
              + 'compute constant back-derived from the reported ratios; they are not measurements '
              + 'of a trained model.',
        }),
        el('p', {
          class: 'stage-sub',
          text: 'The broader lesson matters more than the exact motif: not every layer needs the same '
            + 'memory system, and the schedule itself is part of the architecture.',
        }),
      );

      addReadouts([
        { label: 'G layers per 8', value: String(st.g), tone: 'orange', sub: `${st.gTotal} of ${LAYERS} total` },
        {
          label: `KV cache at ${tokens(T)}`, value: bytes(st.kv), tone: 'red',
          sub: st.gTotal ? 'grows with context' : 'none',
        },
        {
          label: 'Fixed state', value: bytes(st.state), tone: 'aqua',
          sub: 'same at any context length',
        },
        {
          label: 'KV vs 1 G per 8', value: base.kv ? `${fx(st.kv / base.kv, 1)}×` : '—',
          tone: 'red', sub: 'the memory price',
        },
        {
          label: 'Mixing compute', value: `${fx(st.compute, 2)}×`, tone: 'blue',
          sub: 'estimated, vs 1 G per 8',
        },
      ]);
    }

    function addReadouts(list) { list.forEach((r) => ros.append(readout(r).node)); }

    /* ---- the motif drawn as a stack of layers ---- */
    function stack(motif) {
      const cw = 62, ch = 30, gap = 6;
      const W = MOTIF * (cw + gap);
      const g = svg('svg', {
        viewBox: `0 0 ${W} 96`, width: '100%',
        style: 'display:block;max-width:100%;height:auto;margin:4px 0 16px', role: 'img',
        'aria-label': `Depth schedule ${motif}`,
      });
      g.append(svg('text', {
        x: 0, y: 11, 'font-size': 10.5, fill: '#89877f',
        text: 'one 8-layer motif, input on the left',
      }));
      [...motif].forEach((c, i) => {
        const isG = c === 'G';
        const x = i * (cw + gap);
        g.append(
          svg('rect', {
            x, y: 20, width: cw, height: ch, rx: 6,
            fill: isG ? 'rgba(235,104,52,.26)' : 'rgba(27,175,122,.20)',
            stroke: isG ? '#eb6834' : '#1baf7a', 'stroke-width': 1.4,
          }),
          svg('text', {
            x: x + cw / 2, y: 40, 'text-anchor': 'middle', 'font-size': 14,
            'font-weight': 700, fill: isG ? '#f5a07c' : '#5fd2a8', text: c,
          }),
          svg('text', {
            x: x + cw / 2, y: 64, 'text-anchor': 'middle', 'font-size': 9.5,
            fill: '#89877f', text: isG ? 'KV cache' : 'fixed state',
          }),
          svg('text', {
            x: x + cw / 2, y: 76, 'text-anchor': 'middle', 'font-size': 9.5,
            fill: isG ? '#f5a07c' : '#5fd2a8', text: isG ? 'exact reads' : 'compressed',
          }),
          i < MOTIF - 1 ? svg('text', {
            x: x + cw + gap / 2, y: 40, 'text-anchor': 'middle', 'font-size': 11,
            fill: '#4a4a52', text: '›',
          }) : null,
        );
      });
      return g;
    }

    /* ---- every schedule side by side ---- */
    function comparison(ctx) {
      const base = stats('DDDDDDDG', ctx);
      return el('pre', {
        class: 'ascii',
        text: `  all five schedules at T = ${tokens(ctx)}, one sequence:\n\n`
          + `  motif       G/8   KV cache     fixed state   KV vs 1G/8   compute   source\n`
          + `  ${'─'.repeat(80)}\n`
          + SCHEDULES.map((s) => {
            const st = stats(s.motif, ctx);
            return `  ${s.motif}  ${String(st.g).padStart(3)}`
              + `   ${bytes(st.kv).padStart(9)}    ${bytes(st.state).padStart(9)}`
              + `   ${(base.kv ? `${fx(st.kv / base.kv, 1)}×` : '—').padStart(9)}`
              + `   ${(`${fx(st.compute, 2)}×`).padStart(7)}`
              + `   ${s.id === 'DDDGDDDG' ? 'V4 reported' : 'illustration'}`
              + (s.id === sched ? '  ←' : '');
          }).join('\n')
          + `\n\n  The KV column spans a factor of ${fx(stats('GGGGGGGG', ctx).kv / base.kv, 0)};`
          + ` the compute column spans ${fx(stats('GGGGGGGG', ctx).compute / stats('DDDDDDDD', ctx).compute, 2)}.\n`
          + '  That gap is the whole reason a schedule is worth choosing carefully.',
      });
    }

    render();
  },
);
