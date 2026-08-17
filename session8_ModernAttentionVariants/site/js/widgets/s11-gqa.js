/* §11 — Grouped-query attention, and why it is the baseline rather than the answer.

   Every layout is worked all the way through the formula, in substitution form:

       KV cache = 2 × L × H_kv × D × T × B × bytes

   Two presets. The "typical" one (32 layers, 32 query heads, KV 32/8/1) gives the
   memorable 4 GB → 1 GB at 8K and 137 GB → 34 GB at 256K. The "lesson" one keeps
   §10's yardstick geometry so the two sections line up.

   Byte counts are decimal (GB = 1e9), as the lesson quotes them; the binary value
   is printed alongside because 4 GiB / 1 GiB / 128 GiB / 32 GiB are the figures
   people usually remember. */

import {
  register, el, segmented, slider, readout, svg, bytes, commas, tokens, fx, clear,
} from '../lib.js';

const HEAD_DIM = 128;
const BPN = 2;                       // bf16
const BATCH = 1;

const PRESETS = {
  typical: { label: '32 layers · 32 query heads', layers: 32, q: 32, kvs: [32, 8, 1] },
  lesson: { label: '48 layers · 8 query heads', layers: 48, q: 8, kvs: [8, 2, 1] },
};
const NAMES = ['MHA', 'GQA', 'MQA'];
const COLOUR = ['#eb6834', '#3987e5', '#1baf7a'];

const T_VALUES = [8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576];

const cache = (layers, kv, T) => 2 * layers * kv * HEAD_DIM * T * BATCH * BPN;
const gib = (b) => `${(b / 2 ** 30).toFixed(b / 2 ** 30 < 10 ? 2 : 0)} GiB`;

register(
  'w-s11-gqa',
  'Fewer stored heads, same rising line',
  '§11 · MHA vs GQA vs MQA',
  'Switch the head layout and the whole formula is re-substituted and worked through, factor '
  + 'by factor. Sharing K/V heads divides the cache by a constant — but every layout still has '
  + 'a T in it, so every line still climbs.',
  (root) => {
    let preset = 'typical';
    let mode = 1;              // 0 = MHA, 1 = GQA, 2 = MQA
    let T = 8192;

    const stage = el('div', { class: 'stage' });
    const ros = el('div', { class: 'readouts' });

    const presetSel = segmented({
      label: 'Model geometry',
      options: Object.entries(PRESETS).map(([k, p]) => ({ value: k, label: p.label })),
      value: preset,
      onChange: (v) => { preset = v; render(); },
    });
    const modeSel = segmented({
      label: 'Head layout',
      options: NAMES.map((n, i) => ({ value: i, label: n })),
      value: mode,
      onChange: (v) => { mode = v; render(); },
    });
    const tS = slider({
      label: 'Context length T', values: T_VALUES, value: T,
      fmt: (v) => `${tokens(v)} tokens`,
      onInput: (v) => { T = v; render(); },
    });

    root.append(
      el('div', { class: 'controls' }, presetSel.node, modeSel.node, tS.node),
      stage, ros,
    );

    function render() {
      clear(stage); clear(ros);
      const P = PRESETS[preset];
      const kv = P.kvs[mode];
      const mine = cache(P.layers, kv, T);
      const mha = cache(P.layers, P.kvs[0], T);

      stage.append(
        el('p', {
          class: 'stage-title',
          text: `${NAMES[mode]} — ${P.q} query heads, ${kv} K/V head${kv > 1 ? 's' : ''} stored`,
        }),
        el('p', {
          class: 'stage-sub',
          text: 'In ordinary multi-head attention every query head has its own key and value head, '
            + 'so every token adds many K/V vectors to the cache. GQA keeps the separate query heads '
            + '— they can still ask different questions — but stores fewer copies of the keys and '
            + 'values they search through. Only one factor of the formula changes.',
        }),
        sharingDiagram(P, kv),
        worked(P, kv, T),
        comparison(P, T),
        growthChart(P),
        el('pre', {
          class: 'ascii',
          text: `  ${NAMES[0]} cache:  ${P.kvs[0]} × T\n`
            + `  ${NAMES[1]} cache:  ${P.kvs[1]} × T\n`
            + `  ${NAMES[2]} cache:  ${P.kvs[2]} × T\n\n`
            + '  All three still have a T in them. Sharing divides the cache by a constant;\n'
            + '  it does not change the fact that it grows with context. Double the context\n'
            + '  and the GQA cache still doubles.\n\n'
            + `  ${NAMES[mode]} at 1M tokens, one user: ${bytes(cache(P.layers, kv, 1048576))}`
            + `  (${gib(cache(P.layers, kv, 1048576))})`,
        }),
        el('p', {
          class: 'note',
          html: '<b>MQA saves the most and shares the most.</b> More sharing can affect model '
            + 'quality — this widget compares memory scaling only. GQA is a strong practical '
            + 'baseline because K/V sharing buys a large saving at a useful quality cost. It is a '
            + 'baseline rather than the answer because the cache still grows linearly with context. '
            + '§12 attacks the other term.',
        }),
      );

      addReadouts([
        { label: 'Query heads', value: String(P.q), sub: 'unchanged in all three layouts' },
        {
          label: 'K/V heads stored', value: String(kv), tone: 'blue',
          sub: `${P.q / kv} quer${P.q / kv > 1 ? 'ies' : 'y'} share each one`,
        },
        {
          label: `Cache at ${tokens(T)}`, value: bytes(mine), tone: 'violet',
          sub: `${gib(mine)} · one sequence`,
        },
        {
          label: 'Saving vs MHA', value: mode === 0 ? '—' : `${fx(mha / mine, 0)}×`,
          tone: mode === 0 ? '' : 'aqua',
          sub: mode === 0 ? 'this is the baseline' : `${bytes(mha - mine)} less`,
        },
        {
          label: 'Same model at 256K', value: bytes(cache(P.layers, kv, 262144)), tone: 'orange',
          sub: `${gib(cache(P.layers, kv, 262144))} · 32× the 8K figure`,
        },
      ]);
    }

    function addReadouts(list) { list.forEach((r) => ros.append(readout(r).node)); }

    /* ---- the formula, substituted and worked step by step ---- */
    function worked(P, kv, T) {
      const perHead = kv * HEAD_DIM;
      const perTokLayer = 2 * perHead;
      const perLayer = perTokLayer * T;
      const allLayers = perLayer * P.layers;
      const total = allLayers * BPN;
      const pad = (n) => commas(n).padStart(17);

      return el('pre', {
        class: 'ascii',
        text: '  KV cache = 2 × L × H_kv × D × T × B × bytes\n\n'
          + `  substitute:\n`
          + `           = 2 × ${P.layers} × ${kv} × ${HEAD_DIM} × ${T} × ${BATCH} × ${BPN}\n\n`
          + '  what each factor is:\n'
          + `     2${' '.repeat(8)}K + V — one key and one value per token\n`
          + `     ${String(P.layers).padEnd(9)}transformer layers, each with its own cache\n`
          + `     ${String(kv).padEnd(9)}K/V heads per layer${mode === 0 ? '  (= query heads: no sharing)' : `  ← ${NAMES[mode]}: ${P.q} query heads share these`}\n`
          + `     ${String(HEAD_DIM).padEnd(9)}numbers inside each head\n`
          + `     ${String(T).padEnd(9)}tokens in this conversation\n`
          + `     ${String(BATCH).padEnd(9)}one sequence\n`
          + `     ${String(BPN).padEnd(9)}bf16 = 2 bytes per number\n\n`
          + '  step by step:\n'
          + `     H_kv × D${' '.repeat(9)}= ${String(kv).padStart(3)} × ${HEAD_DIM}`
          + `  = ${pad(perHead)}  numbers in one K (or one V), per layer per token\n`
          + `     × 2 for K and V${' '.repeat(2)}= ${pad(perTokLayer)}  numbers per token, per layer\n`
          + `     × ${String(T).padEnd(7)} tokens = ${pad(perLayer)}  numbers per layer\n`
          + `     × ${String(P.layers).padEnd(7)} layers = ${pad(allLayers)}  numbers, whole model\n`
          + `     × ${String(BPN).padEnd(7)} bytes  = ${pad(total)}  BYTES\n\n`
          + `                        ≈ ${bytes(total)}   (${gib(total)})`,
      });
    }

    /* ---- all three layouts, and the 8K → 256K step ---- */
    function comparison(P, T) {
      const at256 = (kv) => cache(P.layers, kv, 262144);
      return el('pre', {
        class: 'ascii',
        text: '  the same model, three head layouts:\n\n'
          + `  layout   H_kv   cache at ${tokens(T)}${' '.repeat(6)}cache at 256K${' '.repeat(9)}vs MHA\n`
          + `  ${'─'.repeat(76)}\n`
          + P.kvs.map((kv, i) => {
            const c = cache(P.layers, kv, T);
            const c256 = at256(kv);
            return `  ${NAMES[i].padEnd(8)} ${String(kv).padStart(3)}`
              + `   ${bytes(c).padStart(9)} (${gib(c).padStart(8)})`
              + `   ${bytes(c256).padStart(9)} (${gib(c256).padStart(8)})`
              + `   ${i === 0 ? 'baseline' : `${fx(cache(P.layers, P.kvs[0], T) / c, 0)}× smaller`}`
              + (i === mode ? '  ←' : '');
          }).join('\n')
          + '\n\n'
          + `  8K → 256K is 32× more tokens, and T is a plain multiplier, so every\n`
          + `  number in the 256K column is exactly 32× its 8K counterpart.\n`
          + `  Nothing about the head layout changes that — it only sets the starting point.`,
      });
    }

    /* ---- head sharing, drawn ---- */
    function sharingDiagram(P, kv) {
      const per = P.q / kv;
      const rowH = P.q > 12 ? 13 : 26;
      const W = 620, H = P.q * rowH + 34;
      const g = svg('svg', {
        viewBox: `0 0 ${W} ${H}`, width: '100%',
        style: 'display:block;max-width:100%;height:auto;margin:2px 0 18px', role: 'img',
        'aria-label': `${P.q} query heads mapped onto ${kv} stored key-value heads`,
      });
      g.append(
        svg('text', {
          x: 0, y: 12, 'font-size': 11, fill: '#89877f', text: `${P.q} query heads`,
        }),
        svg('text', {
          x: W - 1, y: 12, 'text-anchor': 'end', 'font-size': 11, fill: '#89877f',
          text: `${kv} K/V head${kv > 1 ? 's' : ''} in the cache`,
        }),
      );
      const bh = Math.max(8, rowH - 5);
      for (let q = 0; q < P.q; q++) {
        const y = 22 + q * rowH;
        const grp = Math.floor(q / per);
        const gy = 22 + (grp * per + (per - 1) / 2) * rowH;
        g.append(
          svg('rect', {
            x: 0, y, width: 84, height: bh, rx: 4,
            fill: 'rgba(144,133,233,.16)', stroke: 'rgba(144,133,233,.35)',
          }),
          P.q <= 12 ? svg('text', {
            x: 42, y: y + bh / 2 + 4, 'text-anchor': 'middle', 'font-size': 11,
            fill: '#b5aef2', text: `Q${q + 1}`,
          }) : null,
          svg('path', {
            d: `M 88 ${y + bh / 2} C 180 ${y + bh / 2}, 270 ${gy + bh / 2}, 356 ${gy + bh / 2}`,
            fill: 'none', stroke: COLOUR[mode], 'stroke-width': 1.1, opacity: 0.42,
          }),
        );
      }
      for (let h = 0; h < kv; h++) {
        const gy = 22 + (h * per + (per - 1) / 2) * rowH;
        const bhh = Math.max(bh, 16);
        g.append(
          svg('rect', {
            x: 358, y: gy + bh / 2 - bhh / 2, width: 148, height: bhh, rx: 4,
            fill: `${COLOUR[mode]}33`, stroke: COLOUR[mode],
          }),
          svg('text', {
            x: 432, y: gy + bh / 2 + 4, 'text-anchor': 'middle', 'font-size': 10.5,
            fill: COLOUR[mode], text: kv <= 8 ? `K/V head ${String.fromCharCode(65 + h)}` : `K/V ${h + 1}`,
          }),
          kv <= 8 ? svg('text', {
            x: 514, y: gy + bh / 2 + 4, 'font-size': 10.5, fill: '#89877f',
            text: `${per} quer${per > 1 ? 'ies' : 'y'}`,
          }) : null,
        );
      }
      return g;
    }

    /* ---- linear axes on purpose: three slopes, all rising ---- */
    function growthChart(P) {
      const W = 700, H = 290, L = 74, R = 86, TOP = 16, B = 42;
      const iw = W - L - R, ih = H - TOP - B;
      const maxT = T_VALUES[T_VALUES.length - 1];
      const maxB = cache(P.layers, P.kvs[0], maxT);
      const px = (t) => L + (t / maxT) * iw;
      const py = (b) => TOP + ih - (b / maxB) * ih;

      const g = svg('svg', {
        viewBox: `0 0 ${W} ${H}`, width: '100%',
        style: 'display:block;max-width:100%;height:auto;margin:4px 0', role: 'img',
        'aria-label': 'Cache size against context length for all three head layouts, linear axes',
      });
      for (let f = 0; f <= 4; f++) {
        const y = TOP + ih - (f / 4) * ih;
        g.append(
          svg('line', {
            x1: L, x2: L + iw, y1: y, y2: y, stroke: 'rgba(255,255,255,.07)', 'stroke-width': 1,
          }),
          svg('text', {
            x: L - 10, y: y + 4, 'text-anchor': 'end', 'font-size': 10.5, fill: '#89877f',
            text: bytes((maxB * f) / 4, 0),
          }),
        );
      }
      [8192, 262144, 524288, 1048576].forEach((t) => g.append(svg('text', {
        x: px(t), y: H - B + 20, 'text-anchor': 'middle', 'font-size': 10.5,
        fill: '#89877f', text: tokens(t),
      })));
      g.append(svg('text', {
        x: L + iw / 2, y: H - 5, 'text-anchor': 'middle', 'font-size': 11, fill: '#89877f',
        text: 'context length T  ·  one sequence  ·  linear axes',
      }));

      P.kvs.forEach((kv, i) => {
        const sel = i === mode;
        g.append(
          svg('polyline', {
            points: T_VALUES.map((t) => `${fx(px(t), 1)},${fx(py(cache(P.layers, kv, t)), 1)}`).join(' '),
            fill: 'none', stroke: COLOUR[i], 'stroke-width': sel ? 2.8 : 1.6,
            opacity: sel ? 1 : 0.45, 'stroke-linejoin': 'round',
          }),
          svg('text', {
            x: L + iw + 8, y: py(cache(P.layers, kv, maxT)) + 4, 'font-size': 11,
            fill: COLOUR[i], 'font-weight': sel ? 700 : 500,
            text: `${NAMES[i]} · ${kv}`,
          }),
        );
      });
      g.append(
        svg('line', {
          x1: px(T), x2: px(T), y1: TOP, y2: TOP + ih,
          stroke: 'rgba(255,255,255,.26)', 'stroke-width': 1, 'stroke-dasharray': '3 3',
        }),
        svg('circle', {
          cx: px(T), cy: py(cache(P.layers, P.kvs[mode], T)), r: 5, fill: COLOUR[mode],
        }),
      );
      return g;
    }

    render();
  },
);
