/* §3 — The two bills attention sends.
   Deliberately introduces only the two growth patterns: compute ~ T², KV cache ~ T.
   The full byte-level cache formula (layers, kv_heads, head_dim, precision,
   concurrency) belongs to §10 and is not duplicated here. */

import {
  register, el, svg, segmented, slider, readout, compact, tokens, fx, clear,
} from '../lib.js';

const T_VALUES = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768,
  65536, 131072, 262144, 524288, 1048576];
const T_BASE = 128;   // reference point both curves are normalised against

register(
  'w-s03-bills',
  'Two bills, two growth patterns',
  '§3 · compute vs memory',
  'Move the context slider and watch two different curves. The number of query-key '
  + 'comparisons grows quadratically; the saved conversation history grows linearly. '
  + 'These are separate problems that arrive at separate moments — one during the '
  + 'attention calculation, one during generation.',
  (root) => {
    let T = 4096;
    let view = 'curves';

    const stage = el('div', { class: 'stage' });
    const ros = el('div', { class: 'readouts' });

    const viewSel = segmented({
      label: 'View',
      options: [
        { value: 'accumulate', label: 'How the cache accumulates' },
        { value: 'curves', label: 'The two growth curves' },
      ],
      value: view,
      onChange: (v) => { view = v; render(); },
    });

    const tSlider = slider({
      label: 'Context length T',
      values: T_VALUES,
      value: T,
      fmt: (v) => `${tokens(v)} tokens`,
      onInput: (v) => { T = v; render(); },
    });

    const dbl = el('button', {
      class: 'btn',
      onclick: () => {
        const i = T_VALUES.indexOf(T);
        if (i < T_VALUES.length - 1) { T = T_VALUES[i + 1]; tSlider.set(T); render(); }
      },
    }, 'Double the context →');

    root.append(
      el('div', { class: 'controls' }, viewSel.node, tSlider.node,
        el('div', { class: 'ctl' }, dbl)),
      stage,
      ros,
    );

    function render() {
      clear(stage); clear(ros);
      dbl.disabled = T === T_VALUES[T_VALUES.length - 1];

      const scores = T * T;
      const kvPairs = T;
      const computeX = scores / (T_BASE * T_BASE);
      const cacheX = kvPairs / T_BASE;

      if (view === 'accumulate') {
        stage.append(
          el('p', { class: 'stage-title', text: 'One key and one value are kept per token generated' }),
          el('p', {
            class: 'stage-sub',
            text: 'During generation the model has already computed the key and value for every '
              + 'earlier token. Recomputing them each step would be wasteful, so it keeps them. '
              + 'That growing list is the KV cache — the model\'s saved attention history for one '
              + 'active conversation.',
          }),
          accumulateView(),
          el('pre', {
            class: 'ascii',
            text: 'Model weights   → loaded once, shared by every user\n'
              + 'KV cache        → private to one conversation, cannot be shared\n\n'
              + '   user 1 conversation → its own cache\n'
              + '   user 2 conversation → its own cache\n'
              + '   user 3 conversation → its own cache',
          }),
        );
      } else {
        stage.append(
          el('p', { class: 'stage-title', text: 'The same context length, two different slopes' }),
          el('p', {
            class: 'stage-sub',
            text: 'Both axes are logarithmic, so a straight line is a power law and its steepness '
              + 'is the exponent. Compute climbs at twice the slope of memory. Doubling the '
              + 'context multiplies comparisons by 4 and the cache by 2.',
          }),
          chart(T),
        );
      }

      readouts([
        {
          label: 'Query-key scores', value: compact(scores), tone: 'orange',
          sub: `${tokens(T)} × ${tokens(T)}`,
        },
        {
          label: 'KV pairs stored', value: compact(kvPairs), tone: 'blue',
          sub: 'one per earlier token',
        },
        {
          label: 'Compute vs 128 tok', value: `${compact(computeX)}×`, tone: 'orange',
          sub: 'grows as T²',
        },
        {
          label: 'Cache vs 128 tok', value: `${compact(cacheX)}×`, tone: 'blue',
          sub: 'grows as T',
        },
        // NOT a ratio between the two multipliers — that is algebraically identical
        // to the cache multiplier, (T/T₀)²/(T/T₀) = T/T₀, so it carries no new
        // information. What is worth showing is the absolute cost of one more doubling.
        {
          label: `One more doubling → ${tokens(T * 2)}`,
          value: compact(scores * 4), tone: 'violet',
          // tokens() not compact() for the pair count: pairs are per-token, so it
          // should read "1K" like the label, not compact()'s "1.02K".
          sub: `scores, up from ${compact(scores)} · cache ${tokens(kvPairs * 2)} pairs`,
        },
      ]);
    }

    function readouts(list) { list.forEach((r) => ros.append(readout(r).node)); }

    /* ---- accumulate view: the ASCII table from the lesson, drawn ---- */
    function accumulateView() {
      const rows = [
        ['after "The"', 1],
        ['after "The cat"', 2],
        ['after "The cat sat"', 3],
        ['after "The cat sat on"', 4],
      ];
      const wrap = el('div', { style: { display: 'flex', flexDirection: 'column', gap: '9px' } });
      rows.forEach(([label, n]) => {
        const chips = [];
        for (let i = 1; i <= n; i++) {
          chips.push(el('span', {
            style: {
              font: '600 11.5px ui-monospace,monospace',
              border: '1px solid rgba(57,135,229,.45)',
              background: 'rgba(57,135,229,.13)',
              borderRadius: '7px', padding: '5px 8px', whiteSpace: 'nowrap',
              opacity: i === n ? '1' : '.62',
            },
          }, `K${i},V${i}`));
        }
        wrap.append(el('div', {
          style: { display: 'flex', alignItems: 'center', gap: '9px', flexWrap: 'wrap' },
        },
          el('span', {
            style: {
              font: '12px ui-monospace,monospace', color: 'var(--muted)',
              minWidth: '168px', textAlign: 'right',
            },
            text: label,
          }),
          el('span', { style: { color: 'var(--muted)' }, text: '→' }),
          chips,
          el('span', {
            style: { font: '11.5px ui-monospace,monospace', color: 'var(--muted)' },
            text: `${n} pair${n > 1 ? 's' : ''}`,
          })));
      });
      return wrap;
    }

    /* ---- log-log chart of the two growth patterns ---- */
    function chart(current) {
      const W = 720, H = 320, L = 62, R = 18, TOP = 18, B = 46;
      const iw = W - L - R, ih = H - TOP - B;

      const lo = Math.log2(T_VALUES[0]);
      const hi = Math.log2(T_VALUES[T_VALUES.length - 1]);
      const maxY = Math.log10((T_VALUES[T_VALUES.length - 1] / T_BASE) ** 2);

      const px = (t) => L + ((Math.log2(t) - lo) / (hi - lo)) * iw;
      const py = (mult) => TOP + ih - (Math.log10(Math.max(1, mult)) / maxY) * ih;

      const g = svg('svg', {
        viewBox: `0 0 ${W} ${H}`, width: '100%',
        style: 'display:block;max-width:100%;height:auto', role: 'img',
        'aria-label': 'Log-log chart: attention compute grows as T squared while KV cache grows as T',
      });

      // y gridlines at each decade
      for (let d = 0; d <= Math.ceil(maxY); d++) {
        const y = py(10 ** d);
        if (y < TOP) continue;
        g.append(
          svg('line', {
            x1: L, x2: W - R, y1: y, y2: y,
            stroke: 'rgba(255,255,255,.07)', 'stroke-width': 1,
          }),
          svg('text', {
            x: L - 10, y: y + 4, 'text-anchor': 'end', 'font-size': 10.5,
            fill: '#89877f', text: d === 0 ? '1×' : `10${sup(d)}×`,
          }),
        );
      }

      // x ticks on selected powers of two
      T_VALUES.forEach((t) => {
        if (![128, 1024, 8192, 65536, 524288].includes(t)) return;
        g.append(svg('text', {
          x: px(t), y: H - B + 20, 'text-anchor': 'middle', 'font-size': 10.5,
          fill: '#89877f', text: tokens(t),
        }));
      });
      g.append(svg('text', {
        x: L + iw / 2, y: H - 6, 'text-anchor': 'middle', 'font-size': 11,
        fill: '#89877f', text: 'context length T  (log scale)',
      }));

      const line = (fn, color, dash) => svg('polyline', {
        points: T_VALUES.map((t) => `${fx(px(t), 1)},${fx(py(fn(t)), 1)}`).join(' '),
        fill: 'none', stroke: color, 'stroke-width': 2.4,
        'stroke-dasharray': dash || null, 'stroke-linejoin': 'round',
      });

      const compute = (t) => (t / T_BASE) ** 2;
      const cache = (t) => t / T_BASE;

      g.append(
        line(compute, '#eb6834'),
        line(cache, '#3987e5'),
        // current-T marker
        svg('line', {
          x1: px(current), x2: px(current), y1: TOP, y2: TOP + ih,
          stroke: 'rgba(255,255,255,.28)', 'stroke-width': 1, 'stroke-dasharray': '3 3',
        }),
        svg('circle', { cx: px(current), cy: py(compute(current)), r: 5, fill: '#eb6834' }),
        svg('circle', { cx: px(current), cy: py(cache(current)), r: 5, fill: '#3987e5' }),
      );

      // labels riding on the lines
      const last = T_VALUES[T_VALUES.length - 1];
      g.append(
        svg('text', {
          x: px(last) - 6, y: py(compute(last)) + 16, 'text-anchor': 'end',
          'font-size': 11.5, fill: '#f5a07c', 'font-weight': 600, text: 'compute ∝ T²',
        }),
        svg('text', {
          x: px(last) - 6, y: py(cache(last)) - 9, 'text-anchor': 'end',
          'font-size': 11.5, fill: '#8fbcf5', 'font-weight': 600, text: 'KV cache ∝ T',
        }),
      );
      return g;
    }

    const sup = (d) => String(d).split('').map((c) => '⁰¹²³⁴⁵⁶⁷⁸⁹'[+c]).join('');

    render();
  },
);
