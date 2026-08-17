/* §10 — The cache bill, done properly.
   The formula is revealed one factor at a time, because every factor answers a
   simple question and the long version is unreadable if it arrives all at once.

   Architecture is the lesson's yardstick and stays fixed: 48 layers, 8 KV heads,
   head_dim 128, bf16. At T = 32,768 that gives 6.44 GB for one user and
   51.54 GB for eight — decimal GB, as the lesson quotes them. */

import {
  register, el, slider, segmented, readout, bytes, commas, compact, tokens, fx, clear, svg,
} from '../lib.js';

const LAYERS = 48;
const KV_HEADS = 8;
const HEAD_DIM = 128;

const PRECISION = [
  { value: 4, label: 'fp32 · 4 B' },
  { value: 2, label: 'bf16 · 2 B' },
  { value: 1, label: 'fp8 · 1 B' },
];

const T_VALUES = [4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576];

register(
  'w-s10-cache-bill',
  'The cache bill, factor by factor',
  '§10 · 48 layers · 8 KV heads · d=128',
  'Press "Next multiplier" to build the formula one factor at a time — each one answers a '
  + 'simple question. Then change the context length to move one user\'s cache, and the number '
  + 'of active users to see that private cache copied across concurrent conversations.',
  (root) => {
    let shown = 1;
    let T = 32768;
    let batch = 1;
    let bpn = 2;

    const stage = el('div', { class: 'stage' });
    const ros = el('div', { class: 'readouts' });

    /* Every factor, with the question it answers. */
    const factors = () => [
      {
        sym: '2', val: 2,
        why: 'one key and one value are stored for each token',
        means: 'vectors, per KV head, per token, per layer',
      },
      {
        sym: 'kv_heads', val: KV_HEADS,
        why: 'each KV head stores its own key and value',
        means: 'vectors per token, per layer',
      },
      {
        sym: 'head_dim', val: HEAD_DIM,
        why: 'each vector holds this many numbers',
        means: 'numbers per token, per layer',
      },
      {
        sym: 'layers', val: LAYERS,
        why: 'every layer computes and keeps its own K and V',
        means: 'numbers per token, whole model',
      },
      {
        sym: 'T', val: T,
        why: 'every earlier token left its own K/V record behind',
        means: 'numbers for this conversation',
      },
      {
        sym: 'bytes', val: bpn,
        why: `each stored number occupies memory — ${PRECISION.find((p) => p.value === bpn).label}`,
        means: 'BYTES for this conversation',
      },
      {
        sym: 'batch', val: batch,
        why: 'caches are private: one per active conversation, never shared',
        means: 'BYTES across every active user',
      },
    ];

    const next = el('button', {
      class: 'btn primary',
      onclick: () => { shown = Math.min(7, shown + 1); render(); },
    }, 'Next multiplier →');
    const reset = el('button', {
      class: 'btn',
      onclick: () => { shown = 1; render(); },
    }, 'Reset');

    const tS = slider({
      label: 'Context length T', values: T_VALUES, value: T,
      fmt: (v) => `${tokens(v)} tokens`,
      onInput: (v) => { T = v; render(); },
    });
    const bS = slider({
      label: 'Active users', min: 1, max: 64, step: 1, value: batch,
      fmt: (v) => `${v}`,
      onInput: (v) => { batch = v; render(); },
    });
    const pSel = segmented({
      label: 'Cache precision', options: PRECISION, value: bpn,
      onChange: (v) => { bpn = v; render(); },
    });

    root.append(
      el('div', { class: 'controls' },
        el('div', { class: 'ctl' }, next), el('div', { class: 'ctl' }, reset),
        tS.node, bS.node, pSel.node),
      stage,
      ros,
    );

    function render() {
      clear(stage); clear(ros);
      next.disabled = shown >= 7;
      const F = factors();
      const active = F.slice(0, shown);
      const product = active.reduce((p, f) => p * f.val, 1);
      const last = active[active.length - 1];

      // full totals, independent of how much of the formula is revealed
      const perUser = 2 * LAYERS * KV_HEADS * HEAD_DIM * T * bpn;
      const total = perUser * batch;
      const perToken = 2 * LAYERS * KV_HEADS * HEAD_DIM * bpn;

      stage.append(
        el('p', {
          class: 'stage-title',
          text: shown < 7
            ? `Factor ${shown} of 7 — ${last.sym}`
            : 'The complete formula',
        }),
        el('p', { class: 'stage-sub', text: last.why }),

        el('pre', {
          class: 'ascii',
          text: `  cache = ${F.map((f, i) => (i < shown ? f.sym : '·')).join(' × ')}\n`
            + `        = ${active.map((f) => commas(f.val)).join(' × ')}\n`
            + `        = ${commas(product)}   ${last.means}`,
        }),

        // Why each revealed factor exists.
        el('pre', {
          class: 'ascii',
          text: '  factor        value        what it answers\n'
            + `  ${'─'.repeat(74)}\n`
            + active.map((f) => `  ${f.sym.padEnd(13)} ${commas(f.val).padStart(9)}    ${f.why}`)
              .join('\n')
            + (shown < 7
              ? `\n\n  ${7 - shown} factor${7 - shown > 1 ? 's' : ''} still to come.`
              : ''),
        }),
      );

      if (shown >= 6) {
        stage.append(
          el('pre', {
            class: 'ascii',
            text: `  one user  at ${tokens(T)}   ≈ ${bytes(perUser)}\n`
              + `  ${batch} user${batch > 1 ? 's' : ''} at ${tokens(T)}   ≈ ${bytes(total)}\n\n`
              + '  Double the context and both numbers double.\n'
              + '  Hold the context and double the users, and the total doubles too.\n\n'
              + '  Context length sets the cache cost of ONE conversation.\n'
              + '  Concurrency multiplies that cost by the number of active conversations.',
          }),
          shareChart(perUser),
        );
      }

      if (shown >= 7) {
        stage.append(el('p', {
          class: 'note warn',
          html: '<b>This counts only the raw K and V tensors.</b> A real inference server also needs '
            + 'memory for model weights, activations, temporary attention workspaces and allocator '
            + 'headroom. The formula explains the cache bill — it does not predict total accelerator '
            + 'memory by itself.',
        }));
      }

      addReadouts([
        { label: 'Per token', value: bytes(perToken), tone: 'blue', sub: 'across all 48 layers' },
        {
          label: `One user at ${tokens(T)}`, value: bytes(perUser), tone: 'violet',
          sub: 'private to that conversation',
        },
        {
          label: `${batch} active user${batch > 1 ? 's' : ''}`, value: bytes(total), tone: 'orange',
          sub: 'nothing here is shared',
        },
        {
          label: 'Numbers cached', value: compact(perUser / bpn), tone: 'aqua',
          sub: 'per user, at this context',
        },
        {
          label: 'Formula revealed', value: `${shown} / 7`,
          sub: shown < 7 ? 'press Next multiplier' : 'complete',
        },
      ]);
    }

    function addReadouts(list) { list.forEach((r) => ros.append(readout(r).node)); }

    /** Weights are loaded once; caches are not. One bar per active conversation. */
    function shareChart(perUser) {
      const n = Math.min(batch, 16);
      const W = 660, rowH = 22, H = 42 + n * rowH;
      const g = svg('svg', {
        viewBox: `0 0 ${W} ${H}`, width: '100%',
        style: 'display:block;max-width:100%;height:auto;margin:4px 0 2px', role: 'img',
        'aria-label': `${batch} active conversations, each holding its own ${bytes(perUser)} cache`,
      });
      g.append(svg('text', {
        x: 0, y: 12, 'font-size': 11, fill: '#89877f',
        text: `each active conversation holds its own copy — ${bytes(perUser)} each`,
      }));
      for (let i = 0; i < n; i++) {
        const y = 24 + i * rowH;
        g.append(
          svg('text', {
            x: 0, y: y + 13, 'font-size': 10.5, fill: '#89877f', text: `user ${i + 1}`,
          }),
          svg('rect', {
            x: 58, y, width: W - 150, height: 16, rx: 4, fill: 'rgba(57,135,229,.30)',
          }),
          svg('text', {
            x: W - 86, y: y + 13, 'font-size': 10.5, fill: '#8fbcf5', text: bytes(perUser),
          }),
        );
      }
      if (batch > n) {
        g.append(svg('text', {
          x: 58, y: H - 6, 'font-size': 10.5, fill: '#89877f',
          text: `… and ${batch - n} more, ${bytes(perUser * (batch - n))} between them`,
        }));
      }
      return g;
    }

    render();
  },
);
