/* §15 — Long context is a system, not a number.
   Six conditions have to hold at once. The board checks the four that follow from
   a configuration, and is explicit that two of them cannot be computed from one —
   they are claims about a training run and an evaluation suite.

   The second panel converts the same window into human document capacity under
   different tokenizer fertility, so the cost of a 3× language shows up as lost
   pages rather than an abstract ratio. */

import {
  register, el, segmented, slider, toggle, readout, svg,
  bytes, commas, tokens, fx, clear,
} from '../lib.js';
import {
  ATTN, EXTENSION, POSITION, DEFAULT_CONFIG,
  cacheBytes, stateBytes, servingBytes, mixingRelativeToFull, reach, pages,
} from '../costs.js';

const T_VALUES = [8192, 32768, 131072, 262144, 524288, 1048576];
const FERTILITY = [1, 1.5, 2, 2.5, 3, 3.5];

register(
  'w-s15-readiness',
  'Context readiness board',
  '§15 · six conditions, one binding',
  'Choose a target context and a configuration. The board checks each condition and names which '
  + 'one fails first, and by how much. Beside it, the same window is measured in documents under '
  + 'different tokenizer fertility — because the accelerator charges per token either way.',
  (root) => {
    const c = { ...DEFAULT_CONFIG };

    const stage = el('div', { class: 'stage' });
    const ros = el('div', { class: 'readouts' });
    let fertility = 3;

    const ctl = [
      slider({
        label: 'Target context', values: T_VALUES, value: c.targetCtx,
        fmt: (v) => `${tokens(v)} tokens`, onInput: (v) => { c.targetCtx = v; render(); },
      }),
      slider({
        label: 'Trained context', values: [4096, 8192, 16384, 32768, 131072, 1048576],
        value: c.trainedCtx,
        fmt: (v) => `${tokens(v)} tokens`, onInput: (v) => { c.trainedCtx = v; render(); },
      }),
      segmented({
        label: 'Position', options: Object.entries(POSITION).map(([k, p]) => ({ value: k, label: p.label })),
        value: c.position, onChange: (v) => { c.position = v; render(); },
      }),
      segmented({
        label: 'Extension', options: Object.entries(EXTENSION).map(([k, e]) => ({ value: k, label: k === 'none' ? 'native' : e.label })),
        value: c.extension, onChange: (v) => { c.extension = v; render(); },
      }),
      segmented({
        label: 'Attention family', options: Object.entries(ATTN).map(([k, a]) => ({ value: k, label: a.label })),
        value: c.attn, onChange: (v) => { c.attn = v; render(); },
      }),
      segmented({
        label: 'K/V heads', options: [{ value: 32, label: 'MHA 32' }, { value: 8, label: 'GQA 8' }, { value: 1, label: 'MQA 1' }],
        value: c.kvHeads, onChange: (v) => { c.kvHeads = v; render(); },
      }),
      slider({
        label: 'Sparse layers per 8', min: 0, max: 8, step: 1, value: c.gPer8,
        fmt: (v) => `${v} G`, onInput: (v) => { c.gPer8 = v; render(); },
      }),
      slider({
        label: 'Memory budget', min: 40, max: 640, step: 40, value: c.memBudgetGB,
        fmt: (v) => `${v} GB`, onInput: (v) => { c.memBudgetGB = v; render(); },
      }),
      slider({
        label: 'Active users', min: 1, max: 32, step: 1, value: c.batch,
        fmt: (v) => String(v), onInput: (v) => { c.batch = v; render(); },
      }),
      slider({
        label: 'Tokenizer fertility', values: FERTILITY, value: fertility,
        fmt: (v) => `${v}× ${v === 1 ? '(English)' : v === 3 ? '(≈ Telugu)' : ''}`,
        onInput: (v) => { fertility = v; render(); },
      }),
    ];
    const streamSw = toggle({
      label: 'Cross-chunk stream', value: c.stream, onChange: (v) => { c.stream = v; render(); },
    });
    const evalSw = toggle({
      label: 'Eval beyond retrieval', value: c.evalBeyondRetrieval,
      onChange: (v) => { c.evalBeyondRetrieval = v; render(); },
    });

    root.append(
      el('div', { class: 'controls' }, ctl.map((x) => x.node)),
      el('div', { class: 'controls' },
        el('div', { class: 'ctl' }, streamSw.node),
        el('div', { class: 'ctl' }, evalSw.node)),
      stage, ros,
    );

    /* ---- the six conditions ---- */
    function checks() {
      const T = c.targetCtx;
      const r = reach(c);
      const perUser = servingBytes(c, T);
      const total = cacheBytes(c, T) * c.batch + stateBytes(c) * c.batch;
      const budget = c.memBudgetGB * 1e9;
      const rel = mixingRelativeToFull(c, T);
      const factor = T / c.trainedCtx;

      return [
        {
          n: 1, name: 'Position still makes sense',
          ok: r.limit >= T,
          value: `reach ${tokens(r.limit)} vs target ${tokens(T)}`,
          detail: r.why,
          fix: r.limit >= T ? null
            : `short by ${fx(T / r.limit, 1)}× — train longer, or use a method with a higher reported ceiling`,
          computed: true,
        },
        {
          n: 2, name: 'The cache fits',
          ok: total <= budget,
          value: `${bytes(total)} of ${c.memBudgetGB} GB`,
          detail: `${bytes(perUser)} per sequence × ${c.batch} active`,
          fix: total <= budget ? null
            : `over by ${bytes(total - budget)} — share more K/V heads, compress the sequence, or drop G layers`,
          computed: true,
        },
        {
          n: 3, name: 'Compute is affordable',
          ok: rel <= 0.25,
          value: `${rel < 0.01 ? fx(rel * 100, 2) + '%' : fx(rel * 100, 1) + '%'} of full attention`,
          detail: c.gPer8 === 0
            ? 'no attention layers — purely linear/DeltaNet state'
            : `${ATTN[c.attn].label}, ${c.gPer8} sparse layer${c.gPer8 === 1 ? '' : 's'} per 8`,
          fix: rel <= 0.25 ? null
            : `still ${fx(rel * 100, 0)}% of quadratic cost — reduce the read budget or use fewer G layers`,
          computed: true,
        },
        {
          n: 4, name: 'State outside the window is handled',
          ok: !c.chunked || c.stream,
          value: c.stream ? 'stream on — O(1) per boundary' : 'no mechanism',
          detail: c.chunked
            ? 'documents are chunked, so something must cross the boundary'
            : 'not chunked — nothing to carry',
          fix: (!c.chunked || c.stream) ? null
            : 'chunk boundaries discard everything — add a cross-chunk carry or stop chunking',
          computed: true,
        },
        {
          n: 5, name: 'Training exposed the model to this length',
          ok: c.extension === 'none' ? c.trainedCtx >= T : factor <= EXTENSION[c.extension].ceiling,
          value: c.extension === 'none'
            ? `trained at ${tokens(c.trainedCtx)}`
            : `${fx(factor, factor % 1 ? 1 : 0)}× extension, ${EXTENSION[c.extension].label}`,
          detail: 'gaps larger than the training window were never seen — §9',
          fix: null,
          computed: false,
          caveat: 'A reported extension factor is evidence for one model and procedure. This row '
            + 'checks the arithmetic against a reported ceiling; it cannot verify that YOUR run '
            + 'adapted successfully.',
        },
        {
          n: 6, name: 'Evaluation tests understanding, not only retrieval',
          ok: c.evalBeyondRetrieval,
          value: c.evalBeyondRetrieval ? 'asserted' : 'not asserted',
          detail: 'needle-in-a-haystack is useful but not a complete long-context evaluation',
          fix: c.evalBeyondRetrieval ? null
            : 'a model can find one planted sentence while failing to combine evidence',
          computed: false,
          caveat: 'This cannot be computed from a configuration at all. It is a property of your '
            + 'evaluation suite, so the board can only record what you declare.',
        },
      ];
    }

    function render() {
      clear(stage); clear(ros);
      const rows = checks();
      const failing = rows.filter((x) => !x.ok);
      const binding = failing[0];
      const T = c.targetCtx;

      stage.append(
        el('p', {
          class: 'stage-title',
          text: failing.length === 0
            ? `All six conditions hold at ${tokens(T)}`
            : `${failing.length} condition${failing.length > 1 ? 's' : ''} fail at ${tokens(T)} — `
              + `binding constraint: ${binding.name.toLowerCase()}`,
        }),
        el('p', {
          class: 'stage-sub',
          text: '"Our model supports 256K context" sounds like one specification. It is not. For the '
            + 'claim to mean anything, several different things have to work at the same time — and '
            + 'the useful question is always which one fails first.',
        }),
        el('pre', {
          class: 'ascii',
          text: rows.map((x) => `  ${x.ok ? '✓' : '✗'}  ${x.n}. ${x.name.padEnd(52)}`
            + `${x.value}\n`
            + `        ${x.detail}`
            + (x.fix ? `\n        → ${x.fix}` : '')
            + (x.computed ? '' : '\n        ⚠ not computable from a config — see below')).join('\n\n'),
        }),
        el('p', {
          class: 'note warn',
          html: '<b>Rows 5 and 6 are declarations, not calculations.</b> '
            + rows[4].caveat + ' ' + rows[5].caveat,
        }),

        el('p', {
          class: 'stage-title',
          style: { marginTop: '26px' },
          text: 'The same window, measured in documents',
        }),
        el('p', {
          class: 'stage-sub',
          text: 'Context windows are advertised in tokens. Users have documents. Session 3 showed why '
            + 'that difference matters: if the same meaning costs a language roughly three times as '
            + 'many tokens, a fixed window holds roughly a third as much of it. The accelerator does '
            + 'not care — it charges per token either way.',
        }),
        fertilityChart(T),
        el('pre', {
          class: 'ascii',
          text: `  at a ${tokens(T)} window:\n\n`
            + FERTILITY.map((f) => `  fertility ${fx(f, 1)}×`
              + `${f === 1 ? '  (English)   ' : f === 3 ? '  (≈ Telugu)  ' : '               '}`
              + `${fx(pages(T, f), 0).padStart(6)} pages`
              + `   ${fx(100 / f, 0).padStart(3)}% of the English-equivalent capacity`
              + (f === fertility ? '  ←' : '')).join('\n')
            + '\n\n  Rough conversion: ~0.75 English words per token, 500 words per page.\n'
            + '  The ratios are the point, not the absolute page counts.\n\n'
            + '  For an India-first model, long context is therefore not only a premium\n'
            + '  feature. It also recovers some of the effective document length lost when\n'
            + '  a language needs more tokens for the same content.',
        }),
      );

      addReadouts([
        {
          label: 'Conditions holding', value: `${6 - failing.length} / 6`,
          tone: failing.length === 0 ? 'aqua' : 'red',
          sub: binding ? `first failure: #${binding.n}` : 'nothing binding',
        },
        {
          label: 'Serving memory', value: bytes(servingBytes(c, T) * c.batch),
          tone: servingBytes(c, T) * c.batch <= c.memBudgetGB * 1e9 ? 'aqua' : 'red',
          sub: `${c.batch} user${c.batch > 1 ? 's' : ''} of ${c.memBudgetGB} GB`,
        },
        {
          label: 'Mixing compute', value: `${fx(mixingRelativeToFull(c, T) * 100, 1)}%`,
          tone: 'blue', sub: 'of full attention at this length',
        },
        {
          label: 'Positional reach', value: tokens(reach(c).limit),
          tone: reach(c).limit >= T ? 'aqua' : 'red',
          sub: `${POSITION[c.position].label}`,
        },
        {
          label: `Capacity at ${fx(fertility, 1)}× fertility`, value: `${fx(pages(T, fertility), 0)} pages`,
          tone: 'violet', sub: `${fx(100 / fertility, 0)}% of English-equivalent`,
        },
      ]);
    }

    function addReadouts(list) { list.forEach((r) => ros.append(readout(r).node)); }

    /** Document capacity against fertility — the same window, different languages. */
    function fertilityChart(T) {
      const W = 660, H = 40 + FERTILITY.length * 28, L = 116, R = 128;
      const iw = W - L - R;
      const maxP = pages(T, 1);
      const g = svg('svg', {
        viewBox: `0 0 ${W} ${H}`, width: '100%',
        style: 'display:block;max-width:100%;height:auto;margin:6px 0 12px', role: 'img',
        'aria-label': 'Document capacity of the same token window under different tokenizer fertility',
      });
      g.append(svg('text', {
        x: 0, y: 11, 'font-size': 10.5, fill: '#89877f',
        text: `the same ${tokens(T)} token window, in pages of comparable content`,
      }));
      FERTILITY.forEach((f, i) => {
        const y = 22 + i * 28;
        const p = pages(T, f);
        const sel = f === fertility;
        g.append(
          svg('text', {
            x: L - 12, y: y + 15, 'text-anchor': 'end', 'font-size': 11,
            fill: sel ? '#f4f4f2' : '#89877f',
            text: `${fx(f, 1)}×${f === 1 ? ' English' : f === 3 ? ' Telugu' : ''}`,
          }),
          svg('rect', { x: L, y, width: iw, height: 20, rx: 4, fill: 'rgba(255,255,255,.045)' }),
          svg('rect', {
            x: L, y, width: Math.max(2, iw * (p / maxP)), height: 20, rx: 4,
            fill: sel ? '#9085e9' : 'rgba(144,133,233,.35)',
          }),
          svg('text', {
            x: L + iw + 8, y: y + 15, 'font-size': 10.5, fill: sel ? '#b5aef2' : '#89877f',
            text: `${fx(p, 0)} pages`,
          }),
        );
      });
      return g;
    }

    render();
  },
);
