/* §12 — Compressing the sequence itself.
   §11 reduced kv_heads. This reduces the OTHER term: how many token positions get
   an entry at all.

   Two savings that are easy to conflate, so the widget keeps them apart:
     compression (block size m) → fewer entries STORED       → T / m
     top-k selection            → fewer summaries READ       → k

   Geometry matches §11's "typical" preset (32 layers, 8 KV heads, d=128, bf16), so
   the uncompressed 256K figure is the same 34.36 GB that section ended on.

   The summary here is a plain mean. Real systems learn the compression; a mean is
   just the simplest thing that shows what compression costs. */

import {
  register, el, matrixTable as mt, stepper, slider, readout, svg,
  bytes, commas, compact, tokens, fx, clear,
} from '../lib.js';

const LAYERS = 32;
const KV_HEADS = 8;
const HEAD_DIM = 128;
const BPN = 2;
const BYTES_PER_POS = 2 * LAYERS * KV_HEADS * HEAD_DIM * BPN;   // 131,072

const T_VALUES = [8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576];
const BLOCK_SIZES = [1, 2, 4, 8, 16, 32, 64];

/* Four token vectors for the averaging demonstration. */
const TOK = [
  [0.8, -0.2, 0.5, 0.1],
  [-0.6, 0.9, 0.2, -0.4],
  [0.3, 0.4, -0.7, 0.6],
  [0.1, -0.5, 0.9, 0.2],
];
const MEAN = TOK[0].map((_, d) => TOK.reduce((s, t) => s + t[d], 0) / TOK.length);

const STRIP = 48;        // tokens drawn in the illustration

const STEPS = [
  { key: 'step 1', title: 'One entry per token' },
  { key: 'step 2', title: 'Compress nearby tokens' },
  { key: 'step 3', title: 'Read only the top-k' },
  { key: 'step 4', title: 'Two different savings' },
];

register(
  'w-s12-compression',
  'Fewer positions, then fewer reads',
  '§12 · sequence compression + top-k',
  'Raise the block size and watch the number of stored positions fall. Then change top-k and '
  + 'watch expensive attention read only that many summaries. These are two separate savings — '
  + 'one on what is kept, one on what is looked at.',
  (root) => {
    let step = 0;
    let T = 262144;
    let m = 8;
    let k = 4;

    const stage = el('div', { class: 'stage' });
    const ros = el('div', { class: 'readouts' });

    const seq = stepper({ steps: STEPS, value: 0, onChange: (s) => { step = s; render(); } });

    const tS = slider({
      label: 'Context length T', values: T_VALUES, value: T,
      fmt: (v) => `${tokens(v)} tokens`,
      onInput: (v) => { T = v; render(); },
    });
    const mS = slider({
      label: 'Block size m', values: BLOCK_SIZES, value: m,
      fmt: (v) => (v === 1 ? '1 token (no compression)' : `${v} tokens per block`),
      onInput: (v) => { m = v; render(); },
    });
    const kS = slider({
      label: 'Top-k blocks read', min: 1, max: 32, step: 1, value: k,
      fmt: (v) => `${v} block${v > 1 ? 's' : ''}`,
      onInput: (v) => { k = v; render(); },
    });

    root.append(
      seq.node,
      el('div', { class: 'controls' }, tS.node, mS.node, kS.node),
      stage, ros,
    );

    function render() {
      clear(stage); clear(ros);
      const blocks = Math.ceil(T / m);
      const kk = Math.min(k, blocks);
      const fullBytes = T * BYTES_PER_POS;
      const compBytes = blocks * BYTES_PER_POS;

      if (step === 0) stepBaseline(fullBytes);
      else if (step === 1) stepCompress(blocks, fullBytes, compBytes);
      else if (step === 2) stepSelect(blocks, kk);
      else stepBoth(blocks, kk, fullBytes, compBytes);

      addReadouts([
        {
          label: 'Stored positions', value: compact(blocks), tone: 'blue',
          sub: m === 1 ? `T, uncompressed` : `T / m = ${compact(T)} / ${m}`,
        },
        {
          label: 'Cache for one user', value: bytes(compBytes), tone: 'violet',
          sub: m === 1 ? 'no compression yet' : `${fx(fullBytes / compBytes, 0)}× less than T entries`,
        },
        {
          label: 'Summaries read per query', value: String(kk), tone: 'aqua',
          sub: `of ${compact(blocks)} available`,
        },
        {
          label: 'Expensive reads vs full', value: `${fx(T / kk, 0)}× fewer`, tone: 'aqua',
          sub: `${compact(T)} → ${kk}`,
        },
        {
          label: 'Tokens behind one summary', value: String(m),
          tone: m > 8 ? 'red' : m > 1 ? 'orange' : '',
          sub: m === 1 ? 'token-level detail intact' : 'their detail is merged',
        },
      ]);
    }

    function addReadouts(list) { list.forEach((r) => ros.append(readout(r).node)); }

    /* ---- step 1 ---- */
    function stepBaseline(fullBytes) {
      stage.append(
        el('p', { class: 'stage-title', text: 'Even with GQA, every token still gets an entry' }),
        el('p', {
          class: 'stage-sub',
          text: '§11 reduced how many K/V heads are stored per position. It did not reduce the '
            + 'number of positions. That is the term this section attacks.',
        }),
        el('pre', {
          class: 'ascii',
          text: '  t1 → KV1\n  t2 → KV2\n  t3 → KV3\n  t4 → KV4\n  …    one stored entry per token\n\n'
            + `  at T = ${commas(T)} that is ${commas(T)} stored positions\n`
            + `  × ${commas(BYTES_PER_POS)} bytes each  =  ${bytes(fullBytes)}\n\n`
            + `  (${LAYERS} layers × ${KV_HEADS} KV heads × ${HEAD_DIM} dims × 2 for K and V × ${BPN} bytes\n`
            + `   = ${commas(BYTES_PER_POS)} bytes per position — the §11 GQA geometry)`,
        }),
        strip(1, 0),
      );
    }

    /* ---- step 2 ---- */
    function stepCompress(blocks, fullBytes, compBytes) {
      stage.append(
        el('p', {
          class: 'stage-title',
          text: m === 1
            ? 'Block size 1 — nothing is compressed yet'
            : `${m} nearby tokens share one stored entry`,
        }),
        el('p', {
          class: 'stage-sub',
          text: 'Combine neighbouring positions into one block summary. If every block holds m '
            + 'tokens, the number of long-range entries falls from roughly T to T / m. GQA stored '
            + 'fewer heads per position; this stores fewer positions.',
        }),
        strip(m, 0),
        el('pre', {
          class: 'ascii',
          text: `  stored positions before:  ${commas(T)}\n`
            + `  stored positions after:    ${commas(blocks)}   =  T / m  =  ${commas(T)} / ${m}\n\n`
            + `  cache before:  ${bytes(fullBytes).padStart(9)}\n`
            + `  cache after:   ${bytes(compBytes).padStart(9)}`
            + (m > 1 ? `   ${fx(fullBytes / compBytes, 0)}× smaller` : '   unchanged'),
        }),
        el('p', {
          class: 'stage-sub',
          style: { marginTop: '18px' },
          text: 'But a summary speaks for several tokens, and that costs something real. Below is a '
            + 'block of four tokens and their mean — four rather than m, just to keep the grid '
            + 'readable. The effect only gets stronger as m grows.',
        }),
        el('div', { class: 'gridrow' },
          mt({
            caption: '4 tokens in a block', data: TOK,
            rowLabels: TOK.map((_, i) => `t${i + 1}`), colLabels: ['0', '1', '2', '3'],
            cell: () => ({ bg: 'rgba(57,135,229,.12)' }),
          }),
          el('div', { class: 'op', text: '→' }),
          mt({
            caption: 'one summary', data: [MEAN],
            rowLabels: ['mean'], colLabels: ['0', '1', '2', '3'], digits: 3,
            cell: () => ({ bg: 'rgba(144,133,233,.22)' }),
          }),
        ),
        el('pre', {
          class: 'ascii',
          text: MEAN.map((v, d) => `  slot ${d}:  (${TOK.map((t) => fx(t[d])).join(' + ')}) / 4`
            + `  =  ${fx(v, 3)}`).join('\n')
            + '\n\n  Notice how small the summary is compared with its inputs. Slot 0 held\n'
            + '  0.80 and −0.60; they largely cancelled to 0.15. Opposing detail inside a\n'
            + '  block destroys itself, and no query can ever recover t1 from the summary.\n\n'
            + '  A mean is the crudest possible summary — real systems LEARN the compression,\n'
            + '  which cancels far less than this. But something is always lost, and at the\n'
            + `  current setting one stored entry speaks for ${m} token${m > 1 ? 's' : ''}`
            + `${m > 4 ? ' — twice as many as shown here, so more is merged, not less' : ''}.`,
        }),
      );
    }

    /* ---- step 3 ---- */
    function stepSelect(blocks, kk) {
      stage.append(
        el('p', { class: 'stage-title', text: `Run expensive attention on ${kk} summaries, not all ${compact(blocks)}` }),
        el('p', {
          class: 'stage-sub',
          text: 'Compression alone reduced storage. It did not reduce how much gets read for one '
            + 'query. DeepSeek-V4\'s Compressed Sparse Attention adds a second step: after making '
            + 'the block summaries, select only the top-k that look most relevant, and read those.',
        }),
        strip(m, kk),
        el('pre', {
          class: 'ascii',
          text: '  all tokens\n'
            + `      │  compress nearby tokens  (m = ${m})\n`
            + `  ${commas(blocks)} block summaries\n`
            + `      │  select top-k  (k = ${kk})\n`
            + `  ${kk} expensive attention read${kk > 1 ? 's' : ''}\n\n`
            + `  full attention would read   ${commas(T)} positions\n`
            + `  this reads                  ${kk}\n`
            + `                              ${fx(T / kk, 0)}× fewer`,
        }),
        el('p', {
          class: 'note',
          html: '<b>But selecting the best blocks creates the §7 problem again.</b> If we ran full '
            + 'attention over every summary just to find the best ones, little would be saved. '
            + 'DeepSeek uses a small <b>low-rank indexer</b> to rank the summaries cheaply. The '
            + 'indexer does not produce the attention output — it only chooses which blocks deserve '
            + 'the expensive read. A low-rank scorer costs a fraction of a full attention read, '
            + 'which is what makes the ranking worth doing.',
        }),
      );
    }

    /* ---- step 4 ---- */
    function stepBoth(blocks, kk, fullBytes, compBytes) {
      stage.append(
        el('p', { class: 'stage-title', text: 'Two savings, two different costs' }),
        el('p', {
          class: 'stage-sub',
          text: 'These are independent levers and it is worth keeping them apart. One changes what '
            + 'is kept in memory; the other changes what is looked at per query. You can have '
            + 'either without the other.',
        }),
        el('pre', {
          class: 'ascii',
          text: `  compression (m = ${m})   →  fewer entries STORED   →  ${commas(T)} → ${commas(blocks)}\n`
            + `                            ${bytes(fullBytes)} → ${bytes(compBytes)}\n\n`
            + `  top-k (k = ${kk})          →  fewer summaries READ   →  ${commas(blocks)} → ${kk}\n`
            + `                            per query, per layer\n\n`
            + `  ${'─'.repeat(66)}\n`
            + `  storage saving:  ${m > 1 ? `${fx(fullBytes / compBytes, 0)}×` : 'none — m = 1'}\n`
            + `  read saving:     ${fx(blocks / kk, 0)}× against the summaries,`
            + ` ${fx(T / kk, 0)}× against raw tokens`,
        }),
        savingBars(blocks, kk),
        el('pre', {
          class: 'ascii',
          text: '  Sequence compression reduces how much history is STORED.\n'
            + '  Top-k selection reduces how much of that history is READ for one query.',
        }),
        el('p', {
          class: 'note warn',
          html: '<b>Both steps can lose the answer.</b> Compression discards token-level detail, '
            + 'because one summary now speaks for several tokens. Approximate top-k can miss a '
            + 'useful block entirely. A real architecture has to protect important recent detail '
            + 'and train the summaries and the indexer well — this widget demonstrates the '
            + 'storage-and-read scaling, not the model\'s quality.',
        }),
        el('p', {
          class: 'stage-sub',
          text: 'The reported DeepSeek-V4 architecture also interleaves a heavily compressed dense '
            + 'form with the top-k sparse form. The shared idea is the same either way: old history '
            + 'does not have to keep one equally expensive representation for every original token.',
        }),
      );
    }

    /* ---- the token strip, grouped into blocks ---- */
    function strip(mm, kk) {
      const nBlocks = Math.ceil(STRIP / mm);
      const cell = 11, gap = 2, blockGap = 7;
      const W = 700;
      const g = svg('svg', {
        viewBox: `0 0 ${W} 108`, width: '100%',
        style: 'display:block;max-width:100%;height:auto;margin:6px 0 16px', role: 'img',
        'aria-label': `${STRIP} tokens grouped into blocks of ${mm}${kk ? `, ${kk} selected` : ''}`,
      });
      // Deterministic "relevance" so the selected blocks are stable, not random per render.
      const rank = Array.from({ length: nBlocks }, (_, b) => ({
        b, s: Math.abs(Math.sin((b + 1) * 2.399)),
      })).sort((a, x) => x.s - a.s).slice(0, kk).map((r) => r.b);

      g.append(svg('text', {
        x: 0, y: 10, 'font-size': 10.5, fill: '#89877f',
        text: `first ${STRIP} tokens — illustration only; the numbers below use T = ${tokens(T)}`,
      }));

      let x = 0;
      for (let b = 0; b < nBlocks; b++) {
        const n = Math.min(mm, STRIP - b * mm);
        const wBlock = n * cell + (n - 1) * gap;
        const chosen = rank.includes(b);
        for (let i = 0; i < n; i++) {
          g.append(svg('rect', {
            x: x + i * (cell + gap), y: 22, width: cell, height: cell, rx: 2,
            fill: 'rgba(57,135,229,.30)',
          }));
        }
        if (mm > 1) {
          g.append(
            svg('rect', {
              x: x - 3, y: 19, width: wBlock + 6, height: cell + 6, rx: 3,
              fill: 'none', stroke: 'rgba(255,255,255,.20)', 'stroke-width': 1,
            }),
            svg('rect', {
              x: x - 3, y: 52, width: wBlock + 6, height: cell + 4, rx: 3,
              fill: chosen ? 'rgba(27,175,122,.55)' : 'rgba(144,133,233,.24)',
              stroke: chosen ? '#1baf7a' : 'rgba(144,133,233,.4)',
            }),
          );
        } else {
          g.append(svg('rect', {
            x, y: 52, width: cell, height: cell + 4, rx: 2,
            fill: chosen ? 'rgba(27,175,122,.55)' : 'rgba(144,133,233,.24)',
          }));
        }
        x += wBlock + (mm > 1 ? blockGap + 6 : gap);
        if (x > W - 40) break;
      }
      g.append(
        svg('text', { x: 0, y: 46, 'font-size': 10, fill: '#89877f', text: 'tokens' }),
        svg('text', {
          x: 0, y: 82, 'font-size': 10, fill: '#89877f',
          text: mm === 1 ? 'stored entries — one per token' : `stored entries — one per block of ${mm}`,
        }),
        kk ? svg('text', {
          x: 0, y: 98, 'font-size': 10, fill: '#5fd2a8',
          text: `green = the ${kk} block${kk > 1 ? 's' : ''} the indexer proposed for the expensive read`,
        }) : null,
      );
      return g;
    }

    /** Storage and read savings as two bars, since they are different quantities. */
    function savingBars(blocks, kk) {
      const W = 660, H = 104, L = 118, R = 116;
      const iw = W - L - R;
      const g = svg('svg', {
        viewBox: `0 0 ${W} ${H}`, width: '100%',
        style: 'display:block;max-width:100%;height:auto;margin:6px 0 2px', role: 'img',
        'aria-label': 'Storage and read savings shown as separate bars',
      });
      const bar = (y, frac, colour, label, note) => g.append(
        svg('text', {
          x: L - 12, y: y + 16, 'text-anchor': 'end', 'font-size': 11, fill: '#b8b7b0', text: label,
        }),
        svg('rect', { x: L, y, width: iw, height: 22, rx: 4, fill: 'rgba(255,255,255,.05)' }),
        svg('rect', {
          x: L, y, width: Math.max(2, iw * frac), height: 22, rx: 4, fill: colour,
        }),
        svg('text', {
          x: L + iw + 8, y: y + 16, 'font-size': 10.5, fill: '#89877f', text: note,
        }),
      );
      bar(6, blocks / T, '#3987e5', 'stored', `${commas(blocks)} of ${commas(T)}`);
      bar(38, kk / T, '#1baf7a', 'read per query', `${kk} of ${commas(T)}`);
      bar(70, 1, '#eb6834', 'uncompressed', `${commas(T)} of ${commas(T)}`);
      return g;
    }

    render();
  },
);
