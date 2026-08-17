/* Shared helpers for every Session 8 widget.
   DOM building, small-matrix math, and the control primitives.
   Widgets import from here so adding widget N+1 stays cheap. */

/* ---------------- DOM ---------------- */

export function el(tag, props = {}, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (v == null || v === false) continue;
    if (k === 'class') n.className = v;
    else if (k === 'html') n.innerHTML = v;
    else if (k === 'text') n.textContent = v;
    else if (k.startsWith('on')) n.addEventListener(k.slice(2).toLowerCase(), v);
    else if (k === 'style' && typeof v === 'object') Object.assign(n.style, v);
    else n.setAttribute(k, v === true ? '' : v);
  }
  add(n, kids);
  return n;
}

export function svg(tag, attrs = {}, ...kids) {
  const n = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === 'text') n.textContent = v;
    else n.setAttribute(k, v);
  }
  add(n, kids);
  return n;
}

function add(parent, kids) {
  for (const k of kids.flat(Infinity)) {
    if (k == null || k === false) continue;
    parent.append(k instanceof Node ? k : document.createTextNode(String(k)));
  }
}

export const clear = (n) => { while (n.firstChild) n.removeChild(n.firstChild); return n; };

/* ---------------- number formatting ---------------- */

export const fx = (v, d = 2) => (Object.is(v, -0) ? 0 : v).toFixed(d);

/** Thousands separators on an integer-ish number. */
export const commas = (v) => Math.round(v).toLocaleString('en-US');

/** Compact magnitude: 1234 -> "1.23K", 4.1e9 -> "4.10B". */
export function compact(v, d = 2) {
  const a = Math.abs(v);
  if (a >= 1e12) return (v / 1e12).toFixed(d) + 'T';
  if (a >= 1e9) return (v / 1e9).toFixed(d) + 'B';
  if (a >= 1e6) return (v / 1e6).toFixed(d) + 'M';
  if (a >= 1e3) return (v / 1e3).toFixed(d) + 'K';
  return String(Math.round(v));
}

/** Decimal bytes (GB = 1e9), matching the lesson's 6.44 GB figure. */
export function bytes(b, d = 2) {
  if (b >= 1e12) return (b / 1e12).toFixed(d) + ' TB';
  if (b >= 1e9) return (b / 1e9).toFixed(d) + ' GB';
  if (b >= 1e6) return (b / 1e6).toFixed(d) + ' MB';
  if (b >= 1e3) return (b / 1e3).toFixed(d) + ' KB';
  return Math.round(b) + ' B';
}

/** Token counts the way the lesson writes them: 8K, 32K, 256K, 1M. */
export function tokens(t) {
  if (t >= 1e6) return (t / 1e6 % 1 === 0 ? t / 1e6 : (t / 1e6).toFixed(2)) + 'M';
  if (t >= 1024) return Math.round(t / 1024) + 'K';
  return String(t);
}

/* ---------------- small-matrix math ----------------
   Row-major arrays of arrays. Sizes here are tiny (6x6, 4x4), so clarity
   beats performance everywhere in this file. */

export const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);

export const matmul = (A, B) =>
  A.map((row) => B[0].map((_, j) => row.reduce((s, v, k) => s + v * B[k][j], 0)));

export const transpose = (A) => A[0].map((_, j) => A.map((row) => row[j]));

export const scale = (A, s) => A.map((row) => row.map((v) => v * s));

/** Outer product v k^T — the linear-attention state write of §4. */
export const outer = (v, k) => v.map((vi) => k.map((kj) => vi * kj));

export const addM = (A, B) => A.map((row, i) => row.map((v, j) => v + B[i][j]));

export const zeros = (r, c) => Array.from({ length: r }, () => new Array(c).fill(0));

/** Matrix times column vector — reading a linear-attention state, y = S q. */
export const matvec = (A, x) => A.map((row) => dot(row, x));

/** Largest absolute difference between two equal-length vectors. */
export const maxDiff = (a, b) => Math.max(...a.map((v, i) => Math.abs(v - b[i])));

/** Numerically stable softmax over one row. -Infinity entries return 0. */
export function softmax(row) {
  const m = Math.max(...row);
  if (!isFinite(m)) return row.map(() => 0);
  const ex = row.map((v) => (isFinite(v) ? Math.exp(v - m) : 0));
  const sum = ex.reduce((a, b) => a + b, 0);
  return ex.map((v) => v / sum);
}

/** Causal mask as the lesson defines it: 0 allowed, -Infinity for the future. */
export const causalMask = (n) =>
  Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (j <= i ? 0 : -Infinity)));

/** Indices of the k largest entries, ties broken by earlier position. */
export function topKIndices(row, k) {
  return row
    .map((v, i) => [v, i])
    .sort((a, b) => (b[0] - a[0]) || (a[1] - b[1]))
    .slice(0, k)
    .map(([, i]) => i)
    .sort((a, b) => a - b);
}

/** Deterministic PRNG so every reload shows identical numbers. */
export function rng(seed = 8) {
  let s = seed >>> 0 || 1;
  return () => {
    s ^= s << 13; s >>>= 0;
    s ^= s >> 17;
    s ^= s << 5; s >>>= 0;
    return s / 4294967296;
  };
}

/* ---------------- colour ---------------- */

/** Attention-weight heat: transparent -> blue. t in [0,1]. */
export const heat = (t) => `rgba(57,135,229,${(0.06 + 0.82 * Math.max(0, Math.min(1, t))).toFixed(3)})`;

/** Signed score heat: orange for negative, blue for positive. */
export function signedHeat(v, max) {
  const t = max ? Math.max(-1, Math.min(1, v / max)) : 0;
  return t >= 0
    ? `rgba(57,135,229,${(0.05 + 0.55 * t).toFixed(3)})`
    : `rgba(235,104,52,${(0.05 + 0.55 * -t).toFixed(3)})`;
}

/* ---------------- controls ---------------- */

/**
 * Slider. `fmt` renders the live value label; `values` (optional) makes the
 * slider step through a fixed list instead of a numeric range — used for the
 * powers-of-two context sliders.
 */
export function slider({ label, min = 0, max = 1, step = 1, value, values, fmt: f = String, onInput }) {
  const out = el('span', { class: 'val' });
  const idx = values ? Math.max(0, values.indexOf(value)) : 0;
  const input = el('input', {
    type: 'range',
    min: values ? 0 : min,
    max: values ? values.length - 1 : max,
    step: values ? 1 : step,
    value: values ? idx : value,
  });
  const read = () => (values ? values[+input.value] : +input.value);
  const sync = () => { out.textContent = f(read()); };
  input.addEventListener('input', () => { sync(); onInput(read()); });
  sync();
  return {
    node: el('div', { class: 'ctl' },
      el('label', {}, label, ' ', out), input),
    get: read,
    set(v) {
      input.value = values ? Math.max(0, values.indexOf(v)) : v;
      sync();
    },
  };
}

/** Segmented single-choice control. options: [{value,label,title?}] */
export function segmented({ label, options, value, onChange }) {
  const btns = options.map((o) =>
    el('button', {
      type: 'button',
      title: o.title || '',
      'aria-pressed': String(o.value === value),
      onclick: () => { set(o.value); onChange(o.value); },
    }, o.label));
  const set = (v) => btns.forEach((b, i) =>
    b.setAttribute('aria-pressed', String(options[i].value === v)));
  const seg = el('div', { class: 'seg' }, btns);
  return {
    node: label
      ? el('div', { class: 'ctl' }, el('label', {}, label), seg)
      : el('div', { class: 'ctl' }, seg),
    set,
  };
}

/** On/off switch. */
export function toggle({ label, value = false, onChange }) {
  const input = el('input', { type: 'checkbox', checked: value });
  input.addEventListener('change', () => onChange(input.checked));
  return {
    node: el('label', { class: 'switch' }, input, el('span', { text: label })),
    get: () => input.checked,
    set(v) { input.checked = v; },
  };
}

/**
 * Step sequencer — the shared "walk through the mechanism" control.
 * steps: [{key,title}]. Calls onChange(index).
 */
export function stepper({ steps, value = 0, onChange }) {
  const btns = steps.map((s, i) =>
    el('button', {
      type: 'button',
      'aria-pressed': String(i === value),
      onclick: () => { set(i); onChange(i); },
    },
      el('span', { class: 'k', text: s.key }),
      el('span', { class: 't', text: s.title })));
  const set = (i) => btns.forEach((b, j) => b.setAttribute('aria-pressed', String(i === j)));
  return { node: el('div', { class: 'steps' }, btns), set };
}

/** Readout tile. */
export function readout({ label, value, sub, tone = '' }) {
  const v = el('div', { class: 'v', text: value });
  const s = el('div', { class: 'sub', text: sub || '' });
  return {
    node: el('div', { class: `ro ${tone}` },
      el('div', { class: 'lab', text: label }), v, sub ? s : null),
    set(nv, nsub) { v.textContent = nv; if (nsub != null) s.textContent = nsub; },
  };
}

/**
 * Matrix table with optional per-cell classes and background heat.
 * cell(v,i,j) -> {text, cls, bg} lets callers control masking and highlights.
 */
export function matrixTable({ caption, data, rowLabels, colLabels, digits = 2, cell }) {
  const t = el('table', { class: 'grid' });
  if (caption) t.append(el('caption', { text: caption }));
  if (colLabels) {
    t.append(el('tr', {},
      rowLabels ? el('th', {}) : null,
      colLabels.map((c) => el('th', { text: c }))));
  }
  data.forEach((row, i) => {
    t.append(el('tr', {},
      rowLabels ? el('th', { class: 'rowlab', text: rowLabels[i] }) : null,
      row.map((v, j) => {
        const spec = cell ? cell(v, i, j) : {};
        const td = el('td', {
          class: `num ${spec.cls || ''}`,
          text: spec.text != null ? spec.text : fx(v, digits),
        });
        if (spec.bg) td.style.background = spec.bg;
        if (spec.color) td.style.color = spec.color;
        return td;
      })));
  });
  return t;
}

/* ---------------- widget registration ---------------- */

const registry = [];

/** Widgets call this at module scope; main.js mounts them into #id containers. */
export function register(id, title, tag, caption, build) {
  registry.push({ id, title, tag, caption, build });
}

export function mountAll() {
  for (const w of registry) {
    const host = document.getElementById(w.id);
    if (!host) continue;
    const body = el('div', { class: 'widget-body' });
    host.append(el('div', { class: 'widget' },
      el('div', { class: 'widget-head' },
        el('h3', { text: w.title }),
        el('span', { class: 'tag', text: w.tag })),
      body));
    if (w.caption) body.append(el('p', { class: 'widget-cap', text: w.caption }));
    try {
      w.build(body);
    } catch (err) {
      body.append(el('pre', { class: 'ascii', text: `widget "${w.id}" failed: ${err.message}` }));
      console.error(w.id, err);
    }
  }
}
