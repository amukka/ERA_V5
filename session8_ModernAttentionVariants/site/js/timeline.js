/* Renders the chronological timeline from data/mechanisms.js.
   The data file is the source of truth; this file only presents it. */

import { el, clear } from './lib.js';
import { ACTS, FAMILIES, MECHANISMS, NEXT } from '../data/mechanisms.js';

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
const fmtDate = (iso) => {
  const [y, m, d] = iso.split('-').map(Number);
  return `${d} ${MONTHS[m - 1]} ${y}`;
};

/* Assert the data really is in chronological order — a sorted list is the one
   claim this page cannot afford to get wrong, so it is checked at load. */
function assertSorted() {
  const bad = [];
  for (let i = 1; i < MECHANISMS.length; i++) {
    if (MECHANISMS[i].date < MECHANISMS[i - 1].date) {
      bad.push(`${MECHANISMS[i].name} (${MECHANISMS[i].date}) after ${MECHANISMS[i - 1].name} (${MECHANISMS[i - 1].date})`);
    }
  }
  return bad;
}

let filterFamily = 'all';
let keyOnly = false;
let expanded = new Set();

const host = document.getElementById('timeline');
const countEl = document.getElementById('tl-count');

/* ---------------- controls ---------------- */
function buildControls() {
  const bar = document.getElementById('tl-controls');
  clear(bar);

  const chip = (id, label, colour, active) => el('button', {
    class: `chip${active ? ' on' : ''}`,
    type: 'button',
    style: active && colour ? { borderColor: colour, color: colour } : {},
    onclick: () => { filterFamily = id; render(); buildControls(); },
  }, label);

  bar.append(
    el('div', { class: 'chiprow' },
      chip('all', `all ${MECHANISMS.length}`, null, filterFamily === 'all'),
      Object.entries(FAMILIES).map(([id, f]) =>
        chip(id, f.label, f.colour, filterFamily === id))),
    el('div', { class: 'chiprow' },
      el('button', {
        class: `chip${keyOnly ? ' on' : ''}`, type: 'button',
        onclick: () => { keyOnly = !keyOnly; render(); buildControls(); },
      }, keyOnly ? '★ turning points only' : '☆ turning points only'),
      el('button', {
        class: 'chip', type: 'button',
        onclick: () => {
          expanded = expanded.size ? new Set() : new Set(MECHANISMS.map((m) => m.name));
          render();
        },
      }, expanded.size ? 'collapse all' : 'expand all'),
    ),
  );
}

/* ---------------- one entry ---------------- */
function card(m, index) {
  const fam = FAMILIES[m.family];
  const open = expanded.has(m.name);

  const head = el('button', {
    class: 'tl-head', type: 'button',
    'aria-expanded': String(open),
    onclick: () => {
      if (open) expanded.delete(m.name); else expanded.add(m.name);
      render();
    },
  },
    el('span', { class: 'tl-date' },
      fmtDate(m.date),
      m.dateApprox ? el('span', { class: 'approx', text: '≈' }) : null),
    el('span', { class: 'tl-name' },
      m.isKey ? el('span', { class: 'star', text: '★ ' }) : null,
      m.name,
      m.isBaseline ? el('span', { class: 'tag-base', text: 'the baseline' }) : null),
    el('span', { class: 'tl-fam', style: { color: fam.colour, borderColor: fam.colour }, text: fam.label }),
    el('span', { class: 'tl-caret', text: open ? '−' : '+' }),
  );

  if (!open) {
    return el('article', { class: 'tl-item', style: { '--fam': fam.colour } },
      el('span', { class: 'tl-dot', style: { background: fam.colour } }),
      head,
      el('p', { class: 'tl-teaser', text: m.problem }));
  }

  return el('article', { class: 'tl-item open', style: { '--fam': fam.colour } },
    el('span', { class: 'tl-dot', style: { background: fam.colour } }),
    head,
    el('div', { class: 'tl-body' },
      el('p', { class: 'tl-who' },
        m.who, ' · ',
        el('a', { href: m.url, target: '_blank', rel: 'noopener', text: m.where })),

      el('div', { class: 'pa' },
        el('div', { class: 'pa-block problem' },
          el('h4', { text: 'the problem at that moment' }),
          el('p', { text: m.problem })),
        el('div', { class: 'pa-block answer' },
          el('h4', { text: 'what it did about it' }),
          el('p', { text: m.answer }))),

      el('div', { class: 'tradeoff' },
        el('div', { class: 'to-col pros' },
          el('h4', { text: 'what it buys' }),
          el('ul', {}, m.pros.map((p) => el('li', { text: p })))),
        el('div', { class: 'to-col cons' },
          el('h4', { text: 'what it gives up' }),
          el('ul', {}, m.cons.map((p) => el('li', { text: p }))))),

      el('div', { class: 'pickbox' },
        el('h4', { text: 'when you would actually pick it' }),
        el('p', { text: m.pick })),

      m.note ? el('p', { class: 'tl-note', html: `<b>Worth knowing.</b> ${m.note}` }) : null,
      m.warning ? el('p', { class: 'tl-warn', html: `<b>⚠ ${m.warning}</b>` }) : null,

      el('details', { class: 'verif' },
        el('summary', { text: 'date verification' }),
        el('p', { class: 'v-ok', html: `<b>Confirmed:</b> ${m.verified}` }),
        m.caveat ? el('p', { class: 'v-caveat', html: `<b>Not confirmed:</b> ${m.caveat}` }) : null),
    ));
}

/* ---------------- render ---------------- */
function render() {
  clear(host);
  const shown = MECHANISMS.filter((m) =>
    (filterFamily === 'all' || m.family === filterFamily) && (!keyOnly || m.isKey));

  countEl.textContent = shown.length === MECHANISMS.length
    ? `${MECHANISMS.length} mechanisms, oldest first`
    : `showing ${shown.length} of ${MECHANISMS.length}`;

  ACTS.forEach((act) => {
    const items = shown.filter((m) => m.act === act.id);
    if (!items.length) return;
    host.append(
      el('section', { class: 'act', id: `act-${act.id}` },
        el('div', { class: 'act-head' },
          el('span', { class: 'act-span', text: act.span }),
          el('h2', { text: act.name }),
          el('p', { class: 'act-thesis', text: act.thesis })),
        el('div', { class: 'tl-list' }, items.map((m, i) => card(m, i)))),
    );
  });

  if (!shown.length) {
    host.append(el('p', { class: 'tl-empty', text: 'Nothing matches that filter.' }));
  }
}

/* ---------------- what comes next ---------------- */
function renderNext() {
  const n = document.getElementById('next-list');
  if (!n) return;
  clear(n);
  NEXT.forEach((x, i) => n.append(
    el('div', { class: 'nx' },
      el('div', { class: 'nx-n', text: String(i + 1).padStart(2, '0') }),
      el('div', {},
        el('h3', { text: x.q }),
        el('p', { text: x.why }))),
  ));
}

/* ---------------- order check, shown on the page ---------------- */
function renderOrderCheck() {
  const n = document.getElementById('order-check');
  if (!n) return;
  const bad = assertSorted();
  const first = MECHANISMS[0];
  const last = MECHANISMS[MECHANISMS.length - 1];
  n.append(el('pre', {
    class: 'ascii',
    text: bad.length
      ? `ORDER CHECK FAILED — ${bad.length} entr${bad.length > 1 ? 'ies' : 'y'} out of sequence:\n`
        + bad.map((b) => `  · ${b}`).join('\n')
      : `order check: PASS — all ${MECHANISMS.length} entries are in ascending date order\n`
        + `earliest: ${fmtDate(first.date)}  ${first.name}\n`
        + `latest:   ${fmtDate(last.date)}  ${last.name}\n`
        + `span:     ${(new Date(last.date) - new Date(first.date)) / 31557600000 | 0} years\n`
        + `dates flagged approximate: `
        + MECHANISMS.filter((m) => m.dateApprox).map((m) => m.name).join(', '),
  }));
}

buildControls();
render();
renderNext();
renderOrderCheck();
