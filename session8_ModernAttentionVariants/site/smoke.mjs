/* Headless smoke test for the Session 8 site.
   Loads index.html in jsdom, mounts every registered widget, then drives every
   control (steps, segmented buttons, toggles, sliders, buttons) and fails on any
   console error, uncaught exception, or widget that renders an empty stage.

   Run from the session root:
     npm test
   jsdom lives in the session-root devDependencies, not in site/, so the
   published directory stays free of node_modules.
*/

import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { dirname, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
// Resolve from the session root (one level up) where jsdom is installed.
const require = createRequire(resolve(here, '..', 'package.json'));
const { JSDOM, VirtualConsole } = require('jsdom');

const problems = [];
const vc = new VirtualConsole();
vc.on('jsdomError', (e) => problems.push(`jsdomError: ${e.message}`));
vc.on('error', (...a) => problems.push(`console.error: ${a.join(' ')}`));

const dom = new JSDOM(readFileSync(resolve(here, 'session8.html'), 'utf8'), {
  url: pathToFileURL(resolve(here, 'session8.html')).href,
  runScripts: 'outside-only',
  virtualConsole: vc,
  pretendToBeVisual: true,
});

// Expose the jsdom globals the widget modules touch.
const g = dom.window;
globalThis.window = g;
globalThis.document = g.document;
globalThis.Node = g.Node;
globalThis.HTMLElement = g.HTMLElement;

process.on('uncaughtException', (e) => problems.push(`uncaught: ${e.stack}`));

const lib = await import(pathToFileURL(resolve(here, 'js/lib.js')).href);
await import(pathToFileURL(resolve(here, 'js/widgets/s02-walkthrough.js')).href);
await import(pathToFileURL(resolve(here, 'js/widgets/s03-bills.js')).href);
await import(pathToFileURL(resolve(here, 'js/widgets/s04-softmax-off.js')).href);
await import(pathToFileURL(resolve(here, 'js/widgets/s05-add-only-state.js')).href);
await import(pathToFileURL(resolve(here, 'js/widgets/s06-delta-rule.js')).href);
await import(pathToFileURL(resolve(here, 'js/widgets/s07-topk.js')).href);
await import(pathToFileURL(resolve(here, 'js/widgets/s08-rope.js')).href);
await import(pathToFileURL(resolve(here, 'js/widgets/s09-drope.js')).href);
await import(pathToFileURL(resolve(here, 'js/widgets/s10-cache-bill.js')).href);
await import(pathToFileURL(resolve(here, 'js/widgets/s11-gqa.js')).href);
await import(pathToFileURL(resolve(here, 'js/widgets/s12-compression.js')).href);
await import(pathToFileURL(resolve(here, 'js/widgets/s13-schedule.js')).href);
await import(pathToFileURL(resolve(here, 'js/widgets/s14-memory-stream.js')).href);
await import(pathToFileURL(resolve(here, 'js/widgets/s15-readiness.js')).href);
await import(pathToFileURL(resolve(here, 'js/widgets/s16-v5-board.js')).href);

lib.mountAll();

/* ---- every container in index.html must have received a widget ---- */
const containers = [...g.document.querySelectorAll('div[id^="w-s"]')];
if (!containers.length) problems.push('no widget containers found in index.html');
for (const c of containers) {
  if (!c.querySelector('.widget-body')) problems.push(`${c.id}: never mounted`);
  const failed = c.querySelector('.widget-body > pre.ascii');
  if (failed && /widget ".*" failed/.test(failed.textContent)) {
    problems.push(`${c.id}: ${failed.textContent}`);
  }
}

/* ---- drive every control and assert the stage still renders ----
   Controls are re-queried after each step change, because a step can introduce
   its own controls (the key-token selector on §2 step 2, for example) that do
   not exist in the DOM while another step is showing. */
let interactions = 0;

for (const c of containers) {
  const body = c.querySelector('.widget-body');
  const stepCount = body.querySelectorAll('.steps button').length;

  // Visit every step (or just once, for widgets with no step sequencer).
  for (let s = 0; s < Math.max(1, stepCount); s++) {
    if (stepCount) {
      body.querySelectorAll('.steps button')[s].click();
      interactions++;
      assertStage(c);
    }
    driveControlsOnce(c, body, s);
  }
}

/** Drives every control currently in the DOM, re-selecting the step each time
 *  so that clicking a control which re-renders the stage cannot invalidate the
 *  node list we are iterating. */
function driveControlsOnce(c, body, step) {
  const reselect = () => {
    const steps = body.querySelectorAll('.steps button');
    if (steps.length) steps[step].click();
  };

  // Segmented + plain buttons, addressed by index because the stage re-renders.
  const segCount = body.querySelectorAll('.seg button').length;
  for (let i = 0; i < segCount; i++) {
    const b = body.querySelectorAll('.seg button')[i];
    if (!b || b.disabled) continue;
    b.click(); interactions++; assertStage(c);
  }
  reselect();

  const btnCount = body.querySelectorAll('button.btn').length;
  for (let i = 0; i < btnCount; i++) {
    const b = body.querySelectorAll('button.btn')[i];
    if (!b || b.disabled) continue;
    b.click(); interactions++; assertStage(c);
  }
  reselect();

  for (const t of body.querySelectorAll('.switch input')) {
    for (const v of [false, true]) {
      t.checked = v;
      t.dispatchEvent(new g.Event('change'));
      interactions++;
      assertStage(c);
    }
  }

  const rangeCount = body.querySelectorAll('input[type=range]').length;
  for (let i = 0; i < rangeCount; i++) {
    const r = body.querySelectorAll('input[type=range]')[i];
    if (!r) continue;
    for (const v of [r.min, r.max, String(Math.floor((+r.min + +r.max) / 2))]) {
      r.value = v;
      r.dispatchEvent(new g.Event('input'));
      interactions++;
      assertStage(c);
    }
  }
}

function assertStage(c) {
  const stage = c.querySelector('.stage');
  if (!stage) return;
  if (!stage.textContent.trim()) problems.push(`${c.id}: stage rendered empty`);
  if (/NaN|Infinity|undefined/.test(stage.textContent)) {
    const bad = stage.textContent.match(/.{0,40}(NaN|Infinity|undefined).{0,40}/)[0];
    problems.push(`${c.id}: bad number in stage — "${bad.trim()}"`);
  }
  const ro = c.querySelector('.readouts');
  if (ro && /NaN|undefined/.test(ro.textContent)) {
    problems.push(`${c.id}: bad number in readouts`);
  }
}

/* ---- report ---- */
const unique = [...new Set(problems)];
console.log(`widgets mounted : ${containers.length}`);
console.log(`interactions    : ${interactions}`);
if (unique.length) {
  console.log(`\nFAIL — ${unique.length} problem(s):`);
  unique.forEach((p) => console.log('  · ' + p));
  process.exit(1);
}
console.log('\nPASS — no console errors, no empty stages, no NaN/undefined output.');
