/* app.js — live-training controllers + canvas rendering for the four proofs.
 * Depends on NN (nn.js) and DATA (data.js). Browser only. */
(function () {
  "use strict";
  const NN = window.NN, DATA = window.DATA;

  // ---- palette (dark-surface data-viz colors) ----
  const C0 = [57, 135, 229];    // blue  — class 0 (inner ring / moon A)
  const C1 = [235, 104, 52];    // orange — class 1 (outer ring / moon B)
  const NEUTRAL = [26, 26, 25]; // dark surface midpoint
  const CAT_COLORS = {
    animal: [57, 135, 229], fruit: [27, 175, 122], verb: [235, 104, 52],
    the: [144, 133, 233], end: [137, 135, 129],
  };

  function lerp(a, b, t) { return a + (b - a) * t; }
  function mix(a, b, t) { return [lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t)]; }
  function rgb(c, a) { return `rgba(${c[0]|0},${c[1]|0},${c[2]|0},${a==null?1:a})`; }

  // Map from data-space to a fitted [xmin..xmax]x[ymin..ymax] with padding.
  function fitRange(X, pad) {
    pad = pad == null ? 0.15 : pad;
    let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
    for (const p of X) { xmin = Math.min(xmin, p[0]); xmax = Math.max(xmax, p[0]); ymin = Math.min(ymin, p[1]); ymax = Math.max(ymax, p[1]); }
    const dx = (xmax - xmin) * pad, dy = (ymax - ymin) * pad;
    // keep square-ish aspect by using the larger span
    const cx = (xmin + xmax) / 2, cy = (ymin + ymax) / 2;
    let span = Math.max(xmax - xmin + 2 * dx, ymax - ymin + 2 * dy) / 2;
    return { xmin: cx - span, xmax: cx + span, ymin: cy - span, ymax: cy + span };
  }

  // set up a canvas with device-pixel-ratio scaling. returns {ctx, w, h}
  function setupCanvas(canvas) {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const rect = canvas.getBoundingClientRect();
    const w = rect.width || canvas.clientWidth || 300;
    const h = rect.height || canvas.clientHeight || 300;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, w, h };
  }

  // Render a probability field (decision regions) + data points.
  function drawBoundary(canvas, model, X, Y, range, opts) {
    opts = opts || {};
    const { ctx, w, h } = canvas._env || (canvas._env = setupCanvas(canvas));
    const R = range;
    const toPx = (x, y) => [ (x - R.xmin) / (R.xmax - R.xmin) * w, h - (y - R.ymin) / (R.ymax - R.ymin) * h ];
    // background field, coarse grid
    const step = opts.step || 6;
    for (let px = 0; px < w; px += step) {
      for (let py = 0; py < h; py += step) {
        const x = R.xmin + (px + step / 2) / w * (R.xmax - R.xmin);
        const y = R.ymin + (h - (py + step / 2)) / h * (R.ymax - R.ymin);
        const p = model.predictProb([x, y])[0];
        // diverging blue<->orange through dark neutral
        let col;
        if (p < 0.5) col = mix(NEUTRAL, C0, (0.5 - p) * 2 * 0.85);
        else col = mix(NEUTRAL, C1, (p - 0.5) * 2 * 0.85);
        ctx.fillStyle = rgb(col, 1);
        ctx.fillRect(px, py, step, step);
      }
    }
    // decision boundary contour (p≈0.5): draw where neighbors cross — light stroke
    // (approximate: overlay by sampling handled via field; skip explicit contour)
    // data points
    const r = opts.pointR || 3;
    for (let i = 0; i < X.length; i++) {
      const [cx, cy] = toPx(X[i][0], X[i][1]);
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.fillStyle = Y[i] === 0 ? rgb(C0) : rgb(C1);
      ctx.strokeStyle = "rgba(255,255,255,0.75)";
      ctx.lineWidth = 1;
      ctx.fill(); ctx.stroke();
    }
  }

  // A frame-chunked trainer. cfg: {model, X, Y, epochs, lr, batch, l2,
  //   epochsPerFrame, onFrame(progress), onDone()}
  function trainLive(cfg) {
    const n = cfg.X.length;
    const idx = [...Array(n).keys()];
    const batch = Math.min(cfg.batch || 32, n);
    let done = 0;
    let cancelled = false;
    function shuffle() { for (let i = n - 1; i > 0; i--) { const j = (Math.random() * (i + 1)) | 0; const t = idx[i]; idx[i] = idx[j]; idx[j] = t; } }
    function frame() {
      if (cancelled) return;
      const per = cfg.epochsPerFrame || 4;
      for (let e = 0; e < per && done < cfg.epochs; e++, done++) {
        shuffle();
        for (let b = 0; b < n; b += batch) {
          const bi = idx.slice(b, b + batch);
          cfg.model.trainBatch(bi.map((k) => cfg.X[k]), bi.map((k) => cfg.Y[k]), cfg.lr, cfg.l2 || 0);
        }
      }
      const prog = done / cfg.epochs;
      if (cfg.onFrame) cfg.onFrame(prog);
      if (done < cfg.epochs) requestAnimationFrame(frame);
      else if (cfg.onDone) cfg.onDone();
    }
    requestAnimationFrame(frame);
    return { cancel() { cancelled = true; } };
  }

  const $ = (s, r) => (r || document).querySelector(s);
  const fmt = (v) => (v * 100).toFixed(1);

  // ========================================================================
  // S1-1 : rings — linear vs ReLU
  // ========================================================================
  function initS1() {
    const data = DATA.rings(300, 11);
    const range = fitRange(data.X);
    const cvLin = $("#s1-lin"), cvRelu = $("#s1-relu");
    const accLin = $("#s1-acc-lin"), accRelu = $("#s1-acc-relu");
    const btn = $("#s1-run"), status = $("#s1-status");
    let running = null;

    function fresh() {
      const lin = new NN.MLP({ sizes: [2, 1], acts: ["linear"], head: "sigmoid", seed: 2 });
      const relu = new NN.MLP({ sizes: [2, 16, 1], acts: ["relu", "linear"], head: "sigmoid", seed: 2 });
      return { lin, relu };
    }
    function render(m) {
      drawBoundary(cvLin, m.lin, data.X, data.Y, range);
      drawBoundary(cvRelu, m.relu, data.X, data.Y, range);
      accLin.textContent = fmt(m.lin.accuracyBinary(data.X, data.Y)) + "%";
      accRelu.textContent = fmt(m.relu.accuracyBinary(data.X, data.Y)) + "%";
    }
    function run() {
      if (running) running.forEach((r) => r.cancel());
      const m = fresh();
      render(m);
      btn.disabled = true; status.textContent = "training…";
      let a, b;
      a = trainLive({ model: m.lin, X: data.X, Y: data.Y, epochs: 400, lr: 0.3, batch: 32, epochsPerFrame: 8,
        onFrame: () => { drawBoundary(cvLin, m.lin, data.X, data.Y, range); accLin.textContent = fmt(m.lin.accuracyBinary(data.X, data.Y)) + "%"; } });
      b = trainLive({ model: m.relu, X: data.X, Y: data.Y, epochs: 500, lr: 0.15, batch: 32, epochsPerFrame: 6,
        onFrame: () => { drawBoundary(cvRelu, m.relu, data.X, data.Y, range); accRelu.textContent = fmt(m.relu.accuracyBinary(data.X, data.Y)) + "%"; },
        onDone: () => { btn.disabled = false; status.textContent = "done — only the activation changed."; } });
      running = [a, b];
    }
    render(fresh());
    btn.addEventListener("click", run);
    return run;
  }

  // ========================================================================
  // S1-2 : depth without nonlinearity
  // ========================================================================
  function initS2() {
    const data = DATA.rings(300, 11);
    const range = fitRange(data.X);
    const cv1 = $("#s2-l1"), cv5 = $("#s2-l5"), cvR = $("#s2-r5");
    const a1 = $("#s2-acc-l1"), a5 = $("#s2-acc-l5"), aR = $("#s2-acc-r5");
    const btn = $("#s2-run"), status = $("#s2-status"), collapse = $("#s2-collapse");
    let running = null;

    function fresh() {
      return {
        l1: new NN.MLP({ sizes: [2, 1], acts: ["linear"], head: "sigmoid", seed: 5 }),
        l5: new NN.MLP({ sizes: [2, 8, 8, 8, 8, 1], acts: ["linear", "linear", "linear", "linear", "linear"], head: "sigmoid", seed: 5 }),
        r5: new NN.MLP({ sizes: [2, 8, 8, 8, 8, 1], acts: ["relu", "relu", "relu", "relu", "linear"], head: "sigmoid", seed: 5 }),
      };
    }
    function drawAll(m) {
      drawBoundary(cv1, m.l1, data.X, data.Y, range);
      drawBoundary(cv5, m.l5, data.X, data.Y, range);
      drawBoundary(cvR, m.r5, data.X, data.Y, range);
      a1.textContent = fmt(m.l1.accuracyBinary(data.X, data.Y)) + "%";
      a5.textContent = fmt(m.l5.accuracyBinary(data.X, data.Y)) + "%";
      aR.textContent = fmt(m.r5.accuracyBinary(data.X, data.Y)) + "%";
    }
    function showCollapse(m) {
      const c = m.l5.collapseLinear();
      const w = Array.from(c.W).map((v) => v.toFixed(3));
      const b = Array.from(c.b).map((v) => v.toFixed(3));
      collapse.innerHTML =
        `<div class="collapse-eq"><span class="ce-label">W₁·W₂·W₃·W₄·W₅ &nbsp;=&nbsp;</span>` +
        `<span class="ce-mat">[ ${w.join(",&nbsp; ")} ]</span>` +
        `<span class="ce-note">a single 2×1 matrix</span></div>` +
        `<div class="collapse-eq"><span class="ce-label">effective bias =</span> <span class="ce-mat">[ ${b.join(", ")} ]</span></div>` +
        `<p class="ce-caption">Five stacked linear layers (8·8·8·8 hidden units, ~200 weights) multiply out to <strong>one</strong> 2→1 linear map. That is why the 5-linear boundary is the same straight line as the 1-linear one — extra depth added <em>zero</em> new expressive power.</p>`;
    }
    function run() {
      if (running) running.forEach((r) => r.cancel());
      const m = fresh(); drawAll(m); collapse.innerHTML = "";
      btn.disabled = true; status.textContent = "training…";
      const r1 = trainLive({ model: m.l1, X: data.X, Y: data.Y, epochs: 400, lr: 0.3, batch: 32, epochsPerFrame: 8,
        onFrame: () => { drawBoundary(cv1, m.l1, data.X, data.Y, range); a1.textContent = fmt(m.l1.accuracyBinary(data.X, data.Y)) + "%"; } });
      const r2 = trainLive({ model: m.l5, X: data.X, Y: data.Y, epochs: 400, lr: 0.05, batch: 32, epochsPerFrame: 8,
        onFrame: () => { drawBoundary(cv5, m.l5, data.X, data.Y, range); a5.textContent = fmt(m.l5.accuracyBinary(data.X, data.Y)) + "%"; } });
      const r3 = trainLive({ model: m.r5, X: data.X, Y: data.Y, epochs: 600, lr: 0.1, batch: 32, epochsPerFrame: 6,
        onFrame: () => { drawBoundary(cvR, m.r5, data.X, data.Y, range); aR.textContent = fmt(m.r5.accuracyBinary(data.X, data.Y)) + "%"; },
        onDone: () => { btn.disabled = false; status.textContent = "done — ReLU broke the tie."; showCollapse(m); } });
      running = [r1, r2, r3];
    }
    drawAll(fresh());
    btn.addEventListener("click", run);
    return run;
  }

  // ========================================================================
  // S1-3 : embeddings cluster from next-token only
  // ========================================================================
  function initS3() {
    const pairs = DATA.grammarPairs(400, 31);
    const V = DATA.VOCAB.length;
    const cv = $("#s3-canvas");
    const btn = $("#s3-run"), status = $("#s3-status"), nnEl = $("#s3-nn");
    let model = null, running = null;

    function fresh() { return new NN.EmbModel(V, 8, 7); }
    function nearestSame(m) {
      let same = 0, total = 0;
      for (let i = 0; i < V; i++) {
        const cat = DATA.categoryOf(i);
        // singleton categories can't have a same-cat neighbor; skip from denom
        let members = 0; for (let j = 0; j < V; j++) if (DATA.categoryOf(j) === cat) members++;
        if (members < 2) continue;
        total++;
        let best = -1, bd = Infinity;
        for (let j = 0; j < V; j++) if (j !== i) { let d = 0; for (let k = 0; k < 8; k++) { const dd = m.E[i][k] - m.E[j][k]; d += dd * dd; } if (d < bd) { bd = d; best = j; } }
        if (DATA.categoryOf(best) === cat) same++;
      }
      return { same, total };
    }
    function renderScatter(m) {
      const proj = NN.pca2(m.E);
      const range = fitRange(proj, 0.25);
      const { ctx, w, h } = cv._env || (cv._env = setupCanvas(cv));
      ctx.clearRect(0, 0, w, h);
      // subtle grid
      ctx.strokeStyle = "rgba(255,255,255,0.05)"; ctx.lineWidth = 1;
      for (let g = 0; g <= 4; g++) { const x = (g / 4) * w; ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke(); const y = (g / 4) * h; ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke(); }
      const toPx = (x, y) => [ (x - range.xmin) / (range.xmax - range.xmin) * w, h - (y - range.ymin) / (range.ymax - range.ymin) * h ];
      for (let i = 0; i < V; i++) {
        const [cx, cy] = toPx(proj[i][0], proj[i][1]);
        const col = CAT_COLORS[DATA.categoryOf(i)] || [150, 150, 150];
        ctx.beginPath(); ctx.arc(cx, cy, 6, 0, Math.PI * 2);
        ctx.fillStyle = rgb(col); ctx.strokeStyle = "rgba(255,255,255,0.6)"; ctx.lineWidth = 1.5; ctx.fill(); ctx.stroke();
        ctx.fillStyle = "rgba(255,255,255,0.92)"; ctx.font = "600 13px system-ui, sans-serif"; ctx.textBaseline = "middle";
        const label = DATA.VOCAB[i] === "." ? "·(end)" : DATA.VOCAB[i];
        ctx.fillText(label, cx + 10, cy);
      }
      const ns = nearestSame(m);
      nnEl.textContent = `${ns.same} / ${ns.total}`;
    }
    function run() {
      if (running) running.cancel();
      model = fresh(); renderScatter(model);
      btn.disabled = true; status.textContent = "training on next-token prediction…";
      let epoch = 0; const EPOCHS = 300;
      let cancelled = false;
      function frame() {
        if (cancelled) return;
        for (let e = 0; e < 6 && epoch < EPOCHS; e++, epoch++) {
          for (let b = 0; b < pairs.length; b += 32) model.trainBatch(pairs.slice(b, b + 32), 0.5);
        }
        renderScatter(model);
        if (epoch < EPOCHS) requestAnimationFrame(frame);
        else { btn.disabled = false; status.textContent = "done — same-category tokens clustered, though similarity was never supplied."; }
      }
      running = { cancel() { cancelled = true; } };
      requestAnimationFrame(frame);
    }
    model = fresh(); renderScatter(model);
    btn.addEventListener("click", run);
    return run;
  }

  // ========================================================================
  // S1-4 : memorization vs generalization vs data size
  // ========================================================================
  function initS4() {
    const SIZES = [20, 200, 1000];
    const test = DATA.moons(1000, 999, 0.28, 0.0);
    const testRange = fitRange(test.X, 0.12);
    const cards = SIZES.map((N) => ({
      N,
      cv: $(`#s4-cv-${N}`), tr: $(`#s4-tr-${N}`), te: $(`#s4-te-${N}`), gap: $(`#s4-gap-${N}`),
      data: DATA.moons(N, N, 0.28, 0.0),
    }));
    const btn = $("#s4-run"), status = $("#s4-status"), bars = $("#s4-bars");
    let running = null;

    function freshNet() { return new NN.MLP({ sizes: [2, 48, 48, 1], acts: ["relu", "relu", "linear"], head: "sigmoid", seed: 4 }); }
    function renderCard(c, net) {
      drawBoundary(c.cv, net, c.data.X, c.data.Y, testRange, { pointR: c.N > 200 ? 2 : 3, step: 7 });
    }
    function updateStats(c, net) {
      const trL = net.lossBinary(c.data.X, c.data.Y);
      const teL = net.lossBinary(test.X, test.Y);
      const trA = net.accuracyBinary(c.data.X, c.data.Y);
      const teA = net.accuracyBinary(test.X, test.Y);
      c.tr.textContent = fmt(trA) + "%";
      c.te.textContent = fmt(teA) + "%";
      c.gap.textContent = (teL - trL).toFixed(2);
      c._trL = trL; c._teL = teL;
    }
    function renderBars() {
      // gap = test loss - train loss, per size
      const maxGap = Math.max(...cards.map((c) => (c._teL || 0) - (c._trL || 0)), 0.1);
      bars.innerHTML = cards.map((c) => {
        const g = Math.max(0, (c._teL || 0) - (c._trL || 0));
        const pct = (g / maxGap) * 100;
        return `<div class="gapbar-row"><span class="gapbar-label">N=${c.N}</span>` +
          `<span class="gapbar-track"><span class="gapbar-fill" style="width:${pct.toFixed(0)}%"></span></span>` +
          `<span class="gapbar-val">${g.toFixed(2)}</span></div>`;
      }).join("");
    }
    function run() {
      if (running) running.forEach((r) => r.cancel());
      const nets = cards.map(() => freshNet());
      cards.forEach((c, i) => { renderCard(c, nets[i]); updateStats(c, nets[i]); });
      renderBars();
      btn.disabled = true; status.textContent = "training an over-parameterized net at each data size…";
      let doneCount = 0;
      running = cards.map((c, i) => {
        const epochs = c.N <= 20 ? 1600 : c.N <= 200 ? 700 : 220;
        const epf = c.N <= 20 ? 40 : c.N <= 200 ? 16 : 5;
        return trainLive({ model: nets[i], X: c.data.X, Y: c.data.Y, epochs, lr: 0.12, batch: Math.min(16, c.N), epochsPerFrame: epf,
          onFrame: () => { renderCard(c, nets[i]); updateStats(c, nets[i]); renderBars(); },
          onDone: () => { doneCount++; if (doneCount === cards.length) { btn.disabled = false; status.textContent = "done — the train→test gap is huge at N=20 and closes as data grows."; } } });
      });
    }
    const nets0 = cards.map(() => freshNet());
    cards.forEach((c, i) => { renderCard(c, nets0[i]); updateStats(c, nets0[i]); });
    renderBars();
    btn.addEventListener("click", run);
    return run;
  }

  // ---- boot ----
  document.addEventListener("DOMContentLoaded", function () {
    const runners = { s1: initS1(), s2: initS2(), s3: initS3(), s4: initS4() };
    // auto-run a section the first time it scrolls into view
    const seen = {};
    const io = new IntersectionObserver((entries) => {
      entries.forEach((e) => {
        if (e.isIntersecting) {
          const id = e.target.getAttribute("data-run");
          if (id && !seen[id]) { seen[id] = true; runners[id](); }
        }
      });
    }, { threshold: 0.35 });
    document.querySelectorAll("[data-run]").forEach((el) => io.observe(el));
  });
})();
