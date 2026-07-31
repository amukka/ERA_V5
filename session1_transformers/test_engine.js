/* Headless verification of the four claims. Run: node test_engine.js
 * Dev-only — not referenced by the site. Safe to delete before deploying. */
const NN = require("./nn.js");
const DATA = require("./data.js");

function trainMLP(model, X, Y, epochs, lr, batch, l2) {
  const n = X.length;
  const idx = [...Array(n).keys()];
  for (let e = 0; e < epochs; e++) {
    for (let i = n - 1; i > 0; i--) { const j = (Math.random() * (i + 1)) | 0; [idx[i], idx[j]] = [idx[j], idx[i]]; }
    for (let b = 0; b < n; b += batch) {
      const bi = idx.slice(b, b + batch);
      model.trainBatch(bi.map((k) => X[k]), bi.map((k) => Y[k]), lr, l2 || 0);
    }
  }
}

console.log("=== S1-1: rings, linear vs ReLU ===");
{
  const { X, Y } = DATA.rings(300, 11);
  const lin = new NN.MLP({ sizes: [2, 1], acts: ["linear"], head: "sigmoid", seed: 2 });
  const relu = new NN.MLP({ sizes: [2, 16, 1], acts: ["relu", "linear"], head: "sigmoid", seed: 2 });
  trainMLP(lin, X, Y, 400, 0.3, 32);
  trainMLP(relu, X, Y, 500, 0.15, 32);
  console.log("  linear acc:", (lin.accuracyBinary(X, Y) * 100).toFixed(1), "%   (claim: stuck ~50-55%)");
  console.log("  relu   acc:", (relu.accuracyBinary(X, Y) * 100).toFixed(1), "%   (claim: ~99%)");
}

console.log("=== S1-2: 1 linear vs 5 linear vs 5+ReLU ===");
{
  const { X, Y } = DATA.rings(300, 11);
  const l1 = new NN.MLP({ sizes: [2, 1], acts: ["linear"], head: "sigmoid", seed: 5 });
  const l5 = new NN.MLP({ sizes: [2, 8, 8, 8, 8, 1], acts: ["linear", "linear", "linear", "linear", "linear"], head: "sigmoid", seed: 5 });
  const r5 = new NN.MLP({ sizes: [2, 8, 8, 8, 8, 1], acts: ["relu", "relu", "relu", "relu", "linear"], head: "sigmoid", seed: 5 });
  trainMLP(l1, X, Y, 400, 0.3, 32);
  trainMLP(l5, X, Y, 400, 0.05, 32);
  trainMLP(r5, X, Y, 600, 0.1, 32);
  console.log("  1 linear acc:", (l1.accuracyBinary(X, Y) * 100).toFixed(1), "%");
  console.log("  5 linear acc:", (l5.accuracyBinary(X, Y) * 100).toFixed(1), "%  (≈ 1-linear: depth added nothing)");
  console.log("  5 + ReLU acc:", (r5.accuracyBinary(X, Y) * 100).toFixed(1), "%  (ReLU breaks the tie)");
  const c = l5.collapseLinear();
  console.log("  5 linear matrices multiplied -> single 2x1 map W:", Array.from(c.W).map((v) => v.toFixed(3)), " b:", Array.from(c.b).map((v) => v.toFixed(3)));
}

console.log("=== S1-3: embedding clustering (nearest-neighbor same-category) ===");
{
  const pairs = DATA.grammarPairs(400, 31);
  const model = new NN.EmbModel(DATA.VOCAB.length, 8, 7);
  for (let e = 0; e < 300; e++) for (let b = 0; b < pairs.length; b += 32) model.trainBatch(pairs.slice(b, b + 32), 0.5);
  let same = 0, total = 0;
  for (let i = 0; i < DATA.VOCAB.length; i++) {
    const cat = DATA.categoryOf(i);
    let members = 0; for (let j = 0; j < DATA.VOCAB.length; j++) if (DATA.categoryOf(j) === cat) members++;
    if (members < 2) continue; total++;
    let best = -1, bd = Infinity;
    for (let j = 0; j < DATA.VOCAB.length; j++) if (j !== i) { let d = 0; for (let k = 0; k < 8; k++) { const dd = model.E[i][k] - model.E[j][k]; d += dd * dd; } if (d < bd) { bd = d; best = j; } }
    if (DATA.categoryOf(best) === cat) same++;
  }
  console.log("  nearest-neighbor same-category rate:", same, "/", total, " (singletons 'the'/'.' excluded)");
}

console.log("=== S1-4: generalization gap vs data size ===");
{
  const test = DATA.moons(1000, 999, 0.28, 0.0);
  for (const N of [20, 200, 1000]) {
    const tr = DATA.moons(N, N, 0.28, 0.0);
    const net = new NN.MLP({ sizes: [2, 48, 48, 1], acts: ["relu", "relu", "linear"], head: "sigmoid", seed: 4 });
    trainMLP(net, tr.X, tr.Y, N <= 20 ? 1600 : N <= 200 ? 700 : 220, 0.12, Math.min(16, N));
    const trA = net.accuracyBinary(tr.X, tr.Y), teA = net.accuracyBinary(test.X, test.Y);
    const trL = net.lossBinary(tr.X, tr.Y), teL = net.lossBinary(test.X, test.Y);
    console.log(`  N=${String(N).padStart(4)}  train ${(trA * 100).toFixed(1)}% (loss ${trL.toFixed(3)})  test ${(teA * 100).toFixed(1)}% (loss ${teL.toFixed(3)})  gap ${(teL - trL).toFixed(2)}`);
  }
}
