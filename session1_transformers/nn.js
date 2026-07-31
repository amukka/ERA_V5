/* nn.js — tiny neural-net engine (shared by browser + node).
 * Pure JS, manual backprop. No dependencies.
 * Supports: dense MLP with linear / relu activations, sigmoid+BCE or
 * softmax+CE heads, an embedding->softmax next-token model, and PCA.
 */
(function (root) {
  "use strict";

  // ---------- RNG (seedable, so results are reproducible) ----------
  function makeRNG(seed) {
    let s = seed >>> 0;
    return function () {
      // mulberry32
      s |= 0; s = (s + 0x6D2B79F5) | 0;
      let t = Math.imul(s ^ (s >>> 15), 1 | s);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function randn(rng) {
    // Box-Muller
    let u = 0, v = 0;
    while (u === 0) u = rng();
    while (v === 0) v = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  // ---------- MLP ----------
  // spec: { sizes:[in, h1, ..., out], acts:['relu'|'linear', ...] one per layer,
  //         head:'sigmoid'|'softmax', seed }
  // acts has length = sizes.length-1; the head activation is applied on the
  // last layer's pre-activation regardless (sigmoid for 1 output BCE, softmax
  // for multi-class CE). Intermediate acts are 'relu' or 'linear'.
  function MLP(spec) {
    const rng = makeRNG(spec.seed || 1);
    this.sizes = spec.sizes.slice();
    this.acts = spec.acts.slice();
    this.head = spec.head || "sigmoid";
    this.L = this.sizes.length - 1;
    this.W = [];
    this.b = [];
    for (let l = 0; l < this.L; l++) {
      const nin = this.sizes[l], nout = this.sizes[l + 1];
      // He-ish init
      const scale = Math.sqrt(2 / nin);
      const W = new Float64Array(nin * nout);
      for (let i = 0; i < W.length; i++) W[i] = randn(rng) * scale;
      this.W.push(W);
      this.b.push(new Float64Array(nout));
    }
  }

  // forward one sample x (Array length sizes[0]); returns {out, cache}
  MLP.prototype.forwardOne = function (x) {
    const a = [Float64Array.from(x)];
    const z = [null];
    for (let l = 0; l < this.L; l++) {
      const nin = this.sizes[l], nout = this.sizes[l + 1];
      const W = this.W[l], b = this.b[l], ain = a[l];
      const zl = new Float64Array(nout);
      for (let j = 0; j < nout; j++) {
        let s = b[j];
        for (let i = 0; i < nin; i++) s += ain[i] * W[i * nout + j];
        zl[j] = s;
      }
      let al;
      const isLast = l === this.L - 1;
      if (isLast) {
        if (this.head === "sigmoid") {
          al = new Float64Array(nout);
          for (let j = 0; j < nout; j++) al[j] = 1 / (1 + Math.exp(-zl[j]));
        } else { // softmax
          al = softmaxArr(zl);
        }
      } else if (this.acts[l] === "relu") {
        al = new Float64Array(nout);
        for (let j = 0; j < nout; j++) al[j] = zl[j] > 0 ? zl[j] : 0;
      } else { // linear
        al = zl.slice();
      }
      z.push(zl);
      a.push(al);
    }
    return { out: a[this.L], a: a, z: z };
  };

  // Train on a mini-batch. X: Array of samples, Y: Array of targets.
  // For sigmoid head, Y is Array of 0/1. For softmax, Y is Array of class idx.
  // Returns average loss over batch. Accumulates grads and applies SGD.
  MLP.prototype.trainBatch = function (X, Y, lr, l2) {
    const N = X.length;
    l2 = l2 || 0;
    const gW = this.W.map((w) => new Float64Array(w.length));
    const gb = this.b.map((b) => new Float64Array(b.length));
    let loss = 0;
    for (let n = 0; n < N; n++) {
      const fc = this.forwardOne(X[n]);
      const a = fc.a, z = fc.z, out = fc.out;
      // output delta
      let delta;
      if (this.head === "sigmoid") {
        const p = Math.min(1 - 1e-7, Math.max(1e-7, out[0]));
        const y = Y[n];
        loss += -(y * Math.log(p) + (1 - y) * Math.log(1 - p));
        delta = new Float64Array(1);
        delta[0] = p - y; // dL/dz for sigmoid+BCE
      } else {
        const y = Y[n];
        loss += -Math.log(Math.min(1, Math.max(1e-12, out[y])));
        delta = new Float64Array(out.length);
        for (let j = 0; j < out.length; j++) delta[j] = out[j] - (j === y ? 1 : 0);
      }
      // backprop through layers
      for (let l = this.L - 1; l >= 0; l--) {
        const nin = this.sizes[l], nout = this.sizes[l + 1];
        const ain = a[l];
        const W = this.W[l];
        const gWl = gW[l], gbl = gb[l];
        for (let j = 0; j < nout; j++) {
          const d = delta[j];
          gbl[j] += d;
          for (let i = 0; i < nin; i++) gWl[i * nout + j] += ain[i] * d;
        }
        if (l > 0) {
          const nprev = nin;
          const newDelta = new Float64Array(nprev);
          for (let i = 0; i < nprev; i++) {
            let s = 0;
            for (let j = 0; j < nout; j++) s += W[i * nout + j] * delta[j];
            // apply activation derivative of layer l-1
            if (this.acts[l - 1] === "relu") s = z[l][i] > 0 ? s : 0;
            newDelta[i] = s;
          }
          delta = newDelta;
        }
      }
    }
    // SGD update
    for (let l = 0; l < this.L; l++) {
      const W = this.W[l], b = this.b[l], gWl = gW[l], gbl = gb[l];
      for (let k = 0; k < W.length; k++) W[k] -= lr * (gWl[k] / N + l2 * W[k]);
      for (let k = 0; k < b.length; k++) b[k] -= lr * (gbl[k] / N);
    }
    return loss / N;
  };

  MLP.prototype.predictProb = function (x) {
    return this.forwardOne(x).out;
  };

  // accuracy over a dataset (sigmoid head, threshold 0.5)
  MLP.prototype.accuracyBinary = function (X, Y) {
    let c = 0;
    for (let n = 0; n < X.length; n++) {
      const p = this.forwardOne(X[n]).out[0];
      if ((p >= 0.5 ? 1 : 0) === Y[n]) c++;
    }
    return c / X.length;
  };

  MLP.prototype.lossBinary = function (X, Y) {
    let loss = 0;
    for (let n = 0; n < X.length; n++) {
      const p = Math.min(1 - 1e-7, Math.max(1e-7, this.forwardOne(X[n]).out[0]));
      const y = Y[n];
      loss += -(y * Math.log(p) + (1 - y) * Math.log(1 - p));
    }
    return loss / X.length;
  };

  // Collapse product of all weight matrices (linear net) -> single [in x out]
  // matrix and bias. Only meaningful when every intermediate act is 'linear'.
  MLP.prototype.collapseLinear = function () {
    // Effective map: a_out = a_in * (W1 W2 ... WL) + (b1 W2..WL + ... + bL)
    let Wprod = null; // [in x cur]
    let bAcc = null;  // [cur]
    for (let l = 0; l < this.L; l++) {
      const nin = this.sizes[l], nout = this.sizes[l + 1];
      const W = this.W[l], b = this.b[l];
      if (Wprod === null) {
        Wprod = W.slice(); // [nin x nout] with in=sizes[0]
        bAcc = b.slice();
      } else {
        // Wprod is [in0 x nin]; multiply by W [nin x nout] -> [in0 x nout]
        const in0 = this.sizes[0];
        const newW = new Float64Array(in0 * nout);
        for (let i = 0; i < in0; i++)
          for (let j = 0; j < nout; j++) {
            let s = 0;
            for (let k = 0; k < nin; k++) s += Wprod[i * nin + k] * W[k * nout + j];
            newW[i * nout + j] = s;
          }
        // new bias = bAcc * W + b
        const newB = new Float64Array(nout);
        for (let j = 0; j < nout; j++) {
          let s = b[j];
          for (let k = 0; k < nin; k++) s += bAcc[k] * W[k * nout + j];
          newB[j] = s;
        }
        Wprod = newW; bAcc = newB;
      }
    }
    return { W: Wprod, b: bAcc, inDim: this.sizes[0], outDim: this.sizes[this.L] };
  };

  // ---------- Embedding -> softmax next-token model ----------
  // vocab size V, embed dim d. input token id -> embed row -> linear(d->V) -> softmax
  function EmbModel(V, d, seed) {
    const rng = makeRNG(seed || 7);
    this.V = V; this.d = d;
    this.E = []; // V rows of length d (input embeddings)
    for (let i = 0; i < V; i++) {
      const row = new Float64Array(d);
      for (let k = 0; k < d; k++) row[k] = randn(rng) * 0.3;
      this.E.push(row);
    }
    // output layer d->V
    this.Wo = new Float64Array(d * V);
    for (let k = 0; k < this.Wo.length; k++) this.Wo[k] = randn(rng) * (1 / Math.sqrt(d));
    this.bo = new Float64Array(V);
  }
  EmbModel.prototype.forward = function (tok) {
    const e = this.E[tok];
    const logits = new Float64Array(this.V);
    for (let j = 0; j < this.V; j++) {
      let s = this.bo[j];
      for (let k = 0; k < this.d; k++) s += e[k] * this.Wo[k * this.V + j];
      logits[j] = s;
    }
    return { e: e, probs: softmaxArr(logits) };
  };
  EmbModel.prototype.trainBatch = function (pairs, lr) {
    // pairs: array of [tok, nextTok]
    const N = pairs.length;
    const gE = {}; // sparse by tok
    const gWo = new Float64Array(this.Wo.length);
    const gbo = new Float64Array(this.V);
    let loss = 0;
    for (let n = 0; n < N; n++) {
      const tok = pairs[n][0], y = pairs[n][1];
      const fc = this.forward(tok);
      const p = fc.probs, e = fc.e;
      loss += -Math.log(Math.max(1e-12, p[y]));
      const dlogits = new Float64Array(this.V);
      for (let j = 0; j < this.V; j++) dlogits[j] = p[j] - (j === y ? 1 : 0);
      // grad wrt Wo, bo, and e
      const de = new Float64Array(this.d);
      for (let j = 0; j < this.V; j++) {
        const d = dlogits[j];
        gbo[j] += d;
        for (let k = 0; k < this.d; k++) {
          gWo[k * this.V + j] += e[k] * d;
          de[k] += this.Wo[k * this.V + j] * d;
        }
      }
      if (!gE[tok]) gE[tok] = new Float64Array(this.d);
      for (let k = 0; k < this.d; k++) gE[tok][k] += de[k];
    }
    for (let k = 0; k < this.Wo.length; k++) this.Wo[k] -= lr * gWo[k] / N;
    for (let j = 0; j < this.V; j++) this.bo[j] -= lr * gbo[j] / N;
    for (const tok in gE) {
      const row = this.E[tok], g = gE[tok];
      for (let k = 0; k < this.d; k++) row[k] -= lr * g[k] / N;
    }
    return loss / N;
  };

  // ---------- helpers ----------
  function softmaxArr(z) {
    let m = -Infinity;
    for (let i = 0; i < z.length; i++) if (z[i] > m) m = z[i];
    const out = new Float64Array(z.length);
    let s = 0;
    for (let i = 0; i < z.length; i++) { out[i] = Math.exp(z[i] - m); s += out[i]; }
    for (let i = 0; i < z.length; i++) out[i] /= s;
    return out;
  }

  // PCA to 2D via covariance + power iteration. rows: array of Float64Array (n x d)
  function pca2(rows) {
    const n = rows.length, d = rows[0].length;
    const mean = new Float64Array(d);
    for (let i = 0; i < n; i++) for (let k = 0; k < d; k++) mean[k] += rows[i][k];
    for (let k = 0; k < d; k++) mean[k] /= n;
    // covariance d x d
    const C = new Float64Array(d * d);
    for (let i = 0; i < n; i++) {
      for (let a = 0; a < d; a++) {
        const va = rows[i][a] - mean[a];
        for (let b = 0; b < d; b++) C[a * d + b] += va * (rows[i][b] - mean[b]);
      }
    }
    for (let k = 0; k < C.length; k++) C[k] /= n;
    const rng = makeRNG(3);
    function topEigen(C, d, deflate) {
      let v = new Float64Array(d);
      for (let k = 0; k < d; k++) v[k] = randn(rng);
      normalize(v);
      for (let it = 0; it < 200; it++) {
        const nv = matVec(C, v, d);
        if (deflate) {
          // remove component along deflate vector
          let dot = 0; for (let k = 0; k < d; k++) dot += nv[k] * deflate[k];
          for (let k = 0; k < d; k++) nv[k] -= dot * deflate[k];
        }
        normalize(nv);
        v = nv;
      }
      return v;
    }
    const v1 = topEigen(C, d, null);
    const v2 = topEigen(C, d, v1);
    const out = [];
    for (let i = 0; i < n; i++) {
      let x = 0, y = 0;
      for (let k = 0; k < d; k++) { x += (rows[i][k] - mean[k]) * v1[k]; y += (rows[i][k] - mean[k]) * v2[k]; }
      out.push([x, y]);
    }
    return out;
  }
  function matVec(C, v, d) {
    const out = new Float64Array(d);
    for (let a = 0; a < d; a++) { let s = 0; for (let b = 0; b < d; b++) s += C[a * d + b] * v[b]; out[a] = s; }
    return out;
  }
  function normalize(v) {
    let s = 0; for (let k = 0; k < v.length; k++) s += v[k] * v[k];
    s = Math.sqrt(s) || 1;
    for (let k = 0; k < v.length; k++) v[k] /= s;
  }

  const api = { MLP, EmbModel, pca2, makeRNG, randn, softmaxArr };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  root.NN = api;
})(typeof window !== "undefined" ? window : globalThis);
