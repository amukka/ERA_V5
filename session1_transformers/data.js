/* data.js — dataset generators (shared by browser + node). */
(function (root) {
  "use strict";
  const NN = (typeof require !== "undefined") ? require("./nn.js") : root.NN;
  const makeRNG = NN.makeRNG, randn = NN.randn;

  // Two concentric noisy rings. Inner ring = class 0, outer ring = class 1.
  // Returns {X:[[x,y],...], Y:[0/1,...]}. Coordinates roughly in [-1.4,1.4].
  function rings(n, seed, noise) {
    const rng = makeRNG(seed || 11);
    noise = noise == null ? 0.12 : noise;
    const X = [], Y = [];
    for (let i = 0; i < n; i++) {
      const cls = i % 2;              // balanced
      const r = cls === 0 ? 0.45 : 1.05;
      const a = rng() * Math.PI * 2;
      const rr = r + randn(rng) * noise;
      X.push([Math.cos(a) * rr, Math.sin(a) * rr]);
      Y.push(cls);
    }
    return { X, Y };
  }

  // Two interleaving moons — a learnable, nonlinear boundary with label noise.
  // Used for the memorization/generalization demo.
  function moons(n, seed, noise, flip) {
    const rng = makeRNG(seed || 21);
    noise = noise == null ? 0.18 : noise;
    flip = flip == null ? 0.0 : flip; // label-noise fraction
    const X = [], Y = [];
    for (let i = 0; i < n; i++) {
      const cls = i % 2;
      let x, y;
      const t = rng() * Math.PI;
      if (cls === 0) { x = Math.cos(t); y = Math.sin(t); }
      else { x = 1 - Math.cos(t); y = 0.5 - Math.sin(t); }
      x += randn(rng) * noise; y += randn(rng) * noise;
      let label = cls;
      if (rng() < flip) label = 1 - label;
      X.push([x, y]); Y.push(label);
    }
    return { X, Y };
  }

  // Toy grammar for the embedding demo.
  // Categories chosen so same-category tokens share their next-token distribution.
  // Template:  the ANIMAL VERB the FRUIT .
  //   the    -> a noun (animal or fruit)
  //   ANIMAL -> a verb          (all animals share this successor set)
  //   VERB   -> "the"           (all verbs share this successor)
  //   FRUIT  -> "."             (all fruits share this successor)
  // Every content token thus appears as a *current* token with a
  // category-consistent next-token distribution, so its input embedding trains.
  const VOCAB = ["the", "cat", "dog", "cow", "apple", "mango", "eat", "chase", "see", "."];
  const CAT = {           // category id per token index
    the: "the", cat: "animal", dog: "animal", cow: "animal",
    apple: "fruit", mango: "fruit", eat: "verb", chase: "verb", see: "verb", ".": "end",
  };
  const ANIMALS = ["cat", "dog", "cow"], FRUITS = ["apple", "mango"], VERBS = ["eat", "chase", "see"];

  function grammarPairs(nSentences, seed) {
    const rng = makeRNG(seed || 31);
    const idx = {}; VOCAB.forEach((w, i) => (idx[w] = i));
    const pick = (arr) => arr[Math.floor(rng() * arr.length)];
    const pairs = [];
    for (let s = 0; s < nSentences; s++) {
      const subj = pick(ANIMALS);
      const verb = pick(VERBS);
      const obj = pick(FRUITS);
      const sent = ["the", subj, verb, "the", obj, "."];
      for (let i = 0; i < sent.length - 1; i++) pairs.push([idx[sent[i]], idx[sent[i + 1]]]);
    }
    return pairs;
  }

  const api = { rings, moons, grammarPairs, VOCAB, CAT, categoryOf: (i) => CAT[VOCAB[i]] };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  root.DATA = api;
})(typeof window !== "undefined" ? window : globalThis);
