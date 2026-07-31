/* app.js — tokenizer widget. Loads the shipped tokenizer.json + the pinned India
 * pages and computes every number LIVE in the browser with the real byte-level
 * encoder/decoder, so what you see equals what re-running tokenizer.model gives
 * (verified: identical to tokenizer.py / verify.py). */
(function () {
  "use strict";
  var BPE = window.BPE;
  var LANGS = ["en", "hi", "te", "kn"];
  var NAMES = { en: "English", hi: "Hindi", te: "Telugu", kn: "Kannada" };
  var SCRIPT = { en: "Latin", hi: "Devanagari", te: "Telugu", kn: "Kannada" };
  var COLORS = { en: "#3987e5", hi: "#eb6834", te: "#1baf7a", kn: "#9085e9" };
  var $ = function (s, r) { return (r || document).querySelector(s); };

  var TOK = null, META = null, STATS = null, TEXTS = {}, REPORT = {}, SCORE = 0, SORTED = [];

  function boot() {
    return Promise.all([
      fetch("tokenizer.json").then(function (r) { return r.json(); }),
      fetch("data/meta.json").then(function (r) { return r.json(); }),
      fetch("stats.json").then(function (r) { return r.json(); }),
    ]).then(function (arr) {
      var tokJson = arr[0]; META = arr[1]; STATS = arr[2];
      TOK = BPE.fromJson(tokJson);
      return Promise.all(LANGS.map(function (l) {
        return fetch("data/" + l + ".txt").then(function (r) { return r.text(); })
          .then(function (t) { TEXTS[l] = t; });
      }));
    }).then(function () {
      computeScores();
      renderHeadline();
      renderScoreboard();
      renderRoundtrip();
      renderMethod();
      renderExplorer();
      initPlayground();
    });
  }

  function computeScores() {
    for (var i = 0; i < LANGS.length; i++) {
      var l = LANGS[i];
      var ids = TOK.encode(TEXTS[l]);
      var words = TOK.wordCount(TEXTS[l]);
      REPORT[l] = {
        name: NAMES[l], words: words, tokens: ids.length, fertility: ids.length / words,
        merges: STATS.languages[l].merges, revid: META[l].revid, url: META[l].url, title: META[l].title,
      };
    }
    SORTED = LANGS.map(function (l) { return { l: l, f: REPORT[l].fertility }; })
      .sort(function (a, b) { return a.f - b.f; });
    SCORE = 1000 / (SORTED[3].f - SORTED[0].f);
  }

  function renderHeadline() {
    $("#score-val").textContent = SCORE.toFixed(1);
    $("#spread-val").textContent = (SORTED[3].f - SORTED[0].f).toFixed(4);
    $("#vocab-val").textContent = TOK.vocabSize.toLocaleString();
    var en = REPORT.en.fertility;
    var chip = $("#en-check");
    chip.textContent = "English " + en.toFixed(3) + " ≤ 1.2  " + (en <= 1.2 ? "✓" : "✗");
    chip.classList.add(en <= 1.2 ? "ok" : "bad");
  }

  function renderScoreboard() {
    var tb = $("#score-rows");
    var maxF = Math.max.apply(null, LANGS.map(function (l) { return REPORT[l].fertility; }));
    tb.innerHTML = LANGS.map(function (l) {
      var r = REPORT[l];
      var pct = (r.fertility / maxF) * 100;
      var rank = SORTED.findIndex(function (x) { return x.l === l; });
      var isMax = rank === 3, isMin = rank === 0;
      return '<div class="frow">' +
        '<div class="fname"><span class="dot" style="background:' + COLORS[l] + '"></span>' + r.name +
        '<span class="script">' + SCRIPT[l] + '</span></div>' +
        '<div class="fbar"><span class="fbar-fill" style="width:' + pct.toFixed(1) + '%;background:' + COLORS[l] + '"></span></div>' +
        '<div class="fnum">' + r.words.toLocaleString() + '</div>' +
        '<div class="fnum">' + r.tokens.toLocaleString() + '</div>' +
        '<div class="ffert">' + r.fertility.toFixed(4) +
        (isMax ? '<span class="tag max">Xmax</span>' : '') + (isMin ? '<span class="tag min">Xmin</span>' : '') + '</div>' +
        '</div>';
    }).join("");
    $("#calc-line").innerHTML =
      "score = 1000 / (X<sub>max</sub> − X<sub>min</sub>) = 1000 / (" +
      SORTED[3].f.toFixed(4) + " − " + SORTED[0].f.toFixed(4) + ") = 1000 / " +
      (SORTED[3].f - SORTED[0].f).toFixed(4) + " = <b>" + SCORE.toFixed(1) + "</b>";
  }

  // ---- faithful-roundtrip gate, run LIVE on the full pages + samples ----
  function renderRoundtrip() {
    var samples = STATS.roundtrip_samples.concat(["India's population is 1,428,627,663."]);
    var rows = "";
    var allOk = true;
    samples.forEach(function (s) {
      var ok = TOK.decode(TOK.encode(s)) === s; allOk = allOk && ok;
      rows += rtRow(ok, s.length > 44 ? s.slice(0, 44) + "…" : s);
    });
    LANGS.forEach(function (l) {
      var ok = TOK.decode(TOK.encode(TEXTS[l])) === TEXTS[l]; allOk = allOk && ok;
      rows += rtRow(ok, "full " + NAMES[l] + " page — " + TEXTS[l].length.toLocaleString() + " chars");
    });
    $("#rt-rows").innerHTML = rows;
    var badge = $("#rt-badge");
    badge.textContent = allOk ? "GATE PASSED ✓  decode(encode(text)) == text on every case" : "GATE FAILED";
    badge.className = "chip " + (allOk ? "ok" : "bad");
  }
  function rtRow(ok, label) {
    return '<div class="rtrow"><span class="rtmark ' + (ok ? "ok" : "bad") + '">' +
      (ok ? "✓" : "✗") + '</span><span class="rtlabel" dir="auto">' + escapeHtml(label) + '</span></div>';
  }

  function renderMethod() {
    $("#lang-meta").innerHTML = LANGS.map(function (l) {
      var m = META[l];
      return '<tr><td><span class="dot" style="background:' + COLORS[l] + '"></span>' + NAMES[l] + '</td>' +
        '<td dir="auto"><a href="' + m.url + '" target="_blank" rel="noopener">' + m.title + '</a></td>' +
        '<td class="mono">' + m.words.toLocaleString() + '</td>' +
        '<td class="mono">' + REPORT[l].merges.toLocaleString() + '</td></tr>';
    }).join("");
    $("#base-size").textContent = "256";
    $("#merge-size").textContent = (TOK.vocabSize - 256).toLocaleString();
  }

  // ---- tokenizer explorer ----
  function renderExplorer() {
    var vocab = [];
    for (var id = 0; id < TOK.vocabSize; id++) vocab.push(TOK.tokenText(id));
    $("#vocab-count").textContent = TOK.vocabSize.toLocaleString();
    var list = $("#vocab-list"), search = $("#vocab-search");
    function draw(filter) {
      var f = filter.trim();
      var items = f ? vocab.filter(function (t) { return t.indexOf(f) !== -1; }) : vocab;
      var cap = 600;
      var shown = items.slice(0, cap);
      list.innerHTML = shown.map(function (t) {
        return '<span class="vtok">' + escapeHtml(t.replace(/ /g, "␣")) + '</span>';
      }).join("") + (items.length > cap ? '<div class="vmore">…and ' + (items.length - cap).toLocaleString() +
        ' more (refine search or download the full list)</div>' : "");
      $("#vocab-shown").textContent = Math.min(cap, items.length).toLocaleString() + " / " + items.length.toLocaleString();
    }
    search.addEventListener("input", function () { draw(search.value); });
    draw("");
  }
  function escapeHtml(s) { return s.replace(/[&<>]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]; }); }

  // ---- live playground: encode, then decode back, showing the roundtrip ----
  function detectLang(text) {
    var dev = 0, tel = 0, kan = 0, lat = 0;
    for (var i = 0; i < text.length; i++) {
      var c = text.codePointAt(i);
      if (c >= 0x0900 && c <= 0x097f) dev++;
      else if (c >= 0x0c00 && c <= 0x0c7f) tel++;
      else if (c >= 0x0c80 && c <= 0x0cff) kan++;
      else if ((c >= 65 && c <= 90) || (c >= 97 && c <= 122)) lat++;
    }
    var m = Math.max(dev, tel, kan, lat);
    if (m === 0) return "en";
    if (m === dev) return "hi"; if (m === tel) return "te"; if (m === kan) return "kn"; return "en";
  }
  function initPlayground() {
    var ta = $("#pg-input"), out = $("#pg-tokens"), stat = $("#pg-stat"), langBadge = $("#pg-lang"), rt = $("#pg-roundtrip");
    var samples = {
      en: "India's population is 1,428,627,663.",
      hi: "भारत दक्षिण एशिया में स्थित एक देश है।",
      te: "భారతదేశం దక్షిణ ఆసియాలో ఉన్న ఒక దేశం.",
      kn: "ಭಾರತವು ದಕ್ಷಿಣ ಏಷ್ಯಾದ ಒಂದು ದೇಶವಾಗಿದೆ.",
    };
    function run() {
      var text = ta.value;
      var lang = detectLang(text);
      langBadge.textContent = NAMES[lang] + " · " + SCRIPT[lang];
      langBadge.style.background = COLORS[lang];
      var ids = TOK.encode(text);
      out.innerHTML = ids.map(function (id) {
        return '<span class="tok" style="--tc:' + COLORS[lang] + '">' + escapeHtml(TOK.tokenText(id).replace(/ /g, "␣")) + '</span>';
      }).join("");
      var words = TOK.wordCount(text);
      stat.innerHTML = words ? "<b>" + words + "</b> words → <b>" + ids.length + "</b> tokens · fertility <b>" +
        (ids.length / words).toFixed(3) + "</b>" : "type something…";
      var back = TOK.decode(ids);
      var ok = back === text;
      rt.className = "chip " + (ok ? "ok" : "bad");
      rt.textContent = ok ? "decode(encode(text)) == text  ✓ exact roundtrip" : "roundtrip mismatch ✗";
    }
    ta.addEventListener("input", run);
    document.querySelectorAll("[data-sample]").forEach(function (b) {
      b.addEventListener("click", function () { ta.value = samples[b.getAttribute("data-sample")]; run(); });
    });
    ta.value = samples.en; run();
  }

  boot().catch(function (e) {
    document.body.insertAdjacentHTML("afterbegin",
      '<pre style="color:#e66;padding:20px">load error: ' + e.message + "</pre>");
  });
})();
