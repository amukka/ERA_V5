/* Every attention mechanism on the timeline, in the order it actually appeared.
   ---------------------------------------------------------------------------
   DATE RULE: `date` is the arXiv v1 submission date, or for non-paper releases
   the public release date. NOT the conference date, NOT a later revision. Every
   one was checked against the primary source; `verified` records exactly what
   was confirmed and `caveat` records anything that could not be.

   Later arXiv revisions are a real trap: YaRN's v2 is 1 Nov 2023, which is the
   date most secondary sources repeat. Its v1 is 31 Aug 2023, and that is the
   date used here. */

export const ACTS = [
  {
    id: 'exact',
    name: 'Act I — it wants exactness',
    span: '2014 – 2017',
    thesis: 'Attention is invented and it is exact: every token compares itself with every '
      + 'other token, and the answer is the best one available. Nobody minds the cost, because '
      + 'sequences are short and the win over a fixed-length bottleneck is enormous.',
  },
  {
    id: 'bill',
    name: 'Act II — the bill arrives',
    span: '2019 – 2021',
    thesis: 'Sequences get longer and T² starts to hurt. Everything in this act trades away some '
      + 'of that exactness — read fewer keys, or delete the softmax and keep a fixed-size state '
      + 'instead. The field discovers that giving up exactness costs quality, and starts arguing '
      + 'about how much.',
  },
  {
    id: 'length',
    name: 'Act III — it wants length',
    span: '2021 – 2024',
    thesis: 'The bottleneck moves from FLOPs to how far position can reach. Rotations replace '
      + 'lookup tables, then a run of methods rescale those rotations to stretch a short-trained '
      + 'model over a long window. Cheaper than training long, with a ceiling nobody can state '
      + 'precisely.',
  },
  {
    id: 'memory',
    name: 'Act IV — it wants memory back',
    span: '2023 – 2026',
    thesis: 'Serving costs replace training costs as the thing that hurts. The KV cache is now '
      + 'the binding constraint, so attention gets compressed along every axis available — heads, '
      + 'sequence, rank — and the field stops choosing one mechanism per network and starts '
      + 'interleaving them.',
  },
];

export const FAMILIES = {
  core: { label: 'core attention', colour: '#3987e5' },
  position: { label: 'position', colour: '#9085e9' },
  extension: { label: 'context extension', colour: '#c084d8' },
  sparse: { label: 'sparse / local', colour: '#eb6834' },
  linear: { label: 'linear / state', colour: '#1baf7a' },
  cache: { label: 'KV-cache shape', colour: '#e0a852' },
  systems: { label: 'systems / exact', colour: '#7f8fa6' },
  hybrid: { label: 'hybrid schedule', colour: '#d85888' },
};

export const MECHANISMS = [
  /* ────────────────────────── ACT I ────────────────────────── */
  {
    act: 'exact',
    family: 'core',
    date: '2014-09-01',
    name: 'Additive (Bahdanau) attention',
    who: 'Bahdanau, Cho, Bengio',
    where: 'arXiv:1409.0473',
    url: 'https://arxiv.org/abs/1409.0473',
    problem: 'An encoder–decoder had to squeeze an entire source sentence into one fixed-length '
      + 'vector. Long sentences degraded badly, because the vector ran out of room.',
    answer: 'Let the decoder soft-search the source at every output step, scoring each source '
      + 'position with a small feed-forward network and taking a weighted sum.',
    pros: [
      'Removed the fixed-length bottleneck — translation quality stopped collapsing on long sentences.',
      'The weights are interpretable as a soft alignment, which was genuinely new.',
    ],
    cons: [
      'The scoring function is an MLP per pair, so it is slow and does not batch as a single matmul.',
      'Still bolted onto a recurrent model, so the sequence is processed serially.',
    ],
    pick: 'Historical interest only. Every property you want from it is cheaper in dot-product form.',
    verified: 'Title, authors and v1 date 1 Sep 2014 from the arXiv abstract page.',
  },
  {
    act: 'exact',
    family: 'position',
    date: '2017-05-08',
    name: 'Learned absolute position embeddings',
    who: 'Gehring, Auli, Grangier, Yarats, Dauphin (ConvS2S)',
    where: 'arXiv:1705.03122',
    url: 'https://arxiv.org/abs/1705.03122',
    problem: 'Drop recurrence for speed and the model loses all sense of order. Something has to '
      + 'tell it which token came first.',
    answer: 'Keep a trainable lookup table with one row per position and add it to the token '
      + 'embedding.',
    pros: [
      'Trivially simple, and the model learns whatever positional structure the data rewards.',
      'No hand-designed function to get wrong.',
    ],
    cons: [
      'Hard length wall: there is no row for position 4097 in a 4096-row table. Not a degradation — an absence.',
      'Every position is learned independently, so nothing generalises between nearby positions.',
      'Costs vocabulary-sized parameters you could have spent elsewhere.',
    ],
    pick: 'Only when the maximum length is fixed forever and known in advance. For anything that '
      + 'might need to grow, this is the choice Session 7 ruled out.',
    verified: 'Title, authors and v1 date 8 May 2017 from the arXiv abstract page.',
    caveat: 'The abstract page does not state the positional method; the learned position '
      + 'embedding is described in the paper body (§3.1). Date is verified, mechanism attribution '
      + 'is from the full text.',
  },
  {
    act: 'exact',
    family: 'core',
    date: '2017-06-12',
    name: 'Scaled dot-product attention + multi-head + sinusoidal position',
    who: 'Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin',
    where: 'arXiv:1706.03762',
    url: 'https://arxiv.org/abs/1706.03762',
    isBaseline: true,
    problem: 'Recurrence forced sequential computation, so training could not use the whole '
      + 'accelerator. And additive attention was too slow to be the primary operation.',
    answer: 'Replace the MLP scorer with a plain dot product, divide by √d_k to keep softmax out '
      + 'of saturation, run several heads in parallel, and add a fixed sinusoidal signal for '
      + 'position. Drop recurrence entirely.',
    pros: [
      'Two matmuls and a softmax — maps perfectly onto accelerators, and the whole sequence trains in parallel.',
      'Exact: every token genuinely sees every permitted token, with no approximation anywhere.',
      'Sinusoidal position is a function, so it is defined at any length and costs no parameters.',
    ],
    cons: [
      'T² score computations and T² memory for the score matrix. This is the bill everything after it is trying to pay.',
      'A KV cache that grows linearly with context during generation, private to each conversation.',
      'Sinusoidal position extrapolates in principle but works poorly in practice much past training length.',
    ],
    pick: 'The default, and still correct for short contexts. Under a few thousand tokens the T² '
      + 'term is not your bottleneck and every alternative is strictly worse quality for no real gain.',
    verified: 'Title, authors and v1 date 12 Jun 2017 from the arXiv abstract page.',
    caveat: 'The abstract page does not confirm the positional scheme; sinusoidal encoding and the '
      + 'reported near-identical result for learned embeddings are in §3.5 of the full text.',
  },

  /* ────────────────────────── ACT II ────────────────────────── */
  {
    act: 'bill',
    family: 'linear',
    date: '2019-01-09',
    name: 'Transformer-XL segment recurrence',
    who: 'Dai, Yang, Yang, Carbonell, Le, Salakhutdinov',
    where: 'arXiv:1901.02860',
    url: 'https://arxiv.org/abs/1901.02860',
    problem: 'Training chopped documents into fixed segments and threw away everything at each '
      + 'boundary, so no dependency could ever cross one.',
    answer: 'Cache the previous segment\'s hidden states and let the current segment attend to '
      + 'them, with gradients stopped at the boundary. Plus a relative positional scheme, because '
      + 'absolute positions break when the window slides.',
    pros: [
      'Context outlives the segment without the training graph growing to document length.',
      'Introduced relative position two years before RoPE made it standard.',
    ],
    cons: [
      'The cached segment is extra memory that grows with how much history you keep.',
      'Stop-gradient means the model never learns to *write* a good summary for its future self — it only learns to read whatever landed there.',
    ],
    pick: 'The idea, not the implementation. Its stop-gradient carry is exactly what §14\'s Memory '
      + 'Stream is, compressed to a single vector — worth knowing when you build cross-chunk state.',
    verified: 'Title, authors and v1 date 9 Jan 2019 from the arXiv abstract page. Abstract '
      + 'confirms segment-level recurrence.',
    caveat: 'The abstract does not mention stop-gradient specifically; that detail is in the '
      + 'full text.',
  },
  {
    act: 'bill',
    family: 'sparse',
    date: '2019-04-23',
    name: 'Sparse Transformer (factorised sparse attention)',
    who: 'Child, Gray, Radford, Sutskever',
    where: 'arXiv:1904.10509',
    url: 'https://arxiv.org/abs/1904.10509',
    problem: 'First serious attempt on the T² bill. Generating images and audio needed thousands '
      + 'of steps, and the score matrix would not fit.',
    answer: 'Do not let every query read every key. Factorise the attention pattern into strided '
      + 'and local components so each query reads O(√T) keys.',
    pros: [
      'Cut complexity to O(T√T) with a pattern fixed in advance — no routing cost at all.',
      'Enabled sequence lengths that were simply impossible before.',
    ],
    cons: [
      'The pattern is hand-designed and content-blind: a token that matters is dropped if it falls outside the stride.',
      'Needs custom kernels to realise the saving, so it was hard to adopt.',
    ],
    pick: 'When your data has known structure that a fixed pattern captures — images, audio, '
      + 'anything with a natural stride. For text, learned or content-based selection beats it.',
    verified: 'Title, authors and v1 date 23 Apr 2019 from the arXiv abstract page. Abstract '
      + 'confirms O(n√n) sparse factorisation.',
  },
  {
    act: 'bill',
    family: 'cache',
    date: '2019-11-06',
    name: 'Multi-Query Attention (MQA)',
    who: 'Noam Shazeer',
    where: 'arXiv:1911.02150',
    url: 'https://arxiv.org/abs/1911.02150',
    problem: 'Not training — *decoding*. Generating one token means re-reading the whole KV cache '
      + 'from memory, and that bandwidth, not arithmetic, is what makes generation slow.',
    answer: 'Keep all the query heads but give them a single shared key/value head. The cache '
      + 'shrinks by the number of heads.',
    pros: [
      'Enormous cache reduction — the full head-count factor, so 8× or 32× depending on the model.',
      'Decoding gets dramatically faster, because it was memory-bandwidth-bound and you just shrank the memory.',
    ],
    cons: [
      'Quality cost is real: all heads now search the same key space, so they lose independent retrieval.',
      'Usually needs retraining or uptraining — you cannot just collapse the heads of a finished model and hope.',
    ],
    pick: 'When decode throughput dominates and you can afford some quality: high-volume serving, '
      + 'small models, or anything latency-critical. GQA usually beats it now.',
    verified: 'Title, sole author Noam Shazeer, v1 date 6 Nov 2019 from the arXiv abstract page. '
      + 'Abstract confirms keys and values shared across heads.',
    note: 'Published four years before GQA and largely ignored until serving costs made it urgent. '
      + 'The paper title — "One Write-Head is All You Need" — deserves more credit than it got.',
  },
  {
    act: 'bill',
    family: 'sparse',
    date: '2019-11-13',
    name: 'Compressive Transformer',
    who: 'Rae, Potapenko, Jayakumar, Lillicrap',
    where: 'arXiv:1911.05507',
    url: 'https://arxiv.org/abs/1911.05507',
    problem: 'Transformer-XL\'s cache still had to be evicted eventually, and eviction throws '
      + 'information away entirely.',
    answer: 'Instead of discarding old memories, compress them into a coarser secondary memory. '
      + 'Recent history stays fine-grained; distant history becomes summaries.',
    pros: [
      'Old context degrades gracefully rather than vanishing at a hard boundary.',
      'The first clear statement that history does not need one equally expensive representation per token.',
    ],
    cons: [
      'Compression is lossy and the loss is not recoverable — a summary cannot be un-summarised.',
      'Adds a compression network and its own losses to train and tune.',
    ],
    pick: 'The direct ancestor of §12. If you are choosing sequence compression today you would '
      + 'reach for NSA or DSA, but this is where the idea starts.',
    verified: 'Title, authors and v1 date 13 Nov 2019 from the arXiv abstract page.',
  },
  {
    act: 'bill',
    family: 'sparse',
    date: '2020-01-13',
    name: 'Reformer (LSH attention)',
    who: 'Kitaev, Kaiser, Levskaya',
    where: 'arXiv:2001.04451',
    url: 'https://arxiv.org/abs/2001.04451',
    problem: 'Fixed sparse patterns are content-blind. If the useful key is far away and not on '
      + 'the stride, you never see it.',
    answer: 'Use locality-sensitive hashing to bucket queries with the keys they are likely to '
      + 'match, then attend only within buckets. O(T log T).',
    pros: [
      'Content-based rather than positional: the sparsity adapts to what the query is actually looking for.',
      'Reversible layers cut activation memory too, which was a separate real win.',
    ],
    cons: [
      'Hashing is stochastic, so it sometimes misses a genuinely high-scoring pair, and you cannot tell when.',
      'Needs several hash rounds to be reliable, which eats the saving.',
      'Awkward to implement well; adoption never really followed the citation count.',
    ],
    pick: 'Rarely, now. Its lesson survives in every learned-router design: content-based '
      + 'selection beats a fixed pattern, but you must pay something to find the candidates.',
    verified: 'Title, authors and v1 date 13 Jan 2020 from the arXiv abstract page. Abstract '
      + 'confirms LSH and O(L log L).',
  },
  {
    act: 'bill',
    family: 'sparse',
    date: '2020-04-10',
    name: 'Sliding-window / local attention (Longformer)',
    who: 'Beltagy, Peters, Cohan',
    where: 'arXiv:2004.05150',
    url: 'https://arxiv.org/abs/2004.05150',
    problem: 'Documents are long, most dependencies are local, and paying T² to discover that is '
      + 'absurd.',
    answer: 'Each token attends to a fixed window of w neighbours, plus a few designated global '
      + 'tokens that everyone can see. Cost becomes O(T·w).',
    pros: [
      'Genuinely linear in T, with a pattern simple enough to implement efficiently.',
      'Matches how language mostly works — local dependencies dominate.',
      'Global tokens give an escape hatch for the things that must be seen by all.',
    ],
    cons: [
      'A dependency longer than w is invisible unless it routes through a global token or up through layers.',
      'Effective receptive field grows only linearly with depth, so genuine long-range reasoning is hard.',
      'Choosing w is a guess about your data that you cannot easily revisit after training.',
    ],
    pick: 'Very defensible for long-document classification, retrieval and summarisation where '
      + 'you mostly need local coherence. Weak for tasks needing exact recall of one distant fact.',
    verified: 'Title, authors and v1 date 10 Apr 2020 from the arXiv abstract page. Abstract '
      + 'confirms local windowed attention plus task-motivated global attention.',
  },
  {
    act: 'bill',
    family: 'linear',
    date: '2020-06-08',
    name: 'Linformer (low-rank attention)',
    who: 'Wang, Li, Khabsa, Fang, Ma',
    where: 'arXiv:2006.04768',
    url: 'https://arxiv.org/abs/2006.04768',
    problem: 'The score matrix is T × T, but is it actually full-rank? If not, you are storing '
      + 'redundancy.',
    answer: 'Project keys and values down to a fixed length k before attending. The score matrix '
      + 'becomes T × k.',
    pros: [
      'Linear in T with a very small change to the code.',
      'The low-rank observation is empirically well supported for many tasks.',
    ],
    cons: [
      'The projection is over the sequence axis, so it needs a fixed maximum length — it does not extrapolate.',
      'Fatal for autoregressive decoding: you cannot causally project a sequence you have not finished generating.',
    ],
    pick: 'Encoder-only work with a known fixed length. Never for a decoder, which rules it out '
      + 'of most of what we care about.',
    verified: 'Title, authors and v1 date 8 Jun 2020 from the arXiv abstract page. Abstract '
      + 'confirms low-rank approximation and O(n).',
  },
  {
    act: 'bill',
    family: 'linear',
    date: '2020-06-29',
    name: 'Linear attention — "Transformers are RNNs"',
    who: 'Katharopoulos, Vyas, Pappas, Fleuret',
    where: 'arXiv:2006.16236',
    url: 'https://arxiv.org/abs/2006.16236',
    isKey: true,
    problem: 'Every sparse method still keeps a growing list of keys. Can the past be folded into '
      + 'something whose size does not depend on T at all?',
    answer: 'Delete the softmax. Then (QKᵀ)V can be reassociated as Q(KᵀV), and the sequence axis '
      + 'is summed away — leaving a fixed d × d state. Autoregressively this is literally an RNN.',
    pros: [
      'O(T) work and a state whose size never grows. Decoding cost per token is constant in context length.',
      'The regrouping is exact algebra, not an approximation — for a given feature map the two forms agree bit-for-bit.',
      'Reported up to 4000× faster autoregressive generation on long sequences.',
    ],
    cons: [
      'Deleting softmax gives up positive weights, weights summing to one, competition between keys, and a fresh per-query distribution — all four at once.',
      'A fixed state means interference: different memories share the same cells and cannot all be kept separate.',
      'Consistently behind softmax attention on recall-heavy tasks. The gap is smaller now but has never closed.',
    ],
    pick: 'When decode cost or memory is the binding constraint and the task is not recall-heavy. '
      + 'In practice, almost nobody ships this pure — it appears as the D layers of a hybrid.',
    verified: 'Title, authors and v1 date 29 Jun 2020 from the arXiv abstract page. Abstract '
      + 'confirms kernel feature maps, O(N) complexity and the 4000× figure.',
  },
  {
    act: 'bill',
    family: 'sparse',
    date: '2020-07-28',
    name: 'BigBird (window + global + random)',
    who: 'Zaheer, Guruganesh, Dubey, Ainslie, Alberti, Ontanon, Pham, Ravula, Wang, Yang, Ahmed',
    where: 'arXiv:2007.14062',
    url: 'https://arxiv.org/abs/2007.14062',
    problem: 'Sliding windows lose long-range links. Can sparsity keep the theoretical power of '
      + 'full attention?',
    answer: 'Window plus global tokens plus a few random connections — enough to make the '
      + 'attention graph an expander, which preserves universal approximation and Turing '
      + 'completeness.',
    pros: [
      'Actual theory behind the pattern rather than intuition, including O(1) global tokens.',
      'Random links give short graph paths between distant tokens, so depth compounds reach faster.',
    ],
    cons: [
      'The theoretical guarantees need more layers than people use in practice, so they are weaker than they sound.',
      'Random gather patterns are hardware-hostile; the wall-clock win rarely matches the FLOP count.',
    ],
    pick: 'Same territory as Longformer, with better theory and worse kernels. Choose on the '
      + 'quality of the available implementation.',
    verified: 'Title, authors and v1 date 28 Jul 2020 from the arXiv abstract page. Abstract '
      + 'confirms sparse attention and O(1) global tokens.',
  },
  {
    act: 'bill',
    family: 'linear',
    date: '2020-09-30',
    name: 'Performer (FAVOR+)',
    who: 'Choromanski, Likhosherstov, Dohan, Song, Gane, Sarlos, Hawkins, Davis, Mohiuddin, Kaiser, Belanger, Colwell, Weller',
    where: 'arXiv:2009.14794',
    url: 'https://arxiv.org/abs/2009.14794',
    problem: 'Linear attention drops softmax and loses its properties. Could you keep softmax and '
      + 'still get linear cost?',
    answer: 'Approximate the softmax kernel with positive orthogonal random features, so the '
      + 'reassociation trick applies to an unbiased estimate of real softmax attention.',
    pros: [
      'Unbiased estimator of softmax attention, with provable concentration bounds — not an ad-hoc substitute.',
      'Drop-in: you can approximate an already-trained softmax model.',
      'Positivity of the features matters; earlier random-feature attempts were unstable without it.',
    ],
    cons: [
      'Variance is real. Good approximation needs many random features, and the cost creeps back toward what you saved.',
      'Approximation error compounds through layers in ways that are hard to bound end to end.',
    ],
    pick: 'Intellectually the most satisfying answer in this act, and the least used. If you want '
      + 'linear cost with softmax semantics it is still worth a look, but hybrids beat it in practice.',
    verified: 'Title, authors and v1 date 30 Sep 2020 from the arXiv abstract page. Abstract '
      + 'confirms FAVOR+.',
  },
  {
    act: 'bill',
    family: 'linear',
    date: '2021-02-22',
    name: 'The delta rule for fast weights',
    who: 'Schlag, Irie, Schmidhuber',
    where: 'arXiv:2102.11174',
    url: 'https://arxiv.org/abs/2102.11174',
    isKey: true,
    problem: 'A fixed-size linear-attention state is written with `S += v kᵀ`. Write the same key '
      + 'twice and the old value is still in there, with nothing cancelling it.',
    answer: 'Read what the state currently returns for this key, subtract it from what it should '
      + 'return, and write only the difference. The state becomes editable instead of accumulate-only.',
    pros: [
      'The state can now *remove* information, not only add it — a correction can be negative, which add-only cannot express at all.',
      'Directly reduces the interference that makes fixed-size memory lossy.',
      'Names the connection between linear attention and 1990s fast-weight programmers, which reframed the whole area.',
    ],
    cons: [
      'The write now depends on a read, which is sequential — this is exactly why it took until 2024 to train efficiently at scale.',
      'Still a fixed-size state: better bookkeeping does not create capacity that is not there.',
    ],
    pick: 'Always over plain add-only linear attention. The delta rule itself is Widrow–Hoff, '
      + '1960; the contribution here is putting it inside the attention state.',
    verified: 'Title, authors and v1 date 22 Feb 2021 from the arXiv abstract page. Abstract '
      + 'explicitly states they "replace the purely additive outer products by a delta rule-like '
      + 'programming instruction".',
  },

  /* ────────────────────────── ACT III ────────────────────────── */
  {
    act: 'length',
    family: 'position',
    date: '2021-04-20',
    name: 'RoPE — rotary position embedding',
    who: 'Su, Lu, Pan, Murtadha, Wen, Liu',
    where: 'arXiv:2104.09864',
    url: 'https://arxiv.org/abs/2104.09864',
    isKey: true,
    problem: 'Learned tables have a hard length wall. Additive signals mix position into content '
      + 'in ways the dot product cannot cleanly separate. And what attention actually needs is '
      + 'relative distance, not an absolute label.',
    answer: 'Treat each pair of dimensions as a 2D arrow and rotate it by position × θ. Absolute '
      + 'rotations cancel in the dot product, leaving a function of i − j alone. Different pairs '
      + 'rotate at different rates, so one head represents distance at several scales.',
    pros: [
      'Exactly translation-invariant: shift a whole sentence a hundred places and the score is identical to 1e-16.',
      'A function, not a table, so it is defined at any position — no hard wall.',
      'Rotation preserves vector length, so it injects position without changing magnitudes.',
      'No parameters, and it applies to Q and K only, leaving V\'s content untouched.',
    ],
    cons: [
      'Defined at any length is not the same as *working* at any length. Beyond training, the low-frequency pairs are asked for angle ranges they never saw.',
      'Each individual pair wraps, so a single pair genuinely cannot tell gap 0 from gap 2π/θ.',
      'The base frequency is a hyperparameter with long-range consequences that are hard to predict before training.',
    ],
    pick: 'The default for any decoder today, and the thing every extension method in this act is '
      + 'built on top of.',
    verified: 'Title, authors and v1 date 20 Apr 2021 from the arXiv abstract page. Abstract '
      + 'confirms encoding absolute position with a rotation matrix while incorporating relative '
      + 'position dependency.',
  },
  {
    act: 'length',
    family: 'position',
    date: '2021-08-27',
    name: 'ALiBi — attention with linear biases',
    who: 'Press, Smith, Lewis',
    where: 'arXiv:2108.12409',
    url: 'https://arxiv.org/abs/2108.12409',
    problem: 'Even RoPE degrades past training length. What if you stopped embedding position at '
      + 'all and just penalised distance directly?',
    answer: 'Add no positional embedding. Instead subtract a penalty proportional to i − j from '
      + 'each attention score, with a different slope per head.',
    pros: [
      'Extrapolates markedly better than sinusoidal or learned position — that is literally the paper title.',
      'Almost free: one add, no embedding, no rotation, nothing to store.',
      'Different slopes per head give a natural spread of local and global heads.',
    ],
    cons: [
      'The penalty is monotonic in distance, so it bakes in a recency prior. A task needing a token 50k back fights the mechanism.',
      'Less expressive than RoPE — it cannot represent "exactly 5 tokens back matters more than 4", only "nearer is better".',
      'Largely lost to RoPE in practice, partly because RoPE composes with the extension methods that followed.',
    ],
    pick: 'When you need robust behaviour past training length with no extra machinery and your '
      + 'task is recency-friendly. Still used, and its slope-per-head idea keeps reappearing.',
    verified: 'Title, authors and v1 date 27 Aug 2021 from the arXiv abstract page. Abstract '
      + 'confirms biasing scores with a penalty proportional to distance.',
  },
  {
    act: 'length',
    family: 'systems',
    date: '2022-05-27',
    name: 'FlashAttention — exact, but IO-aware',
    who: 'Dao, Fu, Ermon, Rudra, Ré',
    where: 'arXiv:2205.14135',
    url: 'https://arxiv.org/abs/2205.14135',
    isKey: true,
    problem: 'Everyone had been approximating attention to reduce FLOPs. But attention was never '
      + 'FLOP-bound — it was bound by moving the T × T score matrix to and from HBM.',
    answer: 'Never materialise the score matrix. Tile the computation, keep tiles in SRAM, and '
      + 'use online softmax to combine them. Same arithmetic, far fewer memory round-trips.',
    pros: [
      'EXACT. No approximation, no quality loss, no hyperparameter. This is what makes it different from everything else in this timeline.',
      'Memory drops from O(T²) to O(T), which removed the actual wall people were hitting.',
      'Retroactively made a lot of approximate attention unnecessary — if exact got cheap enough, why approximate?',
    ],
    cons: [
      'Still O(T²) arithmetic. It moves the wall, it does not remove it, and at very long context the FLOPs do bind.',
      'Hardware-specific kernels: every new accelerator generation needs the work redone.',
      'Its success arguably slowed adoption of genuinely sub-quadratic methods for several years.',
    ],
    pick: 'Always. There is no reason to run unfused exact attention. And note it is the one entry '
      + 'here that costs you nothing — a rare thing on this page.',
    verified: 'Title, authors and v1 date 27 May 2022 from the arXiv abstract page. Abstract '
      + 'confirms "IO-aware exact attention algorithm".',
    note: 'The honest lesson: before you approximate a mechanism, check whether it is actually '
      + 'bound by the thing you are reducing. Three years of approximate-attention research was '
      + 'aimed at FLOPs when the bottleneck was memory traffic.',
  },
  {
    act: 'length',
    family: 'cache',
    date: '2023-05-22',
    name: 'Grouped-Query Attention (GQA)',
    who: 'Ainslie, Lee-Thorp, de Jong, Zemlyanskiy, Lebrón, Sanghai',
    where: 'arXiv:2305.13245',
    url: 'https://arxiv.org/abs/2305.13245',
    isKey: true,
    problem: 'MQA\'s cache saving was wonderful and its quality cost was too high. Nothing sat '
      + 'between one KV head and all of them.',
    answer: 'Interpolate. Let groups of query heads share a KV head — 8 KV heads for 64 query '
      + 'heads, say. And uptrain from an existing MHA checkpoint rather than training fresh.',
    pros: [
      'Recovers most of MQA\'s cache saving at a small fraction of the quality cost. A genuinely good trade.',
      'Uptraining from an MHA checkpoint costs a few percent of pretraining — you can retrofit finished models.',
      'One tunable knob, so you can place yourself anywhere on the curve.',
    ],
    cons: [
      'Constant-factor fix to a linear-growth problem. It lowers the slope; the line still rises forever.',
      'At 1M context, GQA-8 on a 32-layer model is still tens of GB for one sequence.',
      'Quality cost is small but not zero, and it grows as you share more aggressively.',
    ],
    pick: 'The default now, and correct for almost everything. But §11\'s point stands: it is the '
      + 'baseline, not the answer. If your target is 1M tokens you need something else *as well*.',
    verified: 'Title, authors and v1 date 22 May 2023 from the arXiv abstract page.',
  },
  {
    act: 'length',
    family: 'extension',
    date: '2023-06-27',
    name: 'Position Interpolation (PI)',
    who: 'Chen, Wong, Chen, Tian',
    where: 'arXiv:2306.15595',
    url: 'https://arxiv.org/abs/2306.15595',
    problem: 'Training at 32K costs far more than training at 2K. Could a model trained short be '
      + 'made to work long without paying for long training?',
    answer: 'Do not extrapolate — *interpolate*. Linearly scale position indices down so that '
      + '32K positions map into the 2K range the model already understands, then briefly fine-tune.',
    pros: [
      'Startlingly simple and it works: LLaMA to 32K with minimal fine-tuning.',
      'Opened the entire context-extension field. Everything after it is a refinement of this move.',
    ],
    cons: [
      'Scaling all frequencies equally crushes the high-frequency pairs, which is precisely where local precision lives.',
      'Measurably hurts short-context performance — you pay for the long window even on short inputs.',
      'Still needs fine-tuning, so it is not free.',
    ],
    pick: 'Superseded by NTK-aware and YaRN, which fix the uniform-scaling flaw. Worth '
      + 'understanding because it isolates the core idea cleanly.',
    verified: 'Title, authors and v1 date 27 Jun 2023 from the arXiv abstract page. Abstract '
      + 'confirms extension of RoPE-based models to 32768 with minimal fine-tuning.',
  },
  {
    act: 'length',
    family: 'extension',
    date: '2023-06-28',
    dateApprox: true,
    name: 'NTK-aware scaled RoPE',
    who: 'bloc97 (r/LocalLLaMA), Dynamic-NTK follow-up by emozilla',
    where: 'Reddit r/LocalLLaMA — community post, no formal paper',
    url: 'https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/',
    problem: 'PI scales every rotary frequency by the same factor, which destroys the '
      + 'high-frequency detail the model relies on for local structure.',
    answer: 'Scale the frequencies *unevenly*. Stretch the low frequencies, which carry long-range '
      + 'position, and leave the high frequencies nearly alone. The right abstraction is the '
      + 'frequency, not the position.',
    pros: [
      'Works with zero fine-tuning, which PI could not do.',
      'Preserves local precision because it barely touches the fast pairs.',
      'Correctly identifies which dimensions are actually out of distribution — §9\'s slow pairs.',
    ],
    cons: [
      'No paper, no peer review, no formal evaluation — it spread by reputation and reproduction.',
      'The NTK justification is more analogy than derivation.',
      'Superseded within months by YaRN, which does the same thing more carefully.',
    ],
    pick: 'Historically important and still a reasonable no-fine-tune option. Reach for YaRN if '
      + 'you can fine-tune at all.',
    verified: 'Confirmed as a community r/LocalLLaMA post by bloc97, mid-2023, with a Dynamic-NTK '
      + 'follow-up by emozilla, and confirmed as influencing CodeLlama and Qwen.',
    caveat: 'THE ONE DATE HERE I CANNOT PIN EXACTLY. It is a Reddit post, not a paper — there is '
      + 'no arXiv v1 stamp. Sources agree on late June / early July 2023; the timeline places it '
      + 'at 28 Jun 2023 and flags it as approximate. Treat the ordering against PI and YaRN as '
      + 'reliable, the exact day as not.',
  },
  {
    act: 'length',
    family: 'extension',
    date: '2023-08-31',
    name: 'YaRN',
    who: 'Peng, Quesnelle, Fan, Shippole',
    where: 'arXiv:2309.00071',
    url: 'https://arxiv.org/abs/2309.00071',
    problem: 'NTK-aware scaling worked but was ad hoc, and interpolation methods still disturbed '
      + 'attention entropy in ways that hurt quality.',
    answer: 'Combine per-frequency interpolation ("NTK-by-parts" — leave high frequencies alone, '
      + 'interpolate low ones, blend in between) with a temperature correction to the attention '
      + 'logits that compensates for the changed entropy.',
    pros: [
      'Best-in-class RoPE scaling for a long time, and the attention-temperature fix was a genuinely new observation.',
      'Needs roughly 0.1% of original pretraining tokens to adapt — very cheap.',
      'Widely implemented and battle-tested; the reference point everything else is compared against.',
    ],
    cons: [
      'Several interacting hyperparameters, so tuning is fiddly.',
      'Still fundamentally rescaling a signal the model learned at a different scale — a repair, not a solution.',
      'Has a ceiling. Nobody can tell you in advance where it is for your model.',
    ],
    pick: 'The default extension method if you can fine-tune. Compare against DroPE now that '
      + 'dropping the embedding entirely is a published option.',
    verified: 'Title, authors and v1 date 31 Aug 2023 from the arXiv abstract page. Submission '
      + 'history shows v1 31 Aug 2023, v2 1 Nov 2023, v3 6 Feb 2026.',
    note: 'Watch out: the widely-cited "November 2023" is the v2 revision. v1 is 31 August 2023, '
      + 'and that is what puts it just before StreamingLLM rather than after.',
  },
  {
    act: 'length',
    family: 'sparse',
    date: '2023-09-29',
    name: 'Attention sinks / StreamingLLM',
    who: 'Xiao, Tian, Chen, Han, Lewis',
    where: 'arXiv:2309.17453',
    url: 'https://arxiv.org/abs/2309.17453',
    isKey: true,
    problem: 'Evict the oldest tokens from a window and quality collapses immediately — far worse '
      + 'than losing that little context should cause. Nobody knew why.',
    answer: 'Because models dump huge attention mass on the first few tokens regardless of '
      + 'content — they act as a "sink" for probability that softmax forces to sum to one. Keep '
      + 'four initial tokens permanently and rolling-window attention works fine.',
    pros: [
      'Explains a real, previously mysterious failure mode. The diagnosis is worth more than the fix.',
      'Trivial to implement — pin a handful of tokens — and enables genuinely unbounded streaming.',
      'Reveals something true about softmax: it must put its mass somewhere, so it invents a dumping ground.',
    ],
    cons: [
      'Enables infinite streaming, not infinite *context* — evicted tokens are still gone and unrecoverable.',
      'The sink is a workaround for a softmax artefact, not a mechanism that adds capability.',
      'Number of sink tokens is another empirical constant to carry around.',
    ],
    pick: 'Any streaming or always-on deployment with a rolling window. Also read it for the '
      + 'diagnosis — it changes how you read attention maps.',
    verified: 'Title, authors and v1 date 29 Sep 2023 from the arXiv abstract page. Abstract '
      + 'confirms the attention-sink phenomenon and that keeping initial tokens\' KV recovers '
      + 'window-attention performance.',
  },
  {
    act: 'length',
    family: 'systems',
    date: '2023-10-03',
    name: 'Ring Attention',
    who: 'Liu, Zaharia, Abbeel',
    where: 'arXiv:2310.01889',
    url: 'https://arxiv.org/abs/2310.01889',
    problem: 'Even with FlashAttention, one device cannot hold the activations for a '
      + 'million-token sequence.',
    answer: 'Shard the sequence across devices in a ring and rotate KV blocks around it, '
      + 'overlapping communication with computation so the transfer is free.',
    pros: [
      'Exact attention at essentially unbounded length — the limit becomes how many devices you own.',
      'No approximation and no quality cost whatsoever.',
      'Composes with FlashAttention rather than competing with it.',
    ],
    cons: [
      'Still O(T²) total work. You have parallelised the bill, not reduced it.',
      'Needs high-bandwidth interconnect; on commodity networking the overlap does not hide the transfer.',
      'A distributed-systems answer, so it does nothing for single-device or edge deployment.',
    ],
    pick: 'When you have the interconnect and want long context with zero quality compromise. '
      + 'The honest alternative to every approximation on this page — if you can afford it.',
    verified: 'Title, authors and v1 date 3 Oct 2023 from the arXiv abstract page. Abstract '
      + 'confirms blockwise distribution with overlapped KV communication and no approximation.',
  },
  {
    act: 'length',
    family: 'sparse',
    date: '2023-10-10',
    name: 'Sliding window at scale (Mistral 7B)',
    who: 'Jiang, Sablayrolles, Mensch, Bamford, Chaplot, de las Casas, et al.',
    where: 'arXiv:2310.06825',
    url: 'https://arxiv.org/abs/2310.06825',
    problem: 'Longformer showed sliding windows worked in 2020, but no strong general-purpose '
      + 'decoder had shipped with one.',
    answer: 'Ship a 7B model with sliding-window attention as the default and demonstrate it '
      + 'beating much larger dense models.',
    pros: [
      'Proved a windowed decoder could be genuinely competitive, not a research curiosity.',
      'Fixed cache size regardless of sequence length, which transformed small-model serving economics.',
      'Made SWA a standard option in every inference stack overnight.',
    ],
    cons: [
      'Information beyond the window only propagates through layer stacking, which is indirect and lossy.',
      'Exact long-range recall is weak — a known limitation of the family, inherited wholesale.',
      'Later Mistral models moved away from it, which is itself evidence about the trade.',
    ],
    pick: 'Small-to-mid models where cache size dominates and tasks are locality-friendly. Notable '
      + 'as the moment sparse attention became mainstream rather than academic.',
    verified: 'Title, authors and v1 date 10 Oct 2023 from the arXiv abstract page. Abstract '
      + 'confirms SWA for sequences of arbitrary length at reduced inference cost.',
    caveat: 'The abstract does not state the window size; 4096 is from the model config and '
      + 'implementation, not the abstract.',
  },
  {
    act: 'length',
    family: 'linear',
    date: '2023-12-01',
    name: 'Mamba (selective state space)',
    who: 'Gu, Dao',
    where: 'arXiv:2312.00752',
    url: 'https://arxiv.org/abs/2312.00752',
    problem: 'Linear attention was fast but weak. State-space models were elegant but could not '
      + 'do content-based reasoning, because their dynamics did not depend on the input.',
    answer: 'Make the state-space parameters functions of the input, so the model can selectively '
      + 'propagate or forget. Plus a hardware-aware parallel scan to keep it trainable.',
    pros: [
      'Closed much of the quality gap with attention while keeping O(T) and a constant-size state.',
      'Selectivity is the key idea — an input-dependent gate is what earlier linear methods lacked.',
      'Triggered the whole modern hybrid wave; Gated DeltaNet is explicitly "Mamba2 + delta rule".',
    ],
    cons: [
      'Still a fixed-size state, so exact recall of arbitrary earlier tokens remains its weak spot.',
      'Pure Mamba underperforms attention on in-context retrieval — which is why almost everything ships hybrid.',
      'Needs custom kernels; the naive implementation is not competitive.',
    ],
    pick: 'Rarely pure. Its real influence is as the D-layer of hybrids, and as the paper that '
      + 'convinced the field that state models were worth taking seriously again.',
    verified: 'Title, authors and v1 date 1 Dec 2023 from the arXiv abstract page. Abstract '
      + 'confirms input-dependent SSM parameters enabling selective propagation and forgetting.',
  },
  {
    act: 'length',
    family: 'extension',
    date: '2024-02-21',
    name: 'LongRoPE',
    who: 'Ding, Zhang, Zhang, Xu, Shang, Xu, Yang, Yang',
    where: 'arXiv:2402.13753',
    url: 'https://arxiv.org/abs/2402.13753',
    problem: 'YaRN\'s rescaling was uniform within each frequency band, chosen by hand. Was there '
      + 'a better per-dimension schedule that nobody had found?',
    answer: 'Search for it. Use evolutionary search over per-dimension rescaling factors, plus a '
      + 'progressive extension schedule, and reach 2048K tokens.',
    pros: [
      'Two million tokens, from 1k fine-tuning steps at 256k training length — a striking result.',
      'Correctly treats the rescaling schedule as something to optimise rather than derive.',
      'Explicitly preserves short-context performance, which earlier methods sacrificed.',
    ],
    cons: [
      'The search is a real compute cost, and it must be redone per model.',
      'A found schedule is hard to reason about — you get numbers that work without knowing why.',
      'A 2M window that is reachable is not the same as a 2M window the model uses well.',
    ],
    pick: 'When you want maximum extension from an existing checkpoint and can afford the search. '
      + 'The honest framing: it pushes the ceiling further out without removing it.',
    verified: 'Title, authors and v1 date 21 Feb 2024 from the arXiv abstract page. Abstract '
      + 'confirms 2048k tokens and 1k fine-tuning steps at 256k training length.',
  },

  /* ────────────────────────── ACT IV ────────────────────────── */
  {
    act: 'memory',
    family: 'cache',
    date: '2024-05-07',
    name: 'Multi-head Latent Attention (MLA)',
    who: 'DeepSeek-AI (DeepSeek-V2)',
    where: 'arXiv:2405.04434',
    url: 'https://arxiv.org/abs/2405.04434',
    isKey: true,
    problem: 'GQA reduces cache by sharing heads, which costs quality because the heads genuinely '
      + 'lose independence. Is there a way to shrink the cache without heads sharing anything?',
    answer: 'Compress K and V jointly into a low-rank latent vector, cache only that, and '
      + 'reconstruct per-head K and V on the fly. The compression is along the feature axis, not '
      + 'the head axis or the sequence axis.',
    pros: [
      'Much larger cache reduction than GQA, and DeepSeek reports quality at or above full MHA rather than below it.',
      'Heads keep their independence — nothing is shared, so nothing is lost to sharing.',
      'A genuinely different axis of compression, which is why it composes with the others.',
    ],
    cons: [
      'Significantly more complex: extra projections, and the up-projection can often be folded into neighbouring weights only with care.',
      'Interacts awkwardly with RoPE — DeepSeek carries a separate un-compressed rotary portion, which is a real wart.',
      'Hard to retrofit; effectively an architecture decision made before pretraining.',
    ],
    pick: 'If you are designing a new model and cache is your binding constraint, this is the '
      + 'strongest single answer available. Note it still grows linearly with T — smaller slope, '
      + 'same shape.',
    verified: 'Title, v1 date 7 May 2024, and MLA in the abstract, from the arXiv abstract page. '
      + 'Abstract confirms MLA "compressing the Key-Value (KV) cache into a latent vector".',
  },
  {
    act: 'memory',
    family: 'linear',
    date: '2024-06-10',
    name: 'DeltaNet, parallelised over sequence length',
    who: 'Yang, Wang, Zhang, Shen, Kim',
    where: 'arXiv:2406.06484',
    url: 'https://arxiv.org/abs/2406.06484',
    problem: 'The delta rule had existed since 2021 and nobody could train it at scale, because '
      + 'each write depends on a read of the previous state — inherently sequential.',
    answer: 'Reformulate the delta-rule recurrence using a WY representation so it can be '
      + 'computed with matmuls in parallel across the sequence.',
    pros: [
      'Turned a three-year-old good idea into a trainable one. This is the paper that made DeltaNet practical.',
      'Retains exact delta-rule semantics — it is a reformulation, not an approximation.',
      'Unlocked everything downstream: Gated DeltaNet, Qwen3-Next, Olmo Hybrid all depend on this.',
    ],
    cons: [
      'The kernel work is substantial and hardware-specific — the maths being parallel does not make it fast for free.',
      'Still a fixed-size state with all the capacity limits that implies.',
    ],
    pick: 'The reference implementation of delta-rule linear attention. Worth internalising the '
      + 'general lesson: the gap between a good idea and a usable one is often three years of '
      + 'kernel engineering.',
    verified: 'Title, authors and v1 date 10 Jun 2024 from the arXiv abstract page.',
  },
  {
    act: 'memory',
    family: 'core',
    date: '2024-10-07',
    name: 'Differential attention',
    who: 'Ye, Dong, Xia, Sun, Zhu, Huang, Wei',
    where: 'arXiv:2410.05258',
    url: 'https://arxiv.org/abs/2410.05258',
    problem: 'Softmax assigns non-trivial weight to irrelevant context, and that noise gets worse '
      + 'as context gets longer — part of why long-context models lose the thread.',
    answer: 'Compute two separate softmax attention maps and subtract one from the other, so '
      + 'common-mode noise cancels.',
    pros: [
      'Attacks a quality problem rather than a cost problem — unusual on this page, and needed.',
      'Reported better long-context retrieval and reduced activation outliers, which helps quantisation.',
      'Simple idea, borrowed straight from differential signalling in electronics.',
    ],
    cons: [
      'Two attention maps, so roughly double the attention work — you are spending cost to buy quality.',
      'Fewer effective heads for the same budget, since they now pair up.',
      'Newer and less battle-tested than most entries here.',
    ],
    pick: 'When long-context quality, not cost, is what is failing. Include it in the '
      + 'conversation because it is a reminder that not every problem with attention is a bill.',
    verified: 'Title, authors and v1 date 7 Oct 2024 from the arXiv abstract page. Abstract '
      + 'confirms attention scores as the difference between two softmax maps. ICLR 2025 oral.',
  },
  {
    act: 'memory',
    family: 'linear',
    date: '2024-12-09',
    name: 'Gated DeltaNet',
    who: 'Yang, Kautz, Hatamizadeh',
    where: 'arXiv:2412.06464',
    url: 'https://arxiv.org/abs/2412.06464',
    isKey: true,
    problem: 'The delta rule can correct a stored association but has no way to *forget* '
      + 'wholesale. Mamba2 gates well but writes crudely. Each has what the other lacks.',
    answer: 'Combine them: a gating mechanism for adaptive forgetting plus the delta rule for '
      + 'precise correction. "Improving Mamba2 with Delta Rule", as the title puts it.',
    pros: [
      'Two complementary controls — decide how much to forget AND write only the correction. Neither alone is sufficient.',
      'Beat both parents on language modelling and recall benchmarks.',
      'Became a real production default: Qwen3-Next and Olmo Hybrid both build on it.',
    ],
    cons: [
      'Fixed-size state, still. Better write and forget rules manage capacity; they do not add it.',
      'More gates means more to tune and more that can be mis-initialised.',
      'Ships almost exclusively inside hybrids, which suggests it is still not sufficient alone.',
    ],
    pick: 'The current best choice for the fixed-state layers of a hybrid. If you are building D '
      + 'layers in 2026, build these.',
    verified: 'Title "Gated Delta Networks: Improving Mamba2 with Delta Rule", authors and v1 date '
      + '9 Dec 2024 from the arXiv abstract page.',
  },
  {
    act: 'memory',
    family: 'hybrid',
    date: '2025-01-14',
    name: 'Lightning attention at scale (MiniMax-01)',
    who: 'MiniMax team (89 authors)',
    where: 'arXiv:2501.08313',
    url: 'https://arxiv.org/abs/2501.08313',
    problem: 'Linear attention had never been demonstrated at frontier scale. Every result was on '
      + 'small models, so nobody knew whether the quality gap widened or narrowed.',
    answer: 'Build a frontier-scale model on lightning attention with periodic softmax layers, and '
      + 'publish the result.',
    pros: [
      'First strong evidence that linear-attention hybrids scale to frontier size.',
      'Very long context at serving costs a pure-softmax model of that size could not reach.',
    ],
    cons: [
      'Hybrid, not pure — which is itself the finding: even its authors would not ship linear attention alone.',
      'Architectural details are less reproducible than a focused method paper.',
    ],
    pick: 'Cite it as scale evidence rather than as a mechanism to copy. Its value is answering '
      + '"does this survive at 400B?" with "yes, if you mix it".',
    verified: 'Title and v1 date 14 Jan 2025 from the arXiv abstract page.',
    caveat: 'The abstract does not state the linear-to-softmax layer ratio; that is in the full '
      + 'report, and I have not verified a specific ratio here.',
  },
  {
    act: 'memory',
    family: 'sparse',
    date: '2025-02-16',
    name: 'Native Sparse Attention (NSA)',
    who: 'Yuan, Gao, Dai, Luo, Zhao, Zhang, Xie, Wei, Wang, Xiao, Wang, Ruan, Zhang, Liang, Zeng',
    where: 'arXiv:2502.11089',
    url: 'https://arxiv.org/abs/2502.11089',
    isKey: true,
    problem: 'Sparse attention had always been bolted onto a dense-trained model at inference, so '
      + 'the model never learned to use its own sparsity. And most sparse patterns were '
      + 'hardware-hostile, so the theoretical saving never materialised.',
    answer: 'Three branches at once — compressed coarse tokens, selected fine tokens, and a local '
      + 'sliding window — designed to be both natively trainable end-to-end and aligned with what '
      + 'the hardware actually does fast.',
    pros: [
      'Trained sparse from scratch, so the model adapts to the pattern instead of tolerating it.',
      'Hardware-aligned by design: the wall-clock speedup is real, not just a FLOP count.',
      'Three branches cover different ranges — this is why it does not have the single-scale weakness of a plain window.',
    ],
    cons: [
      'Substantially more complex than any single-mechanism alternative, with more to get wrong.',
      'Native training means you cannot retrofit an existing checkpoint — it is a pretraining commitment.',
      'Block-level selection can still miss a token that matters if its block scores low.',
    ],
    pick: 'A leading answer for a new long-context model if you are willing to commit at '
      + 'pretraining time. The "natively trainable" framing is the important contribution.',
    verified: 'Title, authors and v1 date 16 Feb 2025 from the arXiv abstract page.',
  },
  {
    act: 'memory',
    family: 'hybrid',
    date: '2025-09-11',
    dateApprox: true,
    name: 'Qwen3-Next — 3:1 Gated DeltaNet / Gated Attention',
    who: 'Qwen team, Alibaba',
    where: 'Qwen3-Next-80B-A3B release',
    url: 'https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct',
    isKey: true,
    problem: 'Hybrids clearly worked, but every team picked a different ratio and nobody knew '
      + 'whether the ratio mattered or what it should be.',
    answer: 'Ship 48 layers as twelve repeats of three Gated DeltaNet blocks followed by one '
      + 'Gated Attention block — 36 recurrent, 12 softmax — at 256K context.',
    pros: [
      'Production-scale evidence for a specific, stated ratio rather than a vague "mostly linear".',
      'Serving economics of a mostly-recurrent network with enough attention layers to keep exact recall.',
      'Open weights, so the architecture is inspectable rather than described.',
    ],
    cons: [
      'The 3:1 ratio is a reported choice, not an ablation — same limitation as V4\'s DDDGDDDG.',
      'A hybrid is two mechanisms to implement, tune and serve.',
    ],
    pick: 'The strongest available reference point if you are choosing a hybrid schedule today.',
    verified: 'Confirmed via the vLLM support announcement and the Hugging Face model card: 48 '
      + 'layers, 3:1 Gated DeltaNet to Gated Attention, 36 recurrent and 12 attention layers, '
      + '256K context, released September 2025.',
    caveat: 'Exact release day not pinned to a primary announcement; vLLM day-0 support is dated '
      + '11 Sep 2025, which is what the timeline uses. Treat the month as reliable, the day as '
      + 'approximate.',
    note: 'CONVERGENT EVIDENCE, and worth flagging: 3 DeltaNet : 1 Attention is the same ratio as '
      + 'V4\'s DDDGDDDG (6 D and 2 G per 8 = 3:1). Two independent teams landed on one exact-access '
      + 'layer per four. That is stronger support for the ratio than either team\'s own report, and '
      + 'it partially answers the §16 open question about whether the ratio was arbitrary.',
  },
  {
    act: 'memory',
    family: 'sparse',
    date: '2025-09-29',
    name: 'DeepSeek Sparse Attention (DSA) — V3.2-Exp',
    who: 'DeepSeek-AI',
    where: 'DeepSeek-V3.2-Exp release',
    url: 'https://api-docs.deepseek.com/news/news250929/',
    problem: 'NSA required committing to sparsity at pretraining. Could fine-grained sparsity be '
      + 'introduced into an existing strong model instead?',
    answer: 'A "lightning indexer" cheaply scores which tokens deserve attention, then '
      + 'fine-grained top-k selection reads only those — layered onto V3.1-Terminus rather than '
      + 'trained from scratch.',
    pros: [
      'Fine-grained token-level selection rather than block-level, so less is missed than NSA-style block choice.',
      'Reported near-identical output quality with substantially better long-context efficiency.',
      'The economics were passed on visibly: API prices cut by more than half.',
      'Open weights under MIT.',
    ],
    cons: [
      'The indexer is another component to train and can itself mis-rank — §7\'s catch, relocated rather than removed.',
      'Explicitly experimental ("-Exp"), so treat it as a strong signal rather than a settled design.',
      '128K context, which is shorter than several competitors.',
    ],
    pick: 'The current reference for adding sparsity to a strong existing model. Read alongside '
      + 'NSA — the two bracket the "train sparse" versus "retrofit sparse" choice.',
    verified: 'Release date 29 Sep 2025 and DSA with lightning indexer plus fine-grained top-k '
      + 'confirmed from the official DeepSeek API announcement, the Hugging Face model card and '
      + 'the vLLM day-0 article. 128K context and the 50%+ price cut confirmed from the same.',
  },
  {
    act: 'memory',
    family: 'extension',
    date: '2025-12-13',
    name: 'DroPE — drop the positional embedding',
    who: 'Gelberg, Eguchi, Akiba, Cetin (Sakana AI)',
    where: 'arXiv:2512.12167',
    url: 'https://arxiv.org/abs/2512.12167',
    isKey: true,
    problem: 'Every extension method since PI rescales RoPE, and all of them are repairs to a '
      + 'signal learned at the wrong scale. What if the positional embedding is the problem?',
    answer: 'Treat RoPE as a training-time scaffold. Pretrain with it, then remove positional '
      + 'embeddings entirely and briefly recalibrate at the *original* context length. The model '
      + 'recovers its perplexity and generalises to lengths it never saw.',
    pros: [
      'Zero-shot context extension with no long-context fine-tuning at all — the expensive part is skipped entirely.',
      'Reported to need under 1% of the original pretraining budget to recalibrate.',
      'Reported to outperform established RoPE-scaling methods on LongBench and RULER.',
      'Reframes the problem beautifully: position helps you learn, then holds you back. That is a genuinely new claim.',
    ],
    cons: [
      'Requires a recalibration pass, so it is not free and not purely inference-time.',
      'Very new — December 2025 — with correspondingly little independent replication.',
      'Removing positional information entirely is a large bet; how it interacts with hybrids and sparse attention is not yet characterised.',
    ],
    pick: 'Evaluate it seriously against YaRN if you need extension today. It is the first method '
      + 'in this act that changes the question rather than refining the answer.',
    verified: 'Title "Extending the Context of Pretrained LLMs by Dropping Their Positional '
      + 'Embeddings", authors Gelberg, Eguchi, Akiba, Cetin, v1 date 13 Dec 2025, from the arXiv '
      + 'abstract page. Mechanism and the <1% recalibration budget confirmed from the Sakana AI '
      + 'project page.',
    warning: 'THREE DIFFERENT THINGS ARE CALLED DROPE. See the corrections note on this page — '
      + 'this is the one that means LLM context extension.',
  },
  {
    act: 'memory',
    family: 'hybrid',
    date: '2026-04-03',
    name: 'Olmo Hybrid',
    who: 'Merrill, Li, Romero, Svete, Costello, Dasigi, Groeneveld, Heineman, et al. (AI2)',
    where: 'arXiv:2604.03444',
    url: 'https://arxiv.org/abs/2604.03444',
    problem: 'Hybrid architectures were shipping from labs that did not publish their reasoning. '
      + 'What does the theory actually say about which layers should be which?',
    answer: 'A 7B fully-open hybrid with the sliding-window layers replaced by Gated DeltaNet '
      + 'layers, explicitly connecting expressivity theory to a practical build and back.',
    pros: [
      'Fully open — data, code, weights — so the hybrid choice is auditable rather than asserted.',
      'Takes the theory seriously instead of only reporting benchmarks.',
      'Directly substitutes a state layer where a sliding window used to be, which is a clean comparison.',
    ],
    cons: [
      '7B, so it does not answer the frontier-scale question.',
      'Very recent; little downstream evidence yet.',
    ],
    pick: 'Read it if you are choosing a schedule and want reasoning rather than a leaderboard. '
      + 'Its swap — window layer becomes state layer — is a useful way to think about what each is for.',
    verified: 'Title, author list and v1 date 3 Apr 2026 from the arXiv abstract page. Sliding '
      + 'window layers replaced by Gated DeltaNet layers confirmed from the abstract.',
    caveat: 'The abstract does not state the attention-to-recurrent ratio, so no ratio is claimed here.',
  },
  {
    act: 'memory',
    family: 'position',
    date: '2026-05-27',
    name: 'Periodic RoPE',
    who: 'Simin Huo',
    where: 'arXiv:2605.27980',
    url: 'https://arxiv.org/abs/2605.27980',
    problem: '"Position exhaustion" — once a sequence passes the pretrained positional range, '
      + 'quality degrades, and every extension method only pushes that boundary further out '
      + 'rather than removing it.',
    answer: 'Split the job. Local sliding-window layers use a periodic RoPE within a bounded '
      + 'window; global layers use no positional encoding at all, so they have nothing to '
      + 'exhaust.',
    pros: [
      'Attacks the ceiling itself rather than raising it — if position never exceeds one window, extrapolation never happens.',
      'Sits naturally alongside the DroPE finding that global layers may not need position at all.',
    ],
    cons: [
      'Evaluated at small scale against small baselines; the claims are not frontier-validated.',
      'A theoretical pathway to infinite context is not a demonstration of one.',
      'Two positional regimes in one network is added complexity.',
    ],
    pick: 'Not a production choice. Included because it shows where the position thread is '
      + 'heading — and because it and DroPE arriving within six months of each other, both '
      + 'concluding that global layers can drop position, is a pattern worth watching.',
    verified: 'Title, author and v1 date 27 May 2026 from the arXiv abstract page. Mechanism and '
      + 'the "position exhaustion" framing confirmed from the abstract.',
    caveat: 'Claims are the authors\' own and compared against small baselines (MiniWin vs '
      + 'MiniMind). Treat as a direction, not a result.',
  },
];

/* Where the arc points next — stated as questions, not predictions. */
export const NEXT = [
  {
    q: 'Does position survive at all in the long-range layers?',
    why: 'DroPE (Dec 2025) removes positional embeddings entirely after pretraining and '
      + 'generalises better for it. Periodic RoPE (May 2026) independently puts no positional '
      + 'encoding in its global layers. Two papers, six months apart, reaching the same conclusion '
      + 'from different directions. If that holds, twelve years of positional-encoding design ends '
      + 'up as scaffolding you remove before serving.',
  },
  {
    q: 'Is the hybrid ratio converging on 3:1?',
    why: 'LightningLM V4 used DDDGDDDG — 6 D and 2 G per 8, which is 3:1. Qwen3-Next ships 48 '
      + 'layers as 36 Gated DeltaNet and 12 Gated Attention — also 3:1. Two independent teams, one '
      + 'exact-access layer per four. Nobody has ablated it properly, so this is convergence rather '
      + 'than proof, but it is the most testable open question on the page.',
  },
  {
    q: 'Train sparse, or retrofit sparse?',
    why: 'NSA (Feb 2025) argues sparsity must be native and trained end-to-end. DSA (Sep 2025) '
      + 'layers fine-grained sparsity onto an already-strong dense model and reports near-identical '
      + 'quality. Both from DeepSeek, seven months apart. Whichever wins determines whether '
      + 'long-context is a pretraining commitment or a serving decision.',
  },
  {
    q: 'Which compression axis wins — or do they compose?',
    why: 'GQA compresses heads. MLA compresses features. NSA and DSA compress the sequence. '
      + 'Quantisation compresses bits. These are orthogonal, and almost nobody has published a '
      + 'careful study of all four together. The multiplicative saving is the obvious next thing '
      + 'to measure.',
  },
  {
    q: 'Was the whole approximation programme premature?',
    why: 'FlashAttention (2022) made exact attention so much cheaper that it retired a lot of '
      + '2019–2021 approximate work overnight, and Ring Attention (2023) does exact attention at '
      + 'near-unbounded length given enough interconnect. The uncomfortable question: how much of '
      + 'this timeline is people approximating a mechanism that was memory-bound, not compute-bound?',
  },
];
