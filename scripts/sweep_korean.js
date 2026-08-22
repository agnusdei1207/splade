// Does the lexical layer's Korean blind spot explain the ranking gap, and what
// closes it? pentesting tokenizes on `is_ascii_alphanumeric`, so every Hangul
// character is a separator. This measures the current behaviour against three
// cheap tokenizer changes and against SPLADE.

const fs = require("fs");
const S = require("./sweep_scale.js");

const DUMP = process.argv[2] || "artifacts/scale";
const read = n => JSON.parse(fs.readFileSync(`${DUMP}/${n}`, "utf8"));
const queries = read("queries.json");
const qvec = Object.fromEntries(read("query-vectors.json").map(v => [v.id, v]));
const dvec = read("document-vectors.json");
const corpus = S.buildCorpus();

const HANGUL = /[가-힣]/;
const koRatio = t => {
  const h = (t.match(/[가-힣]/g) || []).length;
  const a = (t.match(/[A-Za-z]/g) || []).length;
  return h + a ? h / (h + a) : 0;
};

// ── tokenizer variants ────────────────────────────────────────────────────────
// current: ASCII alphanumeric runs only (Hangul is a separator)
const current = S.tokenizeWithIdentifiers;

// unigram: keep the ASCII behaviour, add every Hangul syllable as its own term
function hangulUnigram(text) {
  const terms = current(text);
  const seen = new Set(terms);
  for (const ch of text.match(/[가-힣]/g) || []) if (!seen.has(ch)) { terms.push(ch); seen.add(ch); }
  return terms;
}

// bigram: ASCII behaviour plus Hangul character bigrams, the standard CJK analyzer trick
function hangulBigram(text) {
  const terms = current(text);
  const seen = new Set(terms);
  for (const run of text.match(/[가-힣]+/g) || []) {
    if (run.length === 1) { if (!seen.has(run)) { terms.push(run); seen.add(run); } continue; }
    for (let i = 0; i + 1 < run.length; i++) {
      const g = run.slice(i, i + 2);
      if (!seen.has(g)) { terms.push(g); seen.add(g); }
    }
  }
  return terms;
}

// bigram+word: bigrams plus whitespace-delimited Hangul words (catches exact eojeol matches)
function hangulBigramWord(text) {
  const terms = hangulBigram(text);
  const seen = new Set(terms);
  for (const w of text.split(/\s+/)) {
    const clean = w.replace(/[^가-힣]/g, "");
    if (clean.length >= 2 && !seen.has(clean)) { terms.push(clean); seen.add(clean); }
  }
  return terms;
}

const VARIANTS = {
  "현행 (ASCII 전용)": current,
  "한글 유니그램 추가": hangulUnigram,
  "한글 바이그램 추가": hangulBigram,
  "한글 바이그램+어절": hangulBigramWord,
};

// ── evaluation ────────────────────────────────────────────────────────────────
const koQueries = queries.filter(q => HANGUL.test(q.query));
const enQueries = queries.filter(q => !HANGUL.test(q.query));
const pureKo = queries.filter(q => koRatio(q.query) > 0.8);

const fmt = m => ["R@1", "R@5", "R@10", "MRR@10", "nDCG@10"].map(k => m[k].toFixed(4)).join("  ");

console.log("=== 토크나이저 변형별 BM25 ===");
console.log("  변형                  어휘      평균길이   구간        R@1     R@5     R@10    MRR@10  nDCG@10");
const built = {};
for (const [name, tok] of Object.entries(VARIANTS)) {
  const patched = corpus.map(d => d);
  const saved = S.tokenizeWithIdentifiers;
  // buildBm25/bm25Search call the exported tokenizer, so swap the module binding
  require("./sweep_scale.js").tokenizeWithIdentifiers = tok;
  const module_ = require("./sweep_scale.js");
  const ix = buildWith(tok, patched);
  built[name] = ix;
  const rank = {};
  for (const q of queries) rank[q.id] = S.idsOf(searchWith(tok, ix, q.query, 24));
  for (const [label, set] of [["전체", queries], ["한글포함", koQueries], ["한글위주", pureKo], ["영어전용", enQueries]]) {
    const sub = {};
    for (const q of set) sub[q.id] = rank[q.id];
    const m = S.evaluate(set, sub);
    const head = label === "전체" ? `  ${name.padEnd(20)} ${String(ix.df.size).padStart(6)}  ${ix.avg.toFixed(1).padStart(8)}` : "  " + " ".repeat(38);
    console.log(`${head}   ${label.padEnd(9)} ${fmt(m)}`);
  }
  require("./sweep_scale.js").tokenizeWithIdentifiers = saved;
}

// local build/search that take an explicit tokenizer
function buildWith(tok, docs) {
  const terms = new Map(), lengths = new Map(), df = new Map(), lower = new Map();
  let total = 0;
  for (const d of docs) {
    const t = tok(d.text);
    terms.set(d.id, new Set(t));
    lengths.set(d.id, t.length);
    lower.set(d.id, d.text.toLowerCase());
    total += t.length;
    for (const term of new Set(t)) df.set(term, (df.get(term) || 0) + 1);
  }
  return { docs, terms, lengths, df, lower, avg: total / docs.length, n: docs.length };
}
function searchWith(tok, ix, query, limit) {
  const qt = tok(query);
  if (!qt.length) return [];
  const ql = query.toLowerCase();
  const hits = [];
  for (const d of ix.docs) {
    const has = ix.terms.get(d.id), dl = ix.lengths.get(d.id);
    let score = 0;
    for (const term of qt) {
      if (!has.has(term)) continue;
      const dfv = ix.df.get(term) || 0;
      const idf = Math.log((ix.n - dfv + 0.5) / (dfv + 0.5) + 1.0);
      const denom = 1 + 1.4 * (1.0 - 0.75 + 0.75 * (dl / Math.max(ix.avg, 1.0)));
      score += (idf * 2.4) / Math.max(denom, 1e-5);
    }
    if (ix.lower.get(d.id).includes(ql)) score += 0.15;
    if (score > 0) hits.push([d.id, score]);
  }
  hits.sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1));
  return hits.slice(0, limit);
}

// ── best lexical + SPLADE ─────────────────────────────────────────────────────
console.log("\n=== 최적 토크나이저 + SPLADE 결합 ===");
const sparseIx = S.buildSparseIndex(dvec.map(v => S.truncate(v, 256)));
const spRank = {};
for (const q of queries) spRank[q.id] = S.idsOf(S.sparseSearch(sparseIx, S.truncate(qvec[q.id], 32), 24));

const bestTok = hangulBigramWord;
const bestIx = buildWith(bestTok, corpus);
const bestRank = {};
for (const q of queries) bestRank[q.id] = S.idsOf(searchWith(bestTok, bestIx, q.query, 24));
const curIx = buildWith(current, corpus);
const curRank = {};
for (const q of queries) curRank[q.id] = S.idsOf(searchWith(current, curIx, q.query, 24));

console.log("  구성                                  구간        R@10    MRR@10  nDCG@10");
const rows = [
  ["현행 BM25", curRank],
  ["한글대응 BM25", bestRank],
  ["SPLADE 단독", spRank],
];
for (const head of [5, 8]) {
  const r = {};
  for (const q of queries) r[q.id] = S.cascade(bestRank[q.id], spRank[q.id], head, 24);
  rows.push([`한글BM25${head} + SPLADE 보충`, r]);
}
{
  const r = {};
  for (const q of queries) r[q.id] = S.rrf(bestRank[q.id], spRank[q.id], 60, 0.7, 24);
  rows.push(["RRF 한글BM25 0.7 / SPLADE 0.3", r]);
}
for (const [name, rank] of rows) {
  for (const [label, set] of [["전체", queries], ["한글위주", pureKo], ["영어전용", enQueries]]) {
    const sub = {};
    for (const q of set) sub[q.id] = rank[q.id];
    const m = S.evaluate(set, sub);
    const head = label === "전체" ? `  ${name.padEnd(36)}` : "  " + " ".repeat(36);
    console.log(`${head}  ${label.padEnd(9)} ${m["R@10"].toFixed(4)}  ${m["MRR@10"].toFixed(4)}  ${m["nDCG@10"].toFixed(4)}`);
  }
}

console.log("\n=== 색인 비용 ===");
for (const [name, tok] of [["현행", current], ["한글 바이그램+어절", hangulBigramWord]]) {
  const t0 = Date.now();
  const ix = buildWith(tok, corpus);
  const ms = Date.now() - t0;
  const postings = [...ix.terms.values()].reduce((s, x) => s + x.size, 0);
  console.log(`  ${name.padEnd(18)} 어휘 ${String(ix.df.size).padStart(7)}  posting ${String(postings).padStart(8)}  색인 ${String(ms).padStart(5)}ms  추정 인덱스 ${(postings * 4 / 1048576).toFixed(2)} MiB`);
}
