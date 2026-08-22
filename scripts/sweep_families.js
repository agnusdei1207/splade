// Compare every candidate lexical/sparse arrangement across query families and
// languages, using the prepared corpus so all engines see identical text.
//
// Usage: node scripts/sweep_families.js artifacts/families model-dir[:label] ...

const fs = require("fs");

const FAM = process.argv[2] || "artifacts/families";
const MODEL_ARGS = process.argv.slice(3);

const documents = JSON.parse(fs.readFileSync(`${FAM}/documents.json`, "utf8"));
const queries = JSON.parse(fs.readFileSync(`${FAM}/queries.json`, "utf8"));

// ── BM25, mirroring pentesting hybrid_search.rs ───────────────────────────────
const isLower = c => c >= "a" && c <= "z";
const isDigit = c => c >= "0" && c <= "9";
const isUpper = c => c >= "A" && c <= "Z";
function splitIdentifier(token) {
  const out = [];
  for (const part of token.split(/[_-]/).filter(Boolean)) {
    let cur = "";
    for (let i = 0; i < part.length; i++) {
      const ch = part[i];
      const prev = i > 0 && (isLower(part[i - 1]) || isDigit(part[i - 1]));
      if (isUpper(ch) && prev && cur) { const l = cur.toLowerCase(); if (!out.includes(l)) out.push(l); cur = ""; }
      cur += ch;
    }
    const l = cur.toLowerCase();
    if (l && !out.includes(l)) out.push(l);
  }
  return out.length ? out : [token.toLowerCase()];
}
function tokenize(text) {
  const terms = [], seen = new Set();
  for (const tok of text.split(/[^A-Za-z0-9_-]+/)) {
    if (!tok || tok.length < 2) continue;
    const lo = tok.toLowerCase();
    if (!seen.has(lo)) { terms.push(lo); seen.add(lo); }
    for (const sub of splitIdentifier(tok)) if (!seen.has(sub)) { terms.push(sub); seen.add(sub); }
  }
  return terms;
}
function buildBm25(docs) {
  const terms = new Map(), lengths = new Map(), df = new Map(), lower = new Map();
  let total = 0;
  for (const d of docs) {
    const t = tokenize(d.text);
    terms.set(d.id, new Set(t));
    lengths.set(d.id, t.length);
    lower.set(d.id, d.text.toLowerCase());
    total += t.length;
    for (const term of new Set(t)) df.set(term, (df.get(term) || 0) + 1);
  }
  return { docs, terms, lengths, df, lower, avg: total / docs.length, n: docs.length };
}
function bm25Search(ix, query, limit) {
  const qt = tokenize(query);
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
  return hits.slice(0, limit).map(h => h[0]);
}

// ── sparse ────────────────────────────────────────────────────────────────────
function truncate(v, k) {
  if (v.term_ids.length <= k) return v;
  const pairs = v.term_ids.map((t, i) => [t, v.weights[i]]);
  pairs.sort((a, b) => b[1] - a[1] || a[0] - b[0]);
  const kept = pairs.slice(0, k).sort((a, b) => a[0] - b[0]);
  return { id: v.id, term_ids: kept.map(p => p[0]), weights: kept.map(p => p[1]) };
}
function buildSparse(vectors) {
  const postings = new Map();
  let count = 0;
  for (const v of vectors) {
    for (let i = 0; i < v.term_ids.length; i++) {
      const t = v.term_ids[i];
      let list = postings.get(t);
      if (!list) postings.set(t, (list = []));
      list.push([v.id, v.weights[i]]);
      count++;
    }
  }
  return { postings, count };
}
function sparseSearch(ix, qv, limit) {
  const scores = new Map();
  for (let i = 0; i < qv.term_ids.length; i++) {
    const list = ix.postings.get(qv.term_ids[i]);
    if (!list) continue;
    const qw = qv.weights[i];
    for (const [id, dw] of list) scores.set(id, (scores.get(id) || 0) + qw * dw);
  }
  return [...scores.entries()].sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1)).slice(0, limit).map(e => e[0]);
}

// ── metrics ───────────────────────────────────────────────────────────────────
function recall(rank, rel, k) { const S = new Set(rel); let n = 0; for (const d of rank.slice(0, k)) if (S.has(d)) n++; return S.size ? n / S.size : 0; }
function mrr(rank, rel, k) { const S = new Set(rel); for (let i = 0; i < Math.min(k, rank.length); i++) if (S.has(rank[i])) return 1 / (i + 1); return 0; }
function ndcg(rank, rel, k) {
  const S = new Set(rel);
  const dcg = g => g.reduce((s, v, i) => s + v / Math.log2(i + 2), 0);
  const ideal = dcg(new Array(Math.min(k, S.size)).fill(1));
  return ideal ? dcg(rank.slice(0, k).map(d => (S.has(d) ? 1 : 0))) / ideal : 0;
}
function evaluate(qs, rankings) {
  if (!qs.length) return null;
  const m = f => qs.reduce((s, q) => s + f(rankings[q.id] || [], q.relevance), 0) / qs.length;
  return {
    "R@1": m((r, l) => recall(r, l, 1)),
    "R@10": m((r, l) => recall(r, l, 10)),
    "MRR@10": m((r, l) => mrr(r, l, 10)),
    "nDCG@10": m((r, l) => ndcg(r, l, 10)),
  };
}
const fmt = m => (m ? ["R@1", "R@10", "MRR@10", "nDCG@10"].map(k => m[k].toFixed(4)).join("  ") : "     -");

// ── fusion ────────────────────────────────────────────────────────────────────
function cascade(primary, backfill, head, limit) {
  const out = [], seen = new Set();
  const push = d => { if (!seen.has(d) && out.length < limit) { seen.add(d); out.push(d); } };
  primary.slice(0, head).forEach(push);
  backfill.forEach(push);
  primary.slice(head).forEach(push);
  return out;
}
function rrf(a, b, k, wa, limit) {
  const s = new Map();
  a.forEach((d, i) => s.set(d, (s.get(d) || 0) + wa / (k + i + 1)));
  b.forEach((d, i) => s.set(d, (s.get(d) || 0) + (1 - wa) / (k + i + 1)));
  return [...s.entries()].sort((x, y) => y[1] - x[1] || (x[0] < y[0] ? -1 : 1)).slice(0, limit).map(e => e[0]);
}

// ── run ───────────────────────────────────────────────────────────────────────
console.log(`corpus ${documents.length}문서 / 질의 ${queries.length}개`);
let t0 = Date.now();
const bm = buildBm25(documents);
const bmBuildMs = Date.now() - t0;
t0 = Date.now();
const bmRank = {};
for (const q of queries) bmRank[q.id] = bm25Search(bm, q.query, 24);
const bmSearchMs = Date.now() - t0;
console.log(`BM25  어휘 ${bm.df.size}  평균길이 ${bm.avg.toFixed(1)}  색인 ${bmBuildMs}ms  검색 ${(bmSearchMs / queries.length).toFixed(3)}ms/질의`);

const engines = { "BM25 (현행)": bmRank };
const modelMeta = {};
for (const arg of MODEL_ARGS) {
  const [dir, label] = arg.split("=");
  const name = label || dir.split("/").pop();
  const summary = JSON.parse(fs.readFileSync(`${dir}/summary.json`, "utf8"));
  const dv = JSON.parse(fs.readFileSync(`${dir}/document-vectors.json`, "utf8"));
  const qv = Object.fromEntries(JSON.parse(fs.readFileSync(`${dir}/query-vectors.json`, "utf8")).map(v => [v.id, v]));
  modelMeta[name] = summary;
  const docK = Number(process.env.DOC_K || 512);
  const qryK = Number(process.env.QUERY_K || 64);
  const ix = buildSparse(dv.map(v => truncate(v, docK)));
  t0 = Date.now();
  const rank = {};
  for (const q of queries) rank[q.id] = qv[q.id] ? sparseSearch(ix, truncate(qv[q.id], qryK), 24) : [];
  const ms = Date.now() - t0;
  engines[name] = rank;
  console.log(`${name}  vocab ${summary.vocab_size}  posting ${ix.count}  검색 ${(ms / queries.length).toFixed(3)}ms/질의  인코딩 ${summary.documents_per_second.toFixed(2)} docs/s  모델 ${(summary.snapshot_bytes / 1048576).toFixed(0)}MiB`);
}

const FAMILIES = ["heading", "title", "sentence", "identifier", "question"];
const byFamily = Object.fromEntries(FAMILIES.map(f => [f, queries.filter(q => q.family === f)]));
const byLang = { ko: queries.filter(q => q.language === "ko"), en: queries.filter(q => q.language === "en") };

console.log("\n=== 엔진별 · 전체 ===");
console.log("  엔진                          R@1     R@10    MRR@10  nDCG@10");
for (const [name, rank] of Object.entries(engines)) console.log(`  ${name.padEnd(28)} ${fmt(evaluate(queries, rank))}`);

console.log("\n=== 질의 패밀리별 nDCG@10 ===");
const names = Object.keys(engines);
console.log("  패밀리       n      " + names.map(n => n.slice(0, 16).padEnd(17)).join(""));
for (const f of FAMILIES) {
  const qs = byFamily[f];
  console.log(`  ${f.padEnd(11)} ${String(qs.length).padStart(5)}  ` + names.map(n => evaluate(qs, engines[n])["nDCG@10"].toFixed(4).padEnd(17)).join(""));
}

console.log("\n=== 언어별 nDCG@10 ===");
console.log("  언어    n      " + names.map(n => n.slice(0, 16).padEnd(17)).join(""));
for (const [lang, qs] of Object.entries(byLang)) {
  if (!qs.length) continue;
  console.log(`  ${lang.padEnd(6)} ${String(qs.length).padStart(5)}  ` + names.map(n => evaluate(qs, engines[n])["nDCG@10"].toFixed(4).padEnd(17)).join(""));
}

console.log("\n=== BM25 자리 결합안 (sparse 엔진마다) ===");
for (const name of names.slice(1)) {
  console.log(`  --- ${name} ---`);
  const sp = engines[name];
  const combos = { [`${name} 단독`]: sp };
  for (const h of [3, 5, 8]) {
    const r = {}; for (const q of queries) r[q.id] = cascade(sp[q.id], bmRank[q.id], h, 24);
    combos[`캐스케이드 sparse${h}+BM25`] = r;
  }
  for (const h of [3, 5, 8]) {
    const r = {}; for (const q of queries) r[q.id] = cascade(bmRank[q.id], sp[q.id], h, 24);
    combos[`캐스케이드 BM25${h}+sparse`] = r;
  }
  for (const w of [0.3, 0.5, 0.7]) {
    const r = {}; for (const q of queries) r[q.id] = rrf(sp[q.id], bmRank[q.id], 60, w, 24);
    combos[`RRF w_sparse=${w}`] = r;
  }
  console.log("    구성                          전체nDCG   ko-nDCG   en-nDCG");
  console.log(`    ${"BM25 단독(기준)".padEnd(28)} ${evaluate(queries, bmRank)["nDCG@10"].toFixed(4)}    ${evaluate(byLang.ko, bmRank)["nDCG@10"].toFixed(4)}    ${evaluate(byLang.en, bmRank)["nDCG@10"].toFixed(4)}`);
  for (const [label, rank] of Object.entries(combos)) {
    console.log(`    ${label.padEnd(28)} ${evaluate(queries, rank)["nDCG@10"].toFixed(4)}    ${evaluate(byLang.ko, rank)["nDCG@10"].toFixed(4)}    ${evaluate(byLang.en, rank)["nDCG@10"].toFixed(4)}`);
  }
}
