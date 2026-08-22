// Close three gaps the plan currently states as speculation.
//
// 1. Link density. The repo docs carry 214 links; an Obsidian vault carries far
//    more. If the additive prior distortion grows with density, the plan's
//    "probably worse in the real vault" line should be a measurement, not a guess.
// 2. Significance. 5495 queries is enough to put confidence intervals on the
//    headline deltas instead of quoting point estimates.
// 3. Search latency at scale. Index bytes were projected to 30k documents but
//    query latency was not.

const fs = require("fs");
const path = require("path");

const FAM = process.argv[2] || "artifacts/families";
const DENSE = process.argv[3] || "artifacts/dense";
const SPARSE_DIR = process.argv[4] || "artifacts/models/f-multilingual-v1";

const documents = JSON.parse(fs.readFileSync(`${FAM}/documents.json`, "utf8"));
const queries = JSON.parse(fs.readFileSync(`${FAM}/queries.json`, "utf8"));
const denseRankings = JSON.parse(fs.readFileSync(`${DENSE}/rankings.json`, "utf8"));

const RRF_K = 60, LIMIT = 24, SEED_LIMIT = 5, MAX_HOPS = 2;

// ── BM25 ──────────────────────────────────────────────────────────────────────
const isLower = c => c >= "a" && c <= "z", isDigit = c => c >= "0" && c <= "9", isUpper = c => c >= "A" && c <= "Z";
function splitIdentifier(token) {
  const out = [];
  for (const part of token.split(/[_-]/).filter(Boolean)) {
    let cur = "";
    for (let i = 0; i < part.length; i++) {
      const ch = part[i], prev = i > 0 && (isLower(part[i - 1]) || isDigit(part[i - 1]));
      if (isUpper(ch) && prev && cur) { const l = cur.toLowerCase(); if (!out.includes(l)) out.push(l); cur = ""; }
      cur += ch;
    }
    const l = cur.toLowerCase(); if (l && !out.includes(l)) out.push(l);
  }
  return out.length ? out : [token.toLowerCase()];
}
function tokenize(text) {
  const terms = [], seen = new Set();
  for (const tok of text.split(/[^A-Za-z0-9_-]+/)) {
    if (!tok || tok.length < 2) continue;
    const lo = tok.toLowerCase();
    if (!seen.has(lo)) { terms.push(lo); seen.add(lo); }
    for (const s of splitIdentifier(tok)) if (!seen.has(s)) { terms.push(s); seen.add(s); }
  }
  return terms;
}
const bm = (() => {
  const terms = new Map(), lengths = new Map(), df = new Map(), lower = new Map();
  let total = 0;
  for (const d of documents) {
    const t = tokenize(d.text);
    terms.set(d.id, new Set(t)); lengths.set(d.id, t.length); lower.set(d.id, d.text.toLowerCase());
    total += t.length;
    for (const term of new Set(t)) df.set(term, (df.get(term) || 0) + 1);
  }
  return { terms, lengths, df, lower, avg: total / documents.length, n: documents.length };
})();
function bm25(query, limit) {
  const qt = tokenize(query);
  if (!qt.length) return [];
  const ql = query.toLowerCase(), hits = [];
  for (const d of documents) {
    const has = bm.terms.get(d.id), dl = bm.lengths.get(d.id);
    let score = 0;
    for (const term of qt) {
      if (!has.has(term)) continue;
      const dfv = bm.df.get(term) || 0;
      score += (Math.log((bm.n - dfv + 0.5) / (dfv + 0.5) + 1) * 2.4) / Math.max(1 + 1.4 * (0.25 + 0.75 * (dl / Math.max(bm.avg, 1))), 1e-5);
    }
    if (bm.lower.get(d.id).includes(ql)) score += 0.15;
    if (score > 0) hits.push([d.id, score]);
  }
  hits.sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1));
  return hits.slice(0, limit);
}

// ── link graph, with a synthetic densifier ────────────────────────────────────
const WIKI = /\[\[([^\]|#]+)(?:#[^\]|]+)?(?:\|[^\]]+)?\]\]/g;
const MDLINK = /\[[^\]]+\]\(([^)\s]+\.md(?:#[^)]+)?)\)/g;
const byStem = new Map();
for (const d of documents) {
  const stem = path.basename(d.id, ".md").toLowerCase();
  if (!byStem.has(stem)) byStem.set(stem, d.id);
}
const ids = documents.map(d => d.id);
const idSet = new Set(ids);
function realLinks() {
  const adjacency = new Map(ids.map(id => [id, new Set()]));
  for (const d of documents) {
    for (const m of d.text.matchAll(WIKI)) {
      const t = byStem.get(m[1].trim().toLowerCase().replace(/\.md$/, ""));
      if (t && t !== d.id) adjacency.get(d.id).add(t);
    }
    for (const m of d.text.matchAll(MDLINK)) {
      const raw = m[1].split("#")[0];
      const resolved = path.posix.normalize(path.posix.join(path.posix.dirname(d.id), raw));
      const t = idSet.has(resolved) ? resolved : byStem.get(path.basename(raw, ".md").toLowerCase());
      if (t && t !== d.id) adjacency.get(d.id).add(t);
    }
  }
  return adjacency;
}
// Deterministic densifier: adds preferential-attachment links so hub structure
// resembles a real vault rather than a uniform random graph.
function densify(base, linksPerDocument, seed) {
  const adjacency = new Map([...base].map(([k, v]) => [k, new Set(v)]));
  let state = seed >>> 0;
  const next = () => ((state = (state * 1103515245 + 12345) >>> 0) / 4294967296);
  const degree = new Map(ids.map(id => [id, 1]));
  for (const [, targets] of adjacency) for (const t of targets) degree.set(t, degree.get(t) + 1);
  const pool = [];
  for (const id of ids) for (let i = 0; i < degree.get(id); i++) pool.push(id);
  for (const id of ids) {
    for (let i = 0; i < linksPerDocument; i++) {
      const target = pool[Math.floor(next() * pool.length)];
      if (target && target !== id) { adjacency.get(id).add(target); pool.push(target); }
    }
  }
  return adjacency;
}
function graphState(adjacency) {
  const backlinks = new Map(ids.map(id => [id, new Set()]));
  let total = 0;
  for (const [source, targets] of adjacency) {
    total += targets.size;
    for (const t of targets) backlinks.get(t).add(source);
  }
  const N = ids.length, damping = 0.85;
  let rank = new Map(ids.map(id => [id, 1 / N]));
  for (let it = 0; it < 20; it++) {
    const nextRank = new Map(ids.map(id => [id, (1 - damping) / N]));
    let sink = 0;
    for (const id of ids) {
      const out = adjacency.get(id);
      if (!out.size) { sink += rank.get(id); continue; }
      const share = (damping * rank.get(id)) / out.size;
      for (const t of out) nextRank.set(t, nextRank.get(t) + share);
    }
    for (const id of ids) nextRank.set(id, nextRank.get(id) + (damping * sink) / N);
    rank = nextRank;
  }
  const max = Math.max(...rank.values());
  return {
    adjacency, backlinks, total,
    pagerank: new Map([...rank].map(([k, v]) => [k, max ? v / max : 0])),
    backlinkCount: id => backlinks.get(id).size,
  };
}
function graphRanking(state, lexical, semantic) {
  const seedWeights = new Map();
  [...lexical.slice(0, SEED_LIMIT), ...semantic.slice(0, SEED_LIMIT)]
    .forEach(([id, score], rank) => seedWeights.set(id, (seedWeights.get(id) || 0) + score + 1 / (rank + 1)));
  if (!seedWeights.size) return [];
  const depths = new Map(), queue = [];
  for (const s of seedWeights.keys()) { depths.set(s, 0); queue.push([s, 0]); }
  while (queue.length) {
    const [current, depth] = queue.shift();
    if (depth >= MAX_HOPS) continue;
    for (const n of [...(state.adjacency.get(current) || []), ...(state.backlinks.get(current) || [])]) {
      if (!depths.has(n)) { depths.set(n, depth + 1); queue.push([n, depth + 1]); }
    }
  }
  const ranked = [...depths].map(([id, hops]) => [id, (seedWeights.get(id) || 0) + 1 / (hops + 1) + Math.log(state.backlinkCount(id) + 1) * 0.08]);
  ranked.sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1));
  return ranked.slice(0, LIMIT);
}
function fuse(state, rankings, backBoost, pageBoost) {
  const rrf = new Map();
  for (const ranking of Object.values(rankings)) ranking.forEach(([id], i) => rrf.set(id, (rrf.get(id) || 0) + 1 / (RRF_K + i + 1)));
  const scored = [...rrf].map(([id, s]) => [id, s + Math.log(state.backlinkCount(id) + 1) * backBoost + state.pagerank.get(id) * pageBoost]);
  scored.sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1));
  return scored.slice(0, LIMIT).map(e => e[0]);
}

// ── metrics ───────────────────────────────────────────────────────────────────
const ndcgOf = (r, rel, k = 10) => {
  const S = new Set(rel), dcg = g => g.reduce((s, v, i) => s + v / Math.log2(i + 2), 0);
  const ideal = dcg(new Array(Math.min(k, S.size)).fill(1));
  return ideal ? dcg(r.slice(0, k).map(d => (S.has(d) ? 1 : 0))) / ideal : 0;
};
const mean = a => a.reduce((s, x) => s + x, 0) / a.length;

const bmCache = {}, denseCache = {};
for (const q of queries) { bmCache[q.id] = bm25(q.query, LIMIT); denseCache[q.id] = (denseRankings[q.id] || []).slice(0, LIMIT); }

// ── 1. link density sensitivity ───────────────────────────────────────────────
console.log("=== 1) 링크 밀도에 따른 가산항 왜곡 ===");
console.log("  링크수    문서당   가산항 ON   가산항 OFF   손실률");
const base = realLinks();
for (const extra of [0, 1, 2, 4, 8]) {
  const adjacency = extra === 0 ? base : densify(base, extra, 20260822 + extra);
  const state = graphState(adjacency);
  const on = [], off = [];
  for (const q of queries) {
    const lex = bmCache[q.id], sem = denseCache[q.id];
    const g = graphRanking(state, lex, sem);
    on.push(ndcgOf(fuse(state, { lex, sem, g }, 0.08, 0.10), q.relevance));
    off.push(ndcgOf(fuse(state, { lex, sem, g }, 0, 0), q.relevance));
  }
  const onM = mean(on), offM = mean(off);
  console.log(`  ${String(state.total).padStart(6)}  ${(state.total / documents.length).toFixed(2).padStart(7)}   ${onM.toFixed(4).padStart(9)}   ${offM.toFixed(4).padStart(10)}   ${((1 - onM / offM) * 100).toFixed(1).padStart(6)}%`);
}

// ── 2. significance of the headline deltas ────────────────────────────────────
console.log("\n=== 2) 주요 차이의 신뢰구간 (부트스트랩 5,000회) ===");
function cascadeIds(primary, backfill, head) {
  const out = [], seen = new Set();
  const push = ([id]) => { if (!seen.has(id) && out.length < LIMIT) { seen.add(id); out.push(id); } };
  primary.slice(0, head).forEach(push); backfill.forEach(push); primary.slice(head).forEach(push);
  return out;
}
const sparse = (() => {
  const dv = JSON.parse(fs.readFileSync(`${SPARSE_DIR}/document-vectors.json`, "utf8"));
  const qv = Object.fromEntries(JSON.parse(fs.readFileSync(`${SPARSE_DIR}/query-vectors.json`, "utf8")).map(v => [v.id, v]));
  const trunc = (v, k) => {
    if (v.term_ids.length <= k) return v;
    const p = v.term_ids.map((t, i) => [t, v.weights[i]]);
    p.sort((a, b) => b[1] - a[1] || a[0] - b[0]);
    const kept = p.slice(0, k).sort((a, b) => a[0] - b[0]);
    return { term_ids: kept.map(x => x[0]), weights: kept.map(x => x[1]) };
  };
  const postings = new Map();
  for (const v of dv) { const t = trunc(v, 512); for (let i = 0; i < t.term_ids.length; i++) { let l = postings.get(t.term_ids[i]); if (!l) postings.set(t.term_ids[i], (l = [])); l.push([v.id, t.weights[i]]); } }
  return { postings, qv, trunc };
})();
function sparseSearch(qid, limit) {
  const v = sparse.qv[qid]; if (!v) return [];
  const q = sparse.trunc(v, 64), scores = new Map();
  for (let i = 0; i < q.term_ids.length; i++) {
    const list = sparse.postings.get(q.term_ids[i]); if (!list) continue;
    for (const [id, dw] of list) scores.set(id, (scores.get(id) || 0) + q.weights[i] * dw);
  }
  return [...scores].sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1)).slice(0, limit);
}
// mulberry32: a plain LCG loses precision here because the multiply exceeds 2^53
// in double arithmetic, which collapses the sequence and produces confidence
// intervals that do not even contain their own point estimate.
let rngState = 987654321 >>> 0;
const rnd = () => {
  rngState = (rngState + 0x6d2b79f5) >>> 0;
  let t = rngState;
  t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
};
function ci(diffs) {
  const B = 5000, boots = [];
  for (let b = 0; b < B; b++) { let s = 0; for (let i = 0; i < diffs.length; i++) s += diffs[Math.floor(rnd() * diffs.length)]; boots.push(s / diffs.length); }
  boots.sort((a, b) => a - b);
  return [mean(diffs), boots[Math.floor(B * 0.025)], boots[Math.floor(B * 0.975)]];
}
const comparisons = {
  "가산항 제거 − 현행": q => {
    const st = graphState(base), lex = bmCache[q.id], sem = denseCache[q.id], g = graphRanking(st, lex, sem);
    return ndcgOf(fuse(st, { lex, sem, g }, 0, 0), q.relevance) - ndcgOf(fuse(st, { lex, sem, g }, 0.08, 0.10), q.relevance);
  },
  "캐스케이드 − BM25 단독 (lexical만)": q =>
    ndcgOf(cascadeIds(bmCache[q.id], sparseSearch(q.id, LIMIT), 8), q.relevance) - ndcgOf(bmCache[q.id].map(h => h[0]), q.relevance),
  "sparse 단독 − BM25 단독": q =>
    ndcgOf(sparseSearch(q.id, LIMIT).map(h => h[0]), q.relevance) - ndcgOf(bmCache[q.id].map(h => h[0]), q.relevance),
};
console.log("  비교                                    n     차이       95% CI");
for (const [name, fn] of Object.entries(comparisons)) {
  for (const [label, subset] of [["전체", queries], ["ko", queries.filter(q => q.language === "ko")]]) {
    const [d, lo, hi] = ci(subset.map(fn));
    const mark = lo > 0 ? "유의(+)" : hi < 0 ? "유의(−)" : "0 포함";
    console.log(`  ${(name + " [" + label + "]").padEnd(38)} ${String(subset.length).padStart(4)}  ${(d >= 0 ? "+" : "") + d.toFixed(4)}  [${lo.toFixed(4)}, ${hi.toFixed(4)}]  ${mark}`);
  }
}

// ── 3. search latency as the index grows ──────────────────────────────────────
console.log("\n=== 3) 인덱스 규모별 sparse 검색 지연 ===");
const dv = JSON.parse(fs.readFileSync(`${SPARSE_DIR}/document-vectors.json`, "utf8"));
const sample = queries.slice(0, 500);
console.log("  문서수    posting     인덱스MiB   p50 ms   p95 ms");
for (const factor of [1, 4, 16, 85]) {
  const postings = new Map();
  let count = 0;
  for (let copy = 0; copy < factor; copy++) {
    for (const v of dv) {
      const t = sparse.trunc(v, 512);
      for (let i = 0; i < t.term_ids.length; i++) {
        let l = postings.get(t.term_ids[i]); if (!l) postings.set(t.term_ids[i], (l = []));
        l.push([`${copy}/${v.id}`, t.weights[i]]); count++;
      }
    }
  }
  const latencies = [];
  for (const q of sample) {
    const v = sparse.qv[q.id]; if (!v) continue;
    const qt = sparse.trunc(v, 64);
    const started = process.hrtime.bigint();
    const scores = new Map();
    for (let i = 0; i < qt.term_ids.length; i++) {
      const list = postings.get(qt.term_ids[i]); if (!list) continue;
      for (const [id, dw] of list) scores.set(id, (scores.get(id) || 0) + qt.weights[i] * dw);
    }
    [...scores].sort((a, b) => b[1] - a[1]).slice(0, LIMIT);
    latencies.push(Number(process.hrtime.bigint() - started) / 1e6);
  }
  latencies.sort((a, b) => a - b);
  console.log(`  ${String(dv.length * factor).padStart(6)}  ${String(count).padStart(9)}   ${(count * 8 / 1048576).toFixed(1).padStart(9)}   ${latencies[Math.floor(latencies.length * 0.5)].toFixed(3).padStart(6)}   ${latencies[Math.floor(latencies.length * 0.95)].toFixed(3).padStart(6)}`);
}
console.log("  ※ JS 단일스레드 해시맵 기준. Rust 구현은 이보다 빠르다 (기존 실측 0.075ms/354문서 규모).");
