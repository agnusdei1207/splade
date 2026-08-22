// Try to break the two recommendations before proposing them.
//
// E1  The graph priors look purely harmful, but the measurement only covers queries
//     whose answer is one specific document. A backlink prior encodes "this note is
//     important", which could be right for broad queries. If some query subset is
//     better with the priors on, they need rescaling rather than removal.
//
// E2  The cascade's headline win is on Korean queries - but production normalises
//     every query to English before retrieval, so that gain may be unreachable. The
//     realisable mechanism is different: a multilingual encoder indexes Korean
//     document text into a shared space, so an English query can reach a Korean
//     document. That is what has to be measured.
//
// E3  If the priors carry real signal, additive-on-RRF is still the wrong shape.
//     Compare rank tie-break and multiplicative forms against removal.

const fs = require("fs");
const path = require("path");

const FAM = process.argv[2] || "artifacts/families";
const DENSE = process.argv[3] || "artifacts/dense";
const MULTI = process.argv[4] || "artifacts/models/f-multilingual-v1";
const MINI = process.argv[5] || "artifacts/models/f-doc-v2-mini";

const documents = JSON.parse(fs.readFileSync(`${FAM}/documents.json`, "utf8"));
const queries = JSON.parse(fs.readFileSync(`${FAM}/queries.json`, "utf8"));
const denseRankings = JSON.parse(fs.readFileSync(`${DENSE}/rankings.json`, "utf8"));
const docById = new Map(documents.map(d => [d.id, d]));

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

// ── graph ─────────────────────────────────────────────────────────────────────
const WIKI = /\[\[([^\]|#]+)(?:#[^\]|]+)?(?:\|[^\]]+)?\]\]/g;
const MDLINK = /\[[^\]]+\]\(([^)\s]+\.md(?:#[^)]+)?)\)/g;
const byStem = new Map();
for (const d of documents) { const s = path.basename(d.id, ".md").toLowerCase(); if (!byStem.has(s)) byStem.set(s, d.id); }
const ids = documents.map(d => d.id), idSet = new Set(ids);
const adjacency = new Map(ids.map(id => [id, new Set()])), backlinks = new Map(ids.map(id => [id, new Set()]));
for (const d of documents) {
  for (const m of d.text.matchAll(WIKI)) { const t = byStem.get(m[1].trim().toLowerCase().replace(/\.md$/, "")); if (t && t !== d.id) adjacency.get(d.id).add(t); }
  for (const m of d.text.matchAll(MDLINK)) {
    const raw = m[1].split("#")[0];
    const resolved = path.posix.normalize(path.posix.join(path.posix.dirname(d.id), raw));
    const t = idSet.has(resolved) ? resolved : byStem.get(path.basename(raw, ".md").toLowerCase());
    if (t && t !== d.id) adjacency.get(d.id).add(t);
  }
}
for (const [s, ts] of adjacency) for (const t of ts) backlinks.get(t).add(s);
const backlinkCount = id => backlinks.get(id).size;
const pagerank = (() => {
  const N = ids.length, damping = 0.85;
  let rank = new Map(ids.map(id => [id, 1 / N]));
  for (let it = 0; it < 20; it++) {
    const nx = new Map(ids.map(id => [id, (1 - damping) / N]));
    let sink = 0;
    for (const id of ids) {
      const out = adjacency.get(id);
      if (!out.size) { sink += rank.get(id); continue; }
      const share = (damping * rank.get(id)) / out.size;
      for (const t of out) nx.set(t, nx.get(t) + share);
    }
    for (const id of ids) nx.set(id, nx.get(id) + (damping * sink) / N);
    rank = nx;
  }
  const max = Math.max(...rank.values());
  return new Map([...rank].map(([k, v]) => [k, max ? v / max : 0]));
})();
function graphRanking(lexical, semantic) {
  const sw = new Map();
  [...lexical.slice(0, SEED_LIMIT), ...semantic.slice(0, SEED_LIMIT)].forEach(([id, sc], r) => sw.set(id, (sw.get(id) || 0) + sc + 1 / (r + 1)));
  if (!sw.size) return [];
  const depths = new Map(), queue = [];
  for (const s of sw.keys()) { depths.set(s, 0); queue.push([s, 0]); }
  while (queue.length) {
    const [cur, dep] = queue.shift();
    if (dep >= MAX_HOPS) continue;
    for (const n of [...adjacency.get(cur), ...backlinks.get(cur)]) if (!depths.has(n)) { depths.set(n, dep + 1); queue.push([n, dep + 1]); }
  }
  const ranked = [...depths].map(([id, h]) => [id, (sw.get(id) || 0) + 1 / (h + 1) + Math.log(backlinkCount(id) + 1) * 0.08]);
  ranked.sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1));
  return ranked.slice(0, LIMIT);
}

// ── prior shapes ──────────────────────────────────────────────────────────────
const PRIORS = {
  "현행 (가산 0.08/0.10)": (rrf) => [...rrf].map(([id, s]) => [id, s + Math.log(backlinkCount(id) + 1) * 0.08 + pagerank.get(id) * 0.10]),
  "제거": (rrf) => [...rrf],
  "가산 1/10": (rrf) => [...rrf].map(([id, s]) => [id, s + Math.log(backlinkCount(id) + 1) * 0.008 + pagerank.get(id) * 0.010]),
  "RRF 정규화 후 가산 0.05": (rrf) => {
    const max = Math.max(...rrf.values(), 1e-9);
    return [...rrf].map(([id, s]) => [id, s / max + (Math.log(backlinkCount(id) + 1) / Math.log(50) * 0.5 + pagerank.get(id) * 0.5) * 0.05]);
  },
  "동점 tie-break로만": (rrf) => [...rrf].map(([id, s]) => [id, Math.round(s * 1e6) / 1e6 + (Math.log(backlinkCount(id) + 1) * 0.08 + pagerank.get(id) * 0.1) * 1e-9]),
  "곱셈 (1 + 0.1·prior)": (rrf) => [...rrf].map(([id, s]) => [id, s * (1 + 0.1 * (Math.log(backlinkCount(id) + 1) / Math.log(50) + pagerank.get(id)))]),
};
function fuse(rankings, shape) {
  const rrf = new Map();
  for (const r of Object.values(rankings)) r.forEach(([id], i) => rrf.set(id, (rrf.get(id) || 0) + 1 / (RRF_K + i + 1)));
  const scored = PRIORS[shape](rrf);
  scored.sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1));
  return scored.slice(0, LIMIT).map(e => e[0]);
}

// ── sparse ────────────────────────────────────────────────────────────────────
function loadSparse(dir) {
  const dv = JSON.parse(fs.readFileSync(`${dir}/document-vectors.json`, "utf8"));
  const qv = Object.fromEntries(JSON.parse(fs.readFileSync(`${dir}/query-vectors.json`, "utf8")).map(v => [v.id, v]));
  const trunc = (v, k) => {
    if (v.term_ids.length <= k) return v;
    const p = v.term_ids.map((t, i) => [t, v.weights[i]]);
    p.sort((a, b) => b[1] - a[1] || a[0] - b[0]);
    const kept = p.slice(0, k).sort((a, b) => a[0] - b[0]);
    return { term_ids: kept.map(x => x[0]), weights: kept.map(x => x[1]) };
  };
  const postings = new Map();
  for (const v of dv) { const t = trunc(v, 512); for (let i = 0; i < t.term_ids.length; i++) { let l = postings.get(t.term_ids[i]); if (!l) postings.set(t.term_ids[i], (l = [])); l.push([v.id, t.weights[i]]); } }
  return (qid, limit) => {
    const v = qv[qid]; if (!v) return [];
    const q = trunc(v, 64), sc = new Map();
    for (let i = 0; i < q.term_ids.length; i++) {
      const l = postings.get(q.term_ids[i]); if (!l) continue;
      for (const [id, dw] of l) sc.set(id, (sc.get(id) || 0) + q.weights[i] * dw);
    }
    return [...sc].sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1)).slice(0, limit);
  };
}
const multiSearch = loadSparse(MULTI), miniSearch = loadSparse(MINI);

// ── metrics ───────────────────────────────────────────────────────────────────
const ndcgOf = (r, rel, k = 10) => {
  const S = new Set(rel), dcg = g => g.reduce((s, v, i) => s + v / Math.log2(i + 2), 0);
  const ideal = dcg(new Array(Math.min(k, S.size)).fill(1));
  return ideal ? dcg(r.slice(0, k).map(d => (S.has(d) ? 1 : 0))) / ideal : 0;
};
const mean = a => (a.length ? a.reduce((s, x) => s + x, 0) / a.length : 0);

const bmCache = {}, denseCache = {};
for (const q of queries) { bmCache[q.id] = bm25(q.query, LIMIT); denseCache[q.id] = (denseRankings[q.id] || []).slice(0, LIMIT); }

// ── E1: does the prior ever help? ─────────────────────────────────────────────
console.log("=== E1) 가산항이 도움이 되는 구간이 있는가 ===");
const perQuery = queries.map(q => {
  const lex = bmCache[q.id], sem = denseCache[q.id], g = graphRanking(lex, sem);
  const on = ndcgOf(fuse({ lex, sem, g }, "현행 (가산 0.08/0.10)"), q.relevance);
  const off = ndcgOf(fuse({ lex, sem, g }, "제거"), q.relevance);
  const target = q.relevance[0];
  return { q, on, off, delta: on - off, targetBacklinks: target ? backlinkCount(target) : 0, targetPagerank: target ? pagerank.get(target) || 0 : 0 };
});
console.log(`  전체 ${perQuery.length}개 중  가산항이 도움 ${perQuery.filter(r => r.delta > 1e-9).length}  손해 ${perQuery.filter(r => r.delta < -1e-9).length}  동일 ${perQuery.filter(r => Math.abs(r.delta) <= 1e-9).length}`);
console.log("\n  정답 문서의 backlink 수별 (가산항이 정답을 밀어올려야 하는 구간)");
console.log("  backlink   n      가산항ON   가산항OFF   차이");
for (const [label, test] of [["0", r => r.targetBacklinks === 0], ["1", r => r.targetBacklinks === 1], ["2-4", r => r.targetBacklinks >= 2 && r.targetBacklinks <= 4], ["5+", r => r.targetBacklinks >= 5]]) {
  const sub = perQuery.filter(test);
  if (!sub.length) continue;
  const on = mean(sub.map(r => r.on)), off = mean(sub.map(r => r.off));
  console.log(`  ${label.padEnd(9)} ${String(sub.length).padStart(5)}   ${on.toFixed(4)}    ${off.toFixed(4)}    ${(on - off >= 0 ? "+" : "") + (on - off).toFixed(4)}`);
}
console.log("\n  패밀리별");
console.log("  패밀리       n      가산항ON   가산항OFF   차이");
for (const f of ["heading", "title", "sentence", "identifier", "question"]) {
  const sub = perQuery.filter(r => r.q.family === f);
  const on = mean(sub.map(r => r.on)), off = mean(sub.map(r => r.off));
  console.log(`  ${f.padEnd(11)} ${String(sub.length).padStart(5)}   ${on.toFixed(4)}    ${off.toFixed(4)}    ${(on - off >= 0 ? "+" : "") + (on - off).toFixed(4)}`);
}

// ── E2: English query reaching a Korean document ──────────────────────────────
console.log("\n=== E2) 영어 질의로 한국어 문서를 찾는가 (운영 경로) ===");
const koDoc = new Set(documents.filter(d => d.language === "ko").map(d => d.id));
const enQueries = queries.filter(q => q.language === "en");
const enToKo = enQueries.filter(q => q.relevance.some(id => koDoc.has(id)));
const enToEn = enQueries.filter(q => !q.relevance.some(id => koDoc.has(id)));
console.log(`  영어 질의 ${enQueries.length}개 중 정답이 한국어 문서인 것 ${enToKo.length}개 / 영어 문서인 것 ${enToEn.length}개`);
const cascadeIds = (primary, backfill, head) => {
  const out = [], seen = new Set();
  const push = ([id]) => { if (!seen.has(id) && out.length < LIMIT) { seen.add(id); out.push(id); } };
  primary.slice(0, head).forEach(push); backfill.forEach(push); primary.slice(head).forEach(push);
  return out;
};
console.log("\n  구간              n     BM25단독  mini단독  multi단독  BM25₈+mini  BM25₈+multi");
for (const [label, subset] of [["영어질의→한글문서", enToKo], ["영어질의→영문문서", enToEn], ["한글질의(참고)", queries.filter(q => q.language === "ko")]]) {
  if (!subset.length) continue;
  const row = subset.map(q => ({
    bm: ndcgOf(bmCache[q.id].map(h => h[0]), q.relevance),
    mini: ndcgOf(miniSearch(q.id, LIMIT).map(h => h[0]), q.relevance),
    multi: ndcgOf(multiSearch(q.id, LIMIT).map(h => h[0]), q.relevance),
    cMini: ndcgOf(cascadeIds(bmCache[q.id], miniSearch(q.id, LIMIT), 8), q.relevance),
    cMulti: ndcgOf(cascadeIds(bmCache[q.id], multiSearch(q.id, LIMIT), 8), q.relevance),
  }));
  console.log(`  ${label.padEnd(17)} ${String(subset.length).padStart(4)}  ${mean(row.map(r => r.bm)).toFixed(4)}    ${mean(row.map(r => r.mini)).toFixed(4)}    ${mean(row.map(r => r.multi)).toFixed(4)}     ${mean(row.map(r => r.cMini)).toFixed(4)}      ${mean(row.map(r => r.cMulti)).toFixed(4)}`);
}

// ── E3: prior shapes ──────────────────────────────────────────────────────────
console.log("\n=== E3) 가산항 형태별 ===");
console.log("  형태                          전체      ko        en        identifier");
const idFam = queries.filter(q => q.family === "identifier");
const koQ = queries.filter(q => q.language === "ko"), enQ = queries.filter(q => q.language === "en");
for (const shape of Object.keys(PRIORS)) {
  const score = subset => mean(subset.map(q => {
    const lex = bmCache[q.id], sem = denseCache[q.id], g = graphRanking(lex, sem);
    return ndcgOf(fuse({ lex, sem, g }, shape), q.relevance);
  }));
  console.log(`  ${shape.padEnd(28)} ${score(queries).toFixed(4)}    ${score(koQ).toFixed(4)}    ${score(enQ).toFixed(4)}    ${score(idFam).toFixed(4)}`);
}
