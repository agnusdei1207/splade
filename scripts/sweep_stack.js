// Simulate pentesting's full retrieval stack and swap only the lexical engine.
//
// Replicates hybrid_search.rs: lexical + semantic + graph, fused with RRF at k=60,
// then the additive base-score terms. phase_bonus and memory_strength are query and
// clock dependent, so they are held neutral - they scale every candidate the same way
// for a given query and cannot change the ordering being compared here.
//
// Usage: node scripts/sweep_stack.js artifacts/families artifacts/dense sparseDir=label ...

const fs = require("fs");
const path = require("path");

const FAM = process.argv[2];
const DENSE = process.argv[3];
const SPARSE_ARGS = process.argv.slice(4);

const CONFIG = {
  rrfK: 60,
  lexicalLimit: 24,
  semanticLimit: 24,
  graphLimit: 24,
  graphSeedLimit: 5,
  maxGraphHops: 2,
  backlinkBoost: 0.08,
  pagerankBoost: 0.1,
  limit: 24,
};

const documents = JSON.parse(fs.readFileSync(`${FAM}/documents.json`, "utf8"));
const queries = JSON.parse(fs.readFileSync(`${FAM}/queries.json`, "utf8"));
const denseRankings = JSON.parse(fs.readFileSync(`${DENSE}/rankings.json`, "utf8"));

// ── BM25 (hybrid_search.rs) ───────────────────────────────────────────────────
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
      const idf = Math.log((bm.n - dfv + 0.5) / (dfv + 0.5) + 1.0);
      score += (idf * 2.4) / Math.max(1 + 1.4 * (0.25 + 0.75 * (dl / Math.max(bm.avg, 1))), 1e-5);
    }
    if (bm.lower.get(d.id).includes(ql)) score += 0.15;
    if (score > 0) hits.push([d.id, score]);
  }
  hits.sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1));
  return hits.slice(0, limit);
}

// ── graph (graph_parser.rs) ───────────────────────────────────────────────────
const WIKI = /\[\[([^\]|#]+)(?:#[^\]|]+)?(?:\|[^\]]+)?\]\]/g;
const MDLINK = /\[[^\]]+\]\(([^)\s]+\.md(?:#[^)]+)?)\)/g;
const byStem = new Map();
for (const d of documents) {
  const stem = path.basename(d.id, ".md").toLowerCase();
  if (!byStem.has(stem)) byStem.set(stem, d.id);
}
const ids = new Set(documents.map(d => d.id));
const adjacency = new Map(), backlinks = new Map();
for (const d of documents) {
  const targets = new Set();
  for (const m of d.text.matchAll(WIKI)) {
    const t = byStem.get(m[1].trim().toLowerCase().replace(/\.md$/, ""));
    if (t && t !== d.id) targets.add(t);
  }
  for (const m of d.text.matchAll(MDLINK)) {
    const raw = m[1].split("#")[0];
    const resolved = path.posix.normalize(path.posix.join(path.posix.dirname(d.id), raw));
    const t = ids.has(resolved) ? resolved : byStem.get(path.basename(raw, ".md").toLowerCase());
    if (t && t !== d.id) targets.add(t);
  }
  adjacency.set(d.id, targets);
  for (const t of targets) {
    if (!backlinks.has(t)) backlinks.set(t, new Set());
    backlinks.get(t).add(d.id);
  }
}
const backlinkCount = id => (backlinks.get(id) || new Set()).size;
const pagerank = (() => {
  const N = documents.length, damping = 0.85;
  let rank = new Map(documents.map(d => [d.id, 1 / N]));
  for (let iteration = 0; iteration < 20; iteration++) {
    const next = new Map(documents.map(d => [d.id, (1 - damping) / N]));
    let sink = 0;
    for (const d of documents) {
      const out = adjacency.get(d.id);
      if (!out.size) { sink += rank.get(d.id); continue; }
      const share = (damping * rank.get(d.id)) / out.size;
      for (const t of out) next.set(t, next.get(t) + share);
    }
    for (const d of documents) next.set(d.id, next.get(d.id) + (damping * sink) / N);
    rank = next;
  }
  const max = Math.max(...rank.values());
  return new Map([...rank].map(([k, v]) => [k, max ? v / max : 0]));
})();
function graphRanking(lexical, semantic) {
  const seedWeights = new Map();
  const seeds = [...lexical.slice(0, CONFIG.graphSeedLimit), ...semantic.slice(0, CONFIG.graphSeedLimit)];
  seeds.forEach(([id, score], rank) => seedWeights.set(id, (seedWeights.get(id) || 0) + score + 1 / (rank + 1)));
  if (!seedWeights.size) return [];
  const depths = new Map(), queue = [];
  for (const seed of seedWeights.keys()) { depths.set(seed, 0); queue.push([seed, 0]); }
  while (queue.length) {
    const [current, depth] = queue.shift();
    if (depth >= CONFIG.maxGraphHops) continue;
    const neighbours = [...(adjacency.get(current) || []), ...(backlinks.get(current) || [])];
    for (const n of neighbours) if (!depths.has(n)) { depths.set(n, depth + 1); queue.push([n, depth + 1]); }
  }
  const ranked = [...depths].map(([id, hops]) => [
    id,
    (seedWeights.get(id) || 0) + 1 / (hops + 1) + Math.log(backlinkCount(id) + 1) * CONFIG.backlinkBoost,
  ]);
  ranked.sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1));
  return ranked.slice(0, CONFIG.graphLimit);
}

// ── fusion ────────────────────────────────────────────────────────────────────
function fuse(rankings, weights) {
  const rrf = new Map();
  for (const [engine, ranking] of Object.entries(rankings)) {
    const w = weights[engine] ?? 1;
    ranking.forEach(([id], index) => rrf.set(id, (rrf.get(id) || 0) + w / (CONFIG.rrfK + index + 1)));
  }
  const scored = [...rrf].map(([id, score]) => [
    id,
    score + Math.log(backlinkCount(id) + 1) * CONFIG.backlinkBoost + pagerank.get(id) * CONFIG.pagerankBoost,
  ]);
  scored.sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1));
  return scored.slice(0, CONFIG.limit).map(e => e[0]);
}

// ── sparse engines ────────────────────────────────────────────────────────────
function truncate(v, k) {
  if (v.term_ids.length <= k) return v;
  const p = v.term_ids.map((t, i) => [t, v.weights[i]]);
  p.sort((a, b) => b[1] - a[1] || a[0] - b[0]);
  const kept = p.slice(0, k).sort((a, b) => a[0] - b[0]);
  return { term_ids: kept.map(x => x[0]), weights: kept.map(x => x[1]) };
}
const sparseEngines = {};
for (const arg of SPARSE_ARGS) {
  const [dir, label] = arg.split("=");
  const name = label || path.basename(dir);
  const summary = JSON.parse(fs.readFileSync(`${dir}/summary.json`, "utf8"));
  const dv = JSON.parse(fs.readFileSync(`${dir}/document-vectors.json`, "utf8"));
  const qv = Object.fromEntries(JSON.parse(fs.readFileSync(`${dir}/query-vectors.json`, "utf8")).map(v => [v.id, v]));
  const docK = Number(process.env.DOC_K || 512), qryK = Number(process.env.QUERY_K || 64);
  const postings = new Map();
  for (const v of dv) {
    const t = truncate(v, docK);
    for (let i = 0; i < t.term_ids.length; i++) {
      let list = postings.get(t.term_ids[i]);
      if (!list) postings.set(t.term_ids[i], (list = []));
      list.push([v.id, t.weights[i]]);
    }
  }
  sparseEngines[name] = {
    summary,
    search(queryId, limit) {
      const v = qv[queryId];
      if (!v) return [];
      const q = truncate(v, qryK), scores = new Map();
      for (let i = 0; i < q.term_ids.length; i++) {
        const list = postings.get(q.term_ids[i]);
        if (!list) continue;
        for (const [id, dw] of list) scores.set(id, (scores.get(id) || 0) + q.weights[i] * dw);
      }
      return [...scores].sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1)).slice(0, limit);
    },
  };
}

// ── metrics ───────────────────────────────────────────────────────────────────
const recall = (r, rel, k) => { const S = new Set(rel); let n = 0; for (const d of r.slice(0, k)) if (S.has(d)) n++; return S.size ? n / S.size : 0; };
const mrr = (r, rel, k) => { const S = new Set(rel); for (let i = 0; i < Math.min(k, r.length); i++) if (S.has(r[i])) return 1 / (i + 1); return 0; };
const ndcg = (r, rel, k) => {
  const S = new Set(rel), dcg = g => g.reduce((s, v, i) => s + v / Math.log2(i + 2), 0);
  const ideal = dcg(new Array(Math.min(k, S.size)).fill(1));
  return ideal ? dcg(r.slice(0, k).map(d => (S.has(d) ? 1 : 0))) / ideal : 0;
};
function evaluate(qs, rank) {
  if (!qs.length) return null;
  const m = f => qs.reduce((s, q) => s + f(rank[q.id] || [], q.relevance), 0) / qs.length;
  return { "R@1": m((r, l) => recall(r, l, 1)), "R@10": m((r, l) => recall(r, l, 10)), "MRR@10": m((r, l) => mrr(r, l, 10)), "nDCG@10": m((r, l) => ndcg(r, l, 10)) };
}
const f4 = m => (m ? `${m["R@1"].toFixed(4)}  ${m["R@10"].toFixed(4)}  ${m["MRR@10"].toFixed(4)}  ${m["nDCG@10"].toFixed(4)}` : "-");

// ── run configurations ────────────────────────────────────────────────────────
const bmCache = {}, denseCache = {};
for (const q of queries) {
  bmCache[q.id] = bm25(q.query, CONFIG.lexicalLimit);
  denseCache[q.id] = (denseRankings[q.id] || []).slice(0, CONFIG.semanticLimit);
}

const configurations = [];
configurations.push({ name: "현행 (BM25+Dense+Graph)", lexical: id => bmCache[id], weights: { lexical: 1, semantic: 1, graph: 1 } });
for (const name of Object.keys(sparseEngines)) {
  const engine = sparseEngines[name];
  configurations.push({ name: `${name} 대체 (SPLADE+Dense+Graph)`, lexical: id => engine.search(id, CONFIG.lexicalLimit), weights: { lexical: 1, semantic: 1, graph: 1 } });
  configurations.push({
    name: `${name} 추가 (BM25+SPLADE+Dense+Graph)`,
    lexical: id => bmCache[id], extra: id => engine.search(id, CONFIG.lexicalLimit),
    weights: { lexical: 1, semantic: 1, graph: 1, sparse: 1 },
  });
}

const byLang = { ko: queries.filter(q => q.language === "ko"), en: queries.filter(q => q.language === "en") };
const FAMILIES = ["heading", "title", "sentence", "identifier", "question"];

console.log(`corpus ${documents.length}문서  질의 ${queries.length}개  링크 ${[...adjacency.values()].reduce((s, x) => s + x.size, 0)}개`);
console.log(`dense: 수락 ${Object.values(denseRankings).filter(r => r.length).length}  거절/빈결과 ${Object.values(denseRankings).filter(r => !r.length).length}`);

const results = {};
for (const cfg of configurations) {
  const rank = {};
  for (const q of queries) {
    const lexical = cfg.lexical(q.id);
    const semantic = denseCache[q.id];
    const graph = graphRanking(lexical, semantic);
    const engines = { lexical, semantic, graph };
    if (cfg.extra) engines.sparse = cfg.extra(q.id);
    rank[q.id] = fuse(engines, cfg.weights);
  }
  results[cfg.name] = rank;
}

console.log("\n=== 전체 스택 비교 ===");
console.log("  구성                                      R@1     R@10    MRR@10  nDCG@10");
for (const [name, rank] of Object.entries(results)) console.log(`  ${name.padEnd(40)} ${f4(evaluate(queries, rank))}`);

console.log("\n=== 언어별 nDCG@10 ===");
const names = Object.keys(results);
console.log("  언어    n     " + names.map(n => n.slice(0, 22).padEnd(23)).join(""));
for (const [lang, qs] of Object.entries(byLang)) {
  console.log(`  ${lang.padEnd(6)} ${String(qs.length).padStart(4)}  ` + names.map(n => evaluate(qs, results[n])["nDCG@10"].toFixed(4).padEnd(23)).join(""));
}

console.log("\n=== 패밀리별 nDCG@10 ===");
console.log("  패밀리       n     " + names.map(n => n.slice(0, 22).padEnd(23)).join(""));
for (const f of FAMILIES) {
  const qs = queries.filter(q => q.family === f);
  console.log(`  ${f.padEnd(11)} ${String(qs.length).padStart(4)}  ` + names.map(n => evaluate(qs, results[n])["nDCG@10"].toFixed(4).padEnd(23)).join(""));
}

console.log("\n=== RRF 가중치 재설계 (엔진별 가중, 최적 sparse 구성 기준) ===");
const sparseName = Object.keys(sparseEngines)[0];
if (sparseName) {
  const engine = sparseEngines[sparseName];
  console.log("  lex  sem  grf  spr   전체nDCG   ko      en");
  const grid = [
    [1, 1, 1, 0], [1, 1, 0.5, 0], [1, 1, 0.25, 0], [1, 0.5, 0.5, 0],
    [1, 1, 1, 1], [1, 1, 0.5, 1], [1, 1, 0.5, 1.5], [0.5, 1, 0.5, 1.5],
    [1, 1, 0.25, 1], [0, 1, 0.5, 1], [1, 0, 0.5, 1],
  ];
  for (const [wl, ws, wg, wsp] of grid) {
    const rank = {};
    for (const q of queries) {
      const lexical = bmCache[q.id], semantic = denseCache[q.id];
      const graph = graphRanking(lexical, semantic);
      const engines = { lexical, semantic, graph };
      if (wsp > 0) engines.sparse = engine.search(q.id, CONFIG.lexicalLimit);
      rank[q.id] = fuse(engines, { lexical: wl, semantic: ws, graph: wg, sparse: wsp });
    }
    console.log(`  ${wl.toFixed(2)} ${ws.toFixed(2)} ${wg.toFixed(2)} ${wsp.toFixed(2)}   ${evaluate(queries, rank)["nDCG@10"].toFixed(4)}   ${evaluate(byLang.ko, rank)["nDCG@10"].toFixed(4)}  ${evaluate(byLang.en, rank)["nDCG@10"].toFixed(4)}`);
  }
}

// ── diagnostic: how much do the additive graph priors move the ranking? ───────
console.log("\n=== 가산항(backlink/pagerank) 영향 분리 ===");
function fuseWith(rankings, weights, useBoosts) {
  const rrf = new Map();
  for (const [engine, ranking] of Object.entries(rankings)) {
    const w = weights[engine] ?? 1;
    ranking.forEach(([id], index) => rrf.set(id, (rrf.get(id) || 0) + w / (CONFIG.rrfK + index + 1)));
  }
  const scored = [...rrf].map(([id, score]) => [
    id,
    useBoosts
      ? score + Math.log(backlinkCount(id) + 1) * CONFIG.backlinkBoost + pagerank.get(id) * CONFIG.pagerankBoost
      : score,
  ]);
  scored.sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1));
  return scored.slice(0, CONFIG.limit).map(e => e[0]);
}
const variants = {
  "BM25 단독 (참고)": q => bmCache[q.id].map(h => h[0]).slice(0, CONFIG.limit),
  "lexical만 RRF, 가산항 ON": q => fuseWith({ lexical: bmCache[q.id] }, { lexical: 1 }, true),
  "lexical만 RRF, 가산항 OFF": q => fuseWith({ lexical: bmCache[q.id] }, { lexical: 1 }, false),
  "3엔진 RRF, 가산항 ON (현행)": q => { const g = graphRanking(bmCache[q.id], denseCache[q.id]); return fuseWith({ lexical: bmCache[q.id], semantic: denseCache[q.id], graph: g }, { lexical: 1, semantic: 1, graph: 1 }, true); },
  "3엔진 RRF, 가산항 OFF": q => { const g = graphRanking(bmCache[q.id], denseCache[q.id]); return fuseWith({ lexical: bmCache[q.id], semantic: denseCache[q.id], graph: g }, { lexical: 1, semantic: 1, graph: 1 }, false); },
  "lexical+dense, graph 제외, 가산항 OFF": q => fuseWith({ lexical: bmCache[q.id], semantic: denseCache[q.id] }, { lexical: 1, semantic: 1 }, false),
};
console.log("  구성                                      R@1     R@10    MRR@10  nDCG@10");
for (const [name, fn] of Object.entries(variants)) {
  const rank = {};
  for (const q of queries) rank[q.id] = fn(q);
  console.log(`  ${name.padEnd(40)} ${f4(evaluate(queries, rank))}`);
}
