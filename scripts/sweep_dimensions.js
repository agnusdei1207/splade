// Driver: how many sparse dimensions per document and per query does this corpus need,
// and which retrieval arrangement wins once BM25 is the thing being replaced.

const fs = require("fs");
const S = require("./sweep_scale.js");

const DUMP = process.argv[2] || "artifacts/scale";
const read = name => JSON.parse(fs.readFileSync(`${DUMP}/${name}`, "utf8"));

const queries = read("queries.json");
const docVectors = read("document-vectors.json");
const queryVectors = read("query-vectors.json");
const summary = read("summary.json");
const qvecById = Object.fromEntries(queryVectors.map(v => [v.id, v]));

console.log("=== corpus ===");
console.log(`  문서 ${summary.documents}개 / ${(summary.corpus_bytes / 1048576).toFixed(2)} MiB`);
console.log(`  질의 ${summary.queries}개 (정답 문서 2개 이상: ${summary.ambiguous_queries}개)`);
console.log(`  문서 활성차원  min ${summary.document_active_dims.min}  p50 ${summary.document_active_dims.p50}  p90 ${summary.document_active_dims.p90}  p99 ${summary.document_active_dims.p99}  max ${summary.document_active_dims.max}  mean ${summary.document_active_dims.mean.toFixed(1)}`);
console.log(`  질의 활성차원  min ${summary.query_active_dims.min}  p50 ${summary.query_active_dims.p50}  p90 ${summary.query_active_dims.p90}  max ${summary.query_active_dims.max}  mean ${summary.query_active_dims.mean.toFixed(1)}`);

const corpus = S.buildCorpus();
const byId = new Map(corpus.map(d => [d.id, d]));
const missing = docVectors.filter(v => !byId.has(v.id)).length;
if (missing) console.log(`  ⚠ 벡터에는 있으나 corpus 재구성에 없는 문서 ${missing}개`);

console.log("\n=== BM25 기준선 구축 ===");
let t0 = Date.now();
const bm = S.buildBm25(corpus);
console.log(`  색인 ${Date.now() - t0}ms   어휘 ${bm.df.size}   평균 문서길이 ${bm.avg.toFixed(1)} (tf는 모두 1)`);

t0 = Date.now();
const bmRank = {};
for (const q of queries) bmRank[q.id] = S.idsOf(S.bm25Search(bm, q.query, 24));
const bmMs = Date.now() - t0;
const bmMetrics = S.evaluate(queries, bmRank);
console.log(`  검색 ${bmMs}ms / ${queries.length}질의 = ${(bmMs / queries.length).toFixed(2)}ms per query`);

const fmt = m => ["R@1", "R@5", "R@10", "MRR@10", "nDCG@10"].map(k => m[k].toFixed(4)).join("  ");
console.log("\n=== 1) 문서 top-k 스윕 (질의 무제한) ===");
console.log("  doc_k   postings   인덱스MiB   R@1     R@5     R@10    MRR@10  nDCG@10");
const DOC_KS = [64, 128, 192, 256, 384, 512, 1024, 30522];
const perDocK = {};
for (const k of DOC_KS) {
  const vecs = docVectors.map(v => S.truncate(v, k));
  const ix = S.buildSparseIndex(vecs);
  const rank = {};
  for (const q of queries) rank[q.id] = S.idsOf(S.sparseSearch(ix, qvecById[q.id], 24));
  const m = S.evaluate(queries, rank);
  perDocK[k] = { ix, rank, m };
  const mib = (ix.count * 8) / 1048576;
  console.log(`  ${String(k === 30522 ? "무제한" : k).padEnd(7)} ${String(ix.count).padStart(8)}  ${mib.toFixed(2).padStart(9)}   ${fmt(m)}`);
}

console.log("\n=== 2) 질의 top-k 스윕 (문서 top-k 256 고정) ===");
console.log("  query_k  평균활성  검색ms/질의   R@1     R@5     R@10    MRR@10  nDCG@10");
const ix256 = perDocK[256].ix;
for (const qk of [4, 8, 16, 32, 64, 30522]) {
  const rank = {};
  let active = 0;
  const s = Date.now();
  for (const q of queries) {
    const qv = S.truncate(qvecById[q.id], qk);
    active += qv.term_ids.length;
    rank[q.id] = S.idsOf(S.sparseSearch(ix256, qv, 24));
  }
  const ms = Date.now() - s;
  const m = S.evaluate(queries, rank);
  console.log(`  ${String(qk === 30522 ? "무제한" : qk).padEnd(8)} ${(active / queries.length).toFixed(1).padStart(7)}  ${(ms / queries.length).toFixed(3).padStart(10)}   ${fmt(m)}`);
}

console.log("\n=== 3) BM25 자리 대체안 비교 (문서 256 / 질의 32) ===");
const ixMain = perDocK[256].ix;
const spRank = {};
for (const q of queries) spRank[q.id] = S.idsOf(S.sparseSearch(ixMain, S.truncate(qvecById[q.id], 32), 24));
console.log("  구성                          R@1     R@5     R@10    MRR@10  nDCG@10");
const show = (name, rank) => console.log(`  ${name.padEnd(28)} ${fmt(S.evaluate(queries, rank))}`);
show("BM25 단독 (현행 lexical)", bmRank);
show("SPLADE 단독 (완전 대체)", spRank);
for (const head of [3, 5, 8, 12, 16]) {
  const r = {};
  for (const q of queries) r[q.id] = S.cascade(spRank[q.id], bmRank[q.id], head, 24);
  show(`캐스케이드 SPLADE${head}+BM25`, r);
}
for (const w of [0.3, 0.5, 0.7, 0.9]) {
  const r = {};
  for (const q of queries) r[q.id] = S.rrf(spRank[q.id], bmRank[q.id], 60, w, 24);
  show(`RRF w_splade=${w}`, r);
}
{
  const r = {};
  for (const q of queries) r[q.id] = S.cascade(bmRank[q.id], spRank[q.id], 8, 24);
  show("역방향 캐스케이드 BM25+SPLADE8", r);
}

console.log("\n=== 4) 질의 난이도별 (BM25 어휘 겹침 기준) ===");
const overlap = q => {
  const qt = new Set(S.tokenizeWithIdentifiers(q.query));
  let best = 0;
  for (const id of q.relevance) {
    const dt = bm.terms.get(id);
    if (!dt) continue;
    let n = 0;
    for (const t of qt) if (dt.has(t)) n++;
    best = Math.max(best, qt.size ? n / qt.size : 0);
  }
  return best;
};
const buckets = { "낮음 <0.5": [], "중간 0.5~0.8": [], "높음 ≥0.8": [] };
for (const q of queries) {
  const o = overlap(q);
  (o < 0.5 ? buckets["낮음 <0.5"] : o < 0.8 ? buckets["중간 0.5~0.8"] : buckets["높음 ≥0.8"]).push(q);
}
console.log("  구간             n     엔진      R@10    MRR@10  nDCG@10");
for (const [name, qs] of Object.entries(buckets)) {
  if (!qs.length) continue;
  const casc = {};
  for (const q of qs) casc[q.id] = S.cascade(spRank[q.id], bmRank[q.id], 8, 24);
  for (const [label, rank] of [["BM25", bmRank], ["SPLADE", spRank], ["캐스케이드", casc]]) {
    const m = S.evaluate(qs, rank);
    console.log(`  ${name.padEnd(15)} ${String(qs.length).padStart(4)}  ${label.padEnd(9)} ${m["R@10"].toFixed(4)}  ${m["MRR@10"].toFixed(4)}  ${m["nDCG@10"].toFixed(4)}`);
  }
}
