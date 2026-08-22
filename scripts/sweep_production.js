// The decisive comparison: queries shaped the way production actually produces them.
//
// Two paths matter. The normal one is normalize_retrieval_intent()'s English output.
// The fallback one is ascii_identifier_fallback() on the raw request, which is what
// runs when the LLM call times out at 20s or errors - it keeps only ASCII tokens.

const fs = require("fs");

const FAM = "artifacts/families";
const documents = JSON.parse(fs.readFileSync(`${FAM}/documents.json`, "utf8"));
const rows = fs.readFileSync("benchmarks/production-queries.jsonl", "utf8").trim().split("\n").map(JSON.parse);
const known = new Set(documents.map(d => d.id));
const queries = rows.filter(r => known.has(r.doc));
if (queries.length !== rows.length) console.log(`⚠ corpus에 없는 정답 문서 ${rows.length - queries.length}건 제외`);

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
function bm25(query, limit = 24) {
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
// retrieval_intent.rs :: ascii_identifier_fallback
const asciiFallback = raw => raw.split(/\s+/).filter(t => /^[\x00-\x7F]*$/.test(t) && /[a-zA-Z0-9]/.test(t)).slice(0, 24).join(" ");

// ── sparse ────────────────────────────────────────────────────────────────────
function loadSparse(modelDir, prodDir) {
  const dv = JSON.parse(fs.readFileSync(`${modelDir}/document-vectors.json`, "utf8"));
  const trunc = (v, k) => {
    if (v.term_ids.length <= k) return v;
    const p = v.term_ids.map((t, i) => [t, v.weights[i]]);
    p.sort((a, b) => b[1] - a[1] || a[0] - b[0]);
    const kept = p.slice(0, k).sort((a, b) => a[0] - b[0]);
    return { term_ids: kept.map(x => x[0]), weights: kept.map(x => x[1]) };
  };
  const postings = new Map();
  for (const v of dv) {
    const t = trunc(v, 512);
    for (let i = 0; i < t.term_ids.length; i++) {
      let l = postings.get(t.term_ids[i]); if (!l) postings.set(t.term_ids[i], (l = []));
      l.push([v.id, t.weights[i]]);
    }
  }
  const qv = {};
  for (const form of ["raw", "norm"]) {
    qv[form] = Object.fromEntries(JSON.parse(fs.readFileSync(`${prodDir}/query-vectors-${form}.json`, "utf8")).map(v => [v.id, v]));
  }
  return (id, form, limit = 24) => {
    const v = qv[form][id]; if (!v) return [];
    const q = trunc(v, 64), sc = new Map();
    for (let i = 0; i < q.term_ids.length; i++) {
      const l = postings.get(q.term_ids[i]); if (!l) continue;
      for (const [did, dw] of l) sc.set(did, (sc.get(did) || 0) + q.weights[i] * dw);
    }
    return [...sc].sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1)).slice(0, limit);
  };
}
const miniS = loadSparse("artifacts/models/f-doc-v2-mini", "artifacts/prod/mini");
const multiS = loadSparse("artifacts/models/f-multilingual-v1", "artifacts/prod/multi");

// ── metrics ───────────────────────────────────────────────────────────────────
const hitAt = (r, doc, k) => r.slice(0, k).includes(doc) ? 1 : 0;
const rr = (r, doc) => { const i = r.indexOf(doc); return i >= 0 && i < 10 ? 1 / (i + 1) : 0; };
function evaluate(rankFn) {
  const r1 = [], r5 = [], r10 = [], mrr = [];
  for (const q of queries) {
    const r = rankFn(q);
    r1.push(hitAt(r, q.doc, 1)); r5.push(hitAt(r, q.doc, 5)); r10.push(hitAt(r, q.doc, 10)); mrr.push(rr(r, q.doc));
  }
  const m = a => a.reduce((s, x) => s + x, 0) / a.length;
  return { "R@1": m(r1), "R@5": m(r5), "R@10": m(r10), "MRR@10": m(mrr), _mrr: mrr, _r10: r10 };
}
const f = m => `${m["R@1"].toFixed(4)}  ${m["R@5"].toFixed(4)}  ${m["R@10"].toFixed(4)}  ${m["MRR@10"].toFixed(4)}`;
const cascade = (primary, backfill, head, limit = 24) => {
  const out = [], seen = new Set();
  const push = ([id]) => { if (!seen.has(id) && out.length < limit) { seen.add(id); out.push(id); } };
  primary.slice(0, head).forEach(push); backfill.forEach(push); primary.slice(head).forEach(push);
  return out;
};
const ids = r => r.map(h => h[0]);

console.log(`운영 질의 ${queries.length}개  |  corpus ${documents.length}문서`);
console.log(`  원문 언어: 한국어 ${queries.filter(q => /[가-힣]/.test(q.raw)).length}개`);

console.log("\n=== A) 운영 경로 — LLM이 영어로 정규화한 질의 ===");
console.log("  엔진                          R@1     R@5     R@10    MRR@10");
const A = {
  "BM25 (현행)": q => ids(bm25(q.norm)),
  "mini 단독": q => ids(miniS(q.id, "norm")),
  "multi 단독": q => ids(multiS(q.id, "norm")),
  "캐스케이드 BM25₅+mini": q => cascade(bm25(q.norm), miniS(q.id, "norm"), 5),
  "캐스케이드 BM25₅+multi": q => cascade(bm25(q.norm), multiS(q.id, "norm"), 5),
  "캐스케이드 BM25₈+multi": q => cascade(bm25(q.norm), multiS(q.id, "norm"), 8),
  "캐스케이드 multi₅+BM25": q => cascade(multiS(q.id, "norm"), bm25(q.norm), 5),
};
const resA = {};
for (const [name, fn] of Object.entries(A)) { resA[name] = evaluate(fn); console.log(`  ${name.padEnd(28)} ${f(resA[name])}`); }

console.log("\n=== B) fallback 경로 — LLM 실패 시 (원문 그대로 / ASCII만 남김) ===");
console.log("  엔진                          R@1     R@5     R@10    MRR@10");
const B = {
  "BM25 + ascii fallback (현행)": q => ids(bm25(asciiFallback(q.raw))),
  "BM25 + 원문 그대로": q => ids(bm25(q.raw)),
  "mini + 원문": q => ids(miniS(q.id, "raw")),
  "multi + 원문": q => ids(multiS(q.id, "raw")),
  "캐스케이드 BM25(ascii)₅+multi(원문)": q => cascade(bm25(asciiFallback(q.raw)), multiS(q.id, "raw"), 5),
};
const resB = {};
for (const [name, fn] of Object.entries(B)) { resB[name] = evaluate(fn); console.log(`  ${name.padEnd(28)} ${f(resB[name])}`); }

console.log("\n=== C) 유의성 (부트스트랩 10,000회, MRR@10 차이) ===");
let s = 20260822 >>> 0;
const rnd = () => { s = (s + 0x6d2b79f5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
function ci(a, b) {
  const d = a.map((x, i) => x - b[i]);
  const B = 10000, boots = [];
  for (let k = 0; k < B; k++) { let acc = 0; for (let i = 0; i < d.length; i++) acc += d[Math.floor(rnd() * d.length)]; boots.push(acc / d.length); }
  boots.sort((x, y) => x - y);
  const mean = d.reduce((x, y) => x + y, 0) / d.length;
  return [mean, boots[Math.floor(B * 0.025)], boots[Math.floor(B * 0.975)]];
}
const base = resA["BM25 (현행)"];
for (const name of ["mini 단독", "multi 단독", "캐스케이드 BM25₅+mini", "캐스케이드 BM25₅+multi", "캐스케이드 BM25₈+multi"]) {
  const [d, lo, hi] = ci(resA[name]._mrr, base._mrr);
  console.log(`  ${(name + " − BM25").padEnd(34)} ${(d >= 0 ? "+" : "") + d.toFixed(4)}  [${lo.toFixed(4)}, ${hi.toFixed(4)}]  ${lo > 0 ? "유의(+)" : hi < 0 ? "유의(−)" : "0 포함"}`);
}
const [fd, flo, fhi] = ci(resB["캐스케이드 BM25(ascii)₅+multi(원문)"]._mrr, resB["BM25 + ascii fallback (현행)"]._mrr);
console.log(`  [fallback] 캐스케이드 − 현행        ${(fd >= 0 ? "+" : "") + fd.toFixed(4)}  [${flo.toFixed(4)}, ${fhi.toFixed(4)}]  ${flo > 0 ? "유의(+)" : fhi < 0 ? "유의(−)" : "0 포함"}`);

console.log("\n=== D) 정규화가 실제로 얼마나 기여하나 ===");
const normOnly = evaluate(q => ids(bm25(q.norm)));
const rawOnly = evaluate(q => ids(bm25(q.raw)));
const asciiOnly = evaluate(q => ids(bm25(asciiFallback(q.raw))));
console.log(`  BM25 + 정규화 영어질의    MRR@10 ${normOnly["MRR@10"].toFixed(4)}   R@10 ${normOnly["R@10"].toFixed(4)}`);
console.log(`  BM25 + 원문 그대로       MRR@10 ${rawOnly["MRR@10"].toFixed(4)}   R@10 ${rawOnly["R@10"].toFixed(4)}`);
console.log(`  BM25 + ascii fallback  MRR@10 ${asciiOnly["MRR@10"].toFixed(4)}   R@10 ${asciiOnly["R@10"].toFixed(4)}`);
console.log(`  → LLM 정규화의 기여: MRR +${(normOnly["MRR@10"] - asciiOnly["MRR@10"]).toFixed(4)}`);
