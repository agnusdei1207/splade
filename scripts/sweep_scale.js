// Offline sweep over the encoded corpus: dimensionality, truncation, and fusion.
//
// Rebuilds the BM25 side exactly as pentesting's hybrid_search.rs scores it
// (identifier-expanded tokens, deduped so tf is 1, k1=1.4, b=0.75, +0.15 phrase
// bonus) and replays the SPLADE vectors dumped by exp_scale.py. Everything is
// swept from the uncapped vectors so no setting costs another encode pass.

const fs = require("fs");
const path = require("path");

const ROOT = process.env.PENTESTING_ROOT || "C:/workspace/pentesting";
const DUMP = process.argv[2] || "artifacts/scale";
const SKIP = new Set([".git", "node_modules", "target", "_archive", "prompts", ".cache", ".worktrees"]);
const GENERIC = /^(overview|summary|notes|background|references|see also|todo|목적|요약|개요|참고|배경|결론|목차)$/i;

// ── corpus, mirroring exp_scale.py ────────────────────────────────────────────
function markdownFiles(dir, out = []) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    if (SKIP.has(entry.name)) continue;
    const p = path.join(dir, entry.name);
    if (entry.isDirectory()) markdownFiles(p, out);
    else if (entry.name.endsWith(".md")) out.push(p);
  }
  return out;
}
function parseFrontmatter(raw) {
  if (!raw.startsWith("---")) return [{}, raw];
  const end = raw.indexOf("\n---", 3);
  if (end === -1) return [{}, raw];
  const meta = {};
  for (const line of raw.slice(3, end).split("\n")) {
    if (!line.includes(":") || /^[ \t-]/.test(line)) continue;
    const i = line.indexOf(":");
    const v = line.slice(i + 1).trim().replace(/^\[|\]$/g, "");
    if (v) meta[line.slice(0, i).trim().toLowerCase()] = v.split(",").map(s => s.trim().replace(/^["']|["']$/g, "")).filter(Boolean);
  }
  return [meta, raw.slice(end + 4).replace(/^\n+/, "")];
}
const usableHeading = t => {
  const w = t.split(/\s+/).length;
  return w >= 3 && w <= 14 && !GENERIC.test(t) && !/^\d+[.)]?$/.test(t);
};
function buildCorpus() {
  const docs = [];
  for (const p of markdownFiles(ROOT).sort()) {
    const raw = fs.readFileSync(p, "utf8");
    const [meta, body] = parseFrontmatter(raw);
    const id = path.relative(ROOT, p).split(path.sep).join("/");
    const kept = [];
    for (const line of body.split("\n")) {
      const m = /^(#{2,4})[ \t]+(.+?)[ \t]*$/.exec(line);
      if (m && usableHeading(m[2].replace(/[`*_#\[\]]/g, "").trim())) continue;
      kept.push(line);
    }
    const titleMatch = /^#[ \t]+(.+?)[ \t]*$/m.exec(body);
    const title = titleMatch ? titleMatch[1].trim() : path.basename(p, ".md").replace(/[_-]/g, " ");
    const aliases = (meta.aliases || []).join(" ");
    const tags = (meta.tags || []).join(" ");
    docs.push({ id, text: `${title}\n${aliases}\n${tags}\n${kept.join("\n")}` });
  }
  return docs;
}

// ── BM25, mirroring hybrid_search.rs ──────────────────────────────────────────
const isLower = c => c >= "a" && c <= "z";
const isDigit = c => c >= "0" && c <= "9";
const isUpper = c => c >= "A" && c <= "Z";
function splitIdentifier(token) {
  const result = [];
  for (const part of token.split(/[_-]/).filter(Boolean)) {
    let cur = "";
    for (let i = 0; i < part.length; i++) {
      const ch = part[i];
      const prev = i > 0 && (isLower(part[i - 1]) || isDigit(part[i - 1]));
      if (isUpper(ch) && prev && cur) { const l = cur.toLowerCase(); if (!result.includes(l)) result.push(l); cur = ""; }
      cur += ch;
    }
    const l = cur.toLowerCase();
    if (l && !result.includes(l)) result.push(l);
  }
  return result.length ? result : [token.toLowerCase()];
}
function tokenizeWithIdentifiers(text) {
  const terms = [], seen = new Set();
  for (const tok of text.split(/[^A-Za-z0-9_-]+/)) {
    if (!tok || tok.length < 2) continue;
    const lo = tok.toLowerCase();
    if (!seen.has(lo)) { terms.push(lo); seen.add(lo); }
    for (const sub of splitIdentifier(tok)) if (!seen.has(sub)) { terms.push(sub); seen.add(sub); }
  }
  return terms; // deduped, so tf is always 1 — matches dedupe_terms() upstream
}
function buildBm25(docs) {
  const terms = new Map(), lengths = new Map(), df = new Map(), lower = new Map();
  let total = 0;
  for (const d of docs) {
    const t = tokenizeWithIdentifiers(d.text);
    terms.set(d.id, new Set(t));
    lengths.set(d.id, t.length);
    lower.set(d.id, d.text.toLowerCase());
    total += t.length;
    for (const term of new Set(t)) df.set(term, (df.get(term) || 0) + 1);
  }
  return { docs, terms, lengths, df, lower, avg: total / docs.length, n: docs.length };
}
function bm25Search(ix, query, limit) {
  const qt = tokenizeWithIdentifiers(query);
  if (!qt.length) return [];
  const ql = query.toLowerCase();
  const hits = [];
  for (const d of ix.docs) {
    const has = ix.terms.get(d.id);
    const dl = ix.lengths.get(d.id);
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

// ── sparse vectors ────────────────────────────────────────────────────────────
function truncate(vec, k) {
  if (vec.term_ids.length <= k) return vec;
  const pairs = vec.term_ids.map((t, i) => [t, vec.weights[i]]);
  pairs.sort((a, b) => b[1] - a[1] || a[0] - b[0]);
  const kept = pairs.slice(0, k).sort((a, b) => a[0] - b[0]);
  return { id: vec.id, term_ids: kept.map(p => p[0]), weights: kept.map(p => p[1]) };
}
function buildSparseIndex(vectors) {
  const postings = new Map();
  let count = 0;
  for (const v of vectors) {
    for (let i = 0; i < v.term_ids.length; i++) {
      const t = v.term_ids[i];
      if (!postings.has(t)) postings.set(t, []);
      postings.get(t).push([v.id, v.weights[i]]);
      count++;
    }
  }
  return { postings, count, docs: vectors.length };
}
function sparseSearch(ix, qvec, limit) {
  const scores = new Map();
  for (let i = 0; i < qvec.term_ids.length; i++) {
    const list = ix.postings.get(qvec.term_ids[i]);
    if (!list) continue;
    const qw = qvec.weights[i];
    for (const [id, dw] of list) scores.set(id, (scores.get(id) || 0) + qw * dw);
  }
  return [...scores.entries()].sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1)).slice(0, limit);
}

// ── metrics ───────────────────────────────────────────────────────────────────
const idsOf = r => r.map(x => x[0]);
function recall(rank, rel, k) { const S = new Set(rel); let n = 0; for (const d of rank.slice(0, k)) if (S.has(d)) n++; return S.size ? n / S.size : 0; }
function mrr(rank, rel, k) { const S = new Set(rel); for (let i = 0; i < Math.min(k, rank.length); i++) if (S.has(rank[i])) return 1 / (i + 1); return 0; }
function ndcg(rank, rel, k) {
  const S = new Set(rel);
  const dcg = g => g.reduce((s, v, i) => s + v / Math.log2(i + 2), 0);
  const a = dcg(rank.slice(0, k).map(d => (S.has(d) ? 1 : 0)));
  const ideal = dcg(new Array(Math.min(k, S.size)).fill(1));
  return ideal ? a / ideal : 0;
}
function evaluate(queries, rankings) {
  const m = f => queries.reduce((s, q) => s + f(rankings[q.id] || [], q.relevance), 0) / queries.length;
  return {
    "R@1": m((r, l) => recall(r, l, 1)),
    "R@5": m((r, l) => recall(r, l, 5)),
    "R@10": m((r, l) => recall(r, l, 10)),
    "MRR@10": m((r, l) => mrr(r, l, 10)),
    "nDCG@10": m((r, l) => ndcg(r, l, 10)),
  };
}

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

module.exports = {
  buildCorpus, buildBm25, bm25Search, truncate, buildSparseIndex, sparseSearch,
  evaluate, cascade, rrf, idsOf, tokenizeWithIdentifiers, DUMP,
};
