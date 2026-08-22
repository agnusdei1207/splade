"""Experiment: does restoring term frequency change the BM25 baseline and the gate?

Runs entirely from committed artifacts plus the read-only corpus; no model loading.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

from splade_poc import lexical
from splade_poc.corpus import load_corpus, load_queries
from splade_poc.fusion import reciprocal_rank_fusion
from splade_poc.metrics import evaluate_run

RUN_DIR = Path("artifacts/eval/2026-08-20-pentesting-267")


def tokenize_real_tf(text: str) -> list[str]:
    """Same expansion as the shipped tokenizer, but dedup is per token occurrence.

    The shipped `tokenize_with_identifiers` keeps one `seen` set for the whole
    document, so every term collapses to tf=1. Scoping `seen` to a single token
    keeps the intent (do not double count a token's own subword expansion) while
    letting repeated tokens accumulate a real term frequency.
    """
    terms: list[str] = []
    for match in lexical.TOKEN_RE.finditer(text):
        token = match.group(0)
        if len(token) < 2:
            continue
        seen: set[str] = set()
        lexical._push_unique(terms, seen, token.lower())
        for part in lexical._split_identifier(token):
            lexical._push_unique(terms, seen, part)
    return terms


def build_index(documents, tokenizer):
    original = lexical.tokenize_with_identifiers
    lexical.tokenize_with_identifiers = tokenizer
    try:
        return lexical.Bm25Index(documents)
    finally:
        lexical.tokenize_with_identifiers = original


def search_all(index, queries, tokenizer, limit=24):
    original = lexical.tokenize_with_identifiers
    lexical.tokenize_with_identifiers = tokenizer
    try:
        return {q.id: index.search(q.search_text, limit=limit) for q in queries}
    finally:
        lexical.tokenize_with_identifiers = original


def split_metrics(queries, rankings):
    return {
        split: evaluate_run([q for q in queries if q.split == split], rankings)
        for split in ("selection", "validation")
    }


def main() -> None:
    corpus_root = Path(sys.argv[1] if len(sys.argv) > 1 else "/corpus")
    documents = load_corpus(corpus_root)
    queries = load_queries(Path("benchmarks/queries.jsonl"), {d.id for d in documents})

    splade = json.loads((RUN_DIR / "if-opensearch-mini.json").read_text(encoding="utf-8"))
    splade_rankings = {
        qid: [(h["document_id"], h["score"]) for h in hits]
        for qid, hits in splade["rankings"].items()
    }
    splade_ids = {qid: [d for d, _ in hits] for qid, hits in splade_rankings.items()}

    report: dict = {"splade_alone": split_metrics(queries, splade_ids), "variants": {}}

    for name, tokenizer in (
        ("shipped_tf1", lexical.tokenize_with_identifiers),
        ("real_tf", tokenize_real_tf),
    ):
        index = build_index(documents, tokenizer)
        hits = search_all(index, queries, tokenizer)
        bm25_ids = {qid: [h.document_id for h in rows] for qid, rows in hits.items()}
        fused_ids = {}
        for query in queries:
            fused = reciprocal_rank_fusion(
                {
                    "bm25": [(h.document_id, h.score) for h in hits[query.id]],
                    "splade": splade_rankings[query.id],
                },
                limit=24,
            )
            fused_ids[query.id] = [h.document_id for h in fused]

        lengths = [len(index.document_terms[d.id]) for d in documents]
        max_tf = max(max(c.values()) for c in index.frequencies.values())
        report["variants"][name] = {
            "vocabulary": len(index.document_frequency),
            "average_document_length": sum(lengths) / len(lengths),
            "max_term_frequency": max_tf,
            "bm25": split_metrics(queries, bm25_ids),
            "fused": split_metrics(queries, fused_ids),
        }

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
