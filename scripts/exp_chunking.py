"""Experiment: chunking, top-k saturation, throughput and RAM for the sparse encoder.

The shipped pipeline truncates every document to 512 tokens, so most of the corpus
is never indexed. This measures what changes when documents are chunked instead,
how binding the 256-term document cap is, and what the encode cost actually looks
like once the POC's two-CPU limit is lifted.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import time
from pathlib import Path

import torch

from splade_poc.corpus import load_corpus, load_queries
from splade_poc.metrics import evaluate_run
from splade_poc.models import MODEL_SPECS, SparseModel
from splade_poc.sparse import SparseIndex, SparseVector

WINNER = "if-opensearch-mini"
CONTENT_TOKENS = 510  # 512 minus [CLS] and [SEP]


def peak_rss_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def chunk_documents(documents, tokenizer, content_tokens: int, stride: int):
    """Split each document into overlapping token windows, decoded back to text."""
    units: list[tuple[str, str]] = []
    stats = []
    step = content_tokens - stride
    for document in documents:
        ids = tokenizer(document.text, add_special_tokens=False)["input_ids"]
        if not ids:
            continue
        pieces = 0
        for start in range(0, len(ids), step):
            window = ids[start : start + content_tokens]
            if not window:
                break
            units.append((f"{document.id}#{pieces}", tokenizer.decode(window)))
            pieces += 1
            if start + content_tokens >= len(ids):
                break
        stats.append({"id": document.id, "tokens": len(ids), "chunks": pieces})
    return units, stats


def document_scores(index: SparseIndex, query: SparseVector, limit: int) -> list[str]:
    """Search chunks, then max-pool chunk scores up to their parent document."""
    best: dict[str, float] = {}
    for unit_id, score in index.search(query, limit=limit * 8):
        document_id = unit_id.split("#", 1)[0]
        if score > best.get(document_id, float("-inf")):
            best[document_id] = score
    ranked = sorted(best.items(), key=lambda item: (-item[1], item[0]))
    return [document_id for document_id, _ in ranked[:limit]]


def split_metrics(queries, rankings):
    return {
        split: evaluate_run([q for q in queries if q.split == split], rankings)
        for split in ("selection", "validation")
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-root", type=Path, default=Path("/corpus"))
    parser.add_argument("--mode", choices=["truncate", "chunk"], default="chunk")
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--doc-top-k", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--measure-uncapped", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    threads = args.threads or int(os.environ.get("SPLADE_EXP_CPUS", "2"))
    torch.manual_seed(0)
    torch.set_num_threads(threads)

    spec = next(s for s in MODEL_SPECS if s.key == WINNER)
    documents = load_corpus(args.corpus_root)
    queries = load_queries(Path("benchmarks/queries.jsonl"), {d.id for d in documents})

    load_started = time.perf_counter()
    model = SparseModel(spec)
    model_load_seconds = time.perf_counter() - load_started
    tokenizer = model.encoder.tokenizer

    if args.mode == "chunk":
        units, chunk_stats = chunk_documents(documents, tokenizer, CONTENT_TOKENS, args.stride)
    else:
        units = [(d.id, d.text) for d in documents]
        chunk_stats = []

    object.__setattr__(spec, "max_document_terms", args.doc_top_k)
    model.spec = spec

    encode_started = time.perf_counter()
    vectors = model.encode_documents([text for _, text in units], batch_size=args.batch_size)
    encode_seconds = time.perf_counter() - encode_started

    index = SparseIndex({unit_id: vector for (unit_id, _), vector in zip(units, vectors)})

    rankings: dict[str, list[str]] = {}
    latencies: list[float] = []
    query_terms: list[int] = []
    for query in queries:
        started = time.perf_counter()
        vector = model.encode_queries([query.search_text], batch_size=1)[0]
        if args.mode == "chunk":
            ranked = document_scores(index, vector, 24)
        else:
            ranked = [d for d, _ in index.search(vector, limit=24)]
        latencies.append((time.perf_counter() - started) * 1000)
        query_terms.append(len(vector.term_ids))
        rankings[query.id] = ranked

    active = [len(v.term_ids) for v in vectors]
    result = {
        "mode": args.mode,
        "stride": args.stride if args.mode == "chunk" else None,
        "doc_top_k": args.doc_top_k,
        "batch_size": args.batch_size,
        "threads": threads,
        "documents": len(documents),
        "encoded_units": len(units),
        "model_load_seconds": model_load_seconds,
        "encode_seconds": encode_seconds,
        "units_per_second": len(units) / encode_seconds,
        "documents_per_second": len(documents) / encode_seconds,
        "average_active_dims": sum(active) / len(active),
        "capped_units": sum(1 for a in active if a >= args.doc_top_k),
        "average_query_terms": sum(query_terms) / len(query_terms),
        "index_postings": sum(len(p) for p in index.postings.values()),
        "index_bytes_estimate": index.storage_bytes,
        "query_p50_ms": sorted(latencies)[len(latencies) // 2],
        "query_p95_ms": sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)],
        "peak_rss_mib": peak_rss_mib(),
        "metrics": split_metrics(queries, rankings),
        "chunk_stats_summary": {
            "total_tokens": sum(s["tokens"] for s in chunk_stats),
            "average_tokens": (sum(s["tokens"] for s in chunk_stats) / len(chunk_stats)) if chunk_stats else 0,
            "average_chunks": (sum(s["chunks"] for s in chunk_stats) / len(chunk_stats)) if chunk_stats else 0,
            "max_chunks": max((s["chunks"] for s in chunk_stats), default=0),
        },
    }

    if args.measure_uncapped:
        sample = [text for _, text in units[:24]]
        object.__setattr__(spec, "max_document_terms", 30522)
        uncapped = model.encode_documents(sample, batch_size=2)
        counts = sorted(len(v.term_ids) for v in uncapped)
        result["uncapped_active_dims"] = {
            "sample": len(counts),
            "min": counts[0],
            "median": counts[len(counts) // 2],
            "max": counts[-1],
            "mean": sum(counts) / len(counts),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
