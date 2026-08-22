"""Encode chunks once and dump per-query chunk hits so aggregation can be swept offline.

Chunking lost precision against plain truncation. The likely cause is max-pooling:
a document split into 149 chunks gets 149 chances to score high. Dumping the chunk
level hits lets every aggregation rule be compared without paying the encode cost again.
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
from splade_poc.models import MODEL_SPECS, SparseModel
from splade_poc.sparse import SparseIndex

WINNER = "if-opensearch-mini"
CONTENT_TOKENS = 510


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-root", type=Path, default=Path("/corpus"))
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--chunk-hits", type=int, default=400)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    threads = int(os.environ.get("SPLADE_EXP_CPUS", "2"))
    torch.manual_seed(0)
    torch.set_num_threads(threads)

    spec = next(s for s in MODEL_SPECS if s.key == WINNER)
    documents = load_corpus(args.corpus_root)
    queries = load_queries(Path("benchmarks/queries.jsonl"), {d.id for d in documents})

    model = SparseModel(spec)
    tokenizer = model.encoder.tokenizer

    units: list[tuple[str, str]] = []
    chunk_index: dict[str, tuple[str, int, int]] = {}
    step = CONTENT_TOKENS - args.stride
    for document in documents:
        ids = tokenizer(document.text, add_special_tokens=False)["input_ids"]
        if not ids:
            continue
        pieces = 0
        windows = []
        for start in range(0, len(ids), step):
            window = ids[start : start + CONTENT_TOKENS]
            if not window:
                break
            windows.append(window)
            if start + CONTENT_TOKENS >= len(ids):
                break
        for position, window in enumerate(windows):
            unit_id = f"{document.id}#{position}"
            units.append((unit_id, tokenizer.decode(window)))
            chunk_index[unit_id] = (document.id, position, len(windows))
            pieces += 1

    encode_started = time.perf_counter()
    vectors = model.encode_documents([text for _, text in units], batch_size=args.batch_size)
    encode_seconds = time.perf_counter() - encode_started

    index = SparseIndex({unit_id: vector for (unit_id, _), vector in zip(units, vectors)})

    hits: dict[str, list[tuple[str, float]]] = {}
    for query in queries:
        vector = model.encode_queries([query.search_text], batch_size=1)[0]
        hits[query.id] = index.search(vector, limit=args.chunk_hits)

    payload = {
        "encoded_units": len(units),
        "documents": len(documents),
        "threads": threads,
        "batch_size": args.batch_size,
        "stride": args.stride,
        "encode_seconds": encode_seconds,
        "units_per_second": len(units) / encode_seconds,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "index_postings": sum(len(p) for p in index.postings.values()),
        "chunk_index": {k: list(v) for k, v in chunk_index.items()},
        "hits": {qid: [[d, s] for d, s in rows] for qid, rows in hits.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k not in ("chunk_index", "hits")}, indent=2))


if __name__ == "__main__":
    main()
