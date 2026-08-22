"""Run pentesting's own dense encoder so the three-engine fusion can be simulated.

The semantic layer is snowflake-arctic-embed-xs, quantized, shipped inside
`builder_workspace_index/assets`. Reproducing its contract exactly matters more
than convenience here, so this mirrors local_embedding.rs:

  query prefix   "Represent this sentence for searching relevant passages: "
  truncation     512 tokens
  pooling        CLS, that is output[batch, 0, :]
  normalisation  L2, so cosine is a dot product
  language gate  validate_english_query rejects any non-ASCII alphabetic query

The gate is recorded rather than bypassed: a Korean query reaching this layer is
an error in production, and the count of such queries is itself a finding.
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
MAX_SEQUENCE_LENGTH = 512
EMBEDDING_DIM = 384
BATCH = 32


def is_english_dense_query(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    return not any(ch.isalpha() and ord(ch) > 127 for ch in stripped)


def load(asset_dir: Path, threads: int):
    tokenizer = Tokenizer.from_file(str(asset_dir / "tokenizer.json"))
    tokenizer.enable_truncation(max_length=MAX_SEQUENCE_LENGTH)
    tokenizer.enable_padding()
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(asset_dir / "model_quantized.onnx"),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    return tokenizer, session


def embed(tokenizer, session, texts: list[str]) -> np.ndarray:
    wanted = {i.name for i in session.get_inputs()}
    out = np.zeros((len(texts), EMBEDDING_DIM), dtype=np.float32)
    for start in range(0, len(texts), BATCH):
        batch = texts[start : start + BATCH]
        encodings = tokenizer.encode_batch(batch)
        ids = np.array([e.ids for e in encodings], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        types = np.array([e.type_ids for e in encodings], dtype=np.int64)
        feed = {}
        if "input_ids" in wanted:
            feed["input_ids"] = ids
        if "attention_mask" in wanted:
            feed["attention_mask"] = mask
        if "token_type_ids" in wanted:
            feed["token_type_ids"] = types
        hidden = session.run(None, feed)[0]
        cls = np.asarray(hidden)[:, 0, :].astype(np.float32)
        norms = np.linalg.norm(cls, axis=1, keepdims=True)
        out[start : start + len(batch)] = cls / np.maximum(norms, 1e-12)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-root", type=Path, default=Path("/corpus"))
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--semantic-limit", type=int, default=24)
    parser.add_argument("--relative-floor", type=float, default=0.80)
    parser.add_argument("--absolute-floor", type=float, default=0.55)
    args = parser.parse_args()

    asset_dir = (
        args.corpus_root
        / "crates"
        / "builder_workspace_index"
        / "assets"
        / "snowflake-arctic-embed-xs"
    )
    documents = json.loads((args.input_dir / "documents.json").read_text(encoding="utf-8"))
    queries = json.loads((args.input_dir / "queries.json").read_text(encoding="utf-8"))

    load_started = time.perf_counter()
    tokenizer, session = load(asset_dir, args.threads)
    load_seconds = time.perf_counter() - load_started

    document_started = time.perf_counter()
    document_embeddings = embed(tokenizer, session, [d["text"] for d in documents])
    document_seconds = time.perf_counter() - document_started

    accepted = [q for q in queries if is_english_dense_query(q["query"])]
    rejected = [q for q in queries if not is_english_dense_query(q["query"])]

    query_started = time.perf_counter()
    query_embeddings = embed(
        tokenizer, session, [QUERY_PREFIX + q["query"].strip() for q in accepted]
    )
    query_seconds = time.perf_counter() - query_started

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "document-embeddings.npy", document_embeddings)
    np.save(args.output_dir / "query-embeddings.npy", query_embeddings)

    # Reproduce semantic_ranking(): cosine, then the relative/absolute score floor,
    # then the semantic_limit truncation. A query whose best hit is weak returns
    # very few documents, which is a real property of this layer.
    document_ids = [d["id"] for d in documents]
    rankings: dict[str, list] = {}
    kept_counts: list[int] = []
    search_started = time.perf_counter()
    for row, query in enumerate(accepted):
        scores = document_embeddings @ query_embeddings[row]
        order = np.argsort(-scores, kind="stable")
        best = float(scores[order[0]]) if len(order) else 0.0
        floor = max(best * args.relative_floor, args.absolute_floor)
        hits = [
            [document_ids[i], float(scores[i])]
            for i in order[: args.semantic_limit]
            if float(scores[i]) >= floor
        ]
        kept_counts.append(len(hits))
        rankings[query["id"]] = hits
    search_seconds = time.perf_counter() - search_started
    for query in rejected:
        rankings[query["id"]] = []

    (args.output_dir / "rankings.json").write_text(
        json.dumps(rankings, ensure_ascii=False), encoding="utf-8"
    )
    (args.output_dir / "ids.json").write_text(
        json.dumps(
            {
                "documents": document_ids,
                "queries": [q["id"] for q in accepted],
                "rejected_queries": [q["id"] for q in rejected],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    summary = {
        "model": "snowflake-arctic-embed-xs (quantized, from pentesting assets)",
        "embedding_dim": EMBEDDING_DIM,
        "asset_bytes": sum(p.stat().st_size for p in asset_dir.iterdir() if p.is_file()),
        "threads": args.threads,
        "documents": len(documents),
        "queries_total": len(queries),
        "queries_accepted": len(accepted),
        "queries_rejected_non_english": len(rejected),
        "rejected_by_language": {
            language: sum(1 for q in rejected if q.get("language") == language)
            for language in ("ko", "en", "other")
        },
        "model_load_seconds": load_seconds,
        "document_encode_seconds": document_seconds,
        "documents_per_second": len(documents) / document_seconds,
        "query_encode_seconds": query_seconds,
        "queries_per_second": len(accepted) / query_seconds if accepted else 0.0,
        "search_seconds": search_seconds,
        "semantic_limit": args.semantic_limit,
        "relative_floor": args.relative_floor,
        "absolute_floor": args.absolute_floor,
        "hits_after_floor": {
            "mean": (sum(kept_counts) / len(kept_counts)) if kept_counts else 0.0,
            "zero_hit_queries": sum(1 for c in kept_counts if c == 0),
        },
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
