from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import resource
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .corpus import Query, load_corpus, load_queries, write_corpus_manifest
from .fusion import reciprocal_rank_fusion
from .lexical import Bm25Index
from .metrics import evaluate_run
from .models import MODEL_SPECS, ModelSpec, SparseModel
from .report import choose_winner, write_report
from .sparse import SparseIndex


def percentile(values: list[float], percentage: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(len(ordered) * percentage / 100))
    return ordered[rank - 1]


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def merge_run_parts(run_dir: Path) -> dict:
    bm25 = _read_json(run_dir / "bm25.json")
    summary = {"bm25": {"selection": bm25["selection"], "validation": bm25["validation"]}, "models": {}}
    for spec in MODEL_SPECS:
        path = run_dir / f"{spec.key}.json"
        if not path.is_file():
            raise ValueError(f"missing model result: {spec.key}")
        result = _read_json(path)
        result.pop("rankings", None)
        summary["models"][spec.key] = result
    return summary


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def corpus_git_sha(root: Path) -> str:
    supplied = os.environ.get("SPLADE_CORPUS_GIT_SHA", "")
    if len(supplied) == 40 and all(character in "0123456789abcdef" for character in supplied):
        return supplied
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
    ).strip()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_manifest(snapshot: Path) -> tuple[list[dict], int]:
    files: list[dict] = []
    unique_content: dict[str, int] = {}
    for path in sorted(item for item in snapshot.rglob("*") if item.is_file()):
        digest = _file_sha256(path)
        size = path.stat().st_size
        files.append({"path": path.relative_to(snapshot).as_posix(), "bytes": size, "sha256": digest})
        unique_content.setdefault(digest, size)
    return files, sum(unique_content.values())


def _metrics_by_split(queries: list[Query], rankings: dict[str, list[str]]) -> dict:
    return {
        split: evaluate_run([query for query in queries if query.split == split], rankings)
        for split in ("selection", "validation")
    }


def evaluate_bm25(run_dir: Path, corpus_root: Path, query_path: Path) -> None:
    documents = load_corpus(corpus_root)
    queries = load_queries(query_path, {document.id for document in documents})
    index = Bm25Index(documents)
    rankings: dict[str, list[dict]] = {}
    latencies: list[float] = []
    for query in queries:
        started = time.perf_counter()
        hits = index.search(query.search_text, limit=24)
        latencies.append((time.perf_counter() - started) * 1000)
        rankings[query.id] = [asdict(hit) for hit in hits]
    ids = {query_id: [hit["document_id"] for hit in hits] for query_id, hits in rankings.items()}
    metrics = _metrics_by_split(queries, ids)
    _write_json(
        run_dir / "bm25.json",
        {
            **metrics,
            "resources": {"query_p50_ms": percentile(latencies, 50), "query_p95_ms": percentile(latencies, 95)},
            "rankings": rankings,
        },
    )
    write_corpus_manifest(documents, run_dir / "corpus-manifest.json")
    query_bytes = query_path.read_bytes()
    _write_json(
        run_dir / "manifest.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "corpus_git_sha": corpus_git_sha(corpus_root),
            "corpus_documents": len(documents),
            "corpus_bytes": sum(document.bytes for document in documents),
            "queries": len(queries),
            "queries_sha256": hashlib.sha256(query_bytes).hexdigest(),
            "models": [asdict(spec) for spec in MODEL_SPECS],
        },
    )
    _write_json(
        run_dir / "environment.json",
        {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "memory_max": Path("/sys/fs/cgroup/memory.max").read_text().strip(),
            "pids_max": Path("/sys/fs/cgroup/pids.max").read_text().strip(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("sentence-transformers", "torch", "transformers", "matplotlib")
            },
        },
    )


def evaluate_sparse_model(
    spec: ModelSpec, run_dir: Path, corpus_root: Path, query_path: Path
) -> None:
    import torch

    torch.manual_seed(0)
    torch.set_num_threads(2)
    documents = load_corpus(corpus_root)
    queries = load_queries(query_path, {document.id for document in documents})
    load_started = time.perf_counter()
    model = SparseModel(spec)
    model_load_seconds = time.perf_counter() - load_started
    model_files, model_bytes = _snapshot_manifest(model.snapshot)
    encode_started = time.perf_counter()
    document_vectors = model.encode_documents([document.text for document in documents], batch_size=2)
    document_encode_seconds = time.perf_counter() - encode_started
    vectors_by_id = {
        document.id: vector for document, vector in zip(documents, document_vectors, strict=True)
    }
    index = SparseIndex(vectors_by_id)
    bm25 = _read_json(run_dir / "bm25.json")["rankings"]
    model_rankings: dict[str, list[dict]] = {}
    fused_rankings: dict[str, list[str]] = {}
    per_query: list[dict] = []
    latencies: list[float] = []
    query_active_dims: list[int] = []
    for query in queries:
        started = time.perf_counter()
        vector = model.encode_queries([query.search_text], batch_size=1)[0]
        hits = index.search(vector, limit=24)
        elapsed_ms = (time.perf_counter() - started) * 1000
        latencies.append(elapsed_ms)
        query_active_dims.append(len(vector.term_ids))
        model_rankings[query.id] = [
            {"document_id": document_id, "score": score} for document_id, score in hits
        ]
        fused = reciprocal_rank_fusion(
            {
                "bm25": [(hit["document_id"], hit["score"]) for hit in bm25[query.id]],
                "splade": hits,
            },
            limit=24,
        )
        fused_rankings[query.id] = [hit.document_id for hit in fused]
        per_query.append(
            {
                "id": query.id,
                "split": query.split,
                "category": query.category,
                "latency_ms": elapsed_ms,
                "active_dims": len(vector.term_ids),
                "model_ranking": model_rankings[query.id],
                "fused_ranking": fused_rankings[query.id],
            }
        )
    model_ids = {
        query_id: [hit["document_id"] for hit in hits] for query_id, hits in model_rankings.items()
    }
    selection_queries = [query for query in queries if query.split == "selection"]
    validation_queries = [query for query in queries if query.split == "validation"]
    average_document_terms = sum(len(vector.term_ids) for vector in document_vectors) / len(document_vectors)
    result = {
        "key": spec.key,
        "spec": asdict(spec),
        "license": spec.license,
        "selection": {
            "model": evaluate_run(selection_queries, model_ids),
            "fused": evaluate_run(selection_queries, fused_rankings),
        },
        "validation": {
            "model": evaluate_run(validation_queries, model_ids),
            "fused": evaluate_run(validation_queries, fused_rankings),
        },
        "resources": {
            "model_load_seconds": model_load_seconds,
            "document_encode_seconds": document_encode_seconds,
            "documents_per_second": len(documents) / document_encode_seconds,
            "query_p50_ms": percentile(latencies, 50),
            "query_p95_ms": percentile(latencies, 95),
            "query_p99_ms": percentile(latencies, 99),
            "average_query_terms": sum(query_active_dims) / len(query_active_dims),
            "average_document_terms": average_document_terms,
            "index_bytes": index.storage_bytes,
            "projected_index_mib_10k": average_document_terms * 8 * 10000 / (1024 * 1024),
            "model_mib": model_bytes / (1024 * 1024),
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        },
        "model_files": model_files,
        "rankings": model_rankings,
    }
    _write_json(run_dir / f"{spec.key}.json", result)
    with (run_dir / f"per-query-{spec.key}.jsonl").open("w", encoding="utf-8") as stream:
        for row in per_query:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model", choices=["bm25", *(spec.key for spec in MODEL_SPECS), "finalize"], required=True)
    parser.add_argument("--corpus-root", type=Path, default=Path("/corpus"))
    parser.add_argument("--queries", type=Path, default=Path("benchmarks/queries.jsonl"))
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    with (args.run_dir / "commands.log").open("a", encoding="utf-8") as stream:
        stream.write(" ".join(sys.argv) + "\n")
    if args.model == "bm25":
        evaluate_bm25(args.run_dir, args.corpus_root, args.queries)
    elif args.model == "finalize":
        summary = merge_run_parts(args.run_dir)
        _write_json(args.run_dir / "summary.json", summary)
        write_report(summary, choose_winner(summary), args.run_dir)
    else:
        spec = next(spec for spec in MODEL_SPECS if spec.key == args.model)
        evaluate_sparse_model(spec, args.run_dir, args.corpus_root, args.queries)


if __name__ == "__main__":
    main()
