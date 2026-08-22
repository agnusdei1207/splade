"""Encode the corpus and query set with several sparse models for a like-for-like run.

Model choice so far assumed an English-only pipeline, because pentesting normalises
every query to English with an LLM call before retrieval. A multilingual sparse
encoder could remove that call entirely, so the candidates have to be measured on
both the English queries the current pipeline produces and the native-language
queries it currently throws away.

Handles both packaging styles: sentence-transformers SparseEncoder when the repo
ships one, otherwise raw transformers with SPLADE max-pooling.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import time
from pathlib import Path

import torch


def splade_pool(logits: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    masked = logits * attention_mask.unsqueeze(-1).to(logits.dtype)
    return masked.relu().log1p().amax(dim=1)


class RawSpladeEncoder:
    """transformers MLM + SPLADE pooling, used when the repo has no ST config."""

    def __init__(self, snapshot: Path, max_length: int = 512) -> None:
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
        self.model = AutoModelForMaskedLM.from_pretrained(
            snapshot, local_files_only=True
        ).eval()
        self.max_length = max_length
        self.vocab_size = int(self.model.config.vocab_size)

    @torch.inference_mode()
    def encode(self, texts: list[str], batch_size: int) -> list[tuple[list[int], list[float]]]:
        rows: list[tuple[list[int], list[float]]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            logits = self.model(**encoded)[0]
            scores = splade_pool(logits, encoded["attention_mask"])
            for row in scores:
                nonzero = torch.nonzero(row, as_tuple=False).flatten()
                rows.append(
                    (
                        [int(i) for i in nonzero],
                        [float(row[i]) for i in nonzero],
                    )
                )
        return rows


class StEncoder:
    """sentence-transformers SparseEncoder with document/query asymmetry preserved."""

    def __init__(self, snapshot: Path) -> None:
        from splade_poc.models import prepare_sentence_transformers_compat

        prepare_sentence_transformers_compat()
        from sentence_transformers import SparseEncoder

        self.encoder = SparseEncoder(str(snapshot), device="cpu")
        self.vocab_size = int(
            getattr(self.encoder, "max_active_dims", None) or self.encoder[0].auto_model.config.vocab_size
        )

    def _rows(self, tensor) -> list[tuple[list[int], list[float]]]:
        tensor = tensor.coalesce()
        indices = tensor.indices().cpu()
        values = tensor.values().cpu()
        rows: list[tuple[list[int], list[float]]] = [([], []) for _ in range(int(tensor.shape[0]))]
        for offset in range(values.numel()):
            row = int(indices[0, offset])
            rows[row][0].append(int(indices[1, offset]))
            rows[row][1].append(float(values[offset]))
        return rows

    def encode(self, texts: list[str], batch_size: int, kind: str) -> list[tuple[list[int], list[float]]]:
        fn = self.encoder.encode_document if kind == "document" else self.encoder.encode_query
        encoded = fn(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_sparse_tensor=True,
            save_to_cpu=True,
        )
        return self._rows(encoded)


def load_encoder(repo_id: str, revision: str | None):
    from huggingface_hub import snapshot_download

    snapshot = Path(
        snapshot_download(repo_id=repo_id, revision=revision, max_workers=2)
    )
    files = {p.name for p in snapshot.rglob("*")}
    if "modules.json" in files:
        try:
            return StEncoder(snapshot), snapshot, "sentence-transformers"
        except Exception as error:  # noqa: BLE001 - fall back rather than abort the survey
            print(f"    ST 로드 실패({error}); raw 경로로 전환", flush=True)
    return RawSpladeEncoder(snapshot), snapshot, "transformers-raw"


def snapshot_bytes(snapshot: Path) -> int:
    seen: dict[int, int] = {}
    for path in snapshot.rglob("*"):
        if path.is_file():
            stat = path.stat()
            seen.setdefault(stat.st_ino, stat.st_size)
    return sum(seen.values())


def describe(values: list[int]) -> dict:
    values = sorted(values)
    return {
        "min": values[0],
        "p50": values[len(values) // 2],
        "p90": values[int(len(values) * 0.90)],
        "p99": values[min(len(values) - 1, int(len(values) * 0.99))],
        "max": values[-1],
        "mean": sum(values) / len(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--label", required=True)
    parser.add_argument("--input-dir", type=Path, default=Path("artifacts/scale"))
    parser.add_argument("--corpus-root", type=Path, default=Path("/corpus"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--doc-cap", type=int, default=1024)
    parser.add_argument("--query-cap", type=int, default=128)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    threads = int(os.environ.get("SPLADE_EXP_CPUS", "2"))
    torch.manual_seed(0)
    torch.set_num_threads(threads)

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from corpus_support import build_corpus_texts

    documents, queries = build_corpus_texts(args.corpus_root, args.input_dir)

    load_started = time.perf_counter()
    encoder, snapshot, packaging = load_encoder(args.model, args.revision)
    load_seconds = time.perf_counter() - load_started

    def cap(rows, limit):
        out = []
        for term_ids, weights in rows:
            pairs = [(t, w) for t, w in zip(term_ids, weights) if w > 0]
            pairs.sort(key=lambda p: (-p[1], p[0]))
            pairs = sorted(pairs[:limit])
            out.append(([t for t, _ in pairs], [round(w, 6) for _, w in pairs]))
        return out

    doc_started = time.perf_counter()
    if packaging == "sentence-transformers":
        doc_rows = encoder.encode([d["text"] for d in documents], args.batch_size, "document")
    else:
        doc_rows = encoder.encode([d["text"] for d in documents], args.batch_size)
    doc_seconds = time.perf_counter() - doc_started
    doc_rows_full = doc_rows
    doc_rows = cap(doc_rows, args.doc_cap)

    query_started = time.perf_counter()
    if packaging == "sentence-transformers":
        query_rows = encoder.encode([q["query"] for q in queries], 32, "query")
    else:
        query_rows = encoder.encode([q["query"] for q in queries], 32)
    query_seconds = time.perf_counter() - query_started
    query_rows_full = query_rows
    query_rows = cap(query_rows, args.query_cap)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "document-vectors.json").write_text(
        json.dumps(
            [
                {"id": d["id"], "term_ids": t, "weights": w}
                for d, (t, w) in zip(documents, doc_rows, strict=True)
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (args.output_dir / "query-vectors.json").write_text(
        json.dumps(
            [
                {"id": q["id"], "term_ids": t, "weights": w}
                for q, (t, w) in zip(queries, query_rows, strict=True)
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    summary = {
        "label": args.label,
        "model": args.model,
        "packaging": packaging,
        "snapshot_bytes": snapshot_bytes(snapshot),
        "vocab_size": getattr(encoder, "vocab_size", None),
        "threads": threads,
        "batch_size": args.batch_size,
        "documents": len(documents),
        "queries": len(queries),
        "model_load_seconds": load_seconds,
        "document_encode_seconds": doc_seconds,
        "documents_per_second": len(documents) / doc_seconds,
        "query_encode_seconds": query_seconds,
        "queries_per_second": len(queries) / query_seconds,
        "document_active_dims_uncapped": describe([len(t) for t, _ in doc_rows_full]),
        "document_active_dims_capped": describe([len(t) for t, _ in doc_rows]),
        "query_active_dims_uncapped": describe([len(t) for t, _ in query_rows_full]),
        "max_term_id": max((max(t) if t else 0) for t, _ in doc_rows),
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
