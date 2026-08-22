"""Encode the full pentesting Markdown corpus once, uncapped, and dump raw vectors.

The earlier runs used 267 documents and 60 hand-written queries, which was too
small to certify anything. This widens both sides:

- corpus: every Markdown file in the pentesting tree, assembled the way the Rust
  `MarkdownNote::searchable_text` does it (title, aliases, tags, body).
- queries: every section heading, held out of the indexed body so the query is
  not a trivial substring of the document it should retrieve.

Vectors are encoded without a top-k cap so document and query truncation can be
swept offline instead of paying the encode cost once per setting.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import resource
import time
from pathlib import Path

import torch

from splade_poc.models import MODEL_SPECS, SparseModel

WINNER = "if-opensearch-mini"
VOCAB_SIZE = 30522
SKIP_DIRECTORIES = {".git", "node_modules", "target", "_archive", "prompts", ".cache", ".worktrees"}
HEADING_RE = re.compile(r"^(#{2,4})[ \t]+(.+?)[ \t]*$", re.MULTILINE)
GENERIC_HEADING_RE = re.compile(
    r"^(overview|summary|notes|background|references|see also|todo|목적|요약|개요|참고|배경|결론|목차)$",
    re.IGNORECASE,
)


def markdown_files(root: Path) -> list[Path]:
    found: list[Path] = []
    for path in sorted(root.rglob("*.md")):
        if any(part in SKIP_DIRECTORIES for part in path.relative_to(root).parts):
            continue
        if path.is_file():
            found.append(path)
    return found


def parse_frontmatter(raw: str) -> tuple[dict, str]:
    if not raw.startswith("---"):
        return {}, raw
    end = raw.find("\n---", 3)
    if end == -1:
        return {}, raw
    head, body = raw[3:end], raw[end + 4 :]
    meta: dict = {}
    for line in head.splitlines():
        if ":" not in line or line.startswith((" ", "\t", "-")):
            continue
        key, _, value = line.partition(":")
        value = value.strip().strip("[]")
        if not value:
            continue
        meta[key.strip().lower()] = [
            item.strip().strip("\"'") for item in value.split(",") if item.strip()
        ]
    return meta, body.lstrip("\n")


def extract_title(body: str, path: Path) -> str:
    match = re.search(r"^#[ \t]+(.+?)[ \t]*$", body, re.MULTILINE)
    return match.group(1).strip() if match else path.stem.replace("_", " ").replace("-", " ")


def usable_heading(text: str) -> bool:
    words = text.split()
    if not 3 <= len(words) <= 14:
        return False
    if GENERIC_HEADING_RE.match(text):
        return False
    return not re.fullmatch(r"\d+[.)]?", text)


def clean_heading(text: str) -> str:
    return re.sub(r"[`*_#\[\]]", "", text).strip()


def build_corpus(root: Path) -> tuple[list[dict], dict[str, list[str]]]:
    """Returns documents with headings removed, plus heading -> document ids."""
    documents: list[dict] = []
    headings_to_documents: dict[str, list[str]] = {}
    for path in markdown_files(root):
        raw = path.read_text(encoding="utf-8", errors="replace")
        meta, body = parse_frontmatter(raw)
        document_id = path.relative_to(root).as_posix()
        held_out: list[str] = []
        kept_lines: list[str] = []
        for line in body.splitlines():
            match = HEADING_RE.match(line)
            if match:
                heading = clean_heading(match.group(2))
                if usable_heading(heading):
                    held_out.append(heading)
                    continue  # drop the heading line from the indexed body
            kept_lines.append(line)
        stripped_body = "\n".join(kept_lines)
        title = extract_title(body, path)
        aliases = " ".join(meta.get("aliases", []))
        tags = " ".join(meta.get("tags", []))
        searchable = f"{title}\n{aliases}\n{tags}\n{stripped_body}"
        documents.append(
            {
                "id": document_id,
                "title": title,
                "text": searchable,
                "bytes": len(raw.encode("utf-8")),
                "held_out_headings": sorted(set(held_out)),
            }
        )
        for heading in set(held_out):
            headings_to_documents.setdefault(heading.lower(), []).append(document_id)
    return documents, headings_to_documents


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-root", type=Path, default=Path("/corpus"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    threads = int(os.environ.get("SPLADE_EXP_CPUS", "2"))
    torch.manual_seed(0)
    torch.set_num_threads(threads)

    documents, heading_map = build_corpus(args.corpus_root)
    queries = [
        {"id": f"h{index:05d}", "query": heading, "relevance": sorted(document_ids)}
        for index, (heading, document_ids) in enumerate(sorted(heading_map.items()))
    ]

    spec = next(s for s in MODEL_SPECS if s.key == WINNER)
    object.__setattr__(spec, "max_document_terms", VOCAB_SIZE)
    object.__setattr__(spec, "max_query_terms", VOCAB_SIZE)

    load_started = time.perf_counter()
    model = SparseModel(spec)
    model_load_seconds = time.perf_counter() - load_started

    document_started = time.perf_counter()
    document_vectors = model.encode_documents(
        [document["text"] for document in documents], batch_size=args.batch_size
    )
    document_seconds = time.perf_counter() - document_started

    query_started = time.perf_counter()
    query_vectors = model.encode_queries(
        [query["query"] for query in queries], batch_size=64
    )
    query_seconds = time.perf_counter() - query_started

    args.output_dir.mkdir(parents=True, exist_ok=True)

    def rows(items, vectors):
        return [
            {
                "id": item["id"],
                "term_ids": list(vector.term_ids),
                "weights": [round(weight, 6) for weight in vector.weights],
            }
            for item, vector in zip(items, vectors, strict=True)
        ]

    (args.output_dir / "documents.json").write_text(
        json.dumps(
            [{k: v for k, v in d.items() if k != "text"} for d in documents],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (args.output_dir / "queries.json").write_text(
        json.dumps(queries, ensure_ascii=False), encoding="utf-8"
    )
    (args.output_dir / "document-vectors.json").write_text(
        json.dumps(rows(documents, document_vectors), ensure_ascii=False), encoding="utf-8"
    )
    (args.output_dir / "query-vectors.json").write_text(
        json.dumps(rows(queries, query_vectors), ensure_ascii=False), encoding="utf-8"
    )

    document_dims = sorted(len(vector.term_ids) for vector in document_vectors)
    query_dims = sorted(len(vector.term_ids) for vector in query_vectors)

    def describe(values: list[int]) -> dict:
        return {
            "count": len(values),
            "min": values[0],
            "p50": values[len(values) // 2],
            "p90": values[int(len(values) * 0.90)],
            "p99": values[min(len(values) - 1, int(len(values) * 0.99))],
            "max": values[-1],
            "mean": sum(values) / len(values),
        }

    summary = {
        "corpus_root": str(args.corpus_root),
        "documents": len(documents),
        "corpus_bytes": sum(document["bytes"] for document in documents),
        "queries": len(queries),
        "ambiguous_queries": sum(1 for q in queries if len(q["relevance"]) > 1),
        "threads": threads,
        "batch_size": args.batch_size,
        "vocab_size": VOCAB_SIZE,
        "model_load_seconds": model_load_seconds,
        "document_encode_seconds": document_seconds,
        "documents_per_second": len(documents) / document_seconds,
        "query_encode_seconds": query_seconds,
        "queries_per_second": len(queries) / query_seconds,
        "document_active_dims": describe(document_dims),
        "query_active_dims": describe(query_dims),
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
