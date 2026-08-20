from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    text: str
    sha256: str
    bytes: int


@dataclass(frozen=True)
class Query:
    id: str
    query: str
    category: str
    split: str
    relevance: dict[str, int]


def _title(path: Path, text: str) -> str:
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return path.stem.replace("_", " ").replace("-", " ")


def load_corpus(root: Path) -> list[Document]:
    root = root.resolve()
    candidates = [root / "README.md", root / "ARCHITECTURE.md"]
    candidates.extend((root / "docs").rglob("*.md") if (root / "docs").is_dir() else [])
    documents: list[Document] = []
    for path in sorted(set(candidates)):
        if not path.is_file():
            continue
        resolved = path.resolve()
        if not resolved.is_relative_to(root):
            continue
        relative = resolved.relative_to(root).as_posix()
        if "/_archive/" in f"/{relative}/" or "/prompts/" in f"/{relative}/":
            continue
        raw = resolved.read_bytes()
        text = raw.decode("utf-8")
        documents.append(
            Document(
                id=relative,
                title=_title(resolved, text),
                text=text,
                sha256=hashlib.sha256(raw).hexdigest(),
                bytes=len(raw),
            )
        )
    return documents


def write_corpus_manifest(documents: list[Document], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for document in documents:
        metadata = asdict(document)
        metadata.pop("text")
        rows.append(metadata)
    output.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_queries(path: Path, corpus_ids: set[str]) -> list[Query]:
    queries: list[Query] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        query = Query(
            id=str(row["id"]),
            query=str(row["query"]).strip(),
            category=str(row["category"]),
            split=str(row["split"]),
            relevance={str(key): int(value) for key, value in row["relevance"].items()},
        )
        if query.id in seen:
            raise ValueError(f"duplicate query id {query.id} on line {line_number}")
        if not query.query:
            raise ValueError(f"empty query on line {line_number}")
        if query.split not in {"selection", "validation"}:
            raise ValueError(f"invalid split {query.split} on line {line_number}")
        missing = sorted(set(query.relevance) - corpus_ids)
        if missing:
            raise ValueError(f"unknown relevance targets on line {line_number}: {missing}")
        if any(value not in {1, 2} for value in query.relevance.values()):
            raise ValueError(f"invalid relevance grade on line {line_number}")
        seen.add(query.id)
        queries.append(query)
    return queries
