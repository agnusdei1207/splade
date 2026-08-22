"""Shared corpus and query construction, so every model run sees identical inputs.

Mirrors what pentesting's `MarkdownNote::searchable_text` feeds the lexical index
(title, aliases, tags, body) and holds section headings out of the body so a
heading used as a query is not a trivial substring of its own document.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

SKIP_DIRECTORIES = {".git", "node_modules", "target", "_archive", "prompts", ".cache", ".worktrees"}
HEADING_RE = re.compile(r"^(#{2,4})[ \t]+(.+?)[ \t]*$")
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
    meta: dict = {}
    for line in raw[3:end].splitlines():
        if ":" not in line or line.startswith((" ", "\t", "-")):
            continue
        key, _, value = line.partition(":")
        value = value.strip().strip("[]")
        if value:
            meta[key.strip().lower()] = [
                item.strip().strip("\"'") for item in value.split(",") if item.strip()
            ]
    return meta, raw[end + 4 :].lstrip("\n")


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
                    continue
            kept_lines.append(line)
        title_match = re.search(r"^#[ \t]+(.+?)[ \t]*$", body, re.MULTILINE)
        title = (
            title_match.group(1).strip()
            if title_match
            else path.stem.replace("_", " ").replace("-", " ")
        )
        aliases = " ".join(meta.get("aliases", []))
        tags = " ".join(meta.get("tags", []))
        documents.append(
            {
                "id": document_id,
                "title": title,
                "text": f"{title}\n{aliases}\n{tags}\n" + "\n".join(kept_lines),
                "bytes": len(raw.encode("utf-8")),
            }
        )
        for heading in set(held_out):
            headings_to_documents.setdefault(heading.lower(), []).append(document_id)
    return documents, headings_to_documents


def build_queries(headings_to_documents: dict[str, list[str]]) -> list[dict]:
    return [
        {"id": f"h{index:05d}", "query": heading, "relevance": sorted(document_ids)}
        for index, (heading, document_ids) in enumerate(sorted(headings_to_documents.items()))
    ]


def build_corpus_texts(corpus_root: Path, prepared: Path | None = None):
    """Returns (documents, queries).

    A prepared directory wins outright: gen_query_families.py holds more text out
    of each document than the heading-only path here, so rebuilding the corpus
    locally would leak query text back into the index and silently inflate scores.
    """
    if prepared:
        documents_path = prepared / "documents.json"
        queries_path = prepared / "queries.json"
        if documents_path.is_file() and queries_path.is_file():
            return (
                json.loads(documents_path.read_text(encoding="utf-8")),
                json.loads(queries_path.read_text(encoding="utf-8")),
            )
    documents, heading_map = build_corpus(corpus_root)
    queries = build_queries(heading_map)
    if prepared and (prepared / "queries.json").is_file():
        queries = json.loads((prepared / "queries.json").read_text(encoding="utf-8"))
    return documents, queries
