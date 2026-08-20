import json
from pathlib import Path

import pytest

from splade_poc.corpus import load_corpus, load_queries, write_corpus_manifest


def test_load_corpus_uses_stable_relative_ids_and_allowed_markdown(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# Root\nVisible", encoding="utf-8")
    (tmp_path / "ARCHITECTURE.md").write_text("# Architecture", encoding="utf-8")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "guide.md").write_text("# Guide\nBody", encoding="utf-8")
    (tmp_path / "docs" / "ignored.txt").write_text("secret", encoding="utf-8")
    (tmp_path / "private.md").write_text("outside scope", encoding="utf-8")

    documents = load_corpus(tmp_path)

    assert [document.id for document in documents] == [
        "ARCHITECTURE.md",
        "README.md",
        "docs/guide.md",
    ]
    assert documents[2].title == "Guide"
    assert documents[2].text == "# Guide\nBody"


def test_load_corpus_rejects_symlink_that_escapes_root(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.md"
    outside.write_text("do not read", encoding="utf-8")
    docs = tmp_path / "docs"
    docs.mkdir()
    try:
        (docs / "escape.md").symlink_to(outside)
    except OSError:
        pytest.skip("symlinks unavailable")

    assert load_corpus(tmp_path) == []


def test_manifest_contains_hashes_but_not_source_text(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# Root\nSensitive body", encoding="utf-8")
    output = tmp_path / "manifest.json"

    write_corpus_manifest(load_corpus(tmp_path), output)

    raw = output.read_text(encoding="utf-8")
    manifest = json.loads(raw)
    assert "Sensitive body" not in raw
    assert manifest[0]["id"] == "README.md"
    assert len(manifest[0]["sha256"]) == 64


def test_benchmark_has_60_valid_judged_queries() -> None:
    documents = load_corpus(Path("/corpus"))
    queries = load_queries(Path("benchmarks/queries.jsonl"), {doc.id for doc in documents})

    assert len(queries) == 60
    assert sum(query.split == "selection" for query in queries) == 36
    assert sum(query.split == "validation" for query in queries) == 24
    assert sum(not query.relevance for query in queries) == 4
    assert {query.category for query in queries} == {
        "exact",
        "semantic",
        "ko-en",
        "no-answer",
    }
    assert all(query.search_text.isascii() for query in queries if query.category == "ko-en")
