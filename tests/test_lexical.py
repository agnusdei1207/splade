from splade_poc.corpus import Document
from splade_poc.fusion import reciprocal_rank_fusion
from splade_poc.lexical import rank_bm25, tokenize_with_identifiers


def document(doc_id: str, text: str) -> Document:
    return Document(id=doc_id, title=doc_id, text=text, sha256="0" * 64, bytes=len(text))


def test_identifier_tokenizer_expands_snake_and_camel_case() -> None:
    assert tokenize_with_identifiers("workspace_id getHttpClient") == [
        "workspace_id",
        "workspace",
        "id",
        "gethttpclient",
        "get",
        "http",
        "client",
    ]


def test_bm25_like_ranking_matches_identifier_fragments() -> None:
    documents = [
        document("a.md", "workspace_id selects the workspace"),
        document("b.md", "terminal rendering and colors"),
    ]

    ranked = rank_bm25(documents, "workspace identifier", limit=2)

    assert [hit.document_id for hit in ranked] == ["a.md"]
    assert ranked[0].score > 0


def test_current_bm25_like_deduplicates_document_term_frequency() -> None:
    documents = [
        document("a.md", "cache cache cache"),
        document("b.md", "cache"),
    ]

    ranked = rank_bm25(documents, "cache", limit=2)

    assert ranked[0].score == ranked[1].score
    assert [hit.document_id for hit in ranked] == ["a.md", "b.md"]


def test_rrf_breaks_equal_scores_by_document_id() -> None:
    fused = reciprocal_rank_fusion(
        {
            "bm25": [("b.md", 9.0), ("a.md", 8.0)],
            "splade": [("a.md", 0.9), ("b.md", 0.8)],
        },
        k=60,
        limit=2,
    )

    assert [hit.document_id for hit in fused] == ["a.md", "b.md"]
    assert fused[0].score == fused[1].score
    assert fused[0].engines == ("bm25", "splade")
