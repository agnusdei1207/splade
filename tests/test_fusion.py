import pytest

from splade_poc.fusion import cascade, is_confident, reciprocal_rank_fusion


def test_cascade_keeps_the_primary_head_before_backfilling() -> None:
    primary = [("a", 9.0), ("b", 8.0), ("c", 7.0), ("d", 6.0)]
    backfill = [("x", 3.0), ("b", 2.0), ("y", 1.0)]
    hits = cascade(primary, backfill, head=2, limit=6)
    assert [hit.document_id for hit in hits] == ["a", "b", "x", "y", "c", "d"]
    assert [hit.engines for hit in hits[:2]] == [("primary",), ("primary",)]
    assert hits[2].engines == ("backfill",)


def test_cascade_recovers_documents_the_primary_engine_missed() -> None:
    primary = [("a", 9.0), ("b", 8.0)]
    backfill = [("rescued", 5.0)]
    hits = cascade(primary, backfill, head=8, limit=24)
    assert "rescued" in [hit.document_id for hit in hits]


def test_cascade_respects_the_limit_and_rejects_a_negative_head() -> None:
    primary = [(f"p{i}", float(10 - i)) for i in range(10)]
    backfill = [(f"b{i}", float(i)) for i in range(10)]
    assert len(cascade(primary, backfill, head=3, limit=5)) == 5
    with pytest.raises(ValueError):
        cascade(primary, backfill, head=-1)


def test_cascade_with_zero_head_is_backfill_first() -> None:
    hits = cascade([("a", 1.0)], [("b", 2.0)], head=0, limit=4)
    assert [hit.document_id for hit in hits] == ["b", "a"]


def test_is_confident_uses_the_top_score_only() -> None:
    assert is_confident([("a", 8.0), ("b", 99.0)], threshold=7.5)
    assert not is_confident([("a", 7.0)], threshold=7.5)
    assert not is_confident([], threshold=0.0)


def test_reciprocal_rank_fusion_still_records_contributing_engines() -> None:
    hits = reciprocal_rank_fusion(
        {"bm25": [("a", 1.0), ("b", 0.5)], "splade": [("b", 9.0)]}, limit=2
    )
    assert hits[0].document_id == "b"
    assert hits[0].engines == ("bm25", "splade")
