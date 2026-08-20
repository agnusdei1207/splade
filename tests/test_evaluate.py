import json
from pathlib import Path

import pytest

from splade_poc.evaluate import corpus_git_sha, merge_run_parts, percentile


def test_percentile_uses_nearest_rank_without_interpolation() -> None:
    assert percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 95) == 4.0


def test_corpus_git_sha_uses_value_supplied_by_read_only_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SPLADE_CORPUS_GIT_SHA", "a" * 40)

    assert corpus_git_sha(Path("/path/without/git")) == "a" * 40


def test_merge_requires_all_approved_model_results(tmp_path: Path) -> None:
    (tmp_path / "bm25.json").write_text(
        json.dumps({"selection": {}, "validation": {}, "rankings": {}}), encoding="utf-8"
    )
    (tmp_path / "if-bert-tiny.json").write_text(json.dumps({"key": "if-bert-tiny"}), encoding="utf-8")

    with pytest.raises(ValueError, match="missing model result"):
        merge_run_parts(tmp_path)


def test_merge_strips_rankings_from_summary(tmp_path: Path) -> None:
    (tmp_path / "bm25.json").write_text(
        json.dumps({"selection": {"recall@10": 1}, "validation": {"recall@10": 1}, "rankings": {"q": []}}),
        encoding="utf-8",
    )
    for key in ("if-bert-tiny", "if-opensearch-mini", "splade-tiny"):
        (tmp_path / f"{key}.json").write_text(
            json.dumps({"key": key, "rankings": {"q": []}, "selection": {}, "validation": {}}),
            encoding="utf-8",
        )

    summary = merge_run_parts(tmp_path)

    assert summary["bm25"] == {"selection": {"recall@10": 1}, "validation": {"recall@10": 1}}
    assert set(summary["models"]) == {"if-bert-tiny", "if-opensearch-mini", "splade-tiny"}
    assert all("rankings" not in result for result in summary["models"].values())
