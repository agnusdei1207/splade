import json
from pathlib import Path

from splade_poc.report import choose_winner, write_report


def fixture_summary() -> dict:
    base = {"recall@10": 0.5, "ndcg@10": 0.5, "by_category": {"exact": {"recall@10": 1.0}}}
    passing = {
        "spec": {"inference_free_query": True},
        "license": "apache-2.0",
        "selection": {"fused": {"recall@10": 0.7, "ndcg@10": 0.8, "by_category": {"exact": {"recall@10": 1.0}}}},
        "validation": {"fused": {"recall@10": 0.6, "ndcg@10": 0.7}},
        "resources": {
            "query_p50_ms": 1.0,
            "query_p95_ms": 2.0,
            "projected_index_mib_10k": 10.0,
            "model_mib": 17.0,
            "peak_rss_mib": 200.0,
        },
    }
    failing = json.loads(json.dumps(passing))
    failing["selection"]["fused"]["ndcg@10"] = 0.9
    failing["resources"]["projected_index_mib_10k"] = 40.0
    return {"bm25": {"selection": base, "validation": base}, "models": {"a": passing, "b": failing}}


def test_winner_uses_fixed_gates_before_selection_ndcg() -> None:
    decision = choose_winner(fixture_summary())

    assert decision["winner"] == "a"
    assert decision["models"]["a"]["passed"] is True
    assert decision["models"]["b"]["passed"] is False
    assert "index>32MiB/10k" in decision["models"]["b"]["reasons"]


def test_report_and_svg_files_are_generated_from_summary(tmp_path: Path) -> None:
    summary = fixture_summary()
    decision = choose_winner(summary)

    write_report(summary, decision, tmp_path)

    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "Winner: `a`" in report
    for name in ("quality.svg", "latency.svg", "resources.svg"):
        content = (tmp_path / name).read_text(encoding="utf-8")
        assert "<svg" in content
        assert (tmp_path / name.replace(".svg", ".png")).stat().st_size > 1000
