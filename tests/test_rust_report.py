import pytest

from splade_poc.rust_report import comparison_data, ensure_parity


def test_comparison_data_uses_deployment_metrics() -> None:
    python = {
        "resources": {
            "documents_per_second": 4.4,
            "query_p95_ms": 1.7,
            "peak_rss_mib": 858.0,
        }
    }
    rust = {
        "documents_per_second": 0.25,
        "query_and_search_p95_ms": 0.38,
        "peak_rss_mib": 785.0,
    }

    assert comparison_data(python, rust) == {
        "Document throughput (docs/s)": (4.4, 0.25),
        "Query + search p95 (ms)": (1.7, 0.38),
        "Peak RSS (MiB)": (858.0, 785.0),
    }


def test_report_rejects_failed_parity_artifact() -> None:
    with pytest.raises(ValueError, match="parity failed"):
        ensure_parity({"top10_exact_queries": 59, "evaluated_queries": 60})
