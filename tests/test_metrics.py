import math

from splade_poc.corpus import Query
from splade_poc.metrics import evaluate_run, mrr_at_k, ndcg_at_k, recall_at_k


def test_retrieval_metrics_use_literal_hand_checked_values() -> None:
    relevance = {"a": 2, "b": 1}
    ranking = ["b", "a", "x"]

    assert recall_at_k(ranking, relevance, 1) == 0.5
    assert mrr_at_k(ranking, relevance, 10) == 1.0
    expected_dcg = 1.0 + 3.0 / math.log2(3)
    ideal_dcg = 3.0 + 1.0 / math.log2(3)
    assert ndcg_at_k(ranking, relevance, 10) == expected_dcg / ideal_dcg


def test_evaluate_run_separates_categories_and_ignores_no_answer_quality() -> None:
    queries = [
        Query("q1", "exact", "exact", "selection", {"a": 2}),
        Query("q2", "semantic", "semantic", "selection", {"b": 2}),
        Query("q3", "absent", "no-answer", "selection", {}),
    ]
    rankings = {"q1": ["a"], "q2": ["x", "b"], "q3": ["x"]}

    metrics = evaluate_run(queries, rankings)

    assert metrics["judged_queries"] == 2
    assert metrics["no_answer_queries"] == 1
    assert metrics["recall@5"] == 1.0
    assert metrics["mrr@10"] == 0.75
    assert metrics["by_category"]["exact"]["recall@10"] == 1.0
