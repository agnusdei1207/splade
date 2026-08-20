from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable

from .corpus import Query


def recall_at_k(ranking: list[str], relevance: dict[str, int], k: int) -> float:
    if not relevance:
        return 0.0
    found = len(set(ranking[:k]) & set(relevance))
    return found / len(relevance)


def mrr_at_k(ranking: list[str], relevance: dict[str, int], k: int) -> float:
    for rank, document_id in enumerate(ranking[:k], 1):
        if document_id in relevance:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranking: list[str], relevance: dict[str, int], k: int) -> float:
    def dcg(grades: Iterable[int]) -> float:
        return sum((2**grade - 1) / math.log2(rank + 1) for rank, grade in enumerate(grades, 1))

    actual = dcg(relevance.get(document_id, 0) for document_id in ranking[:k])
    ideal = dcg(sorted(relevance.values(), reverse=True)[:k])
    return actual / ideal if ideal else 0.0


def _aggregate(queries: list[Query], rankings: dict[str, list[str]]) -> dict[str, float | int]:
    judged = [query for query in queries if query.relevance]
    if not judged:
        return {
            "judged_queries": 0,
            "no_answer_queries": len(queries),
            "recall@5": 0.0,
            "recall@10": 0.0,
            "mrr@10": 0.0,
            "ndcg@10": 0.0,
        }
    return {
        "judged_queries": len(judged),
        "no_answer_queries": len(queries) - len(judged),
        "recall@5": sum(recall_at_k(rankings.get(q.id, []), q.relevance, 5) for q in judged)
        / len(judged),
        "recall@10": sum(recall_at_k(rankings.get(q.id, []), q.relevance, 10) for q in judged)
        / len(judged),
        "mrr@10": sum(mrr_at_k(rankings.get(q.id, []), q.relevance, 10) for q in judged)
        / len(judged),
        "ndcg@10": sum(ndcg_at_k(rankings.get(q.id, []), q.relevance, 10) for q in judged)
        / len(judged),
    }


def evaluate_run(queries: list[Query], rankings: dict[str, list[str]]) -> dict:
    result = _aggregate(queries, rankings)
    categories: dict[str, list[Query]] = defaultdict(list)
    for query in queries:
        if query.relevance:
            categories[query.category].append(query)
    result["by_category"] = {
        category: _aggregate(category_queries, rankings)
        for category, category_queries in sorted(categories.items())
    }
    return result
