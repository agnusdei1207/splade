from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FusedHit:
    document_id: str
    score: float
    engines: tuple[str, ...]


def reciprocal_rank_fusion(
    rankings: dict[str, list[tuple[str, float]]],
    *,
    k: int = 60,
    limit: int = 24,
) -> list[FusedHit]:
    scores: dict[str, float] = {}
    engines: dict[str, set[str]] = {}
    for engine, ranking in rankings.items():
        for rank, (document_id, _score) in enumerate(ranking, 1):
            scores[document_id] = scores.get(document_id, 0.0) + 1.0 / (k + rank)
            engines.setdefault(document_id, set()).add(engine)
    hits = [
        FusedHit(document_id, score, tuple(sorted(engines[document_id])))
        for document_id, score in scores.items()
    ]
    hits.sort(key=lambda hit: (-hit.score, hit.document_id))
    return hits[:limit]
