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


def cascade(
    primary: list[tuple[str, float]],
    backfill: list[tuple[str, float]],
    *,
    head: int = 8,
    limit: int = 24,
) -> list[FusedHit]:
    """Keep the primary engine's head intact, then backfill with the other engine.

    Equal-weight RRF measurably degrades the sparse ranking: it mixes the weaker
    engine's ordering into positions the stronger engine already had right. This
    keeps the first `head` primary hits in their original order, appends whatever
    the backfill engine found that the primary missed, and only then falls back to
    the primary's tail. Scores from the two engines are never compared, so no
    normalisation is needed.
    """
    if head < 0:
        raise ValueError("head must not be negative")
    hits: list[FusedHit] = []
    seen: set[str] = set()

    def push(document_id: str, score: float, engine: str) -> None:
        if document_id in seen or len(hits) >= limit:
            return
        seen.add(document_id)
        hits.append(FusedHit(document_id, score, (engine,)))

    for document_id, score in primary[:head]:
        push(document_id, score, "primary")
    for document_id, score in backfill:
        push(document_id, score, "backfill")
    for document_id, score in primary[head:]:
        push(document_id, score, "primary")
    return hits


def is_confident(ranking: list[tuple[str, float]], *, threshold: float) -> bool:
    """Whether the top hit clears the abstain threshold.

    A retrieval run with no relevant document still returns a confident looking
    ordering, so the caller needs an explicit way to answer "nothing found".
    """
    return bool(ranking) and ranking[0][1] >= threshold
