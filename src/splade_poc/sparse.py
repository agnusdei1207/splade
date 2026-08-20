from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class SparseVector:
    term_ids: tuple[int, ...]
    weights: tuple[float, ...]

    @classmethod
    def from_pairs(cls, pairs: Iterable[tuple[int, float]], limit: int) -> "SparseVector":
        checked: list[tuple[int, float]] = []
        seen: set[int] = set()
        for term_id, weight in pairs:
            if not 0 <= term_id <= 65535:
                raise ValueError(f"term id outside u16 range: {term_id}")
            if term_id in seen:
                raise ValueError(f"duplicate term id: {term_id}")
            if not math.isfinite(weight) or weight < 0:
                raise ValueError(f"invalid sparse weight: {weight}")
            seen.add(term_id)
            if weight > 0:
                checked.append((term_id, float(weight)))
        selected = sorted(checked, key=lambda pair: (-pair[1], pair[0]))[:limit]
        selected.sort(key=lambda pair: pair[0])
        return cls(
            term_ids=tuple(term_id for term_id, _ in selected),
            weights=tuple(weight for _, weight in selected),
        )

    @property
    def storage_bytes(self) -> int:
        return len(self.term_ids) * 8

    def dot(self, other: "SparseVector") -> float:
        left = right = 0
        score = 0.0
        while left < len(self.term_ids) and right < len(other.term_ids):
            left_id = self.term_ids[left]
            right_id = other.term_ids[right]
            if left_id == right_id:
                score += self.weights[left] * other.weights[right]
                left += 1
                right += 1
            elif left_id < right_id:
                left += 1
            else:
                right += 1
        return score
