import math

import pytest

from splade_poc.sparse import SparseVector


def test_sparse_vector_prunes_by_weight_then_orders_by_term_id() -> None:
    vector = SparseVector.from_pairs([(9, 0.4), (2, 0.8), (5, 0.8)], limit=2)

    assert vector.term_ids == (2, 5)
    assert vector.weights == (0.8, 0.8)


@pytest.mark.parametrize(
    "pairs",
    [
        [(1, -0.1)],
        [(1, math.inf)],
        [(1, math.nan)],
        [(65536, 0.5)],
        [(1, 0.5), (1, 0.4)],
    ],
)
def test_sparse_vector_rejects_invalid_postings(pairs: list[tuple[int, float]]) -> None:
    with pytest.raises(ValueError):
        SparseVector.from_pairs(pairs, limit=256)


def test_sparse_dot_product_uses_only_shared_terms() -> None:
    left = SparseVector.from_pairs([(2, 0.5), (7, 0.2)], limit=256)
    right = SparseVector.from_pairs([(2, 0.4), (9, 1.0)], limit=256)

    assert left.dot(right) == pytest.approx(0.2)
    assert left.storage_bytes == 16
