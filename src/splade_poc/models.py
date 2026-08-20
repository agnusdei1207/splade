from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .sparse import SparseVector


@dataclass(frozen=True)
class ModelSpec:
    key: str
    hf_id: str
    revision: str
    vocab_size: int
    inference_free_query: bool
    license: str
    max_query_terms: int = 32
    max_document_terms: int = 256


MODEL_SPECS = (
    ModelSpec(
        key="if-bert-tiny",
        hf_id="tomaarsen/inference-free-splade-bert-tiny-nq",
        revision="37bc323e08e4e99d9e36d2c55d25877a579b3745",
        vocab_size=30522,
        inference_free_query=True,
        license="apache-2.0",
    ),
    ModelSpec(
        key="if-opensearch-mini",
        hf_id="opensearch-project/opensearch-neural-sparse-encoding-doc-v2-mini",
        revision="4af867a426867dfdd744097531046f4289a32fdd",
        vocab_size=30522,
        inference_free_query=True,
        license="apache-2.0",
    ),
    ModelSpec(
        key="splade-tiny",
        hf_id="rasyosef/splade-tiny",
        revision="7391972eac4411e33efff5fad27b886ec97895c0",
        vocab_size=30522,
        inference_free_query=False,
        license="mit",
    ),
)


def prepare_sentence_transformers_compat() -> None:
    from sentence_transformers.sparse_encoder import models

    if not hasattr(models, "IDF"):
        models.IDF = models.SparseStaticEmbedding
    try:
        from sentence_transformers.sparse_encoder import modules
    except ImportError:
        return
    if not hasattr(modules, "IDF"):
        modules.IDF = modules.SparseStaticEmbedding


def download_snapshot(spec: ModelSpec) -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=spec.hf_id,
            revision=spec.revision,
            max_workers=1,
        )
    )


def _rows_from_sparse_tensor(tensor: Any, limit: int) -> list[SparseVector]:
    tensor = tensor.coalesce()
    indices = tensor.indices().cpu()
    values = tensor.values().cpu()
    batch_size = int(tensor.shape[0])
    rows: list[list[tuple[int, float]]] = [[] for _ in range(batch_size)]
    for offset in range(values.numel()):
        row = int(indices[0, offset])
        term_id = int(indices[1, offset])
        rows[row].append((term_id, float(values[offset])))
    return [SparseVector.from_pairs(row, limit) for row in rows]


class SparseModel:
    def __init__(self, spec: ModelSpec) -> None:
        prepare_sentence_transformers_compat()
        from sentence_transformers import SparseEncoder

        self.spec = spec
        self.snapshot = download_snapshot(spec)
        self.encoder = SparseEncoder(str(self.snapshot), device="cpu")

    def encode_documents(self, texts: list[str], batch_size: int = 4) -> list[SparseVector]:
        encoded = self.encoder.encode_document(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_sparse_tensor=True,
            save_to_cpu=True,
            max_active_dims=self.spec.max_document_terms,
        )
        return _rows_from_sparse_tensor(encoded, self.spec.max_document_terms)

    def encode_queries(self, texts: list[str], batch_size: int = 16) -> list[SparseVector]:
        encoded = self.encoder.encode_query(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_sparse_tensor=True,
            save_to_cpu=True,
            max_active_dims=self.spec.max_query_terms,
        )
        return _rows_from_sparse_tensor(encoded, self.spec.max_query_terms)
