from splade_poc.models import MODEL_SPECS, prepare_sentence_transformers_compat


def test_registry_contains_exactly_the_three_approved_models() -> None:
    assert [spec.hf_id for spec in MODEL_SPECS] == [
        "tomaarsen/inference-free-splade-bert-tiny-nq",
        "opensearch-project/opensearch-neural-sparse-encoding-doc-v2-mini",
        "rasyosef/splade-tiny",
    ]
    assert [spec.key for spec in MODEL_SPECS] == [
        "if-bert-tiny",
        "if-opensearch-mini",
        "splade-tiny",
    ]
    assert all(spec.vocab_size <= 65536 for spec in MODEL_SPECS)
    assert all(spec.max_document_terms == 256 for spec in MODEL_SPECS)
    assert all(spec.max_query_terms == 32 for spec in MODEL_SPECS)


def test_legacy_idf_module_resolves_to_sparse_static_embedding() -> None:
    prepare_sentence_transformers_compat()

    from sentence_transformers.sparse_encoder.models import IDF, SparseStaticEmbedding

    assert IDF is SparseStaticEmbedding
