use std::path::PathBuf;

use serde::Deserialize;
use splade::{DocumentEncoder, QueryEncoder, SparseVector};

#[derive(Deserialize)]
struct Fixture {
    model: String,
    revision: String,
    weight_tolerance: f32,
    documents: Vec<Row>,
    queries: Vec<Row>,
}

#[derive(Deserialize)]
struct Row {
    text: String,
    term_ids: Vec<u16>,
    weights: Vec<f32>,
}

fn fixture() -> Fixture {
    serde_json::from_str(include_str!("../fixtures/python-parity.json")).unwrap()
}

fn model_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/if-opensearch-mini")
}

fn trusted_manifest() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("artifacts/eval/2026-08-20-pentesting-267/rust-export-manifest.json")
}

fn assert_vector(expected: &Row, actual: &SparseVector, tolerance: f32) {
    assert_eq!(actual.term_ids(), expected.term_ids);
    assert_eq!(actual.weights().len(), expected.weights.len());
    for (position, (actual, expected)) in actual.weights().iter().zip(&expected.weights).enumerate()
    {
        assert!(
            (actual - expected).abs() <= tolerance,
            "weight mismatch at {position}: {actual} != {expected}"
        );
    }
}

#[test]
fn rust_query_encoder_matches_python_fixture() {
    let fixture = fixture();
    assert_eq!(fixture.model, "if-opensearch-mini");
    assert_eq!(fixture.revision, "4af867a426867dfdd744097531046f4289a32fdd");
    let encoder = QueryEncoder::from_dir_with_manifest(model_dir(), trusted_manifest()).unwrap();
    for row in &fixture.queries {
        let actual = encoder.encode(&row.text).unwrap();
        assert_vector(row, &actual, fixture.weight_tolerance);
    }
}

#[test]
fn rust_document_encoder_matches_python_fixture() {
    let fixture = fixture();
    let encoder = DocumentEncoder::from_dir_with_manifest(model_dir(), trusted_manifest()).unwrap();
    let texts = fixture
        .documents
        .iter()
        .map(|row| row.text.as_str())
        .collect::<Vec<_>>();
    let actual = encoder.encode_batch(&texts).unwrap();
    for (row, actual) in fixture.documents.iter().zip(&actual) {
        assert_vector(row, actual, fixture.weight_tolerance);
    }
}

#[test]
fn trusted_manifest_rejects_tampered_query_weights() {
    let temporary = tempfile::tempdir().unwrap();
    std::fs::copy(
        model_dir().join("tokenizer.json"),
        temporary.path().join("tokenizer.json"),
    )
    .unwrap();
    std::fs::copy(
        model_dir().join("query.safetensors"),
        temporary.path().join("query.safetensors"),
    )
    .unwrap();
    let mut bytes = std::fs::read(temporary.path().join("query.safetensors")).unwrap();
    let last = bytes.last_mut().unwrap();
    *last ^= 1;
    std::fs::write(temporary.path().join("query.safetensors"), bytes).unwrap();

    let error = QueryEncoder::from_dir_with_manifest(temporary.path(), trusted_manifest())
        .err()
        .expect("tampered weights must be rejected");
    assert!(error.to_string().contains("SHA-256"));
}
