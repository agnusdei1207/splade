use std::collections::{BTreeMap, HashMap, HashSet};
use std::env;
use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use splade::{DocumentEncoder, QueryEncoder, SparseIndex};

const DOCUMENT_BATCH_SIZE: usize = 4;
const RANKING_LIMIT: usize = 24;
const RUN_DIR: &str = "artifacts/eval/2026-08-20-pentesting-267";
const BENCHMARK_COMMAND: &str = "cargo run --release --locked --bin rust-benchmark --";

#[derive(Deserialize)]
struct Query {
    id: String,
    query: String,
    retrieval_query: Option<String>,
}

#[derive(Deserialize)]
struct EvaluationManifest {
    corpus_git_sha: String,
    corpus_documents: usize,
    corpus_bytes: u64,
    queries: usize,
    queries_sha256: String,
}

#[derive(Deserialize)]
struct CorpusRecord {
    id: String,
    sha256: String,
    bytes: u64,
}

#[derive(Deserialize)]
struct ExpectedEvaluation {
    key: String,
    spec: ExpectedModel,
    rankings: HashMap<String, Vec<ExpectedRank>>,
}

#[derive(Deserialize)]
struct ExpectedModel {
    revision: String,
}

#[derive(Deserialize, Serialize)]
struct ExportManifest {
    model: String,
    revision: String,
    files: BTreeMap<String, AssetRecord>,
}

#[derive(Deserialize, Serialize)]
struct AssetRecord {
    bytes: u64,
    sha256: String,
}

#[derive(Deserialize)]
struct ExpectedRank {
    document_id: String,
    score: f64,
}

struct Document {
    id: String,
    text: String,
    bytes: u64,
    sha256: String,
}

#[derive(Serialize)]
struct ContainerLimits {
    memory_max: String,
    cpu_max: String,
    pids_max: String,
}

#[derive(Serialize)]
struct InputEvidence {
    corpus_git_sha: String,
    queries_sha256: String,
    corpus_manifest_sha256: String,
    python_evaluation_sha256: String,
    export_manifest_sha256: String,
    cargo_lock_sha256: String,
}

#[derive(Serialize)]
struct RustPortEvaluation {
    command: &'static str,
    build_profile: &'static str,
    rustc: String,
    container_limits: ContainerLimits,
    inputs: InputEvidence,
    model: String,
    revision: String,
    model_files: BTreeMap<String, AssetRecord>,
    corpus_documents: usize,
    corpus_bytes: u64,
    evaluated_queries: usize,
    top10_exact_queries: usize,
    top10_position_agreement: f64,
    top10_set_recall: f64,
    max_score_abs_error: f64,
    model_load_seconds: f64,
    document_encode_seconds: f64,
    documents_per_second: f64,
    query_and_search_p50_ms: f64,
    query_and_search_p95_ms: f64,
    average_query_terms: f64,
    average_document_terms: f64,
    serialized_index_bytes: usize,
    projected_serialized_index_mib_10k: f64,
    runtime_model_bytes: u64,
    peak_rss_mib: f64,
}

fn main() -> Result<()> {
    let corpus_root = env::args().nth(1).unwrap_or_else(|| "/corpus".to_owned());
    let output_path = env::args()
        .nth(2)
        .unwrap_or_else(|| format!("{RUN_DIR}/rust-port.json"));
    let model_dir = Path::new("models/if-opensearch-mini");
    let query_path = Path::new("benchmarks/queries.jsonl");
    let corpus_manifest_path = Path::new(RUN_DIR).join("corpus-manifest.json");
    let evaluation_manifest_path = Path::new(RUN_DIR).join("manifest.json");
    let python_evaluation_path = Path::new(RUN_DIR).join("if-opensearch-mini.json");
    let export_manifest_path = Path::new(RUN_DIR).join("rust-export-manifest.json");

    let evaluation_manifest: EvaluationManifest = read_json(&evaluation_manifest_path)?;
    let export_manifest: ExportManifest = read_json(&export_manifest_path)?;
    let expected: ExpectedEvaluation = read_json(&python_evaluation_path)?;
    let corpus_git_sha = env::var("SPLADE_CORPUS_GIT_SHA")
        .context("SPLADE_CORPUS_GIT_SHA was not supplied by the capped wrapper")?;
    ensure!(
        corpus_git_sha == evaluation_manifest.corpus_git_sha,
        "corpus Git SHA differs from Python evaluation"
    );
    ensure!(
        sha256_file(query_path)? == evaluation_manifest.queries_sha256,
        "query file SHA-256 differs from Python evaluation"
    );
    ensure!(
        expected.key == export_manifest.model,
        "model key provenance differs"
    );
    ensure!(
        expected.spec.revision == export_manifest.revision,
        "model revision provenance differs"
    );

    let documents = load_documents(Path::new(&corpus_root))?;
    verify_corpus(&documents, &corpus_manifest_path, &evaluation_manifest)?;
    let queries = load_queries(query_path)?;
    ensure!(
        queries.len() == evaluation_manifest.queries,
        "query count differs from Python evaluation"
    );

    let load_started = Instant::now();
    let document_encoder =
        DocumentEncoder::from_dir_with_manifest(model_dir, &export_manifest_path)?;
    let query_encoder = QueryEncoder::from_dir_with_manifest(model_dir, &export_manifest_path)?;
    let model_load_seconds = load_started.elapsed().as_secs_f64();

    let document_started = Instant::now();
    let mut encoded_documents = Vec::with_capacity(documents.len());
    for batch in documents.chunks(DOCUMENT_BATCH_SIZE) {
        let texts = batch
            .iter()
            .map(|document| document.text.as_str())
            .collect::<Vec<_>>();
        let vectors = document_encoder.encode_batch(&texts)?;
        encoded_documents.extend(
            batch
                .iter()
                .zip(vectors)
                .map(|(document, vector)| (document.id.clone(), vector)),
        );
    }
    let document_encode_seconds = document_started.elapsed().as_secs_f64();
    let average_document_terms = encoded_documents
        .iter()
        .map(|(_, vector)| vector.active_terms())
        .sum::<usize>() as f64
        / encoded_documents.len() as f64;
    let index = SparseIndex::build(encoded_documents)?;
    let serialized_index_bytes = index.to_bytes().len();

    let mut query_latencies = Vec::with_capacity(queries.len());
    let mut query_terms = 0usize;
    let mut exact_queries = 0usize;
    let mut position_matches = 0usize;
    let mut set_matches = 0usize;
    let mut max_score_abs_error = 0.0f64;
    for query in &queries {
        let started = Instant::now();
        let search_text = query.retrieval_query.as_deref().unwrap_or(&query.query);
        let vector = query_encoder.encode(search_text)?;
        query_terms += vector.active_terms();
        let actual = index.search(&vector, RANKING_LIMIT);
        query_latencies.push(started.elapsed().as_secs_f64() * 1000.0);
        let expected = expected
            .rankings
            .get(&query.id)
            .with_context(|| format!("missing Python ranking for {}", query.id))?;
        let actual_top10 = actual
            .iter()
            .take(10)
            .map(|row| row.0.as_str())
            .collect::<Vec<_>>();
        let expected_top10 = expected
            .iter()
            .take(10)
            .map(|row| row.document_id.as_str())
            .collect::<Vec<_>>();
        if actual_top10 == expected_top10 {
            exact_queries += 1;
        }
        position_matches += actual_top10
            .iter()
            .zip(&expected_top10)
            .filter(|(actual, expected)| actual == expected)
            .count();
        let expected_set = expected_top10.iter().copied().collect::<HashSet<_>>();
        set_matches += actual_top10
            .iter()
            .filter(|document_id| expected_set.contains(**document_id))
            .count();
        for ((actual_id, actual_score), expected_row) in actual.iter().zip(expected) {
            if actual_id == &expected_row.document_id {
                max_score_abs_error =
                    max_score_abs_error.max((f64::from(*actual_score) - expected_row.score).abs());
            }
        }
    }
    query_latencies.sort_by(f64::total_cmp);
    let compared_slots = queries.len() * 10;
    let corpus_bytes = documents.iter().map(|document| document.bytes).sum();
    let runtime_model_bytes = export_manifest
        .files
        .values()
        .map(|record| record.bytes)
        .sum();
    let result = RustPortEvaluation {
        command: BENCHMARK_COMMAND,
        build_profile: "release",
        rustc: rustc_version()?,
        container_limits: container_limits()?,
        inputs: InputEvidence {
            corpus_git_sha,
            queries_sha256: sha256_file(query_path)?,
            corpus_manifest_sha256: sha256_file(&corpus_manifest_path)?,
            python_evaluation_sha256: sha256_file(&python_evaluation_path)?,
            export_manifest_sha256: sha256_file(&export_manifest_path)?,
            cargo_lock_sha256: sha256_file(Path::new("Cargo.lock"))?,
        },
        model: export_manifest.model,
        revision: export_manifest.revision,
        model_files: export_manifest.files,
        corpus_documents: documents.len(),
        corpus_bytes,
        evaluated_queries: queries.len(),
        top10_exact_queries: exact_queries,
        top10_position_agreement: position_matches as f64 / compared_slots as f64,
        top10_set_recall: set_matches as f64 / compared_slots as f64,
        max_score_abs_error,
        model_load_seconds,
        document_encode_seconds,
        documents_per_second: documents.len() as f64 / document_encode_seconds,
        query_and_search_p50_ms: percentile(&query_latencies, 50),
        query_and_search_p95_ms: percentile(&query_latencies, 95),
        average_query_terms: query_terms as f64 / queries.len() as f64,
        average_document_terms,
        serialized_index_bytes,
        projected_serialized_index_mib_10k: serialized_index_bytes as f64 / documents.len() as f64
            * 10_000.0
            / (1024.0 * 1024.0),
        runtime_model_bytes,
        peak_rss_mib: peak_rss_mib().unwrap_or(0.0),
    };
    ensure!(
        exact_queries == queries.len(),
        "Rust top-10 rankings differ from Python on {}/{} queries",
        queries.len() - exact_queries,
        queries.len()
    );
    let output_path = Path::new(&output_path);
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(output_path, serde_json::to_string_pretty(&result)? + "\n")?;
    let mut commands = OpenOptions::new()
        .create(true)
        .append(true)
        .open(Path::new(RUN_DIR).join("commands.log"))?;
    writeln!(commands, "{BENCHMARK_COMMAND}")?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}

fn load_documents(root: &Path) -> Result<Vec<Document>> {
    let mut paths = vec![root.join("README.md"), root.join("ARCHITECTURE.md")];
    collect_markdown(&root.join("docs"), &mut paths)?;
    paths.sort();
    paths.dedup();
    let mut documents = Vec::new();
    for path in paths {
        if !path.is_file() {
            continue;
        }
        let id = path
            .strip_prefix(root)?
            .to_string_lossy()
            .replace('\\', "/");
        if id
            .split('/')
            .any(|part| part == "_archive" || part == "prompts")
        {
            continue;
        }
        let raw = fs::read(&path)?;
        documents.push(Document {
            id,
            text: String::from_utf8(raw.clone())?,
            bytes: u64::try_from(raw.len())?,
            sha256: sha256_bytes(&raw),
        });
    }
    Ok(documents)
}

fn verify_corpus(
    documents: &[Document],
    manifest_path: &Path,
    evaluation: &EvaluationManifest,
) -> Result<()> {
    ensure!(
        documents.len() == evaluation.corpus_documents,
        "corpus document count differs from Python evaluation"
    );
    ensure!(
        documents.iter().map(|document| document.bytes).sum::<u64>() == evaluation.corpus_bytes,
        "corpus bytes differ from Python evaluation"
    );
    let expected = read_json::<Vec<CorpusRecord>>(manifest_path)?
        .into_iter()
        .map(|record| (record.id.clone(), record))
        .collect::<HashMap<_, _>>();
    ensure!(
        expected.len() == documents.len(),
        "corpus manifest length differs"
    );
    for document in documents {
        let record = expected
            .get(&document.id)
            .with_context(|| format!("corpus manifest is missing {}", document.id))?;
        ensure!(record.bytes == document.bytes, "corpus byte size differs");
        ensure!(record.sha256 == document.sha256, "corpus SHA-256 differs");
    }
    Ok(())
}

fn collect_markdown(directory: &Path, paths: &mut Vec<PathBuf>) -> Result<()> {
    if !directory.is_dir() {
        return Ok(());
    }
    for entry in fs::read_dir(directory)? {
        let path = entry?.path();
        if path.is_dir() {
            collect_markdown(&path, paths)?;
        } else if path.extension().and_then(|extension| extension.to_str()) == Some("md") {
            paths.push(path);
        }
    }
    Ok(())
}

fn load_queries(path: &Path) -> Result<Vec<Query>> {
    fs::read_to_string(path)?
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line).map_err(Into::into))
        .collect()
}

fn percentile(sorted: &[f64], percentage: usize) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let rank = (sorted.len() * percentage).div_ceil(100).max(1);
    sorted[rank - 1]
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    serde_json::from_str(&fs::read_to_string(path)?).map_err(Into::into)
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn rustc_version() -> Result<String> {
    let output = Command::new("rustc").arg("--version").output()?;
    ensure!(output.status.success(), "rustc --version failed");
    Ok(String::from_utf8(output.stdout)?.trim().to_owned())
}

fn container_limits() -> Result<ContainerLimits> {
    Ok(ContainerLimits {
        memory_max: read_trimmed("/sys/fs/cgroup/memory.max")?,
        cpu_max: read_trimmed("/sys/fs/cgroup/cpu.max")?,
        pids_max: read_trimmed("/sys/fs/cgroup/pids.max")?,
    })
}

fn read_trimmed(path: &str) -> Result<String> {
    Ok(fs::read_to_string(path)?.trim().to_owned())
}

fn peak_rss_mib() -> Option<f64> {
    let status = fs::read_to_string("/proc/self/status").ok()?;
    let kilobytes = status
        .lines()
        .find_map(|line| line.strip_prefix("VmHWM:"))?
        .split_whitespace()
        .next()?
        .parse::<f64>()
        .ok()?;
    Some(kilobytes / 1024.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentile_matches_python_nearest_rank_definition() {
        let values = (1..=60).map(f64::from).collect::<Vec<_>>();
        assert_eq!(percentile(&values, 50), 30.0);
        assert_eq!(percentile(&values, 95), 57.0);
        assert_eq!(percentile(&[], 95), 0.0);
    }
}
