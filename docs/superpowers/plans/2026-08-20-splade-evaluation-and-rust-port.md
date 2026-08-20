# SPLADE Evaluation and Rust Port Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compare three SPLADE models on real `pentesting` documents, select one by fixed gates, and port only the winner to Rust.

**Architecture:** A capped Docker Python harness builds a read-only corpus, evaluates BM25 and three sparse models, and emits raw metrics plus SVG reports. A small Rust crate then reproduces the winning model's sparse vectors and search scores.

**Tech Stack:** Python 3.12, Sentence Transformers SparseEncoder, PyTorch CPU, pytest, Matplotlib, Rust 1.96, tract-onnx, tokenizers, Docker.

**Spec:** `docs/superpowers/specs/2026-08-20-splade-rust-poc-design.md`

**Status:** Completed on 2026-08-20. Raw results and verification evidence are under `artifacts/eval/2026-08-20-pentesting-267`.

## Global Constraints

- Run Python and Cargo only in Docker with `--memory 4g --memory-swap 4g --cpus 2 --pids-limit 512`.
- Mount `C:\workspace\pentesting` read-only.
- Compare exactly the three models named in the spec before choosing a winner.
- Do not commit model weights, caches, source corpus text, or secrets.
- Persist raw measurements, environment metadata, commands, and generated SVGs.
- Port no model to Rust until `decision.json` names a passing winner.
- Do not modify `pentesting` during this plan.

---

### Task 1: Reproducible Docker Harness

**Files:**
- Create: `.gitignore`
- Create: `pyproject.toml`
- Create: `scripts/poc.ps1`
- Create: `tests/test_harness_contract.py`

**Interfaces:**
- Produces: `scripts/poc.ps1 <pytest|lock|evaluate|report|parity>`.

- [ ] Write a failing contract test that checks Docker memory, CPU, swap, PID, read-only corpus mount, and cache exclusions.
- [ ] Run `scripts/poc.ps1 pytest tests/test_harness_contract.py` and confirm failure because the harness is absent.
- [ ] Implement the capped `ghcr.io/astral-sh/uv:python3.12-bookworm-slim` runner and dependency manifest.
- [ ] Re-run the contract test and commit the passing harness.

### Task 2: Corpus and Relevance Judgments

**Files:**
- Create: `src/splade_poc/corpus.py`
- Create: `benchmarks/queries.jsonl`
- Create: `tests/test_corpus.py`
- Create: `docs/research/model-selection.md`

**Interfaces:**
- Produces: `load_corpus(root: Path) -> list[Document]` and 60 judged queries split deterministically 60/40.

- [ ] Write failing tests for stable IDs, Markdown-only inclusion, excluded generated/private paths, and absence of source text in saved benchmark artifacts.
- [ ] Run the focused tests and confirm the missing corpus API failure.
- [ ] Implement read-only Markdown loading and 60 concise queries covering exact, semantic, Korean-to-English, relational, and no-answer cases.
- [ ] Run tests, inspect every target ID against the source snapshot, and record corpus selection reasons.

### Task 3: BM25, Fusion, and Metrics

**Files:**
- Create: `src/splade_poc/lexical.py`
- Create: `src/splade_poc/fusion.py`
- Create: `src/splade_poc/metrics.py`
- Create: `tests/test_lexical.py`
- Create: `tests/test_metrics.py`

**Interfaces:**
- Produces: `rank_bm25`, `reciprocal_rank_fusion`, and `evaluate_run`.

- [ ] Write failing examples for identifier expansion, current BM25-like ranking, deterministic RRF ties, Recall, MRR, and nDCG.
- [ ] Run focused tests and verify expected assertion failures.
- [ ] Implement only the tested ranking and metric behavior.
- [ ] Run focused tests and the full Python suite.

### Task 4: Sparse Model Adapters

**Files:**
- Create: `src/splade_poc/models.py`
- Create: `src/splade_poc/sparse.py`
- Create: `tests/test_sparse.py`
- Create: `tests/test_models.py`

**Interfaces:**
- Produces: `SparseVector(term_ids, weights)`, `ModelSpec`, `encode_queries`, and `encode_documents`.

- [ ] Write failing tests for top-k pruning, stable tie ordering, finite non-negative weights, `u16` limits, and the exact three-model registry.
- [ ] Run focused tests and confirm missing API failures.
- [ ] Implement the pure sparse utilities and Sentence Transformers adapters.
- [ ] Run unit tests, then one-document smoke inference for each pinned model and capture revisions and SHA-256 values.

### Task 5: Evaluation and Human-Readable Reports

**Files:**
- Create: `src/splade_poc/evaluate.py`
- Create: `src/splade_poc/report.py`
- Create: `tests/test_report.py`
- Create: `docs/research/evaluation.md`

**Interfaces:**
- Produces: `artifacts/eval/<run-id>/{manifest,environment,metrics,per-query,decision}.json*`, three SVGs, and `report.md`.

- [ ] Write failing tests using a fixed miniature result set for gate decisions, report tables, and SVG existence.
- [ ] Run focused tests and confirm failure because reporting is absent.
- [ ] Implement timed batched evaluation, peak-RSS sampling, gate evaluation, decision generation, and graphs sourced only from raw JSON.
- [ ] Run unit tests and a miniature end-to-end fixture evaluation.

### Task 6: Full Three-Model Experiment

**Files:**
- Generate: `artifacts/eval/<run-id>/*`
- Update: `docs/research/evaluation.md`
- Create: `docs/research/integration-decision.md`

**Interfaces:**
- Consumes: fixed corpus snapshot, 60 qrels, and three pinned model specs.
- Produces: immutable `decision.json` with a winner or `no_winner`.

- [ ] Record the corpus Git SHA, environment, package lock, model revisions, and commands.
- [ ] Run all three models on the same corpus and query split under the capped container.
- [ ] Regenerate tables and SVGs from raw output and verify their values against `metrics.json`.
- [ ] Apply the gates without manual overrides and document the decision and limitations concisely.

### Task 7: Rust Port of the Winner

**Files:**
- Create: `Cargo.toml`
- Create: `src/lib.rs`
- Create: `src/vector.rs`
- Create: `src/index.rs`
- Create: `src/encoder.rs`
- Create: `tests/vector.rs`
- Create: `tests/index.rs`
- Create: `tests/parity.rs`
- Create: `scripts/rust-test.ps1`
- Create: `docs/research/rust-port.md`

**Interfaces:**
- Produces: `SparseVector`, `SparseIndex`, and `DocumentEncoder` for the selected model only.

- [ ] Abort this task when `decision.json` is `no_winner`; record that Rust was correctly not started.
- [ ] For a winner, write failing Rust tests from committed non-sensitive Python parity fixtures.
- [ ] Run the focused test in capped Docker and confirm the expected missing implementation failure.
- [ ] Implement sparse vector validation, inverted-index dot product, tokenizer, ONNX inference, pooling, and top-k pruning.
- [ ] Run focused and full Rust tests in capped Docker and record peak resources.

### Task 8: Final Parity, Audit, and Publication

**Files:**
- Update: `README.md`
- Update: `docs/research/rust-port.md`
- Update: `docs/research/integration-decision.md`

**Interfaces:**
- Produces: evidence for or against adding SPLADE as the fourth `pentesting` retrieval layer.

- [ ] Re-run Python tests, full evaluation report generation, Rust tests when applicable, and `git diff --check`.
- [ ] Confirm Python/Rust top-256 IDs and `1e-4` weight tolerance when a model was ported.
- [ ] Audit every spec gate against raw artifacts and list any unmet gate without softening it.
- [ ] Commit and push code, concise research notes, raw metrics, and SVGs; exclude caches, weights, and corpus text.
