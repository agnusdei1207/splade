from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch
from torch import Tensor, nn

from .models import MODEL_SPECS, ModelSpec, SparseModel
from .sparse import SparseVector


WINNER_KEY = "if-opensearch-mini"
PARITY_DOCUMENTS = (
    "The terminal listener upgrades a shell to a pseudo terminal and verifies the prompt.",
    "Docker resource limits protect the host from compiler memory exhaustion.",
    "Evidence-backed mission completion remains separate from a model stop event.",
)
PARITY_QUERIES = (
    "PTY shell upgrade",
    "Docker memory limit",
    "verified mission completion",
)


class SpladeDocumentWrapper(nn.Module):
    def __init__(self, model: nn.Module, top_k: int = 256) -> None:
        super().__init__()
        self.model = model
        self.top_k = top_k

    def forward(
        self, input_ids: Tensor, attention_mask: Tensor, token_type_ids: Tensor
    ) -> tuple[Tensor, Tensor]:
        logits = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )[0]
        masked = logits * attention_mask.unsqueeze(-1).to(logits.dtype)
        scores = masked.relu().log1p().amax(dim=1)
        return scores.topk(self.top_k, dim=1, largest=True, sorted=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _winner_spec(decision_path: Path) -> ModelSpec:
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    winner = decision.get("winner")
    if winner != WINNER_KEY:
        raise ValueError(f"expected evaluated winner {WINNER_KEY!r}, got {winner!r}")
    return next(spec for spec in MODEL_SPECS if spec.key == winner)


def _from_topk(values: Tensor, indices: Tensor, limit: int) -> SparseVector:
    return SparseVector.from_pairs(
        (
            (int(term_id), float(weight))
            for term_id, weight in zip(indices, values, strict=True)
        ),
        limit,
    )


def _assert_document_parity(
    sparse_model: SparseModel,
    wrapper: SpladeDocumentWrapper,
    tokenizer: object,
) -> None:
    expected = sparse_model.encode_documents(list(PARITY_DOCUMENTS), batch_size=3)
    encoded = tokenizer(
        list(PARITY_DOCUMENTS),
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    token_type_ids = encoded.get(
        "token_type_ids", torch.zeros_like(encoded["input_ids"])
    )
    with torch.inference_mode():
        values, indices = wrapper(
            encoded["input_ids"], encoded["attention_mask"], token_type_ids
        )
    actual = [
        _from_topk(row_values, row_indices, sparse_model.spec.max_document_terms)
        for row_values, row_indices in zip(values, indices, strict=True)
    ]
    for row, (wanted, observed) in enumerate(zip(expected, actual, strict=True)):
        if wanted.term_ids != observed.term_ids:
            raise AssertionError(f"document wrapper term ids differ at row {row}")
        for term_id, wanted_weight, observed_weight in zip(
            wanted.term_ids, wanted.weights, observed.weights, strict=True
        ):
            if abs(wanted_weight - observed_weight) > 1e-5:
                raise AssertionError(
                    f"document wrapper weight differs at row {row}, term {term_id}: "
                    f"{wanted_weight} != {observed_weight}"
                )


def _write_parity_fixture(sparse_model: SparseModel, fixture_path: Path) -> None:
    documents = sparse_model.encode_documents(list(PARITY_DOCUMENTS), batch_size=3)
    queries = sparse_model.encode_queries(list(PARITY_QUERIES), batch_size=3)

    def record(text: str, vector: SparseVector) -> dict:
        return {
            "text": text,
            "term_ids": list(vector.term_ids),
            "weights": [round(weight, 8) for weight in vector.weights],
        }

    payload = {
        "model": sparse_model.spec.key,
        "revision": sparse_model.spec.revision,
        "weight_tolerance": 1e-4,
        "documents": [
            record(text, vector)
            for text, vector in zip(PARITY_DOCUMENTS, documents, strict=True)
        ],
        "queries": [
            record(text, vector)
            for text, vector in zip(PARITY_QUERIES, queries, strict=True)
        ],
    }
    fixture_path.parent.mkdir(parents=True, exist_ok=True)
    fixture_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def export_winner(
    decision_path: Path, output_dir: Path, fixture_path: Path, record_path: Path
) -> dict:
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    spec = _winner_spec(decision_path)
    sparse_model = SparseModel(spec)
    snapshot = sparse_model.snapshot
    model = AutoModelForMaskedLM.from_pretrained(
        snapshot, local_files_only=True, use_safetensors=True
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    wrapper = SpladeDocumentWrapper(model, top_k=spec.max_document_terms).eval()
    _assert_document_parity(sparse_model, wrapper, tokenizer)

    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_name in (
        "document-tokenizer.json",
        "document-tokenizer-config.json",
        "query-tokenizer.json",
        "query-tokenizer-config.json",
    ):
        (output_dir / stale_name).unlink(missing_ok=True)
    sample = tokenizer(
        [PARITY_DOCUMENTS[0]],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    token_type_ids = sample.get(
        "token_type_ids", torch.zeros_like(sample["input_ids"])
    )
    onnx_path = output_dir / "document.onnx"
    torch.onnx.export(
        wrapper,
        (sample["input_ids"], sample["attention_mask"], token_type_ids),
        onnx_path,
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["values", "term_ids"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "token_type_ids": {0: "batch", 1: "sequence"},
            "values": {0: "batch"},
            "term_ids": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )

    document_tokenizer = snapshot / "tokenizer.json"
    query_tokenizer = snapshot / "query_0_SparseStaticEmbedding" / "tokenizer.json"
    if _sha256(document_tokenizer) != _sha256(query_tokenizer):
        raise AssertionError("document and query tokenizers differ")
    copies = {
        "tokenizer.json": document_tokenizer,
        "query.safetensors": snapshot
        / "query_0_SparseStaticEmbedding"
        / "model.safetensors",
    }
    for destination, source in copies.items():
        shutil.copyfile(source, output_dir / destination)

    _write_parity_fixture(sparse_model, fixture_path)
    files = {}
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "manifest.json":
            files[path.name] = {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
    manifest = {
        "model": spec.key,
        "hf_id": spec.hf_id,
        "revision": spec.revision,
        "license": spec.license,
        "vocab_size": spec.vocab_size,
        "document_top_k": spec.max_document_terms,
        "query_top_k": spec.max_query_terms,
        "onnx_opset": 17,
        "files": files,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the evaluated winner for Rust")
    parser.add_argument(
        "--decision",
        type=Path,
        default=Path("artifacts/eval/2026-08-20-pentesting-267/decision.json"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("models/if-opensearch-mini")
    )
    parser.add_argument(
        "--fixture", type=Path, default=Path("fixtures/python-parity.json")
    )
    parser.add_argument(
        "--record",
        type=Path,
        default=Path(
            "artifacts/eval/2026-08-20-pentesting-267/rust-export-manifest.json"
        ),
    )
    args = parser.parse_args()
    manifest = export_winner(args.decision, args.output, args.fixture, args.record)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
