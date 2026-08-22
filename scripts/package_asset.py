"""Assemble the sparse encoder into the asset layout pentesting already uses for dense.

`builder_workspace_index/assets/snowflake-arctic-embed-xs` ships a quantized ONNX,
its tokenizer, a MANIFEST.json with SHA-256 per file, plus the model card and
licence. Matching that layout means the new encoder inherits the same integrity
check and the same review surface rather than inventing a second convention.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="opensearch-project/opensearch-neural-sparse-encoding-doc-v2-mini")
    parser.add_argument("--revision", default="4af867a426867dfdd744097531046f4289a32fdd")
    parser.add_argument("--onnx", type=Path, default=Path("artifacts/quant/mini/document-int8.onnx"))
    parser.add_argument("--document-top-k", type=int, default=512)
    parser.add_argument("--query-top-k", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    snapshot = Path(snapshot_download(repo_id=args.model, revision=args.revision, max_workers=2))
    args.output.mkdir(parents=True, exist_ok=True)

    document_tokenizer = snapshot / "tokenizer.json"
    query_tokenizer = snapshot / "query_0_SparseStaticEmbedding" / "tokenizer.json"
    if query_tokenizer.is_file() and sha256(document_tokenizer) != sha256(query_tokenizer):
        raise SystemExit("document and query tokenizers differ; ship both or reconcile")

    copies = {
        "document_quantized.onnx": args.onnx,
        "tokenizer.json": document_tokenizer,
        "query_weights.safetensors": snapshot / "query_0_SparseStaticEmbedding" / "model.safetensors",
    }
    for name in ("README.md", "LICENSE"):
        source = snapshot / name
        if source.is_file():
            copies["MODEL_CARD.md" if name == "README.md" else "LICENSE-APACHE-2.0.txt"] = source

    for destination, source in copies.items():
        if not source.is_file():
            raise SystemExit(f"missing source asset: {source}")
        shutil.copyfile(source, args.output / destination)

    import json as _json

    config = _json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    manifest = {
        "model_id": args.model,
        "revision": args.revision,
        "license": "Apache-2.0",
        "quantization": "int8 dynamic (onnxruntime QInt8, MatMulConstBOnly)",
        "vocab_size": config["vocab_size"],
        "max_sequence_length": 512,
        "pooling": "SPLADE max over relu(log1p(masked logits))",
        "document_top_k": args.document_top_k,
        "query_top_k": args.query_top_k,
        "query_is_inference_free": True,
        "files": {
            name: {"bytes": (args.output / name).stat().st_size, "sha256": sha256(args.output / name)}
            for name in sorted(copies)
        },
    }
    (args.output / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    total = sum(record["bytes"] for record in manifest["files"].values())
    print(json.dumps(manifest, indent=2))
    print(f"\n총 배포 크기: {total / 1048576:.2f} MiB")


if __name__ == "__main__":
    main()
