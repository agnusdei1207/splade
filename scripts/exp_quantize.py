"""Export a sparse document encoder to ONNX, quantize it, and measure the real cost.

Binary size is the binding constraint. pentesting already embeds its dense encoder
with `include_bytes!` at 22.97 MB, and that file is int8 quantized - so the question
is not "how big is the fp32 checkpoint" but "how big is the same treatment applied
to a sparse encoder, and what does it cost in quality and speed".

Measures, for each candidate: fp32 ONNX bytes, int8 ONNX bytes, encode throughput
for both, and the sparse vectors themselves so ranking quality can be compared.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import time
from pathlib import Path

import torch
from torch import Tensor, nn


class SpladeDocumentWrapper(nn.Module):
    """MLM logits -> SPLADE max pooling -> top-k, all inside the graph."""

    def __init__(self, model: nn.Module, top_k: int) -> None:
        super().__init__()
        self.model = model
        self.top_k = top_k

    def forward(self, input_ids: Tensor, attention_mask: Tensor, token_type_ids: Tensor):
        logits = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )[0]
        masked = logits * attention_mask.unsqueeze(-1).to(logits.dtype)
        scores = masked.relu().log1p().amax(dim=1)
        return scores.topk(self.top_k, dim=1, largest=True, sorted=True)


def export_onnx(snapshot: Path, out: Path, top_k: int, sample_text: str) -> None:
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    model = AutoModelForMaskedLM.from_pretrained(
        snapshot, local_files_only=True, use_safetensors=True
    ).eval()
    wrapper = SpladeDocumentWrapper(model, top_k).eval()
    encoded = tokenizer(
        [sample_text], padding=True, truncation=True, max_length=512, return_tensors="pt"
    )
    type_ids = encoded.get("token_type_ids", torch.zeros_like(encoded["input_ids"]))
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (encoded["input_ids"], encoded["attention_mask"], type_ids),
        out,
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


def quantize(src: Path, dst: Path) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(
        model_input=str(src),
        model_output=str(dst),
        weight_type=QuantType.QInt8,
        extra_options={"MatMulConstBOnly": True},
    )


def run_onnx(path: Path, tokenizer, texts: list[str], threads: int, batch_size: int):
    import numpy as np
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    wanted = {i.name for i in session.get_inputs()}
    rows = []
    started = time.perf_counter()
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch, padding=True, truncation=True, max_length=512, return_tensors="np"
        )
        feed = {"input_ids": encoded["input_ids"].astype("int64")}
        if "attention_mask" in wanted:
            feed["attention_mask"] = encoded["attention_mask"].astype("int64")
        if "token_type_ids" in wanted:
            feed["token_type_ids"] = encoded.get(
                "token_type_ids", np.zeros_like(encoded["input_ids"])
            ).astype("int64")
        values, term_ids = session.run(None, feed)
        for row in range(values.shape[0]):
            keep = values[row] > 0
            pairs = sorted(zip(term_ids[row][keep].tolist(), values[row][keep].tolist()))
            rows.append(([t for t, _ in pairs], [round(float(w), 6) for _, w in pairs]))
    return rows, time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--label", required=True)
    parser.add_argument("--input-dir", type=Path, default=Path("artifacts/families"))
    parser.add_argument("--top-k", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    threads = int(os.environ.get("SPLADE_EXP_CPUS", "2"))
    torch.manual_seed(0)
    torch.set_num_threads(threads)

    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    snapshot = Path(snapshot_download(repo_id=args.model, revision=args.revision, max_workers=2))
    documents = json.loads((args.input_dir / "documents.json").read_text(encoding="utf-8"))
    texts = [d["text"] for d in documents]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fp32 = args.output_dir / "document-fp32.onnx"
    int8 = args.output_dir / "document-int8.onnx"

    print(f"[{args.label}] ONNX export ...", flush=True)
    export_onnx(snapshot, fp32, args.top_k, texts[0][:2000])
    print(f"[{args.label}] quantize ...", flush=True)
    quantize(fp32, int8)

    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    results = {}
    for name, path in (("fp32", fp32), ("int8", int8)):
        rows, seconds = run_onnx(path, tokenizer, texts, threads, args.batch_size)
        results[name] = {
            "bytes": path.stat().st_size,
            "encode_seconds": seconds,
            "documents_per_second": len(texts) / seconds,
            "mean_active_dims": sum(len(t) for t, _ in rows) / len(rows),
        }
        (args.output_dir / f"document-vectors-{name}.json").write_text(
            json.dumps(
                [
                    {"id": d["id"], "term_ids": t, "weights": w}
                    for d, (t, w) in zip(documents, rows, strict=True)
                ],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"[{args.label}] {name}: {path.stat().st_size / 1048576:.1f} MiB, "
              f"{len(texts) / seconds:.2f} docs/s", flush=True)

    # the query side is a static weight table, tiny but it still ships
    query_weights = snapshot / "query_0_SparseStaticEmbedding" / "model.safetensors"
    tokenizer_json = snapshot / "tokenizer.json"
    summary = {
        "label": args.label,
        "model": args.model,
        "top_k": args.top_k,
        "threads": threads,
        "batch_size": args.batch_size,
        "documents": len(documents),
        "fp32": results["fp32"],
        "int8": results["int8"],
        "size_ratio_int8_over_fp32": results["int8"]["bytes"] / results["fp32"]["bytes"],
        "speedup_int8_over_fp32": results["int8"]["documents_per_second"] / results["fp32"]["documents_per_second"],
        "query_weight_bytes": query_weights.stat().st_size if query_weights.is_file() else None,
        "tokenizer_bytes": tokenizer_json.stat().st_size if tokenizer_json.is_file() else None,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    shippable = results["int8"]["bytes"] + (summary["query_weight_bytes"] or 0) + (summary["tokenizer_bytes"] or 0)
    summary["shippable_int8_bytes"] = shippable
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
