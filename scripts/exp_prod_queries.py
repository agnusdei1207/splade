"""Encode the production-shaped query set only, reusing the document vectors already on disk.

The earlier query families were derived from document text, which favours a lexical
scorer and cannot settle whether SPLADE earns its place. These queries are written
the way a developer actually asks, then normalised with the exact system prompt
`retrieval_intent.rs` uses, so both the production path (normalised English) and the
fallback path (raw text, when the LLM call times out) can be measured.

Documents are untouched, so only the queries need an encoder pass.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp_models import StEncoder, RawSpladeEncoder, load_encoder  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--queries", type=Path, default=Path("benchmarks/production-queries.jsonl"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--query-cap", type=int, default=128)
    args = parser.parse_args()

    threads = int(os.environ.get("SPLADE_EXP_CPUS", "2"))
    torch.manual_seed(0)
    torch.set_num_threads(threads)

    rows = [
        json.loads(line)
        for line in args.queries.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    encoder, snapshot, packaging = load_encoder(args.model, args.revision)

    def cap(encoded):
        out = []
        for term_ids, weights in encoded:
            pairs = sorted(
                ((t, w) for t, w in zip(term_ids, weights) if w > 0),
                key=lambda p: (-p[1], p[0]),
            )[: args.query_cap]
            pairs.sort()
            out.append(([t for t, _ in pairs], [round(w, 6) for _, w in pairs]))
        return out

    def encode(texts: list[str]):
        started = time.perf_counter()
        if packaging == "sentence-transformers":
            encoded = encoder.encode(texts, 32, "query")
        else:
            encoded = encoder.encode(texts, 32)
        return cap(encoded), time.perf_counter() - started

    forms = {}
    timings = {}
    for form in ("raw", "norm"):
        encoded, seconds = encode([r[form] for r in rows])
        forms[form] = encoded
        timings[form] = seconds

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for form, encoded in forms.items():
        (args.output_dir / f"query-vectors-{form}.json").write_text(
            json.dumps(
                [
                    {"id": row["id"], "term_ids": t, "weights": w}
                    for row, (t, w) in zip(rows, encoded, strict=True)
                ],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    summary = {
        "model": args.model,
        "packaging": packaging,
        "queries": len(rows),
        "threads": threads,
        "encode_seconds": timings,
        "active_dims": {
            form: {
                "mean": sum(len(t) for t, _ in encoded) / len(encoded),
                "zero": sum(1 for t, _ in encoded if not t),
            }
            for form, encoded in forms.items()
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
