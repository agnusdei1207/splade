from __future__ import annotations

import argparse
import json
from pathlib import Path


def ensure_parity(rust: dict) -> None:
    if rust["top10_exact_queries"] != rust["evaluated_queries"]:
        raise ValueError(
            "Rust parity failed: "
            f"{rust['top10_exact_queries']}/{rust['evaluated_queries']} exact queries"
        )


def comparison_data(python: dict, rust: dict) -> dict[str, tuple[float, float]]:
    resources = python["resources"]
    return {
        "Document throughput (docs/s)": (
            resources["documents_per_second"],
            rust["documents_per_second"],
        ),
        "Query + search p95 (ms)": (
            resources["query_p95_ms"],
            rust["query_and_search_p95_ms"],
        ),
        "Peak RSS (MiB)": (resources["peak_rss_mib"], rust["peak_rss_mib"]),
    }


def write_port_chart(run_dir: Path) -> None:
    import matplotlib.pyplot as plt

    python = json.loads((run_dir / "if-opensearch-mini.json").read_text(encoding="utf-8"))
    rust = json.loads((run_dir / "rust-port.json").read_text(encoding="utf-8"))
    ensure_parity(rust)
    comparisons = comparison_data(python, rust)

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    colors = ("#4C78A8", "#F58518")
    for axis, (title, values) in zip(axes, comparisons.items(), strict=True):
        bars = axis.bar(("Python\nPyTorch", "Rust\ntract"), values, color=colors)
        axis.set_title(title, fontsize=11, weight="bold")
        axis.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
        axis.set_ylim(0, max(values) * 1.18)
        axis.grid(axis="x", visible=False)
    figure.suptitle("Selected SPLADE model: Python vs Rust", fontsize=14, weight="bold")
    figure.tight_layout()
    for suffix in ("svg", "png"):
        figure.savefig(run_dir / f"rust-port.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(figure)

    (run_dir / "rust-port-report.md").write_text(
        "\n".join(
            (
                "# Rust port result",
                "",
                "| Check | Result |",
                "|---|---:|",
                f"| Python top-10 exact queries | {rust['top10_exact_queries']}/{rust['evaluated_queries']} |",
                f"| Maximum score error | {rust['max_score_abs_error']:.8f} |",
                f"| Document throughput | {rust['documents_per_second']:.3f} docs/s |",
                f"| Query + search p95 | {rust['query_and_search_p95_ms']:.3f} ms |",
                f"| Peak RSS | {rust['peak_rss_mib']:.1f} MiB |",
                f"| Runtime model files | {rust['runtime_model_bytes'] / (1024 * 1024):.2f} MiB |",
                f"| Serialized index / 10k docs | {rust['projected_serialized_index_mib_10k']:.2f} MiB |",
                "",
                "Decision: parity passed. Use Rust query encoding online; run document encoding only during indexing.",
                "",
            )
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Render Python/Rust port comparison")
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    write_port_chart(args.run_dir)


if __name__ == "__main__":
    main()
