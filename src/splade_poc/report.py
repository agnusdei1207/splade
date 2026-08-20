from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ALLOWED_LICENSES = {"apache-2.0", "mit"}


def choose_winner(summary: dict) -> dict:
    baseline = summary["bm25"]["selection"]
    baseline_exact = baseline.get("by_category", {}).get("exact", {}).get("recall@10", 0.0)
    decisions: dict[str, dict] = {}
    for key, result in sorted(summary["models"].items()):
        fused = result["selection"]["fused"]
        resources = result["resources"]
        reasons: list[str] = []
        if fused["recall@10"] < baseline["recall@10"]:
            reasons.append("recall<bm25")
        exact = fused.get("by_category", {}).get("exact", {}).get("recall@10", 0.0)
        if exact < baseline_exact:
            reasons.append("exact-recall<bm25")
        if resources["projected_index_mib_10k"] > 32.0:
            reasons.append("index>32MiB/10k")
        if result["spec"]["inference_free_query"] and resources["query_p95_ms"] > 10.0:
            reasons.append("query-p95>10ms")
        if result["license"].lower() not in ALLOWED_LICENSES:
            reasons.append("license")
        decisions[key] = {"passed": not reasons, "reasons": reasons}
    passed = [key for key, value in decisions.items() if value["passed"]]
    winner = max(
        passed,
        key=lambda key: (summary["models"][key]["selection"]["fused"]["ndcg@10"], key),
        default=None,
    )
    return {"winner": winner or "no_winner", "models": decisions}


def _save_quality(summary: dict, output: Path) -> None:
    labels = ["BM25"] + list(summary["models"])
    recall = [summary["bm25"]["validation"]["recall@10"]]
    ndcg = [summary["bm25"]["validation"]["ndcg@10"]]
    for result in summary["models"].values():
        recall.append(result["validation"]["fused"]["recall@10"])
        ndcg.append(result["validation"]["fused"]["ndcg@10"])
    x = list(range(len(labels)))
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar([value - 0.2 for value in x], recall, width=0.4, label="Recall@10")
    axis.bar([value + 0.2 for value in x], ndcg, width=0.4, label="nDCG@10")
    axis.set_title("Validation retrieval quality")
    axis.set_ylabel("Score")
    axis.set_xticks(x, labels, rotation=15, ha="right")
    axis.set_ylim(0, 1)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, format="svg")
    figure.savefig(output.with_suffix(".png"), dpi=150)
    plt.close(figure)


def _save_latency(summary: dict, output: Path) -> None:
    labels = list(summary["models"])
    p50 = [result["resources"]["query_p50_ms"] for result in summary["models"].values()]
    p95 = [result["resources"]["query_p95_ms"] for result in summary["models"].values()]
    x = list(range(len(labels)))
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar([value - 0.2 for value in x], p50, width=0.4, label="p50")
    axis.bar([value + 0.2 for value in x], p95, width=0.4, label="p95")
    axis.set_title("Query encode and search latency")
    axis.set_ylabel("Milliseconds")
    axis.set_xticks(x, labels, rotation=15, ha="right")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, format="svg")
    figure.savefig(output.with_suffix(".png"), dpi=150)
    plt.close(figure)


def _save_resources(summary: dict, output: Path) -> None:
    labels = list(summary["models"])
    fields = [
        ("model_mib", "Model (MiB)"),
        ("projected_index_mib_10k", "Index / 10k docs (MiB)"),
        ("peak_rss_mib", "Peak RSS (MiB)"),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(11, 4.5))
    for axis, (field, title) in zip(axes, fields, strict=True):
        values = [result["resources"][field] for result in summary["models"].values()]
        axis.bar(labels, values)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=25)
        for tick in axis.get_xticklabels():
            tick.set_horizontalalignment("right")
    figure.suptitle("Resource cost")
    figure.tight_layout()
    figure.savefig(output, format="svg")
    figure.savefig(output.with_suffix(".png"), dpi=150)
    plt.close(figure)


def write_report(summary: dict, decision: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    lines = ["# SPLADE evaluation", "", f"Winner: `{decision['winner']}`", ""]
    lines.extend(
        [
            "| Method | Validation Recall@10 | Validation nDCG@10 | Query p95 ms | Index MiB/10k | Gate |",
            "|---|---:|---:|---:|---:|---|",
            f"| BM25 | {summary['bm25']['validation']['recall@10']:.4f} | {summary['bm25']['validation']['ndcg@10']:.4f} | – | – | baseline |",
        ]
    )
    for key, result in summary["models"].items():
        metrics = result["validation"]["fused"]
        resources = result["resources"]
        gate = "pass" if decision["models"][key]["passed"] else ", ".join(decision["models"][key]["reasons"])
        lines.append(
            f"| {key} | {metrics['recall@10']:.4f} | {metrics['ndcg@10']:.4f} | "
            f"{resources['query_p95_ms']:.3f} | {resources['projected_index_mib_10k']:.2f} | {gate} |"
        )
    lines.extend(["", "Metrics come from `summary.json`; SVG files are generated from the same values.", ""])
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    (output / "decision.json").write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    _save_quality(summary, output / "quality.svg")
    _save_latency(summary, output / "latency.svg")
    _save_resources(summary, output / "resources.svg")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    summary = json.loads((args.run_dir / "summary.json").read_text(encoding="utf-8"))
    write_report(summary, choose_winner(summary), args.run_dir)


if __name__ == "__main__":
    main()
