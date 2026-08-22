"""Query the Hugging Face API for sparse-retrieval candidates and report real specs.

Model choice was previously made from memory and a size guess. This pulls the
actual repository metadata - parameter count, file sizes, license, vocabulary and
tokenizer type - so the shortlist is grounded in what is downloadable today.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.parse
import urllib.request

API = "https://huggingface.co/api"

CANDIDATES = [
    # currently selected
    "opensearch-project/opensearch-neural-sparse-encoding-doc-v2-mini",
    # english, larger rungs
    "opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill",
    "opensearch-project/opensearch-neural-sparse-encoding-doc-v3-distill",
    "opensearch-project/opensearch-neural-sparse-encoding-doc-v3-gte",
    "opensearch-project/opensearch-neural-sparse-encoding-v2-distill",
    "prithivida/Splade_PP_en_v1",
    "prithivida/Splade_PP_en_v2",
    "naver/splade-v3",
    "naver/splade-v3-distilbert",
    "tomaarsen/inference-free-splade-bert-tiny-nq",
    # multilingual candidates
    "BAAI/bge-m3",
    "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1",
    "naver/splade-v3-lexical",
    "aken12/splade-japanese-v3",
    "hotchpotch/japanese-splade-v2",
    "yosefw/splade-multilingual",
]

SEARCH_QUERIES = [
    ("sparse retrieval multilingual", {"search": "sparse", "filter": "feature-extraction"}),
    ("splade", {"search": "splade"}),
    ("neural sparse", {"search": "neural-sparse"}),
]


def get(url: str) -> object | None:
    request = urllib.request.Request(url, headers={"User-Agent": "splade-poc-survey"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        print(f"  ! {url} -> HTTP {error.code}", file=sys.stderr)
    except Exception as error:  # noqa: BLE001 - survey should not abort on one repo
        print(f"  ! {url} -> {error}", file=sys.stderr)
    return None


def model_report(repo_id: str) -> dict | None:
    info = get(f"{API}/models/{urllib.parse.quote(repo_id)}")
    if not isinstance(info, dict):
        return None
    siblings = info.get("siblings") or []
    names = [s.get("rfilename", "") for s in siblings]
    config = get(f"https://huggingface.co/{repo_id}/resolve/main/config.json")
    tokenizer_config = get(f"https://huggingface.co/{repo_id}/resolve/main/tokenizer_config.json")
    safetensors = info.get("safetensors") or {}
    parameters = safetensors.get("total")
    if parameters is None and isinstance(safetensors.get("parameters"), dict):
        parameters = sum(safetensors["parameters"].values())
    return {
        "id": repo_id,
        "license": (info.get("cardData") or {}).get("license") or info.get("license"),
        "downloads": info.get("downloads"),
        "likes": info.get("likes"),
        "parameters": parameters,
        "architecture": (config or {}).get("architectures", [None])[0] if config else None,
        "model_type": (config or {}).get("model_type") if config else None,
        "vocab_size": (config or {}).get("vocab_size") if config else None,
        "hidden_size": (config or {}).get("hidden_size") if config else None,
        "layers": (config or {}).get("num_hidden_layers") if config else None,
        "max_position": (config or {}).get("max_position_embeddings") if config else None,
        "tokenizer_class": (tokenizer_config or {}).get("tokenizer_class") if tokenizer_config else None,
        "has_onnx": any(name.endswith(".onnx") for name in names),
        "has_safetensors": any(name.endswith(".safetensors") for name in names),
        "has_sparse_linear": any("sparse" in name.lower() for name in names),
        "files": len(names),
    }


def main() -> None:
    print("=== 지정 후보 조회 ===")
    reports = []
    for repo_id in CANDIDATES:
        report = model_report(repo_id)
        if report:
            reports.append(report)
            print(f"  ok  {repo_id}")
        else:
            print(f"  --  {repo_id} (없음/비공개)")

    print("\n=== 검색으로 추가 발굴 ===")
    discovered: dict[str, dict] = {}
    for label, params in SEARCH_QUERIES:
        query = urllib.parse.urlencode({**params, "sort": "downloads", "direction": "-1", "limit": 40})
        results = get(f"{API}/models?{query}")
        if not isinstance(results, list):
            continue
        print(f"  [{label}] {len(results)}건")
        for item in results:
            repo_id = item.get("modelId") or item.get("id")
            if repo_id and repo_id not in {r["id"] for r in reports}:
                discovered[repo_id] = {
                    "id": repo_id,
                    "downloads": item.get("downloads"),
                    "likes": item.get("likes"),
                    "tags": [t for t in (item.get("tags") or []) if not t.startswith("region")][:12],
                }

    ranked = sorted(discovered.values(), key=lambda r: -(r["downloads"] or 0))[:30]
    payload = {"shortlist": reports, "discovered": ranked}
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
