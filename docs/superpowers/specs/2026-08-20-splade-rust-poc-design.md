# Rust SPLADE POC 설계

## 목표

세 SPLADE 모델을 실제 `pentesting` 검색 자료로 평가한다. 합격 모델 하나만 Rust로 이식하고, 이후 `BM25 + Dense + SPLADE + Graph` 검색에 사용할 수 있는지 판단한다.

## 비교 모델

| 모델 | 용도 | 선택 이유 |
|---|---|---|
| `tomaarsen/inference-free-splade-bert-tiny-nq` | 최소 비용 후보 | 4.42M 파라미터, 질의 추론 없음 |
| `opensearch-project/opensearch-neural-sparse-encoding-doc-v2-mini` | 실사용 후보 | 22.7M 파라미터, 공개 BEIR 성능 우수 |
| `rasyosef/splade-tiny` | 일반 SPLADE 기준 | 4.42M 파라미터, 질의 확장 비용과 효과 비교 |

모델 이름, revision, 라이선스, 파일 SHA-256을 실행 기록에 남긴다. 가중치 파일은 Git에 넣지 않는다.

## 진행 순서

1. `pentesting` 문서를 읽기 전용으로 추출한다.
2. 대표 질의 50~100개와 정답 문서 ID를 작성한다.
3. Python 기준 구현으로 세 모델과 BM25를 같은 조건에서 평가한다.
4. 원시 결과에서 표, SVG 그래프, Markdown 보고서를 생성한다.
5. 합격 모델 하나를 Rust로 이식한다.
6. Python과 Rust의 희소 벡터와 검색 결과가 일치하는지 검증한다.
7. 통과하면 `pentesting`의 네 번째 검색 레이어 통합안을 작성한다.

## 평가 데이터

질의는 다음 범주를 균형 있게 포함한다.

- CVE, 경로, 함수명, 오류 코드 등 정확 검색
- 같은 뜻을 다른 표현으로 묻는 의미 검색
- 한국어 질문과 영어 문서가 섞인 검색
- 여러 문서 관계를 따라가야 하는 검색
- 관련 문서가 없는 질의

각 정답에는 문서 ID, 관련도 `0~2`, 판단 근거를 기록한다. 원문과 비밀값은 저장소에 복사하지 않는다.
질의는 범주별 비율을 유지해 선택용 60%와 최종 검증용 40%로 나눈다. 최종 검증 결과를 보기 전에는 모델과 설정을 고정한다.

## 지표

- Recall@5, Recall@10
- MRR@10, nDCG@10
- exact identifier Recall@10
- 질의 지연시간 p50, p95, p99
- 문서 인덱싱 처리량과 peak RSS
- 모델 및 인덱스 크기
- BM25와 결합했을 때 추가로 찾은 정답 수

## 모델 합격 기준

다음을 모두 만족하는 모델 중 `BM25 + 모델`의 선택용 nDCG@10이 가장 높은 모델을 고른다.

- `BM25 + 모델`의 Recall@10이 BM25보다 낮아지지 않는다.
- exact identifier Recall@10은 BM25와 결합했을 때 낮아지지 않는다.
- 희소 인덱스는 문서 1만 개당 32 MiB 이하이다.
- inference-free 모델은 질의 p95 추가 비용이 10ms 이하이다.
- 모델 라이선스가 프로젝트 배포 조건과 맞는다.

어느 모델도 통과하지 못하면 Rust 이식을 중단하고 결과만 보고한다.

## Rust 이식

Rust 런타임은 먼저 `tract`를 사용한다. ONNX 호환 문제를 재현하고 해결할 수 없을 때만 `ort`를 검토한다.

```rust
pub struct SparseVector {
    pub term_ids: Vec<u16>,
    pub weights: Vec<f32>,
}
```

- 문서 벡터는 최대 256개 항목만 저장한다.
- 질의 벡터는 최대 32개 항목만 저장한다.
- 전체 어휘 크기의 dense 배열은 인덱스에 저장하지 않는다.
- 검색은 inverted index의 sparse dot product로 수행한다.

Rust 합격 기준은 Python top-256 토큰 ID 일치와 가중치 오차 `1e-4` 이하이다.

## 결과물

```text
artifacts/eval/<run-id>/
  manifest.json
  environment.json
  commands.log
  metrics.json
  per-query.jsonl
  quality.svg
  latency.svg
  resources.svg
  report.md

docs/research/
  model-selection.md
  evaluation.md
  rust-port.md
  integration-decision.md
```

연구 문서는 가설, 실행 명령, 관찰 결과, 결론만 적는다. 같은 설명을 여러 문서에 반복하지 않는다.

## `pentesting` 통합 경계

POC 단계에서는 `pentesting`을 수정하지 않는다. Rust 이식까지 통과한 후 다음 두 구성을 비교한다.

- 기존: `BM25 + Dense + Graph`
- 후보: `BM25 + Dense + SPLADE + Graph`

네 검색 결과는 RRF로 합친다. SPLADE가 품질을 높이지 못하거나 자원 기준을 넘으면 통합하지 않는다.
