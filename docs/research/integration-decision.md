# 통합 판단

## 현재 결정

`opensearch-project/opensearch-neural-sparse-encoding-doc-v2-mini`만 Rust로 이식한다.

이유:

- 고정 선택셋의 품질 gate를 통과했다.
- 일반 SPLADE-Tiny보다 질의 p95가 약 5배 짧다.
- 1만 문서 인덱스 환산값이 32MiB 제한 안에 있다.
- Apache-2.0이라 배포 조건이 명확하다.

## 예정 검색 구성

Rust 동등성까지 통과하면 `pentesting`에서 다음 두 구성을 비교한다.

- 기존: `BM25 + Dense + Graph`
- 후보: `BM25 + Dense + SPLADE + Graph`

아직 `pentesting` 코드는 수정하지 않는다. Python과 Rust의 top-256 토큰 ID와 가중치가 일치한 뒤 통합한다.
