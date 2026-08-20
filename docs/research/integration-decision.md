# 통합 판단

## 현재 결정

`opensearch-project/opensearch-neural-sparse-encoding-doc-v2-mini`만 Rust로 이식한다.

이유:

- 고정 선택셋의 품질 gate를 통과했다.
- 일반 SPLADE-Tiny보다 질의 p95가 약 5배 짧다.
- 1만 문서 인덱스 환산값이 32MiB 제한 안에 있다.
- Apache-2.0이라 배포 조건이 명확하다.

## POC 판정

Rust 이식은 합격했다.

- 267개 문서, 60개 질의의 top-10 순위 60/60 일치
- 최대 점수 오차 0.00003288
- Rust 질의+검색 p95 0.075ms
- 런타임 모델 132.39MiB, 최대 RSS 724.1MiB
- 실제 직렬화 인덱스 20.30MiB/1만 문서
- release 문서 인코딩 1.115 docs/s: 온라인 요청에서는 금지하고 사전 인덱싱에서만 사용

## 검색 구성

`pentesting`에서 다음 구성을 비교한다.

- 기존: `BM25 + Dense + Graph`
- 후보: `BM25 + Dense + SPLADE + Graph`

SPLADE는 BM25를 대체하지 않는다. 네 레이어 결과를 각각 보존한 뒤 기존 결합 정책에 한 입력으로 추가한다. 모델 파일은 Git에 넣지 않고 해시 검증된 설치 자산으로 공급한다.
