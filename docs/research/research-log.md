# 연구 기록

| 단계 | 한 일 | 판단 |
|---|---|---|
| 1 | 실제 `pentesting` 문서 267개와 qrel 60개 고정 | 선택 36 / 검증 24로 누수 방지 |
| 2 | BM25와 SPLADE 3종을 동일 조건에서 실행 | OpenSearch mini만 Rust 이식 대상으로 선택 |
| 3 | 선택셋 gate 후 검증셋 공개 | IF BERT-Tiny의 검증 고득점으로 선택을 뒤집지 않음 |
| 4 | 문서 모델 ONNX export, 질의 정적 가중치 분리 | Rust에서 dense 30,522차원 저장 불필요 |
| 5 | fixture 벡터 Python↔Rust 비교 | query 특수 토큰 포함 차이를 찾아 수정 |
| 6 | 267문서·60질의 Rust 재평가 | top-10 60/60 일치, 품질 동등성 통과 |
| 7 | release 처리량·메모리 실측 | 질의는 온라인, 문서는 사전 인덱싱으로 제한 |
| 8 | 리뷰 후 provenance·타이밍·인덱스 재검증 | SHA 강제, 질의+검색 범위 통일, 숫자 posting 직렬화 |

원시 JSON, 명령 기록, 그래프는 `artifacts/eval/2026-08-20-pentesting-267`에 둔다. 원문 문서와 전체 sparse vector는 복제하지 않는다.
