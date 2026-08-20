# 평가 결과

## 방법

- corpus: `pentesting` Markdown 267개, 2,890,634바이트
- 질의: 60개(선택 36, 최종 검증 24)
- 비교: BM25, 세 SPLADE 모델, `BM25 + 각 모델` RRF
- 실행: Docker 4 GiB, CPU 2, swap 0, PID 512
- 원시 결과: `artifacts/eval/2026-08-20-pentesting-267`

한국어 질의는 실제 검색 배선처럼 사람이 작성한 영어 retrieval query로 평가했다. 모델과 설정은 최종 검증 결과를 보기 전에 고정했다.

## 최종 검증 결과

| 방법 | Recall@10 | nDCG@10 | 질의 p95 | 인덱스/1만 문서 |
|---|---:|---:|---:|---:|
| BM25 | 0.6818 | 0.5019 | 10.12ms¹ | – |
| BM25 + IF BERT-Tiny | 0.8182 | 0.5693 | 1.80ms | 15.69MiB |
| BM25 + OpenSearch mini | 0.7727 | 0.5586 | 1.73ms | 18.15MiB |
| BM25 + SPLADE-Tiny | 0.7727 | 0.5560 | 8.99ms | 18.16MiB |

¹ Python 기준선 구현의 검색 시간이다. Rust 현재값으로 해석하지 않는다.

## 선택

`if-opensearch-mini`가 합격 모델이다.

- 선택셋 Recall@10: 0.6765(BM25 0.4706)
- 선택셋 nDCG@10: 0.4261(BM25 0.2734)
- exact Recall@10: 0.7143(BM25 0.6429)
- inference-free 질의 p95: 1.73ms
- projected index: 18.15MiB/1만 문서
- 라이선스: Apache-2.0

IF BERT-Tiny는 최종 검증값은 가장 높았지만 선택셋 Recall gate를 통과하지 못해 탈락했다. 최종 검증 결과를 보고 선택을 뒤집지 않았다.

## 실행 중 발견한 문제

| 문제 | 조치 |
|---|---|
| 기본 PyTorch가 CUDA 패키지 약 2GiB를 요청 | CPU wheel 저장소로 고정 |
| 최신 Sentence Transformers가 구형 `IDF` 이름을 읽지 못함 | 모델 작성 시기 버전과 호환 alias 사용 |
| Hugging Face 비인증 병렬 요청 429 | 고정 revision을 단일 worker로 먼저 다운로드 |
| 컨테이너에 Git 없음 | 호스트가 corpus SHA만 전달 |

## 한계

- 실제 `.pentesting/knowledge` vault가 없어 저장소 문서를 대체 corpus로 사용했다.
- 60개 qrel은 POC용이며 운영 품질 보증 자료가 아니다.
- 네 번째 레이어 도입 판단은 Rust 동등성 확인 후 별도로 한다.
