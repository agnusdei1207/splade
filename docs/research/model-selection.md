# 모델 선택

## 결정

Python에서 세 모델을 먼저 비교하고, 합격 모델 하나만 Rust로 옮긴다.

| 모델 | 비교 목적 |
|---|---|
| `inference-free-splade-bert-tiny-nq` | 최소 모델·최저 질의 비용 |
| `opensearch-neural-sparse-encoding-doc-v2-mini` | 실사용 품질 후보 |
| `splade-tiny` | 일반 SPLADE의 질의 확장 효과 확인 |

한국어 inference-free 모델은 약 570 MiB이고 문서 벡터가 지나치게 커서 제외했다. Naver 원본 가중치는 비상업 라이선스라 제외했다.

## 평가 자료

`pentesting`의 `README.md`, `ARCHITECTURE.md`, `docs/**/*.md`를 읽기 전용으로 사용한다. 원문은 결과 저장소에 복사하지 않는다.

- 질의: 60개
- 모델 선택: 36개
- 최종 검증: 24개
- 범주: exact, semantic, 한국어→영어, no-answer
