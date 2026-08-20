# splade

SPLADE 모델을 비교하고, 합격 모델 하나를 Rust로 이식하기 위한 실험 저장소입니다.

진행 순서와 합격 기준은 [설계 문서](docs/superpowers/specs/2026-08-20-splade-rust-poc-design.md)에 기록합니다.

## 원칙

- 같은 문서와 질의로 모델을 비교합니다.
- 원시 측정값에서 표와 그래프를 자동 생성합니다.
- 모델 파일은 커밋하지 않고 버전과 SHA-256을 고정합니다.
- 합격 모델만 Rust로 이식합니다.
- `pentesting`은 Rust POC가 통과한 뒤 수정합니다.

## 결론

| 항목 | 결과 |
|---|---|
| 합격 모델 | OpenSearch neural sparse v2 mini |
| Python↔Rust 순위 | top-10 60/60 일치 |
| Rust 질의+검색 p95 | 0.075ms |
| Rust 문서 인코딩 | 1.115 docs/s |
| 런타임 모델 파일 | 132.39MiB |
| 직렬화 인덱스 환산 | 20.30MiB/1만 문서 |

Rust 이식은 합격했다. 질의 인코딩은 온라인 경로에 적합하지만, tract 문서 인코딩은 느리므로 인덱싱 작업에서만 실행한다. 상세 수치는 [평가](docs/research/evaluation.md)와 [Rust 이식](docs/research/rust-port.md)에 있다.

## 재현

모든 Python/Rust 작업은 메모리 4GiB, CPU 2개로 제한된 Docker 안에서 실행한다.

```powershell
scripts/poc.ps1 pytest -q
scripts/poc.ps1 export
scripts/rust-test.ps1 test
scripts/rust-test.ps1 bench
```
