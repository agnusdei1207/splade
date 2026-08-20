# splade

SPLADE 모델을 비교하고, 합격 모델 하나를 Rust로 이식하기 위한 실험 저장소입니다.

진행 순서와 합격 기준은 [설계 문서](docs/superpowers/specs/2026-08-20-splade-rust-poc-design.md)에 기록합니다.

## 원칙

- 같은 문서와 질의로 모델을 비교합니다.
- 원시 측정값에서 표와 그래프를 자동 생성합니다.
- 모델 파일은 커밋하지 않고 버전과 SHA-256을 고정합니다.
- 합격 모델만 Rust로 이식합니다.
- `pentesting`은 Rust POC가 통과한 뒤 수정합니다.
