# 통합 지점 분석 (초안 — 플랜 문서로 병합 예정)

## 현재 검색 경로 (코드 실측)

```
사용자 질의 (임의 언어)
  │
  ├─ retrieval_intent.rs :: normalize_retrieval_intent()
  │     LLM 호출, max_tokens 256, 타임아웃 20초
  │     "Convert the user request into one concise English retrieval query."
  │     실패/타임아웃 → ascii_identifier_fallback(): ASCII 토큰만 남김 (한국어 전부 소실)
  │     bound_english_query(): is_english_dense_query() 통과 필수 = non-ASCII 알파벳 금지
  │
  ▼ 영어 질의
hybrid_search.rs :: search_with_details()
  ├─ lexical_ranking()    BM25  tf≡1, k1=1.4, b=0.75, +0.15 구문 보너스, +tag_boost 0.3×n
  │                        → lexical_limit 24
  ├─ semantic_ranking()   snowflake-arctic-embed-xs int8, 384차원 코사인
  │                        점수 바닥: max(best×0.80, 0.55) → semantic_limit 24
  └─ graph_ranking()      lexical 상위 5 + semantic 상위 5를 시드로 ≤2홉 → graph_limit 24
        │
        ▼ inject_rankings() — RRF k=60, 세 엔진 동일 가중
  base_score = rrf_score
             + phase_bonus(query.phase, kind)
             + ln(backlinks+1) × 0.08
             + pagerank × 0.10
             + min(matched_tags × 0.03, 0.12)
  base_score ×= memory_strength(now)          ← 시간 감쇠
  필터: archive layer / tombstone / sensitive / malicious / kind
```

## SPLADE가 들어갈 때 건드려야 하는 지점

| # | 위치 | 현재 | 변경 |
|---|---|---|---|
| 1 | `refresh_from_memory_using()` | `term_frequencies` / `document_lengths` / `document_frequencies` 구축 | 여기서 sparse 벡터도 생성. dense가 쓰는 `previous_search_texts` 비교 + 콘텐츠 LRU 패턴을 그대로 재사용 |
| 2 | `lexical_ranking()` | BM25 점수 | sparse dot product로 교체 또는 병렬 추가 |
| 3 | `query_terms` (`dedupe_terms(tokenize_with_identifiers(q))`) | BM25 질의어이자 `matched_tags` 입력 | **제거 불가.** 태그 매칭이 이 토큰에 의존하므로 lexical을 바꿔도 토크나이저는 남는다 |
| 4 | `graph_ranking(lexical, semantic)` | lexical 상위 5를 그래프 시드로 사용 | lexical 순위가 바뀌면 그래프 결과도 바뀐다. 2차 효과이며 별도 검증 필요 |
| 5 | `HybridSearchConfig` | `rrf_k`, `lexical_limit`, `tag_boost` … | sparse 관련 설정 추가. 기본값 변경은 스냅샷 테스트에 영향 |
| 6 | `retrieval_intent.rs` | LLM 영어 정규화 (20초 타임아웃) | **다국어 모델 채택 시에만** 제거 가능. 영어 모델을 쓰면 그대로 둬야 한다 |
| 7 | `builder_workspace_index` | `DenseEncoder`, `EMBEDDING_DIM = 384` | sparse 인코더를 같은 크레이트에 둘지 신규 크레이트로 뺄지 결정 필요 |

## 이미 있어서 재사용할 수 있는 것

- `WorkspaceRetrieverCacheKey { cwd, vault_files: Vec<MarkdownFileStamp>, supplemental_notes_hash }` — 경로·바이트수·mtime·SHA-256 기반 변경 감지. sparse 인덱스도 같은 키로 무효화하면 된다.
- `embed_documents_with_cache()` — 콘텐츠 해시 LRU로 미변경 문서의 추론을 건너뛴다. sparse도 동일 구조로 O(delta) 인코딩이 가능하다.
- `WORKSPACE_RETRIEVER_CACHE_LIMIT = 2` — 리트리버 인스턴스 상한. sparse 인덱스 메모리가 이 배수로 잡힌다.
- tract-onnx 런타임이 이미 `builder_workspace_index`에 있다. sparse 모델도 같은 런타임을 쓸 수 있다.

## 자료형 제약

현재 POC의 Rust `SparseVector`는 `term_ids: Vec<u16>`이다. 어휘 30,522는 u16에 들어가지만,
다국어 모델의 어휘 105,879는 **u16 상한 65,535를 넘는다.** 채택 시 `u16 → u32` 변경이 강제된다.

- `rust/vector.rs`: `Vec<u16>` → `Vec<u32>`, `u16::try_from` → `u32::try_from`
- `rust/index.rs`: `HashMap<u16, …>` → `HashMap<u32, …>`, 직렬화 포맷 `term_id.to_le_bytes()` 2바이트 → 4바이트 (포맷 매직 `SPLADE01` → `SPLADE02`)
- `rust/encoder.rs`: `VOCAB_SIZE` 상수, 정적 질의 가중치 shape 검증

posting 자체는 `(u32 ordinal, f32 weight)` 8바이트로 변하지 않는다. 벡터 저장만 항당 6→8바이트.
