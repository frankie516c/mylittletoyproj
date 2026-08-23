# 현재 상태 — 2026-08-23 기준

> **공개 저장소 안내** — 이 문서는 별도의 **비공개** 저장소 `dog-training-rag` 에서 진행한
> 작업을 기록한 것이다. 아래에 언급되는 브랜치·기존 모듈(`backend/app/`, EvidenceCard 스키마,
> 데이터 계약 등)과 방법론 조사 문서(`docs/rag-research/`, 10편)는 그 비공개 저장소에 있고
> 여기에는 포함하지 않았다. 이 폴더에는 데모 코드만 들어 있다.

한 줄 요약: **데모는 끝까지 동작하지만 검색 품질이 낮다.** 원인 규명 중간에 멈춰 있고,
다음 할 일은 임베딩 모델 비교다.

작업 이력은 [`HISTORY.md`](HISTORY.md) 참고. 방법론 조사 문서(`docs/rag-research/`)는 비공개 저장소에 있다.

---

## 브랜치·커밋

| 항목 | 값 |
|---|---|
| 브랜치 | `feature/bite-rag-research` (로컬 전용, push 안 함) |
| HEAD | `a641622` feat(ui): 아이소메트릭 방 씬과 강아지 리그 런타임 |
| 베이스 | `23cd433` (구 `feature/room-landing-ui` HEAD) |

`a641622`는 이 세션이 아니라 사용자가 직접 커밋한 frontend 작업이다.

**untracked (커밋 안 됨)**

- `demo/` — 데모 전체
- `docs/rag-research/` — 리서치 문서 10개
- `scripts/collect_targeted_candidates.py` — 세션 이전부터 있던 파일
- `HISTORY.md`, `STATUS.md` — 이 문서들

`data/`는 `.gitignore` 대상이라 원문 마크다운·청크·벡터DB·평가셋 전부 추적되지 않는다.
`.env`도 마찬가지(OpenAI 키 포함).

## 실행 방법

```bash
# 서버는 현재 꺼져 있음
.venv/Scripts/python.exe demo/server.py     # http://127.0.0.1:8100

# 재색인이 필요하면 (서버를 먼저 끌 것)
.venv/Scripts/python.exe demo/ingest.py --skip-fetch

# 검색 평가만
.venv/Scripts/python.exe demo/evaluate.py
```

> **Qdrant 임베디드 모드는 저장 디렉터리를 잠근다.** `ingest.py` / `evaluate.py` /
> `compare_models.py`와 `server.py`를 동시에 켜면 안 된다.

## 구성

| 파일 | 역할 | 상태 |
|---|---|---|
| `demo/config.py` | 경로·파라미터·`.env` 로더 | 동작 |
| `demo/sources.py` | 소스 10편 URL·메타데이터 | 동작 |
| `demo/fetcher.py` | HTML → 마크다운 (trafilatura) | 동작 |
| `demo/chunker.py` | 구조 인식 마크다운 청킹 | 동작 |
| `demo/embedder.py` | fastembed ONNX 로더 + 폴백 | 동작 (양자화본 사용 중) |
| `demo/store.py` | Qdrant 임베디드 | 동작 |
| `demo/ingest.py` | 색인 파이프라인 CLI | 동작 |
| `demo/server.py` | FastAPI + OpenAI 생성 | 동작 — **질의 로그 미구현** |
| `demo/web/index.html` | 채팅 UI | 동작 — **예시 질문이 입질 중심 그대로** |
| `demo/make_evalset.py` | 평가셋 생성 | 동작 |
| `demo/evaluate.py` | 검색 평가 (hit rate / MRR) | 동작 |
| `demo/compare_models.py` | 임베딩 모델 비교 | **미실행** |
| `demo/README.md` | 설계 근거·파라미터 설명 | 코퍼스 2편 기준으로 쓰여 있음 — 갱신 필요 |

## 색인 상태

```
문서 10편 / 청크 155개 / 384차원 / Qdrant embedded (data/demo/qdrant)
임베딩: intfloat/multilingual-e5-small (int8 양자화 ONNX, ~118MB)
생성:   gpt-4o-mini (DEMO_OPENAI_MODEL로 변경 가능)
```

| 주제 | 청크 |
|---|---|
| 짖음 | 40 |
| 배변 훈련 | 36 |
| 입질·물기 | 33 |
| 분리불안 | 31 |
| 씹기·물어뜯기 | 15 |

청킹 파라미터: `MAX_CHARS=1100` / `MIN_CHARS=280` / `OVERLAP_CHARS=160`
→ 실제 분포 최소 41자 / 중앙 약 900자 / 최대 1,259자

## 성능

### 지연시간 — 문제 없음

검색 7ms, 생성 약 5.3초(gpt-4o-mini).

### 검색 품질 — 문제 있음

평가셋 `data/demo/eval/golden_v1.jsonl` (질문 10개, seed 42, gpt-4o-mini 생성) 기준:

| 지표 | 청크 단위 | 문서 단위 |
|---|---|---|
| hit rate@1 | **0.000** | 0.400 |
| hit rate@3 | 0.000 | 0.400 |
| hit rate@5 | **0.100** | 0.800 |
| hit rate@10 | 0.500 | 0.900 |
| MRR@10 | **0.087** | — |

상세: `data/demo/eval/retrieval_results_v1.json`

## 알려진 문제

### 1. 임베딩 점수가 변별력이 없다 (최우선)

top-20 전체의 점수 폭이 **0.0125**다 (1위 0.8040 / 5위 0.7970 / 20위 0.7915).
사실상 모든 청크에 같은 점수를 매기고 있어 순위가 노이즈다.

- 파이프라인 버그는 아니다 — 정답 청크 본문을 질의로 넣으면 자기 자신을 9/10 찾아온다
- 범위 밖 질문("고양이 화장실 훈련")도 0.79로 나와 점수 임계값을 쓸 수 없다
- **유력한 원인: int8 양자화.** fp32 ONNX가 320MiB에서 반복적으로 끊겨 양자화본으로 갈아탄 결과다

검증 방법: `demo/compare_models.py` 실행 (재색인 없이 메모리에서 비교).
후보는 e5-small int8 / e5-small O4 / 접두사 제거 / MiniLM-L12 multilingual.

### 2. 평가셋의 정답 라벨이 과하게 엄격하다

청크 하나만 정답으로 보는데, 같은 문서의 옆 청크가 똑같이 답이 되는 경우가 많다.
문서 단위로 재면 @5가 0.8로 올라간다. 다만 이걸 감안해도 문서 단위 @1 0.4는 나쁘다.

개선안: 정답을 청크 집합으로 두거나, 문서 단위 지표를 병기한다.

### 3. 합성 평가셋의 편향

청크에서 역생성한 질문이라 실사용 질문보다 쉬운 쪽으로 치우친다. 상한이 아니라
하한 점검용으로만 써야 한다. 사람이 만든 질문으로 점차 대체해야 한다.

### 4. 코퍼스 커버리지 구멍

주제 5개 밖의 질문은 답하지 못한다. 5개 안이어도 세부 케이스는 비어 있다 —
예: "패드에 앞발만 올리고 소변" 같은 조준 문제는 어느 문서에도 없다(패드 크기 언급만).

### 5. 서버에 질의 로그가 없다

무엇이 실패했는지 누적되지 않아 커버리지 구멍을 데이터로 찾을 수 없다.
JSONL 로깅 추가를 시도했으나 반영 전에 중단됐다.

## 다음 액션

우선순위 순.

1. **`demo/compare_models.py` 실행** — 양자화가 원인인지 확정.
   원인이면 청킹·하이브리드를 손대는 건 다 헛수고가 된다. 여기서 시작해야 한다.
2. 원인이 양자화로 확인되면 대안 확보 — O4 변형, MiniLM, 또는 fp32 재다운로드
   (320MiB 스톨 회피 방법 필요: 재개 다운로드, `HF_TOKEN` 설정, 미러 등)
3. 그래도 부족하면 **하이브리드 검색(BM25 + dense, RRF)** 추가.
   리서치 문서가 한국어에서 BM25 비중이 큰 이유를 다룬다 (비공개 저장소)
4. 평가셋을 30~50문항으로 확장하고 정답을 청크 집합으로 재정의
5. 서버 질의 로그(JSONL) + UI 커버리지 표시 추가
6. `demo/README.md`를 현재 코퍼스(10편/155청크) 기준으로 갱신

## 결정이 필요한 사항

- **데모와 본 시스템의 관계** — 데모는 "작은 모델 + 웹 문서 원문 인덱싱"이고,
  리서치 종합 문서는 본 시스템에 "BGE-M3 하이브리드 + EvidenceCard claim만 인덱싱"을 권한다.
  의도적으로 갈라진 상태다. 데모를 본 시스템에 합칠지, 별도로 둘지
- 리서치 문서가 남긴 미결 사항 — 입질 스코프 3분할 확정, 유료 생성 API 연결 여부
  (비공개 저장소의 종합 문서 12절)
- `demo/`와 `docs/rag-research/`를 커밋할지
