# dog-training-rag

강아지 훈련 질문에 **검색된 근거만으로** 답하는 최소 RAG 데모. 웹 문서를 마크다운으로
바꿔 구조 인식 청킹하고, 로컬 임베딩 + 로컬 벡터 DB로 검색한 뒤, 생성만 OpenAI에 맡긴다.

RAG 프레임워크(LangChain / LlamaIndex)를 쓰지 않고 각 단계를 직접 구현했다.
검색 품질을 눈이 아니라 **hit rate / MRR로 재는 평가 하네스**가 같이 들어 있다.

```
웹 문서 10편 ──trafilatura──▶ 마크다운 ──헤딩 트리 청킹──▶ 155개 청크
                                                              │
                                   multilingual-e5-small (384d, 로컬 ONNX)
                                                              │
                                                      Qdrant 임베디드 DB
                                                              │
질문 ──임베딩──▶ 코사인 top-5 ──▶ 근거 프롬프트 ──▶ OpenAI ──▶ 인용 달린 한국어 답변
```

## 지금 상태 — 검색 품질이 낮다

숨기지 않고 적는다. 데모는 끝까지 동작하지만 **검색이 제대로 순위를 못 매긴다.**

| 지표 | 청크 단위 | 문서 단위 |
|---|---|---|
| hit rate@1 | 0.000 | 0.400 |
| hit rate@5 | 0.100 | 0.800 |
| hit rate@10 | 0.500 | 0.900 |
| MRR@10 | 0.087 | — |

원인은 **top-20 전체의 점수 폭이 0.0125**라는 것이다 (1위 0.8040 / 20위 0.7915).
임베딩이 모든 청크에 사실상 같은 점수를 줘서 순위가 노이즈다. 파이프라인 버그는 아니다 —
정답 청크 본문을 그대로 질의로 넣으면 자기 자신을 9/10 찾아온다.

유력한 원인은 **int8 양자화 모델**이다. fp32 ONNX(약 470MB)가 320MiB 지점에서 반복적으로
끊겨 118MB 양자화본으로 갈아탄 결과다. 검증용으로 `demo/compare_models.py`를 만들어 뒀고
아직 실행하지 않았다. 자세한 내용은 [`STATUS.md`](STATUS.md).

## 빠른 시작

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt   # Windows
# source .venv/bin/activate && pip install -r requirements.txt  # macOS/Linux

cp .env.example .env        # OPENAI_API_KEY 를 채운다

python demo/ingest.py       # 수집 → 청킹 → 임베딩 → 색인
python demo/server.py       # http://127.0.0.1:8100
```

검색 성능만 재려면:

```bash
python demo/make_evalset.py   # 청크에서 질문 10개 역생성 (OpenAI 사용)
python demo/evaluate.py       # hit rate / MRR — 생성 호출 없음
python demo/compare_models.py # 임베딩 모델 후보 비교
```

> **Qdrant 임베디드 모드는 저장 디렉터리를 잠근다.** `ingest.py` / `evaluate.py` /
> `compare_models.py` 와 `server.py` 를 동시에 켜면 안 된다.

## 구성

| 파일 | 역할 |
|---|---|
| `demo/config.py` | 경로·파라미터·`.env` 로더 (의존성 없이 직접 파싱) |
| `demo/sources.py` | 수집 대상 10편의 URL·메타데이터 |
| `demo/fetcher.py` | HTML → 본문만 남긴 마크다운 (헤딩 구조 보존) |
| `demo/chunker.py` | **구조 인식 마크다운 청킹** — 이 프로젝트의 핵심 |
| `demo/embedder.py` | fastembed(ONNX) 로더. torch를 끌어오지 않는다 |
| `demo/store.py` | Qdrant 임베디드 컬렉션 |
| `demo/ingest.py` | 색인 파이프라인 CLI |
| `demo/server.py` | FastAPI — 검색 + OpenAI 생성 + 정적 페이지 |
| `demo/web/index.html` | 채팅 UI. 인용 번호를 누르면 해당 근거로 스크롤 |
| `demo/make_evalset.py` | 평가셋 생성 |
| `demo/evaluate.py` | 검색 평가 (hit rate / MRR) |
| `demo/compare_models.py` | 임베딩 모델 후보 비교 |

## 구조 인식 청킹

고정 길이로 자르면 "3단계 방법"의 2번과 3번이 다른 청크로 흩어진다. 그 청크만 검색되면
답변이 반쪽이 된다. 그래서 **문서의 헤딩 트리를 1차 경계**로 쓴다.

1. **헤딩 트리 파싱** — `#`~`######`을 스택으로 추적해 각 본문에 `문서 > H2 > H3` 경로를 부여.
   코드 펜스 안의 `#`은 헤딩으로 보지 않는다
2. **짧은 섹션 병합** — 280자 미만 섹션은 같은 부모를 공유하는 직전 섹션에 흡수
3. **긴 섹션 분할** — 1,100자를 넘으면 문단/리스트 덩어리 경계에서만 분할.
   연속된 리스트 항목은 한 덩어리로 유지해 절차가 쪼개지지 않게 한다
4. **문장 경계 폴백** — 한 문단이 통째로 상한을 넘으면(실제로 1,899자짜리가 있었다)
   문장 경계에서 재분할. 그래도 안 되면 마지막 수단으로만 하드 컷
5. **오버랩** — 이어지는 조각에 직전 청크의 꼬리 160자를 문장 경계에서 잘라 붙인다
6. **컨텍스트 헤더** — 임베딩 텍스트 앞에 `[문서 > H2 > H3]`를 붙인다. 청크 본문에 주어가
   없고 위 섹션에만 있는 경우를 헤딩이 보완한다. 표시할 때는 본문만 쓴다

결과: 155개 청크, 최소 41자 / 중앙 약 900자 / 최대 1,259자.
파라미터는 `demo/config.py`의 `MAX_CHARS` / `MIN_CHARS` / `OVERLAP_CHARS`.

## 임베딩 모델

`intfloat/multilingual-e5-small` — 118M 파라미터, 384차원. "작으면서 한국어가 되는" 지점.
torch 없이 ONNX Runtime으로 돌리려고 fastembed에 커스텀 모델로 등록해 쓴다.

- e5 계열은 접두사가 필수다: 문서는 `passage: `, 질의는 `query: `
- 수집 자료가 영어이고 질문은 한국어라 **교차언어 검색**이다. 다국어 모델이어야 한다
- 로드에 실패하면 `paraphrase-multilingual-MiniLM-L12-v2`로 자동 폴백한다

## 수집 자료

전부 tier-1 출처(ASPCA / AKC / Humane Society)다. 원문은 저작권 문제로 커밋하지 않는다 —
`demo/ingest.py`가 URL에서 다시 받아온다.

| 주제 | 문서 | 청크 |
|---|---|---|
| 짖음 | 1 | 40 |
| 배변 훈련 | 5 | 36 |
| 입질·물기 | 2 | 33 |
| 분리불안 | 1 | 31 |
| 씹기·물어뜯기 | 1 | 15 |

주제 5개 밖의 질문은 "제공된 자료로는 답할 수 없습니다"로 거부하는 것이 정상 동작이다.

## 안전 정책

`demo/server.py`의 시스템 프롬프트가 **근거 문서보다 우선하는** 규칙을 강제한다.

- 체벌·주둥이 잡기·알파롤·프롱/전기충격 목줄·배변 실수에 코 박기 등 혐오 기법 제안 금지
- 사람이 다쳤거나 공격성 신호가 있거나 아이·노인이 노출된 상황이면 전문가 상담을 먼저 안내
- 갑작스러운 배변 실패나 성견의 급성 물기는 의학적 원인 가능성을 언급하고 수의사 진료 권고
- 답변마다 개별 진단이 아님을 명시

## 문서

- [`STATUS.md`](STATUS.md) — 현재 상태, 알려진 문제, 다음 액션
- [`HISTORY.md`](HISTORY.md) — 만들어진 과정과 도중에 부딪힌 문제들
- [`prompt.txt`](prompt.txt) — 루프 엔지니어링 프롬프트 초안과, 이 파이프라인에
  `ralph-loop`을 그대로 걸면 왜 안 되는지에 대한 검토

## 일부러 안 한 것

- **하이브리드 검색(BM25 + dense)** — 코퍼스가 커지면 1순위 추가 항목
- **리랭킹** — 후보 20개를 top-5로 자르기만 하고 cross-encoder는 쓰지 않는다
- **쿼리 재작성 / HyDE**
- **증분 색인** — `ingest.py`는 매번 컬렉션을 지우고 다시 만든다
- **질의 로그 / 트레이싱** — 무엇이 실패했는지 누적되지 않는다
