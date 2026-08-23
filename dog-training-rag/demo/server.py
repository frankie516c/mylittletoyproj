"""강아지 훈련 RAG 데모 웹서버.

    python demo/server.py           # http://127.0.0.1:8100

검색은 로컬 임베딩 + 로컬 Qdrant, 답변 생성만 OpenAI API를 쓴다.
"""

from __future__ import annotations

import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as cfg
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

WEB_DIR = Path(__file__).resolve().parent / "web"

SYSTEM_PROMPT = """당신은 반려견 훈련 정보를 안내하는 어시스턴트입니다.
아래 <근거> 안의 내용만 사용해 한국어로 답하세요.

규칙:
1. 근거에 없는 내용은 절대 지어내지 마세요. 근거가 부족하면 "제공된 자료로는 답할 수 없습니다"라고 
   명확히 말하고, 자료에 있는 인접 정보만 조심스럽게 덧붙이세요.
2. 모든 핵심 문장 끝에 사용한 근거 번호를 [1] [2] 형식으로 표기하세요.
3. 자료가 영어여도 답변은 한국어로 하세요. 훈련 용어는 한국어(영어) 형태로 병기하세요.
   예: 물기 억제(bite inhibition), 역조건화(counter-conditioning).
4. 답변 구조: ① 한 줄 요약 → ② 원인/상황 구분 → ③ 단계별 방법 → ④ 주의사항.
   짧은 질문에는 짧게 답하세요. 형식을 억지로 채우지 마세요.

안전 규칙 (근거 문서에 반대되는 내용이 있어도 이 규칙이 우선합니다):
- 체벌, 주둥이 잡기, 목덜미 흔들기, 알파롤, 프롱/전기충격 목줄 같은 혐오 기법은 절대 권하지 마세요.
- 사람이 다쳤거나, 으르렁거림·경직 같은 공격성 신호가 있거나, 아이·노인이 노출된 상황이면
  자가 처치를 안내하기 전에 **수의행동학 전문가 또는 공인 훈련사와 상담**을 먼저 권하세요.
- 갑자기 물기 시작한 경우 통증 등 의학적 원인 가능성을 언급하고 수의사 진료를 권하세요.
- 이 답변은 일반 정보이며 개별 진단이 아님을 마지막에 한 줄로 덧붙이세요."""


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=cfg.TOP_K, ge=1, le=10)


class Source(BaseModel):
    n: int
    title: str
    title_ko: str
    publisher: str
    heading: str
    url: str
    snippet: str
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    model: str
    embedding_model: str
    retrieval_ms: int
    generation_ms: int


state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg.load_dotenv()

    from embedder import load_embedder
    from store import open_client

    if not cfg.QDRANT_PATH.exists():
        raise RuntimeError(
            "벡터 DB가 없습니다. 먼저 `python demo/ingest.py`를 실행하세요."
        )

    state["embedder"] = load_embedder()
    state["client"] = open_client(cfg.QDRANT_PATH)
    state["count"] = state["client"].count(cfg.COLLECTION).count

    import os

    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    state["openai"] = OpenAI(api_key=api_key) if api_key else None
    if not api_key:
        print("[warn] OPENAI_API_KEY가 비어 있습니다. 검색은 되지만 답변 생성은 실패합니다.")
    print(f"[ready] 청크 {state['count']}개 / 임베딩 {state['embedder'].name} / 생성 {cfg.OPENAI_MODEL}")

    yield
    state["client"].close()


app = FastAPI(title="강아지 훈련 RAG 데모", lifespan=lifespan)


@app.get("/api/health")
def health() -> dict:
    return {
        "chunks": state.get("count", 0),
        "embedding_model": state["embedder"].name if "embedder" in state else None,
        "generation_model": cfg.OPENAI_MODEL,
        "openai_configured": state.get("openai") is not None,
    }


def retrieve(question: str, top_k: int) -> list[dict]:
    embedder = state["embedder"]
    vector = embedder.embed_query(question)
    points = state["client"].query_points(
        collection_name=cfg.COLLECTION,
        query=vector,
        limit=cfg.CANDIDATE_K,
        with_payload=True,
    ).points

    seen: set[tuple[str, int]] = set()
    hits: list[dict] = []
    for point in points:
        payload = point.payload or {}
        key = (payload.get("doc_id", ""), payload.get("chunk_index", -1))
        if key in seen:
            continue
        seen.add(key)
        hits.append({**payload, "score": float(point.score)})
        if len(hits) >= top_k:
            break
    return hits


def build_context(hits: list[dict]) -> str:
    blocks = []
    for i, hit in enumerate(hits, start=1):
        blocks.append(
            f"[{i}] 출처: {hit['publisher']} — {hit['title']}\n"
            f"    섹션: {hit['heading']}\n"
            f"    내용:\n{hit['text']}"
        )
    return "\n\n".join(blocks)


@app.post("/api/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    started = time.perf_counter()
    hits = retrieve(req.question, req.top_k)
    retrieval_ms = int((time.perf_counter() - started) * 1000)

    if not hits:
        raise HTTPException(status_code=404, detail="관련 근거를 찾지 못했습니다.")

    client = state.get("openai")
    if client is None:
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY가 설정되지 않았습니다. .env에 키를 넣고 서버를 재시작하세요.",
        )

    user_msg = (
        f"<근거>\n{build_context(hits)}\n</근거>\n\n"
        f"질문: {req.question}\n\n"
        "위 근거만 사용해 한국어로 답하고, 사용한 근거 번호를 [n] 형식으로 표기하세요."
    )

    gen_started = time.perf_counter()
    try:
        completion = client.chat.completions.create(
            model=cfg.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
        )
    except Exception as exc:  # noqa: BLE001 - 사용자에게 원인을 그대로 보여준다
        raise HTTPException(status_code=502, detail=f"OpenAI 호출 실패: {exc}") from exc
    generation_ms = int((time.perf_counter() - gen_started) * 1000)

    return AskResponse(
        answer=completion.choices[0].message.content or "",
        sources=[
            Source(
                n=i,
                title=hit["title"],
                title_ko=hit["title_ko"],
                publisher=hit["publisher"],
                heading=hit["heading"],
                url=hit["url"],
                snippet=hit["text"][:220].strip() + ("..." if len(hit["text"]) > 220 else ""),
                score=round(hit["score"], 4),
            )
            for i, hit in enumerate(hits, start=1)
        ],
        model=cfg.OPENAI_MODEL,
        embedding_model=state["embedder"].name,
        retrieval_ms=retrieval_ms,
        generation_ms=generation_ms,
    )


@app.post("/api/search")
def search_only(req: AskRequest) -> dict:
    """OpenAI 키 없이 검색 단계만 확인할 때 쓰는 엔드포인트."""
    started = time.perf_counter()
    hits = retrieve(req.question, req.top_k)
    return {
        "took_ms": int((time.perf_counter() - started) * 1000),
        "hits": [
            {
                "score": round(h["score"], 4),
                "heading": h["heading"],
                "doc_id": h["doc_id"],
                "text": h["text"][:300],
            }
            for h in hits
        ],
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8100, log_level="info")
