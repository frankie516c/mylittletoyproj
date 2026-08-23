"""데모 파이프라인 공통 설정."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = REPO_ROOT / "data" / "demo" / "raw"
CHUNK_FILE = REPO_ROOT / "data" / "demo" / "chunks.jsonl"
QDRANT_PATH = REPO_ROOT / "data" / "demo" / "qdrant"
COLLECTION = "dog_training_demo_v1"

# 작으면서 한국어가 되는 순서대로. fastembed가 지원하는 첫 모델을 쓴다.
EMBEDDING_CANDIDATES = [
    "intfloat/multilingual-e5-small",          # 118M / 384d — 1순위
    "intfloat/multilingual-e5-base",           # 278M / 768d
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",  # 118M / 384d
    "BAAI/bge-small-en-v1.5",                  # 최후 폴백(영어 전용)
]

# e5 계열은 접두사를 요구한다. 다른 모델이면 빈 문자열로 대체된다.
E5_QUERY_PREFIX = "query: "
E5_PASSAGE_PREFIX = "passage: "

# 청킹 파라미터 (한국어는 1토큰당 대략 1.5~2자라 문자 기준으로 잡는다)
MAX_CHARS = 1100
MIN_CHARS = 280
OVERLAP_CHARS = 160

TOP_K = 5
CANDIDATE_K = 20

OPENAI_MODEL = os.getenv("DEMO_OPENAI_MODEL", "gpt-4o-mini")


def load_dotenv(path: Path | None = None) -> None:
    """의존성 없이 .env를 읽어 os.environ에 채운다 (기존 값은 덮지 않는다)."""
    env_path = path or (REPO_ROOT / ".env")
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value and not os.environ.get(key):
            os.environ[key] = value
