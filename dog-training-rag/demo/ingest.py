"""수집 → 마크다운 변환 → 구조 인식 청킹 → 임베딩 → 로컬 벡터 DB 적재.

    python demo/ingest.py                # 전체 실행
    python demo/ingest.py --skip-fetch   # 이미 받아둔 마크다운 재사용
    python demo/ingest.py --dry-run      # 청킹까지만 하고 임베딩은 생략
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as cfg
from chunker import chunk_markdown
from fetcher import fetch_markdown, frontmatter, strip_frontmatter
from sources import SOURCES

META_KEYS = (
    "doc_id", "url", "title", "title_ko", "publisher",
    "evidence_tier", "lang", "topic", "topic_ko",
)


def collect_markdown(skip_fetch: bool) -> list[tuple[dict, str]]:
    """소스를 마크다운으로 확보한다. 개별 실패는 건너뛰고 끝에 요약한다."""
    cfg.RAW_DIR.mkdir(parents=True, exist_ok=True)
    out: list[tuple[dict, str]] = []
    failed: list[tuple[str, str]] = []

    for source in SOURCES:
        path = cfg.RAW_DIR / f"{source['doc_id']}.md"
        if skip_fetch and path.exists():
            body = strip_frontmatter(path.read_text(encoding="utf-8"))
            print(f"[fetch] 캐시  {source['doc_id']:26} {len(body):>7,}자")
            out.append((source, body))
            continue
        try:
            body = fetch_markdown(source["url"])
        except Exception as exc:  # noqa: BLE001 - 한 소스 실패로 전체를 멈추지 않는다
            print(f"[fetch] 실패  {source['doc_id']:26} {type(exc).__name__}")
            failed.append((source["doc_id"], str(exc)))
            continue
        path.write_text(frontmatter(source) + body, encoding="utf-8")
        print(f"[fetch] 수집  {source['doc_id']:26} {len(body):>7,}자")
        out.append((source, body))

    if failed:
        print(f"[fetch] {len(failed)}개 소스를 가져오지 못했습니다: {', '.join(d for d, _ in failed)}")
    if not out:
        raise SystemExit("가져온 문서가 하나도 없습니다.")
    return out


def build_chunks(docs: list[tuple[dict, str]]) -> list[dict]:
    records: list[dict] = []
    for source, body in docs:
        chunks = chunk_markdown(
            body,
            source["title"],
            max_chars=cfg.MAX_CHARS,
            min_chars=cfg.MIN_CHARS,
            overlap_chars=cfg.OVERLAP_CHARS,
        )
        sizes = sorted(c.char_len for c in chunks)
        print(
            f"[chunk] {source['doc_id']:26} {len(chunks):>3}개  "
            f"최소 {sizes[0]:>4} / 중앙 {sizes[len(sizes) // 2]:>4} / 최대 {sizes[-1]:>4}자"
        )
        base = {k: source[k] for k in META_KEYS}
        for chunk in chunks:
            records.append(
                {
                    **base,
                    "chunk_index": chunk.index,
                    "part": chunk.part,
                    "parts": chunk.parts,
                    "heading_path": chunk.heading_path,
                    "heading": " > ".join(chunk.heading_path),
                    "text": chunk.text,
                    "embed_text": chunk.embed_text,
                    "char_len": chunk.char_len,
                }
            )
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-fetch", action="store_true", help="저장된 마크다운 재사용")
    parser.add_argument("--dry-run", action="store_true", help="임베딩/적재 생략")
    args = parser.parse_args()

    docs = collect_markdown(args.skip_fetch)
    records = build_chunks(docs)

    cfg.CHUNK_FILE.parent.mkdir(parents=True, exist_ok=True)
    with cfg.CHUNK_FILE.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[chunk] 총 {len(records)}개 → {cfg.CHUNK_FILE.relative_to(cfg.REPO_ROOT)}")

    if args.dry_run:
        print("[dry-run] 임베딩/적재를 건너뜁니다.")
        return 0

    from embedder import load_embedder
    from store import open_client, recreate_collection, upsert

    embedder = load_embedder()
    print(f"[embed] {len(records)}개 청크 임베딩 중...")
    vectors = embedder.embed_passages([r["embed_text"] for r in records])
    print(f"[embed] 완료 (차원 {len(vectors[0])})")

    client = open_client(cfg.QDRANT_PATH)
    recreate_collection(client, cfg.COLLECTION, embedder.dim)
    payloads = [{k: v for k, v in r.items() if k != "embed_text"} for r in records]
    upsert(client, cfg.COLLECTION, vectors, payloads)
    count = client.count(cfg.COLLECTION).count
    client.close()

    topics: dict[str, int] = {}
    for record in records:
        topics[record["topic_ko"]] = topics.get(record["topic_ko"], 0) + 1

    meta = {
        "collection": cfg.COLLECTION,
        "embedding_model": embedder.name,
        "dim": embedder.dim,
        "chunks": count,
        "docs": [s["doc_id"] for s, _ in docs],
        "topics": topics,
        "chunking": {
            "strategy": "structure-aware markdown (heading tree + block packing)",
            "max_chars": cfg.MAX_CHARS,
            "min_chars": cfg.MIN_CHARS,
            "overlap_chars": cfg.OVERLAP_CHARS,
        },
    }
    (cfg.QDRANT_PATH.parent / "index_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[store] {count}개 포인트 적재 완료 → {cfg.QDRANT_PATH.relative_to(cfg.REPO_ROOT)}")
    print(f"[store] 주제별 청크: {topics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
