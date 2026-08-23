"""검색 단계만 평가한다 — hit rate와 MRR.

    python demo/evaluate.py
    python demo/evaluate.py -k 10

생성(OpenAI)은 전혀 호출하지 않는다. 질문을 임베딩해 Qdrant에서 top-k를 가져오고,
정답 청크가 몇 등으로 돌아왔는지만 본다.

- hit rate@k : 정답 청크가 상위 k 안에 들어온 질문의 비율
- MRR@k      : 정답 청크 순위의 역수 평균 (1등이면 1.0, 3등이면 0.333, 밖이면 0)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as cfg

EVAL_DIR = cfg.REPO_ROOT / "data" / "demo" / "eval"


def load_evalset(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"{path} 가 없습니다. 먼저 demo/make_evalset.py 를 실행하세요.")
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", "--top-k", type=int, default=10, help="평가할 최대 순위")
    parser.add_argument("--evalset", default=None)
    args = parser.parse_args()

    from embedder import load_embedder
    from store import open_client

    evalset_path = Path(args.evalset) if args.evalset else EVAL_DIR / "golden_v1.jsonl"
    evalset = load_evalset(evalset_path)

    embedder = load_embedder()
    client = open_client(cfg.QDRANT_PATH)
    total_chunks = client.count(cfg.COLLECTION).count

    rows = []
    for item in evalset:
        points = client.query_points(
            collection_name=cfg.COLLECTION,
            query=embedder.embed_query(item["question"]),
            limit=args.top_k,
            with_payload=True,
        ).points

        rank = None
        for i, point in enumerate(points, start=1):
            payload = point.payload or {}
            if (payload.get("doc_id") == item["gold_doc_id"]
                    and payload.get("chunk_index") == item["gold_chunk_index"]):
                rank = i
                break

        top = points[0].payload if points else {}
        rows.append({
            "qid": item["qid"],
            "question": item["question"],
            "gold": f"{item['gold_doc_id']}#{item['gold_chunk_index']}",
            "gold_topic": item["gold_topic_ko"],
            "rank": rank,
            "rr": 1.0 / rank if rank else 0.0,
            "top1": f"{top.get('doc_id')}#{top.get('chunk_index')}" if top else None,
            "top1_score": round(float(points[0].score), 4) if points else None,
            "top1_topic": top.get("topic_ko") if top else None,
        })
    client.close()

    n = len(rows)
    cutoffs = [c for c in (1, 3, 5, 10) if c <= args.top_k]
    hit = {c: sum(1 for r in rows if r["rank"] and r["rank"] <= c) / n for c in cutoffs}
    mrr = sum(r["rr"] for r in rows) / n

    print(f"\n{'=' * 78}")
    print(f"검색 평가 — 질문 {n}개 / 코퍼스 {total_chunks}개 청크 / {embedder.name}")
    print("=" * 78)
    print(f"{'qid':<5} {'순위':>4}  {'RR':>5}  {'정답 청크':<28} 질문")
    print("-" * 78)
    for row in rows:
        rank = str(row["rank"]) if row["rank"] else "밖"
        print(f"{row['qid']:<5} {rank:>4}  {row['rr']:>5.3f}  {row['gold']:<28} {row['question'][:36]}")
    print("-" * 78)
    for c in cutoffs:
        print(f"  hit rate@{c:<2} = {hit[c]:.3f}  ({int(hit[c] * n)}/{n})")
    print(f"  MRR@{args.top_k:<7} = {mrr:.3f}")
    print("=" * 78)

    missed = [r for r in rows if not r["rank"]]
    if missed:
        print(f"\n상위 {args.top_k}위 안에 못 들어온 질문 {len(missed)}개:")
        for row in missed:
            print(f"  [{row['qid']}] {row['question']}")
            print(f"        정답 {row['gold']} ({row['gold_topic']})")
            print(f"        1위  {row['top1']} ({row['top1_topic']}, {row['top1_score']})")

    result = {
        "evalset": str(evalset_path.relative_to(cfg.REPO_ROOT)),
        "questions": n,
        "corpus_chunks": total_chunks,
        "embedding_model": embedder.name,
        "top_k": args.top_k,
        "hit_rate": {f"@{c}": round(hit[c], 4) for c in cutoffs},
        "mrr": round(mrr, 4),
        "per_question": rows,
    }
    out = EVAL_DIR / "retrieval_results_v1.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[save] {out.relative_to(cfg.REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
