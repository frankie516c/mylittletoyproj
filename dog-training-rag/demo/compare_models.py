"""임베딩 모델 후보를 같은 평가셋으로 비교한다.

    python demo/compare_models.py

Qdrant를 거치지 않고 메모리에서 코사인 랭킹을 계산한다. 재색인 없이 모델만 갈아끼워
hit rate / MRR / 점수 분포를 나란히 본다. 점수 폭(1위-20위)이 좁으면 그 모델은
이 코퍼스에서 순위를 매기지 못하고 있다는 뜻이다.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as cfg
from fastembed import TextEmbedding
from fastembed.common.model_description import ModelSource, PoolingType

E5_REPO = "intfloat/multilingual-e5-small"

CANDIDATES = [
    # (표시 이름, fastembed 모델명, hf 리포|None, onnx 파일|None, 질의 접두사, 문서 접두사)
    ("e5-small int8", "e5s-int8", E5_REPO, "onnx/model_qint8_avx512_vnni.onnx", "query: ", "passage: "),
    ("e5-small O4", "e5s-o4", E5_REPO, "onnx/model_O4.onnx", "query: ", "passage: "),
    ("e5-small int8 (접두사 없음)", "e5s-int8", E5_REPO, "onnx/model_qint8_avx512_vnni.onnx", "", ""),
    ("MiniLM-L12 multilingual", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", None, None, "", ""),
]


def register(name: str, repo: str, model_file: str) -> None:
    if name in {m["model"] for m in TextEmbedding.list_supported_models()}:
        return
    TextEmbedding.add_custom_model(
        model=name,
        pooling=PoolingType.MEAN,
        normalization=True,
        sources=ModelSource(hf=repo),
        dim=384,
        model_file=model_file,
    )


def load_data() -> tuple[list[dict], list[dict]]:
    chunks = [json.loads(l) for l in cfg.CHUNK_FILE.open(encoding="utf-8") if l.strip()]
    eval_path = cfg.REPO_ROOT / "data" / "demo" / "eval" / "golden_v1.jsonl"
    evalset = [json.loads(l) for l in eval_path.open(encoding="utf-8") if l.strip()]
    return chunks, evalset


def evaluate(model: TextEmbedding, chunks: list[dict], evalset: list[dict],
             qp: str, pp: str) -> dict:
    doc_vecs = np.array([v for v in model.embed([pp + c["embed_text"] for c in chunks])])
    doc_vecs /= np.linalg.norm(doc_vecs, axis=1, keepdims=True)

    q_vecs = np.array([v for v in model.embed([qp + it["question"] for it in evalset])])
    q_vecs /= np.linalg.norm(q_vecs, axis=1, keepdims=True)

    sims = q_vecs @ doc_vecs.T
    ranks, spreads, doc_hits = [], [], []
    for i, item in enumerate(evalset):
        order = np.argsort(-sims[i])
        gold = next(
            j for j, c in enumerate(chunks)
            if c["doc_id"] == item["gold_doc_id"] and c["chunk_index"] == item["gold_chunk_index"]
        )
        rank = int(np.where(order == gold)[0][0]) + 1
        ranks.append(rank)
        spreads.append(float(sims[i][order[0]] - sims[i][order[19]]))
        doc_hits.append(chunks[order[0]]["doc_id"] == item["gold_doc_id"])

    n = len(ranks)
    return {
        "hit@1": sum(r <= 1 for r in ranks) / n,
        "hit@3": sum(r <= 3 for r in ranks) / n,
        "hit@5": sum(r <= 5 for r in ranks) / n,
        "hit@10": sum(r <= 10 for r in ranks) / n,
        "mrr": sum(1 / r if r <= 10 else 0 for r in ranks) / n,
        "doc_hit@1": sum(doc_hits) / n,
        "spread": sum(spreads) / n,
        "median_rank": sorted(ranks)[n // 2],
    }


def main() -> int:
    chunks, evalset = load_data()
    print(f"청크 {len(chunks)}개 / 질문 {len(evalset)}개\n")

    header = f"{'모델':<28} {'hit@1':>6} {'hit@3':>6} {'hit@5':>6} {'hit@10':>7} {'MRR':>6} {'문서hit@1':>9} {'점수폭':>7} {'중앙순위':>8}"
    print(header)
    print("-" * len(header))

    results = {}
    for label, name, repo, model_file, qp, pp in CANDIDATES:
        try:
            if repo:
                register(name, repo, model_file)
            model = TextEmbedding(model_name=name)
            r = evaluate(model, chunks, evalset, qp, pp)
        except Exception as exc:  # noqa: BLE001
            print(f"{label:<28} 실패: {type(exc).__name__}: {str(exc)[:60]}")
            continue
        results[label] = r
        print(f"{label:<28} {r['hit@1']:>6.2f} {r['hit@3']:>6.2f} {r['hit@5']:>6.2f} "
              f"{r['hit@10']:>7.2f} {r['mrr']:>6.3f} {r['doc_hit@1']:>9.2f} "
              f"{r['spread']:>7.4f} {r['median_rank']:>8}")

    out = cfg.REPO_ROOT / "data" / "demo" / "eval" / "model_comparison.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[save] {out.relative_to(cfg.REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
