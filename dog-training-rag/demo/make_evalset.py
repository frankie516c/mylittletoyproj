"""청크를 무작위로 뽑아 질문을 만들고 검색 평가셋을 구성한다.

    python demo/make_evalset.py            # 기본 10개
    python demo/make_evalset.py -n 20      # 개수 변경
    python demo/make_evalset.py --seed 7   # 표본 변경

각 항목은 "이 질문의 정답 청크는 이것" 이라는 한 쌍이다. 검색이 그 청크를 다시 찾아오는지로
hit rate / MRR을 잰다. 질문 생성에만 OpenAI를 쓰고, 평가 자체는 검색만으로 한다.

주의: 이렇게 만든 합성 평가셋은 상한이 아니라 하한 점검용이다. 청크에서 파생된 질문이라
실제 사용자 질문보다 쉬운 쪽으로 치우친다. 사람 검수를 거친 질문으로 대체해 나가야 한다.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as cfg

EVAL_DIR = cfg.REPO_ROOT / "data" / "demo" / "eval"

PROMPT = """다음은 반려견 훈련 자료에서 잘라낸 한 조각입니다.

주제: {topic_ko}
문서: {title}
섹션: {heading}
---
{text}
---

이 조각을 읽어야만 답할 수 있는 **한국어 질문 하나**를 만드세요.

조건:
- 실제 반려인이 검색창에 칠 법한 자연스러운 구어체 질문. 존댓말.
- 조각의 문장을 그대로 베끼지 마세요. 영어 단어를 그대로 옮기지 말고 한국어로 바꾸세요.
- "이 문서에 따르면" 같은 메타 표현 금지.
- 조각에 실제로 답이 있는 것만 물으세요.
- 한 문장, 60자 이내.
- 질문만 출력하세요. 따옴표나 번호를 붙이지 마세요."""


def load_chunks() -> list[dict]:
    if not cfg.CHUNK_FILE.exists():
        raise SystemExit(f"{cfg.CHUNK_FILE} 가 없습니다. 먼저 demo/ingest.py 를 실행하세요.")
    with cfg.CHUNK_FILE.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg.load_dotenv()
    import os

    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("OPENAI_API_KEY 가 없습니다.")
    client = OpenAI(api_key=api_key)

    chunks = load_chunks()
    # 너무 짧아 질문거리가 안 되는 청크는 후보에서 뺀다
    pool = [c for c in chunks if c["char_len"] >= 300]
    print(f"[pool] 전체 {len(chunks)}개 중 300자 이상 {len(pool)}개에서 표본 추출")

    rng = random.Random(args.seed)
    sample = rng.sample(pool, min(args.count, len(pool)))

    records = []
    for i, chunk in enumerate(sample, start=1):
        completion = client.chat.completions.create(
            model=cfg.OPENAI_MODEL,
            messages=[{
                "role": "user",
                "content": PROMPT.format(
                    topic_ko=chunk["topic_ko"],
                    title=chunk["title"],
                    heading=chunk["heading"],
                    text=chunk["text"],
                ),
            }],
            temperature=0.4,
        )
        question = (completion.choices[0].message.content or "").strip().strip('"')
        records.append({
            "qid": f"q{i:02d}",
            "question": question,
            "gold_doc_id": chunk["doc_id"],
            "gold_chunk_index": chunk["chunk_index"],
            "gold_topic_ko": chunk["topic_ko"],
            "gold_heading": chunk["heading"],
            "gold_text": chunk["text"],
        })
        print(f"[{i:2}/{len(sample)}] {chunk['doc_id']}#{chunk['chunk_index']}  {question}")

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else EVAL_DIR / "golden_v1.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    meta = {
        "count": len(records),
        "seed": args.seed,
        "generator_model": cfg.OPENAI_MODEL,
        "source_chunks": str(cfg.CHUNK_FILE.relative_to(cfg.REPO_ROOT)),
        "note": "청크에서 역생성한 합성 질문. 실사용 질문보다 쉬운 쪽으로 편향됨.",
    }
    (EVAL_DIR / "golden_v1.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n[save] {len(records)}개 → {out.relative_to(cfg.REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
