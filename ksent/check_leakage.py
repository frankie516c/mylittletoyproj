# -*- coding: utf-8 -*-
"""수제 테스트 문장이 NSMC train/test에 이미 존재하는지 검사.
모델 3종이 모두 NSMC로 학습됐으므로, 스위트가 학습 데이터와 겹치면
'행동 테스트'가 아니라 '암기 테스트'가 되어버린다."""
import re
import sys
from collections import Counter

sys.path.insert(0, ".")
import suite_data as s


def norm(t):
    return re.sub(r"\s+", "", t).strip()


def char_ngrams(t, n=5):
    t = norm(t)
    return {t[i:i + n] for i in range(max(0, len(t) - n + 1))}


def load(path):
    out = []
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) == 3:
                out.append(p[1])
    return out


def suite_texts():
    ts = []
    for cap, cases in s.MFT.items():
        for t, y, d in cases:
            ts.append((f"MFT/{cap}", t))
    for t in s.NEUTRAL:
        ts.append(("UNC", t))
    for name, b, v, d in s.DIR_PAIRS:
        ts += [("DIR", b), ("DIR", v)]
    for k, seq in s.LADDERS.items():
        ts += [("LADDER", t) for t in seq]
    for a, b in s.INV_MANUAL:
        ts += [("INV", a), ("INV", b)]
    for y, g in s.PARAPHRASE_GROUPS:
        ts += [("PARA", t) for t in g]
    return ts


def main():
    train = load("data/ratings_train.txt")
    test = load("data/ratings_test.txt")
    train_set = {norm(t) for t in train}
    test_set = {norm(t) for t in test}

    ts = suite_texts()
    print(f"suite texts: {len(ts)}  | NSMC train {len(train)} / test {len(test)}")

    exact_tr = [(c, t) for c, t in ts if norm(t) in train_set]
    exact_te = [(c, t) for c, t in ts if norm(t) in test_set]
    print(f"\n[정확 일치] train {len(exact_tr)} / test {len(exact_te)}")
    for c, t in exact_tr[:20]:
        print("   TRAIN", c, "|", t)
    for c, t in exact_te[:20]:
        print("   TEST ", c, "|", t)

    # 5-gram 자카드 기준 근접 중복 (짧은 문장 위주라 보수적으로 확인)
    print("\n[근접 중복 상위] (5-gram Jaccard >= 0.6)")
    train_idx = {}
    for t in train:
        for g in char_ngrams(t):
            train_idx.setdefault(g, []).append(t)
    hits = 0
    for c, t in ts:
        gs = char_ngrams(t)
        if not gs:
            continue
        cand = Counter()
        for g in gs:
            for o in train_idx.get(g, ())[:50]:
                cand[o] += 1
        for o, k in cand.most_common(3):
            og = char_ngrams(o)
            j = k / len(gs | og)
            if j >= 0.6:
                print(f"   J={j:.2f} {c} | {t}  <->  {o}")
                hits += 1
                break
    print(f"   총 {hits}건")


if __name__ == "__main__":
    main()
