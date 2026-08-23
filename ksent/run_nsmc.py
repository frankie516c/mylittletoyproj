# -*- coding: utf-8 -*-
"""NSMC held-out 벤치마크 (파트 A) — 모델별 예측을 개별로 캐싱한 뒤 통계를 낸다.

3개 트랜스포머를 한 프로세스에 동시 상주시켰을 때 CPU 추론 중 프로세스가
조용히 죽는 현상이 있었으므로, 모델을 하나씩 로드/예측/해제하고
결과를 .npy로 저장한다. 중단돼도 이어서 돌릴 수 있다.

  python run_nsmc.py predict <model>   # 한 모델 예측 -> results/nsmc_<model>.npy
  python run_nsmc.py report            # 캐시된 예측으로 파트 A 통계 전부 출력
"""
import gc
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as M
from run_eval import load_nsmc

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
N = int(os.environ.get("KSENT_N", "5000"))
ORDER = ["mbert-nsmc", "koelectra-gen", "kcelectra-ko", "tfidf-lr", "lexicon-baseline"]


def rows():
    return load_nsmc("data/ratings_test.txt", n=N)


def path(name):
    return os.path.join(OUT, f"nsmc_{name}_{N}.npy")


def predict(name):
    import predictor as PR
    r = rows()
    texts = [x[0] for x in r]
    if name == "tfidf-lr":
        from tfidf_baseline import TfidfBaseline
        m = TfidfBaseline().fit()
    elif name == "lexicon-baseline":
        m = PR.LexiconBaseline()
    else:
        m = PR.SentimentModel(name, PR.MODELS[name], batch_size=32)
        m.calibrate_polarity()
        print(f"  polarity: {m.polarity_note}", flush=True)
    import time
    t0 = time.time()
    out = np.empty(len(texts), dtype=np.float64)
    for i in range(0, len(texts), 500):
        out[i:i + 500] = m.predict_proba(texts[i:i + 500])
        print(f"  {min(i+500,len(texts))}/{len(texts)}  {time.time()-t0:.0f}s", flush=True)
    np.save(path(name), out)
    del m
    gc.collect()
    print(f"saved {path(name)}  ({time.time()-t0:.0f}s, {len(texts)/(time.time()-t0):.0f} it/s)")


def log(*a):
    print(*a, flush=True)


def report():
    r = rows()
    texts = [x[0] for x in r]
    y = np.array([x[1] for x in r])
    probs = {}
    for n in ORDER:
        if os.path.exists(path(n)):
            probs[n] = np.load(path(n))
    if not probs:
        print("캐시된 예측이 없습니다. 먼저 predict를 돌리세요.")
        return
    # 트랜스포머 평균 앙상블
    tf = [n for n in ("mbert-nsmc", "koelectra-gen", "kcelectra-ko") if n in probs]
    if len(tf) > 1:
        probs["ensemble-mean"] = np.mean([probs[n] for n in tf], axis=0)
    names = [n for n in ORDER if n in probs]
    if "ensemble-mean" in probs:
        names.insert(3, "ensemble-mean")
    preds = {n: (probs[n] >= 0.5).astype(int) for n in names}
    store = {}

    log("=" * 82)
    log(f"파트 A. NSMC held-out 벤치마크  (n={len(y)}, 긍정 {int(y.sum())} / 부정 {int((1-y).sum())})")
    log("=" * 82)

    log("\n[A-1] 기본 성능")
    log(f"  {'model':<18}{'acc':>8}{'95% CI':>18}{'F1(pos)':>9}{'F1(neg)':>9}{'AUC':>8}{'Brier':>8}")
    for n in names:
        p, pd_ = probs[n], preds[n]
        acc = float((pd_ == y).mean())
        lo, hi = M.bootstrap_ci((pd_ == y).astype(float))
        _, _, f1p = M.prf(y, pd_, 1)
        _, _, f1n = M.prf(y, pd_, 0)
        log(f"  {n:<18}{acc:>8.4f}{f'[{lo:.4f},{hi:.4f}]':>18}{f1p:>9.4f}{f1n:>9.4f}"
            f"{M.roc_auc(y,p):>8.4f}{M.brier(y,p):>8.4f}")
        store[n] = dict(acc=acc, ci=[lo, hi], f1_pos=f1p, f1_neg=f1n,
                        auc=M.roc_auc(y, p), brier=M.brier(y, p))

    log("\n[A-2] 보정 — ECE/MCE, 최적 임계값, 평균 확신도")
    log(f"  {'model':<18}{'ECE':>8}{'MCE':>8}{'thr*':>8}{'acc@thr*':>10}{'acc@0.5':>10}{'평균확신도':>11}")
    for n in names:
        p = probs[n]
        ece, mce, rows_bin = M.ece_mce(y, p)
        t, a = M.best_threshold(y, p)
        conf = float(np.mean(np.maximum(p, 1 - p)))
        log(f"  {n:<18}{ece:>8.4f}{mce:>8.4f}{t:>8.3f}{a:>10.4f}"
            f"{float(((p>=0.5).astype(int)==y).mean()):>10.4f}{conf:>11.4f}")
        store[n].update(ece=ece, mce=mce, best_thr=t, acc_best_thr=a, mean_conf=conf,
                        reliability=rows_bin)

    log("\n[A-3] 신뢰도 구간별 실제 정확도")
    for n in names:
        _, _, rb = M.ece_mce(y, probs[n])
        log(f"  {n}")
        log("    " + " ".join(f"{lo:.2f}-{hi:.2f}:n={c},acc={a:.3f}" if c else f"{lo:.2f}-{hi:.2f}:-"
                              for lo, hi, c, a, _ in rb))

    log("\n[A-4] 선택적 예측 — 확신도 상위 X%만 답할 때 오류율")
    log(f"  {'model':<18}{'AURC':>9}{'err@20%':>10}{'err@50%':>10}{'err@80%':>10}{'err@100%':>10}")
    for n in names:
        aurc, pts, _ = M.risk_coverage(y, probs[n])
        log(f"  {n:<18}{aurc:>9.4f}{pts[0.2]:>10.4f}{pts[0.5]:>10.4f}{pts[0.8]:>10.4f}{pts[1.0]:>10.4f}")
        store[n].update(aurc=aurc, risk_coverage={str(k): v for k, v in pts.items()})

    log("\n[A-5] 온도 스케일링 (앞 절반으로 T 적합 -> 뒤 절반 평가)")
    half = len(y) // 2
    log(f"  {'model':<18}{'T*':>7}{'ECE전':>9}{'ECE후':>9}{'Brier전':>10}{'Brier후':>10}{'acc변화':>9}")
    for n in names:
        p = probs[n]
        t, _ = M.fit_temperature(y[:half], p[:half])
        pe, ye = p[half:], y[half:]
        pt = M.apply_temperature(pe, t)
        e0, _, _ = M.ece_mce(ye, pe)
        e1, _, _ = M.ece_mce(ye, pt)
        a0 = float(((pe >= 0.5).astype(int) == ye).mean())
        a1 = float(((pt >= 0.5).astype(int) == ye).mean())
        log(f"  {n:<18}{t:>7.2f}{e0:>9.4f}{e1:>9.4f}{M.brier(ye,pe):>10.4f}{M.brier(ye,pt):>10.4f}{a1-a0:>+9.4f}")
        store[n].update(temperature=t, ece_before=e0, ece_after=e1)

    log("\n[A-6] 모델 쌍 비교 — McNemar 정확검정 (쌍체, 양측)")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = (preds[names[i]] == y), (preds[names[j]] == y)
            b01, b10, pv = M.mcnemar_exact(a, b)
            log(f"  {names[i]:<16} vs {names[j]:<16} onlyA={b01:<5} onlyB={b10:<5} "
                f"p={pv:.3g}  {'유의차 O' if pv < 0.05 else '유의차 X'}")

    log("\n[A-7] 길이별 정확도")
    buckets = [(0, 10), (10, 20), (20, 35), (35, 60), (60, 10 ** 9)]
    lens = np.array([len(t) for t in texts])
    log("  " + "model".ljust(18) + "".join(
        f"{(str(lo)+'-'+(str(hi) if hi < 10**9 else '')+'자'):>15}" for lo, hi in buckets))
    for n in names:
        c = (preds[n] == y)
        cells = ""
        for lo, hi in buckets:
            m_ = (lens >= lo) & (lens < hi)
            cells += f"{f'{c[m_].mean():.3f}(n={int(m_.sum())})':>15}" if m_.sum() else f"{'-':>15}"
        log(f"  {n:<18}{cells}")

    log("\n[A-8] 가장 확신하며 틀린 사례 (모델별 상위 5)")
    for n in names:
        p, pd_ = probs[n], preds[n]
        wrong = np.where(pd_ != y)[0]
        conf = np.maximum(p, 1 - p)
        for i in wrong[np.argsort(-conf[wrong])][:5]:
            log(f"  [{n:<14}] gold={y[i]} P(pos)={p[i]:.3f} | {texts[i][:66]}")

    if len(tf) >= 3:
        allwrong = np.ones(len(y), dtype=bool)
        for n in tf:
            allwrong &= (preds[n] != y)
        idx = np.where(allwrong)[0]
        log(f"\n[A-9] 파인튜닝 트랜스포머 3종이 모두 골드와 반대로 예측: "
            f"{len(idx)}건 ({len(idx)/len(y)*100:.2f}%)")
        log("      -> NSMC 골드 라벨 노이즈의 상한 추정치. 샘플 12건:")
        for i in idx[:12]:
            log(f"     gold={y[i]} | {texts[i][:70]}")
        store["_all_wrong"] = dict(count=int(len(idx)), rate=float(len(idx) / len(y)),
                                   samples=[texts[i] for i in idx[:40]],
                                   golds=[int(y[i]) for i in idx[:40]])

        log("\n[A-10] 모델 간 예측 일치도 (쌍별 라벨 일치 비율)")
        for i in range(len(tf)):
            for j in range(i + 1, len(tf)):
                ag = float((preds[tf[i]] == preds[tf[j]]).mean())
                log(f"  {tf[i]:<16} vs {tf[j]:<16} 일치 {ag:.4f}")

    with open(os.path.join(OUT, f"nsmc_report_{N}.json"), "w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=False, indent=1, default=float)
    log(f"\n저장: results/nsmc_report_{N}.json")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    if cmd == "predict":
        predict(sys.argv[2])
    else:
        report()
