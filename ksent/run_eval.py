# -*- coding: utf-8 -*-
"""한국어 영화리뷰 감성분류 모델 종합 검증.

  파트 A. NSMC held-out 벤치마크 (정확도/AUC/보정/선택적예측/모델간 유의성)
  파트 B. 행동 테스트 스위트 (MFT / INV / DIR / UNC)
  파트 C. 편향·일관성 프로브

사용: python run_eval.py [--n-nsmc 5000] [--models mbert-nsmc,koelectra-gen,kcelectra-ko]
"""
import argparse
import csv
import json
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as M
import perturb as P
import suite_data as S

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)


def log(*a):
    print(*a, flush=True)


def hr(title=""):
    log("\n" + "=" * 78)
    if title:
        log(title)
        log("=" * 78)


# ------------------------------------------------------------------ 데이터
def load_nsmc(path, n=None, seed=0):
    rows = []
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) == 3 and p[1].strip():
                rows.append((p[1], int(p[2])))
    if n and n < len(rows):
        rng = random.Random(seed)
        # 라벨 균형 유지 층화 샘플
        pos = [r for r in rows if r[1] == 1]
        neg = [r for r in rows if r[1] == 0]
        rng.shuffle(pos); rng.shuffle(neg)
        rows = pos[: n // 2] + neg[: n - n // 2]
        rng.shuffle(rows)
    return rows


# ------------------------------------------------------------------ 파트 A
def part_a_nsmc(models, rows, store):
    texts = [r[0] for r in rows]
    y = np.array([r[1] for r in rows])
    hr(f"파트 A. NSMC held-out 벤치마크  (n={len(rows)}, 긍정 {int(y.sum())} / 부정 {int((1-y).sum())})")

    preds, probs = {}, {}
    for m in models:
        t0 = time.time()
        p = m.predict_proba(texts)
        dt = time.time() - t0
        probs[m.name] = p
        preds[m.name] = (p >= 0.5).astype(int)
        log(f"  {m.name}: {dt:.1f}s ({len(texts)/dt:.0f} it/s)")

    log("\n[A-1] 기본 성능")
    log(f"  {'model':<18}{'acc':>8}{'95% CI':>18}{'F1(pos)':>9}{'F1(neg)':>9}{'AUC':>8}{'Brier':>8}")
    for m in models:
        pr = probs[m.name]; pd_ = preds[m.name]
        acc = float((pd_ == y).mean())
        lo, hi = M.bootstrap_ci((pd_ == y).astype(float))
        _, _, f1p = M.prf(y, pd_, 1)
        _, _, f1n = M.prf(y, pd_, 0)
        auc = M.roc_auc(y, pr)
        br = M.brier(y, pr)
        log(f"  {m.name:<18}{acc:>8.4f}{f'[{lo:.4f},{hi:.4f}]':>18}{f1p:>9.4f}{f1n:>9.4f}{auc:>8.4f}{br:>8.4f}")
        store["nsmc"][m.name] = dict(acc=acc, ci=[lo, hi], f1_pos=f1p, f1_neg=f1n, auc=auc, brier=br)

    log("\n[A-2] 보정(calibration) — ECE/MCE, 최적 임계값")
    log(f"  {'model':<18}{'ECE':>8}{'MCE':>8}{'thr*':>8}{'acc@thr*':>10}{'acc@0.5':>10}{'평균확신도':>12}")
    for m in models:
        pr = probs[m.name]
        ece, mce, rows_bin = M.ece_mce(y, pr)
        t, a = M.best_threshold(y, pr)
        conf = float(np.mean(np.maximum(pr, 1 - pr)))
        acc05 = float(((pr >= 0.5).astype(int) == y).mean())
        log(f"  {m.name:<18}{ece:>8.4f}{mce:>8.4f}{t:>8.3f}{a:>10.4f}{acc05:>10.4f}{conf:>12.4f}")
        store["nsmc"][m.name].update(ece=ece, mce=mce, best_thr=t, acc_best_thr=a, mean_conf=conf,
                                     reliability=rows_bin)

    log("\n[A-3] 신뢰도 구간별 실제 정확도 (reliability table)")
    for m in models:
        ece, mce, rows_bin = M.ece_mce(y, probs[m.name])
        cells = " ".join(f"{lo:.2f}-{hi:.2f}:n={n},acc={a:.3f}" if n else f"{lo:.2f}-{hi:.2f}:-"
                         for lo, hi, n, a, c in rows_bin)
        log(f"  {m.name}\n    {cells}")

    log("\n[A-4] 선택적 예측 (확신도 상위 X%만 답할 때의 오류율)")
    log(f"  {'model':<18}{'AURC':>9}{'err@20%':>10}{'err@50%':>10}{'err@80%':>10}{'err@100%':>10}")
    for m in models:
        aurc, pts, _ = M.risk_coverage(y, probs[m.name])
        log(f"  {m.name:<18}{aurc:>9.4f}{pts[0.2]:>10.4f}{pts[0.5]:>10.4f}{pts[0.8]:>10.4f}{pts[1.0]:>10.4f}")
        store["nsmc"][m.name].update(aurc=aurc, risk_coverage=pts)

    log("\n[A-4b] 온도 스케일링으로 보정이 고쳐지는가 (전반 절반으로 T 적합 -> 후반 절반 평가)")
    half = len(y) // 2
    log(f"  {'model':<18}{'T*':>7}{'ECE(전)':>10}{'ECE(후)':>10}{'Brier(전)':>11}{'Brier(후)':>11}{'acc변화':>9}")
    for m in models:
        pr = probs[m.name]
        t, _ = M.fit_temperature(y[:half], pr[:half])
        pe = pr[half:]; ye = y[half:]
        pt = M.apply_temperature(pe, t)
        e0, _, _ = M.ece_mce(ye, pe); e1, _, _ = M.ece_mce(ye, pt)
        a0 = float(((pe >= 0.5).astype(int) == ye).mean())
        a1 = float(((pt >= 0.5).astype(int) == ye).mean())
        log(f"  {m.name:<18}{t:>7.2f}{e0:>10.4f}{e1:>10.4f}{M.brier(ye,pe):>11.4f}{M.brier(ye,pt):>11.4f}{a1-a0:>+9.4f}")
        store["nsmc"][m.name].update(temperature=t, ece_before=e0, ece_after=e1)

    log("\n[A-5] 모델 쌍 비교 — McNemar 정확검정 (쌍체, 양측)")
    names = [m.name for m in models]
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = (preds[names[i]] == y); b = (preds[names[j]] == y)
            b01, b10, p = M.mcnemar_exact(a, b)
            verdict = "유의차 있음" if p < 0.05 else "유의차 없음"
            log(f"  {names[i]:<16} vs {names[j]:<16} only-A={b01:<6} only-B={b10:<6} p={p:.3g}  {verdict}")

    log("\n[A-6] 길이별 정확도")
    buckets = [(0, 10), (10, 20), (20, 35), (35, 60), (60, 10 ** 9)]
    lens = np.array([len(t) for t in texts])
    log("  " + "model".ljust(18) + "".join(f"{f'{lo}-{hi if hi<10**9 else str()}자':>14}" for lo, hi in buckets))
    for m in models:
        c = (preds[m.name] == y)
        cells = ""
        for lo, hi in buckets:
            msk = (lens >= lo) & (lens < hi)
            cells += f"{f'{c[msk].mean():.3f}(n={msk.sum()})':>14}" if msk.sum() else f"{'-':>14}"
        log(f"  {m.name:<18}{cells}")

    log("\n[A-7] 가장 확신하며 틀린 사례 (모델별 상위 5)")
    for m in models:
        pr = probs[m.name]; pd_ = preds[m.name]
        wrong = np.where(pd_ != y)[0]
        conf = np.maximum(pr, 1 - pr)
        top = wrong[np.argsort(-conf[wrong])][:5]
        log(f"  -- {m.name}")
        for i in top:
            log(f"     gold={y[i]} pred={pd_[i]} P(pos)={pr[i]:.3f} | {texts[i][:70]}")

    # NSMC 라벨 노이즈 추정: 파인튜닝된 트랜스포머 전부가 골드와 반대로 예측한 사례
    core = [m for m in models if m.name not in ("lexicon-baseline", "ensemble-mean", "tfidf-lr")]
    if len(core) >= 3:
        allwrong = np.ones(len(y), dtype=bool)
        for m in core:
            allwrong &= (preds[m.name] != y)
        idx = np.where(allwrong)[0]
        log(f"\n[A-8] 전 모델이 동일하게 골드와 반대로 예측: {len(idx)}건 ({len(idx)/len(y)*100:.2f}%)")
        log("      -> 상당수는 NSMC 라벨 노이즈일 가능성. 샘플 8건:")
        for i in idx[:8]:
            log(f"     gold={y[i]} | {texts[i][:74]}")
        store["nsmc_all_models_wrong"] = dict(count=int(len(idx)), rate=float(len(idx) / len(y)),
                                              samples=[texts[i] for i in idx[:30]],
                                              golds=[int(y[i]) for i in idx[:30]])
    store["_nsmc_probs"] = {k: v.tolist() for k, v in probs.items()}
    store["_nsmc_gold"] = y.tolist()
    return probs


# ------------------------------------------------------------------ 파트 B
def part_b_mft(models, store, case_rows):
    hr("파트 B-1. MFT — 능력별 최소기능 테스트")
    caps = sorted(S.MFT)
    all_texts, meta = [], []
    for cap in caps:
        for t, y, d in S.MFT[cap]:
            all_texts.append(t); meta.append((cap, y, d))

    res = {}
    for m in models:
        pr = m.predict_proba(all_texts)
        res[m.name] = pr
        for (cap, y, d), t, p in zip(meta, all_texts, pr):
            case_rows.append(dict(model=m.name, part="MFT", capability=cap, text=t,
                                  gold=y, p_pos=round(float(p), 4),
                                  pred=int(p >= 0.5), correct=int((p >= 0.5) == y), difficulty=d))

    log("  ('다수'=해당 소분류에서 다수 라벨만 찍었을 때의 정확도. 이 값을 못 넘으면 무의미)")
    hdr = f"  {'capability':<28}{'n':>4}{'다수':>6}"
    for m in models:
        hdr += f"{m.name[:14]:>16}"
    log(hdr)
    for cap in caps:
        idx = [i for i, mm in enumerate(meta) if mm[0] == cap]
        g = np.array([meta[i][1] for i in idx])
        maj = max(g.mean(), 1 - g.mean())
        line = f"  {cap:<28}{len(idx):>4}{maj:>6.2f}"
        for m in models:
            pr = res[m.name][idx]
            gold = np.array([meta[i][1] for i in idx])
            c = ((pr >= 0.5).astype(int) == gold)
            lo, hi = M.wilson_ci(int(c.sum()), len(idx))
            line += f"{f'{c.mean():.2f}[{lo:.2f}-{hi:.2f}]':>16}"
            store["mft"].setdefault(m.name, {})[cap] = dict(
                n=len(idx), acc=float(c.mean()), wilson=[lo, hi])
        log(line)
    _g = np.array([x[1] for x in meta])
    line = f"  {'** 전체 **':<26}{len(meta):>4}{max(_g.mean(), 1-_g.mean()):>6.2f}"
    for m in models:
        gold = np.array([x[1] for x in meta])
        c = ((res[m.name] >= 0.5).astype(int) == gold)
        lo, hi = M.wilson_ci(int(c.sum()), len(c))
        line += f"{f'{c.mean():.2f}[{lo:.2f}-{hi:.2f}]':>16}"
        store["mft"].setdefault(m.name, {})["_overall"] = dict(n=len(c), acc=float(c.mean()), wilson=[lo, hi])
    log(line)

    log("\n[B-1b] 난이도별 정확도 (1=사전적 / 2=보통 / 3=추론 필요)")
    for m in models:
        gold = np.array([x[1] for x in meta]); diff = np.array([x[2] for x in meta])
        c = ((res[m.name] >= 0.5).astype(int) == gold)
        cells = "".join(f"  d{d}: {c[diff==d].mean():.3f}(n={int((diff==d).sum())})" for d in (1, 2, 3))
        log(f"  {m.name:<18}{cells}")

    log("\n[B-1c] 파인튜닝 트랜스포머 전부가 실패한 케이스 (베이스라인/앙상블 제외)")
    gold = np.array([x[1] for x in meta])
    core = [m for m in models if m.name not in ("lexicon-baseline", "ensemble-mean", "tfidf-lr")]
    common = np.ones(len(meta), dtype=bool)
    for m in core:
        common &= ((res[m.name] >= 0.5).astype(int) != gold)
    for i in np.where(common)[0]:
        ps = " ".join(f"{m.name[:10]}={res[m.name][i]:.2f}" for m in models)
        log(f"  [{meta[i][0]}] gold={meta[i][1]} | {all_texts[i][:56]} | {ps}")
    store["mft_common_fail"] = [dict(cap=meta[i][0], gold=int(meta[i][1]), text=all_texts[i])
                                for i in np.where(common)[0]]
    return res, meta, all_texts


def part_b_unc(models, store, case_rows):
    hr("파트 B-2. UNC — 의견 없는 중립 문장에서의 과신 여부")
    log("  (정답 라벨 없음. 이상적 모델이라면 P(pos)가 0.5 근처여야 함)")
    for m in models:
        pr = m.predict_proba(S.NEUTRAL)
        conf = np.maximum(pr, 1 - pr)
        log(f"  {m.name:<18} 평균확신도={conf.mean():.3f}  확신도>0.9 비율={float((conf>0.9).mean()):.2f}  "
            f"P(pos) 범위=[{pr.min():.2f},{pr.max():.2f}]  긍정판정={int((pr>=0.5).sum())}/{len(pr)}")
        store["unc"][m.name] = dict(mean_conf=float(conf.mean()),
                                    frac_over_90=float((conf > 0.9).mean()),
                                    p_min=float(pr.min()), p_max=float(pr.max()))
        for t, p in zip(S.NEUTRAL, pr):
            case_rows.append(dict(model=m.name, part="UNC", capability="neutral", text=t,
                                  gold="", p_pos=round(float(p), 4), pred=int(p >= 0.5),
                                  correct="", difficulty=""))
    log("\n  문장별 P(pos):")
    for i, t in enumerate(S.NEUTRAL):
        vals = " ".join(f"{m.name[:10]}={m.predict_proba([t])[0]:.2f}" for m in models)
        log(f"    {t[:44]:<46} {vals}")


def part_b_dir(models, store, case_rows, margin=0.02):
    hr(f"파트 B-3. DIR — 방향성 기대 테스트 (유효 변화폭 > {margin})")
    bases = [d[1] for d in S.DIR_PAIRS]
    vars_ = [d[2] for d in S.DIR_PAIRS]
    for m in models:
        pb = m.predict_proba(bases); pv = m.predict_proba(vars_)
        ok = 0
        log(f"  -- {m.name}")
        for (name, b, v, d), x, yv in zip(S.DIR_PAIRS, pb, pv):
            delta = yv - x
            passed = (delta > margin) if d == "up" else (delta < -margin)
            ok += passed
            log(f"     {'PASS' if passed else 'FAIL'} {name:<18} {d:<5} "
                f"{x:.3f} -> {yv:.3f} (Δ={delta:+.3f})")
            case_rows.append(dict(model=m.name, part="DIR", capability=name, text=f"{b} => {v}",
                                  gold=d, p_pos=round(float(yv), 4), pred="",
                                  correct=int(passed), difficulty=""))
        lo, hi = M.wilson_ci(ok, len(S.DIR_PAIRS))
        log(f"     통과율 {ok}/{len(S.DIR_PAIRS)} = {ok/len(S.DIR_PAIRS):.2f} [{lo:.2f}-{hi:.2f}]")
        store["dir"][m.name] = dict(passed=int(ok), n=len(S.DIR_PAIRS), rate=ok / len(S.DIR_PAIRS))

    hr("파트 B-4. 강도 사다리 — P(pos) 단조성")
    for key, seq in S.LADDERS.items():
        want = "증가" if key == "pos_ladder" else "감소"
        log(f"  [{key}] 기대: 아래로 갈수록 P(pos) {want}")
        for m in models:
            pr = m.predict_proba(seq)
            diffs = np.diff(pr)
            mono = bool(np.all(diffs > 0)) if key == "pos_ladder" else bool(np.all(diffs < 0))
            log(f"    {m.name:<18} " + " -> ".join(f"{p:.3f}" for p in pr) +
                f"   단조성 {'OK' if mono else 'X'}")
            store["ladder"].setdefault(m.name, {})[key] = dict(probs=[float(x) for x in pr], monotonic=mono)
        for t in seq:
            pass
        log("")


def part_b_inv(models, store, case_rows, mft_texts, mft_meta, seed=7):
    hr("파트 B-5. INV — 변형 불변성 (예측 라벨이 바뀌면 실패)")
    log("  의미보존군: despace / punct_strip / emoji_append / pad_context")
    log("  오타군    : drop_jongseong / swap_adjacent  (감성어는 보호했으나 드물게 의미가 흔들릴 수 있음)")
    rng = random.Random(seed)
    variants = {}
    for pname, fn in P.PERTURBATIONS.items():
        r = random.Random(seed)
        variants[pname] = [fn(t, r) for t in mft_texts]

    log(f"\n  {'model':<18}{'perturbation':<18}{'flip율':>9}{'|Δp| 평균':>11}{'|Δp| p95':>10}")
    for m in models:
        base = m.predict_proba(mft_texts)
        base_lab = (base >= 0.5).astype(int)
        for pname in P.PERTURBATIONS:
            same = [i for i, (a, b) in enumerate(zip(mft_texts, variants[pname])) if a != b]
            if not same:
                continue
            pv = m.predict_proba([variants[pname][i] for i in same])
            lab = (pv >= 0.5).astype(int)
            flips = (lab != base_lab[same])
            dp = np.abs(pv - base[same])
            log(f"  {m.name:<18}{pname:<18}{f'{flips.mean():.3f}({int(flips.sum())}/{len(same)})':>9}"
                f"{dp.mean():>11.3f}{np.quantile(dp,0.95):>10.3f}")
            store["inv"].setdefault(m.name, {})[pname] = dict(
                flip_rate=float(flips.mean()), n=len(same),
                mean_abs_delta=float(dp.mean()), p95_abs_delta=float(np.quantile(dp, 0.95)))
            for k, i in enumerate(same):
                if flips[k]:
                    case_rows.append(dict(model=m.name, part="INV", capability=pname,
                                          text=f"{mft_texts[i]} => {variants[pname][i]}",
                                          gold=mft_meta[i][1], p_pos=round(float(pv[k]), 4),
                                          pred=int(lab[k]), correct=0, difficulty=""))

    log("\n[B-5b] 수동 작성 문체 변형 쌍 (구어체/맞춤법오류/띄어쓰기/이모지)")
    a = [x[0] for x in S.INV_MANUAL]; b = [x[1] for x in S.INV_MANUAL]
    for m in models:
        pa = m.predict_proba(a); pb = m.predict_proba(b)
        flip = ((pa >= 0.5).astype(int) != (pb >= 0.5).astype(int))
        log(f"  {m.name:<18} flip {int(flip.sum())}/{len(a)}  평균|Δp|={np.abs(pa-pb).mean():.3f}")
        for i in np.where(flip)[0]:
            log(f"      FLIP: '{a[i]}'({pa[i]:.2f}) -> '{b[i]}'({pb[i]:.2f})")
        store["inv_manual"][m.name] = dict(flips=int(flip.sum()), n=len(a),
                                           mean_abs_delta=float(np.abs(pa - pb).mean()))


# ------------------------------------------------------------------ 파트 C
def part_c_bias(models, store, case_rows):
    hr("파트 C-1. 개체명 치환 편향 — 같은 문장, 배우 이름만 교체")
    for tmpl, gold in S.ENTITY_TEMPLATES:
        log(f"  템플릿: {tmpl}  (기대 라벨 {gold})")
        texts = [tmpl.format(e) for e in S.ENTITIES]
        for m in models:
            pr = m.predict_proba(texts)
            acc = float(((pr >= 0.5).astype(int) == gold).mean())
            order = np.argsort(pr)
            log(f"    {m.name:<18} 정확도={acc:.2f}  P(pos) 범위={pr.min():.3f}~{pr.max():.3f} "
                f"(폭 {pr.max()-pr.min():.3f}, sd {pr.std():.3f})")
            log(f"        최저 {S.ENTITIES[order[0]]}={pr[order[0]]:.3f} / "
                f"최고 {S.ENTITIES[order[-1]]}={pr[order[-1]]:.3f}")
            store["entity_bias"].setdefault(m.name, {})[tmpl] = dict(
                acc=acc, spread=float(pr.max() - pr.min()), sd=float(pr.std()),
                per_entity={e: float(p) for e, p in zip(S.ENTITIES, pr)})
            for e, p in zip(S.ENTITIES, pr):
                case_rows.append(dict(model=m.name, part="BIAS-entity", capability=tmpl, text=e,
                                      gold=gold, p_pos=round(float(p), 4), pred=int(p >= 0.5),
                                      correct=int((p >= 0.5) == gold), difficulty=""))
        log("")

    hr("파트 C-2. 장르어 confound — 같은 문장, 장르만 교체")
    for tmpl, gold in S.GENRE_TEMPLATES:
        log(f"  템플릿: {tmpl}  (기대 라벨 {gold})")
        texts = [tmpl.format(g) for g in S.GENRES]
        for m in models:
            pr = m.predict_proba(texts)
            acc = float(((pr >= 0.5).astype(int) == gold).mean())
            order = np.argsort(pr)
            log(f"    {m.name:<18} 정확도={acc:.2f}  폭={pr.max()-pr.min():.3f}  "
                f"최저 {S.GENRES[order[0]]}={pr[order[0]]:.3f} / 최고 {S.GENRES[order[-1]]}={pr[order[-1]]:.3f}")
            store["genre_bias"].setdefault(m.name, {})[tmpl] = dict(
                acc=acc, spread=float(pr.max() - pr.min()),
                per_genre={g: float(p) for g, p in zip(S.GENRES, pr)})
            for g, p in zip(S.GENRES, pr):
                case_rows.append(dict(model=m.name, part="BIAS-genre", capability=tmpl, text=g,
                                      gold=gold, p_pos=round(float(p), 4), pred=int(p >= 0.5),
                                      correct=int((p >= 0.5) == gold), difficulty=""))
        log("")

    hr("파트 C-3. 패러프레이즈 일관성 — 같은 뜻, 다른 표현")
    for m in models:
        log(f"  -- {m.name}")
        for gi, (gold, group) in enumerate(S.PARAPHRASE_GROUPS):
            pr = m.predict_proba(group)
            lab = (pr >= 0.5).astype(int)
            consistent = len(set(lab.tolist())) == 1
            log(f"     group{gi}(gold={gold}) 일관={'O' if consistent else 'X'} "
                f"acc={float((lab==gold).mean()):.2f} P={' '.join(f'{p:.2f}' for p in pr)}")
            store["paraphrase"].setdefault(m.name, {})[f"group{gi}"] = dict(
                gold=gold, consistent=consistent, acc=float((lab == gold).mean()),
                probs=[float(p) for p in pr])


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-nsmc", type=int, default=5000)
    ap.add_argument("--models", default="mbert-nsmc,koelectra-gen,kcelectra-ko")
    ap.add_argument("--no-lexicon", action="store_true")
    ap.add_argument("--no-tfidf", action="store_true")
    ap.add_argument("--skip-nsmc", action="store_true")
    args = ap.parse_args()

    import predictor as PR
    hr("모델 로딩 및 극성 방향 실측 검증")
    log(f"  device = {PR.DEVICE}")
    names = [x for x in args.models.split(",") if x]
    models = PR.load_all(names, include_lexicon=not args.no_lexicon,
                         include_tfidf=not args.no_tfidf)
    for m in models:
        flag = "" if getattr(m, "polarity_ok", True) else "   <-- 경고: 극성 분리 약함"
        log(f"  {m.name:<18} {m.hub_id}")
        log(f"      declared id2label = {m.declared_id2label}")
        log(f"      실측: {m.polarity_note}{flag}")

    store = dict(nsmc={}, mft={}, unc={}, dir={}, ladder={}, inv={}, inv_manual={},
                 entity_bias={}, genre_bias={}, paraphrase={},
                 meta=dict(device=PR.DEVICE, models={m.name: m.hub_id for m in models},
                           n_nsmc=args.n_nsmc))
    case_rows = []

    if not args.skip_nsmc:
        rows = load_nsmc("data/ratings_test.txt", n=args.n_nsmc)
        part_a_nsmc(models, rows, store)

    mft_res, mft_meta, mft_texts = part_b_mft(models, store, case_rows)
    part_b_unc(models, store, case_rows)
    part_b_dir(models, store, case_rows)
    part_b_inv(models, store, case_rows, mft_texts, mft_meta)
    part_c_bias(models, store, case_rows)

    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=False, indent=1, default=float)
    with open(os.path.join(OUT, "cases.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "part", "capability", "text", "gold",
                                          "p_pos", "pred", "correct", "difficulty"])
        w.writeheader(); w.writerows(case_rows)
    hr("저장 완료")
    log(f"  {OUT}/results.json   ({len(case_rows)} case rows -> cases.csv)")


if __name__ == "__main__":
    main()
