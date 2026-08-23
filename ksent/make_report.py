# -*- coding: utf-8 -*-
"""results.json -> 요약 마크다운. run_eval.py 실행 후 사용."""
import json
import os
import sys

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def main():
    with open(os.path.join(R, "results.json"), encoding="utf-8") as f:
        d = json.load(f)
    out = []
    A = out.append

    models = list(d["meta"]["models"]) + [k for k in d["mft"] if k not in d["meta"]["models"]]
    models = list(dict.fromkeys(list(d["mft"].keys())))

    A("# 한국어 영화리뷰 감성분류 검증 리포트\n")
    A(f"- device: `{d['meta']['device']}` / NSMC 표본 n={d['meta']['n_nsmc']}")
    for k, v in d["meta"]["models"].items():
        A(f"- `{k}` = {v}")
    A("")

    if d.get("nsmc"):
        A("## 1. NSMC held-out 성능\n")
        A("| model | acc | 95% CI | F1(pos) | F1(neg) | AUC | Brier | ECE | T* | AURC |")
        A("|---|---|---|---|---|---|---|---|---|---|")
        for m, v in d["nsmc"].items():
            A(f"| {m} | {v['acc']:.4f} | [{v['ci'][0]:.4f}, {v['ci'][1]:.4f}] | {v['f1_pos']:.4f} | "
              f"{v['f1_neg']:.4f} | {v['auc']:.4f} | {v['brier']:.4f} | {v['ece']:.4f} | "
              f"{v.get('temperature', float('nan')):.2f} | {v.get('aurc', float('nan')):.4f} |")
        A("")
        if "nsmc_all_models_wrong" in d:
            w = d["nsmc_all_models_wrong"]
            A(f"트랜스포머 전 모델이 골드와 반대로 예측: **{w['count']}건 ({w['rate']*100:.2f}%)** "
              f"— 라벨 노이즈 상한 추정치.\n")

    A("## 2. 행동 테스트 (MFT) — 능력별 정확도\n")
    caps = sorted({c for v in d["mft"].values() for c in v if c != "_overall"})
    A("| capability | n | " + " | ".join(models) + " |")
    A("|---" * (len(models) + 2) + "|")
    for c in caps:
        n = next((d["mft"][m][c]["n"] for m in models if c in d["mft"][m]), "-")
        cells = []
        for m in models:
            v = d["mft"][m].get(c)
            cells.append(f"{v['acc']:.2f} [{v['wilson'][0]:.2f}-{v['wilson'][1]:.2f}]" if v else "-")
        A(f"| {c} | {n} | " + " | ".join(cells) + " |")
    cells = [f"**{d['mft'][m]['_overall']['acc']:.2f}**" for m in models]
    A("| **전체** | " + str(d["mft"][models[0]]["_overall"]["n"]) + " | " + " | ".join(cells) + " |")
    A("")

    A("## 3. 강건성 / 일관성\n")
    A("| model | INV flip율(평균) | DIR 통과율 | 중립문 평균확신도 | 개체명 P폭(최대) |")
    A("|---|---|---|---|---|")
    for m in models:
        inv = d["inv"].get(m, {})
        fl = sum(v["flip_rate"] for v in inv.values()) / len(inv) if inv else float("nan")
        dr = d["dir"].get(m, {}).get("rate", float("nan"))
        un = d["unc"].get(m, {}).get("mean_conf", float("nan"))
        eb = d["entity_bias"].get(m, {})
        sp = max((v["spread"] for v in eb.values()), default=float("nan"))
        A(f"| {m} | {fl:.3f} | {dr:.2f} | {un:.3f} | {sp:.3f} |")
    A("")

    if d.get("mft_common_fail"):
        A("## 4. 트랜스포머 전 모델 공통 실패\n")
        for x in d["mft_common_fail"]:
            A(f"- `{x['cap']}` (gold={x['gold']}) {x['text']}")
        A("")

    p = os.path.join(R, "REPORT.md")
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print("wrote", p)


if __name__ == "__main__":
    main()
