# -*- coding: utf-8 -*-
"""통계 유틸. scipy 없이 동작하도록 순수 numpy/math로 구현."""
import math
import numpy as np


def wilson_ci(k, n, z=1.96):
    """이항 비율의 Wilson score 신뢰구간. n이 작은 소분류 셀에서 정확도가 중요."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_ci(correct, n_boot=5000, seed=0, alpha=0.05):
    """정확도의 부트스트랩 백분위 신뢰구간."""
    correct = np.asarray(correct, dtype=float)
    n = len(correct)
    if n == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    accs = correct[idx].mean(axis=1)
    return (float(np.quantile(accs, alpha / 2)), float(np.quantile(accs, 1 - alpha / 2)))


def mcnemar_exact(a_correct, b_correct):
    """두 모델의 쌍체 비교. 정확 이항검정(양측).
    반환: (b01, b10, p_value)  b01 = A만 맞음, b10 = B만 맞음"""
    a = np.asarray(a_correct, dtype=bool)
    b = np.asarray(b_correct, dtype=bool)
    b01 = int(np.sum(a & ~b))
    b10 = int(np.sum(~a & b))
    n = b01 + b10
    if n == 0:
        return b01, b10, 1.0
    k = min(b01, b10)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return b01, b10, min(1.0, 2 * tail)


def roc_auc(y_true, score):
    """랭크 기반 AUC (동점 처리 포함)."""
    y = np.asarray(y_true)
    s = np.asarray(score, dtype=float)
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        ranks[order[i:j + 1]] = avg
        i = j + 1
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def brier(y_true, p):
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y_true, dtype=float)) ** 2))


def ece_mce(y_true, p, n_bins=10):
    """Expected / Maximum Calibration Error.
    확신도 = max(p, 1-p), 정답 = argmax 예측이 맞았는지."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y_true)
    conf = np.maximum(p, 1 - p)
    pred = (p >= 0.5).astype(int)
    acc = (pred == y).astype(float)
    bins = np.linspace(0.5, 1.0, n_bins + 1)
    ece, mce, rows = 0.0, 0.0, []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if m.sum() == 0:
            rows.append((float(lo), float(hi), 0, float("nan"), float("nan")))
            continue
        a, c = acc[m].mean(), conf[m].mean()
        gap = abs(a - c)
        ece += m.sum() / len(p) * gap
        mce = max(mce, gap)
        rows.append((float(lo), float(hi), int(m.sum()), float(a), float(c)))
    return float(ece), float(mce), rows


def risk_coverage(y_true, p):
    """선택적 예측: 확신도 순으로 정렬해 커버리지별 오류율. AURC 반환."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y_true)
    conf = np.maximum(p, 1 - p)
    pred = (p >= 0.5).astype(int)
    err = (pred != y).astype(float)
    order = np.argsort(-conf, kind="mergesort")
    err_sorted = err[order]
    cum_err = np.cumsum(err_sorted) / np.arange(1, len(err) + 1)
    cov = np.arange(1, len(err) + 1) / len(err)
    aurc = float(np.mean(cum_err))
    pts = {}
    for target in (0.2, 0.5, 0.8, 1.0):
        i = min(len(cov) - 1, int(round(target * len(cov))) - 1)
        pts[target] = float(cum_err[i])
    return aurc, pts, (cov, cum_err)


def fit_temperature(y_true, p, grid=None):
    """이진 확률에 대한 온도 스케일링. logit을 T로 나눠 NLL을 최소화하는 T를 격자탐색."""
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(y_true, dtype=float)
    z = np.log(p / (1 - p))
    grid = grid if grid is not None else np.concatenate([np.arange(0.2, 5.01, 0.02)])
    best_t, best_nll = 1.0, None
    for t in grid:
        q = 1 / (1 + np.exp(-z / t))
        q = np.clip(q, 1e-9, 1 - 1e-9)
        nll = -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))
        if best_nll is None or nll < best_nll:
            best_t, best_nll = float(t), float(nll)
    return best_t, best_nll


def apply_temperature(p, t):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p))
    return 1 / (1 + np.exp(-z / t))


def prf(y_true, y_pred, positive=1):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = int(np.sum((y_pred == positive) & (y_true == positive)))
    fp = int(np.sum((y_pred == positive) & (y_true != positive)))
    fn = int(np.sum((y_pred != positive) & (y_true == positive)))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def best_threshold(y_true, p):
    """정확도를 최대화하는 임계값 탐색 (기본 0.5가 최적인지 확인용)."""
    y = np.asarray(y_true)
    p = np.asarray(p, dtype=float)
    cands = np.unique(np.concatenate([[0.0, 0.5, 1.0], p]))
    best_t, best_a = 0.5, ((p >= 0.5).astype(int) == y).mean()
    for t in cands:
        a = ((p >= t).astype(int) == y).mean()
        if a > best_a:
            best_t, best_a = float(t), float(a)
    return best_t, float(best_a)
