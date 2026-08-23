# -*- coding: utf-8 -*-
"""한국어 리뷰 텍스트 감성 분석.

허브 모델의 id2label 을 그대로 믿지 않고, 논란의 여지가 없는 프로브 문장으로
극성 방향(어느 인덱스가 positive 인지)을 실측 검증한 뒤 사용한다.
(ksent/predictor.py 와 같은 방식)
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

import numpy as np

from .config import TEXT_MODEL_ID

POLARITY_PROBES_POS = [
    "정말 재미있고 훌륭한 제품입니다. 강력 추천합니다.",
    "최고였어요. 너무 좋았습니다.",
    "완벽합니다. 아주 만족스러워요.",
]
POLARITY_PROBES_NEG = [
    "정말 지루하고 최악입니다. 절대 사지 마세요.",
    "돈이 아까운 쓰레기였어요. 시간 낭비.",
    "형편없고 별로입니다. 최악이에요.",
]

_bundle = None                       # (tokenizer, model, pos_index, note)
_lock = threading.Lock()


def _load():
    global _bundle
    if _bundle is None:
        with _lock:
            if _bundle is None:
                import torch
                from transformers import (AutoModelForSequenceClassification,
                                          AutoTokenizer)
                tok = AutoTokenizer.from_pretrained(TEXT_MODEL_ID)
                model = AutoModelForSequenceClassification.from_pretrained(TEXT_MODEL_ID)
                model.eval()
                torch.set_grad_enabled(False)

                def probs(texts):
                    enc = tok(list(texts), padding=True, truncation=True,
                              max_length=256, return_tensors="pt")
                    return torch.softmax(model(**enc).logits.float(), dim=-1).cpu().numpy()

                gap = probs(POLARITY_PROBES_POS).mean(0) - probs(POLARITY_PROBES_NEG).mean(0)
                pos_index = int(np.argmax(gap))
                note = (f"pos_index={pos_index} "
                        f"(declared={model.config.id2label.get(pos_index)}), "
                        f"gap={float(gap[pos_index]):.3f}")
                _bundle = (tok, model, pos_index, note)
    return _bundle


def load():
    """모델 예열용 공개 진입점."""
    return _load()


@dataclass
class TextResult:
    text: str = ""
    positive: float = 0.5             # P(positive)
    label: str = ""
    chunks: list[dict] = field(default_factory=list)
    calibration: str = ""
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error and bool(self.text.strip())


def _split(text: str, max_chars: int = 220) -> list[str]:
    """긴 리뷰는 문장 단위로 잘라 각각 예측한 뒤 평균낸다."""
    import re
    sents = [s.strip() for s in re.split(r"(?<=[.!?。？！])\s+|\n+", text) if s.strip()]
    if not sents:
        return [text.strip()] if text.strip() else []

    out, cur = [], ""
    for s in sents:
        if len(cur) + len(s) + 1 <= max_chars:
            cur = f"{cur} {s}".strip()
        else:
            if cur:
                out.append(cur)
            cur = s[:max_chars]
    if cur:
        out.append(cur)
    return out


def analyze(text: str) -> TextResult:
    text = (text or "").strip()
    if not text:
        return TextResult(error="인식된 텍스트가 없습니다.")

    try:
        import torch
        tok, model, pos_index, note = _load()
    except Exception as exc:                     # noqa: BLE001
        return TextResult(text=text, error=f"텍스트 감성 모델 로딩 실패: {exc}")

    parts = _split(text)
    try:
        enc = tok(parts, padding=True, truncation=True, max_length=256,
                  return_tensors="pt")
        p = torch.softmax(model(**enc).logits.float(), dim=-1).cpu().numpy()[:, pos_index]
    except Exception as exc:                     # noqa: BLE001
        return TextResult(text=text, error=f"텍스트 감성 분석 실패: {exc}")

    # 길이 가중 평균 — 짧은 감탄사가 긴 본문을 흔들지 않도록.
    weights = np.array([max(len(s), 1) for s in parts], dtype=np.float64)
    positive = float((p * weights).sum() / weights.sum())

    return TextResult(
        text=text,
        positive=positive,
        label="긍정" if positive >= 0.5 else "부정",
        chunks=[{"text": s, "positive": float(v)} for s, v in zip(parts, p)],
        calibration=note,
    )
