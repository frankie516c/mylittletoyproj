# -*- coding: utf-8 -*-
"""텍스트 감성 + 표정 평균 감정을 합쳐 최종 리뷰 판정을 만든다."""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import FACE_WEIGHT, TEXT_WEIGHT


@dataclass
class Verdict:
    score: float = 0.5                # 최종 P(positive) 0~1
    label: str = "판정 불가"
    tone: str = "unknown"             # positive | neutral | negative | unknown
    confidence: float = 0.0
    text_positive: float | None = None
    face_positive: float | None = None
    weights: dict[str, float] = field(default_factory=dict)
    agreement: str = ""
    notes: list[str] = field(default_factory=list)


def _label(score: float) -> tuple[str, str]:
    if score >= 0.60:
        return "긍정", "positive"
    if score <= 0.40:
        return "부정", "negative"
    return "중립", "neutral"


def combine(text_result, face_result) -> Verdict:
    v = Verdict()

    text_ok = text_result.ok
    face_ok = face_result.ok

    w_text = TEXT_WEIGHT if text_ok else 0.0
    # 얼굴은 '몇 프레임에서 실제로 검출됐는지'(커버리지)에 비례해 신뢰한다.
    coverage = (face_result.faces_found / face_result.frames_scanned
                if face_result.frames_scanned else 0.0)
    w_face = FACE_WEIGHT * coverage if face_ok else 0.0

    if w_text + w_face <= 0:
        v.notes.append("음성과 얼굴 어느 쪽에서도 분석할 신호를 찾지 못했습니다.")
        if text_result.error:
            v.notes.append(text_result.error)
        if face_result.error:
            v.notes.append(face_result.error)
        return v

    if text_ok:
        v.text_positive = text_result.positive
    if face_ok:
        v.face_positive = face_result.positive_score

    total = w_text + w_face
    score = ((w_text * (v.text_positive or 0.0)) +
             (w_face * (v.face_positive or 0.0))) / total

    v.score = float(score)
    v.label, v.tone = _label(v.score)
    v.weights = {"text": round(w_text / total, 3), "face": round(w_face / total, 3)}

    # 확신도: 0.5 에서 얼마나 떨어져 있는지 + 두 모달리티가 얼마나 일치하는지
    margin = abs(v.score - 0.5) * 2
    if text_ok and face_ok:
        gap = abs(v.text_positive - v.face_positive)
        same_side = (v.text_positive - 0.5) * (v.face_positive - 0.5) >= 0
        v.agreement = "일치" if same_side else "불일치"
        if not same_side:
            v.notes.append(
                f"말의 내용({v.text_positive:.0%})과 표정({v.face_positive:.0%})의 "
                "방향이 다릅니다. 반어·억지 미소·무표정 낭독일 수 있습니다."
            )
        v.confidence = float(max(0.0, margin * (1.0 - gap)))
    else:
        v.agreement = "단일 모달"
        only = "음성 텍스트" if text_ok else "표정"
        v.notes.append(f"{only}만으로 판정했습니다.")
        v.confidence = float(margin * 0.7)

    if face_ok and coverage < 0.5:
        v.notes.append(
            f"얼굴이 {face_result.frames_scanned}장 중 {face_result.faces_found}장에서만 "
            "잡혀 표정 가중치를 낮췄습니다."
        )
    if text_result.error and not text_ok:
        v.notes.append(text_result.error)
    if face_result.error and not face_ok:
        v.notes.append(face_result.error)

    return v
