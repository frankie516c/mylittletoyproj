# -*- coding: utf-8 -*-
"""faster-whisper 로 음성을 텍스트로 옮긴다 (지연 로딩 + 프로세스 내 캐시)."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

import numpy as np

from .config import (
    SAMPLE_RATE,
    WHISPER_COMPUTE,
    WHISPER_DEVICE,
    WHISPER_LANG,
    WHISPER_MODEL,
)

_model = None
_lock = threading.Lock()


def get_model():
    """최초 호출 시 모델을 내려받아 캐시한다 (small ≈ 460MB)."""
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                from faster_whisper import WhisperModel
                _model = WhisperModel(
                    WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE
                )
    return _model


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class Transcript:
    text: str = ""
    language: str = ""
    language_probability: float = 0.0
    duration: float = 0.0
    segments: list[Segment] = field(default_factory=list)
    error: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.text.strip())


def transcribe(audio: np.ndarray | None) -> Transcript:
    """16kHz mono float32 배열을 받아 인식 결과를 돌려준다."""
    if audio is None or len(audio) < SAMPLE_RATE * 0.2:
        return Transcript(error="음성 트랙이 없거나 너무 짧습니다.")

    try:
        model = get_model()
        segments, info = model.transcribe(
            audio.astype(np.float32),
            language=WHISPER_LANG or None,
            beam_size=5,
            vad_filter=True,                     # 무음 구간 제거 → 환청(hallucination) 억제
        )
        segs = [Segment(float(s.start), float(s.end), s.text.strip())
                for s in segments if s.text.strip()]
    except Exception as exc:                     # noqa: BLE001
        return Transcript(error=f"STT 실패: {exc}")

    return Transcript(
        text=" ".join(s.text for s in segs).strip(),
        language=getattr(info, "language", "") or "",
        language_probability=float(getattr(info, "language_probability", 0.0) or 0.0),
        duration=float(getattr(info, "duration", 0.0) or 0.0),
        segments=segs,
    )
