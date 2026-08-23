# -*- coding: utf-8 -*-
"""PyAV 기반 미디어 디코딩.

ffmpeg 실행 파일 없이 동영상에서
  - 16kHz mono float32 오디오 (faster-whisper 입력 규격)
  - 시간축으로 고르게 분포된 N장의 RGB 프레임
을 뽑아낸다. mp4 뿐 아니라 브라우저 MediaRecorder 가 만드는 webm 도 지원한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import av
import numpy as np

from .config import NUM_FRAMES, SAMPLE_RATE


@dataclass
class Sample:
    """추출한 프레임 1장."""
    index: int
    timestamp: float          # 초
    image: np.ndarray         # HxWx3 RGB uint8


@dataclass
class MediaBundle:
    audio: np.ndarray | None = None       # float32 mono 16kHz, [-1, 1]
    audio_duration: float = 0.0
    frames: list[Sample] = field(default_factory=list)
    video_duration: float = 0.0
    errors: list[str] = field(default_factory=list)


def _decode_audio(container) -> tuple[np.ndarray | None, float]:
    """오디오 스트림을 16kHz mono float32 로 리샘플해 하나로 잇는다."""
    stream = next((s for s in container.streams if s.type == "audio"), None)
    if stream is None:
        return None, 0.0

    resampler = av.audio.resampler.AudioResampler(
        format="s16", layout="mono", rate=SAMPLE_RATE
    )
    chunks: list[np.ndarray] = []

    def _collect(out):
        # PyAV 버전에 따라 단일 프레임 또는 리스트를 돌려준다.
        if out is None:
            return
        for f in (out if isinstance(out, list) else [out]):
            if f is not None:
                chunks.append(f.to_ndarray().reshape(-1))

    for frame in container.decode(stream):
        _collect(resampler.resample(frame))
    _collect(resampler.resample(None))          # flush

    if not chunks:
        return None, 0.0

    pcm = np.concatenate(chunks).astype(np.float32) / 32768.0
    return pcm, len(pcm) / SAMPLE_RATE


def _decode_frames(container, n_frames: int) -> tuple[list[Sample], float]:
    """시간축으로 고르게 n_frames 장을 남긴다.

    컨테이너가 총 프레임 수/재생시간을 알려주지 않는 경우(webm 등)가 흔하므로
    총 길이를 모른 채로도 동작하는 '스트라이드 배증 데시메이션'을 쓴다.
    버퍼가 2N 을 넘으면 짝수 번째만 남기고 스트라이드를 2배로 올린다.
    결과적으로 메모리는 O(N), 최종 표본은 시간축에 균등하게 남는다.
    """
    stream = next((s for s in container.streams if s.type == "video"), None)
    if stream is None:
        return [], 0.0
    stream.thread_type = "AUTO"

    time_base = float(stream.time_base) if stream.time_base else 0.0
    kept: list[Sample] = []
    stride = 1
    seen = 0
    last_ts = 0.0

    for frame in container.decode(stream):
        ts = float(frame.pts * time_base) if (frame.pts is not None and time_base) else last_ts
        last_ts = max(last_ts, ts)
        if seen % stride == 0:
            kept.append(Sample(index=seen, timestamp=ts,
                               image=frame.to_ndarray(format="rgb24")))
            if len(kept) > 2 * n_frames:
                kept = kept[::2]
                stride *= 2
        seen += 1

    if len(kept) > n_frames:
        # 남은 것 중 시간축 균등하게 n_frames 장 선택
        idx = np.linspace(0, len(kept) - 1, n_frames).round().astype(int)
        kept = [kept[i] for i in dict.fromkeys(idx.tolist())]

    return kept, last_ts


def extract(path: str, n_frames: int = NUM_FRAMES) -> MediaBundle:
    """동영상 파일에서 오디오와 대표 프레임을 뽑는다."""
    bundle = MediaBundle()

    # 오디오와 비디오는 각각 디코딩한다(하나의 순회로는 두 스트림 위치가 엉킨다).
    try:
        with av.open(path) as container:
            bundle.audio, bundle.audio_duration = _decode_audio(container)
    except Exception as exc:                      # noqa: BLE001 - 데모: 사유만 표시
        bundle.errors.append(f"오디오 디코딩 실패: {exc}")

    try:
        with av.open(path) as container:
            bundle.frames, bundle.video_duration = _decode_frames(container, n_frames)
    except Exception as exc:                      # noqa: BLE001
        bundle.errors.append(f"영상 디코딩 실패: {exc}")

    return bundle
