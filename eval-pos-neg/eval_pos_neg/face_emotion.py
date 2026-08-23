# -*- coding: utf-8 -*-
"""프레임에서 얼굴을 찾아 표정(감정)을 분류하고 평균 감정을 만든다.

- 얼굴 검출: OpenCV YuNet (ONNX, 약 230KB). OpenCV 5 는 haarcascade 를 번들하지
  않으므로 최초 실행 시 모델 파일을 내려받는다.
- 표정 분류: HuggingFace ViT (기본 trpakov/vit-face-expression, FER 7클래스).
"""

from __future__ import annotations

import base64
import threading
import urllib.request
from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import (
    EMOTION_MODEL_ID,
    FACE_NMS_THRESHOLD,
    FACE_SCORE_THRESHOLD,
    YUNET_PATH,
    YUNET_URL,
)

# 감정 → 정서가(valence). -1(부정) ~ +1(긍정).
# surprise 는 긍/부정 양쪽에 걸치므로 약한 양수만 준다.
VALENCE = {
    "happy": 1.0,
    "surprise": 0.2,
    "neutral": 0.0,
    "sad": -0.7,
    "fear": -0.7,
    "disgust": -0.9,
    "angry": -1.0,
}

EMOTION_KO = {
    "happy": "행복", "surprise": "놀람", "neutral": "무표정",
    "sad": "슬픔", "fear": "두려움", "disgust": "혐오", "angry": "분노",
}

_detector = None
_detector_lock = threading.Lock()
_emotion = None                     # (processor, model, id2label)
_emotion_lock = threading.Lock()


# --------------------------------------------------------------------------- #
# 모델 로딩
# --------------------------------------------------------------------------- #
def _ensure_yunet() -> str:
    if not YUNET_PATH.exists():
        YUNET_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(YUNET_URL, YUNET_PATH)
    return str(YUNET_PATH)


def get_detector():
    global _detector
    if _detector is None:
        with _detector_lock:
            if _detector is None:
                _detector = cv2.FaceDetectorYN.create(
                    _ensure_yunet(), "", (320, 320),
                    FACE_SCORE_THRESHOLD, FACE_NMS_THRESHOLD, 5000,
                )
    return _detector


def get_emotion_model():
    """최초 호출 시 HF 허브에서 표정 분류기를 내려받아 캐시한다."""
    global _emotion
    if _emotion is None:
        with _emotion_lock:
            if _emotion is None:
                import torch
                from transformers import (AutoImageProcessor,
                                          AutoModelForImageClassification)
                processor = AutoImageProcessor.from_pretrained(EMOTION_MODEL_ID)
                model = AutoModelForImageClassification.from_pretrained(EMOTION_MODEL_ID)
                model.eval()
                torch.set_grad_enabled(False)
                id2label = {int(k): str(v).lower()
                            for k, v in model.config.id2label.items()}
                _emotion = (processor, model, id2label)
    return _emotion


# --------------------------------------------------------------------------- #
# 결과 자료구조
# --------------------------------------------------------------------------- #
@dataclass
class FrameEmotion:
    index: int
    timestamp: float
    detected: bool
    box: tuple[int, int, int, int] | None = None
    det_score: float = 0.0
    top_label: str = ""
    top_label_ko: str = ""
    probs: dict[str, float] = field(default_factory=dict)
    valence: float = 0.0
    thumbnail: str = ""              # data:image/jpeg;base64,...


@dataclass
class FaceResult:
    frames: list[FrameEmotion] = field(default_factory=list)
    mean_probs: dict[str, float] = field(default_factory=dict)
    mean_valence: float = 0.0
    dominant: str = ""
    dominant_ko: str = ""
    faces_found: int = 0
    frames_scanned: int = 0
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.faces_found > 0

    @property
    def positive_score(self) -> float:
        """valence(-1~1) 를 P(positive) 스케일(0~1)로 옮긴 값."""
        return (self.mean_valence + 1.0) / 2.0


# --------------------------------------------------------------------------- #
# 핵심 로직
# --------------------------------------------------------------------------- #
def _largest_face(rgb: np.ndarray):
    """가장 큰 얼굴 하나의 (x, y, w, h, score) 를 돌려준다. 없으면 None."""
    h, w = rgb.shape[:2]
    detector = get_detector()
    detector.setInputSize((w, h))
    _, faces = detector.detect(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if faces is None or len(faces) == 0:
        return None
    best = max(faces, key=lambda f: float(f[2]) * float(f[3]))
    x, y, fw, fh = (int(round(v)) for v in best[:4])
    return x, y, fw, fh, float(best[-1])


def _crop(rgb: np.ndarray, box, margin: float = 0.25) -> np.ndarray:
    x, y, w, h = box
    mx, my = int(w * margin), int(h * margin)
    H, W = rgb.shape[:2]
    x0, y0 = max(0, x - mx), max(0, y - my)
    x1, y1 = min(W, x + w + mx), min(H, y + h + my)
    return rgb[y0:y1, x0:x1]


def _thumb(rgb: np.ndarray, size: int = 128) -> str:
    if rgb.size == 0:
        return ""
    small = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(small, cv2.COLOR_RGB2BGR),
                           [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def analyze(samples) -> FaceResult:
    """프레임 목록(media.Sample)을 받아 프레임별 표정과 평균 감정을 만든다."""
    result = FaceResult(frames_scanned=len(samples))
    if not samples:
        result.error = "분석할 영상 프레임이 없습니다."
        return result

    try:
        import torch
        processor, model, id2label = get_emotion_model()
    except Exception as exc:                     # noqa: BLE001
        result.error = f"감정 분류기 로딩 실패: {exc}"
        return result

    crops: list[np.ndarray] = []
    pending: list[FrameEmotion] = []

    for s in samples:
        try:
            det = _largest_face(s.image)
        except Exception as exc:                 # noqa: BLE001
            result.error = result.error or f"얼굴 검출 실패: {exc}"
            det = None

        if det is None:
            result.frames.append(FrameEmotion(index=s.index, timestamp=s.timestamp,
                                              detected=False))
            continue

        x, y, w, h, score = det
        face = _crop(s.image, (x, y, w, h))
        if face.size == 0:
            result.frames.append(FrameEmotion(index=s.index, timestamp=s.timestamp,
                                              detected=False))
            continue

        fe = FrameEmotion(index=s.index, timestamp=s.timestamp, detected=True,
                          box=(x, y, w, h), det_score=score, thumbnail=_thumb(face))
        crops.append(face)
        pending.append(fe)
        result.frames.append(fe)

    result.frames.sort(key=lambda f: f.index)

    if not crops:
        result.error = result.error or "영상에서 얼굴을 찾지 못했습니다."
        return result

    # 검출된 얼굴을 한 번에 배치 추론
    try:
        inputs = processor(images=crops, return_tensors="pt")
        logits = model(**inputs).logits.float()
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
    except Exception as exc:                     # noqa: BLE001
        result.error = f"표정 분류 실패: {exc}"
        return result

    labels = [id2label[i] for i in range(probs.shape[1])]
    for fe, p in zip(pending, probs):
        d = {lab: float(v) for lab, v in zip(labels, p)}
        fe.probs = d
        fe.top_label = max(d, key=d.get)
        fe.top_label_ko = EMOTION_KO.get(fe.top_label, fe.top_label)
        fe.valence = float(sum(VALENCE.get(lab, 0.0) * v for lab, v in d.items()))

    # 프레임 평균 = 확률 분포의 산술평균 (요청하신 "평균 감정")
    mean = probs.mean(axis=0)
    result.mean_probs = {lab: float(v) for lab, v in zip(labels, mean)}
    result.mean_valence = float(
        sum(VALENCE.get(lab, 0.0) * v for lab, v in result.mean_probs.items())
    )
    result.dominant = max(result.mean_probs, key=result.mean_probs.get)
    result.dominant_ko = EMOTION_KO.get(result.dominant, result.dominant)
    result.faces_found = len(crops)
    return result
