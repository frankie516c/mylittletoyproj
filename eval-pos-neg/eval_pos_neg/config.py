# -*- coding: utf-8 -*-
"""/eval-pos-neg 데모 설정값."""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
MODEL_DIR = PROJECT_ROOT / "models"

# --- 얼굴 검출 (OpenCV YuNet, ONNX) ---
# OpenCV 5.x 는 haarcascade 를 더 이상 번들하지 않으므로 YuNet 을 사용한다.
YUNET_PATH = MODEL_DIR / "face_detection_yunet_2023mar.onnx"
YUNET_URL = (
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
FACE_SCORE_THRESHOLD = 0.6
FACE_NMS_THRESHOLD = 0.3

# --- 표정(감정) 분류기 : 최초 실행 시 HF 허브에서 자동 다운로드 ---
EMOTION_MODEL_ID = os.getenv("EMOTION_MODEL_ID", "trpakov/vit-face-expression")

# --- 한국어 텍스트 감성 분류기 ---
TEXT_MODEL_ID = os.getenv(
    "TEXT_MODEL_ID", "Copycats/koelectra-base-v3-generalized-sentiment-analysis"
)

# --- STT (faster-whisper) ---
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE = os.getenv("WHISPER_COMPUTE", "int8")
WHISPER_LANG = os.getenv("WHISPER_LANG", "ko")  # 빈 문자열이면 자동 감지

# --- 프레임 샘플링 ---
NUM_FRAMES = int(os.getenv("NUM_FRAMES", "10"))
SAMPLE_RATE = 16000            # faster-whisper 입력 규격
MAX_UPLOAD_BYTES = 200 * 1024 * 1024

# --- 멀티모달 융합 가중치 ---
TEXT_WEIGHT = float(os.getenv("TEXT_WEIGHT", "0.65"))
FACE_WEIGHT = float(os.getenv("FACE_WEIGHT", "0.35"))
