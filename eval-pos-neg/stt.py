"""로컬 Whisper STT (faster-whisper / tiny 모델)

사용법:
    python stt.py                  # 샘플 음성 자동 다운로드 후 인식
    python stt.py audio.mp3        # 원하는 파일 인식
    python stt.py audio.mp3 --model base --lang ko
"""

import argparse
import sys
import urllib.request
from pathlib import Path

from faster_whisper import WhisperModel

# 11초짜리 JFK 연설 샘플 (openai/whisper 테스트 파일)
SAMPLE_URL = "https://raw.githubusercontent.com/openai/whisper/main/tests/jfk.flac"
SAMPLE_PATH = Path(__file__).parent / "sample.flac"


def ensure_sample() -> Path:
    """샘플 음성이 없으면 내려받는다."""
    if not SAMPLE_PATH.exists():
        print(f"샘플 음성 다운로드 중... ({SAMPLE_URL})")
        urllib.request.urlretrieve(SAMPLE_URL, SAMPLE_PATH)
        print(f"저장 완료: {SAMPLE_PATH}")
    return SAMPLE_PATH


def fmt(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    return f"{int(m):02d}:{s:05.2f}"


def main() -> int:
    p = argparse.ArgumentParser(description="로컬 Whisper 음성 인식")
    p.add_argument("audio", nargs="?", help="음성 파일 경로 (생략하면 샘플 사용)")
    p.add_argument("--model", default="tiny",
                   help="모델 크기: tiny(기본), tiny.en, base, small, medium, large-v3")
    p.add_argument("--lang", default=None, help="언어 코드 (ko, en ...). 생략하면 자동 감지")
    args = p.parse_args()

    audio = Path(args.audio) if args.audio else ensure_sample()
    if not audio.exists():
        print(f"파일을 찾을 수 없습니다: {audio}", file=sys.stderr)
        return 1

    # 첫 실행 시 모델(tiny ≈ 75MB)을 자동으로 내려받아 캐시한다.
    print(f"모델 로딩 중: {args.model} (CPU, int8)")
    model = WhisperModel(args.model, device="cpu", compute_type="int8")

    print(f"인식 중: {audio}\n")
    segments, info = model.transcribe(audio.as_posix(), language=args.lang, beam_size=5)

    print(f"감지 언어: {info.language} (확률 {info.language_probability:.2f}), "
          f"길이 {info.duration:.1f}초\n")

    parts = []
    for seg in segments:
        print(f"[{fmt(seg.start)} -> {fmt(seg.end)}] {seg.text.strip()}")
        parts.append(seg.text.strip())

    print("\n--- 전체 텍스트 ---")
    print(" ".join(parts))
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
