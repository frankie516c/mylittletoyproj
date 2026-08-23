# -*- coding: utf-8 -*-
"""/eval-pos-neg — 동영상 리뷰 긍/부정 분석 데모 라우터.

기존 FastAPI 앱에 붙이려면:

    from eval_pos_neg import router as eval_pos_neg_router
    app.include_router(eval_pos_neg_router)
"""

from __future__ import annotations

import os
import tempfile
import time
from dataclasses import asdict

from fastapi import APIRouter, File, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool

from . import face_emotion, fusion, media, stt_engine, text_sentiment
from .config import (
    BASE_DIR,
    EMOTION_MODEL_ID,
    MAX_UPLOAD_BYTES,
    NUM_FRAMES,
    TEXT_MODEL_ID,
    WHISPER_MODEL,
)

router = APIRouter(prefix="/eval-pos-neg", tags=["eval-pos-neg"])
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

_SUFFIX_BY_TYPE = {
    "video/mp4": ".mp4", "video/webm": ".webm", "video/quicktime": ".mov",
    "video/x-matroska": ".mkv", "audio/webm": ".webm", "audio/mpeg": ".mp3",
    "audio/wav": ".wav", "audio/x-wav": ".wav",
}


def _suffix(upload: UploadFile) -> str:
    ext = os.path.splitext(upload.filename or "")[1].lower()
    if ext:
        return ext
    return _SUFFIX_BY_TYPE.get((upload.content_type or "").split(";")[0], ".webm")


def _pipeline(path: str) -> dict:
    """디코딩 → STT → 표정 → 융합. (동기, 스레드풀에서 실행)"""
    t0 = time.perf_counter()

    bundle = media.extract(path, n_frames=NUM_FRAMES)
    t_decode = time.perf_counter()

    transcript = stt_engine.transcribe(bundle.audio)
    t_stt = time.perf_counter()

    faces = face_emotion.analyze(bundle.frames)
    t_face = time.perf_counter()

    text_res = text_sentiment.analyze(transcript.text)
    verdict = fusion.combine(text_res, faces)
    t_end = time.perf_counter()

    return {
        "ok": True,
        "verdict": asdict(verdict),
        "stt": {
            "text": transcript.text,
            "language": transcript.language,
            "language_probability": transcript.language_probability,
            "duration": transcript.duration,
            "segments": [asdict(s) for s in transcript.segments],
            "error": transcript.error,
        },
        "text_sentiment": asdict(text_res),
        "face": {
            "frames": [asdict(f) for f in faces.frames],
            "mean_probs": faces.mean_probs,
            "mean_valence": faces.mean_valence,
            "dominant": faces.dominant,
            "dominant_ko": faces.dominant_ko,
            "faces_found": faces.faces_found,
            "frames_scanned": faces.frames_scanned,
            "positive_score": faces.positive_score if faces.ok else None,
            "emotion_ko": face_emotion.EMOTION_KO,
            "error": faces.error,
        },
        "media": {
            "audio_duration": bundle.audio_duration,
            "video_duration": bundle.video_duration,
            "frames_extracted": len(bundle.frames),
            "errors": bundle.errors,
        },
        "timing": {
            "decode": round(t_decode - t0, 2),
            "stt": round(t_stt - t_decode, 2),
            "face": round(t_face - t_stt, 2),
            "text": round(t_end - t_face, 2),
            "total": round(t_end - t0, 2),
        },
        "models": {
            "stt": f"faster-whisper/{WHISPER_MODEL}",
            "emotion": EMOTION_MODEL_ID,
            "text": TEXT_MODEL_ID,
        },
    }


@router.get("", response_class=HTMLResponse)
@router.get("/", response_class=HTMLResponse)
async def page(request: Request):
    return templates.TemplateResponse(
        request, "eval_pos_neg.html",
        {"num_frames": NUM_FRAMES, "whisper_model": WHISPER_MODEL,
         "emotion_model": EMOTION_MODEL_ID, "text_model": TEXT_MODEL_ID},
    )


@router.post("/analyze")
async def analyze(video: UploadFile = File(...)):
    data = await video.read()
    if not data:
        return JSONResponse({"ok": False, "error": "빈 파일입니다."}, status_code=400)
    if len(data) > MAX_UPLOAD_BYTES:
        return JSONResponse(
            {"ok": False, "error": f"파일이 너무 큽니다 (최대 {MAX_UPLOAD_BYTES // 1024 // 1024}MB)."},
            status_code=413,
        )

    fd, path = tempfile.mkstemp(suffix=_suffix(video))
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        result = await run_in_threadpool(_pipeline, path)
    except Exception as exc:                     # noqa: BLE001
        return JSONResponse({"ok": False, "error": f"분석 중 오류: {exc}"}, status_code=500)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass

    result["upload"] = {"filename": video.filename,
                        "content_type": video.content_type,
                        "bytes": len(data)}
    return JSONResponse(result)


@router.post("/warmup")
async def warmup():
    """무거운 모델 3종을 미리 로딩해 첫 분석의 대기 시간을 줄인다."""
    def _load():
        out = {}
        for name, fn in (("stt", stt_engine.get_model),
                         ("face_detector", face_emotion.get_detector),
                         ("emotion", face_emotion.get_emotion_model),
                         ("text", text_sentiment.load)):
            try:
                fn()
                out[name] = "ready"
            except Exception as exc:             # noqa: BLE001
                out[name] = f"error: {exc}"
        return out

    return JSONResponse({"ok": True, "models": await run_in_threadpool(_load)})
