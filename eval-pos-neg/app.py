# -*- coding: utf-8 -*-
"""FastAPI 엔트리포인트.

실행:
    .venv/Scripts/python -m uvicorn app:app --reload --port 8000
    → http://127.0.0.1:8000/eval-pos-neg

라우터를 추가할 때는 아래 include_router 목록에 한 줄만 더한다.
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, RedirectResponse

from eval_pos_neg import router as eval_pos_neg_router

CA_CERT = Path(__file__).parent / "certs" / "eval-pos-neg-ca.crt"

app = FastAPI(
    title="Review Sentiment Demo",
    description="동영상 리뷰에서 음성(STT)과 표정을 함께 읽어 긍/부정을 판정하는 데모",
    version="0.1.0",
)

app.include_router(eval_pos_neg_router)


@app.get("/", include_in_schema=False)
def index():
    return RedirectResponse("/eval-pos-neg")


@app.get("/healthz", include_in_schema=False)
def healthz():
    return {"status": "ok"}


@app.get("/ca.crt", include_in_schema=False)
def ca_cert():
    """개발용 로컬 CA 인증서 배포.

    휴대폰에서 HTTPS 로 녹화하려면 이 CA 를 기기에 신뢰 설치해야 한다.
    (iOS 는 신뢰되지 않은 인증서의 오리진에 카메라 접근을 허용하지 않는다.)
    """
    return FileResponse(
        CA_CERT,
        media_type="application/x-x509-ca-cert",
        filename="eval-pos-neg-ca.crt",
    )
