# /eval-pos-neg — 동영상 리뷰 멀티모달 감정 분석

동영상 리뷰에서 **말한 내용(STT)** 과 **얼굴 표정** 을 함께 읽어 최종 긍/부정을 판정하는 FastAPI 데모.

## 실행

```bash
.venv/Scripts/python -m uvicorn app:app --reload --port 8000
# http://127.0.0.1:8000/eval-pos-neg
```

> 브라우저 녹화(`getUserMedia`)는 보안 컨텍스트가 필요합니다. `localhost` 는 허용되지만
> 다른 PC에서 IP로 접속하면 HTTPS 가 있어야 카메라가 열립니다.

## 기존 FastAPI 앱에 붙이기

```python
from eval_pos_neg import router as eval_pos_neg_router
app.include_router(eval_pos_neg_router)
```

라우터가 `prefix="/eval-pos-neg"` 를 이미 갖고 있고 템플릿 경로도 모듈 내부를 가리키므로,
`eval_pos_neg/` 폴더와 `models/` 폴더만 함께 옮기면 그대로 동작합니다.

## 엔드포인트

| 메서드 | 경로 | 설명 |
|---|---|---|
| GET | `/eval-pos-neg` | 데모 페이지 (업로드 / 녹화 UI) |
| POST | `/eval-pos-neg/analyze` | `multipart/form-data`, 필드명 `video` → 분석 결과 JSON |
| POST | `/eval-pos-neg/warmup` | 모델 4종 사전 로딩 (페이지 진입 시 자동 호출) |

## 처리 파이프라인

```
업로드/녹화 영상
   │
   ├─ PyAV 디코딩 ─┬─ 오디오 → 16kHz mono float32
   │               └─ 프레임 → 시간축 균등 10장
   │
   ├─ faster-whisper (small, int8, VAD) ─→ 리뷰 텍스트
   │        └─ KoELECTRA 감성분류 ─→ P(positive)_text
   │
   ├─ YuNet 얼굴검출 → 가장 큰 얼굴 crop (10장 배치)
   │        └─ ViT 표정분류 (7클래스) ─→ 프레임별 확률
   │              └─ 평균 확률 → 평균 정서가(valence) → P(positive)_face
   │
   └─ 가중 융합 → 최종 판정 (긍정 / 중립 / 부정 + 확신도)
```

### 사용 모델 (전부 무료·로컬 실행)

| 역할 | 모델 | 비고 |
|---|---|---|
| STT | `Systran/faster-whisper-small` | CPU int8, VAD 필터로 환청 억제 |
| 얼굴 검출 | OpenCV **YuNet** ONNX (~230KB) | `models/` 에 없으면 자동 다운로드. OpenCV 5 는 haarcascade 를 번들하지 않음 |
| 표정 분류 | `trpakov/vit-face-expression` | FER 7클래스: angry / disgust / fear / happy / neutral / sad / surprise |
| 텍스트 감성 | `Copycats/koelectra-base-v3-generalized-sentiment-analysis` | 한국어 |

첫 실행 시 HuggingFace 캐시로 자동 다운로드됩니다. `ffmpeg` 실행 파일은 필요 없습니다(PyAV 내장).

## 판정 로직

**표정 → 정서가(valence)**: 감정별 가중치의 기댓값. `-1`(부정) ~ `+1`(긍정)

| happy | surprise | neutral | sad | fear | disgust | angry |
|---|---|---|---|---|---|---|
| +1.0 | +0.2 | 0.0 | −0.7 | −0.7 | −0.9 | −1.0 |

`P(positive)_face = (valence + 1) / 2`

**융합**: 기본 가중치는 텍스트 0.65 / 표정 0.35. 표정 쪽은 **얼굴 검출 커버리지**(10장 중 몇 장에서
얼굴이 잡혔는지)에 비례해 깎은 뒤 정규화합니다. 한쪽 모달리티가 없으면 나머지 하나로만 판정하고
그 사실을 결과에 남깁니다.

```
score = (w_text · P_text + w_face · P_face) / (w_text + w_face)
```

`score ≥ 0.60` → 긍정, `≤ 0.40` → 부정, 그 사이는 중립.

**확신도**는 0.5 로부터의 거리 × 두 모달리티의 일치도로 계산합니다. 말은 긍정인데 표정은
부정처럼 **방향이 어긋나면** 확신도를 낮추고 "반어·억지 미소·무표정 낭독일 수 있음" 경고를 답니다.

### 극성 캘리브레이션

허브 모델의 `id2label` 이 `0/1` 처럼 무의미한 경우가 많아, 명백한 긍/부정 프로브 문장을 넣어
**어느 인덱스가 실제로 positive 인지 실측**한 뒤 사용합니다 (감성 모델 평가 하네스 ksent 와 동일한 방식).
결과 JSON 의 `text_sentiment.calibration` 에 실측값이 담깁니다.

## 튜닝 (환경 변수)

| 변수 | 기본값 | 설명 |
|---|---|---|
| `WHISPER_MODEL` | `small` | `tiny` / `base` / `medium` / `large-v3` |
| `WHISPER_LANG` | `ko` | 빈 문자열이면 자동 감지 |
| `NUM_FRAMES` | `10` | 표정 분석에 쓸 프레임 수 |
| `TEXT_WEIGHT` / `FACE_WEIGHT` | `0.65` / `0.35` | 융합 가중치 |
| `EMOTION_MODEL_ID` | `trpakov/vit-face-expression` | 표정 분류기 교체 |
| `TEXT_MODEL_ID` | `Copycats/koelectra-...` | 텍스트 감성 모델 교체 |

## 파일 구성

```
app.py                          FastAPI 엔트리포인트
eval_pos_neg/
  router.py                     라우트 + 파이프라인 조립
  media.py                      PyAV 오디오/프레임 추출
  stt_engine.py                 faster-whisper 래퍼
  face_emotion.py               YuNet 검출 + ViT 표정 분류 + 평균 감정
  text_sentiment.py             KoELECTRA + 극성 캘리브레이션
  fusion.py                     멀티모달 융합/판정
  config.py                     설정값
  templates/eval_pos_neg.html   데모 UI
models/
  face_detection_yunet_2023mar.onnx
```

## 알려진 한계

- 표정 분류기는 FER2013 계열이라 정면·조명 좋은 얼굴에서 가장 정확합니다. 옆얼굴·저조도에서는
  `neutral` / `sad` 쪽으로 쏠리는 경향이 있습니다.
- 프레임당 **가장 큰 얼굴 하나**만 봅니다. 여러 명이 나오는 영상은 대상이 흔들릴 수 있습니다.
- valence 가중치는 데모용 휴리스틱입니다. 실제 서비스에서는 라벨링된 리뷰 데이터로
  융합 가중치와 valence 를 함께 학습시키는 편이 낫습니다.
