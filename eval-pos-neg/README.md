# eval-pos-neg — 동영상 리뷰 멀티모달 감정 분석

동영상 리뷰에서 **말한 내용(STT)** 과 **얼굴 표정** 을 함께 읽어 최종 긍/부정을 판정하는
FastAPI 데모. 브라우저에서 바로 녹화하거나, 찍어둔 영상을 업로드하면 됩니다.

전부 **무료·로컬 실행**이며 외부 API 호출이 없습니다.

| 파일 | 내용 |
|---|---|
| `app.py` | FastAPI 엔트리포인트 |
| [`eval_pos_neg/`](eval_pos_neg/) | 분석 패키지 — 파이프라인·판정 로직·튜닝은 [패키지 README](eval_pos_neg/README.md) 참고 |
| `stt.py` | faster-whisper 로컬 STT 최소 예제 (CLI). 본 데모와 독립적으로 돌아가는 참고용 스크립트 |

---

## 빠른 시작

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m uvicorn app:app --reload --port 8000
# → http://127.0.0.1:8000/eval-pos-neg
```

모델 가중치는 저장소에 없습니다. 첫 실행 시 HuggingFace 캐시와 `models/` 로 **자동 다운로드**됩니다
(합계 약 800MB).

`ffmpeg` 실행 파일은 **필요 없습니다** — 디코딩은 PyAV 내장 ffmpeg 라이브러리로 처리합니다.

> 브라우저 내장 녹화(`getUserMedia`)는 보안 컨텍스트를 요구합니다. `localhost` 는 허용되지만
> 다른 기기에서 IP 로 접속하면 HTTPS 가 있어야 카메라가 열립니다 (아래 6번 항목).
> **녹화 없이 파일 업로드만 쓸 거라면** 이 과정은 전부 건너뛰어도 됩니다.

### 검증 상태

합성 테스트 영상(얼굴 이미지 + 한국어 음성)으로 확인한 결과:

| 케이스 | 결과 |
|---|---|
| mp4 (얼굴+음성) | 200, 11.9s · STT 원문 정확 일치 · 얼굴 10/10 검출 |
| webm vp8+opus (MediaRecorder 형식) | 200, 8.3s |
| 얼굴 없는 영상 | 텍스트 단일 모달로 판정 + 사유 표시 |
| 오디오 없는 영상 | 표정 단일 모달로 판정 + 사유 표시 |

데스크톱 브라우저 UI는 렌더링과 엔드포인트 응답까지 확인했습니다.
**브라우저 내장 녹화 버튼은 실제 카메라로 검증하지 못했습니다** (개발 환경에 카메라 없음).
iPhone Chrome 에서는 인증서 문제로 녹화가 막히는 것까지 확인했고, 해결책은 아래 6번 항목에 있습니다.

---

## ⚠️ 인증서 개인키 취급

HTTPS 로 녹화를 테스트하려면 개발용 로컬 CA 를 만들게 되는데, 그 개인키(`ca-key.pem`)는
**절대 저장소에 올리지 마세요.** 이 CA 를 테스트 기기에 "신뢰됨"으로 설치하기 때문에,
키가 유출되면 **그 기기의 모든 HTTPS 트래픽을 위조할 수 있습니다.**

- 이 폴더와 저장소 루트의 `.gitignore` 가 `certs/`, `*.pem`, `*.crt` 를 이중으로 막습니다.
- 테스트가 끝나면 기기에서 프로파일을 지우세요:
  **설정 → 일반 → VPN 및 기기 관리 → 프로파일 삭제**
- PC 쪽은 `rm -rf certs/` 로 정리합니다.

인증서를 만드는 명령은 아래 6번 항목에 전부 적어 두었습니다.

---

## 삽질 기록

라이브러리 버전이 올라가면서 인터넷의 예제 코드가 대부분 그대로는 안 도는 상태였습니다.
같은 함정에 다시 빠지지 않도록 *증상 → 원인 → 해결* 로 남깁니다.

### 1. OpenCV 5 에는 `CascadeClassifier` 가 아예 없다

**증상**
```python
cv2.data.haarcascades          # 경로는 있는데 XML 이 하나도 없음
cv2.CascadeClassifier(...)     # AttributeError: module 'cv2' has no attribute 'CascadeClassifier'
```

**원인** OpenCV 5.0 부터 haarcascade XML 번들이 빠졌고, `CascadeClassifier` API 자체가 제거됐습니다.
"얼굴 인식 = haarcascade" 로 시작하는 예제는 전부 4.x 기준입니다.

**해결** **YuNet**(ONNX, 약 230KB)으로 갈아탔습니다. `cv2.FaceDetectorYN` 은 5.x 에 그대로 있고,
정확도도 haar 보다 낫습니다. 부수 효과로 5점 랜드마크와 검출 신뢰도까지 얻습니다.

```python
det = cv2.FaceDetectorYN.create(onnx_path, "", (320, 320), 0.6, 0.3, 5000)
det.setInputSize((w, h))
_, faces = det.detect(bgr)     # Nx15: x,y,w,h + 랜드마크 10 + score
```

> `setPreferableTarget` 관련 경고가 한 줄 뜨는데 동작에는 지장 없습니다.

### 2. opencv_zoo 의 raw 링크는 파일이 아니라 LFS 포인터다

**증상** YuNet 모델을 받았는데 **131 바이트**. 로딩하면 당연히 실패.

**원인** `raw.githubusercontent.com` 은 Git LFS 로 관리되는 파일에 대해 실제 바이너리가 아니라
포인터 텍스트를 돌려줍니다.

**해결** 호스트를 `media.githubusercontent.com` 으로 바꾸면 실물(232KB)이 옵니다.

```bash
# ✗ 131 bytes (LFS 포인터)
curl -L https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
# ✓ 232 KB
curl -L https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
```

### 3. ffmpeg 없이 동영상을 다뤄야 했다

**증상** `ffmpeg` 가 PATH 에 없음. 대부분의 튜토리얼은 `subprocess` 로 ffmpeg 를 부릅니다.

**해결** **PyAV** 하나로 끝냅니다. 바이너리 휠에 ffmpeg 라이브러리가 들어 있어 별도 설치가 필요 없고,
vp8/vp9/opus/aac/h264 인코딩까지 전부 됩니다. 오디오는 리샘플러로 바로 whisper 입력 규격을 만듭니다.

```python
resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=16000)
```

주의할 점 두 가지:
- `resampler.resample()` 은 PyAV 버전에 따라 **단일 프레임 또는 리스트**를 돌려줍니다. 둘 다 받도록 처리.
- 끝나고 `resample(None)` 으로 **flush** 하지 않으면 마지막 몇 백 ms 가 잘립니다.
- 오디오와 비디오는 **컨테이너를 따로 열어** 각각 순회해야 합니다. 한 번의 순회로 두 스트림을 뽑으면
  디코딩 위치가 서로 엉킵니다.

### 4. MediaRecorder 가 만든 webm 은 길이를 모른다

**증상** 브라우저 녹화 파일에서 `stream.frames == 0`, `duration == None`.
"전체 프레임 수 ÷ N" 으로 균등 샘플링하는 코드가 통째로 무력화됩니다.

**원인** MediaRecorder 는 스트리밍 출력이라 컨테이너 헤더에 총 길이를 못 적습니다.

**해결** 총 길이를 **모르는 채로도** 시간축 균등 샘플이 나오는 *스트라이드 배증 데시메이션*을 썼습니다.
버퍼가 2N 을 넘으면 짝수 번째만 남기고 스트라이드를 2배로 올립니다. 메모리는 O(N) 고정.

```python
if seen % stride == 0:
    kept.append(frame)
    if len(kept) > 2 * n_frames:
        kept = kept[::2]
        stride *= 2
```

프레임 → ndarray 변환이 비싼 작업인데, 이 방식은 변환 횟수도 `O(N log(total/N))` 으로 묶입니다
(900프레임·N=10 기준 약 110회).

### 5. HuggingFace 모델의 `id2label` 을 믿으면 안 된다

**증상** 감성 모델을 바꿨더니 긍/부정이 통째로 뒤집힘.

**원인** 허브의 한국어 감성 모델 상당수가 `id2label` 을 `{0: 'LABEL_0', 1: 'LABEL_1'}` 또는
`{0: '0', 1: '1'}` 로 방치해 뒀습니다. 어느 쪽이 positive 인지 메타데이터로는 알 수 없습니다.

**해결** 로딩 직후 **명백한 긍/부정 프로브 문장**을 밀어 넣고 확률 차이가 가장 커지는 인덱스를
positive 로 **실측**합니다. 한국어 감성 모델 평가 하네스(ksent)에서 쓰던 방식을 그대로 가져왔습니다.

```python
gap = probs(POSITIVE_PROBES).mean(0) - probs(NEGATIVE_PROBES).mean(0)
pos_index = int(np.argmax(gap))
```

실측값은 결과 JSON 의 `text_sentiment.calibration` 에 남겨서 눈으로 확인할 수 있게 했습니다
(예: `pos_index=1 (declared=1), gap=0.979`).

### 6. iOS 에서 브라우저 녹화가 막히는 진짜 이유는 두 개다

**증상** iPhone Chrome 에서 `https://192.168.0.66:8443` 접속은 되는데 녹화 버튼을 누르면:
```
NotAllowedError: The request is not allowed by the user agent or the platform
in the current context, possibly because the user denied permission.
```
권한 팝업 자체가 안 뜹니다.

**원인 두 가지가 겹쳐 있었습니다.**

1. **유효기간.** 처음 만든 자체서명 인증서가 825일짜리였는데, **Safari/WebKit 은 398일을 넘는 서버
   인증서를 무조건 거부**합니다. iOS 13 부터의 정책입니다. 이것만으로 이미 탈락.
2. **신뢰 경로.** 자체서명 인증서는 경고를 "무시하고 진입"해도 iOS 가 그 오리진을 **완전한 보안
   컨텍스트로 인정하지 않습니다.** 화면은 뜨지만 `getUserMedia` 같은 강력 기능은 차단됩니다.
   데스크톱 Chrome 이 관대해서 헷갈리기 쉬운 부분입니다.

**해결** 로컬 CA 를 만들고, 거기서 **397일짜리** 리프 인증서를 발급한 뒤 CA 를 기기에 신뢰 설치합니다.

```bash
# 1) 로컬 루트 CA (CA 자신은 398일 제한 대상이 아님)
openssl req -x509 -newkey rsa:2048 -nodes -days 3650 -sha256 \
  -keyout ca-key.pem -out ca-cert.pem \
  -subj "//CN=eval-pos-neg local CA\OU=dev" \
  -addext "basicConstraints=critical,CA:TRUE,pathlen:0" \
  -addext "keyUsage=critical,keyCertSign,cRLSign"

# 2) 서버 키 + CSR
openssl req -newkey rsa:2048 -nodes -keyout dev-key.pem -out dev.csr \
  -subj "//CN=192.168.0.66"

# 3) 리프 확장 — SAN 에 IP 필수, serverAuth 필수
cat > leaf.ext <<'EOF'
subjectAltName=IP:192.168.0.66,IP:127.0.0.1,DNS:localhost
basicConstraints=CA:FALSE
keyUsage=critical,digitalSignature,keyEncipherment
extendedKeyUsage=serverAuth
EOF

# 4) 397일 발급 (398일 제한 충족)
openssl x509 -req -in dev.csr -CA ca-cert.pem -CAkey ca-key.pem -CAcreateserial \
  -out dev-cert.pem -days 397 -sha256 -extfile leaf.ext

cat dev-cert.pem ca-cert.pem > dev-fullchain.pem
openssl x509 -in ca-cert.pem -outform DER -out eval-pos-neg-ca.crt   # iOS 설치용

# 5) 기동
uvicorn app:app --host 0.0.0.0 --port 8443 \
  --ssl-keyfile certs/dev-key.pem --ssl-certfile certs/dev-fullchain.pem
```

> Git Bash(MSYS)에서는 `-subj` 값 앞에 `//` 를 붙여야 경로로 오인되지 않습니다.

**기기 설치 순서** — 3번을 빠뜨리는 실수가 잦습니다.

1. **Safari 로** `http://<IP>:8000/ca.crt` 접속 (iOS Chrome 은 구성 프로파일을 처리하지 못함)
2. 설정 → 다운로드된 프로파일 → 설치
3. **설정 → 일반 → 정보 → 인증서 신뢰 설정 → 해당 CA 토글 ON**
   ← 설치만으로는 신뢰되지 않습니다
4. `https://<IP>:8443/eval-pos-neg` → 주소창에 **자물쇠** 확인

**녹화가 꼭 필요 없다면** 이 과정을 전부 건너뛸 수 있습니다. 폰 카메라 앱으로 찍어서
평문 HTTP 로 **업로드**하면 분석은 동일하게 동작합니다. 업로드는 보안 컨텍스트가 필요 없습니다.

### 7. Windows 방화벽과 네트워크 분류

**증상** 서버는 `0.0.0.0` 바인딩인데 폰에서 연결이 안 됨.

**원인** 두 가지. (a) 인바운드 허용 규칙 없음, (b) 네트워크가 **Public** 으로 분류되어 기본 차단.

**해결** 관리자 PowerShell 에서 LAN 대역으로 좁혀 허용합니다.

```powershell
New-NetFirewallRule -DisplayName "eval-pos-neg dev (LAN)" `
  -Direction Inbound -Action Allow -Protocol TCP `
  -LocalPort 8000,8443 -RemoteAddress 192.168.0.0/24 -Profile Any
# 정리: Remove-NetFirewallRule -DisplayName "eval-pos-neg dev (LAN)"
```

### 8. curl(Schannel)은 `--cacert` 를 무시한다

**증상** 인증서 체인을 제대로 만들었는데 `curl --cacert ca-cert.pem https://...` 가 계속 실패
(`could not establish a secure connection`, exit 000). 인증서를 몇 번이나 다시 만들었습니다.

**원인** Windows 의 curl 은 **Schannel** 백엔드입니다 (`curl -V` 로 확인). Schannel 은 Windows
인증서 저장소만 보기 때문에 PEM 파일로 준 `--cacert` 를 무시합니다. **인증서 문제가 아니었습니다.**

**해결** 검증은 OpenSSL 이나 Python 으로 합니다.

```bash
openssl s_client -connect 192.168.0.66:8443 -CAfile certs/ca-cert.pem   # → Verification: OK
```
```python
ctx = ssl.create_default_context(cafile="certs/ca-cert.pem")
urllib.request.urlopen("https://192.168.0.66:8443/eval-pos-neg", context=ctx)   # → 200
```

### 9. 동작이 확인된 버전 조합

최신 메이저 버전들이 한꺼번에 올라간 시점이라, 아래 조합이 실제로 맞물리는 것을 확인했습니다.

| 패키지 | 버전 | 메모 |
|---|---|---|
| Python | 3.14.7 | |
| torch | 2.13.0+cpu | |
| **torchvision** | 설치 필수 | 없으면 `AutoImageProcessor` 가 ImportError. transformers 의존성에 안 딸려옴 |
| transformers | 5.15.1 | |
| opencv-python | 5.0.0 | haarcascade 없음 (1번 항목) |
| av (PyAV) | 18.1.0 | libvpx / libopus / libx264 인코더 전부 포함 |
| faster-whisper | 1.2.1 | `vad_filter=True` 에 onnxruntime 필요 (자동 설치됨) |
| fastapi | 0.141.1 | |

---

## 라이선스 · 출처

- 얼굴 검출: [OpenCV Zoo YuNet](https://github.com/opencv/opencv_zoo)
- 표정 분류: [`trpakov/vit-face-expression`](https://huggingface.co/trpakov/vit-face-expression)
- 텍스트 감성: [`Copycats/koelectra-base-v3-generalized-sentiment-analysis`](https://huggingface.co/Copycats/koelectra-base-v3-generalized-sentiment-analysis)
- STT: [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
