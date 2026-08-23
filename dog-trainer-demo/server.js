/**
 * 강아지 훈련 Q&A 데모 서버
 *  - 정적 파일 서빙 (public/)
 *  - POST /api/stt   : 녹음 오디오 -> OpenAI 음성인식(텍스트)
 *  - POST /api/ask   : 질문 -> 웹검색(블로그/유튜브/논문) 기반 답변 + 출처
 *  - POST /api/eval  : 답변 -> 신뢰도/정확성 평가(LLM 심판)
 *
 * 외부 npm 의존성 없음 (Node 18+ 내장 fetch/FormData/Blob 사용)
 */
const http = require("http");
const https = require("https");
const os = require("os");
const fs = require("fs");
const path = require("path");

const ROOT = __dirname;
const PUBLIC_DIR = path.join(ROOT, "public");
const CERT_DIR = path.join(ROOT, "certs");
const PORT = process.env.PORT || 3000;
const HTTPS_PORT = process.env.HTTPS_PORT || 3443;

// ---------------------------------------------------------------- env
function loadEnv() {
  const f = path.join(ROOT, ".env");
  if (!fs.existsSync(f)) return;
  for (const line of fs.readFileSync(f, "utf8").split(/\r?\n/)) {
    const m = line.match(/^\s*([A-Z0-9_]+)\s*=\s*(.*)\s*$/i);
    if (!m) continue;
    const v = m[2].replace(/^['"]|['"]$/g, "");
    if (!process.env[m[1]]) process.env[m[1]] = v;
  }
}
loadEnv();

const API_KEY = process.env.OPENAI_API_KEY;
const MODEL = process.env.OPENAI_MODEL || "gpt-4o";
const STT_MODEL = process.env.OPENAI_STT_MODEL || "gpt-4o-transcribe";
const API_BASE = process.env.OPENAI_BASE_URL || "https://api.openai.com/v1";

// ---------------------------------------------------------------- utils
const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".svg": "image/svg+xml",
  ".ico": "image/x-icon",
  ".json": "application/json; charset=utf-8",
};

function sendJSON(res, code, obj) {
  const body = JSON.stringify(obj);
  res.writeHead(code, {
    "Content-Type": "application/json; charset=utf-8",
    "Content-Length": Buffer.byteLength(body),
  });
  res.end(body);
}

function readBody(req, limit = 30 * 1024 * 1024) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let size = 0;
    req.on("data", (c) => {
      size += c.length;
      if (size > limit) {
        reject(new Error("요청 본문이 너무 큽니다."));
        req.destroy();
        return;
      }
      chunks.push(c);
    });
    req.on("end", () => {
      try {
        resolve(JSON.parse(Buffer.concat(chunks).toString("utf8") || "{}"));
      } catch (e) {
        reject(new Error("JSON 파싱 실패"));
      }
    });
    req.on("error", reject);
  });
}

async function openai(endpoint, payload) {
  const r = await fetch(`${API_BASE}${endpoint}`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) {
    const msg = (data && data.error && data.error.message) || `OpenAI 오류 (HTTP ${r.status})`;
    const err = new Error(msg);
    err.status = r.status;
    err.raw = data;
    throw err;
  }
  return data;
}

/** Responses API 응답에서 본문 텍스트만 뽑기 */
function extractText(resp) {
  if (typeof resp.output_text === "string" && resp.output_text.trim()) {
    return resp.output_text;
  }
  const parts = [];
  for (const item of resp.output || []) {
    for (const c of item.content || []) {
      if (typeof c.text === "string") parts.push(c.text);
    }
  }
  return parts.join("\n").trim();
}

/** Responses API 응답에서 url_citation 주석 수집 */
function extractCitations(resp) {
  const seen = new Map();
  for (const item of resp.output || []) {
    for (const c of item.content || []) {
      for (const a of c.annotations || []) {
        if (a.type !== "url_citation" || !a.url) continue;
        if (seen.has(a.url)) continue;
        seen.set(a.url, { url: a.url, title: a.title || a.url });
      }
    }
  }
  return [...seen.values()];
}

/** 도메인으로 소스 종류 추정 */
function classify(url) {
  let host = "";
  try {
    host = new URL(url).hostname.replace(/^www\./, "");
  } catch (e) {
    return { host: url, kind: "웹" };
  }
  const h = host.toLowerCase();
  const is = (...xs) => xs.some((x) => h.includes(x));
  if (is("youtube.com", "youtu.be", "vimeo.com")) return { host, kind: "영상" };
  if (
    is("arxiv.org", "pubmed", "ncbi.nlm.nih.gov", "doi.org", "sciencedirect",
      "springer", "nature.com", "wiley", "frontiersin", "mdpi.com", "plos",
      "researchgate", "scholar.google", "biorxiv", "tandfonline", "sagepub")
  ) {
    return { host, kind: "논문" };
  }
  if (is(".edu", ".ac.", ".gov", ".go.kr", ".or.kr", "vet", "avma.org", "avsab",
    "aaha.org", "akc.org", "rspca", "aspca.org", "wsava", "iaabc", "apdt",
    "ccpdt", "fediaf", "thekennelclub", "bluecross", "dogstrust", "battersea",
    "petmd", "vin.com", "merckvetmanual", "cornell.edu", "ufaw")) {
    return { host, kind: "기관/전문" };
  }
  if (is("blog", "tistory", "naver.com", "brunch", "medium.com", "velog",
    "wordpress", "substack")) {
    return { host, kind: "블로그" };
  }
  return { host, kind: "웹" };
}

function profileLine(p) {
  p = p || {};
  const name = p.name || "코코";
  const breed = p.breed || "푸들(토이푸들)";
  const sex = p.sex || "수컷";
  const age = p.age == null ? 4 : p.age;
  const weight = p.weight || "5.2kg";
  const notes = p.notes || "실내 생활, 중성화 완료, 하루 2회 산책";
  return [
    `- 이름: ${name}`,
    `- 견종: ${breed}`,
    `- 성별: ${sex}`,
    `- 나이: 만 ${age}세`,
    `- 체중: ${weight}`,
    `- 특이사항: ${notes}`,
  ].join("\n");
}

// ---------------------------------------------------------------- STT
async function handleSTT(req, res) {
  const t0 = Date.now();
  const body = await readBody(req);
  if (!body.audio) return sendJSON(res, 400, { error: "오디오 데이터가 없습니다." });

  const buf = Buffer.from(body.audio, "base64");
  const mime = body.mime || "audio/webm";
  // OpenAI는 확장자로 포맷을 판별하므로 MIME과 맞춰 준다
  const EXT_BY_MIME = [
    ["webm", "webm"], ["ogg", "ogg"], ["opus", "ogg"],
    ["mp4", "mp4"], ["m4a", "m4a"], ["aac", "m4a"],
    ["wav", "wav"], ["x-wav", "wav"], ["flac", "flac"],
    ["mpeg", "mp3"], ["mp3", "mp3"],
  ];
  const hit = EXT_BY_MIME.find(([k]) => mime.toLowerCase().includes(k));
  const ext = hit ? hit[1] : "webm";

  // ⚠ prompt(도메인 힌트)는 절대 넣지 말 것.
  // 실측 결과, 어휘 목록만 넣어도 모델이 그 목록을 그대로 받아쓰거나 없는 질문을 지어낸다.
  // (영어 음성 입력 시 "배변 훈련은 어떻게 시작하나요?" 를 창작함)
  // language=ko 만으로 한국어 인식 정확도는 충분하다.
  async function transcribe(model) {
    const form = new FormData();
    form.append("file", new Blob([buf], { type: mime }), `speech.${ext}`);
    form.append("model", model);
    form.append("language", "ko");

    const r = await fetch(`${API_BASE}/audio/transcriptions`, {
      method: "POST",
      headers: { Authorization: `Bearer ${API_KEY}` },
      body: form,
    });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) {
      const e = new Error((data && data.error && data.error.message) || "음성 인식에 실패했습니다.");
      e.status = r.status;
      throw e;
    }
    return (data.text || "").trim();
  }

  const hasHangul = (s) => /[가-힣]/.test(s);

  let text;
  let usedModel = STT_MODEL;
  let retried = false;
  try {
    text = await transcribe(STT_MODEL);

    // 한글이 하나도 없으면 언어를 잘못 잡은 것 → whisper-1로 한 번 더 시도
    if (text && !hasHangul(text) && STT_MODEL !== "whisper-1") {
      const second = await transcribe("whisper-1").catch(() => "");
      if (second && hasHangul(second)) {
        text = second;
        usedModel = "whisper-1";
        retried = true;
      }
    }
  } catch (e) {
    return sendJSON(res, e.status || 500, { error: e.message });
  }

  // 두 모델 모두 한글을 못 냈으면 인식 실패로 처리 (엉뚱한 외국어 환각 방지)
  if (text && !hasHangul(text)) {
    return sendJSON(res, 200, {
      text: "",
      warning: "한국어로 인식되지 않았습니다. 조금 더 크고 또렷하게, 1초 이상 말해 주세요.",
      heard: text.slice(0, 80),
      model: usedModel,
      ms: Date.now() - t0,
      bytes: buf.length,
    });
  }

  sendJSON(res, 200, {
    text,
    model: usedModel,
    retried,
    ms: Date.now() - t0,
    bytes: buf.length,
  });
}

// ---------------------------------------------------------------- ASK
const ASK_SYSTEM = `너는 반려견 행동 교정 전문가(반려동물 행동학 + 임상 수의학 지식 보유)다.
아래 규칙을 지켜 한국어로 답한다.

1. 반드시 웹 검색 도구로 최신 자료를 찾아본 뒤 답한다.
2. 소스 종류를 가리지 않는다: 학술 논문, 수의/행동학 기관 가이드, 전문가 블로그, 유튜브 영상 모두 활용한다.
   가능하면 서로 성격이 다른 소스를 3개 이상 참고한다.
3. 답변 구조(마크다운 헤딩 그대로 사용):
## 한 줄 요약
## 왜 그런가
## 이렇게 해보세요
## 하지 말아야 할 것
## 병원 상담이 필요한 신호
4. 분량을 반드시 지킨다. 짧고 밀도 높게 쓴다.
   - 한 줄 요약: 2문장 이내
   - 왜 그런가: 불릿 3개 이하, 각 1~2문장
   - 이렇게 해보세요: 번호 4~5단계, 각 2문장 이내. 각 단계에 소요 기간이나 반복 횟수를 넣는다.
   - 하지 말아야 할 것: 불릿 3개 이하, 각 1문장
   - 병원 상담이 필요한 신호: 불릿 3개 이하, 각 1문장
   - 전체 900자를 넘기지 않는다.
5. 위 5개 섹션 외에 어떤 것도 쓰지 않는다.
   특히 마지막에 프로필을 다시 요약하는 문단, "도와드릴게요" 같은 맺음말, 총평 문단을 절대 붙이지 않는다.
6. 프로필에 적힌 사실만 사용한다. 프로필에 없는 증상이나 병명을 있다고 전제하지 않는다.
   (예: 프로필에 분리불안이 없으면 분리불안이 있다고 단정하지 않는다. 가능성으로만 언급하되 확인이 필요하다고 쓴다.)
7. 견종·나이·체중에 맞춰 구체적으로 조정한다. 소형견·성견 특성을 반영한다.
8. 근거가 약하거나 전문가 의견이 갈리는 부분은 "논란 있음"이라고 솔직히 밝힌다.
9. 진단이나 투약을 단정하지 않는다. 의학적 판단이 필요하면 수의사 상담을 권한다.`;

const ASK_SYSTEM_NOSEARCH = ASK_SYSTEM.replace(
  "1. 반드시 웹 검색 도구로 최신 자료를 찾아본 뒤 답한다.",
  "1. 지금은 웹 검색을 쓸 수 없다. 알고 있는 지식으로 답하되 근거의 한계를 답변 첫 줄에 명시한다."
);

function askInput(system, question, profile) {
  return [
    { role: "system", content: system },
    {
      role: "user",
      content: `[반려견 프로필]\n${profileLine(profile)}\n\n[보호자 질문]\n${question}`,
    },
  ];
}

async function handleAsk(req, res) {
  const t0 = Date.now();
  const body = await readBody(req);
  const question = (body.question || "").trim();
  if (!question) return sendJSON(res, 400, { error: "질문이 비어 있습니다." });

  // 모델/계정에 따라 웹검색 도구 이름이 다를 수 있어 순차 시도
  let resp = null;
  let usedTool = null;
  let lastErr = null;
  for (const toolType of ["web_search", "web_search_preview"]) {
    try {
      resp = await openai("/responses", {
        model: MODEL,
        tools: [{ type: toolType }],
        tool_choice: "auto",
        input: askInput(ASK_SYSTEM, question, body.profile),
      });
      usedTool = toolType;
      break;
    } catch (e) {
      lastErr = e;
      if (!/tool|unsupported|invalid_value|not supported|hosted/i.test(e.message)) break;
    }
  }

  // 웹검색이 아예 불가하면 검색 없이라도 답변
  if (!resp) {
    try {
      resp = await openai("/responses", {
        model: MODEL,
        input: askInput(ASK_SYSTEM_NOSEARCH, question, body.profile),
      });
      usedTool = "none";
    } catch (e) {
      return sendJSON(res, e.status || 500, {
        error: e.message || (lastErr && lastErr.message) || "답변 생성 실패",
      });
    }
  }

  const citations = extractCitations(resp).map((c) => Object.assign({}, c, classify(c.url)));
  sendJSON(res, 200, {
    answer: extractText(resp),
    citations,
    model: MODEL,
    tool: usedTool,
    ms: Date.now() - t0,
    usage: resp.usage || null,
  });
}

// ---------------------------------------------------------------- EVAL
const EVAL_SYSTEM = `너는 반려견 훈련 정보의 품질을 감사하는 심사관이다.
[질문], [답변], [참고 소스 목록]을 보고 냉정하게 평가한다. 후하게 주지 마라.

평가 기준
- reliability(신뢰도 0~100): 소스의 권위·다양성·최신성. 논문이나 수의/행동학 기관 자료가 섞여 있으면 가점,
  개인 블로그·영상만이면 감점, 소스가 아예 없으면 40점 이하.
- accuracy(정확성 0~100): 현대 반려견 행동학(긍정강화 중심) 합의와 일치하는가.
  지배이론, 체벌 권장, 과잉 일반화가 있으면 크게 감점. 프로필(견종/나이/체중) 반영 여부도 포함.
- checks: 답변의 핵심 주장 2~4개를 뽑아 각각 "근거 충분"/"부분 근거"/"근거 부족" 중 하나로 판정.
- risks: 보호자가 주의해야 할 위험 요소(없으면 빈 배열).

반드시 아래 JSON 형식만 출력한다. 다른 텍스트 금지.
{
  "reliability": 0-100 정수,
  "reliability_reason": "한국어 1~2문장",
  "accuracy": 0-100 정수,
  "accuracy_reason": "한국어 1~2문장",
  "source_diversity": "높음" 또는 "보통" 또는 "낮음",
  "risks": ["문자열"],
  "checks": [{"claim": "문자열", "verdict": "근거 충분", "note": "문자열"}],
  "summary": "한 줄 총평"
}`;

async function handleEval(req, res) {
  const t0 = Date.now();
  const body = await readBody(req);
  const question = body.question || "";
  const answer = body.answer || "";
  const citations = body.citations || [];
  if (!answer) return sendJSON(res, 400, { error: "평가할 답변이 없습니다." });

  const srcList = citations.length
    ? citations.map((c, i) => `${i + 1}. [${c.kind}] ${c.title} (${c.host})`).join("\n")
    : "(참고 소스 없음 — 모델 내부 지식만 사용)";

  let resp;
  try {
    resp = await openai("/responses", {
      model: MODEL,
      text: { format: { type: "json_object" } },
      input: [
        { role: "system", content: EVAL_SYSTEM },
        {
          role: "user",
          content: `[반려견 프로필]\n${profileLine(body.profile)}\n\n[질문]\n${question}\n\n[답변]\n${answer}\n\n[참고 소스 목록]\n${srcList}`,
        },
      ],
    });
  } catch (e) {
    return sendJSON(res, e.status || 500, { error: e.message });
  }

  let parsed;
  try {
    parsed = JSON.parse(extractText(resp).replace(/^```json\s*|\s*```$/g, ""));
  } catch (e) {
    return sendJSON(res, 502, { error: "평가 결과 JSON 파싱에 실패했습니다." });
  }

  sendJSON(res, 200, Object.assign({}, parsed, { ms: Date.now() - t0, model: MODEL }));
}

// ---------------------------------------------------------------- static
function serveStatic(req, res) {
  let p = decodeURIComponent(req.url.split("?")[0]);
  if (p === "/") p = "/index.html";
  const file = path.join(PUBLIC_DIR, path.normalize(p).replace(/^[/\\]+/, ""));
  if (!file.startsWith(PUBLIC_DIR) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) {
    res.writeHead(404, { "Content-Type": "text/plain; charset=utf-8" });
    return res.end("404 Not Found");
  }
  res.writeHead(200, {
    "Content-Type": MIME[path.extname(file)] || "application/octet-stream",
    "Cache-Control": "no-cache",
  });
  fs.createReadStream(file).pipe(res);
}

// ---------------------------------------------------------------- server
const routes = {
  "/api/stt": handleSTT,
  "/api/ask": handleAsk,
  "/api/eval": handleEval,
};

async function handle(req, res) {
  const url = req.url.split("?")[0];
  if (url === "/api/health") {
    return sendJSON(res, 200, { ok: true, hasKey: !!API_KEY, model: MODEL, stt: STT_MODEL });
  }
  if (routes[url]) {
    if (req.method !== "POST") return sendJSON(res, 405, { error: "POST만 허용됩니다." });
    if (!API_KEY) {
      return sendJSON(res, 500, {
        error: "OPENAI_API_KEY가 설정되지 않았습니다. dog-trainer-demo/.env 파일을 확인하세요.",
      });
    }
    try {
      return await routes[url](req, res);
    } catch (e) {
      console.error(e);
      return sendJSON(res, 500, { error: e.message || "서버 오류" });
    }
  }
  serveStatic(req, res);
}

function lanIPs() {
  const out = [];
  for (const list of Object.values(os.networkInterfaces())) {
    for (const i of list || []) {
      if (i.family === "IPv4" && !i.internal) out.push(i.address);
    }
  }
  return out;
}

/** certs/ 에 인증서가 있으면 HTTPS도 함께 띄운다 (휴대폰 마이크용) */
function tlsOptions() {
  const key = path.join(CERT_DIR, "key.pem");
  const cert = path.join(CERT_DIR, "cert.pem");
  if (!fs.existsSync(key) || !fs.existsSync(cert)) return null;
  try {
    return { key: fs.readFileSync(key), cert: fs.readFileSync(cert) };
  } catch (e) {
    console.error("  ⚠ 인증서를 읽지 못했습니다:", e.message);
    return null;
  }
}

const ips = lanIPs();
const tls = tlsOptions();

http.createServer(handle).listen(PORT, "0.0.0.0", () => {
  console.log("");
  console.log("  🐩 강아지 훈련 Q&A 데모");
  console.log(`  ▶ 이 PC에서:  http://localhost:${PORT}`);

  if (tls) {
    https.createServer(tls, handle).listen(HTTPS_PORT, "0.0.0.0", () => {
      for (const ip of ips) console.log(`  ▶ 휴대폰에서: https://${ip}:${HTTPS_PORT}   (마이크 사용 가능)`);
      console.log("     └ 인증서 경고가 뜨면 [고급] → [계속] 을 누르세요. 자체 서명 인증서라 정상입니다.");
      tail();
    });
  } else {
    for (const ip of ips) console.log(`  ▶ 휴대폰에서: http://${ip}:${PORT}   (⚠ 마이크 불가 — 텍스트 입력만)`);
    console.log("     └ 휴대폰에서 마이크를 쓰려면 `node make-cert.js` 실행 후 서버를 다시 켜세요.");
    tail();
  }
});

function tail() {
  console.log("");
  console.log(`  답변 모델: ${MODEL} / 음성인식: ${STT_MODEL}`);
  console.log(API_KEY ? "  API 키: 확인됨" : "  ⚠ OPENAI_API_KEY 미설정 — .env 파일을 만들어 주세요.");
  console.log("");
}
