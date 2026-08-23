/* 멍톡 데모 — 음성 녹음(2회 클릭) → STT → 웹검색 답변 → 품질 분석 */

const PROFILE = {
  name: "코코",
  breed: "토이푸들",
  sex: "수컷",
  age: 4,
  weight: "5.2kg",
  notes: "실내 생활, 중성화 완료, 하루 2회 산책",
};

/* 펼친 채로 둘 섹션 (나머지는 접어서 보여준다) */
const OPEN_SECTIONS = ["요약", "이렇게"];

const KIND_SLUG = { "논문": "paper", "영상": "video", "블로그": "blog", "기관/전문": "org", "웹": "web" };
const slug = (k) => KIND_SLUG[k] || "web";

const $ = (s) => document.querySelector(s);
const el = (tag, cls, html) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (html != null) n.innerHTML = html;
  return n;
};
const esc = (s) => String(s == null ? "" : s)
  .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
  .replace(/"/g, "&quot;");

const chat = $("#chat");
const micBtn = $("#mic-btn");
const micIcon = $("#mic-icon");
const micLabel = $("#mic-label");
const micTimer = $("#mic-timer");
const levelBars = $("#level-bars");

/* ───────────────────────── 레벨 미터 막대 생성 ───────────────────────── */
const BAR_COUNT = 22;
for (let i = 0; i < BAR_COUNT; i++) levelBars.appendChild(document.createElement("i"));
const bars = [...levelBars.children];

/* ───────────────────────── 마크다운(최소) ───────────────────────── */
function md(src) {
  const lines = String(src || "").replace(/\r/g, "").split("\n");
  let out = "";
  let list = null; // "ul" | "ol"
  const closeList = () => { if (list) { out += `</${list}>`; list = null; } };
  const inline = (t) => esc(t)
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>")
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>')
    // 각주형 인용 표기 제거용 잔여 정리
    .replace(/\(\s*\)/g, "");

  for (const raw of lines) {
    const line = raw.trimEnd();
    if (!line.trim()) { closeList(); continue; }

    let m;
    if ((m = line.match(/^###\s+(.*)$/))) { closeList(); out += `<h3>${inline(m[1])}</h3>`; continue; }
    if ((m = line.match(/^##\s+(.*)$/))) { closeList(); out += `<h2>${inline(m[1])}</h2>`; continue; }
    if ((m = line.match(/^#\s+(.*)$/))) { closeList(); out += `<h2>${inline(m[1])}</h2>`; continue; }
    if ((m = line.match(/^\s*[-*•]\s+(.*)$/))) {
      if (list !== "ul") { closeList(); out += "<ul>"; list = "ul"; }
      out += `<li>${inline(m[1])}</li>`; continue;
    }
    if ((m = line.match(/^\s*\d+[.)]\s+(.*)$/))) {
      if (list !== "ol") { closeList(); out += "<ol>"; list = "ol"; }
      out += `<li>${inline(m[1])}</li>`; continue;
    }
    closeList();
    out += `<p>${inline(line)}</p>`;
  }
  closeList();
  return out;
}

/**
 * ## 헤딩 단위로 잘라서 접을 수 있는 섹션으로 재구성한다.
 * 요약과 실행 단계는 펼친 채로, 배경·주의·병원 신호는 접어둔다.
 */
function sectionize(html) {
  const src = document.createElement("div");
  src.innerHTML = html;

  const out = document.createDocumentFragment();
  let body = null;

  const flush = () => { body = null; };

  for (const node of [...src.childNodes]) {
    if (node.nodeName === "H2") {
      flush();
      const title = node.textContent.trim();
      const open = OPEN_SECTIONS.some((k) => title.includes(k));

      const det = document.createElement("details");
      det.className = "sec";
      if (open) det.open = true;

      const sum = document.createElement("summary");
      sum.textContent = title;
      det.appendChild(sum);

      body = document.createElement("div");
      body.className = "sec-body";
      det.appendChild(body);

      out.appendChild(det);
      continue;
    }
    (body || out).appendChild(node);
  }

  // ## 헤딩이 하나도 없으면 원본 그대로
  return out.querySelector("details") ? out : null;
}

/* ───────────────────────── 채팅 렌더 ───────────────────────── */
function clearEmpty() {
  const e = $("#empty-state");
  if (e) e.remove();
}
function scrollDown() {
  requestAnimationFrame(() => { chat.scrollTop = chat.scrollHeight; });
}

function addUser(text, meta) {
  clearEmpty();
  const wrap = el("div", "msg user");
  wrap.appendChild(el("div", "who", "🧑"));
  const b = el("div", "bubble");
  b.appendChild(el("div", null, esc(text)));
  if (meta) b.appendChild(el("div", "stt-meta", esc(meta)));
  wrap.appendChild(b);
  chat.appendChild(wrap);
  scrollDown();
  return wrap;
}

function addBot() {
  clearEmpty();
  const wrap = el("div", "msg bot");
  wrap.appendChild(el("div", "who", "🐩"));
  const b = el("div", "bubble");
  b.innerHTML = '<span class="typing"><i></i><i></i><i></i></span><span class="step">준비 중…</span>';
  wrap.appendChild(b);
  chat.appendChild(wrap);
  scrollDown();
  return {
    node: wrap,
    bubble: b,
    step(t) {
      const s = b.querySelector(".step");
      if (s) s.textContent = t;
      scrollDown();
    },
    error(msg) {
      b.classList.add("err");
      b.innerHTML = `<b>문제가 생겼어요</b><p>${esc(msg)}</p>`;
      scrollDown();
    },
    answer(text, citations) {
      b.classList.remove("err");
      const secs = sectionize(md(text));
      if (secs) { b.innerHTML = ""; b.appendChild(secs); }
      else b.innerHTML = md(text);
      if (citations && citations.length) {
        const box = el("div", "cites");
        box.appendChild(el("div", "cites-title", `참고한 자료 ${citations.length}건`));
        for (const c of citations) {
          const a = el("a", "cite");
          a.href = c.url; a.target = "_blank"; a.rel = "noopener";
          a.innerHTML = `<span class="k k-${slug(c.kind)}">${esc(c.kind)}</span>` +
            `<span class="t">${esc(c.title)}</span>`;
          box.appendChild(a);
        }
        b.appendChild(box);
      }
      scrollDown();
    },
  };
}

/* ───────────────────────── 분석 패널 ───────────────────────── */
const CIRC = 2 * Math.PI * 42;

function setGauge(key, value) {
  const g = document.querySelector(`.gauge[data-key="${key}"]`);
  const fg = g.querySelector(".g-fg");
  const num = g.querySelector(".num");
  const v = Math.max(0, Math.min(100, Math.round(value)));
  fg.style.strokeDashoffset = String(CIRC * (1 - v / 100));
  fg.style.stroke = v >= 80 ? "var(--green)" : v >= 60 ? "var(--amber)" : "var(--red)";
  // 카운트업
  const start = performance.now();
  const from = parseInt(num.textContent, 10) || 0;
  const tick = (now) => {
    const p = Math.min(1, (now - start) / 800);
    num.textContent = String(Math.round(from + (v - from) * (1 - Math.pow(1 - p, 3))));
    if (p < 1) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

function speedScore(totalMs) {
  // 4초 이하 100점, 22초 이상 25점, 그 사이 선형
  const s = 100 - ((totalMs - 4000) / (22000 - 4000)) * 75;
  return Math.max(20, Math.min(100, s));
}

function renderAnalysis({ timings, evalData, citations }) {
  $("#analysis-empty").classList.add("hidden");
  $("#analysis").classList.remove("hidden");

  const total = (timings.stt || 0) + (timings.ask || 0) + (timings.eval || 0);
  const spd = speedScore(total);

  setGauge("reliability", evalData.reliability ?? 0);
  setGauge("accuracy", evalData.accuracy ?? 0);
  setGauge("speed", spd);

  $("#verdict").textContent = evalData.summary || "—";

  // 속도 분해 막대
  const seg = [
    { k: "stt", cls: "t-stt", label: "음성인식", ms: timings.stt || 0 },
    { k: "ask", cls: "t-ask", label: "검색+답변", ms: timings.ask || 0 },
    { k: "eval", cls: "t-eval", label: "품질평가", ms: timings.eval || 0 },
  ].filter((s) => s.ms > 0);
  const timing = $("#timing");
  timing.innerHTML = "";
  for (const s of seg) {
    const pct = (s.ms / total) * 100;
    const span = el("span", s.cls, pct > 14 ? `${s.label}` : "");
    span.style.width = pct + "%";
    span.title = `${s.label} ${(s.ms / 1000).toFixed(1)}초`;
    timing.appendChild(span);
  }
  $("#timing-total").innerHTML =
    seg.map((s) => `<span>${s.label} <b>${(s.ms / 1000).toFixed(1)}s</b></span>`).join("") +
    `<span>총 <b>${(total / 1000).toFixed(1)}초</b></span>`;

  // 신뢰도
  $("#reliability-reason").textContent =
    `${evalData.reliability_reason || ""} (소스 다양성: ${evalData.source_diversity || "-"})`;

  const mix = {};
  for (const c of citations || []) mix[c.kind] = (mix[c.kind] || 0) + 1;
  const mixBox = $("#src-mix");
  mixBox.innerHTML = Object.keys(mix).length
    ? Object.entries(mix)
        .map(([k, n]) => `<span class="k k-${slug(k)}">${esc(k)} ${n}</span>`)
        .join("")
    : '<span class="k k-web">참고 소스 없음</span>';

  const list = $("#src-list");
  list.innerHTML = "";
  for (const c of (citations || []).slice(0, 8)) {
    const li = document.createElement("li");
    li.innerHTML = `<a href="${esc(c.url)}" target="_blank" rel="noopener">${esc(c.title)}` +
      `<span class="host"> ${esc(c.host)}</span></a>`;
    list.appendChild(li);
  }

  // 정확성
  $("#accuracy-reason").textContent = evalData.accuracy_reason || "";
  const checks = $("#checks");
  checks.innerHTML = "";
  for (const c of evalData.checks || []) {
    const v = c.verdict || "";
    const cls = v.includes("충분") ? "v-full" : v.includes("부분") ? "v-part" : "v-none";
    const li = document.createElement("li");
    li.innerHTML = `<span class="v ${cls}">${esc(v)}</span>${esc(c.claim)}` +
      (c.note ? `<span class="note">${esc(c.note)}</span>` : "");
    checks.appendChild(li);
  }

  // 위험
  const riskSec = $("#risk-section");
  const risks = $("#risks");
  risks.innerHTML = "";
  if (evalData.risks && evalData.risks.length) {
    riskSec.classList.remove("hidden");
    for (const r of evalData.risks) {
      const li = document.createElement("li");
      li.textContent = r;
      risks.appendChild(li);
    }
  } else {
    riskSec.classList.add("hidden");
  }
}

/* ───────────────────────── API ───────────────────────── */
async function post(url, body) {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.error || `요청 실패 (${r.status})`);
  return data;
}

/* ───────────────────────── 질문 처리 파이프라인 ───────────────────────── */
let busy = false;

async function ask(question, sttMs) {
  if (busy) return;
  busy = true;
  const bot = addBot();
  const timings = { stt: sttMs || 0, ask: 0, eval: 0 };

  try {
    bot.step("블로그 · 유튜브 · 논문 검색 중…");
    const t1 = performance.now();
    const res = await post("/api/ask", { question, profile: PROFILE });
    timings.ask = Math.round(performance.now() - t1);
    bot.answer(res.answer, res.citations);

    if (res.tool === "none") {
      bot.bubble.appendChild(
        el("div", "stt-meta", "※ 이 답변은 웹 검색 없이 모델 지식만으로 생성되었습니다.")
      );
    }

    // 품질 평가
    const t2 = performance.now();
    const ev = await post("/api/eval", {
      question,
      answer: res.answer,
      citations: res.citations,
      profile: PROFILE,
    });
    timings.eval = Math.round(performance.now() - t2);

    renderAnalysis({ timings, evalData: ev, citations: res.citations });
  } catch (e) {
    bot.error(e.message);
  } finally {
    busy = false;
    setMicState("idle");
  }
}

/* ───────────────────────── 녹음 (2회 클릭) ───────────────────────── */
let mediaRecorder = null;
let stream = null;
let audioCtx = null;
let analyser = null;
let rafId = null;
let chunks = [];
let startedAt = 0;
let timerId = null;
let peakLevel = 0; // 녹음 중 관측된 최대 음량 (0~1) — 무음 녹음 진단용

function setMicState(state) {
  micBtn.classList.remove("recording", "busy");
  if (state === "recording") {
    micBtn.classList.add("recording");
    micIcon.textContent = "⏹";
    micLabel.textContent = "듣는 중… 다 말했으면 한 번 더";
  } else if (state === "busy") {
    micBtn.classList.add("busy");
    micIcon.textContent = "⋯";
    micLabel.textContent = "처리 중…";
  } else {
    micIcon.textContent = "🎙️";
    micLabel.textContent = busy ? "답변 생성 중…" : "눌러서 말하기 시작";
    micTimer.textContent = "00:00";
    resetBars();
  }
}

function resetBars() {
  for (const b of bars) { b.style.height = "4px"; b.style.background = "#e2dcd2"; }
}

function fmt(ms) {
  const s = Math.floor(ms / 1000);
  return `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
}

function pickMime() {
  const cands = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus", "audio/mp4"];
  for (const c of cands) {
    if (window.MediaRecorder && MediaRecorder.isTypeSupported(c)) return c;
  }
  return "";
}

function visualize() {
  const buf = new Uint8Array(analyser.frequencyBinCount);
  const loop = () => {
    analyser.getByteFrequencyData(buf);
    const step = Math.floor(buf.length / BAR_COUNT / 2) || 1;
    for (let i = 0; i < BAR_COUNT; i++) {
      const v = buf[i * step] / 255;
      if (v > peakLevel) peakLevel = v;
      const h = 4 + v * 22;
      bars[i].style.height = h.toFixed(1) + "px";
      bars[i].style.background = v > 0.08 ? "var(--brand)" : "#e2dcd2";
    }
    rafId = requestAnimationFrame(loop);
  };
  loop();
}

/** 브라우저가 마이크를 쓸 수 있는 상태인지 (HTTPS 또는 localhost 여야 함) */
function micSupport() {
  const secure = window.isSecureContext || location.hostname === "localhost" || location.hostname === "127.0.0.1";
  if (!secure) return { ok: false, reason: "insecure" };
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) return { ok: false, reason: "unsupported" };
  if (!window.MediaRecorder) return { ok: false, reason: "unsupported" };
  return { ok: true };
}

async function startRecording() {
  const sup = micSupport();
  if (!sup.ok) { showMicBlocked(sup.reason); return; }

  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true },
    });
  } catch (e) {
    const n = e && e.name;
    let msg;
    if (n === "NotAllowedError" || n === "SecurityError") {
      msg = "마이크 권한이 거부되었습니다.\n\n주소창 왼쪽의 자물쇠/설정 아이콘 → 사이트 설정 → 마이크 → 허용으로 바꾼 뒤 새로고침해 주세요.";
    } else if (n === "NotFoundError" || n === "DevicesNotFoundError") {
      msg = "마이크 장치를 찾지 못했습니다. 마이크가 연결되어 있는지 확인해 주세요.";
    } else if (n === "NotReadableError") {
      msg = "다른 앱이 마이크를 사용 중입니다. 해당 앱을 끄고 다시 시도해 주세요.";
    } else {
      msg = "마이크를 열 수 없습니다.\n(" + (e && e.message ? e.message : e) + ")";
    }
    alert(msg);
    setMicState("idle");
    return;
  }

  const mime = pickMime();
  chunks = [];
  peakLevel = 0;
  mediaRecorder = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
  mediaRecorder.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
  mediaRecorder.onstop = handleStop;
  mediaRecorder.start();

  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  analyser = audioCtx.createAnalyser();
  analyser.fftSize = 256;
  audioCtx.createMediaStreamSource(stream).connect(analyser);
  visualize();

  startedAt = Date.now();
  micTimer.textContent = "00:00";
  timerId = setInterval(() => {
    micTimer.textContent = fmt(Date.now() - startedAt);
    if (Date.now() - startedAt > 60000) stopRecording(); // 안전장치: 60초
  }, 200);

  setMicState("recording");
}

function stopRecording() {
  if (!mediaRecorder || mediaRecorder.state === "inactive") return;
  mediaRecorder.stop();
  clearInterval(timerId);
  cancelAnimationFrame(rafId);
  resetBars();
  setMicState("busy");
}

async function handleStop() {
  if (stream) stream.getTracks().forEach((t) => t.stop());
  if (audioCtx) audioCtx.close().catch(() => {});
  stream = null; audioCtx = null;

  const durMs = Date.now() - startedAt;
  const type = (mediaRecorder && mediaRecorder.mimeType) || "audio/webm";
  const blob = new Blob(chunks, { type });

  if (blob.size < 2000 || durMs < 900) {
    setMicState("idle");
    micLabel.textContent = "너무 짧아요. 1초 이상 말해 주세요";
    return;
  }

  try {
    const b64 = await blobToBase64(blob);
    const t0 = performance.now();
    const res = await post("/api/stt", { audio: b64, mime: type });
    const sttMs = Math.round(performance.now() - t0);

    if (!res.text) {
      setMicState("idle");
      micLabel.textContent = res.warning
        ? "한국어로 안 들렸어요. 다시 말해 주세요"
        : "말소리를 못 들었어요. 다시 시도해 주세요";
      if (res.warning) {
        clearEmpty();
        const note = el("div", "msg bot");
        note.appendChild(el("div", "who", "🐩"));
        const b = el("div", "bubble err");
        b.innerHTML = `<b>음성을 한국어로 인식하지 못했어요</b><p>${esc(res.warning)}</p>` +
          (res.heard ? `<p class="stt-meta">들린 내용: ${esc(res.heard)}</p>` : "") +
          (peakLevel < 0.06 ? '<p class="stt-meta">🔈 녹음된 소리가 매우 작습니다. 마이크에 더 가까이서 말해 주세요.</p>' : "");
        note.appendChild(b);
        chat.appendChild(note);
        scrollDown();
      }
      return;
    }
    const retag = res.retried ? " · 재시도" : "";
    addUser(res.text, `🎙️ ${(durMs / 1000).toFixed(1)}초 발화 · ${sttMs}ms · ${res.model}${retag}`);
    setMicState("idle");
    ask(res.text, sttMs);
  } catch (e) {
    setMicState("idle");
    addUser("(음성 인식 실패)");
    const bot = addBot();
    bot.error(e.message);
  }
}

function blobToBase64(blob) {
  return new Promise((resolve, reject) => {
    const fr = new FileReader();
    fr.onload = () => resolve(String(fr.result).split(",")[1]);
    fr.onerror = reject;
    fr.readAsDataURL(blob);
  });
}

/* ───────────────────────── 마이크 불가 안내 ───────────────────────── */
function showMicBlocked(reason) {
  if ($("#mic-blocked")) return;

  const httpsUrl = `https://${location.hostname}:3443`;
  const box = el("div", "mic-blocked");
  box.id = "mic-blocked";

  if (reason === "insecure") {
    box.innerHTML =
      "<b>이 주소에서는 마이크를 쓸 수 없어요</b>" +
      "<p>브라우저는 <b>HTTPS</b> 또는 <b>localhost</b>에서만 마이크를 허용합니다. " +
      "지금은 <code>http://</code>로 접속해서 권한 요청 자체가 뜨지 않습니다.</p>" +
      `<p class="fix">👉 아래 주소로 다시 접속해 보세요<br /><a href="${httpsUrl}">${httpsUrl}</a></p>` +
      '<p class="dim">인증서 경고가 뜨면 [고급] → [계속]을 누르면 됩니다.<br />' +
      "(서버에서 <code>node make-cert.js</code>를 한 번 실행해 두어야 합니다.)</p>" +
      '<p class="dim">그때까지는 아래 <b>직접 입력</b>으로 이용할 수 있어요.</p>';
  } else {
    box.innerHTML =
      "<b>이 브라우저는 녹음을 지원하지 않아요</b>" +
      "<p>최신 Chrome, Edge, Safari에서 열어 주세요. 아래 <b>직접 입력</b>은 그대로 사용할 수 있습니다.</p>";
  }

  micBtn.classList.add("busy");
  micBtn.disabled = true;
  micIcon.textContent = "🚫";
  micLabel.textContent = "마이크 사용 불가";
  micTimer.textContent = "";
  levelBars.classList.add("hidden");
  $(".mic-guide").replaceWith(box);
}

/* ───────────────────────── 이벤트 ───────────────────────── */
micBtn.addEventListener("click", () => {
  if (busy) return;
  if (mediaRecorder && mediaRecorder.state === "recording") stopRecording();
  else startRecording();
});

$("#text-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const input = $("#text-input");
  const q = input.value.trim();
  if (!q || busy) return;
  input.value = "";
  addUser(q, "⌨️ 직접 입력");
  ask(q, 0);
});

document.querySelectorAll(".chip").forEach((c) => {
  c.addEventListener("click", () => {
    if (busy) return;
    addUser(c.textContent, "⌨️ 예시 질문");
    ask(c.textContent, 0);
  });
});

$("#reset-btn").addEventListener("click", () => {
  chat.innerHTML = "";
  const e = el("div", "empty-state");
  e.id = "empty-state";
  e.innerHTML = '<div class="empty-emoji">🦴</div>' +
    "<p><b>코코</b>에 대해 궁금한 훈련법을 말해 보세요.</p>" +
    '<p class="dim">답변은 블로그 · 유튜브 · 논문 등 실제 웹 자료를 검색해서 만들어집니다.</p>';
  chat.appendChild(e);
  $("#analysis").classList.add("hidden");
  $("#analysis-empty").classList.remove("hidden");
});

/* 서버 상태 표시 */
fetch("/api/health").then((r) => r.json()).then((h) => {
  const pill = $("#engine-pill");
  if (h.hasKey) {
    pill.textContent = `${h.model} · ${h.stt}`;
    pill.classList.add("ok");
  } else {
    pill.textContent = "API 키 미설정";
    pill.classList.add("bad");
  }
}).catch(() => {});

resetBars();

/* 페이지를 열자마자 마이크 가능 여부를 검사해서, 안 되면 미리 안내한다 */
(function () {
  const sup = micSupport();
  if (!sup.ok) showMicBlocked(sup.reason);
})();
