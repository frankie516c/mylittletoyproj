/**
 * 로컬 개발용 자체 서명 인증서 생성기
 *
 *   node make-cert.js
 *
 * certs/key.pem, certs/cert.pem 을 만들고, 이 PC의 모든 랜 IP를 SAN에 넣는다.
 * 휴대폰에서 https://<랜IP>:3443 으로 접속하면 마이크(getUserMedia)가 동작한다.
 */
const { execFileSync } = require("child_process");
const fs = require("fs");
const os = require("os");
const path = require("path");

const CERT_DIR = path.join(__dirname, "certs");

/** openssl 실행 파일 찾기 (PowerShell에선 PATH에 없을 수 있어 Git 설치 경로도 뒤진다) */
function findOpenssl() {
  const candidates = [
    "openssl",
    "C:\\Program Files\\Git\\mingw64\\bin\\openssl.exe",
    "C:\\Program Files\\Git\\usr\\bin\\openssl.exe",
    "C:\\Program Files (x86)\\Git\\mingw64\\bin\\openssl.exe",
    "/usr/bin/openssl",
  ];
  for (const c of candidates) {
    try {
      execFileSync(c, ["version"], { stdio: "pipe" });
      return c;
    } catch (e) { /* 다음 후보 */ }
  }
  return null;
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

const openssl = findOpenssl();
if (!openssl) {
  console.error("openssl을 찾지 못했습니다. Git Bash에서 실행하거나 openssl을 설치해 주세요.");
  process.exit(1);
}

const ips = lanIPs();
const san = [
  "DNS:localhost",
  "IP:127.0.0.1",
  ...ips.map((ip) => `IP:${ip}`),
].join(",");

fs.mkdirSync(CERT_DIR, { recursive: true });

execFileSync(openssl, [
  "req", "-x509", "-newkey", "rsa:2048", "-nodes",
  "-keyout", path.join(CERT_DIR, "key.pem"),
  "-out", path.join(CERT_DIR, "cert.pem"),
  "-days", "825",
  "-subj", "/CN=dog-trainer-demo",
  "-addext", `subjectAltName=${san}`,
], { stdio: "pipe" });

console.log("");
console.log("  인증서를 만들었습니다: certs/cert.pem, certs/key.pem");
console.log("  포함된 주소: " + san.replace(/,/g, ", "));
console.log("");
console.log("  이제 `node server.js` 로 서버를 켜고, 휴대폰에서 아래 주소로 접속하세요:");
for (const ip of ips) console.log(`    https://${ip}:3443`);
console.log("");
console.log("  ※ 자체 서명 인증서라 '연결이 비공개로 설정되어 있지 않습니다' 경고가 뜹니다.");
console.log("    [고급] → [안전하지 않음(계속)] 을 누르면 됩니다. 그 뒤 마이크 권한 창이 정상적으로 뜹니다.");
console.log("");
