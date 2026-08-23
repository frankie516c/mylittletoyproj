"""웹 페이지를 본문만 남긴 마크다운으로 변환한다.

trafilatura로 보일러플레이트(네비/광고/푸터)를 걷어내고 헤딩 구조를 보존한 마크다운을
뽑는다. 헤딩이 살아 있어야 뒤의 구조 인식 청킹이 의미를 갖는다.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

import trafilatura

UA = "dog-training-rag-demo/0.1 (personal learning project; contact via repo)"


def fetch_markdown(url: str) -> str:
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        raise RuntimeError(f"페이지를 가져오지 못했습니다: {url}")
    md = trafilatura.extract(
        downloaded,
        output_format="markdown",
        include_formatting=True,
        include_tables=True,
        include_links=False,
        include_comments=False,
        favor_recall=True,
    )
    if not md:
        raise RuntimeError(f"본문 추출에 실패했습니다: {url}")
    return normalize(md)


def normalize(md: str) -> str:
    """빈 줄 정리, 헤딩 앞뒤 공백 보정 등 최소 정규화."""
    md = md.replace("\r\n", "\n").replace("\xa0", " ")
    md = re.sub(r"[ \t]+\n", "\n", md)
    md = re.sub(r"\n{3,}", "\n\n", md)
    md = re.sub(r"(?<!\n)\n(#{1,6} )", r"\n\n\1", md)
    return md.strip() + "\n"


def frontmatter(source: dict) -> str:
    fetched = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines = ["---"]
    for key in ("doc_id", "url", "title", "title_ko", "publisher", "evidence_tier", "lang", "topic"):
        lines.append(f"{key}: {source[key]}")
    lines.append(f"fetched_at: {fetched}")
    lines.append("---")
    return "\n".join(lines) + "\n\n"


def strip_frontmatter(text: str) -> str:
    if text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        if end != -1:
            return text[end + 5 :].lstrip("\n")
    return text
