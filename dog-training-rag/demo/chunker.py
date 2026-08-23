"""구조 인식(structure-aware) 마크다운 청킹.

마크다운 헤딩 트리를 그대로 살려 섹션 단위로 자르고, 섹션이 너무 길면 블록
(문단 / 리스트 덩어리) 경계에서만 나눈다. 그래도 넘치면 문장 경계에서 나눈다.
문장 중간에서 잘리는 일은 없다.

각 청크에는 `문서 제목 > H2 > H3` 형태의 헤딩 경로를 컨텍스트 헤더로 붙인다.
검색용 임베딩은 이 헤더를 포함한 텍스트로 만들고, 사용자에게 보여줄 때는
본문만 쓴다. (contextual chunk header의 값싼 버전)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
FENCE_RE = re.compile(r"^\s*(```|~~~)")
LIST_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|(?<=다\.)\s+|(?<=요\.)\s+")


@dataclass
class Section:
    """헤딩 하나와 그 아래 본문."""

    path: list[str]
    body: str

    @property
    def heading(self) -> str:
        return self.path[-1] if self.path else ""


@dataclass
class Chunk:
    index: int
    heading_path: list[str]
    text: str          # 본문만 (표시용)
    embed_text: str    # 헤딩 경로 헤더 + 본문 (임베딩·검색용)
    char_len: int
    part: int = 0      # 같은 섹션을 쪼갠 경우의 순번
    parts: int = 1
    meta: dict = field(default_factory=dict)


def parse_sections(markdown: str, doc_title: str) -> list[Section]:
    """마크다운을 헤딩 경로가 붙은 섹션 리스트로 분해한다."""
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []  # (level, heading)
    buf: list[str] = []
    in_fence = False

    def flush() -> None:
        body = "\n".join(buf).strip()
        buf.clear()
        if not body:
            return
        path = [doc_title] + [h for _, h in stack]
        # 문서 제목과 H1이 같은 흔한 경우처럼 연속 중복된 헤딩은 접는다
        path = [x for i, x in enumerate(path) if i == 0 or x != path[i - 1]]
        sections.append(Section(path=path, body=body))

    for line in markdown.splitlines():
        if FENCE_RE.match(line):
            in_fence = not in_fence
            buf.append(line)
            continue
        if in_fence:
            buf.append(line)
            continue

        m = HEADING_RE.match(line)
        if m:
            flush()
            level = len(m.group(1))
            heading = m.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, heading))
            continue
        buf.append(line)

    flush()
    return sections


def _split_blocks(body: str) -> list[str]:
    """문단/리스트 덩어리 단위로 나눈다. 연속된 리스트 항목은 한 덩어리로 묶는다."""
    raw_blocks = [b.strip() for b in re.split(r"\n\s*\n", body) if b.strip()]
    blocks: list[str] = []
    for block in raw_blocks:
        if len(block) <= 900 or not LIST_RE.match(block.splitlines()[0]):
            blocks.append(block)
            continue
        # 아주 긴 리스트는 항목 경계에서 추가로 쪼갠다
        current: list[str] = []
        for line in block.splitlines():
            if LIST_RE.match(line) and sum(len(x) + 1 for x in current) > 600:
                blocks.append("\n".join(current))
                current = []
            current.append(line)
        if current:
            blocks.append("\n".join(current))
    return blocks


def _split_oversized(block: str, max_chars: int) -> list[str]:
    """max_chars를 넘는 단일 블록을 줄 → 문장 경계 순으로 나눈다."""
    units: list[str] = []
    for line in block.splitlines():
        if len(line) <= max_chars:
            units.append(line)
            continue
        units.extend(s for s in SENTENCE_RE.split(line) if s.strip())

    pieces: list[str] = []
    current: list[str] = []
    size = 0
    for unit in units:
        if current and size + len(unit) + 1 > max_chars:
            pieces.append(" ".join(current).strip())
            current, size = [], 0
        if len(unit) > max_chars:
            # 문장 하나가 상한을 넘으면(구두점 없는 텍스트) 마지막 수단으로 하드 컷
            if current:
                pieces.append(" ".join(current).strip())
                current, size = [], 0
            for i in range(0, len(unit), max_chars):
                pieces.append(unit[i : i + max_chars])
            continue
        current.append(unit)
        size += len(unit) + 1
    if current:
        pieces.append(" ".join(current).strip())
    return [p for p in pieces if p.strip()]


def _overlap_tail(text: str, limit: int) -> str:
    """직전 청크의 꼬리를 문장 경계에서 잘라 오버랩으로 쓴다."""
    if limit <= 0 or len(text) <= limit:
        return text.strip()
    tail = text[-limit:]
    for marker in ("\n", ". ", "다. ", "요. ", "! ", "? "):
        pos = tail.find(marker)
        if 0 <= pos < limit // 2:
            return tail[pos + len(marker):].strip()
    return tail.strip()


def _pack(blocks: list[str], max_chars: int, overlap_chars: int) -> list[str]:
    """블록들을 max_chars 안에서 탐욕적으로 묶고, 이어지는 조각에 오버랩을 준다."""
    packed: list[str] = []
    current: list[str] = []
    size = 0

    def close() -> None:
        nonlocal current, size
        if current:
            packed.append("\n\n".join(current))
            current.clear()
            size = 0

    for block in blocks:
        block_len = len(block)
        if size and size + block_len + 2 > max_chars:
            close()
        if block_len > max_chars:
            close()
            packed.extend(_split_oversized(block, max_chars))
            continue
        current.append(block)
        size += block_len + 2
    close()

    if overlap_chars <= 0 or len(packed) < 2:
        return packed

    with_overlap = [packed[0]]
    for i in range(1, len(packed)):
        tail = _overlap_tail(packed[i - 1], overlap_chars)
        with_overlap.append(f"{tail}\n\n{packed[i]}" if tail else packed[i])
    return with_overlap


def chunk_markdown(
    markdown: str,
    doc_title: str,
    *,
    max_chars: int,
    min_chars: int,
    overlap_chars: int,
) -> list[Chunk]:
    """마크다운 문서를 구조 인식 청크 리스트로 만든다."""
    sections = parse_sections(markdown, doc_title)

    # 너무 짧은 섹션은 같은 부모를 공유하는 다음 섹션에 흡수시킨다
    merged: list[Section] = []
    for section in sections:
        if (
            merged
            and len(section.body) < min_chars
            and merged[-1].path[:-1] == section.path[:-1]
            and len(merged[-1].body) + len(section.body) <= max_chars
        ):
            prev = merged[-1]
            prev.body = f"{prev.body}\n\n### {section.heading}\n{section.body}"
            continue
        merged.append(Section(path=list(section.path), body=section.body))

    chunks: list[Chunk] = []
    for section in merged:
        pieces = (
            [section.body]
            if len(section.body) <= max_chars
            else _pack(_split_blocks(section.body), max_chars, overlap_chars)
        )
        for part, piece in enumerate(pieces):
            header = " > ".join(section.path)
            suffix = f" (part {part + 1}/{len(pieces)})" if len(pieces) > 1 else ""
            chunks.append(
                Chunk(
                    index=len(chunks),
                    heading_path=list(section.path),
                    text=piece,
                    embed_text=f"[{header}{suffix}]\n{piece}",
                    char_len=len(piece),
                    part=part,
                    parts=len(pieces),
                )
            )
    return chunks
