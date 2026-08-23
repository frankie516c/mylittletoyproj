# -*- coding: utf-8 -*-
"""의미 보존 변형(INV용). 감성 어휘는 보호해서 라벨이 바뀌지 않도록 한다."""
import random
import re

SBASE, TCOUNT = 0xAC00, 28
SEND = 0xD7A3

# 이 어휘 안의 글자는 오타 변형에서 제외한다 (건드리면 라벨이 흔들릴 수 있음)
PROTECTED = [
    "재미", "재밌", "재미없", "노잼", "꿀잼", "존잼", "핵노잼", "좋", "최고", "최악",
    "훌륭", "완벽", "감동", "추천", "비추", "명작", "졸작", "띵작", "만족", "실망",
    "지루", "아깝", "아까", "별로", "발연기", "쓰레기", "낭비", "형편없", "짜증",
    "안", "못", "없", "않",  # 부정 표지는 절대 보호
]


def _protected_mask(text):
    mask = [False] * len(text)
    for w in PROTECTED:
        for m in re.finditer(re.escape(w), text):
            for i in range(m.start(), m.end()):
                mask[i] = True
    return mask


def _is_hangul(ch):
    return SBASE <= ord(ch) <= SEND


def despace(text, rng=None):
    return re.sub(r"\s+", "", text)


def punct_strip(text, rng=None):
    return re.sub(r"[.,!?~…]+", "", text).strip()


def emoji_append(text, rng=None):
    return text + " 🎬"


def drop_jongseong(text, rng=None):
    """받침 하나를 떨어뜨린다 (매우 흔한 한국어 오타). 감성어는 건드리지 않음."""
    rng = rng or random
    mask = _protected_mask(text)
    cands = [i for i, ch in enumerate(text)
             if _is_hangul(ch) and (ord(ch) - SBASE) % TCOUNT != 0 and not mask[i]]
    if not cands:
        return text
    i = rng.choice(cands)
    code = ord(text[i])
    return text[:i] + chr(code - (code - SBASE) % TCOUNT) + text[i + 1:]


def swap_adjacent(text, rng=None):
    """인접 음절 자리바꿈 (타이핑 오류). 감성어 내부는 제외."""
    rng = rng or random
    mask = _protected_mask(text)
    cands = [i for i in range(len(text) - 1)
             if _is_hangul(text[i]) and _is_hangul(text[i + 1])
             and not mask[i] and not mask[i + 1]]
    if not cands:
        return text
    i = rng.choice(cands)
    return text[:i] + text[i + 1] + text[i] + text[i + 2:]


def pad_context(text, rng=None):
    return "어제 친구랑 같이 극장에서 봤습니다. " + text + " 참고로 주차는 3시간 무료였습니다."


PERTURBATIONS = {
    "despace": despace,
    "punct_strip": punct_strip,
    "emoji_append": emoji_append,
    "drop_jongseong": drop_jongseong,
    "swap_adjacent": swap_adjacent,
    "pad_context": pad_context,
}
