# -*- coding: utf-8 -*-
"""모델 로딩 + 배치 추론. P(positive)를 반환한다.

중요: 허브 모델의 id2label을 그대로 믿지 않는다.
      라벨이 없거나(None) 0/1로만 되어 있는 모델이 많으므로,
      명백한 긍/부정 프로브 문장으로 극성 방향을 실측 검증한 뒤 사용한다.
"""
import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 극성 방향 검증용 프로브 (사전적으로 논란의 여지가 없는 문장만)
POLARITY_PROBES_POS = [
    "정말 재미있고 훌륭한 영화입니다. 강력 추천합니다.",
    "최고의 작품이었어요. 너무 좋았습니다.",
    "완벽한 영화. 인생영화로 남을 것 같아요.",
]
POLARITY_PROBES_NEG = [
    "정말 지루하고 최악인 영화입니다. 절대 보지 마세요.",
    "돈이 아까운 쓰레기 영화였어요. 시간 낭비.",
    "재미없고 형편없는 작품. 최악입니다.",
]

MODELS = {
    "mbert-nsmc": "sangrimlee/bert-base-multilingual-cased-nsmc",
    "koelectra-gen": "Copycats/koelectra-base-v3-generalized-sentiment-analysis",
    "kcelectra-ko": "matthewburke/korean_sentiment",
}


class SentimentModel:
    def __init__(self, name, hub_id, max_len=128, batch_size=64):
        self.name = name
        self.hub_id = hub_id
        self.max_len = max_len
        self.batch_size = batch_size
        self.tok = AutoTokenizer.from_pretrained(hub_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(hub_id).to(DEVICE).eval()
        self.n_labels = self.model.config.num_labels
        self.declared_id2label = dict(self.model.config.id2label)
        self.pos_index = None
        self.polarity_note = ""

    # ---- 저수준: 모든 클래스에 대한 확률 ----
    @torch.no_grad()
    def _probs(self, texts):
        out = []
        for i in range(0, len(texts), self.batch_size):
            chunk = list(texts[i:i + self.batch_size])
            enc = self.tok(chunk, padding=True, truncation=True,
                           max_length=self.max_len, return_tensors="pt").to(DEVICE)
            logits = self.model(**enc).logits.float()
            out.append(torch.softmax(logits, dim=-1).cpu().numpy())
        return np.concatenate(out, axis=0)

    # ---- 극성 방향 실측 ----
    def calibrate_polarity(self):
        p_pos = self._probs(POLARITY_PROBES_POS).mean(axis=0)
        p_neg = self._probs(POLARITY_PROBES_NEG).mean(axis=0)
        gap = p_pos - p_neg                       # 긍정 프로브에서 더 커지는 클래스
        self.pos_index = int(np.argmax(gap))
        neg_index = int(np.argmin(gap))
        self.polarity_note = (
            f"pos_index={self.pos_index} (declared={self.declared_id2label.get(self.pos_index)}), "
            f"neg_index={neg_index}, gap={gap[self.pos_index]:.3f}, "
            f"probe_sep={p_pos[self.pos_index] - p_pos[neg_index]:+.3f}"
        )
        # 방향 분리가 약하면 경고
        self.polarity_ok = bool(gap[self.pos_index] > 0.5 and self.n_labels == 2)
        return self.polarity_note

    # ---- 공개 API: P(positive) ----  (앙상블이 같은 문장을 재요청하므로 캐시)
    def predict_proba(self, texts):
        if self.pos_index is None:
            self.calibrate_polarity()
        if not hasattr(self, "_cache"):
            self._cache = {}
        texts = list(texts)
        todo = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if todo:
            p = self._probs(todo)[:, self.pos_index]
            self._cache.update(zip(todo, p))
        return np.array([self._cache[t] for t in texts])


class LexiconBaseline:
    """감성 사전 기반 최소 베이스라인. 트랜스포머가 실제로 뭘 더 하는지 보기 위한 바닥선."""
    name = "lexicon-baseline"
    hub_id = "(rule-based)"
    POS = ["재미", "재밌", "좋", "최고", "훌륭", "완벽", "감동", "추천", "명작", "띵작",
           "만족", "인생영화", "몰입", "꿀잼", "존잼", "강추", "여운", "탄탄", "압권", "낫"]
    NEG = ["지루", "노잼", "최악", "실망", "아까", "별로", "재미없", "발연기", "쓰레기",
           "졸작", "비추", "낭비", "형편없", "어색", "뻔한", "망", "하품", "짜증", "혹평"]

    def __init__(self):
        self.pos_index = 1
        self.polarity_note = "(rule-based, fixed)"
        self.polarity_ok = True
        self.declared_id2label = {0: "negative", 1: "positive"}
        self.n_labels = 2

    def calibrate_polarity(self):
        return self.polarity_note

    def predict_proba(self, texts):
        out = []
        for t in texts:
            p = sum(w in t for w in self.POS)
            n = sum(w in t for w in self.NEG)
            score = (p - n) / max(1.0, (p + n))
            out.append(1 / (1 + np.exp(-2.5 * score)))
        return np.array(out)


class EnsembleMean:
    """트랜스포머 3종의 P(pos) 평균. 단일 최고 모델을 실제로 이기는지 확인용."""
    hub_id = "(mean of transformer P(pos))"

    def __init__(self, members):
        self.members = members
        self.name = "ensemble-mean"
        self.pos_index = 1
        self.polarity_note = f"(mean of {[m.name for m in members]})"
        self.polarity_ok = True
        self.declared_id2label = {0: "negative", 1: "positive"}
        self.n_labels = 2
        self._cache = {}

    def calibrate_polarity(self):
        return self.polarity_note

    def predict_proba(self, texts):
        texts = list(texts)
        return np.mean([m.predict_proba(texts) for m in self.members], axis=0)


def load_all(names=None, include_lexicon=True, include_tfidf=True, include_ensemble=True):
    names = names or list(MODELS)
    models = []
    for n in names:
        m = SentimentModel(n, MODELS[n])
        m.calibrate_polarity()
        models.append(m)
    transformers_only = list(models)
    if include_ensemble and len(transformers_only) > 1:
        models.append(EnsembleMean(transformers_only))
    if include_tfidf:
        from tfidf_baseline import TfidfBaseline
        models.append(TfidfBaseline().fit())
    if include_lexicon:
        models.append(LexiconBaseline())
    return models
