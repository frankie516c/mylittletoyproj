# -*- coding: utf-8 -*-
"""NSMC train으로 학습하는 문자 n-gram TF-IDF + 로지스틱회귀 베이스라인.

목적: 트랜스포머가 '표면 어휘 통계' 대비 실제로 무엇을 더 하는지 분리해 보기 위함.
      NSMC 테스트 점수는 비슷한데 행동 테스트(부정/반어/대조)에서만 크게 벌어진다면
      그 격차가 곧 트랜스포머가 실제로 얻은 능력이다.
"""
import os
import pickle

import numpy as np

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "tfidf.pkl")


class TfidfBaseline:
    name = "tfidf-lr"
    hub_id = "(char 2-4gram TF-IDF + LogisticRegression, NSMC train 150k)"

    def __init__(self):
        self.pos_index = 1
        self.polarity_note = "(supervised, fixed)"
        self.polarity_ok = True
        self.declared_id2label = {0: "negative", 1: "positive"}
        self.n_labels = 2
        self.pipe = None

    def calibrate_polarity(self):
        return self.polarity_note

    def fit(self, train_path="data/ratings_train.txt"):
        if os.path.exists(CACHE):
            with open(CACHE, "rb") as f:
                self.pipe = pickle.load(f)
            return self
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline

        X, y = [], []
        with open(train_path, encoding="utf-8") as f:
            next(f)
            for line in f:
                p = line.rstrip("\n").split("\t")
                if len(p) == 3 and p[1].strip():
                    X.append(p[1]); y.append(int(p[2]))
        self.pipe = make_pipeline(
            TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=3, max_features=300000,
                            sublinear_tf=True),
            LogisticRegression(C=4.0, max_iter=2000, solver="liblinear"),
        )
        self.pipe.fit(X, y)
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        with open(CACHE, "wb") as f:
            pickle.dump(self.pipe, f)
        return self

    def predict_proba(self, texts):
        if self.pipe is None:
            self.fit()
        return self.pipe.predict_proba(list(texts))[:, 1]
