"""데모에 적재할 훈련 자료 소스.

전부 tier-1(ASPCA / AKC / Humane Society) 문서다. 주제(topic)는 UI의 커버리지 표시와
검색 메타데이터에 쓴다. 여기 없는 주제는 데모가 "답할 수 없다"고 답하는 것이 정상이다.
"""

SOURCES = [
    # ── 배변 훈련 ────────────────────────────────────────────────
    {
        "doc_id": "akc-housetrain-adult",
        "url": "https://www.akc.org/expert-advice/training/how-to-housetrain-an-adult-dog/",
        "title": "How to Potty Train an Older Dog: Housetraining Adult Dogs",
        "title_ko": "성견 배변 훈련",
        "publisher": "AKC",
        "evidence_tier": 1,
        "lang": "en",
        "topic": "house_training",
        "topic_ko": "배변 훈련",
    },
    {
        "doc_id": "akc-potty-pad",
        "url": "https://www.akc.org/expert-advice/training/the-ins-and-outs-of-potty-pad-training/",
        "title": "Housetraining Dogs: Puppy Potty Pad and Paper Training",
        "title_ko": "배변패드·신문지 훈련",
        "publisher": "AKC",
        "evidence_tier": 1,
        "lang": "en",
        "topic": "house_training",
        "topic_ko": "배변 훈련",
    },
    {
        "doc_id": "akc-potty-apartment",
        "url": "https://www.akc.org/expert-advice/training/potty-train-dog-in-apartment/",
        "title": "How to Potty Train a Dog When You Live in an Apartment",
        "title_ko": "아파트에서의 배변 훈련",
        "publisher": "AKC",
        "evidence_tier": 1,
        "lang": "en",
        "topic": "house_training",
        "topic_ko": "배변 훈련",
    },
    {
        "doc_id": "hsus-potty-training",
        "url": "https://www.humanesociety.org/resources/how-potty-train-your-dog-or-puppy",
        "title": "How to Potty Train a Puppy: Essential Housebreaking Tips",
        "title_ko": "배변 훈련 기본",
        "publisher": "Humane Society",
        "evidence_tier": 1,
        "lang": "en",
        "topic": "house_training",
        "topic_ko": "배변 훈련",
    },
    {
        "doc_id": "aspca-house-training",
        "url": "https://www.aspca.org/news/house-training-your-dog-or-puppy",
        "title": "House Training Your Dog or Puppy",
        "title_ko": "배변 훈련 요약",
        "publisher": "ASPCA",
        "evidence_tier": 1,
        "lang": "en",
        "topic": "house_training",
        "topic_ko": "배변 훈련",
    },
    # ── 입질 ────────────────────────────────────────────────────
    {
        "doc_id": "aspca-puppy-mouthing",
        "url": "https://www.aspca.org/pet-care/dog-care/common-dog-behavior-issues/mouthing-nipping-and-biting-puppies",
        "title": "Mouthing, Nipping and Biting in Puppies",
        "title_ko": "어린 강아지의 입질",
        "publisher": "ASPCA",
        "evidence_tier": 1,
        "lang": "en",
        "topic": "mouthing",
        "topic_ko": "입질·물기",
    },
    {
        "doc_id": "aspca-adult-mouthing",
        "url": "https://www.aspca.org/pet-care/dog-care/common-dog-behavior-issues/mouthing-nipping-and-play-biting-adult-dogs",
        "title": "Mouthing, Nipping and Play Biting in Adult Dogs",
        "title_ko": "성견의 입질",
        "publisher": "ASPCA",
        "evidence_tier": 1,
        "lang": "en",
        "topic": "mouthing",
        "topic_ko": "입질·물기",
    },
    # ── 그 밖의 흔한 문제행동 ──────────────────────────────────────
    {
        "doc_id": "aspca-barking",
        "url": "https://www.aspca.org/pet-care/dog-care/common-dog-behavior-issues/barking",
        "title": "Barking",
        "title_ko": "짖음",
        "publisher": "ASPCA",
        "evidence_tier": 1,
        "lang": "en",
        "topic": "barking",
        "topic_ko": "짖음",
    },
    {
        "doc_id": "aspca-separation-anxiety",
        "url": "https://www.aspca.org/pet-care/dog-care/common-dog-behavior-issues/separation-anxiety",
        "title": "Separation Anxiety",
        "title_ko": "분리불안",
        "publisher": "ASPCA",
        "evidence_tier": 1,
        "lang": "en",
        "topic": "separation_anxiety",
        "topic_ko": "분리불안",
    },
    {
        "doc_id": "aspca-destructive-chewing",
        "url": "https://www.aspca.org/pet-care/dog-care/common-dog-behavior-issues/destructive-chewing",
        "title": "Destructive Chewing",
        "title_ko": "물건 파괴·씹기",
        "publisher": "ASPCA",
        "evidence_tier": 1,
        "lang": "en",
        "topic": "chewing",
        "topic_ko": "씹기·물어뜯기",
    },
]

TOPICS_KO = []
for _s in SOURCES:
    if _s["topic_ko"] not in TOPICS_KO:
        TOPICS_KO.append(_s["topic_ko"])
