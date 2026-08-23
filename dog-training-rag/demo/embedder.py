"""임베딩 모델 로더.

fastembed(ONNX Runtime)만 쓴다. torch를 끌어오지 않아 설치가 가볍고 CPU에서 충분히 빠르다.

`intfloat/multilingual-e5-small` (118M / 384d) — 작으면서 한국어가 되는 지점.
fastembed 기본 카탈로그에 없어서 커스텀 ONNX 모델로 등록해 쓴다.
int8 양자화본(약 118MB)을 1순위로 시도하고, 실패하면 fp32(약 470MB), 그래도 안 되면
카탈로그에 있는 다국어 소형 모델로 폴백한다.
"""

from __future__ import annotations

from dataclasses import dataclass

from fastembed import TextEmbedding
from fastembed.common.model_description import ModelSource, PoolingType

E5_REPO = "intfloat/multilingual-e5-small"
FALLBACK = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# (등록 이름, 리포 내 onnx 파일, 설명)
E5_VARIANTS = [
    ("intfloat/multilingual-e5-small-q8", "onnx/model_qint8_avx512_vnni.onnx", "int8 양자화 (~118MB)"),
    ("intfloat/multilingual-e5-small", "onnx/model.onnx", "fp32 (~470MB)"),
]


@dataclass
class Embedder:
    name: str
    dim: int
    query_prefix: str
    passage_prefix: str
    _model: TextEmbedding

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        prefixed = [self.passage_prefix + t for t in texts]
        return [v.tolist() for v in self._model.embed(prefixed)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._model.query_embed(self.query_prefix + text))).tolist()


def _register(name: str, model_file: str) -> None:
    known = {m["model"] for m in TextEmbedding.list_supported_models()}
    if name in known:
        return
    TextEmbedding.add_custom_model(
        model=name,
        pooling=PoolingType.MEAN,
        normalization=True,
        sources=ModelSource(hf=E5_REPO),
        dim=384,
        model_file=model_file,
        description="Multilingual E5 small, 384d, 512 tokens, query:/passage: prefixes required",
        license="mit",
    )


def load_embedder(verbose: bool = True) -> Embedder:
    for name, model_file, note in E5_VARIANTS:
        try:
            _register(name, model_file)
            model = TextEmbedding(model_name=name)
            next(iter(model.embed(["passage: 워밍업"])))  # 다운로드/추론 실제 확인
            if verbose:
                print(f"[embedder] {E5_REPO} {note} 로드 완료 (384d)")
            return Embedder(E5_REPO, 384, "query: ", "passage: ", model)
        except Exception as exc:  # noqa: BLE001 - 폴백이 목적
            if verbose:
                print(f"[embedder] {note} 실패: {type(exc).__name__}: {exc}")

    if verbose:
        print(f"[embedder] {FALLBACK} 로 폴백합니다")
    return Embedder(FALLBACK, 384, "", "", TextEmbedding(model_name=FALLBACK))
