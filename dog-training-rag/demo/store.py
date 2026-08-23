"""로컬 벡터 DB (Qdrant 임베디드 모드).

서버를 따로 띄우지 않고 `data/demo/qdrant/` 디렉터리에 그대로 저장한다.
주의: 임베디드 모드는 디렉터리를 잠그므로 색인 스크립트와 웹서버를 동시에 켤 수 없다.
"""

from __future__ import annotations

from pathlib import Path

from qdrant_client import QdrantClient, models


def open_client(path: Path) -> QdrantClient:
    path.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(path))


def recreate_collection(client: QdrantClient, name: str, dim: int) -> None:
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(
        collection_name=name,
        vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
    )
    # 페이로드 인덱스는 임베디드 모드에서 효과가 없다. 서버 모드로 올릴 때를 위해
    # 필드 목록만 남겨 둔다: doc_id / topic / publisher


def upsert(client: QdrantClient, name: str, vectors: list[list[float]], payloads: list[dict]) -> None:
    client.upsert(
        collection_name=name,
        points=[
            models.PointStruct(id=i, vector=vec, payload=payload)
            for i, (vec, payload) in enumerate(zip(vectors, payloads))
        ],
    )


def search(client: QdrantClient, name: str, vector: list[float], limit: int) -> list:
    return client.query_points(
        collection_name=name,
        query=vector,
        limit=limit,
        with_payload=True,
    ).points
