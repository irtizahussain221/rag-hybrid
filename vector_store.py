from pymilvus import MilvusClient
from embedder import VECTOR_DIMENSIONS
from dataclasses import dataclass


@dataclass
class Vector:
    id: str
    vector: list[float]


class VectorStore:
    def __init__(self, db_name: str, collection_name: str):
        self.client = MilvusClient(db_name)
        if not self.client.has_collection(collection_name=collection_name):
            self.client.create_collection(
                collection_name=collection_name, dimension=VECTOR_DIMENSIONS
            )

    def upsert(self, vectors: Vector):
        self.client.upsert(
            collection_name=self.collection_name,
            data=[{"id": vector.id, "vector": vector.vector} for vector in vectors],
        )
