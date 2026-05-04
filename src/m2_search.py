"""Module 2: Hybrid Search - BM25 (Vietnamese) + Dense + RRF."""

import math
import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    BM25_TOP_K,
    COLLECTION_NAME,
    DENSE_TOP_K,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    HYBRID_TOP_K,
    QDRANT_HOST,
    QDRANT_PORT,
)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


class _FallbackEncoder:
    """Small deterministic encoder used when sentence-transformers is unavailable."""

    def __init__(self, dim: int):
        self.dim = dim

    def _encode_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for token in segment_vietnamese(text).split():
            vector[hash(token) % self.dim] += 1.0
        return vector

    def encode(self, texts, show_progress_bar: bool = False):
        if isinstance(texts, str):
            return self._encode_one(texts)
        return [self._encode_one(text) for text in texts]


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words."""
    text = text.strip()
    if not text:
        return ""

    try:
        from underthesea import word_tokenize

        return word_tokenize(text, format="text")
    except Exception:
        return text


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        self.documents = chunks or []
        self.corpus_tokens = [
            segment_vietnamese(chunk.get("text", "")).split()
            for chunk in self.documents
        ]

        if not self.corpus_tokens:
            self.bm25 = None
            return

        try:
            from rank_bm25 import BM25Okapi

            self.bm25 = BM25Okapi(self.corpus_tokens)
        except Exception:
            self.bm25 = None

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        if not self.documents:
            return []

        tokenized_query = segment_vietnamese(query).split()
        if not tokenized_query:
            return []

        if self.bm25 is not None:
            scores = list(self.bm25.get_scores(tokenized_query))
        else:
            query_terms = set(tokenized_query)
            scores = [
                float(sum(1 for token in doc_tokens if token in query_terms))
                for doc_tokens in self.corpus_tokens
            ]

        top_indices = sorted(
            range(len(scores)),
            key=lambda idx: scores[idx],
            reverse=True,
        )[:top_k]

        return [
            SearchResult(
                text=self.documents[idx].get("text", ""),
                score=float(scores[idx]),
                metadata=self.documents[idx].get("metadata", {}),
                method="bm25",
            )
            for idx in top_indices
        ]


class DenseSearch:
    def __init__(self):
        self.client = None
        self._encoder = None
        self.backend = "memory"
        self._memory_items = []

    def _get_client(self):
        if self.client is not None:
            return self.client

        try:
            from qdrant_client import QdrantClient

            client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
            client.get_collections()
            self.client = client
            self.backend = "qdrant"
            return self.client
        except Exception:
            self.client = None
            self.backend = "memory"
            return None

    def _get_encoder(self):
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._encoder = SentenceTransformer(EMBEDDING_MODEL)
            except Exception:
                self._encoder = _FallbackEncoder(EMBEDDING_DIM)
        return self._encoder

    @staticmethod
    def _as_vector_list(vectors) -> list[list[float]]:
        if hasattr(vectors, "tolist"):
            vectors = vectors.tolist()
        return [list(map(float, vector)) for vector in vectors]

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _store_in_memory(self, chunks: list[dict], vectors: list[list[float]]) -> None:
        self._memory_items = [
            {
                "text": chunk.get("text", ""),
                "metadata": chunk.get("metadata", {}),
                "vector": vector,
            }
            for chunk, vector in zip(chunks, vectors)
        ]
        self.backend = "memory"

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant."""
        if not chunks:
            self._memory_items = []
            return

        texts = [chunk.get("text", "") for chunk in chunks]
        vectors = self._as_vector_list(
            self._get_encoder().encode(texts, show_progress_bar=False)
        )
        self._store_in_memory(chunks, vectors)

        client = self._get_client()
        if client is None:
            return

        try:
            from qdrant_client.models import Distance, PointStruct, VectorParams

            if hasattr(client, "collection_exists") and client.collection_exists(collection):
                client.delete_collection(collection)

            if hasattr(client, "recreate_collection"):
                client.recreate_collection(
                    collection_name=collection,
                    vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
                )
            else:
                client.create_collection(
                    collection_name=collection,
                    vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
                )

            points = [
                PointStruct(
                    id=idx,
                    vector=vector,
                    payload={"text": chunk.get("text", ""), **chunk.get("metadata", {})},
                )
                for idx, (chunk, vector) in enumerate(zip(chunks, vectors))
            ]
            client.upsert(collection_name=collection, points=points)
            self.backend = "qdrant"
        except Exception:
            self.backend = "memory"

    def search(
        self,
        query: str,
        top_k: int = DENSE_TOP_K,
        collection: str = COLLECTION_NAME,
    ) -> list[SearchResult]:
        """Search using dense vectors."""
        if not query.strip():
            return []

        query_vector = self._as_vector_list([self._get_encoder().encode(query)])[0]

        if self.backend == "qdrant" and self.client is not None:
            try:
                hits = self.client.search(
                    collection_name=collection,
                    query_vector=query_vector,
                    limit=top_k,
                )
                return [
                    SearchResult(
                        text=hit.payload.get("text", ""),
                        score=float(hit.score),
                        metadata={k: v for k, v in hit.payload.items() if k != "text"},
                        method="dense",
                    )
                    for hit in hits
                ]
            except Exception:
                self.backend = "memory"

        if not self._memory_items:
            return []

        scored = sorted(
            (
                (self._cosine_similarity(query_vector, item["vector"]), item)
                for item in self._memory_items
            ),
            key=lambda pair: pair[0],
            reverse=True,
        )[:top_k]

        return [
            SearchResult(
                text=item["text"],
                score=float(score),
                metadata=item["metadata"],
                method="dense",
            )
            for score, item in scored
        ]


def reciprocal_rank_fusion(
    results_list: list[list[SearchResult]],
    k: int = 60,
    top_k: int = HYBRID_TOP_K,
) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = sum 1 / (k + rank)."""
    fused = {}

    for result_list in results_list:
        for rank, result in enumerate(result_list):
            entry = fused.setdefault(result.text, {"score": 0.0, "result": result})
            entry["score"] += 1.0 / (k + rank + 1)

    merged = sorted(fused.values(), key=lambda entry: entry["score"], reverse=True)[:top_k]
    return [
        SearchResult(
            text=entry["result"].text,
            score=float(entry["score"]),
            metadata=entry["result"].metadata,
            method="hybrid",
        )
        for entry in merged
    ]


class HybridSearch:
    """Combines BM25 + Dense + RRF."""

    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print("Original:  Nhan vien duoc nghi phep nam")
    print(f"Segmented: {segment_vietnamese('Nhan vien duoc nghi phep nam')}")
