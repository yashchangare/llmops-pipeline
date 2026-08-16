from zenml import step
from qdrant_client import QdrantClient
import google.generativeai as genai
from typing import List
import os
import logging

logger = logging.getLogger(__name__)

COLLECTION = os.getenv("QDRANT_COLLECTION", "llmops_docs")
QDRANT_URL = os.getenv("QDRANT_URL", "").strip().rstrip("/")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "").strip()


def get_query_embedding(query: str) -> List[float]:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    result = genai.embed_content(
        model="models/gemini-embedding-001",
        content=query,
        task_type="retrieval_query",
    )
    return result["embedding"]


def semantic_search(query: str, top_k: int = 5) -> List[dict]:
    client = QdrantClient(path="data/qdrant_storage")
    query_vector = get_query_embedding(query)
    results = client.query_points(
        collection_name=COLLECTION,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    ).points
    return [
        {"text": r.payload["text"], "score": r.score, "source": r.payload["source"]}
        for r in results
    ]


def keyword_search(query: str, top_k: int = 5) -> List[dict]:
    client = QdrantClient(path="data/qdrant_storage")
    results, _ = client.scroll(
        collection_name=COLLECTION,
        limit=200,
        with_payload=True,
    )
    keywords = query.lower().split()
    scored = []
    for r in results:
        text_lower = r.payload["text"].lower()
        score = sum(1 for kw in keywords if kw in text_lower) / len(keywords)
        if score > 0:
            scored.append({
                "text": r.payload["text"],
                "score": score,
                "source": r.payload["source"],
            })
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def hybrid_retrieve(query: str, top_k: int = 5) -> List[dict]:
    """Hybrid search: semantic + keyword, merged and re-ranked."""
    semantic = semantic_search(query, top_k=top_k)
    keyword = keyword_search(query, top_k=top_k)

    seen = set()
    merged = []
    for chunk in semantic + keyword:
        key = chunk["text"][:100]
        if key not in seen:
            seen.add(key)
            merged.append(chunk)

    merged.sort(key=lambda x: x["score"], reverse=True)
    results = merged[:top_k]
    logger.info(f"Retrieved {len(results)} chunks for: '{query[:50]}'")
    return results