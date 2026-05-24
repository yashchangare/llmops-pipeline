from zenml import step
from qdrant_client import QdrantClient
from llama_index.embeddings.gemini import GeminiEmbedding
from typing import List
import os
import logging

logger = logging.getLogger(__name__)

COLLECTION = os.getenv("QDRANT_COLLECTION", "llmops_docs")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")


def get_client():
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def get_embed_model():
    return GeminiEmbedding(
        model_name="models/embedding-001",
        api_key=os.getenv("GOOGLE_API_KEY"),
    )


def semantic_search(query: str, top_k: int = 5) -> List[dict]:
    client = get_client()
    embed_model = get_embed_model()
    query_vector = embed_model.get_text_embedding(query)
    results = client.search(
        collection_name=COLLECTION,
        query_vector=query_vector,
        limit=top_k,
        with_payload=True,
    )
    return [
        {"text": r.payload["text"], "score": r.score, "source": r.payload["source"]}
        for r in results
    ]


def keyword_search(query: str, top_k: int = 5) -> List[dict]:
    client = get_client()
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


@step
def hybrid_retrieve(query: str, top_k: int = 5) -> List[dict]:
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