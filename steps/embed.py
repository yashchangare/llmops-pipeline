from zenml import step
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import google.generativeai as genai
from typing import List
import mlflow
import os
import logging

logger = logging.getLogger(__name__)

COLLECTION = os.getenv("QDRANT_COLLECTION", "llmops_docs")
QDRANT_URL = os.getenv("QDRANT_URL", "").strip().rstrip("/")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "").strip()
EMBED_DIM = 3072


def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Get embeddings from Google Gemini API."""
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    embeddings = []
    for text in texts:
        result = genai.embed_content(
            model="models/gemini-embedding-001",
            content=text,
            task_type="retrieval_document",
        )
        embeddings.append(result["embedding"])
    return embeddings


@step
def embed_and_store(chunks: List[dict]) -> int:
    """Embed chunks with Gemini and store in Qdrant Cloud."""
    with mlflow.start_run(run_name="embed", nested=True):

        client = QdrantClient(path="data/qdrant_storage")

        existing = [c.name for c in client.get_collections().collections]
        if COLLECTION not in existing:
            client.create_collection(
                collection_name=COLLECTION,
                vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
            )
            logger.info(f"Created collection: {COLLECTION}")

        logger.info(f"Embedding {len(chunks)} chunks with Gemini...")
        texts = [c["text"] for c in chunks]
        embeddings = get_embeddings(texts)

        points = [
            PointStruct(
                id=i,
                vector=embeddings[i],
                payload={
                    "text": chunks[i]["text"],
                    "source": chunks[i]["source"],
                    "chunk_id": chunks[i]["id"],
                },
            )
            for i in range(len(chunks))
        ]

        client.upsert(collection_name=COLLECTION, points=points)

        mlflow.log_metric("vectors_stored", len(points))
        mlflow.log_param("embed_model", "gemini-embedding-001")
        mlflow.log_param("vector_db", "qdrant-cloud")
        logger.info(f"Stored {len(points)} vectors in Qdrant Cloud")

    return len(points)