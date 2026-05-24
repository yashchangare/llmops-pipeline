from zenml import step
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from llama_index.embeddings.gemini import GeminiEmbedding
from typing import List
import mlflow
import os
import logging

logger = logging.getLogger(__name__)

COLLECTION = os.getenv("QDRANT_COLLECTION", "llmops_docs")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
EMBED_DIM = 768


@step
def embed_and_store(chunks: List[dict]) -> int:
    with mlflow.start_run(run_name="embed", nested=True):

        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

        existing = [c.name for c in client.get_collections().collections]
        if COLLECTION not in existing:
            client.create_collection(
                collection_name=COLLECTION,
                vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
            )
            logger.info(f"Created Qdrant collection: {COLLECTION}")

        embed_model = GeminiEmbedding(
            model_name="models/embedding-001",
            api_key=os.getenv("GOOGLE_API_KEY"),
        )

        points = []
        texts = [c["text"] for c in chunks]

        logger.info(f"Embedding {len(texts)} chunks...")
        embeddings = embed_model.get_text_embedding_batch(texts, show_progress=True)

        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            points.append(
                PointStruct(
                    id=i,
                    vector=embedding,
                    payload={
                        "text": chunk["text"],
                        "source": chunk["metadata"].get("file_name", "unknown"),
                        "chunk_id": chunk["id"],
                    },
                )
            )

        client.upsert(collection_name=COLLECTION, points=points)

        mlflow.log_metric("vectors_stored", len(points))
        mlflow.log_param("embed_model", "gemini-embedding-001")
        mlflow.log_param("vector_db", "qdrant-cloud")
        logger.info(f"Stored {len(points)} vectors in Qdrant Cloud")

    return len(points)