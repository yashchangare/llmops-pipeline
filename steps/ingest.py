from zenml import step
from typing import List
import mlflow
import os
import logging

logger = logging.getLogger(__name__)


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks by word count."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


@step
def ingest_documents(data_dir: str = "data/raw") -> List[dict]:
    """Load all .txt and .md files, chunk them, return as list of dicts."""
    with mlflow.start_run(run_name="ingest", nested=True):
        chunks = []
        files = [f for f in os.listdir(data_dir) if f.endswith((".txt", ".md"))]
        logger.info(f"Found {len(files)} files in {data_dir}")

        for filename in files:
            filepath = os.path.join(data_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()

            file_chunks = chunk_text(text)
            for i, chunk in enumerate(file_chunks):
                chunks.append({
                    "id": f"{filename}_{i}",
                    "text": chunk,
                    "source": filename,
                })

        mlflow.log_metric("docs_loaded", len(files))
        mlflow.log_metric("chunks_created", len(chunks))
        mlflow.log_param("chunk_size", 512)
        mlflow.log_param("chunk_overlap", 50)
        logger.info(f"Created {len(chunks)} chunks from {len(files)} files")

    return chunks