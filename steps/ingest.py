from zenml import step
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from typing import List
import mlflow
import logging

logger = logging.getLogger(__name__)


@step
def ingest_documents(data_dir: str = "data/raw") -> List[dict]:
    """
    Load documents from data/raw/, chunk them, return as list of dicts.
    ZenML automatically versions this artifact.
    MLflow logs how many docs and chunks were created.
    """
    with mlflow.start_run(run_name="ingest", nested=True):

        # Load all files from data/raw/ (PDFs, txt, md — LlamaIndex handles all)
        logger.info(f"Loading documents from {data_dir}")
        reader = SimpleDirectoryReader(data_dir, recursive=True)
        documents = reader.load_data()
        mlflow.log_metric("docs_loaded", len(documents))
        logger.info(f"Loaded {len(documents)} documents")

        # Chunk into overlapping segments
        # chunk_size=512 tokens, overlap=50 — good default for RAG
        splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
        nodes = splitter.get_nodes_from_documents(documents)
        mlflow.log_metric("chunks_created", len(nodes))
        mlflow.log_param("chunk_size", 512)
        mlflow.log_param("chunk_overlap", 50)
        logger.info(f"Created {len(nodes)} chunks")

        # Convert to plain dicts so ZenML can serialize/version them
        chunks = [
            {
                "id": node.node_id,
                "text": node.text,
                "metadata": node.metadata,
            }
            for node in nodes
        ]

    return chunks
