from zenml import pipeline
from steps.ingest import ingest_documents
from steps.embed import embed_and_store
from steps.retrieve import hybrid_retrieve
from steps.generate import generate_answer
import mlflow
import os


@pipeline(name="rag_indexing_pipeline")
def indexing_pipeline(data_dir: str = "data/raw"):
    """
    Indexing pipeline — run this once (or when data changes).
    Ingests documents, chunks them, embeds and stores in Qdrant.
    ZenML versions every artifact automatically.
    DVC tracks the raw data files separately (see .dvc/ folder).
    """
    chunks = ingest_documents(data_dir=data_dir)
    vectors_stored = embed_and_store(chunks=chunks)
    return vectors_stored


@pipeline(name="rag_query_pipeline")
def query_pipeline(query: str):
    """
    Query pipeline — run this for every user question.
    Retrieves relevant chunks, generates answer via LiteLLM gateway.
    Opik traces every LLM call automatically via the decorator in generate.py.
    """
    chunks = hybrid_retrieve(query=query)
    result = generate_answer(query=query, context_chunks=chunks)
    return result
