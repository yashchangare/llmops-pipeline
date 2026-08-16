from zenml import pipeline
from steps.ingest import ingest_documents
from steps.embed import embed_and_store
from steps.retrieve import hybrid_retrieve
from steps.generate import generate_answer


@pipeline(name="rag_indexing_pipeline")
def indexing_pipeline(data_dir: str = "data/raw"):
    chunks = ingest_documents(data_dir=data_dir)
    embed_and_store(chunks=chunks)


def run_query_pipeline(query: str) -> dict:
    chunks = hybrid_retrieve(query=query)
    result = generate_answer(query=query, context_chunks=chunks)
    return result