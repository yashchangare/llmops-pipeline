from zenml import step
import google.generativeai as genai
from dotenv import load_dotenv
from typing import List
import mlflow
import os
import logging

import opik

load_dotenv()
logger = logging.getLogger(__name__)

import requests

@step
def generate_answer(query: str, context_chunks: List[dict]) -> dict:
    context = "\n\n---\n\n".join([c["text"] for c in context_chunks])
    sources = list(set([c["source"] for c in context_chunks]))

    prompt = f"""Answer using ONLY the context below.

Context:
{context}

Question: {query}

Answer:"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "qwen2.5:0.5b",
            "prompt": prompt,
            "stream": False,
        }
    )

    answer = response.json()["response"]
    logger.info(f"Answer generated for: '{query[:50]}'")

    # Manual Opik trace
    opik_client = opik.Opik()
    trace = opik_client.trace(
        name="rag_generate",
        input={"query": query, "context_chunks_count": len(context_chunks)},
        output={"answer": answer, "sources": sources},
        metadata={"model": "qwen2.5:0.5b"},
    )
    trace.end()

    with mlflow.start_run(run_name="generate", nested=True):
        mlflow.log_param("model", "qwen2.5:0.5b")
        mlflow.log_metric("context_chunks_used", len(context_chunks))
        mlflow.log_param("query", query)
        mlflow.log_param("sources_used", str(sources))
        mlflow.log_text(
            "\n\n---\n\n".join([f"Source: {c['source']}\n{c['text']}" for c in context_chunks]),
            "retrieved_chunks.txt"
        )
        mlflow.log_text(answer, "answer.txt")

    return {
        "query": query,
        "answer": answer,
        "sources": sources,
        "model_used": "qwen2.5:0.5b",
        "context_chunks": context_chunks,
    }