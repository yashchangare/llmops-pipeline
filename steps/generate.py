from zenml import step
from openai import OpenAI  # LiteLLM is OpenAI-compatible, same client
import opik
from opik.integrations.openai import track_openai
from typing import List
import os
import logging

logger = logging.getLogger(__name__)

# Point OpenAI client at LiteLLM gateway instead of OpenAI directly
# LiteLLM handles routing to Ollama or OpenAI based on litellm_config.yaml
client = OpenAI(
    base_url=os.getenv("LITELLM_BASE_URL", "http://localhost:4000"),
    api_key="sk-llmops-local-key",  # matches master_key in litellm_config.yaml
)

# Wrap client with Opik — this is all you need for full tracing
# Every call now appears in Opik dashboard automatically
client = track_openai(client)

# Initialize Opik project
opik.configure(
    api_key=os.getenv("OPIK_API_KEY"),
    project_name=os.getenv("OPIK_PROJECT_NAME", "llmops-pipeline"),
)


@opik.track(name="rag_generate")  # decorator creates a trace in Opik
@step
def generate_answer(query: str, context_chunks: List[dict]) -> dict:
    """
    Generate answer using retrieved context, routed through LiteLLM.
    Opik traces every call: prompt, response, token count, latency, model used.
    """
    # Build context string from retrieved chunks
    context = "\n\n---\n\n".join([c["text"] for c in context_chunks])
    sources = list(set([c["source"] for c in context_chunks]))

    system_prompt = """You are a helpful assistant. Answer the question using ONLY 
the provided context. If the answer is not in the context, say so clearly.
Do not make up information."""

    user_prompt = f"""Context:
{context}

Question: {query}

Answer:"""

    logger.info(f"Generating answer for: '{query[:50]}...'")

    response = client.chat.completions.create(
        model="primary",  # LiteLLM routes this to Ollama, falls back to OpenAI
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,  # low temp for factual RAG
        max_tokens=512,
    )

    answer = response.choices[0].message.content
    model_used = response.model

    logger.info(f"Answer generated using model: {model_used}")

    return {
        "query": query,
        "answer": answer,
        "sources": sources,
        "model_used": model_used,
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
    }
