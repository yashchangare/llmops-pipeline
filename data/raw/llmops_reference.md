# LLMOps & MLOps Reference Guide

## What is MLOps?

MLOps (Machine Learning Operations) is the practice of automating and managing the full lifecycle of machine learning models in production. It bridges data science and DevOps, applying software engineering principles to ML workflows.

Key components of MLOps include:
- **Data versioning**: Tracking datasets like code using tools like DVC
- **Experiment tracking**: Logging metrics, parameters, and artifacts for every training run using MLflow
- **Pipeline orchestration**: Defining reproducible ML workflows as DAGs using tools like ZenML or Kubeflow
- **Model registry**: Versioning and promoting models through staging to production
- **CI/CD for ML**: Automated testing, evaluation, and deployment of models
- **Monitoring**: Detecting model drift and data drift in production

## What is LLMOps?

LLMOps extends MLOps specifically for Large Language Models. Because LLMs behave differently from traditional ML models, new operational patterns are needed.

Key differences from traditional MLOps:
- **Prompt engineering as code**: Prompts are versioned, tested, and deployed like code
- **RAG systems**: Retrieval-Augmented Generation adds a retrieval layer that must be operated separately
- **Non-deterministic outputs**: LLMs produce different outputs for the same input, requiring statistical evaluation
- **LLM-as-judge evaluation**: Using powerful LLMs to score outputs from other LLMs at scale
- **Model routing**: Sending requests to different models based on cost, latency, or task type

## Retrieval-Augmented Generation (RAG)

RAG is the dominant architecture for production LLM applications. It works by:

1. **Indexing**: Documents are chunked, embedded into vectors, and stored in a vector database
2. **Retrieval**: User queries are embedded and used to find semantically similar document chunks
3. **Generation**: Retrieved chunks are passed as context to an LLM to generate a grounded response

### Hybrid Search

Pure semantic (vector) search can miss exact keyword matches. Hybrid search combines:
- **Dense retrieval**: Vector similarity search for semantic meaning
- **Sparse retrieval**: BM25 or keyword-based search for exact term matching
- **Re-ranking**: Combining and re-scoring results from both methods

This improves recall for queries with specific names, codes, or technical terms.

## Evaluation in LLMOps

Evaluation is the most critical and time-consuming part of production LLMOps. Key metrics:

### RAGAS Metrics
- **Faithfulness**: Does the answer stick to the retrieved context? (measures hallucination)
- **Answer Relevancy**: Is the answer relevant to the user's question?
- **Context Recall**: Did the retrieval step find the right documents?

### LLM-as-Judge
A separate, more powerful LLM evaluates the outputs of the RAG system. This scales better than human evaluation and can be automated in CI/CD pipelines.

### Golden Datasets
A curated set of question-answer pairs with known correct answers. Used as regression tests to ensure quality doesn't degrade between deployments.

## ZenML Pipelines

ZenML is an MLOps framework that lets you define pipelines as Python functions with decorators:

```python
from zenml import step, pipeline

@step
def ingest_data() -> list:
    # load your data
    return data

@step  
def train_model(data: list) -> dict:
    # train your model
    return model

@pipeline
def ml_pipeline():
    data = ingest_data()
    model = train_model(data)
```

ZenML automatically:
- Versions every artifact (data, models, metrics)
- Tracks lineage between steps
- Runs steps on any infrastructure (local, Kubernetes, SageMaker, Vertex AI)

## MLflow Experiment Tracking

MLflow tracks experiments with a simple API:

```python
import mlflow

with mlflow.start_run():
    mlflow.log_param("chunk_size", 512)
    mlflow.log_metric("faithfulness", 0.87)
    mlflow.log_artifact("model.pkl")
```

The MLflow UI at localhost:5000 shows all runs, metrics, and artifacts in a dashboard.

## LiteLLM Model Gateway

LiteLLM provides a unified API across all LLM providers:

```python
from litellm import completion

# Same code works for any model
response = completion(model="ollama/llama3.2", messages=[...])
response = completion(model="gpt-4o-mini", messages=[...])
```

As a gateway/proxy, it handles:
- **Routing**: Send traffic to different models based on rules
- **Fallbacks**: Automatically retry with a different model on failure
- **Cost tracking**: Log spend per request across providers
- **Caching**: Cache repeated queries to reduce latency and cost
