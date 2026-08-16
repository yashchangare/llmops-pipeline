from zenml import step
import opik
from opik.evaluation.metrics import Hallucination, AnswerRelevance, ContextPrecision
import mlflow
import json
import os
import logging
from typing import List

logger = logging.getLogger(__name__)

GOLDEN_DATASET_PATH = "evaluation/golden_dataset.json"

def load_golden_dataset() -> List[dict]:
    with open(GOLDEN_DATASET_PATH) as f:
        return json.load(f)


def evaluate_pipeline(results: List[dict]) -> dict:
    """Opik LLM-as-judge evaluation. Scores hallucination, relevance, context precision."""
    with mlflow.start_run(run_name="evaluate", nested=True):

        scores = {"hallucination": [], "answer_relevance": [], "context_precision": []}

        for r in results:
            context = " ".join([c["text"] for c in r.get("context_chunks", [])])

            ##-- Add your OpenAI key, Opik's LLM-as-judge uses gpt-4o by default which needs an OpenAI key. 
            # scores["hallucination"].append(
            #     Hallucination().score(input=r["query"], output=r["answer"], context=context).value
            # )
            # scores["answer_relevance"].append(
            #     AnswerRelevance().score(input=r["query"], output=r["answer"]).value
            # )
            # scores["context_precision"].append(
            #     ContextPrecision().score(input=r["query"], output=r["answer"], context=context).value
            # )

            scores["hallucination"].append(
                Hallucination(model="ollama/qwen2.5:0.5b").score(input=r["query"], output=r["answer"], context=context).value
            )
            scores["answer_relevance"].append(
                AnswerRelevance(model="ollama/qwen2.5:0.5b").score(input=r["query"], output=r["answer"]).value
            )
            scores["context_precision"].append(
                ContextPrecision(model="ollama/qwen2.5:0.5b").score(input=r["query"], output=r["answer"], context=context).value
            )

        final_scores = {k: round(sum(v) / len(v), 3) for k, v in scores.items() if v}

        for metric, score in final_scores.items():
            mlflow.log_metric(f"opik_{metric}", score)

        logger.info(f"Evaluation scores: {final_scores}")

    return final_scores