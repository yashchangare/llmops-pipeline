from zenml import step
import opik
from opik.evaluation import evaluate
from opik.evaluation.metrics import Hallucination, AnswerRelevance, ContextRecall
from ragas import evaluate as ragas_evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from datasets import Dataset
import mlflow
import json
import os
import logging
from typing import List

logger = logging.getLogger(__name__)

GOLDEN_DATASET_PATH = "evaluation/golden_dataset.json"


def load_golden_dataset() -> List[dict]:
    """Load ground truth Q&A pairs used as regression baseline."""
    with open(GOLDEN_DATASET_PATH) as f:
        return json.load(f)


@step
def evaluate_pipeline(results: List[dict]) -> dict:
    """
    Two-layer evaluation:
    1. Opik — LLM-as-judge scores each response (hallucination, relevance)
    2. RAGAS — dataset-level metrics against golden dataset
    Both logged to MLflow so you can track quality across runs.
    """
    with mlflow.start_run(run_name="evaluate", nested=True):

        # ── LAYER 1: Opik LLM-as-judge ──────────────────────────────────────
        # Opik sends each response to an LLM judge and scores it 0-1
        logger.info("Running Opik LLM-as-judge evaluation...")

        opik_dataset = [
            opik.DatasetItem(
                input={"query": r["query"]},
                expected_output=r.get("expected_answer", ""),
                metadata={"context": r.get("context", ""), "sources": r.get("sources", [])},
            )
            for r in results
        ]

        opik_results = evaluate(
            dataset=opik_dataset,
            task=lambda x: {"output": x["query"]},  # plug in your generate step here
            scoring_metrics=[
                Hallucination(),     # did the model make things up?
                AnswerRelevance(),   # is the answer relevant to the question?
                ContextRecall(),     # did it use the context correctly?
            ],
            project_name=os.getenv("OPIK_PROJECT_NAME", "llmops-pipeline"),
        )

        avg_hallucination = opik_results.get("hallucination_score", 0)
        avg_relevance = opik_results.get("answer_relevance_score", 0)

        mlflow.log_metric("opik_hallucination_score", avg_hallucination)
        mlflow.log_metric("opik_answer_relevance", avg_relevance)
        logger.info(f"Opik scores — Hallucination: {avg_hallucination:.3f}, Relevance: {avg_relevance:.3f}")

        # ── LAYER 2: RAGAS dataset-level metrics ────────────────────────────
        # RAGAS evaluates at the dataset level against your golden dataset
        logger.info("Running RAGAS evaluation against golden dataset...")

        golden = load_golden_dataset()

        ragas_data = Dataset.from_list([
            {
                "question": g["question"],
                "answer": next(
                    (r["answer"] for r in results if r["query"] == g["question"]),
                    ""
                ),
                "contexts": [g["context"]],
                "ground_truth": g["answer"],
            }
            for g in golden
        ])

        ragas_scores = ragas_evaluate(
            ragas_data,
            metrics=[faithfulness, answer_relevancy, context_recall],
        )

        mlflow.log_metric("ragas_faithfulness", float(ragas_scores["faithfulness"]))
        mlflow.log_metric("ragas_answer_relevancy", float(ragas_scores["answer_relevancy"]))
        mlflow.log_metric("ragas_context_recall", float(ragas_scores["context_recall"]))

        logger.info(f"RAGAS scores — {ragas_scores}")

        final_scores = {
            "opik_hallucination": avg_hallucination,
            "opik_relevance": avg_relevance,
            "ragas_faithfulness": float(ragas_scores["faithfulness"]),
            "ragas_answer_relevancy": float(ragas_scores["answer_relevancy"]),
            "ragas_context_recall": float(ragas_scores["context_recall"]),
        }

    return final_scores
