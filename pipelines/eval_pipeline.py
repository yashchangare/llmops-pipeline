from steps.retrieve import hybrid_retrieve
from steps.generate import generate_answer
from steps.evaluate import evaluate_pipeline
import json


def load_test_queries(path: str = "evaluation/golden_dataset.json"):
    with open(path) as f:
        return [item["question"] for item in json.load(f)]


def evaluation_pipeline():
    queries = load_test_queries()
    all_results = []

    for query in queries:
        chunks = hybrid_retrieve(query=query)
        result = generate_answer(query=query, context_chunks=chunks)
        all_results.append(result)

    scores = evaluate_pipeline(results=all_results)
    return scores