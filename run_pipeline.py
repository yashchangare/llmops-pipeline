"""
run_pipeline.py — entry point for the full LLMOps pipeline.

Usage:
    python run_pipeline.py --mode index          # ingest + embed documents
    python run_pipeline.py --mode query          # run a single query
    python run_pipeline.py --mode eval           # run full eval against golden dataset
    python run_pipeline.py --mode all            # run everything end to end
"""
import argparse
import os
import mlflow
from dotenv import load_dotenv

load_dotenv()

from pipelines.rag_pipeline import indexing_pipeline
from pipelines.eval_pipeline import evaluation_pipeline
from pipelines.rag_pipeline import run_query_pipeline

# Point ZenML's MLflow integration at your local server
os.environ["MLFLOW_TRACKING_URI"] = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

def run_index():
    print("\n── Running indexing pipeline ──")
    print("This will load docs from data/raw/, chunk, embed, and store in Qdrant.")
    print("Make sure: Docker (Qdrant) + Ollama (nomic-embed-text) are running.\n")
    indexing_pipeline(data_dir="data/raw")
    print("✓ Indexing complete. Check Qdrant dashboard: http://localhost:6333/dashboard")

def run_query(query: str = None):
    if not query:
        query = input("\nEnter your query: ")
    print(f"\n── Running query pipeline ──\nQuery: {query}\n")
    result = run_query_pipeline(query=query)
    print(f"\nAnswer: {result['answer']}")
    print(f"Sources: {result['sources']}")
    print(f"Model used: {result['model_used']}")
    print(f"\nCheck Opik traces: http://localhost:5173")


def run_eval():
    print("\n── Running evaluation pipeline ──")
    print("Running all golden dataset questions through RAG, scoring with Opik + RAGAS.\n")
    scores = evaluation_pipeline()
    print("\n── Evaluation Results ──")
    for metric, score in scores.items():
        status = "✓" if score > 0.7 else "✗"
        print(f"  {status} {metric}: {score:.3f}")
    print(f"\nCheck MLflow: http://localhost:5000")

    # CI gate: fail if faithfulness < 0.7
    if scores.get("ragas_faithfulness", 0) < 0.7:
        print("\n✗ QUALITY GATE FAILED: faithfulness below 0.7 threshold")
        exit(1)
    print("\n✓ All quality gates passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["index", "query", "eval", "all"], default="all")
    parser.add_argument("--query", type=str, help="Query string for query mode")
    args = parser.parse_args()

    if args.mode == "index":
        run_index()
    elif args.mode == "query":
        run_query(args.query)
    elif args.mode == "eval":
        run_eval()
    elif args.mode == "all":
        run_index()
        run_query("What is retrieval-augmented generation?")
        run_eval()
