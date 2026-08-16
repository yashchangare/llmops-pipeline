# LLMOps Pipeline
RAG + Observability + Model Routing

**Stack:** ZenML · LlamaIndex · Qdrant · MLflow · DVC · Opik · LiteLLM · GitHub Actions

---

## Setup (Day 1 — do this first)

### 1. Clone & create environment
```bash
git clone https://github.com/YOUR_USERNAME/llmops-pipeline.git
cd llmops-pipeline
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. Copy env file and fill in your keys
```bash
copy .env.example .env
```

### 3. Start Qdrant (vector DB) via Docker
```bash
docker run -p 6333:6333 qdrant/qdrant
```

### 4. Start MLflow tracking server
```bash
mlflow server --host 0.0.0.0 --port 5000
```

### 5. Start LiteLLM gateway
```bash
litellm --config gateway/litellm_config.yaml --port 4000
```

### 6. Initialize ZenML
```bash
zenml init
zenml integration install mlflow qdrant -y
```

### 7. Initialize DVC
```bash
dvc init
dvc add data/raw/
git add .
git commit -m "init: project setup"
```

### 8. Run the full pipeline
```bash
python run_pipeline.py
```

### 9. Open dashboards
- MLflow:  http://localhost:5000
- Opik:    http://localhost:5173  (run: `opik server start`)
- Qdrant:  http://localhost:6333/dashboard

---

## Project Structure
```
llmops-pipeline/
├── data/
│   ├── raw/              # source documents (tracked by DVC)
│   └── processed/        # chunked + embedded docs
├── steps/
│   ├── ingest.py         # load + chunk documents
│   ├── embed.py          # embed + store in Qdrant
│   ├── retrieve.py       # hybrid search retrieval
│   ├── generate.py       # LLM generation via LiteLLM
│   └── evaluate.py       # Opik + RAGAS evaluation
├── pipelines/
│   ├── rag_pipeline.py   # ZenML RAG pipeline
│   └── eval_pipeline.py  # ZenML eval pipeline
├── evaluation/
│   └── golden_dataset.json  # ground truth Q&A pairs
├── gateway/
│   └── litellm_config.yaml  # model routing config
├── .github/workflows/
│   └── ci.yml            # GitHub Actions CI/CD
├── run_pipeline.py
├── requirements.txt
└── .env.example
```

---

## What each tool does in this project

| Tool | Role |
|------|------|
| ZenML | Orchestrates the pipeline steps, handles artifacts |
| LlamaIndex | Document loading, chunking, hybrid search |
| Qdrant | Vector database storing embeddings |
| MLflow | Tracks experiments, logs metrics per run |
| DVC | Versions the raw data files (like Git for data) |
| Opik | Traces every LLM call, runs LLM-as-judge eval |
| LiteLLM | Routes requests to Ollama or OpenAI with fallback |
| GitHub Actions | Runs eval on every push, gates on quality threshold |
