# Claude Review Notes

This is an agentic data-science RAG project built around LangGraph, ChromaDB,
MLflow, local Ollama models, and OpenAI for high-stakes reasoning nodes.

## Quick Run

1. Create and activate a virtual environment.
2. Install dependencies from `requirements.txt`.
3. Set environment variables:

```bash
export OPENAI_API_KEY="..."
export OLLAMA_MODEL="llama3"
export OLLAMA_BASE_URL="http://localhost:11434"
```

4. From this directory, run:

```bash
python main.py data/sample_binary_classification.csv --output-dir outputs/review_run
```

The included sample dataset is intentionally tiny to keep review runs cheap.
The RAG knowledge base is optional for this smoke run; if Chroma is empty, RAG
retrieval degrades to no-context mode instead of blocking on embedding downloads.

## What To Check

- LangGraph flow in `graph.py`.
- RAG integration and metadata filtering in `rag.py`.
- LLM structured-output schemas in `schemas.py`.
- Data-science nodes in `nodes/`.
- Final report and generated plots under `outputs/review_run/` after a run.
