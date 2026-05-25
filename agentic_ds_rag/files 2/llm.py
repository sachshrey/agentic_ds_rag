# utils/llm.py
"""
LLM factory. Routes each node to the right model.
High-stakes nodes → GPT-4.1 (best reasoning quality)
Efficient nodes   → Ollama Llama 3.1 8B (free, local, fast)
Temperature = 0 everywhere. DS decisions must be deterministic.

Changes vs prior version:
- ChatOllama now has timeout=30s. Without this, a cold or hung Ollama
  instance silently burns 100+ seconds per call (observed: 100.98s on telco run).
  If the call times out, model_selection_node falls back to the heuristic top candidate.
- max_tokens added to ChatOllama (prevents runaway generation on verbose models).
"""
from __future__ import annotations
import os
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

# Nodes where correct reasoning significantly impacts downstream results
HIGH_STAKES_NODES = {"task_detector", "critic", "explainer", "report"}

# Nodes where simpler reasoning is acceptable (heuristics do heavy lifting)
EFFICIENT_NODES = {
    "eda", "preprocessing", "feature_engineering",
    "model_selection", "baseline",
}

_OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "30"))  # seconds; override via env


def get_llm(node_name: str):
    """Return the appropriate LLM for the given node."""
    if node_name in HIGH_STAKES_NODES:
        return ChatOpenAI(
            model="gpt-4.1",
            temperature=0,
            max_tokens=2000,
            api_key=os.environ["OPENAI_API_KEY"],
        )
    # Local Ollama — free, no API key, runs on Mac/Linux
    return ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
        temperature=0,
        num_ctx=4096,
        num_predict=512,          # cap output length; efficient nodes never need more
        timeout=_OLLAMA_TIMEOUT,  # hard timeout — surfaces hung instances immediately
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )
