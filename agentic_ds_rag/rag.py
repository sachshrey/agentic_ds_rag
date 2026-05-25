# rag.py
"""
ChromaDB-based RAG layer.
Metadata-filtered retrieval — the most important RAG implementation detail.
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional
import os
import sqlite3

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

logger = logging.getLogger(__name__)

COLLECTION_NAME = "ml_knowledge"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# Module-level singletons (initialized once per process)
_client: chromadb.PersistentClient | None = None
_collection: chromadb.Collection | None = None


def _stored_embedding_count(cache_dir: str = "cache") -> int:
    """Read local Chroma row count without initializing embedding models."""
    db_path = Path(cache_dir) / "chroma_db" / "chroma.sqlite3"
    if not db_path.exists():
        return 0
    try:
        with sqlite3.connect(db_path) as conn:
            return int(conn.execute("select count(*) from embeddings").fetchone()[0])
    except Exception:
        return 0


def get_collection(cache_dir: str = "cache") -> chromadb.Collection:
    """Get or create the ChromaDB collection. Idempotent."""
    global _client, _collection
    if _collection is not None:
        return _collection

    Path(cache_dir).mkdir(exist_ok=True)

    _client = chromadb.PersistentClient(path=f"{cache_dir}/chroma_db")

    # Local embedding model — free, fast, good quality
    _EMBED_DEVICE = os.getenv("EMBED_DEVICE", "cpu")
    logger.info(f"RAG: Embedding device = {_EMBED_DEVICE}")
    ef = SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL,
        device=_EMBED_DEVICE,
        normalize_embeddings=True,
    )

    _collection = _client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    logger.info(
        f"RAG: ChromaDB collection ready — {_collection.count()} documents"
    )
    return _collection


def retrieve_for_node(
    node_name: str,
    task_type: str,
    domain: Optional[str],
    query: str,
    k: int = 4,
) -> list[str]:
    """
    Metadata-filtered retrieval.
    Only returns documents tagged for this node's category.
    Without this filter, retrieval confuses preprocessing docs with eval docs.
    """
    if _collection is None and _stored_embedding_count() == 0:
        logger.warning("RAG: Knowledge base is empty. Run `python ingest_docs.py` first.")
        return []

    collection = get_collection()
    total = collection.count()

    if total == 0:
        logger.warning("RAG: Knowledge base is empty. Run `python ingest_docs.py` first.")
        return []

    # Filter strictly by node category
    where_filter = {"category": {"$eq": node_name}}

    try:
        results = collection.query(
            query_texts=[query],
            n_results=min(k, total),
            where=where_filter,
        )
        docs = results["documents"][0] if results["documents"] else []
        logger.debug(f"RAG: {node_name} retrieved {len(docs)} docs for query: {query[:60]}...")
        return docs

    except Exception as e:
        logger.warning(f"RAG: Retrieval failed for {node_name}: {e}")
        return []


def add_experiment_to_rag(
    task_type: str,
    domain: str | None,
    model: str,
    metrics: dict[str, float],
    key_decisions: list[str],
    run_id: str,
) -> None:
    """
    Store experiment result back into RAG.
    Next run on a similar dataset can retrieve this as a prior experiment.
    This is what gives your system learning-from-experience capability.
    """
    if _collection is None and _stored_embedding_count() == 0:
        logger.info("RAG: Skipping experiment memory write because knowledge base is empty.")
        return

    try:
        collection = get_collection()
    except Exception as e:
        logger.warning(f"RAG: Skipping experiment memory write; collection unavailable: {e}")
        return

    doc = (
        f"Past experiment: {task_type} on {domain or 'general'} dataset. "
        f"Best model: {model}. "
        f"Metrics: {metrics}. "
        f"Key decisions: {'; '.join(key_decisions[-3:])}. "
        f"MLflow run ID: {run_id}"
    )

    try:
        collection.upsert(
            documents=[doc],
            metadatas=[{
                "category": "model_selection",  # Retrievable by model selection node
                "task_type": task_type,
                "domain": domain or "general",
                "model": model,
                "run_id": run_id,
            }],
            ids=[f"exp_{run_id}"],
        )
        logger.info(f"RAG: Experiment {run_id} stored for future retrieval.")
    except Exception as e:
        logger.warning(f"RAG: Failed to store experiment: {e}")
