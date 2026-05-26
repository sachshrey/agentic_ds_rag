# utils/caching.py
"""
Three-layer caching strategy:
1. Profile cache     — SHA256 of file → skip expensive pandas profiling on reruns
2. LLM call cache    — SQLite → same prompt = instant return during development
3. RAG result cache  — lru_cache in memory → same query per session = single DB hit
"""
from __future__ import annotations
import hashlib
import logging
from functools import lru_cache
from pathlib import Path

from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache

from state import DatasetProfile

logger = logging.getLogger(__name__)


def setup_llm_cache(cache_dir: str = "cache", enabled: bool = True) -> None:
    """
    Cache LLM responses to SQLite. Massively speeds up development iteration.
    Disable for production or when fresh reasoning is required (critic node).
    """
    if not enabled:
        return
    Path(cache_dir).mkdir(exist_ok=True)
    set_llm_cache(SQLiteCache(database_path=f"{cache_dir}/llm_cache.db"))
    logger.info("LLM response caching enabled (SQLite).")


def get_file_hash(file_path: str) -> str:
    """SHA256 of file content — used as cache key for profiling."""
    with open(file_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def get_cached_profile(file_path: str, cache_dir: str = "cache") -> DatasetProfile | None:
    file_hash = get_file_hash(file_path)
    cache_path = Path(cache_dir) / f"profile_{file_hash}.json"
    if cache_path.exists():
        logger.info(f"Profile cache hit: {file_path}")
        return DatasetProfile.model_validate_json(cache_path.read_text())
    return None


def save_profile_cache(
    file_path: str, profile: DatasetProfile, cache_dir: str = "cache"
) -> None:
    Path(cache_dir).mkdir(exist_ok=True)
    file_hash = get_file_hash(file_path)
    cache_path = Path(cache_dir) / f"profile_{file_hash}.json"
    cache_path.write_text(profile.model_dump_json())


@lru_cache(maxsize=256)
def cached_retrieve(node_name: str, query: str) -> tuple[str, ...]:
    """
    In-memory cache for RAG retrievals within a session.
    Same node + same query → single ChromaDB hit.
    Returns tuple (hashable) instead of list.
    """
    from rag import retrieve_for_node
    results = retrieve_for_node(node_name=node_name, task_type="all",
                                domain=None, query=query)
    return tuple(results)
