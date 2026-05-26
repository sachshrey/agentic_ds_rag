# main.py
"""
Agentic Data Science RAG System — CLI Entry Point
Usage:
  python main.py data/titanic.csv
  python main.py data/house_prices.csv --context "Predicting sale price. Iowa housing market."
  python main.py data/sales.csv --ingest-docs --output-dir my_outputs
"""
from __future__ import annotations
import argparse
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()  # Load .env before any LangChain/OpenAI imports

from utils.caching import setup_llm_cache
from graph import compile_graph
from state import AgentState

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-20s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def _state_get(state: object, key: str, default=None):
    """Read from LangGraph's dict-like state or a Pydantic AgentState."""
    if hasattr(state, "get"):
        return state.get(key, default)
    return getattr(state, key, default)


def run(
    file_path: str,
    user_context: str | None = None,
    output_dir: str = "outputs",
    use_cache: bool = True,
) -> dict:
    """Main pipeline runner. Returns final state as dict."""
    # Validate file
    fp = Path(file_path)
    if not fp.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")
    if fp.suffix.lower() not in (".csv", ".xlsx", ".xls", ".parquet"):
        raise ValueError(f"Unsupported file type: {fp.suffix}")
    if fp.stat().st_size > 500 * 1024 * 1024:  # 500MB limit
        raise ValueError("File exceeds 500MB limit.")

    # Create output directories
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path("cache").mkdir(exist_ok=True)

    # Setup LLM caching (disable for fresh reasoning runs)
    setup_llm_cache(cache_dir="cache", enabled=use_cache)

    logger.info(f"🚀 Starting Agentic DS RAG | File: {file_path}")

    initial_state = AgentState(
        file_path=str(fp.resolve()),
        user_context=user_context,
        output_dir=output_dir,
    )

    graph = compile_graph(enable_checkpointing=True)
    config = {
        "configurable": {"thread_id": "pipeline_main"},
        "recursion_limit": 80,
    }

    final_state = graph.invoke(initial_state, config=config)

    # Summary
    report = _state_get(final_state, "report_path", "N/A") or "N/A"
    retries = _state_get(final_state, "retry_count", 0)
    errors = _state_get(final_state, "errors", [])
    experiment = _state_get(final_state, "experiment")
    metrics = (
        experiment.metrics
        if hasattr(experiment, "metrics")
        else (experiment or {}).get("metrics", {})
    )

    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    logger.info(f"✅ Pipeline complete!")
    logger.info(f"   Report:  {report}")
    logger.info(f"   Metrics: {metrics}")
    logger.info(f"   Retries: {retries}")
    if errors:
        logger.warning(f"   Errors ({len(errors)}): {errors}")
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    return final_state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Agentic Data Science RAG System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("file", help="Path to dataset (CSV, Excel, Parquet)")
    parser.add_argument(
        "--context",
        default=None,
        help='Optional hint about the dataset, e.g. "Telecom churn prediction dataset"',
    )
    parser.add_argument(
        "--output-dir", default="outputs", help="Directory for all output files"
    )
    parser.add_argument(
        "--ingest-docs",
        action="store_true",
        help="(Re-)ingest RAG knowledge base before running pipeline",
    )
    parser.add_argument(
        "--no-cache", action="store_true", help="Disable LLM response caching"
    )

    args = parser.parse_args()

    if args.ingest_docs:
        logger.info("Ingesting RAG knowledge base...")
        from ingest_docs import ingest
        ingest()

    run(
        file_path=args.file,
        user_context=args.context,
        output_dir=args.output_dir,
        use_cache=not args.no_cache,
    )
