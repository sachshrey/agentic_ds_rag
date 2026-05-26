# nodes/task_detector.py
from __future__ import annotations
import logging
from typing import Optional

from state import AgentState, TaskDetection
from schemas import TaskDetectionOutput
from prompts import TASK_DETECTION_PROMPT, TASK_HEURISTICS, LEAKAGE_PATTERNS
from rag import retrieve_for_node
from utils.llm import get_llm

logger = logging.getLogger(__name__)

DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "medical":   ["glucose", "blood_pressure", "diagnosis", "patient", "clinical",
                  "hospital", "medication", "bmi", "cholesterol", "symptoms"],
    "financial": ["loan", "credit", "fraud", "transaction", "balance", "income",
                  "debt", "interest", "default", "payment", "account"],
    "ecommerce": ["product", "cart", "order", "customer", "purchase", "sku",
                  "category", "discount", "churn", "subscription"],
    "hr":        ["employee", "salary", "department", "attrition", "performance",
                  "tenure", "promotion", "satisfaction"],
    "telco":     ["churn", "contract", "monthly_charges", "internet_service",
                  "phone_service", "streaming"],
}


def _detect_domain(columns: list[str]) -> Optional[str]:
    """Keyword-based domain detection — no LLM needed."""
    col_text = " ".join(col.lower() for col in columns)
    scores = {
        domain: sum(1 for kw in kws if kw in col_text)
        for domain, kws in DOMAIN_KEYWORDS.items()
    }
    best = max(scores, key=scores.get)
    return best if scores[best] >= 2 else None


def _find_likely_target(columns: list[str], dtypes: dict[str, str]) -> Optional[str]:
    """Heuristic target column detection."""
    TARGET_KEYWORDS = [
        "target", "label", "class", "churn", "survived", "outcome",
        "y", "output", "result", "default", "fraud", "purchase",
        "price", "sales", "revenue", "salary", "score", "attrition",
    ]
    for kw in TARGET_KEYWORDS:
        for col in columns:
            if kw in col.lower():
                return col
    # Convention: last column is often the target
    return columns[-1] if columns else None


def task_detector_node(state: AgentState) -> AgentState:
    profile = state.profile
    columns = list(profile.dtypes.keys())

    # ── Deterministic domain detection ───────────────────────────────────
    detected_domain = _detect_domain(columns)

    # ── Conditional RAG retrieval ─────────────────────────────────────────
    rag_query = (
        f"ML task detection best practices for "
        f"{'domain:' + detected_domain if detected_domain else 'general tabular'} dataset"
    )
    rag_docs = retrieve_for_node(
        node_name="task_detector",
        task_type="all",
        domain=detected_domain,
        query=rag_query,
        k=3,
    )
    rag_context = "\n".join(rag_docs) if rag_docs else "No prior context available."
    state.rag_retrievals.append(
        f"TASK_DETECTOR: {len(rag_docs)} docs retrieved | domain={detected_domain}"
    )

    # ── Build profile summary (never dump full DataFrame into prompt) ─────
    profile_summary = {
        "rows": profile.rows,
        "columns": profile.columns,
        "dtypes": profile.dtypes,
        "null_percentages": profile.null_percentages,
        "cardinality": profile.cardinality,
        "high_correlations": profile.correlations_high[:10],  # top 10
        "skewness_extremes": {
            k: v for k, v in profile.skewness.items() if abs(v) > 2.0
        },
        "size_category": profile.size_category,
        "duplicate_rows": profile.duplicate_rows,
        "detected_domain": detected_domain,
        "likely_target_hint": _find_likely_target(columns, profile.dtypes),
        "user_context": state.user_context or "None provided",
    }

    # ── LLM call with structured output ──────────────────────────────────
    llm = get_llm("task_detector").with_structured_output(TaskDetectionOutput)

    result: TaskDetectionOutput = llm.invoke(
        TASK_DETECTION_PROMPT.format(
            profile=profile_summary,
            heuristics=TASK_HEURISTICS,
            leakage_patterns=LEAKAGE_PATTERNS,
            rag_context=rag_context,
            user_context=state.user_context or "None",
        )
    )

    # ── Convert to state model ────────────────────────────────────────────
    state.task = TaskDetection(
        task_type=result.task_type,
        target_column=result.target_column,
        confidence=result.confidence,
        reasoning=result.reasoning,
        leakage_suspects=result.leakage_suspects,
        dataset_category=result.dataset_category,
        domain_detected=result.domain_detected or detected_domain,
        recommended_metrics=result.recommended_metrics,
        recommended_validation=result.recommended_validation,
    )

    # ── Drop high-confidence leakage suspects immediately ─────────────────
    confirmed_leakage = [
        col for col in result.leakage_suspects if col in profile.dtypes
    ]
    state.dropped_columns.extend(confirmed_leakage)

    state.reasoning_log.append(
        f"TASK DETECTOR: task={state.task.task_type} | "
        f"target={state.task.target_column} | "
        f"confidence={state.task.confidence:.0%} | "
        f"domain={state.task.domain_detected} | "
        f"leakage_dropped={confirmed_leakage} | "
        f"reasoning={state.task.reasoning}"
    )
    return state
