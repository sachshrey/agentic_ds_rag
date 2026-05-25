# nodes/model_selection.py
"""
Model selection node.

Changes vs prior version:
- Deterministic cardinality summary passed to LLM (no hallucination on high-card cols)
- RAG retrieval reduced to k=2 (was 4) — fewer chunks, less prompt bloat
- Trimmed profile dict before injecting into prompt (excludes raw dtypes/skewness blobs)
- Timeout fallback: if LLM call fails or exceeds timeout, use heuristic top candidate
- model_source recorded in state so report can explain how model was chosen
"""
from __future__ import annotations
import logging

from state import AgentState, ModelCandidate
from schemas import ModelSelectionOutput
from prompts import MODEL_SELECTION_PROMPT
from rag import retrieve_for_node
from utils.llm import get_llm

logger = logging.getLogger(__name__)

MODEL_REGISTRY: dict[str, list[str]] = {
    "binary_classification": ["catboost", "xgboost", "lightgbm", "random_forest", "logistic_regression"],
    "multiclass_classification": ["catboost", "xgboost", "lightgbm", "random_forest"],
    "regression": ["catboost", "xgboost", "lightgbm", "random_forest", "ridge"],
    "time_series_forecasting": ["lightgbm", "xgboost"],
}


def _compute_heuristic_scores(state: AgentState) -> list[ModelCandidate]:
    """Score candidate models against dataset characteristics. Pure heuristics — no LLM."""
    profile = state.profile
    task = state.task
    avg_null = sum(profile.null_percentages.values()) / max(len(profile.null_percentages), 1)
    high_card_count = sum(1 for v in profile.cardinality.values() if v > 50)

    eda_stats = state.processed_data.get("eda_stats", {})
    imbalance_ratio = eda_stats.get("imbalance_ratio", 1.0)

    candidates = []
    for model_name in MODEL_REGISTRY.get(task.task_type, []):
        score, pros, cons, points = 0.0, [], [], []

        # Size-based scoring
        if profile.size_category == "large":
            if model_name == "lightgbm":   score += 2.5; points.append("Fastest on large data (histogram algorithm)")
            elif model_name == "xgboost":  score += 2.0; points.append("Scales well to large data")
            elif model_name == "random_forest": score -= 1.0; cons.append("Slower than gradient boosting on large data")
        elif profile.size_category == "small":
            if model_name == "logistic_regression": score += 1.5; pros.append("Robust on small datasets, low overfit risk")
            elif model_name in ("xgboost", "lightgbm", "catboost"): score -= 0.5; cons.append("Overfit risk on small data")

        # Missing value handling
        if avg_null > 5.0:
            if model_name == "catboost":                score += 2.0; pros.append("Native NaN handling — no imputation needed")
            elif model_name in ("xgboost", "lightgbm"): score += 1.0; points.append("Handles NaN reasonably in tree splits")

        # High cardinality categoricals — uses computed count, not LLM guess
        if high_card_count >= 2:
            if model_name == "catboost":             score += 2.0; pros.append(f"Native handling of {high_card_count} high-cardinality cols")
            elif model_name == "logistic_regression": score -= 1.5; cons.append("Needs heavy preprocessing for high-cardinality cols")

        # Class imbalance
        if imbalance_ratio > 5.0:
            if model_name in ("xgboost", "lightgbm"):  score += 1.5; pros.append("scale_pos_weight handles imbalance directly")
            elif model_name == "random_forest":         score += 1.0; pros.append("class_weight='balanced' is effective")

        # Estimated training time
        if profile.size_category == "large":
            time_est = {"lightgbm": "2-8 min", "xgboost": "5-15 min", "catboost": "10-20 min"}.get(model_name, "3-10 min")
        else:
            time_est = {"logistic_regression": "<1 min", "ridge": "<1 min"}.get(model_name, "1-5 min")

        candidates.append(ModelCandidate(
            name=model_name,
            heuristic_score=round(score, 2),
            reasoning_points=points,
            pros=pros,
            cons=cons,
            estimated_training_time=time_est,
        ))

    return sorted(candidates, key=lambda x: x.heuristic_score, reverse=True)[:4]


def _slim_profile(state: AgentState) -> dict:
    """
    Build a compact profile dict for the model selection prompt.
    Excludes dtypes, skewness, and raw correlations — these bloat the prompt
    without adding value for model selection decisions.
    Includes pre-computed cardinality summary so LLM never needs to infer it.
    """
    profile = state.profile
    high_card_cols = {
        col: card
        for col, card in profile.cardinality.items()
        if card > 50
    }
    return {
        "rows": profile.rows,
        "columns": profile.columns,
        "size_category": profile.size_category,
        "memory_usage_mb": round(profile.memory_usage_mb, 2),
        "avg_null_pct": round(
            sum(profile.null_percentages.values()) / max(len(profile.null_percentages), 1), 2
        ),
        "high_cardinality_cols": high_card_cols,      # exact counts — no LLM inference
        "high_cardinality_count": len(high_card_cols),
        "duplicate_rows": profile.duplicate_rows,
    }


def model_selection_node(state: AgentState) -> AgentState:
    candidates = _compute_heuristic_scores(state)
    state.model_candidates = candidates

    # RAG: prior experiment results — k=2 keeps prompt small
    rag_docs = retrieve_for_node(
        node_name="model_selection",
        task_type=state.task.task_type,
        domain=state.task.domain_detected,
        query=(
            f"best model for {state.task.domain_detected or 'tabular'} "
            f"{state.task.task_type} similar dataset"
        ),
        k=2,
    )
    state.rag_retrievals.append(f"MODEL_SELECTION: {len(rag_docs)} prior experiment docs")

    # Slim profile — exclude raw blobs that bloat the prompt
    slim_profile = _slim_profile(state)

    # Slim task — only the fields relevant to model selection
    slim_task = {
        "task_type": state.task.task_type,
        "domain_detected": state.task.domain_detected,
        "recommended_validation": state.task.recommended_validation,
    }

    # LLM validates heuristic ranking — with timeout fallback
    llm = get_llm("model_selection").with_structured_output(ModelSelectionOutput)
    result: ModelSelectionOutput | None = None
    model_source = "llm"

    try:
        result = llm.invoke(
            MODEL_SELECTION_PROMPT.format(
                candidates=[
                    {
                        "name": c.name,
                        "heuristic_score": c.heuristic_score,
                        "pros": c.pros,
                        "cons": c.cons,
                        "estimated_training_time": c.estimated_training_time,
                    }
                    for c in candidates
                ],
                profile=slim_profile,
                task=slim_task,
                baseline_metrics=state.baseline_metrics,
                rag_prior_experiments="\n".join(rag_docs) or "No prior experiments found.",
            )
        )
    except Exception as e:
        logger.warning(
            f"MODEL_SELECTION: LLM call failed ({e}). "
            f"Falling back to heuristic top candidate: {candidates[0].name}"
        )
        model_source = "heuristic_fallback"

    # If LLM succeeded, validate its choice is in candidates
    if result is not None:
        valid_names = {c.name for c in candidates}
        if result.chosen_model not in valid_names:
            logger.warning(
                f"LLM chose invalid model '{result.chosen_model}', "
                f"falling back to heuristic top: {candidates[0].name}"
            )
            result = result.model_copy(update={"chosen_model": candidates[0].name})
            model_source = "heuristic_fallback"
        state.chosen_model = result.chosen_model
    else:
        # Full fallback — take heuristic top
        state.chosen_model = candidates[0].name

    state.model_source = model_source

    chosen_score = next(
        (c.heuristic_score for c in candidates if c.name == state.chosen_model), "?"
    )
    reasoning = result.reasoning if result else "Heuristic fallback (LLM unavailable)."
    state.reasoning_log.append(
        f"MODEL SELECTED: {state.chosen_model} "
        f"(heuristic_score={chosen_score}, source={model_source}) | "
        f"REASONING: {reasoning}"
    )
    return state
