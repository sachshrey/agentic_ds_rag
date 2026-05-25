# nodes/explainer.py
from __future__ import annotations
import logging
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from state import AgentState
from schemas import ExplainabilityNarration
from prompts import SHAP_NARRATION_PROMPT
from utils.llm import get_llm

logger = logging.getLogger(__name__)


def explainer_node(state: AgentState) -> AgentState:
    if state.trained_model is None:
        state.errors.append("Explainer skipped — no trained model.")
        return state

    model = state.trained_model
    X_test = state.processed_data["X_test"]
    Path(state.output_dir).mkdir(parents=True, exist_ok=True)

    try:
        # ── Choose SHAP explainer by model type ───────────────────────────
        if hasattr(model, "get_booster") or hasattr(model, "feature_importances_"):
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X_test)
        else:
            explainer = shap.LinearExplainer(model, X_test)
            shap_vals = explainer.shap_values(X_test)

        # Binary classification: use positive class SHAP
        plot_vals = shap_vals[1] if isinstance(shap_vals, list) else shap_vals

        # ── Beeswarm summary plot ─────────────────────────────────────────
        plt.figure(figsize=(10, 8))
        shap.summary_plot(plot_vals, X_test, show=False)
        plt.tight_layout()
        summary_path = f"{state.output_dir}/shap_summary.png"
        plt.savefig(summary_path, dpi=150, bbox_inches="tight")
        plt.close()

        # ── Waterfall for first test instance ─────────────────────────────
        base_val = (
            explainer.expected_value[1]
            if isinstance(explainer.expected_value, (list, np.ndarray))
            else explainer.expected_value
        )
        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(
            shap.Explanation(
                values=plot_vals[0],
                base_values=base_val,
                data=X_test.iloc[0],
                feature_names=list(X_test.columns),
            ),
            show=False,
        )
        plt.tight_layout()
        waterfall_path = f"{state.output_dir}/shap_waterfall.png"
        plt.savefig(waterfall_path, dpi=150, bbox_inches="tight")
        plt.close()

        # ── Top features for LLM narration ────────────────────────────────
        mean_shap = np.asarray(np.abs(plot_vals).mean(axis=0)).reshape(-1)
        top_features = sorted(
            zip(X_test.columns, mean_shap), key=lambda x: x[1], reverse=True
        )[:10]
        shap_table = [
            {"feature": str(feature), "mean_abs_shap": round(float(value), 6)}
            for feature, value in top_features
        ]
        state.shap_importance_table = shap_table
        if state.experiment is not None:
            state.experiment.feature_importance = {
                row["feature"]: row["mean_abs_shap"] for row in shap_table
            }

        state.experiment.shap_values_path = summary_path

        # ── LLM narrates in plain English ─────────────────────────────────
        llm = get_llm("explainer").with_structured_output(ExplainabilityNarration)
        narration: ExplainabilityNarration = llm.invoke(
            SHAP_NARRATION_PROMPT.format(
                top_features=[(f, round(float(v), 4)) for f, v in top_features],
                task_type=state.task.task_type,
                target_column=state.task.target_column,
                domain=state.task.domain_detected or "general",
                model_name=state.chosen_model,
                metrics=state.experiment.metrics,
            )
        )

        state.processed_data["shap_narration"] = narration.narration
        state.processed_data["shap_top_features"] = top_features
        state.processed_data["shap_actionable_insights"] = narration.actionable_insights

        state.reasoning_log.append(
            f"SHAP: Top 3 features: {[f for f, _ in top_features[:3]]} | "
            f"Plots: shap_summary.png, shap_waterfall.png"
        )
        state.reasoning_log.append("SHAP importance table computed.")

    except Exception as e:
        state.errors.append(f"SHAP explainer failed: {e}")
        logger.warning(f"SHAP failed: {e}. Continuing without explainability plots.")

    return state
