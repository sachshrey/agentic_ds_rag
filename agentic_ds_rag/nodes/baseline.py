# nodes/baseline.py
"""
Always train the simplest possible model before any advanced model.
Two purposes:
1. Sanity check — if logistic regression gets 0.95 AUC, XGBoost may not be needed
2. Critic comparison — advanced model must beat this meaningfully
"""
from __future__ import annotations
import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import f1_score, roc_auc_score, r2_score, mean_absolute_error, mean_squared_error

from state import AgentState

logger = logging.getLogger(__name__)


def _positive_class_scores(model, y_true, X):
    classes = list(getattr(model, "classes_", []))
    preferred = ["Yes", "yes", "Y", "y", "True", True, 1, "1", "Churn"]
    positive_label = next((label for label in preferred if label in classes), classes[1])
    positive_idx = classes.index(positive_label)
    y_prob = model.predict_proba(X)[:, positive_idx]
    y_binary = (pd.Series(y_true).to_numpy() == positive_label).astype(int)
    return y_binary, y_prob


def baseline_trainer_node(state: AgentState) -> AgentState:
    X_train = state.processed_data["X_train"]
    X_test = state.processed_data["X_test"]
    y_train = state.processed_data["y_train"]
    y_test = state.processed_data["y_test"]
    task = state.task.task_type

    if task not in (
        "binary_classification",
        "multiclass_classification",
        "regression",
        "time_series_forecasting",
    ):
        state.reasoning_log.append(
            f"BASELINE: Skipped for task={task} (no simple baseline defined)"
        )
        return state

    # Select simplest possible model
    if "classification" in task:
        model = LogisticRegression(
            max_iter=1000, random_state=42, class_weight="balanced"
        )
        model_name = "LogisticRegression"
    else:
        model = Ridge(alpha=1.0)
        model_name = "Ridge"

    try:
        model.fit(X_train, y_train)
    except Exception as e:
        state.errors.append(f"Baseline training failed: {e}")
        return state

    # Compute baseline metrics
    metrics: dict[str, float] = {}
    if task == "binary_classification":
        y_pred = model.predict(X_test)
        y_binary, y_prob = _positive_class_scores(model, y_test, X_test)
        metrics["accuracy"] = round(float(model.score(X_test, y_test)), 4)
        metrics["f1_weighted"] = round(float(f1_score(y_test, y_pred, average="weighted")), 4)
        metrics["roc_auc"] = round(float(roc_auc_score(y_binary, y_prob)), 4)
    elif task == "multiclass_classification":
        y_pred = model.predict(X_test)
        metrics["accuracy"] = round(float(model.score(X_test, y_test)), 4)
        metrics["f1_weighted"] = round(float(f1_score(y_test, y_pred, average="weighted")), 4)
    elif task in ("regression", "time_series_forecasting"):
        y_pred = model.predict(X_test)
        metrics["r2"] = round(float(r2_score(y_test, y_pred)), 4)
        metrics["rmse"] = round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4)
        metrics["mae"] = round(float(mean_absolute_error(y_test, y_pred)), 4)

    state.baseline_metrics = metrics
    if state.low_confidence_mode:
        state.chosen_model = model_name
        state.trained_model = model
        state.model_source = "safe_path_baseline"
        state.reasoning_log.append(
            f"SAFE PATH: {model_name} baseline trained and written to final model state."
        )
    state.reasoning_log.append(
        f"BASELINE ({model_name}): {metrics} — Advanced model must beat this meaningfully."
    )
    return state
