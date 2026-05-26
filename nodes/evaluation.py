# nodes/evaluation.py
"""Auto metric selection based on task type and imbalance."""
from __future__ import annotations
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    roc_auc_score, average_precision_score,
    r2_score, mean_absolute_error, mean_squared_error,
)

from state import AgentState, ExperimentResult

logger = logging.getLogger(__name__)

VALIDATION_STRATEGIES = {
    "binary_classification_small":  "stratified_kfold_10_repeated",
    "binary_classification_medium": "stratified_kfold_5",
    "binary_classification_large":  "stratified_kfold_3",
    "multiclass_classification_small": "stratified_kfold_10",
    "multiclass_classification_medium": "stratified_kfold_5",
    "regression_small":  "kfold_10",
    "regression_medium": "kfold_5",
    "regression_large":  "kfold_3",
    "time_series_forecasting": "walk_forward_validation",
}


def _positive_class_and_scores(model, y_true, X):
    """Return positive label and score vector for binary classifiers."""
    classes = list(getattr(model, "classes_", []))
    if len(classes) != 2:
        raise ValueError(f"Expected binary classifier with 2 classes, got {classes}")

    preferred = ["Yes", "yes", "Y", "y", "True", True, 1, "1", "Churn"]
    positive_label = next((label for label in preferred if label in classes), classes[1])
    positive_idx = classes.index(positive_label)
    y_prob = model.predict_proba(X)[:, positive_idx]
    y_binary = (pd.Series(y_true).to_numpy() == positive_label).astype(int)
    return positive_label, y_binary, y_prob


def evaluation_node(state: AgentState) -> AgentState:
    if state.trained_model is None:
        state.errors.append("Evaluation skipped — no trained model.")
        return state

    model = state.trained_model
    X_train = state.processed_data["X_train"]
    X_test = state.processed_data["X_test"]
    y_train = state.processed_data["y_train"]
    y_test = state.processed_data["y_test"]
    task = state.task.task_type

    metrics: dict[str, float] = {}

    if task == "binary_classification":
        y_pred = model.predict(X_test)
        positive_label, y_binary, y_prob = _positive_class_and_scores(model, y_test, X_test)
        metrics["accuracy"] = round(accuracy_score(y_test, y_pred), 4)
        metrics["balanced_accuracy"] = round(balanced_accuracy_score(y_test, y_pred), 4)
        metrics["f1_weighted"] = round(f1_score(y_test, y_pred, average="weighted"), 4)
        metrics["roc_auc"] = round(roc_auc_score(y_binary, y_prob), 4)
        metrics["average_precision"] = round(average_precision_score(y_binary, y_prob), 4)
        state.reasoning_log.append(
            f"EVALUATION: Binary positive label for ranking metrics = {positive_label!r}"
        )

    elif task == "multiclass_classification":
        y_pred = model.predict(X_test)
        metrics["accuracy"] = round(accuracy_score(y_test, y_pred), 4)
        metrics["balanced_accuracy"] = round(balanced_accuracy_score(y_test, y_pred), 4)
        metrics["f1_macro"] = round(f1_score(y_test, y_pred, average="macro"), 4)
        metrics["f1_weighted"] = round(f1_score(y_test, y_pred, average="weighted"), 4)

    elif task in ("regression", "time_series_forecasting"):
        y_pred = model.predict(X_test)
        metrics["r2"] = round(r2_score(y_test, y_pred), 4)
        metrics["rmse"] = round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4)
        metrics["mae"] = round(mean_absolute_error(y_test, y_pred), 4)
        if task == "time_series_forecasting":
            mask = y_test != 0
            if mask.any():
                mape = float(np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100)
                metrics["mape_pct"] = round(mape, 4)

    # Overfitting check
    train_score = float(model.score(X_train, y_train))
    val_score = float(model.score(X_test, y_test))

    # Feature importance
    feature_importance: dict[str, float] = {}
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
        if len(state.selected_features) == len(importance):
            feature_importance = {
                col: round(float(v), 6)
                for col, v in zip(state.selected_features, importance)
            }

    val_key = f"{task}_{state.profile.size_category}"
    state.experiment = ExperimentResult(
        model_name=state.chosen_model,
        metrics=metrics,
        baseline_metrics=state.baseline_metrics,
        validation_strategy=VALIDATION_STRATEGIES.get(val_key, "kfold_5"),
        train_score=round(train_score, 4),
        val_score=round(val_score, 4),
        feature_importance=feature_importance,
    )

    # Primary key defined for all supported task types
    if "classification" in task:
        primary_key = "f1_weighted"
        higher_is_better = True
    elif task == "time_series_forecasting":
        primary_key = "rmse"
        higher_is_better = False
    else:
        primary_key = "r2"
        higher_is_better = True

    baseline_val = state.baseline_metrics.get(primary_key, 0)
    model_val = metrics.get(primary_key, 0)
    improvement = (baseline_val - model_val) if not higher_is_better else (model_val - baseline_val)

    state.reasoning_log.append(
        f"EVALUATION: {state.chosen_model} | metrics={metrics} | "
        f"train={train_score:.4f} | val={val_score:.4f} | "
        f"improvement_vs_baseline={improvement:+.4f}"
    )
    return state
