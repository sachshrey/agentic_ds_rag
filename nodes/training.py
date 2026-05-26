# nodes/training.py
"""Pure computation. No LLM. No RAG. Just sklearn/XGBoost/LightGBM/CatBoost."""
from __future__ import annotations
import logging

import pandas as pd

from state import AgentState

logger = logging.getLogger(__name__)


def _build_model(model_name: str, state: AgentState):
    """Instantiate the chosen model with sensible defaults."""
    import xgboost as xgb
    import lightgbm as lgb
    from catboost import CatBoostClassifier, CatBoostRegressor
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LogisticRegression, Ridge

    task = state.task.task_type
    is_clf = "classification" in task
    X_train: pd.DataFrame = state.processed_data["X_train"]
    y_train = state.processed_data["y_train"]

    # Class weight / scale_pos_weight
    scale_pos_weight = 1.0
    if is_clf:
        counts = y_train.value_counts()
        if len(counts) == 2:
            scale_pos_weight = float(counts.max() / counts.min())

    if model_name == "xgboost":
        import numpy as np

        n_classes = len(np.unique(y_train)) if is_clf else 0
        if is_clf:
            eval_metric = "mlogloss" if n_classes > 2 else "logloss"
            return xgb.XGBClassifier(
                n_estimators=300, learning_rate=0.05, max_depth=6,
                random_state=42, eval_metric=eval_metric,
                early_stopping_rounds=20, verbosity=0,
                scale_pos_weight=scale_pos_weight,
            )
        return xgb.XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            random_state=42, eval_metric="rmse",
            early_stopping_rounds=20, verbosity=0,
        )

    elif model_name == "lightgbm":
        kw = dict(n_estimators=300, learning_rate=0.05, num_leaves=31,
                  random_state=42, verbose=-1)
        if is_clf:
            return lgb.LGBMClassifier(class_weight="balanced", **kw)
        return lgb.LGBMRegressor(**kw)

    elif model_name == "catboost":
        # CatBoost handles object columns natively — pass their indices
        cat_feature_indices = [
            i for i, col in enumerate(X_train.columns)
            if X_train[col].dtype == object
        ]
        kw = dict(
            iterations=300,
            learning_rate=0.05,
            depth=6,
            cat_features=cat_feature_indices,
            random_seed=42,
            verbose=False,
            early_stopping_rounds=30,
        )
        if is_clf:
            return CatBoostClassifier(auto_class_weights="Balanced", **kw)
        return CatBoostRegressor(**kw)

    elif model_name == "random_forest":
        kw = dict(n_estimators=200, random_state=42, n_jobs=-1)
        if is_clf:
            return RandomForestClassifier(class_weight="balanced", **kw)
        return RandomForestRegressor(**kw)

    elif model_name in ("logistic_regression", "ridge"):
        return (
            LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
            if is_clf else Ridge(alpha=1.0)
        )

    raise ValueError(f"Unknown model: {model_name}")


def training_node(state: AgentState) -> AgentState:
    X_train = state.processed_data["X_train"]
    X_test = state.processed_data["X_test"]
    y_train = state.processed_data["y_train"]
    y_test = state.processed_data["y_test"]

    try:
        model = _build_model(state.chosen_model, state)

        # Models with early stopping need eval_set
        if state.chosen_model in ("xgboost", "lightgbm"):
            model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        elif state.chosen_model == "catboost":
            from catboost import Pool

            eval_pool = Pool(X_test, y_test)
            model.fit(X_train, y_train, eval_set=eval_pool, use_best_model=True, verbose=False)
        else:
            model.fit(X_train, y_train)

        state.trained_model = model
        state.tool_calls_log.append(
            f"TRAINING: {state.chosen_model} — "
            f"{len(X_train)} train rows, {len(X_train.columns)} features"
        )

    except Exception as e:
        state.errors.append(f"Training failed: {state.chosen_model}: {e}")
        state.retry_count += 1
        logger.error(f"Training error: {e}")

    return state
