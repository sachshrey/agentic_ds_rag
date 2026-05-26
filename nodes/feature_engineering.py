# nodes/feature_engineering.py
from __future__ import annotations
import logging

import numpy as np
import pandas as pd

from state import AgentState
from schemas import FeatureEngineeringPlan, FeatureDef
from prompts import FEATURE_ENG_PROMPT
from rag import retrieve_for_node
from utils.llm import get_llm

logger = logging.getLogger(__name__)


def _create_feature(
    X_train: pd.DataFrame, X_test: pd.DataFrame, feat: FeatureDef
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply a single feature definition to both train and test sets."""
    cols = feat.source_columns

    # Validate columns exist
    missing = [c for c in cols if c not in X_train.columns]
    if missing:
        raise ValueError(f"Source columns not found: {missing}")

    def apply_op(df: pd.DataFrame) -> pd.Series:
        if feat.operation == "ratio":
            return df[cols[0]] / (df[cols[1]].replace(0, np.nan) + 1e-9)
        elif feat.operation == "product":
            return df[cols[0]] * df[cols[1]]
        elif feat.operation == "difference":
            return df[cols[0]] - df[cols[1]]
        elif feat.operation == "log":
            return np.log1p(df[cols[0]].clip(lower=0))
        elif feat.operation == "bin":
            bins = feat.params.get("bins", 5)
            return pd.qcut(df[cols[0]], q=bins, labels=False, duplicates="drop")
        elif feat.operation == "interaction":
            combo = df[cols[0]].astype(str) + "_" + df[cols[1]].astype(str)
            freq_map = X_train[cols[0]].astype(str).add("_").add(
                X_train[cols[1]].astype(str)
            ).value_counts(normalize=True).to_dict()
            return combo.map(freq_map).fillna(0.0)
        else:
            raise ValueError(f"Unknown operation: {feat.operation}")

    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[feat.name] = apply_op(X_train)
    X_test[feat.name] = apply_op(X_test)
    return X_train, X_test


def feature_engineering_node(state: AgentState) -> AgentState:
    X_train: pd.DataFrame = state.processed_data["X_train"].copy()
    X_test: pd.DataFrame = state.processed_data["X_test"].copy()

    # RAG: domain-specific feature patterns
    rag_docs = retrieve_for_node(
        node_name="feature_engineering",
        task_type=state.task.task_type,
        domain=state.task.domain_detected,
        query=(
            f"feature engineering ideas for {state.task.domain_detected or 'tabular'} "
            f"{state.task.task_type} dataset"
        ),
    )
    state.rag_retrievals.append(f"FEATURE_ENG: {len(rag_docs)} domain pattern docs")

    # LLM proposes features
    # Build validated column lists to prevent LLM hallucinating column names
    valid_numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    valid_all_cols = X_train.columns.tolist()

    llm = get_llm("feature_engineering").with_structured_output(FeatureEngineeringPlan)
    plan: FeatureEngineeringPlan = llm.invoke(
        FEATURE_ENG_PROMPT.format(
            task=state.task.model_dump(),
            available_columns=valid_all_cols,
            numeric_columns=valid_numeric_cols,
            high_correlations=state.profile.correlations_high,
            rag_context="\n".join(rag_docs) or "No RAG context available.",
        )
    )

    valid_col_set = set(valid_all_cols)
    validated_features = []
    for feat in plan.features_to_create:
        bad_cols = [c for c in feat.source_columns if c not in valid_col_set]
        if bad_cols:
            state.errors.append(
                f"FEATURE ENG: Skipping '{feat.name}' — hallucinated column(s): {bad_cols}"
            )
            continue
        if feat.operation == "interaction":
            state.errors.append(
                f"FEATURE ENG: Skipping '{feat.name}' — 'interaction' op is disabled."
            )
            continue
        validated_features.append(feat)
    plan = plan.model_copy(update={"features_to_create": validated_features})

    # Drop redundant features first
    for col in plan.features_to_drop:
        if col in X_train.columns and col != state.task.target_column:
            X_train = X_train.drop(columns=[col])
            X_test = X_test.drop(columns=[col])
            state.preprocessing_steps.append(f"DROPPED: {col} — LLM flagged as redundant feature")

    # Create new features (tools execute, not LLM)
    created = 0
    for feat_def in plan.features_to_create:
        try:
            X_train, X_test = _create_feature(X_train, X_test, feat_def)
            state.preprocessing_steps.append(
                f"FEATURE CREATED: {feat_def.name} "
                f"({feat_def.operation} of {feat_def.source_columns}) — {feat_def.reasoning}"
            )
            created += 1
        except Exception as e:
            state.errors.append(f"Feature creation failed: {feat_def.name}: {e}")

    # Time-series lag features — always add for TS tasks
    if state.task.task_type == "time_series_forecasting" and state.task.target_column:
        target_col = state.task.target_column
        if target_col in X_train.columns:
            # Concatenate train+test so test boundary lags look back into training history
            combined = pd.concat([X_train, X_test], axis=0)
            n_train = len(X_train)

            for lag in [1, 7, 14, 30]:
                col_name = f"{target_col}_lag_{lag}"
                combined[col_name] = combined[target_col].shift(lag)
                state.preprocessing_steps.append(f"LAG FEATURE: {col_name} (lag={lag})")

            X_train = combined.iloc[:n_train].copy()
            X_test = combined.iloc[n_train:].copy()

            # Drop NaN rows created by the largest lag from training set only
            max_lag = 30
            rows_before = len(X_train)
            y_train = state.processed_data["y_train"]
            X_train = X_train.iloc[max_lag:].reset_index(drop=True)
            y_train = y_train.iloc[max_lag:].reset_index(drop=True)
            state.processed_data["y_train"] = y_train
            state.preprocessing_steps.append(
                f"LAG NaN DROP: removed first {max_lag} train rows "
                f"({rows_before} → {len(X_train)} rows)"
            )

    state.processed_data["X_train"] = X_train
    state.processed_data["X_test"] = X_test
    state.selected_features = list(X_train.columns)

    state.reasoning_log.append(
        f"FEATURE ENG: Created {created} features | "
        f"Dropped {len(plan.features_to_drop)} redundant | "
        f"Final feature count: {len(state.selected_features)}"
    )
    return state
