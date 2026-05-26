# nodes/preprocessing.py
"""
LLM decides strategy. Tools execute it.
Critical correctness: scaler fitted on X_train only, then applied to X_test.
SMOTE applied on X_train only, after split.

Changes vs prior version:
- preprocessing_steps cleared at entry (fixes duplicate log on retry)
- Numeric coercion before dtype check (fixes TotalCharges/object-dtype bug)
- Deterministic ID-column detection and drop (fixes customerID noise)
- retry_context built from critic_findings and passed to LLM prompt
- retry_count injected as cache-bust salt so LLM cache never returns stale response
"""
from __future__ import annotations
import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from state import AgentState
from schemas import PreprocessingDecision
from prompts import PREPROCESSING_PROMPT
from rag import retrieve_for_node
from utils.llm import get_llm

logger = logging.getLogger(__name__)

SAFE_DROP_REASONS = {
    "id_column",
    "near_constant",
    "duplicate",
    "leakage",
    "high_nulls",
}

DOMAIN_PROTECTED_COLUMNS = {
    "OnlineSecurity", "TechSupport", "StreamingTV", "StreamingMovies",
    "DeviceProtection", "OnlineBackup", "MonthlyCharges", "tenure",
    "Contract", "InternetService", "Partner", "Dependents",
    "PhoneService", "MultipleLines", "PaperlessBilling", "PaymentMethod",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _coerce_numeric_objects(df: pd.DataFrame, target_col: str | None) -> pd.DataFrame:
    """
    Coerce object columns that are actually numeric (e.g. TotalCharges stored
    as str due to empty strings in raw CSV).
    Threshold: ≥90% of non-null values must parse as float.
    """
    for col in df.select_dtypes(include=["object"]).columns:
        if col == target_col:
            continue
        converted = pd.to_numeric(df[col], errors="coerce")
        non_null = df[col].notna().sum()
        if non_null > 0 and converted.notna().sum() / non_null >= 0.90:
            df[col] = converted
            logger.info(f"PREPROCESSING: coerced '{col}' object→numeric")
    return df


def _drop_id_columns(
    df: pd.DataFrame,
    profile,
    task,
    state: AgentState,
) -> pd.DataFrame:
    """
    Deterministically drop near-unique ID columns before LLM sees them.
    Criteria (either is sufficient):
      1. Column name contains 'id', '_key', '_uuid', '_guid', '_code' (case-insensitive)
         AND cardinality > 50% of rows
      2. Cardinality ≥ 95% of rows (regardless of name)
    Never drops the target column.
    """
    id_keywords = {"id", "_key", "_uuid", "_guid", "_code"}
    rows = profile.rows

    for col in list(df.columns):
        if col == task.target_column:
            continue
        card = profile.cardinality.get(col, 0)
        name_lower = col.lower()
        name_is_id = any(kw in name_lower for kw in id_keywords)
        near_unique = card >= 0.95 * rows
        name_id_and_high = name_is_id and card > 0.50 * rows

        if near_unique or name_id_and_high:
            df = df.drop(columns=[col])
            state.preprocessing_steps.append(
                f"DROPPED: {col} — ID column "
                f"(cardinality={card}, rows={rows}, name_is_id={name_is_id})"
            )
            logger.info(f"PREPROCESSING: dropped ID column '{col}'")

    return df


def _execute_preprocessing(
    df: pd.DataFrame, decision: PreprocessingDecision, state: AgentState
) -> pd.DataFrame:
    """
    Apply imputation and encoding decisions.
    Returns the transformed DataFrame.
    Scaler and SMOTE are applied post-split — see caller.
    """
    from sklearn.impute import KNNImputer

    # Drop columns per LLM recommendation
    for col in decision.columns_to_drop:
        if col in df.columns and col != state.task.target_column:
            df = df.drop(columns=[col])
            state.preprocessing_steps.append(f"DROPPED: {col} — LLM flagged as redundant")

    # Imputation
    for col, strategy in decision.imputation_strategy.items():
        if col not in df.columns:
            continue
        if df[col].isnull().sum() == 0:
            continue

        if strategy == "drop":
            if col != state.task.target_column:
                df = df.drop(columns=[col])
                state.preprocessing_steps.append(f"DROPPED: {col} — imputation strategy=drop")
        elif strategy == "median":
            df[col] = df[col].fillna(df[col].median())
            state.preprocessing_steps.append(f"IMPUTED: {col} → median")
        elif strategy == "mean":
            df[col] = df[col].fillna(df[col].mean())
            state.preprocessing_steps.append(f"IMPUTED: {col} → mean")
        elif strategy == "mode":
            df[col] = df[col].fillna(df[col].mode().iloc[0])
            state.preprocessing_steps.append(f"IMPUTED: {col} → mode")
        elif strategy == "unknown":
            df[col] = df[col].fillna("Unknown")
            state.preprocessing_steps.append(f"IMPUTED: {col} → 'Unknown'")
        elif strategy == "knn":
            if pd.api.types.is_numeric_dtype(df[col]):
                imp = KNNImputer(n_neighbors=5)
                df[col] = imp.fit_transform(df[[col]])[:, 0]
                state.preprocessing_steps.append(f"IMPUTED: {col} → KNN(k=5)")

    # Encoding
    for col, strategy in decision.encoding_strategy.items():
        if col not in df.columns or col == state.task.target_column:
            continue

        if strategy == "onehot":
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True, dtype=np.int8)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
            state.preprocessing_steps.append(
                f"ENCODED: {col} → one-hot ({len(dummies.columns)} new cols)"
            )
        elif strategy == "frequency":
            freq_map = df[col].value_counts(normalize=True).to_dict()
            df[col] = df[col].map(freq_map).fillna(0.0).astype(np.float32)
            state.preprocessing_steps.append(f"ENCODED: {col} → frequency")
        elif strategy == "ordinal":
            df[col] = df[col].astype("category").cat.codes.astype(np.int16)
            state.preprocessing_steps.append(f"ENCODED: {col} → ordinal codes")
        # "catboost" strategy: leave as object — CatBoost handles it natively

    # Safety: fill remaining numeric NaNs with column median
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    return df


def _apply_scaling(
    X_train: pd.DataFrame, X_test: pd.DataFrame, strategy: str, state: AgentState
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit scaler on X_train only. Transform both. Store fitted scaler in state."""
    from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

    if strategy == "none":
        return X_train, X_test

    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return X_train, X_test

    scaler_map = {"standard": StandardScaler, "robust": RobustScaler, "minmax": MinMaxScaler}
    scaler = scaler_map.get(strategy, StandardScaler)()

    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    state.scaler = scaler
    state.preprocessing_steps.append(
        f"SCALED: {len(numeric_cols)} numeric cols → {strategy}"
    )
    return X_train, X_test


def _apply_smote(
    X_train: pd.DataFrame, y_train: pd.Series, state: AgentState
) -> tuple[pd.DataFrame, pd.Series]:
    """SMOTE applied on training data only. Never touch test set."""
    try:
        from imblearn.over_sampling import SMOTE
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X_train, y_train)
        X_res = pd.DataFrame(X_res, columns=X_train.columns)
        y_res = pd.Series(y_res, name=y_train.name)
        dist = pd.Series(y_res).value_counts().to_dict()
        state.preprocessing_steps.append(f"SMOTE: Training set rebalanced → {dist}")
        return X_res, y_res
    except Exception as e:
        state.errors.append(f"SMOTE failed: {e}. Proceeding without.")
        return X_train, y_train


def _decide_preprocessing_rules(
    df: pd.DataFrame,
    profile,
    task,
    llm_suggested_drops: list[str],
) -> list[str]:
    """Validate LLM column-drop suggestions — only safe deterministic drops allowed."""
    columns_to_drop: list[str] = []
    n_rows = profile.rows

    for col in llm_suggested_drops:
        if col not in df.columns or col == task.target_column:
            continue
        card = profile.cardinality.get(col, 0)

        # Protected column guard — never drop unless leakage or >80% nulls
        if col in DOMAIN_PROTECTED_COLUMNS:
            null_pct_val = profile.null_percentages.get(col, 0.0)
            if null_pct_val < 80.0:
                continue  # Skip drop — protected column

        # Reject vague LLM reasons — only drop for deterministic reasons
        drop_reason = None
        if card <= 1:
            drop_reason = "near_constant"
        elif (card / max(n_rows, 1) > 0.95 and
              any(kw in col.lower() for kw in ["id", "_id", "uuid", "index", "key"])):
            drop_reason = "id_column"

        if drop_reason not in SAFE_DROP_REASONS:
            continue  # Not a confirmed safe drop reason — keep column

        columns_to_drop.append(col)

    return columns_to_drop


def _rule_based_preprocessing_decision(
    df: pd.DataFrame,
    profile,
    task,
) -> "PreprocessingDecision":
    """
    Pure deterministic rules — no LLM, fully reproducible.
    LLM is retained only for columns_to_drop (genuinely ambiguous).
    Note: numeric coercion and ID-column drops run BEFORE this function,
    so df.dtypes reflect the true column types here.
    """
    from utils.config import cfg
    imputation: dict[str, str] = {}
    encoding: dict[str, str] = {}

    for col in df.columns:
        if col == task.target_column:
            continue

        null_pct = profile.null_percentages.get(col, 0.0)
        card = profile.cardinality.get(col, 0)
        is_num = pd.api.types.is_numeric_dtype(df[col])

        # Imputation rules
        if null_pct == 0:
            pass
        elif null_pct < 5 and is_num:
            imputation[col] = "median"
        elif null_pct < 5 and not is_num:
            imputation[col] = "mode"
        elif null_pct < 20 and is_num:
            imputation[col] = "knn"
        elif null_pct < 20 and not is_num:
            imputation[col] = "unknown"
        else:
            imputation[col] = "median" if is_num else "unknown"

        # Encoding rules — only for non-numeric columns
        if not is_num and col in df.columns:
            if card <= 10:
                encoding[col] = "onehot"
            elif card <= 50:
                encoding[col] = "frequency"
            else:
                encoding[col] = "frequency"

    # Scaling rule (tree models don't need scaling)
    if "classification" in task.task_type or task.task_type == "time_series_forecasting":
        scaling = "none"
    else:
        scaling = "standard"

    return PreprocessingDecision(
        imputation_strategy=imputation,
        encoding_strategy=encoding,
        scaling_strategy=scaling,
        imbalance_strategy="none",
        columns_to_drop=[],
        reasoning="Rule-based preprocessing decision from null rates, cardinality, and task type.",
    )


def _build_retry_context(state: AgentState) -> str:
    """
    Build a concise retry context string from critic findings.
    Injected into the LLM prompt so it knows WHY we are retrying
    and can make different column-drop decisions.
    Includes retry_count as a salt so the LLM cache is busted between retries.
    """
    if not state.critic_findings or state.retry_count == 0:
        return f"[Attempt {state.retry_count + 1}] No prior critic findings."

    lines = [f"[Attempt {state.retry_count + 1}] CRITIC FLAGGED THE FOLLOWING ISSUES — fix them now:"]
    for f in state.critic_findings:
        lines.append(f"  - [{f.severity.upper()}] {f.issue_type}: {f.detail}")
        lines.append(f"    Suggested fix: {f.fix}")
    lines.append(
        "ACTION REQUIRED: be more aggressive about dropping suspicious columns. "
        "If any column looks like it could encode the target, drop it."
    )
    return "\n".join(lines)


# ── Main node ─────────────────────────────────────────────────────────────────

def preprocessing_node(state: AgentState) -> AgentState:
    df: pd.DataFrame = state.raw_data.copy()

    # FIX 1: Clear preprocessing_steps at entry so retries don't duplicate logs.
    state.preprocessing_steps = []

    # Drop confirmed leakage columns (from task detection)
    for col in state.dropped_columns:
        if col in df.columns:
            df = df.drop(columns=[col])
            state.preprocessing_steps.append(f"DROPPED: {col} — confirmed target leakage")

    # Drop high-null columns (>70%) — deterministic, no LLM
    for col, null_pct in state.profile.null_percentages.items():
        if null_pct > 70 and col in df.columns and col != state.task.target_column:
            df = df.drop(columns=[col])
            state.preprocessing_steps.append(
                f"DROPPED: {col} — {null_pct:.1f}% null (threshold: 70%)"
            )

    # FIX 2: Coerce object columns that are actually numeric (e.g. TotalCharges).
    # Must run BEFORE rule-based decision so dtypes are correct.
    df = _coerce_numeric_objects(df, state.task.target_column)

    # FIX 3: Deterministically drop ID columns before LLM sees them.
    df = _drop_id_columns(df, state.profile, state.task, state)

    # Rule-based decision for imputation + encoding (no LLM — deterministic and reproducible)
    decision = _rule_based_preprocessing_decision(df, state.profile, state.task)

    # Imbalance strategy: rule-based
    eda_stats = state.processed_data.get("eda_stats", {})
    imbalance_ratio = eda_stats.get("imbalance_ratio", 1.0)
    if "classification" in state.task.task_type and imbalance_ratio > 3.0:
        decision = decision.model_copy(update={"imbalance_strategy": "smote"})

    # LLM retained ONLY for columns_to_drop (genuinely ambiguous redundancy detection)
    rag_docs = retrieve_for_node(
        node_name="preprocessing",
        task_type=state.task.task_type,
        domain=state.task.domain_detected,
        query=f"redundant features to drop for {state.task.task_type}",
        k=2,  # reduced from 4 — preprocessing RAG is for drop hints only
    )
    state.rag_retrievals.append(f"PREPROCESSING: {len(rag_docs)} strategy docs")

    # FIX 4: Pass retry context so LLM knows about prior critic findings.
    # retry_count is embedded in the string so each retry produces a distinct
    # prompt, busting the LLM cache automatically.
    retry_context = _build_retry_context(state)

    llm = get_llm("preprocessing").with_structured_output(PreprocessingDecision)
    llm_drop_decision: PreprocessingDecision = llm.invoke(
        PREPROCESSING_PROMPT.format(
            profile=state.profile.model_dump(),
            task=state.task.model_dump(),
            remaining_columns=list(df.columns),
            rag_context="\n".join(rag_docs) or "No RAG context.",
            retry_context=retry_context,
        )
    )
    # Merge: rule-based for imputation/encoding; LLM drops filtered for safety
    safe_drops = _decide_preprocessing_rules(
        df, state.profile, state.task, llm_drop_decision.columns_to_drop
    )
    decision = decision.model_copy(update={"columns_to_drop": safe_drops})

    state.reasoning_log.append(
        f"PREPROCESSING: rule-based imputation={list(decision.imputation_strategy)} | "
        f"encoding={list(decision.encoding_strategy)} | "
        f"scaling={decision.scaling_strategy} | "
        f"LLM-drops={decision.columns_to_drop}"
    )

    # Tools execute
    df = _execute_preprocessing(df, decision, state)

    # Train / test split
    target = state.task.target_column
    if target not in df.columns:
        state.errors.append(f"Target column '{target}' not found after preprocessing.")
        return state

    X = df.drop(columns=[target])
    y = df[target]

    if state.task.task_type == "time_series_forecasting":
        # Chronological split — NEVER random for time series
        split_idx = int(len(df) * 0.8)
        X_train, X_test = X.iloc[:split_idx].copy(), X.iloc[split_idx:].copy()
        y_train, y_test = y.iloc[:split_idx].copy(), y.iloc[split_idx:].copy()
        state.preprocessing_steps.append("SPLIT: Chronological 80/20 (time-series)")
    else:
        stratify = (
            y if state.task.task_type in ("binary_classification", "multiclass_classification")
            else None
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify
        )
        state.preprocessing_steps.append("SPLIT: Stratified random 80/20")

    # Scale (fit on X_train, apply to both)
    X_train, X_test = _apply_scaling(X_train, X_test, decision.scaling_strategy, state)

    # SMOTE (training set only, after split)
    if decision.imbalance_strategy == "smote" and "classification" in state.task.task_type:
        X_train, y_train = _apply_smote(X_train, y_train, state)

    state.processed_data.update({
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "imbalance_strategy": decision.imbalance_strategy,
    })
    state.selected_features = list(X_train.columns)
    return state
