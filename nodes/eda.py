# nodes/eda.py
from __future__ import annotations
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — server safe
import matplotlib.pyplot as plt
import seaborn as sns

from state import AgentState
from schemas import EDAInterpretation
from prompts import EDA_INTERPRETATION_PROMPT
from utils.llm import get_llm
from utils.config import cfg

logger = logging.getLogger(__name__)


def _generate_eda_plots(df: pd.DataFrame, target: str | None, output_dir: str, task_type: str) -> None:
    """All plot generation. Pure tool code — no LLM."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Target distribution
    if target and target in df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        if "classification" in task_type:
            df[target].value_counts().plot(kind="bar", ax=ax, color="steelblue")
            ax.set_xlabel("Class")
        else:
            df[target].hist(bins=50, ax=ax, color="steelblue")
        ax.set_title(f"Target Distribution: {target}")
        ax.set_ylabel("Count")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/target_distribution.png", dpi=150)
        plt.close()

    # Correlation heatmap
    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) > 1:
        fig, ax = plt.subplots(figsize=(min(14, len(numeric_df.columns) + 2), 10))
        annotate = len(numeric_df.columns) <= 15
        sns.heatmap(
            numeric_df.corr(), annot=annotate, fmt=".2f",
            cmap="coolwarm", center=0, ax=ax,
        )
        ax.set_title("Feature Correlation Matrix")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/correlation_heatmap.png", dpi=150)
        plt.close()

    # Missing value pattern
    if df.isnull().any().any():
        null_cols = df.columns[df.isnull().any()]
        fig, ax = plt.subplots(figsize=(max(10, len(null_cols)), 6))
        sns.heatmap(df[null_cols].isnull(), yticklabels=False, cbar=False, ax=ax, cmap="viridis")
        ax.set_title("Missing Value Pattern (yellow = missing)")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/missing_values.png", dpi=150)
        plt.close()


def eda_node(state: AgentState) -> AgentState:
    df: pd.DataFrame = state.raw_data
    target = state.task.target_column

    # ── Sample if large ───────────────────────────────────────────────────
    if state.eda_mode == "sampled":
        df_eda = df.sample(n=min(cfg.data.eda_sample_rows, len(df)), random_state=42)
        state.reasoning_log.append(
            f"EDA: Sampled 10,000 rows from {state.profile.rows:,} for speed."
        )
    else:
        df_eda = df.copy()

    # ── TOOLS compute all statistics ──────────────────────────────────────
    eda_stats: dict = {}

    # Class balance
    if target and target in df_eda.columns and "classification" in state.task.task_type:
        counts = df_eda[target].value_counts(normalize=True)
        eda_stats["class_distribution"] = {str(k): round(float(v), 4) for k, v in counts.items()}
        if len(counts) >= 2:
            majority = float(counts.max())
            minority = float(counts.min())
            majority_to_minority = majority / minority
            minority_to_majority = minority / majority
            if minority_to_majority >= 0.8:
                balance_description = "well balanced"
            elif minority_to_majority >= 0.4:
                balance_description = "moderately imbalanced"
            else:
                balance_description = "highly imbalanced"
            eda_stats["imbalance_ratio"] = round(majority_to_minority, 2)
            eda_stats["class_balance_ratio"] = round(minority_to_majority, 2)
            eda_stats["class_balance_summary"] = (
                "Class balance ratio (minority/majority): "
                f"{minority_to_majority:.2f}. Dataset is {balance_description}. "
                "The majority/minority imbalance ratio used for SMOTE checks is "
                f"{majority_to_minority:.2f}:1."
            )

    # Missing value summary
    null_pcts = state.profile.null_percentages
    eda_stats["missing_value_summary"] = {
        "total_cols_with_nulls": sum(1 for p in null_pcts.values() if p > 0),
        "severe_null_cols": {c: p for c, p in null_pcts.items() if p > 30},
        "moderate_null_cols": {c: p for c, p in null_pcts.items() if 5 < p <= 30},
    }

    # Outlier detection (IQR method)
    outlier_cols: dict[str, int] = {}
    for col in df_eda.select_dtypes(include=[np.number]).columns:
        Q1, Q3 = df_eda[col].quantile(0.25), df_eda[col].quantile(0.75)
        IQR = Q3 - Q1
        n_outliers = int(((df_eda[col] < Q1 - 1.5 * IQR) | (df_eda[col] > Q3 + 1.5 * IQR)).sum())
        if n_outliers > 0:
            outlier_cols[col] = n_outliers
    eda_stats["outlier_summary"] = outlier_cols

    # Highly skewed features
    eda_stats["highly_skewed"] = {
        col: skew for col, skew in state.profile.skewness.items() if abs(skew) > 2.0
    }

    eda_stats["high_correlation_pairs"] = state.profile.correlations_high[:10]

    # ── LLM INTERPRETS — does not compute ────────────────────────────────
    llm = get_llm("eda").with_structured_output(EDAInterpretation)

    interpretation: EDAInterpretation = llm.invoke(
        EDA_INTERPRETATION_PROMPT.format(
            eda_stats=eda_stats,
            task_type=state.task.task_type,
            target_column=target,
            profile_summary={
                "rows": state.profile.rows,
                "columns": state.profile.columns,
                "size_category": state.profile.size_category,
                "domain": state.task.domain_detected,
            },
        )
    )

    # ── Generate plots ────────────────────────────────────────────────────
    _generate_eda_plots(df_eda, target, state.output_dir, state.task.task_type)

    state.reasoning_log.append(
        f"EDA: imbalance_ratio={eda_stats.get('imbalance_ratio', 'N/A')} | "
        f"severe_null_cols={list(eda_stats['missing_value_summary']['severe_null_cols'])} | "
        f"outlier_cols={len(outlier_cols)} | "
        f"primary_insight={interpretation.primary_insight}"
    )

    # Store for preprocessing node context
    state.processed_data["eda_stats"] = eda_stats
    state.processed_data["eda_interpretation"] = interpretation.model_dump()

    return state
