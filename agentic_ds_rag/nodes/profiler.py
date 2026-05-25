# nodes/profiler.py
"""
Deterministic profiling. Zero LLM calls. Zero tokens spent.
Gives downstream reasoning nodes exactly what they need in structured form.
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from ydata_profiling import ProfileReport

from state import AgentState, DatasetProfile
from utils.config import cfg
from utils.caching import get_cached_profile, save_profile_cache

logger = logging.getLogger(__name__)


def profiler_node(state: AgentState) -> AgentState:
    from pathlib import Path

    Path(state.output_dir).mkdir(parents=True, exist_ok=True)

    # ── Load dataset ─────────────────────────────────────────────────────
    ext = state.file_path.rsplit(".", 1)[-1].lower()
    loaders = {
        "csv": pd.read_csv,
        "parquet": pd.read_parquet,
        "xls": lambda p: pd.read_excel(p, engine="xlrd"),
        "xlsx": lambda p: pd.read_excel(p, engine="openpyxl"),
    }
    if ext not in loaders:
        raise ValueError(f"Unsupported file type: .{ext}  (supported: csv, parquet, xls, xlsx)")

    df: pd.DataFrame = loaders[ext](state.file_path)
    state.raw_data = df

    from utils.caching import get_file_hash
    state.dataset_hash = get_file_hash(state.file_path)
    state.reasoning_log.append(f"PROFILER: dataset_hash={state.dataset_hash[:12]}...")

    # ── Check profile cache ───────────────────────────────────────────────
    cached = get_cached_profile(state.file_path)
    if cached:
        state.profile = cached
        minimal_mode = cached.size_category == "large"
        report_path = f"{state.output_dir}/profile_report.html"
        if not Path(report_path).exists():
            report = ProfileReport(
                df, minimal=minimal_mode, title="Dataset Profile", progress_bar=False
            )
            report.to_file(report_path)
        state.profile_report_path = report_path
        state.reasoning_log.append(
            f"PROFILER: Loaded from cache — {cached.rows} rows, {cached.columns} cols"
        )
        return state

    # ── High-correlation pairs ────────────────────────────────────────────
    numeric_df = df.select_dtypes(include=[np.number])
    high_corr: list[tuple[str, str, float]] = []
    if len(numeric_df.columns) > 1:
        corr = numeric_df.corr().abs()
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                val = corr.iloc[i, j]
                if val > cfg.data.correlation_drop_threshold:
                    high_corr.append((corr.columns[i], corr.columns[j], round(float(val), 3)))

    # ── Size category ─────────────────────────────────────────────────────
    rows = len(df)
    size_cat = "small" if rows < cfg.data.eda_sample_rows else ("medium" if rows < 100_000 else "large")

    # ── Build profile ─────────────────────────────────────────────────────
    profile = DatasetProfile(
        rows=rows,
        columns=len(df.columns),
        dtypes={col: str(df[col].dtype) for col in df.columns},
        null_percentages={
            col: round(df[col].isnull().mean() * 100, 2) for col in df.columns
        },
        cardinality={col: int(df[col].nunique()) for col in df.columns},
        skewness={
            col: round(float(df[col].skew()), 3)
            for col in df.columns
            if pd.api.types.is_numeric_dtype(df[col])
        },
        correlations_high=high_corr,
        memory_usage_mb=round(df.memory_usage(deep=True).sum() / 1e6, 2),
        duplicate_rows=int(df.duplicated().sum()),
        size_category=size_cat,
    )
    state.profile = profile
    save_profile_cache(state.file_path, profile)

    # ── ydata-profiling HTML (for human viewer only — never sent to LLM) ──
    minimal_mode = size_cat == "large"
    report = ProfileReport(df, minimal=minimal_mode, title="Dataset Profile", progress_bar=False)
    report_path = f"{state.output_dir}/profile_report.html"
    report.to_file(report_path)
    state.profile_report_path = report_path

    avg_null = (
        sum(profile.null_percentages.values()) / len(df.columns)
        if df.columns.any() else 0
    )
    state.reasoning_log.append(
        f"PROFILER: {rows:,} rows | {len(df.columns)} cols | "
        f"size={size_cat} | avg_null={avg_null:.1f}% | "
        f"high_corr_pairs={len(high_corr)} | duplicates={profile.duplicate_rows}"
    )
    return state
