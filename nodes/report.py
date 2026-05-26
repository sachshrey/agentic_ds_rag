# nodes/report.py
from __future__ import annotations
import json
import logging
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

from state import AgentState
from rag import add_experiment_to_rag
from utils.config import cfg

logger = logging.getLogger(__name__)


def _render_shap_table(state: AgentState) -> str:
    if not state.shap_importance_table:
        return "_No SHAP importance table available._"

    rows = "\n".join(
        f"| {i + 1} | `{row['feature']}` | {row['mean_abs_shap']:.4f} |"
        for i, row in enumerate(state.shap_importance_table)
    )
    return (
        "| Rank | Feature | Mean \\|SHAP\\| |\n"
        "|---:|---|---:|\n"
        f"{rows}"
    )


def _render_output_files(state: AgentState) -> str:
    files = [
        ("profile_report.html", "Full ydata-profiling interactive report"),
        ("target_distribution.png", "Target variable distribution"),
        ("correlation_heatmap.png", "Feature correlation matrix"),
        ("missing_values.png", "Missing value pattern heatmap"),
        ("shap_summary.png", "SHAP beeswarm importance plot"),
        ("shap_waterfall.png", "Individual prediction waterfall"),
        ("final_report.md", "This report"),
    ]
    rows = []
    for fname, description in files:
        if (Path(state.output_dir) / fname).exists() or fname == "final_report.md":
            rows.append(f"| `{fname}` | {description} |")
    return "\n".join(rows)


def is_effectively_retained(raw_col: str, final_features: list[str]) -> bool:
    # direct retention
    if raw_col in final_features:
        return True
    # one-hot encoded retention
    prefix = raw_col + "_"
    return any(f.startswith(prefix) for f in final_features)


def _log_to_mlflow(state: AgentState) -> str:
    mlflow.set_tracking_uri(str(Path(cfg.paths.mlflow_tracking_uri).resolve()))
    mlflow.set_experiment("agentic-ds-rag")

    with mlflow.start_run(
        run_name=f"{state.task.task_type}_{state.chosen_model}"
    ) as run:
        # Params
        mlflow.log_params({
            "task_type": state.task.task_type,
            "dataset_category": state.task.dataset_category,
            "target_column": state.task.target_column,
            "chosen_model": state.chosen_model,
            "domain_detected": state.task.domain_detected,
            "dataset_rows": state.profile.rows,
            "dataset_cols": state.profile.columns,
            "size_category": state.profile.size_category,
            "retry_count": state.retry_count,
            "leakage_suspects_dropped": str(state.task.leakage_suspects),
            "preprocessing_steps_count": len(state.preprocessing_steps),
        })

        # Log dataset hash for reproducibility tracing
        if state.dataset_hash:
            mlflow.log_param("dataset_hash", state.dataset_hash)

        # Log node timing breakdown
        if state.node_timings:
            mlflow.log_metrics({
                f"time_{k}": v for k, v in state.node_timings.items()
            })
            mlflow.log_metric("time_total_pipeline_s", round(sum(state.node_timings.values()), 2))

        # Metrics
        mlflow.log_metrics(state.experiment.metrics)
        mlflow.log_metrics({f"baseline_{k}": v for k, v in state.baseline_metrics.items()})

        # Text artifacts
        mlflow.log_text("\n\n".join(state.reasoning_log), "reasoning_audit_trail.txt")
        mlflow.log_text("\n\n".join(state.rag_retrievals), "rag_retrievals.txt")
        mlflow.log_text("\n".join(state.preprocessing_steps), "preprocessing_steps.txt")
        mlflow.log_text(
            json.dumps([f.model_dump() for f in state.critic_findings], indent=2),
            "critic_findings.json",
        )
        mlflow.log_text(
            json.dumps(state.experiment.feature_importance, indent=2),
            "feature_importance.json",
        )
        mlflow.log_text(
            json.dumps(state.shap_importance_table, indent=2),
            "shap_importance_table.json",
        )

        # File artifacts
        for fname in [
            "shap_summary.png", "shap_waterfall.png",
            "target_distribution.png", "correlation_heatmap.png",
            "missing_values.png", "profile_report.html", "final_report.md",
        ]:
            fpath = Path(state.output_dir) / fname
            if fpath.exists():
                mlflow.log_artifact(str(fpath))

        # Trained model
        if state.trained_model is not None:
            try:
                mlflow.sklearn.log_model(
                    state.trained_model,
                    artifact_path="model",
                    registered_model_name=f"{state.task.task_type}_{state.chosen_model}",
                )
            except Exception as e:
                logger.warning(f"MLflow model logging failed: {e}")

        return run.info.run_id


def report_node(state: AgentState) -> AgentState:
    run_id = _log_to_mlflow(state)
    state.experiment.mlflow_run_id = run_id

    # Store experiment in RAG for future retrieval
    add_experiment_to_rag(
        task_type=state.task.task_type,
        domain=state.task.domain_detected,
        model=state.chosen_model,
        metrics=state.experiment.metrics,
        key_decisions=state.reasoning_log,
        run_id=run_id,
    )

    # ── Determine primary metric ──────────────────────────────────────────
    is_clf = "classification" in state.task.task_type
    if is_clf:
        primary_key = "f1_weighted"
        primary_label = "F1-Weighted"
        higher_is_better = True
    elif state.task.task_type == "time_series_forecasting":
        primary_key = "rmse"
        primary_label = "RMSE"
        higher_is_better = False
    else:
        primary_key = "r2"
        primary_label = "R²"
        higher_is_better = True
    primary_val = state.experiment.metrics.get(primary_key, 0)
    baseline_val = state.baseline_metrics.get(primary_key, 0)
    improvement = (primary_val - baseline_val) if higher_is_better else (baseline_val - primary_val)

    # ── Format sections ───────────────────────────────────────────────────
    avg_null = (
        sum(state.profile.null_percentages.values()) / max(state.profile.columns, 1)
    )

    preprocessing_str = "\n".join(
        f"  {i + 1}. {step}" for i, step in enumerate(state.preprocessing_steps)
    )

    candidates_str = "\n".join(
        f"  - **{c.name}** (score {c.heuristic_score}): {' | '.join(c.reasoning_points)}"
        for c in state.model_candidates
    )

    metrics_comparison = "\n".join(
        f"| {k} | {state.baseline_metrics.get(k, 'N/A')} | **{v}** | {v - state.baseline_metrics.get(k, 0):+.4f} |"
        for k, v in state.experiment.metrics.items()
    )

    top_features_str = _render_shap_table(state)
    output_files_str = _render_output_files(state)

    critic_str = ""
    if state.critic_findings:
        critic_str = "\n## Issues Detected & Resolved\n\n"
        for f in state.critic_findings:
            critic_str += (
                f"- **{f.issue_type}** ({f.severity}): {f.detail}\n"
                f"  Fix applied: `{f.fix}`\n\n"
            )

    leakage_str = (
        "\n".join(f"  - `{col}` — flagged as possible leakage" for col in state.task.leakage_suspects)
        if state.task.leakage_suspects
        else "  None detected."
    )

    reasoning_str = "\n\n".join(state.reasoning_log)
    shap_narration = state.processed_data.get("shap_narration", "_SHAP explainer did not run._")
    actionable_insights = state.processed_data.get("shap_actionable_insights", [])

    # Feature summary stats
    original_feature_count = state.profile.columns - 1  # minus target
    dropped_count = len(state.dropped_columns) + len([
        s for s in state.preprocessing_steps if s.startswith("DROPPED:")
    ])
    encoded_count = len([s for s in state.preprocessing_steps if s.startswith("ENCODED:")])
    final_feature_count = len(state.selected_features)

    # Protected columns status
    from nodes.preprocessing import DOMAIN_PROTECTED_COLUMNS
    all_cols = set(state.profile.dtypes.keys())
    final_features = list(state.selected_features)
    protected_retained = [
        c for c in DOMAIN_PROTECTED_COLUMNS
        if c in all_cols and is_effectively_retained(c, final_features)
    ]
    protected_dropped = [
        c for c in DOMAIN_PROTECTED_COLUMNS
        if c in all_cols and not is_effectively_retained(c, final_features)
    ]

    # Runtime summary
    total_time = sum(state.node_timings.values()) if state.node_timings else 0
    openai_nodes = {"task_detector", "critic", "explainer", "report"}
    ollama_nodes = {"eda", "preprocessing", "feature_engineering", "model_selection"}
    openai_calls = len([n for n in state.node_timings if n in openai_nodes])
    ollama_calls = len([n for n in state.node_timings if n in ollama_nodes])
    avg_latency = round(total_time / max(len(state.node_timings), 1), 2)

    # ── Build report ──────────────────────────────────────────────────────
    report = f"""# Data Science Analysis Report

**System:** Agentic Data Science RAG System  
**MLflow Run ID:** `{run_id}`  
**Dataset:** `{Path(state.file_path).name}`  
**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

---

## 1. Dataset Summary

| Property | Value |
|---|---|
| Rows | {state.profile.rows:,} |
| Columns | {state.profile.columns} |
| Memory Usage | {state.profile.memory_usage_mb} MB |
| Dataset Size | {state.profile.size_category} |
| Avg Null Rate | {avg_null:.1f}% |
| Duplicate Rows | {state.profile.duplicate_rows} |
| High-Corr Pairs | {len(state.profile.correlations_high)} |
| Domain Detected | {state.task.domain_detected or 'General'} |

### Target Leakage Check
{leakage_str}

---

## 2. Task Detection

| Property | Value |
|---|---|
| Problem Type | **{state.task.task_type}** |
| Target Column | `{state.task.target_column}` |
| Confidence | {state.task.confidence:.0%} |
| Dataset Category | {state.task.dataset_category} |
| Recommended Metrics | {', '.join(state.task.recommended_metrics)} |
| Validation Strategy | {state.experiment.validation_strategy} |

**Reasoning:** {state.task.reasoning}

---

## 3. Preprocessing Audit Trail

{preprocessing_str}

---

## 4. Model Selection

**Candidates evaluated:**
{candidates_str}

**Chosen model:** `{state.chosen_model}`

---

## 5. Results

### Performance vs Baseline

| Metric | Baseline (LogReg/Ridge) | {state.chosen_model} | Δ |
|---|---|---|---|
{metrics_comparison}

**Primary metric ({primary_label}):** {primary_val:.4f}  
**Improvement over baseline:** {improvement:+.4f}

### Overfitting Check
| Score | Value |
|---|---|
| Train | {state.experiment.train_score:.4f} |
| Validation | {state.experiment.val_score:.4f} |
| Gap | {state.experiment.train_score - state.experiment.val_score:.4f} |

---

## 5b. Feature Summary

| Metric | Count |
|---|---|
| Original features (excl. target) | {original_feature_count} |
| Features dropped | {dropped_count} |
| Features encoded | {encoded_count} |
| Final feature count | {final_feature_count} |

### Protected Domain Columns
**Retained:** {', '.join(f'`{c}`' for c in protected_retained) if protected_retained else 'None'}

**Dropped (investigate):** {', '.join(f'`{c}`' for c in protected_dropped) if protected_dropped else 'None — all protected columns preserved ✅'}

Categorical protected columns may appear under one-hot encoded feature names.
Examples:
- Contract → Contract_Two year
- OnlineSecurity → OnlineSecurity_Yes

---

## 5c. Runtime Summary

| Metric | Value |
|---|---|
| Total pipeline runtime | {total_time:.1f}s ({total_time/60:.1f} min) |
| OpenAI (GPT-4.1) calls | {openai_calls} |
| Ollama (llama3.2:3b) calls | {ollama_calls} |
| Avg node latency | {avg_latency}s |

---

## 6. Explainability (SHAP)

### Top 10 Features by Importance
{top_features_str}

### Plain-English Explanation
{shap_narration}

### Actionable Insights
{chr(10).join('- ' + insight for insight in actionable_insights) if actionable_insights else '_None generated._'}

*See `shap_summary.png` and `shap_waterfall.png` for visual explanations.*

---

{critic_str}

## 7. Retry History

- Retries triggered: **{state.retry_count}** / {state.max_retries} max
- Errors encountered: {len(state.errors)}
{chr(10).join('- ' + e for e in state.errors) if state.errors else '- None'}

---

## 8. Full Reasoning Audit Trail

> Every decision the agent made, in order:

```
{reasoning_str}
```

---

## 9. Output Files

| File | Description |
|---|---|
{output_files_str}

**MLflow Dashboard:** Run `mlflow ui --port 5000` then open http://localhost:5000  
**Run ID:** `{run_id}`
"""

    report_path = Path(state.output_dir) / "final_report.md"
    report_path.write_text(report, encoding="utf-8")
    state.report_path = str(report_path)

    logger.info(f"✅ Report written: {report_path}")
    return state
