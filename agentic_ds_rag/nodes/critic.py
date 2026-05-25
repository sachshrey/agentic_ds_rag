# nodes/critic.py
"""
Self-correction loop. Detects issues and triggers retry or passes through.
The 5 checks below catch the most common ML pipeline failures.
"""
from __future__ import annotations
import logging

from state import AgentState, CriticFinding
from rag import retrieve_for_node
from utils.config import cfg

logger = logging.getLogger(__name__)


def compute_leakage_threshold(n_samples: int) -> float:
    """Scale leakage suspicion by dataset size."""
    if n_samples < 100:
        return 1.0
    if n_samples < 500:
        return 0.99
    if n_samples < 2000:
        return 0.98
    return 0.97


def check_for_leakage(state: AgentState, metrics: dict[str, float]) -> tuple[bool, list[str]]:
    n_samples = state.profile.rows if state.profile else len(state.raw_data)
    threshold = compute_leakage_threshold(n_samples)
    accuracy = metrics.get("accuracy", 0.0)
    findings: list[str] = []

    if accuracy < threshold:
        return False, findings

    train_val_gap = abs(state.experiment.train_score - state.experiment.val_score)
    findings.append(
        f"Accuracy {accuracy:.3f} >= leakage threshold {threshold:.2f} "
        f"for dataset size {n_samples}; train/validation gap={train_val_gap:.4f}."
    )

    if n_samples < 100:
        findings.append(
            f"Small dataset (n={n_samples}) can be genuinely separable; "
            "recording warning without retry."
        )
        return False, findings

    leakage_confirmed = n_samples >= 500 or train_val_gap < 0.02
    if not leakage_confirmed:
        findings.append(
            "High accuracy on small/medium data without confirming structural signals; "
            "skipping retry."
        )
    return leakage_confirmed, findings


def critic_node(state: AgentState) -> AgentState:
    if state.experiment is None:
        state.current_node = "explainer"
        return state

    metrics = state.experiment.metrics
    task = state.task.task_type
    findings: list[CriticFinding] = []

    # ── Check 1: Suspiciously high accuracy → possible leakage ────────────
    if "classification" in task:
        leakage_detected, leakage_details = check_for_leakage(state, metrics)
        if leakage_detected:
            findings.append(CriticFinding(
                issue_type="leakage_suspected",
                severity="high",
                detail=" ".join(leakage_details),
                fix="recheck_leakage_and_reprocess",
            ))
        elif leakage_details:
            findings.append(CriticFinding(
                issue_type="high_accuracy_small_dataset",
                severity="low",
                detail=" ".join(leakage_details),
                fix="document_no_retry",
            ))

    # ── Check 2: Overfitting (train/val gap) ──────────────────────────────
    gap = state.experiment.train_score - state.experiment.val_score
    if gap > cfg.model.overfit_gap_medium:
        findings.append(CriticFinding(
            issue_type="overfitting",
            severity="high" if gap > cfg.model.overfit_gap_high else "medium",
            detail=f"Train {state.experiment.train_score:.4f} vs val {state.experiment.val_score:.4f} — gap {gap:.4f}.",
            fix="increase_regularization_or_reduce_features",
        ))

    # ── Check 3: Imbalance ignored (accuracy >> F1) ───────────────────────
    if "classification" in task:
        acc = metrics.get("accuracy", 0)
        f1 = metrics.get("f1_weighted", acc)  # Default to accuracy if F1 not computed
        if acc - f1 > 0.10:
            findings.append(CriticFinding(
                issue_type="imbalance_ignored",
                severity="medium",
                detail=f"Accuracy {acc:.2%} vs F1 {f1:.2%} — gap {acc - f1:.2%} signals imbalance handled poorly.",
                fix="apply_smote_or_class_weights_retry",
            ))

    # ── Check 4: No meaningful improvement over baseline ──────────────────
    if "classification" in task:
        primary = "f1_weighted"
        higher_is_better = True
    elif task == "time_series_forecasting":
        primary = "rmse"
        higher_is_better = False
    else:
        primary = "r2"
        higher_is_better = True

    adv_score = metrics.get(primary, 0)
    base_score = state.baseline_metrics.get(primary, 0)

    # Skip check when no baseline available (e.g. time series — baseline node skips it)
    if base_score > 0 and primary in state.baseline_metrics:
        if higher_is_better:
            no_improvement = (adv_score - base_score) < cfg.model.min_improvement_over_baseline
        else:
            no_improvement = (base_score - adv_score) < cfg.model.min_improvement_over_baseline
        if no_improvement:
            findings.append(CriticFinding(
                issue_type="no_improvement_over_baseline",
                severity="medium",
                detail=(
                    f"{state.chosen_model} {primary}={adv_score:.4f} vs "
                    f"baseline {base_score:.4f} — "
                    f"only {abs(adv_score - base_score):.4f} gain."
                ),
                fix="review_features_or_accept_baseline",
            ))

    # ── Check 5: Negative R² (regression) ────────────────────────────────
    if task == "regression" and metrics.get("r2", 1.0) < 0:
        findings.append(CriticFinding(
            issue_type="model_worse_than_mean",
            severity="high",
            detail=f"R² = {metrics['r2']:.4f} — model is worse than predicting the mean.",
            fix="check_target_transform_or_outlier_removal",
        ))

    state.critic_findings = findings

    # ── RAG: retrieve fix strategies for found issues ─────────────────────
    if findings:
        rag_docs = retrieve_for_node(
            node_name="critic",
            task_type=state.task.task_type,
            domain=state.task.domain_detected,
            query=f"fix strategy for {[f.issue_type for f in findings]}",
        )
        state.rag_retrievals.append(
            f"CRITIC: {len(rag_docs)} fix strategy docs for {[f.issue_type for f in findings]}"
        )

    # Feature retention check — warn if over-pruning hurt the model
    original_col_count = state.profile.columns
    final_col_count = len(state.selected_features)
    drop_pct = (original_col_count - final_col_count) / max(original_col_count, 1)
    primary = "f1_weighted" if "classification" in task else "r2"
    adv_score = metrics.get(primary, 0)
    base_score = state.baseline_metrics.get(primary, 0)

    if drop_pct > 0.30 and (adv_score - base_score) < 0.02:
        state.reasoning_log.append(
            f"CRITIC WARNING: Over-pruning suspected — {drop_pct:.0%} of columns dropped "
            f"({original_col_count} → {final_col_count}) and improvement over baseline is only "
            f"{adv_score - base_score:+.4f}. Consider restoring domain-relevant columns "
            f"on next run. Check DOMAIN_PROTECTED_COLUMNS in preprocessing.py."
        )

    # ── Routing decision ──────────────────────────────────────────────────
    high_sev = [f for f in findings if f.severity == "high"]

    if high_sev and state.retry_count < state.max_retries:
        # Decide where to retry from
        fix_types = " ".join(f.fix for f in high_sev)
        if "leakage" in fix_types:
            state.retry_target_node = "preprocessing"
        elif "regularization" in fix_types or "features" in fix_types:
            state.retry_target_node = "model_selection"
        else:
            state.retry_target_node = "preprocessing"

        state.retry_count += 1
        state.current_node = state.retry_target_node

        state.reasoning_log.append(
            f"CRITIC: {len(high_sev)} HIGH severity issues → "
            f"RETRY {state.retry_count}/{state.max_retries} from {state.retry_target_node}. "
            f"Issues: {[f.issue_type for f in high_sev]}"
        )
    else:
        state.current_node = "explainer"
        if findings:
            state.reasoning_log.append(
                f"CRITIC: {len(findings)} issue(s) found but max retries reached or only low/medium. "
                f"Proceeding with documented caveats."
            )
        else:
            state.reasoning_log.append("CRITIC: No significant issues. Proceeding to explainer.")

    return state


def route_after_critic(state: AgentState) -> str:
    return state.current_node  # "preprocessing" / "model_selection" / "explainer"
