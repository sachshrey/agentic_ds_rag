# state.py
from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, Field, ConfigDict


class DatasetProfile(BaseModel):
    rows: int
    columns: int
    dtypes: dict[str, str]
    null_percentages: dict[str, float]       # column → null %
    cardinality: dict[str, int]              # column → unique count
    skewness: dict[str, float]               # numeric columns only
    correlations_high: list[tuple[str, str, float]]  # (col_a, col_b, corr)
    memory_usage_mb: float
    duplicate_rows: int
    size_category: Literal["small", "medium", "large"]


class TaskDetection(BaseModel):
    task_type: Literal[
        "binary_classification",
        "multiclass_classification",
        "regression",
        "time_series_forecasting",
    ]
    target_column: str | None = None
    confidence: float
    reasoning: str
    leakage_suspects: list[str] = Field(default_factory=list)
    dataset_category: Literal["tabular", "time_series"]
    domain_detected: str | None = None
    recommended_metrics: list[str] = Field(default_factory=list)
    recommended_validation: str = "kfold_5"


class ModelCandidate(BaseModel):
    name: str
    heuristic_score: float
    reasoning_points: list[str] = Field(default_factory=list)
    pros: list[str] = Field(default_factory=list)
    cons: list[str] = Field(default_factory=list)
    estimated_training_time: str = "unknown"


class ExperimentResult(BaseModel):
    model_name: str
    metrics: dict[str, float] = Field(default_factory=dict)
    baseline_metrics: dict[str, float] = Field(default_factory=dict)
    validation_strategy: str
    train_score: float = 0.0
    val_score: float = 0.0
    feature_importance: dict[str, float] = Field(default_factory=dict)
    shap_values_path: str | None = None
    mlflow_run_id: str | None = None


class CriticFinding(BaseModel):
    issue_type: str
    severity: Literal["low", "medium", "high"]
    detail: str
    fix: str


class AgentState(BaseModel):
    """
    Central state object passed through every LangGraph node.
    Design note: DataFrames stored as Any to satisfy Pydantic;
    arbitrary_types_allowed handles this cleanly in v2.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # ── Input ────────────────────────────────────────────────────────────
    file_path: str
    raw_data: Any = None                    # pd.DataFrame
    user_context: str | None = None

    # ── Profiling ────────────────────────────────────────────────────────
    profile: DatasetProfile | None = None
    dataset_hash: str = ""
    profile_report_path: str | None = None

    # ── Routing ──────────────────────────────────────────────────────────
    eda_mode: Literal["full", "sampled"] = "full"
    active_path: str | None = None
    low_confidence_mode: bool = False

    # ── Task ─────────────────────────────────────────────────────────────
    task: TaskDetection | None = None

    # ── Data Processing ───────────────────────────────────────────────────
    processed_data: dict[str, Any] = Field(default_factory=dict)
    selected_features: list[str] = Field(default_factory=list)
    dropped_columns: list[str] = Field(default_factory=list)
    preprocessing_steps: list[str] = Field(default_factory=list)
    scaler: Any = None                      # Fitted sklearn scaler (apply to test)

    # ── Baseline ─────────────────────────────────────────────────────────
    baseline_metrics: dict[str, float] = Field(default_factory=dict)

    # ── Modeling ─────────────────────────────────────────────────────────
    model_candidates: list[ModelCandidate] = Field(default_factory=list)
    chosen_model: str | None = None
    model_source: str | None = None
    trained_model: Any = None
    experiment: ExperimentResult | None = None
    shap_importance_table: list[dict[str, Any]] = Field(default_factory=list)

    # ── Reasoning Audit Trail ─────────────────────────────────────────────
    reasoning_log: list[str] = Field(default_factory=list)
    node_timings: dict[str, float] = Field(default_factory=dict)
    rag_retrievals: list[str] = Field(default_factory=list)
    tool_calls_log: list[str] = Field(default_factory=list)

    # ── Retry & Error ────────────────────────────────────────────────────
    retry_count: int = 0
    max_retries: int = 3
    errors: list[str] = Field(default_factory=list)
    critic_findings: list[CriticFinding] = Field(default_factory=list)
    current_node: str = "input"
    retry_target_node: str | None = None

    # Human-in-the-loop approval is not implemented yet. Future work can add
    # LangGraph interrupt() gates after task detection and model selection.

    # ── Outputs ──────────────────────────────────────────────────────────
    report_path: str | None = None
    output_dir: str = "outputs"
