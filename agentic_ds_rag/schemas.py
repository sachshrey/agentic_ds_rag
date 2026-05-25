# schemas.py
"""
Pydantic models for every LLM structured output.
Rule: every LLM call returns one of these. Never parse raw text.
"""
from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, Field


# ── Task Detection ─────────────────────────────────────────────────────────────

class TaskDetectionOutput(BaseModel):
    """Returned by task_detector_node LLM call."""
    task_type: Literal[
        "binary_classification",
        "multiclass_classification",
        "regression",
        "time_series_forecasting",
    ]
    target_column: str | None = Field(
        None, description="Most likely target column name, or null if none"
    )
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str = Field(..., description="Concise explanation of the detection logic")
    leakage_suspects: list[str] = Field(
        default_factory=list,
        description="Column names that are likely target leakage"
    )
    dataset_category: Literal["tabular", "time_series"]
    domain_detected: str | None = Field(
        None, description="Detected domain: medical/financial/ecommerce/hr/telco/general"
    )
    recommended_metrics: list[str] = Field(default_factory=list)
    recommended_validation: str = "kfold_5"


# ── EDA ────────────────────────────────────────────────────────────────────────

class EDAInterpretation(BaseModel):
    """Returned by eda_node LLM call. LLM interprets computed stats — does NOT compute."""
    primary_insight: str = Field(
        ..., description="Single most important finding from the data"
    )
    data_quality_issues: list[str] = Field(
        default_factory=list,
        description="Quality problems that need fixing before modeling"
    )
    modeling_implications: list[str] = Field(
        default_factory=list,
        description="What EDA findings mean for model and feature choices"
    )
    recommended_preprocessing: list[str] = Field(
        default_factory=list,
        description="Specific preprocessing steps implied by EDA findings"
    )


# ── Preprocessing ──────────────────────────────────────────────────────────────

class PreprocessingDecision(BaseModel):
    """Returned by preprocessing_node LLM call."""
    imputation_strategy: dict[str, str] = Field(
        default_factory=dict,
        description="column_name -> strategy: median/mean/mode/unknown/knn/drop"
    )
    encoding_strategy: dict[str, str] = Field(
        default_factory=dict,
        description="column_name -> strategy: onehot/frequency/ordinal/catboost"
    )
    scaling_strategy: Literal["standard", "robust", "minmax", "none"] = "standard"
    imbalance_strategy: str | None = Field(
        None, description="smote / class_weight / none"
    )
    columns_to_drop: list[str] = Field(
        default_factory=list,
        description="Redundant, constant, or problematic columns to drop"
    )
    reasoning: str


# ── Feature Engineering ────────────────────────────────────────────────────────

class FeatureDef(BaseModel):
    """Single feature creation instruction."""
    name: str = Field(..., description="New column name, snake_case")
    operation: Literal["ratio", "product", "difference", "log", "bin", "interaction"]
    source_columns: list[str] = Field(..., min_length=1)
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="e.g. {'bins': 5} for bin operation"
    )
    reasoning: str = Field(..., description="Why this feature should improve the model")


class FeatureEngineeringPlan(BaseModel):
    """Returned by feature_engineering_node LLM call."""
    features_to_create: list[FeatureDef] = Field(
        default_factory=list,
        description="Max 5 features. Only suggest genuinely useful ones."
    )
    features_to_drop: list[str] = Field(
        default_factory=list,
        description="Existing features that are redundant or harmful"
    )
    reasoning: str


# ── Model Selection ────────────────────────────────────────────────────────────

class ModelSelectionOutput(BaseModel):
    """Returned by model_selection_node LLM call."""
    chosen_model: str = Field(
        ..., description="Exact model name from candidates list"
    )
    reasoning: str = Field(
        ..., description="Why this model over the alternatives, referencing dataset characteristics"
    )
    expected_improvement_over_baseline: str = Field(
        ..., description="Qualitative expectation: slight/moderate/significant improvement"
    )


# ── Explainability ─────────────────────────────────────────────────────────────

class ExplainabilityNarration(BaseModel):
    """Returned by explainer_node LLM call."""
    narration: str = Field(
        ..., description="2-3 sentence plain-English explanation of what drives predictions"
    )
    top_feature_explanations: list[str] = Field(
        default_factory=list,
        description="One sentence per top-3 feature explaining its impact in plain English"
    )
    actionable_insights: list[str] = Field(
        default_factory=list,
        description="2-3 business-relevant insights from the model"
    )
