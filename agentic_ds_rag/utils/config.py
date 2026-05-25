# utils/config.py
from __future__ import annotations
from pathlib import Path
from pydantic import BaseModel
import yaml


class DataConfig(BaseModel):
    null_drop_threshold_pct: float = 70.0
    eda_sample_rows: int = 10000
    correlation_drop_threshold: float = 0.80


class RagConfig(BaseModel):
    retrieval_k: int = 4


class ModelConfig(BaseModel):
    min_improvement_over_baseline: float = 0.02
    overfit_gap_medium: float = 0.15
    overfit_gap_high: float = 0.25
    perfect_accuracy_threshold: float = 0.98


class TaskConfig(BaseModel):
    low_confidence_threshold: float = 0.65


class PathsConfig(BaseModel):
    output_dir: str = "outputs"
    cache_dir: str = "cache"
    mlflow_tracking_uri: str = "mlruns"


class PipelineConfig(BaseModel):
    data: DataConfig = DataConfig()
    rag: RagConfig = RagConfig()
    model: ModelConfig = ModelConfig()
    task_detection: TaskConfig = TaskConfig()
    paths: PathsConfig = PathsConfig()


def load_config(path: str = "config.yaml") -> PipelineConfig:
    config_path = Path(path)
    if config_path.exists():
        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}
        return PipelineConfig(**raw)
    return PipelineConfig()


# Module-level singleton — import cfg from here everywhere
cfg = load_config()
