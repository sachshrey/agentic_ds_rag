"""
Lightweight sidecar checkpoints for LangGraph runs.

The graph state contains DataFrames and fitted models, so LangGraph's default
MemorySaver cannot safely serialize the whole state. These helpers write a
JSON-safe subset after each node instead.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel


_HEAVY_FIELDS = {
    "raw_data",
    "trained_model",
    "scaler",
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _json_safe(value.model_dump())
    if isinstance(value, dict):
        return {
            str(k): _json_safe(v)
            for k, v in value.items()
            if k not in _HEAVY_FIELDS
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)


def state_to_checkpoint_dict(state: Any) -> dict[str, Any]:
    raw = state.model_dump() if isinstance(state, BaseModel) else dict(state)
    return {
        key: _json_safe(value)
        for key, value in raw.items()
        if key not in _HEAVY_FIELDS
    }


def save_sidecar_checkpoint(state: Any, run_id: str, node_name: str) -> Path:
    output_dir = getattr(state, "output_dir", "outputs") or "outputs"
    checkpoint_dir = Path(output_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    snapshot = {
        "run_id": run_id,
        "node": node_name,
        "timestamp": dt.datetime.utcnow().isoformat(),
        "state": state_to_checkpoint_dict(state),
    }
    path = checkpoint_dir / f"{run_id}_{node_name}.json"
    path.write_text(json.dumps(snapshot, indent=2, default=str), encoding="utf-8")
    return path
