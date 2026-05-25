# graph.py
"""
Complete LangGraph state machine.
All nodes, all edges, all conditional routing.
Includes node timing wrapper and confidence-based routing after task detection.
"""
from __future__ import annotations
import time
import uuid
from functools import wraps

from langgraph.graph import StateGraph, END

from state import AgentState
from checkpointing import save_sidecar_checkpoint
from nodes.profiler import profiler_node
from nodes.size_router import size_router_node
from nodes.task_detector import task_detector_node
from nodes.eda import eda_node
from nodes.preprocessing import preprocessing_node
from nodes.feature_engineering import feature_engineering_node
from nodes.baseline import baseline_trainer_node
from nodes.model_selection import model_selection_node
from nodes.training import training_node
from nodes.evaluation import evaluation_node
from nodes.critic import critic_node, route_after_critic
from nodes.explainer import explainer_node
from nodes.report import report_node
from utils.config import cfg


def timed_node(node_name: str, fn, run_id: str | None = None):
    """Wrap any node function to record wall-clock execution time in state."""
    @wraps(fn)
    def wrapper(state: AgentState) -> AgentState:
        t0 = time.perf_counter()
        result = fn(state)
        elapsed = round(time.perf_counter() - t0, 2)
        result.node_timings[node_name] = elapsed
        result.reasoning_log.append(f"PROFILER: {node_name} completed in {elapsed}s")
        if run_id:
            save_sidecar_checkpoint(result, run_id=run_id, node_name=node_name)
        return result
    return wrapper


def route_after_task_detection(state: AgentState) -> str:
    """
    Route to full pipeline (high confidence) or safe path (low confidence).
    Safe path: skips feature engineering and advanced model selection.
    Threshold configured in config.yaml under task_detection.low_confidence_threshold.
    """
    confidence = state.task.confidence if state.task else 0.0
    threshold = cfg.task_detection.low_confidence_threshold

    if confidence >= threshold:
        return "eda"

    # Low confidence: log warning and take safe path
    state.low_confidence_mode = True
    state.reasoning_log.append(
        f"ROUTING: Low task confidence ({confidence:.2f} < {threshold}) → "
        "safe path (skip feature engineering, run baseline only)."
    )
    return "eda_safe"


def build_graph(run_id: str | None = None) -> StateGraph:
    workflow = StateGraph(AgentState)

    # ── Register all nodes with timing wrapper ────────────────────────────
    workflow.add_node("profiler", timed_node("profiler", profiler_node, run_id))
    workflow.add_node("size_router", timed_node("size_router", size_router_node, run_id))
    workflow.add_node("task_detector", timed_node("task_detector", task_detector_node, run_id))
    workflow.add_node("eda", timed_node("eda", eda_node, run_id))
    workflow.add_node("eda_safe", timed_node("eda_safe", eda_node, run_id))
    workflow.add_node("preprocessing", timed_node("preprocessing", preprocessing_node, run_id))
    workflow.add_node("preprocessing_safe", timed_node("preprocessing_safe", preprocessing_node, run_id))
    workflow.add_node("feature_engineering", timed_node("feature_engineering", feature_engineering_node, run_id))
    workflow.add_node("baseline", timed_node("baseline", baseline_trainer_node, run_id))
    workflow.add_node("baseline_safe", timed_node("baseline_safe", baseline_trainer_node, run_id))
    workflow.add_node("model_selection", timed_node("model_selection", model_selection_node, run_id))
    workflow.add_node("training", timed_node("training", training_node, run_id))
    workflow.add_node("evaluation", timed_node("evaluation", evaluation_node, run_id))
    workflow.add_node("critic", timed_node("critic", critic_node, run_id))
    workflow.add_node("explainer", timed_node("explainer", explainer_node, run_id))
    workflow.add_node("report", timed_node("report", report_node, run_id))

    # ── Entry point ───────────────────────────────────────────────────────
    workflow.set_entry_point("profiler")

    # ── Linear pipeline: profiler → size_router → task_detector ──────────
    workflow.add_edge("profiler", "size_router")
    workflow.add_edge("size_router", "task_detector")

    # ── Confidence-based routing after task detection ─────────────────────
    workflow.add_conditional_edges(
        "task_detector",
        route_after_task_detection,
        {
            "eda": "eda",
            "eda_safe": "eda_safe",
        },
    )

    # ── FULL pipeline path ────────────────────────────────────────────────
    workflow.add_edge("eda", "preprocessing")
    workflow.add_edge("preprocessing", "feature_engineering")
    workflow.add_edge("feature_engineering", "baseline")
    workflow.add_edge("baseline", "model_selection")
    workflow.add_edge("model_selection", "training")
    workflow.add_edge("training", "evaluation")
    workflow.add_edge("evaluation", "critic")

    # ── SAFE path (low confidence) ────────────────────────────────────────
    workflow.add_edge("eda_safe", "preprocessing_safe")
    workflow.add_edge("preprocessing_safe", "baseline_safe")
    workflow.add_edge("baseline_safe", "evaluation")

    # ── Conditional edge: critic → retry or proceed ───────────────────────
    workflow.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "preprocessing": "preprocessing",
            "model_selection": "model_selection",
            "explainer": "explainer",
        },
    )

    # ── Final edges ───────────────────────────────────────────────────────
    workflow.add_edge("explainer", "report")
    workflow.add_edge("report", END)

    return workflow


def compile_graph(enable_checkpointing: bool = True):
    """Compile graph with optional lightweight sidecar checkpoints."""
    run_id = str(uuid.uuid4())[:8] if enable_checkpointing else None
    workflow = build_graph(run_id=run_id)
    return workflow.compile()
