# nodes/size_router.py
from state import AgentState


def size_router_node(state: AgentState) -> AgentState:
    """Deterministic — no LLM. Sets EDA mode based on row count."""
    cat = state.profile.size_category

    if cat in ("small", "medium"):
        state.eda_mode = "full"
        state.reasoning_log.append(
            f"SIZE ROUTER: {cat} dataset ({state.profile.rows:,} rows). Full EDA mode."
        )
    else:
        state.eda_mode = "sampled"
        state.reasoning_log.append(
            f"SIZE ROUTER: large dataset ({state.profile.rows:,} rows). "
            f"Sampled EDA (10k rows). Full data used for training."
        )
    return state
