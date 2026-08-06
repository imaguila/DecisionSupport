"""
Workspace Module.

Serves as the main orchestrator for rendering the visual workspace layout, 
combining executive summary views, dataset previews, and interactive 
decision-space maps.
"""

from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from core.workspace_maps import render_maps
from core.workspace_summary import render_summary


# =====================================================
# HELPER FUNCTIONS
# =====================================================


def get_workspace_dimensions(dataset: Dict[str, Any]) -> List[str]:
    """
    Extracts active optimization metrics and selected indicator dimensions.

    Parameters
    ----------
    dataset : Dict[str, Any]
        Global dataset context dictionary.

    Returns
    -------
    List[str]
        List of active dimension column names.
    """
    if not dataset:
        return []

    metrics = dataset.get("metrics", []) or []
    indicators = dataset.get("selected_indicators", []) or []

    return list(metrics) + list(indicators)


def render_empty_workspace_message() -> None:
    """Renders an error message when no valid dataset DataFrame is available."""
    st.error("No dataset is available for the workspace.")


def render_no_map_message() -> None:
    """Renders a warning message when insufficient dimensions exist for mapping."""
    st.warning(
        "At least two dimensions are required to render decision-space maps."
    )


# =====================================================
# MAIN WORKSPACE ENTRY POINT
# =====================================================


def render_workspace(
    df: Optional[pd.DataFrame],
    dataset: Dict[str, Any],
    show_ids: bool = False,
) -> None:
    """
    Renders the primary visual workspace UI.

    Integrates high-level summary metrics, solution set preview tables, and 
    interactive decision-space maps.

    Parameters
    ----------
    df : Optional[pd.DataFrame]
        Active solution space DataFrame.
    dataset : Dict[str, Any]
        Global dataset context dictionary.
    show_ids : bool, default=False
        Whether to display solution ID text labels across workspace maps.
    """
    if df is None or df.empty:
        render_empty_workspace_message()
        return

    dimensions = get_workspace_dimensions(dataset)

    # ==================== SUMMARY & CURRENT SET ====================
    render_summary(df, dataset)

    # ====================== DECISION MAPS ==========================
    if len(dimensions) < 2:
        render_no_map_message()
    else:
        render_maps(df, dataset, dimensions, show_ids)