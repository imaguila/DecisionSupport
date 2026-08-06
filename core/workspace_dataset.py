"""
Workspace Dataset Module.

Provides utilities for column ordering, dynamic labeling, and rendering 
interactive data table previews for active solution sets.
"""

from typing import Any, Dict, List

import pandas as pd
import streamlit as st


# =====================================================
# HELPER FUNCTIONS
# =====================================================


def get_ordered_columns(df: pd.DataFrame, dataset: Dict[str, Any]) -> List[str]:
    """
    Orders DataFrame columns logically by category: ID, objectives, indicators, 
    miscellaneous metadata, and decision variables.

    Parameters
    ----------
    df : pd.DataFrame
        Active solution space DataFrame.
    dataset : Dict[str, Any]
        Global dataset context dictionary.

    Returns
    -------
    List[str]
        List of column names in prioritized ordering.
    """
    if df is None or df.empty:
        return []

    if not dataset:
        dataset = {}

    config = dataset.get("config", {})
    var_prefix = config.get("var_prefix", "x_")

    objective_cols = dataset.get("metrics", [])
    indicator_cols = dataset.get("selected_indicators", [])

    decision_cols = [
        col for col in df.columns if var_prefix and col.startswith(var_prefix)
    ]
    control_cols = {"highlight", "highlight_label", "label"}

    other_cols = [
        col
        for col in df.columns
        if (
            col not in objective_cols
            and col not in indicator_cols
            and col not in decision_cols
            and col not in control_cols
            and col != "id"
        )
    ]

    raw_ordered_cols = (
        (["id"] if "id" in df.columns else [])
        + objective_cols
        + indicator_cols
        + other_cols
        + decision_cols
    )

    # Deduplicate while preserving order and ensuring columns exist in df
    seen = set()
    ordered_cols: List[str] = []
    for col in raw_ordered_cols:
        if col in df.columns and col not in seen:
            seen.add(col)
            ordered_cols.append(col)

    return ordered_cols


def get_current_set_label() -> str:
    """
    Retrieves UI label corresponding to the active solution set state.

    Returns
    -------
    str
        Human-readable label for the current set.
    """
    if st.session_state.get("css_enabled", False):
        return "Current CSS"
    return "Current Decision Set"


# =====================================================
# UI RENDERING COMPONENTS
# =====================================================


def render_dataset_table(df: pd.DataFrame, dataset: Dict[str, Any]) -> None:
    """
    Renders an interactive Streamlit DataFrame table with prioritized column ordering.

    Parameters
    ----------
    df : pd.DataFrame
        Solution set DataFrame to display.
    dataset : Dict[str, Any]
        Global dataset context dictionary.
    """
    if df is None or df.empty:
        st.info("No solutions available in the current dataset.")
        return

    label = get_current_set_label()
    st.markdown(f"#### 📋 {label}")

    ordered_cols = get_ordered_columns(df, dataset)
    display_df = df[ordered_cols] if ordered_cols else df

    st.dataframe(
        display_df,
        use_container_width=True,
        height=420,
        hide_index=True,
    )


def render_dataset_preview(df: pd.DataFrame, dataset: Dict[str, Any]) -> None:
    """
    Renders a collapsible expander containing the interactive dataset table preview.

    Parameters
    ----------
    df : pd.DataFrame
        Solution set DataFrame to display.
    dataset : Dict[str, Any]
        Global dataset context dictionary.
    """
    if df is None or df.empty:
        return

    label = get_current_set_label()
    config = dataset.get("config", {}) if dataset else {}
    var_prefix = config.get("var_prefix", "x_")

    with st.expander(f"📋 {label} (prefix: {var_prefix})", expanded=False):
        render_dataset_table(df, dataset)