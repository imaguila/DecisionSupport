"""
Workspace Dataset Module.

Provides utilities for column ordering, dynamic labeling, and rendering 
interactive data table previews for active solution sets.
"""

from typing import Any, Dict, List
import pandas as pd
import streamlit as st


def get_ordered_columns(df: pd.DataFrame, dataset_config: Dict[str, Any]) -> List[str]:
    """Orders DataFrame columns logically: ID, objectives, indicators, meta, decision vars."""
    if df is None or df.empty:
        return []

    var_prefix = dataset_config.get("var_prefix", "x_")
    objective_cols = dataset_config.get("metrics", [])
    indicator_cols = dataset_config.get("selected_indicators", [])

    decision_cols = [col for col in df.columns if var_prefix and col.startswith(var_prefix)]
    control_cols = {"highlight", "highlight_label", "label"}

    other_cols = [
        col for col in df.columns
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

    seen = set()
    ordered_cols: List[str] = []
    for col in raw_ordered_cols:
        if col in df.columns and col not in seen:
            seen.add(col)
            ordered_cols.append(col)

    return ordered_cols


def render_dataset_table(df: pd.DataFrame, dataset_config: Dict[str, Any]) -> None:
    """Renders an interactive Streamlit DataFrame table with prioritized column ordering."""
    if df is None or df.empty:
        st.info("No solutions available in the current dataset.")
        return

    st.markdown("#### 📋 Current Decision Set")
    ordered_cols = get_ordered_columns(df, dataset_config)
    display_df = df[ordered_cols] if ordered_cols else df

    st.dataframe(
        display_df,
        use_container_width=True,
        height=400,
        hide_index=True,
    )