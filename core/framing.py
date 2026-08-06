"""
Context Framing Module.

Provides range-based bounding filters on numeric metrics and derived indicators 
to dynamically reduce the visible solution decision space.
"""

from typing import Any, Dict, List, Tuple

import pandas as pd
import streamlit as st

from ui.phase_help import render_phase_help_icon


# =====================================================
# HELPER FUNCTIONS
# =====================================================


def get_framing_dimensions(dataset: Dict[str, Any]) -> List[str]:
    """
    Extracts all filterable metric and indicator dimension names from dataset context.

    Parameters
    ----------
    dataset : Dict[str, Any]
        Global dataset configuration metadata.

    Returns
    -------
    List[str]
        List of dimension column names available for range filtering.
    """
    if not dataset:
        return []

    metrics = dataset.get("metrics", [])
    indicators = dataset.get("selected_indicators", [])

    return list(metrics) + list(indicators)


def is_valid_numeric_dimension(df: pd.DataFrame, column: str) -> bool:
    """
    Checks whether a column exists in the DataFrame and contains numeric data.

    Parameters
    ----------
    df : pd.DataFrame
        Target solution space DataFrame.
    column : str
        Column name to evaluate.

    Returns
    -------
    bool
        True if column exists and is numeric; False otherwise.
    """
    if df is None or column not in df.columns:
        return False

    return pd.api.types.is_numeric_dtype(df[column])


def apply_dimension_filter(
    filtered_df: pd.DataFrame,
    metric: str,
    selected_range: Tuple[float, float],
) -> pd.DataFrame:
    """
    Filters a DataFrame based on a closed numeric interval [min, max].

    Parameters
    ----------
    filtered_df : pd.DataFrame
        DataFrame to be filtered.
    metric : str
        Target column name to apply bounding condition.
    selected_range : Tuple[float, float]
        Lower and upper bounds for filtering.

    Returns
    -------
    pd.DataFrame
        Filtered solution space DataFrame.
    """
    if filtered_df is None or filtered_df.empty:
        return filtered_df

    return filtered_df[
        (filtered_df[metric] >= selected_range[0])
        & (filtered_df[metric] <= selected_range[1])
    ]


# =====================================================
# UI RENDERING & SUMMARY
# =====================================================


def render_framing_summary(
    original_df: pd.DataFrame, filtered_df: pd.DataFrame
) -> None:
    """
    Renders progress bar and ratio metrics summarizing solution space reduction.

    Parameters
    ----------
    original_df : pd.DataFrame
        Original un-filtered solution space DataFrame.
    filtered_df : pd.DataFrame
        Filtered active solution space DataFrame.
    """
    total_solutions = len(original_df) if original_df is not None else 0
    remaining_solutions = len(filtered_df) if filtered_df is not None else 0

    ratio = remaining_solutions / max(total_solutions, 1)
    st.progress(ratio)

    st.markdown(
        f"""
        <div style="text-align:center">
            <div style="font-size:0.9rem;color:gray;">
                Remaining Solutions
            </div>
            <div style="font-size:1.8rem;font-weight:bold;">
                {remaining_solutions}/{total_solutions}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(f"{ratio:.0%} of the decision space is visible.")


# =====================================================
# MAIN PIPELINE ENTRY POINT
# =====================================================


def apply_framing(dataset: Dict[str, Any]) -> pd.DataFrame:
    """
    Renders UI sliders for context framing and returns the filtered solution space.

    Parameters
    ----------
    dataset : Dict[str, Any]
        Global dataset context dictionary containing working DataFrame and metrics.

    Returns
    -------
    pd.DataFrame
        Range-bounded solution space DataFrame.
    """
    if not dataset or "df" not in dataset or dataset["df"] is None:
        return pd.DataFrame()

    df = dataset["df"].copy()
    if df.empty:
        return df

    filtered_df = df.copy()
    dimensions = get_framing_dimensions(dataset)

    with st.sidebar.expander("🎛️ Context Framing", expanded=False):
        col_label, col_help = st.columns(
            [0.85, 0.15], vertical_alignment="center"
        )

        with col_label:
            st.markdown("**Bounded Range Filters**")

        with col_help:
            render_phase_help_icon("framing", key="help_input_phase")

        for metric in dimensions:
            if not is_valid_numeric_dimension(df, metric):
                continue

            min_v = float(df[metric].min())
            max_v = float(df[metric].max())

            if pd.isna(min_v) or pd.isna(max_v) or min_v >= max_v:
                continue

            step_val = (max_v - min_v) / 1000.0

            selected_range = st.slider(
                metric,
                min_value=min_v,
                max_value=max_v,
                value=(min_v, max_v),
                step=step_val,
                key=f"framing_{metric}",
            )

            unchanged = (
                abs(selected_range[0] - min_v) < 1e-6
                and abs(selected_range[1] - max_v) < 1e-6
            )
            if unchanged:
                continue

            filtered_df = apply_dimension_filter(
                filtered_df, metric, selected_range
            )

        render_framing_summary(df, filtered_df)

    return filtered_df