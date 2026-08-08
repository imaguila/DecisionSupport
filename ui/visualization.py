"""
Visualization Module.

Pure Plotly figure generation functions (decoupled from Streamlit layout where possible)
and Streamlit chart wrappers.
"""

from typing import Optional, List
import pandas as pd
import plotly.express as px
import streamlit as st


# =====================================================
# COLOR & HOVER UTILITIES
# =====================================================

def infer_lens_color_column(df: pd.DataFrame, user_color: Optional[str] = None) -> Optional[str]:
    """Infers the most informative color encoding column based on active lens metadata."""
    if "group_label" in df.columns:
        return "group_label"
    if "cluster_str" in df.columns:
        return "cluster_str"
    if "preference_score" in df.columns:
        return "preference_score"
    if "efficiency_score" in df.columns:
        return "efficiency_score"
    if "consensus_score" in df.columns:
        return "consensus_score"
    if "domain_match_count" in df.columns:
        return "domain_match_count"
    return user_color


def is_discrete_color(df: pd.DataFrame, color_column: Optional[str]) -> bool:
    """Checks if a color column should be treated as discrete/categorical."""
    if color_column is None or color_column not in df.columns:
        return False
    if color_column in [
        "group_label",
        "cluster_str",
        "preference_method",
        "efficiency_method",
        "domain_matched_metrics",
    ]:
        return True
    return pd.api.types.is_object_dtype(df[color_column])


def build_hover_columns(df: pd.DataFrame) -> List[str]:
    """Selects suitable columns to display in interactive plot tooltips."""
    excluded_prefixes = ("req_", "var_", "x_")
    excluded_cols = {"label", "highlight", "highlight_label"}
    return [
        col for col in df.columns
        if col not in excluded_cols and not col.startswith(excluded_prefixes)
    ]


# =====================================================
# PLOTLY FIGURE BUILDERS
# =====================================================

def render_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    size: Optional[str] = None,
    color: Optional[str] = None,
    show_ids: bool = False,
    key: Optional[str] = None,
) -> None:
    """Renders a single 2D Scatter or Bubble chart in Streamlit."""
    df_plot = df.copy()

    if x not in df_plot.columns or y not in df_plot.columns:
        st.warning("Selected axes are not available in the current dataset.")
        return

    text_column = None
    if show_ids:
        if "id" in df_plot.columns:
            text_column = "id"
        elif "ID" in df_plot.columns:
            text_column = "ID"

    plot_color = infer_lens_color_column(df_plot, user_color=color)
    discrete_color = is_discrete_color(df_plot, plot_color)
    hover_cols = build_hover_columns(df_plot)

    if discrete_color and plot_color is not None:
        df_plot[plot_color] = df_plot[plot_color].astype(str)

    fig = px.scatter(
        df_plot,
        x=x,
        y=y,
        size=size if size in df_plot.columns else None,
        color=plot_color if plot_color in df_plot.columns else None,
        text=text_column,
        hover_data=hover_cols,
        template="plotly_white",
    )
    fig.update_layout(margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig, use_container_width=True, key=key)


def render_coordinated_maps(
    df: pd.DataFrame,
    x: str,
    y: str,
    z: str,
    key_prefix: str,
    show_ids: bool = False,
) -> None:
    """Renders two coordinated side-by-side scatter plots (X vs Y and X vs Z)."""
    col1, col2 = st.columns(2)
    with col1:
        st.caption(f"**Map A:** {x} vs {y}")
        render_scatter(df, x=x, y=y, show_ids=show_ids, key=f"{key_prefix}_a")
    with col2:
        st.caption(f"**Map B:** {x} vs {z}")
        render_scatter(df, x=x, y=z, show_ids=show_ids, key=f"{key_prefix}_b")


def render_distribution(
    df: pd.DataFrame,
    metric: str,
    mode: str = "Violin",
    key: Optional[str] = None,
) -> None:
    """Renders statistical distribution plots (Violin or Box)."""
    if metric not in df.columns:
        st.warning(f"Metric '{metric}' not found in dataframe.")
        return

    color_col = infer_lens_color_column(df)

    if mode == "Violin":
        fig = px.violin(df, y=metric, color=color_col, box=True, points="all", template="plotly_white")
    else:
        fig = px.box(df, y=metric, color=color_col, points="all", template="plotly_white")

    fig.update_layout(margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig, use_container_width=True, key=key)

