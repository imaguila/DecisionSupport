"""
Candidate Solution Set (CSS) Streamlit UI Module.

Renders interactive sidebar panels and comparison components for Streamlit apps,
consuming logic from `css_core`.
"""

from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st

from core.css import (
    apply_css_filtering,
    compute_baseline_differences,
    compute_decision_variable_summary,
    compute_similarity_matrix,
    compute_variable_metric_correlation,
    create_baseline_difference_fig,
    create_decision_variable_distribution_fig,
    create_decision_variable_matrix_fig,
    create_parallel_coordinates_fig,
    create_solution_similarity_fig,
    create_tradeoff_radar_fig,
    create_variable_metric_correlation_fig,
    get_decision_variable_columns,
    get_numeric_dimensions,
    sanitize_ids,
)


# =====================================================
# SESSION STATE INITIALIZATION
# =====================================================


def ensure_css_state() -> None:
    """Ensures all session state keys required for CSS management are initialized."""
    defaults: Dict[str, Any] = {
        "css_enabled": False,
        "css_source": "Current set",
        "css_manual_ids": [],
        "css_highlight_ids": [],
        "show_css_comparison": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# =====================================================
# SIDEBAR PANEL
# =====================================================


def render_css_panel(
    current_df: Optional[pd.DataFrame],
    dataset: Optional[Dict[str, Any]] = None,
) -> Optional[pd.DataFrame]:
    """Renders sidebar controls for managing the Candidate Solution Set (CSS)."""
    ensure_css_state()

    if current_df is None or current_df.empty:
        return current_df

    valid_ids = (
        current_df["id"].dropna().tolist() if "id" in current_df.columns else []
    )

    st.session_state.css_manual_ids = sanitize_ids(
        st.session_state.css_manual_ids, valid_ids
    )
    st.session_state.css_highlight_ids = sanitize_ids(
        st.session_state.css_highlight_ids, valid_ids
    )

    with st.sidebar.expander("🎯 Candidate Solution Set", expanded=False):
        st.session_state.css_enabled = st.checkbox(
            "Lock current set as CSS",
            value=st.session_state.css_enabled,
            help="Create a Candidate Solution Set from current filtered set or manual selection.",
        )

        if not st.session_state.css_enabled:
            st.caption(f"Current set available: {len(current_df)} solutions")
            return apply_css_filtering(current_df, enabled=False)

        sources = ["Current set", "Manual selection"]
        source_idx = (
            sources.index(st.session_state.css_source)
            if st.session_state.css_source in sources
            else 0
        )

        st.session_state.css_source = st.radio(
            "CSS source",
            sources,
            index=source_idx,
            horizontal=True,
        )

        if st.session_state.css_source == "Manual selection":
            st.session_state.css_manual_ids = st.multiselect(
                "Solutions included in CSS",
                options=valid_ids,
                default=st.session_state.css_manual_ids,
                key="css_manual_ids_widget",
                help="Select the exact solutions that form the Candidate Solution Set.",
            )

        st.session_state.show_css_comparison = st.checkbox(
            "Open detailed comparison",
            value=st.session_state.show_css_comparison,
            help="Open detailed visual comparison section for the current CSS.",
        )

    css_df = apply_css_filtering(
        df=current_df,
        enabled=st.session_state.css_enabled,
        source=st.session_state.css_source,
        manual_ids=st.session_state.css_manual_ids,
        highlight_ids=st.session_state.css_highlight_ids,
    )

    if st.session_state.css_enabled:
        st.sidebar.info(f"CSS size: {len(css_df)} solutions")

    return css_df


# =====================================================
# RENDER WIDGETS
# =====================================================


def render_tradeoff_radar(
    compare_df: pd.DataFrame, css_df: pd.DataFrame, dataset: Dict[str, Any]
) -> None:
    numeric_dimensions = get_numeric_dimensions(css_df, dataset)

    if len(numeric_dimensions) < 3:
        st.info(
            "At least three numeric objectives or indicators are required to create a radar chart."
        )
        return

    selected_metrics = st.multiselect(
        "Objectives and indicators for radar profile",
        numeric_dimensions,
        default=numeric_dimensions[: min(5, len(numeric_dimensions))],
        key="css_tradeoff_metrics",
    )

    if len(selected_metrics) < 3:
        st.warning("Select at least three objectives or indicators.")
        return

    metric_goals = {}
    cols = st.columns(len(selected_metrics))

    for idx, metric in enumerate(selected_metrics):
        with cols[idx]:
            metric_goals[metric] = st.selectbox(
                metric,
                ["Maximize", "Minimize"],
                key=f"css_goal_{metric}",
            )

    fig = create_tradeoff_radar_fig(compare_df, selected_metrics, metric_goals)
    st.plotly_chart(fig, use_container_width=True)


def render_parallel_coordinates(
    compare_df: pd.DataFrame, dataset: Dict[str, Any]
) -> None:
    numeric_dims = get_numeric_dimensions(compare_df, dataset)

    if len(numeric_dims) < 2:
        st.info(
            "At least two numerical metrics are required for Parallel Coordinates."
        )
        return

    selected_dims = st.multiselect(
        "Metrics for Parallel Coordinates",
        numeric_dims,
        default=numeric_dims[: min(6, len(numeric_dims))],
        key="css_parcoords_dims",
    )

    if not selected_dims:
        return

    fig = create_parallel_coordinates_fig(compare_df, selected_dims)
    st.plotly_chart(fig, use_container_width=True)


def render_baseline_difference_chart(
    compare_df: pd.DataFrame, dataset: Dict[str, Any]
) -> None:
    numeric_dims = get_numeric_dimensions(compare_df, dataset)
    if not numeric_dims or "id" not in compare_df.columns:
        return

    col_base, col_metrics = st.columns([1, 2])

    with col_base:
        baseline_id = st.selectbox(
            "Select Baseline Solution",
            options=compare_df["id"].tolist(),
            format_func=lambda x: f"ID {int(x)}",
            key="css_baseline_id",
        )

    with col_metrics:
        selected_metrics = st.multiselect(
            "Metrics to compare vs Baseline",
            numeric_dims,
            default=numeric_dims[: min(4, len(numeric_dims))],
            key="css_baseline_metrics",
        )

    if not selected_metrics or baseline_id is None:
        return

    diff_df = compute_baseline_differences(
        compare_df, baseline_id, selected_metrics
    )

    if diff_df.empty:
        st.info(
            "Select at least one additional solution to compare against the Baseline."
        )
        return

    fig = create_baseline_difference_fig(diff_df)
    st.plotly_chart(fig, use_container_width=True)


def render_solution_similarity_matrix(
    compare_df: pd.DataFrame, dataset: Dict[str, Any]
) -> None:
    var_cols = get_decision_variable_columns(compare_df, dataset)
    if not var_cols or len(compare_df) < 2:
        st.info(
            "Requires decision variables and at least 2 solutions to compute similarity."
        )
        return

    sim_matrix = compute_similarity_matrix(compare_df, var_cols)
    fig = create_solution_similarity_fig(sim_matrix)
    st.plotly_chart(fig, use_container_width=True)


def render_decision_variable_matrix(
    compare_df: pd.DataFrame, dataset: Dict[str, Any]
) -> None:
    variable_cols = get_decision_variable_columns(compare_df, dataset)
    config = dataset.get("config", {}) if dataset else {}
    var_prefix = config.get("var_prefix", "x_")

    if not variable_cols:
        st.info(
            f"No numeric decision-variable columns with prefix '{var_prefix}' found."
        )
        return

    fig = create_decision_variable_matrix_fig(compare_df, variable_cols)
    st.plotly_chart(fig, use_container_width=True)


def render_decision_variable_distribution(
    css_df: pd.DataFrame, dataset: Dict[str, Any]
) -> None:
    variable_cols = get_decision_variable_columns(css_df, dataset)
    config = dataset.get("config", {}) if dataset else {}
    var_prefix = config.get("var_prefix", "x_")

    if not variable_cols:
        st.info(
            f"No numeric decision-variable columns with prefix '{var_prefix}' found."
        )
        return

    variable_summary = compute_decision_variable_summary(css_df, variable_cols)
    max_variables = min(50, len(variable_summary))

    if max_variables < 1:
        st.info("No decision variables can be summarized.")
        return

    top_n = st.slider(
        "Decision variables to show",
        min_value=1,
        max_value=max_variables,
        value=min(20, max_variables),
        key="css_decision_variable_top_n",
    )

    plot_df = variable_summary.head(top_n)
    fig = create_decision_variable_distribution_fig(plot_df)
    st.plotly_chart(fig, use_container_width=True)


def render_variable_metric_correlation(
    compare_df: pd.DataFrame, dataset: Dict[str, Any]
) -> None:
    var_cols = get_decision_variable_columns(compare_df, dataset)
    metric_cols = get_numeric_dimensions(compare_df, dataset)

    if not var_cols or not metric_cols:
        st.info(
            "Both decision variables and numeric metrics are required to compute mapping."
        )
        return

    if len(compare_df) < 2:
        st.info("Select at least 2 solutions to calculate correlation.")
        return

    xy_corr = compute_variable_metric_correlation(
        compare_df, var_cols, metric_cols
    )

    if xy_corr.empty:
        st.info(
            "Could not calculate variance/correlation for the selected subset."
        )
        return

    fig = create_variable_metric_correlation_fig(xy_corr, len(var_cols))
    st.plotly_chart(fig, use_container_width=True)


# =====================================================
# MAIN CSS COMPARISON PIPELINE
# =====================================================


def render_css_comparison(
    css_df: pd.DataFrame, dataset: Dict[str, Any]
) -> None:
    """Main entry point for rendering the detailed CSS comparison panel."""
    if not st.session_state.get("show_css_comparison", False):
        return

    with st.expander("🆚 Detailed comparison", expanded=True):
        if css_df is None or css_df.empty:
            st.info("No Candidate Solution Set is available for comparison.")
            return

        if "id" not in css_df.columns:
            st.warning("The current CSS does not contain an 'id' column.")
            return

        css_ids = css_df["id"].dropna().astype(int).tolist()
        default_ids = st.session_state.get("css_highlight_ids", [])
        default_ids = [sid for sid in default_ids if sid in css_ids]

        compare_ids = st.multiselect(
            "Pick solutions to compare & highlight",
            css_ids,
            default=default_ids,
            key="css_compare_ids",
        )

        st.session_state.css_highlight_ids = compare_ids

        if len(compare_ids) < 2:
            st.info("Select at least 2 solutions to compare.")
            return

        compare_df = css_df[css_df["id"].isin(compare_ids)].copy()

        tab_metrics, tab_vars, tab_sim, tab_mapping = st.tabs(
            [
                "📊 Metrics & Trade-offs",
                "📋 Decision Variables",
                "🔀 Structural Similarity",
                "🔗 X → Y Mapping",
            ]
        )

        with tab_metrics:
            render_tradeoff_radar(compare_df, css_df, dataset)
            st.divider()
            render_parallel_coordinates(compare_df, dataset)
            st.divider()
            render_baseline_difference_chart(compare_df, dataset)

        with tab_vars:
            render_decision_variable_matrix(compare_df, dataset)
            st.divider()
            render_decision_variable_distribution(css_df, dataset)

        with tab_sim:
            render_solution_similarity_matrix(compare_df, dataset)

        with tab_mapping:
            render_variable_metric_correlation(compare_df, dataset)