"""
Candidate Solution Set (CSS) Comparison Module.

Provides visual trade-off analysis, structural decision-variable inspection, 
parallel coordinate mapping, baseline difference metrics, and X -> Y 
correlation heatmaps for candidate solution subsets.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# =====================================================
# HELPER FUNCTIONS
# =====================================================


def get_numeric_dimensions(
    df: pd.DataFrame, dataset: Dict[str, Any]
) -> List[str]:
    """
    Extracts numeric objective and indicator column names present in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Active solution space DataFrame.
    dataset : Dict[str, Any]
        Global dataset context dictionary.

    Returns
    -------
    List[str]
        List of verified numeric metric and indicator column names.
    """
    if df is None or df.empty or not dataset:
        return []

    metrics = dataset.get("metrics", []) or []
    indicators = dataset.get("selected_indicators", []) or []
    dimensions = list(metrics) + list(indicators)

    return [
        col
        for col in dimensions
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
    ]


def get_decision_variable_columns(
    df: pd.DataFrame, dataset: Dict[str, Any]
) -> List[str]:
    """
    Retrieves decision variable column names matching the configured variable prefix.

    Parameters
    ----------
    df : pd.DataFrame
        Active solution space DataFrame.
    dataset : Dict[str, Any]
        Global dataset context dictionary.

    Returns
    -------
    List[str]
        List of numeric decision variable column names.
    """
    if df is None or df.empty or not dataset:
        return []

    config = dataset.get("config", {}) if dataset else {}
    var_prefix = config.get("var_prefix", "x_")

    return [
        col
        for col in df.columns
        if var_prefix
        and col.startswith(var_prefix)
        and pd.api.types.is_numeric_dtype(df[col])
    ]


def normalize_metric(series: pd.Series, goal: str) -> pd.Series:
    """
    Normalizes a numeric Pandas Series to the range [0.0, 1.0] based on optimization goal.

    Parameters
    ----------
    series : pd.Series
        Numeric metric values to normalize.
    goal : str
        Optimization goal ("Maximize" or "Minimize").

    Returns
    -------
    pd.Series
        Normalized metric values scaled from 0.0 to 1.0.
    """
    if series.empty:
        return series

    min_v = series.min()
    max_v = series.max()

    if max_v <= min_v:
        return pd.Series(0.5, index=series.index)

    normalized = (series - min_v) / (max_v - min_v)

    if goal == "Minimize":
        normalized = 1.0 - normalized

    return normalized


# =====================================================
# TRADE-OFF RADAR CHART
# =====================================================


def render_tradeoff_radar(
    compare_df: pd.DataFrame, css_df: pd.DataFrame, dataset: Dict[str, Any]
) -> None:
    """
    Renders a polar radar chart comparing normalized solution profiles across objectives.

    Parameters
    ----------
    compare_df : pd.DataFrame
        DataFrame containing selected solutions to compare.
    css_df : pd.DataFrame
        Full Candidate Solution Set DataFrame used for dimension discovery.
    dataset : Dict[str, Any]
        Global dataset context dictionary.
    """
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

    radar_df = compare_df.copy()

    for metric in selected_metrics:
        radar_df[metric] = normalize_metric(
            radar_df[metric], metric_goals[metric]
        )

    fig = go.Figure()

    for _, row in radar_df.iterrows():
        values = row[selected_metrics].tolist()
        values.append(values[0])
        theta = selected_metrics + [selected_metrics[0]]

        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=theta,
                mode="lines+markers",
                name=f"ID {int(row['id'])}",
            )
        )

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        template="plotly_white",
        height=450,
        margin=dict(t=40, b=40),
    )

    st.plotly_chart(fig, use_container_width=True)


# =====================================================
# PARALLEL COORDINATES
# =====================================================


def render_parallel_coordinates(
    compare_df: pd.DataFrame, dataset: Dict[str, Any]
) -> None:
    """
    Renders a Parallel Coordinates plot mapping multi-dimensional solution tradeoffs.

    Parameters
    ----------
    compare_df : pd.DataFrame
        DataFrame containing selected solutions to compare.
    dataset : Dict[str, Any]
        Global dataset context dictionary.
    """
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

    dimensions_config = [
        dict(
            range=[compare_df[col].min(), compare_df[col].max()],
            label=col,
            values=compare_df[col],
        )
        for col in selected_dims
    ]

    fig = go.Figure(
        data=go.Parcoords(
            line=dict(
                color=compare_df["id"],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Solution ID"),
            ),
            dimensions=dimensions_config,
        )
    )

    fig.update_layout(
        template="plotly_white", height=400, margin=dict(t=40, b=40)
    )
    st.plotly_chart(fig, use_container_width=True)


# =====================================================
# BASELINE DIFFERENCE / GAP ANALYSIS
# =====================================================


def render_baseline_difference_chart(
    compare_df: pd.DataFrame, dataset: Dict[str, Any]
) -> None:
    """
    Renders a relative percentage difference bar chart relative to a selected baseline solution.

    Parameters
    ----------
    compare_df : pd.DataFrame
        DataFrame containing selected solutions to compare.
    dataset : Dict[str, Any]
        Global dataset context dictionary.
    """
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

    baseline_row = compare_df[compare_df["id"] == baseline_id].iloc[0]
    other_df = compare_df[compare_df["id"] != baseline_id].copy()

    if other_df.empty:
        st.info(
            "Select at least one additional solution to compare against the Baseline."
        )
        return

    diff_data = []
    for _, row in other_df.iterrows():
        for metric in selected_metrics:
            base_val = baseline_row[metric]
            curr_val = row[metric]

            if base_val != 0:
                pct_change = ((curr_val - base_val) / abs(base_val)) * 100
            else:
                pct_change = 0.0 if curr_val == 0 else np.nan

            diff_data.append(
                {
                    "Solution": f"ID {int(row['id'])}",
                    "Metric": metric,
                    "Relative Change (%)": pct_change,
                    "Absolute Difference": curr_val - base_val,
                }
            )

    diff_df = pd.DataFrame(diff_data)

    fig = px.bar(
        diff_df,
        x="Metric",
        y="Relative Change (%)",
        color="Solution",
        barmode="group",
        hover_data=["Absolute Difference"],
    )

    fig.add_hline(y=0, line_dash="dash", line_color="black")
    fig.update_layout(template="plotly_white", height=400)
    st.plotly_chart(fig, use_container_width=True)


# =====================================================
# SOLUTION SIMILARITY MATRIX
# =====================================================


def render_solution_similarity_matrix(
    compare_df: pd.DataFrame, dataset: Dict[str, Any]
) -> None:
    """
    Computes and displays pairwise decision-variable similarity correlation between solutions.

    Parameters
    ----------
    compare_df : pd.DataFrame
        DataFrame containing selected solutions to compare.
    dataset : Dict[str, Any]
        Global dataset context dictionary.
    """
    var_cols = get_decision_variable_columns(compare_df, dataset)
    if not var_cols or len(compare_df) < 2:
        st.info(
            "Requires decision variables and at least 2 solutions to compute similarity."
        )
        return

    matrix_df = compare_df.set_index("id")[var_cols]
    sim_matrix = matrix_df.T.corr().fillna(0.0)

    sim_matrix.index = [f"ID {int(i)}" for i in sim_matrix.index]
    sim_matrix.columns = [f"ID {int(c)}" for c in sim_matrix.columns]

    fig = px.imshow(
        sim_matrix,
        text_auto=".2f",
        color_continuous_scale="RdBu",
        zmin=-1.0,
        zmax=1.0,
        labels=dict(color="Correlation"),
    )

    fig.update_layout(template="plotly_white", height=450)
    st.plotly_chart(fig, use_container_width=True)


# =====================================================
# DECISION-VARIABLE MATRIX
# =====================================================


def render_decision_variable_matrix(
    compare_df: pd.DataFrame, dataset: Dict[str, Any]
) -> None:
    """
    Renders a structural heatmap matrix of decision variable values per candidate solution.

    Parameters
    ----------
    compare_df : pd.DataFrame
        DataFrame containing selected solutions to compare.
    dataset : Dict[str, Any]
        Global dataset context dictionary.
    """
    variable_cols = get_decision_variable_columns(compare_df, dataset)
    config = dataset.get("config", {}) if dataset else {}
    var_prefix = config.get("var_prefix", "x_")

    if not variable_cols:
        st.info(
            f"No numeric decision-variable columns with prefix '{var_prefix}' found."
        )
        return

    matrix_df = compare_df.set_index("id")[variable_cols].copy()
    matrix_df.index = [f"ID {int(idx)}" for idx in matrix_df.index]

    fig = px.imshow(
        matrix_df,
        labels=dict(x="Decision variables", y="Solutions", color="Value"),
        color_continuous_scale=[[0, "#e0e0e0"], [1, "#00e676"]],
    )

    fig.update_layout(
        template="plotly_white",
        coloraxis_showscale=False,
        xaxis=dict(tickangle=-45, showgrid=False),
        yaxis=dict(autorange="reversed", showgrid=False),
        height=520,
    )

    fig.update_traces(
        xgap=3,
        ygap=3,
        hovertemplate="<b>%{y}</b><br>Variable: %{x}<br>Value: %{z}<extra></extra>",
    )

    st.plotly_chart(fig, use_container_width=True)


# =====================================================
# DECISION-VARIABLE DISTRIBUTION
# =====================================================


def render_decision_variable_distribution(
    css_df: pd.DataFrame, dataset: Dict[str, Any]
) -> None:
    """
    Renders a bar chart summarizing average activation/selection rates across decision variables.

    Parameters
    ----------
    css_df : pd.DataFrame
        Full Candidate Solution Set DataFrame.
    dataset : Dict[str, Any]
        Global dataset context dictionary.
    """
    variable_cols = get_decision_variable_columns(css_df, dataset)
    config = dataset.get("config", {}) if dataset else {}
    var_prefix = config.get("var_prefix", "x_")

    if not variable_cols:
        st.info(
            f"No numeric decision-variable columns with prefix '{var_prefix}' found."
        )
        return

    variable_summary = css_df[variable_cols].mean().reset_index()
    variable_summary.columns = ["decision_variable", "selection_rate"]
    variable_summary = variable_summary.sort_values(
        "selection_rate", ascending=False
    )

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

    fig = px.bar(
        plot_df,
        x="decision_variable",
        y="selection_rate",
        labels={
            "decision_variable": "Decision variable",
            "selection_rate": "Mean Value / Selection rate",
        },
    )

    fig.update_layout(template="plotly_white", height=420, xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)


# =====================================================
# VARIABLE TO METRIC MAPPING (X vs Y Correlation)
# =====================================================


def render_variable_metric_correlation(
    compare_df: pd.DataFrame, dataset: Dict[str, Any]
) -> None:
    """
    Renders an X -> Y correlation matrix heatmap (Decision Variables vs. Metrics).

    Parameters
    ----------
    compare_df : pd.DataFrame
        DataFrame containing selected solutions to compare.
    dataset : Dict[str, Any]
        Global dataset context dictionary.
    """
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

    combined_df = compare_df[var_cols + metric_cols]
    corr_matrix = combined_df.corr()

    xy_corr = (
        corr_matrix.loc[var_cols, metric_cols].dropna(how="all").fillna(0.0)
    )

    if xy_corr.empty:
        st.info(
            "Could not calculate variance/correlation for the selected subset."
        )
        return

    fig = px.imshow(
        xy_corr,
        labels=dict(
            x="Metrics / Objectives (Y)",
            y="Decision Variables (X)",
            color="Correlation",
        ),
        color_continuous_scale="RdBu",
        zmin=-1.0,
        zmax=1.0,
        aspect="auto",
    )

    fig.update_layout(
        template="plotly_white",
        height=max(400, len(var_cols) * 20),
        xaxis=dict(tickangle=-45),
    )

    st.plotly_chart(fig, use_container_width=True)


# =====================================================
# MAIN CSS COMPARISON PIPELINE
# =====================================================


def render_css_comparison(
    css_df: pd.DataFrame, dataset: Dict[str, Any]
) -> None:
    """
    Main entry point for rendering the detailed Candidate Solution Set (CSS) comparison panel.

    Parameters
    ----------
    css_df : pd.DataFrame
        Active Candidate Solution Set DataFrame.
    dataset : Dict[str, Any]
        Global dataset context dictionary.
    """
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