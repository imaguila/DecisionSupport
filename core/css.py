"""
Candidate Solution Set (CSS) Core Analytics & Plotting Module.

Contains pure data processing, matrix calculation, and Plotly figure generation 
decoupled from Streamlit. Suitable for Jupyter Notebooks, batch scripts, and automated pipelines.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# =====================================================
# DATA EXTRACTION & UTILITY FUNCTIONS
# =====================================================


def get_numeric_dimensions(
    df: pd.DataFrame, dataset: Dict[str, Any]
) -> List[str]:
    """Extracts numeric objective and indicator column names present in the DataFrame."""
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
    """Retrieves decision variable column names matching the configured variable prefix."""
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
    """Normalizes a numeric Pandas Series to [0.0, 1.0] based on optimization goal."""
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


def sanitize_ids(ids: List[Any], valid_ids: List[Any]) -> List[Any]:
    """Filters an ID list to retain only IDs present in valid_ids."""
    if not ids or not valid_ids:
        return []

    valid_set = set(valid_ids)
    return [solution_id for solution_id in ids if solution_id in valid_set]


def apply_css_filtering(
    df: pd.DataFrame,
    enabled: bool = False,
    source: str = "Current set",
    manual_ids: Optional[List[Any]] = None,
    highlight_ids: Optional[List[Any]] = None,
) -> pd.DataFrame:
    """Filters DataFrame according to CSS rules independently of Streamlit."""
    if df is None or df.empty:
        return pd.DataFrame()

    css_df = df.copy()

    if not enabled:
        css_df["highlight"] = False
        return css_df

    if source == "Manual selection" and manual_ids is not None:
        css_df = css_df[css_df["id"].isin(manual_ids)].copy()

    if "id" in css_df.columns and highlight_ids:
        css_df["highlight"] = css_df["id"].isin(highlight_ids)
    else:
        css_df["highlight"] = False

    return css_df


# =====================================================
# ANALYTICS & PLOTLY FIGURE GENERATORS
# =====================================================


def create_tradeoff_radar_fig(
    compare_df: pd.DataFrame,
    selected_metrics: List[str],
    metric_goals: Dict[str, str],
) -> go.Figure:
    """Creates a polar radar chart comparing normalized solution profiles."""
    radar_df = compare_df.copy()

    for metric in selected_metrics:
        goal = metric_goals.get(metric, "Maximize")
        radar_df[metric] = normalize_metric(radar_df[metric], goal)

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
    return fig


def create_parallel_coordinates_fig(
    compare_df: pd.DataFrame, selected_dims: List[str]
) -> go.Figure:
    """Creates a Parallel Coordinates plot mapping multi-dimensional solution tradeoffs."""
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
    return fig


def compute_baseline_differences(
    compare_df: pd.DataFrame, baseline_id: Any, selected_metrics: List[str]
) -> pd.DataFrame:
    """Calculates relative and absolute differences relative to a baseline solution."""
    if compare_df.empty or baseline_id is None or not selected_metrics:
        return pd.DataFrame()

    baseline_rows = compare_df[compare_df["id"] == baseline_id]
    if baseline_rows.empty:
        return pd.DataFrame()

    baseline_row = baseline_rows.iloc[0]
    other_df = compare_df[compare_df["id"] != baseline_id].copy()

    if other_df.empty:
        return pd.DataFrame()

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

    return pd.DataFrame(diff_data)


def create_baseline_difference_fig(diff_df: pd.DataFrame) -> go.Figure:
    """Creates a percentage difference bar chart."""
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
    return fig


def compute_similarity_matrix(
    compare_df: pd.DataFrame, var_cols: List[str]
) -> pd.DataFrame:
    """Computes pairwise correlation matrix on decision variables."""
    matrix_df = compare_df.set_index("id")[var_cols]
    sim_matrix = matrix_df.T.corr().fillna(0.0)
    sim_matrix.index = [f"ID {int(i)}" for i in sim_matrix.index]
    sim_matrix.columns = [f"ID {int(c)}" for c in sim_matrix.columns]
    return sim_matrix


def create_solution_similarity_fig(sim_matrix: pd.DataFrame) -> go.Figure:
    """Creates a solution similarity heatmap from a correlation matrix."""
    fig = px.imshow(
        sim_matrix,
        text_auto=".2f",
        color_continuous_scale="RdBu",
        zmin=-1.0,
        zmax=1.0,
        labels=dict(color="Correlation"),
    )
    fig.update_layout(template="plotly_white", height=450)
    return fig


def create_decision_variable_matrix_fig(
    compare_df: pd.DataFrame, variable_cols: List[str]
) -> go.Figure:
    """Creates a structural heatmap matrix of decision variables per candidate solution."""
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
    return fig


def compute_decision_variable_summary(
    css_df: pd.DataFrame, variable_cols: List[str]
) -> pd.DataFrame:
    """Computes mean selection/activation rates across decision variables."""
    variable_summary = css_df[variable_cols].mean().reset_index()
    variable_summary.columns = ["decision_variable", "selection_rate"]
    return variable_summary.sort_values("selection_rate", ascending=False)


def create_decision_variable_distribution_fig(
    plot_df: pd.DataFrame,
) -> go.Figure:
    """Creates a bar chart of decision variable activation rates."""
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
    return fig


def compute_variable_metric_correlation(
    compare_df: pd.DataFrame, var_cols: List[str], metric_cols: List[str]
) -> pd.DataFrame:
    """Computes X -> Y correlation matrix between variables and metrics."""
    combined_df = compare_df[var_cols + metric_cols]
    corr_matrix = combined_df.corr()
    return corr_matrix.loc[var_cols, metric_cols].dropna(how="all").fillna(0.0)


def create_variable_metric_correlation_fig(
    xy_corr: pd.DataFrame, var_cols_count: int
) -> go.Figure:
    """Creates an X -> Y correlation heatmap."""
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
        height=max(400, var_cols_count * 20),
        xaxis=dict(tickangle=-45),
    )
    return fig