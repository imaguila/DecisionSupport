"""
Efficiency Lens Module.

Ranks candidate solutions based on benefit-cost trade-offs using raw ratios,
min-max normalized efficiency, composite cost aggregation, or Euclidean distance
to ideal target states in objective space.
"""

from typing import Any, Dict, List, Optional, Union

import pandas as pd
import streamlit as st

# Global small constant to prevent division by zero
EPS: float = 1e-9


# =====================================================
# UI RENDERING
# =====================================================


def render_params(
    dataset: Dict[str, Any], working_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Renders Streamlit UI controls for efficiency ranking parameters.

    Parameters
    ----------
    dataset : Dict[str, Any]
        Global dataset context containing metric and indicator keys.
    working_df : pd.DataFrame
        Current working solution space DataFrame.

    Returns
    -------
    Dict[str, Any]
        Dictionary of user-selected efficiency parameters.
    """
    dimensions = dataset.get("metrics", []) + dataset.get(
        "selected_indicators", []
    )
    params: Dict[str, Any] = {}
    max_n = max(len(working_df), 1)
    default_n = min(5, max_n)

    if len(dimensions) < 2:
        st.info(
            "At least two dimensions are required for the Efficiency lens."
        )
        params["method"] = "Benefit/Cost Ratio"
        params["benefit"] = None
        params["cost"] = None
        params["top_n"] = default_n
        return params

    params["method"] = st.selectbox(
        "Efficiency Method",
        [
            "Benefit/Cost Ratio",
            "Normalized Ratio",
            "Distance to Ideal",
            "Composite Cost Ratio",
        ],
        key="eff_method",
    )

    params["benefit"] = st.selectbox(
        "Benefit Metric", dimensions, key="eff_benefit"
    )

    cost_options = [d for d in dimensions if d != params["benefit"]]

    if params["method"] == "Composite Cost Ratio":
        params["cost"] = st.multiselect(
            "Cost Metrics",
            cost_options,
            default=cost_options[: min(2, len(cost_options))],
            key="eff_costs",
        )
    else:
        params["cost"] = st.selectbox(
            "Cost Metric", cost_options, key="eff_cost"
        )

    params["top_n"] = st.slider(
        "Top N Solutions", 1, max_n, default_n, key="eff_top_n"
    )

    st.caption(
        "Efficiency methods rank solutions by benefit-cost trade-off."
    )

    return params


# =====================================================
# HELPER FUNCTIONS
# =====================================================


def _normalize_series(series: pd.Series) -> pd.Series:
    """
    Normalizes a numeric pandas Series to the range [0.0, 1.0] via Min-Max scaling.

    Parameters
    ----------
    series : pd.Series
        Numeric input series to normalize.

    Returns
    -------
    pd.Series
        Min-Max normalized series, or zeros if min equals max.
    """
    min_v = series.min()
    max_v = series.max()

    if max_v > min_v:
        return (series - min_v) / (max_v - min_v)

    return pd.Series(0.0, index=series.index)


def _resolve_cost_metrics(
    result: pd.DataFrame,
    benefit: str,
    cost: Optional[Union[str, List[str]]],
) -> List[str]:
    """
    Resolves and validates cost metric column names present in the DataFrame.

    Parameters
    ----------
    result : pd.DataFrame
        Input working solution space DataFrame.
    benefit : str
        Selected benefit metric name.
    cost : Optional[Union[str, List[str]]]
        Single cost metric name or list of cost metric names.

    Returns
    -------
    List[str]
        Filtered list of valid cost column names excluding the benefit metric.
    """
    if cost is None:
        return []

    if isinstance(cost, str):
        cost_metrics = [cost]
    else:
        cost_metrics = [c for c in cost if c in result.columns]

    return [c for c in cost_metrics if c != benefit]


# =====================================================
# SCORE METHOD ENGINES
# =====================================================


def _benefit_cost_ratio(
    result: pd.DataFrame, benefit: str, cost_metrics: List[str]
) -> pd.Series:
    """Calculates unnormalized Benefit / Cost ratio."""
    cost_metric = cost_metrics[0]
    safe_cost = result[cost_metric].replace(0, EPS)
    return result[benefit] / safe_cost


def _normalized_ratio(
    result: pd.DataFrame, benefit: str, cost_metrics: List[str]
) -> pd.Series:
    """Calculates Min-Max normalized Benefit / Normalized Cost ratio."""
    cost_metric = cost_metrics[0]
    benefit_norm = _normalize_series(result[benefit])
    cost_norm = _normalize_series(result[cost_metric])
    return benefit_norm / (cost_norm + EPS)


def _distance_to_ideal(
    result: pd.DataFrame, benefit: str, cost_metrics: List[str]
) -> pd.Series:
    """
    Calculates proximity score based on Euclidean distance to ideal state (1.0 benefit, 0.0 cost).
    """
    cost_metric = cost_metrics[0]
    benefit_norm = _normalize_series(result[benefit])
    cost_norm = _normalize_series(result[cost_metric])

    distance_to_ideal = (
        (1.0 - benefit_norm) ** 2 + (cost_norm) ** 2
    ) ** 0.5
    max_distance = 2.0**0.5

    return 1.0 - (distance_to_ideal / max_distance)


def _composite_cost_ratio(
    result: pd.DataFrame, benefit: str, cost_metrics: List[str]
) -> pd.Series:
    """Calculates normalized Benefit / Average Composite Normalized Costs ratio."""
    benefit_norm = _normalize_series(result[benefit])
    composite_cost = pd.Series(0.0, index=result.index)

    for cost_metric in cost_metrics:
        composite_cost += _normalize_series(result[cost_metric])

    composite_cost /= len(cost_metrics)
    return benefit_norm / (composite_cost + EPS)


# =====================================================
# MAIN PIPELINE ENTRY POINT
# =====================================================


def apply(
    df: pd.DataFrame, params: Dict[str, Any], dataset: Dict[str, Any]
) -> pd.DataFrame:
    """
    Applies selected efficiency lens method to calculate score and rank solutions.

    Parameters
    ----------
    df : pd.DataFrame
        Input solution space DataFrame.
    params : Dict[str, Any]
        Efficiency configuration parameters.
    dataset : Dict[str, Any]
        Global context dataset metadata.

    Returns
    -------
    pd.DataFrame
        Ranked top N subset of solutions enriched with efficiency scores.
    """
    if df is None or df.empty:
        return df

    result = df.copy()
    method = params.get("method", "Benefit/Cost Ratio")
    benefit = params.get("benefit")
    cost = params.get("cost")

    if benefit is None or benefit not in result.columns:
        return result

    cost_metrics = _resolve_cost_metrics(result, benefit, cost)
    if not cost_metrics:
        return result

    top_n = min(params.get("top_n", len(result)), len(result))

    if method == "Benefit/Cost Ratio":
        score = _benefit_cost_ratio(result, benefit, cost_metrics)
    elif method == "Normalized Ratio":
        score = _normalized_ratio(result, benefit, cost_metrics)
    elif method == "Distance to Ideal":
        score = _distance_to_ideal(result, benefit, cost_metrics)
    elif method == "Composite Cost Ratio":
        score = _composite_cost_ratio(result, benefit, cost_metrics)
        result["efficiency_costs"] = ", ".join(cost_metrics)
    else:
        return result

    result["efficiency_score"] = score
    result = result.sort_values(
        "efficiency_score", ascending=False
    ).copy()

    result["efficiency_rank"] = range(1, len(result) + 1)
    result["efficiency_method"] = method
    result["efficiency_benefit"] = benefit
    result["efficiency_primary_cost"] = cost_metrics[0]

    return result.head(top_n)


# =====================================================
# FEEDBACK UI
# =====================================================


def render_feedback(lens_df: pd.DataFrame) -> None:
    """
    Displays UI summary metadata and indicators for applied efficiency ranking.

    Parameters
    ----------
    lens_df : pd.DataFrame
        Filtered/ranked output DataFrame containing efficiency metadata.
    """
    if lens_df is None or lens_df.empty:
        st.warning("No efficiency results available.")
        return

    if "efficiency_method" in lens_df.columns:
        method = lens_df["efficiency_method"].dropna().iloc[0]
        st.info(f"Efficiency method: {method}")

    if "efficiency_benefit" in lens_df.columns:
        benefit = lens_df["efficiency_benefit"].dropna().iloc[0]
        st.caption(f"Benefit metric: {benefit}")

    if "efficiency_costs" in lens_df.columns:
        costs = lens_df["efficiency_costs"].dropna().iloc[0]
        st.caption(f"Composite costs: {costs}")
    elif "efficiency_primary_cost" in lens_df.columns:
        cost = lens_df["efficiency_primary_cost"].dropna().iloc[0]
        st.caption(f"Cost metric: {cost}")