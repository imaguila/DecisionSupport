"""
Preference Lens Module.

Implements Multi-Criteria Decision Making (MCDM) ranking algorithms to score
and order Pareto-optimal solutions based on Decision Maker (DM) preferences.
Supported methods: Weighted Sum, TOPSIS, VIKOR, and Reference Point.
"""

import logging
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)


def render_params(
    dataset: Dict[str, Any], working_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Renders UI controls for MCDM method selection and metric optimization goals.

    Parameters
    ----------
    dataset : Dict[str, Any]
        Global dataset configuration containing metric metadata.
    working_df : pd.DataFrame
        Current working solution space DataFrame.

    Returns
    -------
    Dict[str, Any]
        Dictionary of selected preference configuration parameters.
    """
    dimensions = dataset.get("metrics", []) + dataset.get(
        "selected_indicators", []
    )
    max_n = max(len(working_df), 1)
    default_n = min(5, max_n)

    params: Dict[str, Any] = {}

    params["method"] = st.selectbox(
        "Scoring Method",
        ["Weighted Sum", "TOPSIS", "VIKOR", "Reference Point"],
        key="pref_method",
    )

    st.caption("All preference methods currently assign equal weight to selected criteria.")

    params["maximize"] = st.multiselect(
        "Metrics to Maximize", dimensions, key="pref_maximize"
    )

    minimize_options = [
        d for d in dimensions if d not in params.get("maximize", [])
    ]

    params["minimize"] = st.multiselect(
        "Metrics to Minimize", minimize_options, key="pref_minimize"
    )

    params["top_n"] = st.slider(
        "Top N Solutions", 1, max_n, default_n, key="pref_top_n"
    )

    return params


def _sanitize_criteria(
    df: pd.DataFrame, maximize: List[str], minimize: List[str]
) -> Tuple[List[str], List[str], List[str]]:
    """
    Validates and cleans user-selected criteria columns against DataFrame schema.
    """
    valid_max = [m for m in maximize if m in df.columns]
    valid_min = [m for m in minimize if m in df.columns and m not in valid_max]
    return valid_max, valid_min, valid_max + valid_min


def _minmax_normalize(df: pd.DataFrame, criteria: List[str]) -> pd.DataFrame:
    """
    Applies Min-Max normalization across selected evaluation criteria.
    """
    norm = pd.DataFrame(index=df.index)
    for metric in criteria:
        min_v = df[metric].min()
        max_v = df[metric].max()
        if max_v > min_v:
            norm[metric] = (df[metric] - min_v) / (max_v - min_v)
        else:
            norm[metric] = 0.0
    return norm


def _weighted_sum(
    df: pd.DataFrame, maximize: List[str], minimize: List[str]
) -> pd.Series:
    """
    Calculates score via Weighted Sum Model (WSM).
    """
    criteria = maximize + minimize
    norm = _minmax_normalize(df, criteria)
    score = pd.Series(0.0, index=df.index)
    weight = 1.0 / len(criteria)

    for metric in criteria:
        if metric in maximize:
            val = norm[metric]
        else:
            val = 1.0 - norm[metric]
        score += weight * val

    return score


def _topsis(
    df: pd.DataFrame, maximize: List[str], minimize: List[str]
) -> pd.Series:
    """
    Vectorized TOPSIS (Technique for Order Preference by Similarity to Ideal Solution).
    """
    criteria = maximize + minimize
    vals = df[criteria].to_numpy(dtype=float)

    # Vectorized L2 Normalization
    norms = np.linalg.norm(vals, axis=0)
    norms[norms == 0] = 1.0
    norm_vals = (vals / norms) * (1.0 / len(criteria))

    ideal = np.zeros(len(criteria))
    anti_ideal = np.zeros(len(criteria))

    for idx, metric in enumerate(criteria):
        if metric in maximize:
            ideal[idx] = norm_vals[:, idx].max()
            anti_ideal[idx] = norm_vals[:, idx].min()
        else:
            ideal[idx] = norm_vals[:, idx].min()
            anti_ideal[idx] = norm_vals[:, idx].max()

    # Vectorized Euclidean Distances
    d_plus = np.linalg.norm(norm_vals - ideal, axis=1)
    d_minus = np.linalg.norm(norm_vals - anti_ideal, axis=1)

    denom = d_plus + d_minus
    scores = np.where(denom != 0, d_minus / denom, 0.0)

    return pd.Series(scores, index=df.index)


def _vikor(
    df: pd.DataFrame, maximize: List[str], minimize: List[str], v: float = 0.5
) -> pd.Series:
    """
    Calculates VIKOR compromise ranking index (Q).
    """
    criteria = maximize + minimize
    weight = 1.0 / len(criteria)
    regret = pd.DataFrame(index=df.index)

    for metric in criteria:
        if metric in maximize:
            best, worst = df[metric].max(), df[metric].min()
        else:
            best, worst = df[metric].min(), df[metric].max()

        denom = abs(best - worst)
        if denom == 0:
            regret[metric] = 0.0
        else:
            regret[metric] = weight * abs(best - df[metric]) / denom

    s_value = regret.sum(axis=1)
    r_value = regret.max(axis=1)

    s_range = s_value.max() - s_value.min()
    s_norm = (s_value - s_value.min()) / s_range if s_range > 0 else 0.0

    r_range = r_value.max() - r_value.min()
    r_norm = (r_value - r_value.min()) / r_range if r_range > 0 else 0.0

    q_value = v * s_norm + (1.0 - v) * r_norm
    return 1.0 - q_value  # Higher score implies better rank


def _reference_point(
    df: pd.DataFrame, maximize: List[str], minimize: List[str]
) -> pd.Series:
    """
    Vectorized distance to ideal reference point (1.0 in normalized objective space).
    """
    criteria = maximize + minimize
    norm = _minmax_normalize(df, criteria)
    oriented = pd.DataFrame(index=df.index)

    for metric in criteria:
        if metric in maximize:
            oriented[metric] = norm[metric]
        else:
            oriented[metric] = 1.0 - norm[metric]

    # Vectorized Euclidean Distance to Ideal Point [1, 1, ..., 1]
    oriented_vals = oriented.to_numpy(dtype=float)
    distances = np.linalg.norm(1.0 - oriented_vals, axis=1)

    max_dist = distances.max()
    if max_dist > 0:
        scores = 1.0 - (distances / max_dist)
    else:
        scores = np.ones(len(df))

    return pd.Series(scores, index=df.index)


def apply(
    df: pd.DataFrame, params: Dict[str, Any], dataset: Dict[str, Any]
) -> pd.DataFrame:
    """
    Applies the specified preference lens and ranks solutions accordingly.

    Parameters
    ----------
    df : pd.DataFrame
        Input decision space candidate solutions.
    params : Dict[str, Any]
        Configuration mapping including optimization directions and method.
    dataset : Dict[str, Any]
        Global context dataset metadata.

    Returns
    -------
    pd.DataFrame
        Ranked and truncated DataFrame containing top N solutions.
    """
    if df is None or df.empty:
        return df

    result = df.copy()
    maximize, minimize, criteria = _sanitize_criteria(
        result, params.get("maximize", []), params.get("minimize", [])
    )

    if not criteria:
        return result

    method = params.get("method", "Weighted Sum")
    top_n = min(params.get("top_n", len(result)), len(result))

    if method == "Weighted Sum":
        score = _weighted_sum(result, maximize, minimize)
    elif method == "TOPSIS":
        score = _topsis(result, maximize, minimize)
    elif method == "VIKOR":
        score = _vikor(result, maximize, minimize)
    elif method == "Reference Point":
        score = _reference_point(result, maximize, minimize)
    else:
        return result

    result["preference_score"] = score
    result = result.sort_values("preference_score", ascending=False).copy()
    result["preference_rank"] = range(1, len(result) + 1)
    result["preference_method"] = method

    return result.head(top_n)


def render_feedback(lens_df: pd.DataFrame) -> None:
    """
    Displays UI feedback summarizing the applied preference scoring.
    """
    if lens_df is None or lens_df.empty:
        return

    if "preference_method" in lens_df.columns:
        method = lens_df["preference_method"].dropna().iloc[0]
        st.info(f"Preference method applied: **{method}**")

    if "preference_score" in lens_df.columns:
        st.caption("Solutions ranked and sorted by highest `preference_score`.")