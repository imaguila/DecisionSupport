"""
Indicator Lens Module.

Provides multi-criteria selection methods based on domain indicators:
1. Top-N Matches: Aggregates top solutions across individual target dimensions.
2. Non-Dominated Sorting: Identifies Pareto-optimal solutions within the enriched 
   indicator space.
"""

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st


# =====================================================
# UI RENDERING
# =====================================================


def render_params(
    dataset: Dict[str, Any], working_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Renders Streamlit UI controls for indicator lens options.

    Parameters
    ----------
    dataset : Dict[str, Any]
        Global dataset configuration metadata containing metrics and indicators.
    working_df : pd.DataFrame
        Current working solution space DataFrame.

    Returns
    -------
    Dict[str, Any]
        Dictionary of user-selected criteria and algorithm parameters.
    """
    dimensions = dataset.get("metrics", []) + dataset.get(
        "selected_indicators", []
    )
    indicators = dataset.get("selected_indicators", [])

    params: Dict[str, Any] = {}
    max_n = max(len(working_df) if working_df is not None else 0, 1)
    default_n = min(5, max_n)

    if not dimensions:
        st.info(
            "No dimensions are currently available. "
            "Select objectives or enable indicators first."
        )
        params["method"] = "Top-N Matches"
        params["maximize"] = []
        params["minimize"] = []
        params["top_n"] = default_n
        return params

    params["method"] = st.selectbox(
        "Indicator Method",
        ["Top-N Matches", "Non-dominated"],
        key="indicator_method",
    )

    if params["method"] == "Top-N Matches":
        available_criteria = dimensions
        st.caption(
            "Top-N Matches can use both original objectives and enriched indicators."
        )
    else:
        available_criteria = indicators
        if not available_criteria:
            st.info(
                "Non-dominated analysis currently uses enriched indicators. "
                "Enable indicators in Data Enrichment first."
            )
            params["maximize"] = []
            params["minimize"] = []
            params["top_n"] = None
            return params

        st.caption("Non-dominated analysis uses enriched indicators.")

    params["maximize"] = st.multiselect(
        "Dimensions to Maximize", available_criteria, key="indicator_maximize"
    )

    minimize_options = [
        c for c in available_criteria if c not in params["maximize"]
    ]

    params["minimize"] = st.multiselect(
        "Dimensions to Minimize", minimize_options, key="indicator_minimize"
    )

    if params["method"] == "Top-N Matches":
        params["top_n"] = st.slider(
            "Top N per Dimension", 1, max_n, default_n, key="indicator_top_n"
        )
        st.caption(
            "This method counts how often each solution appears "
            "among the best candidates for the selected dimensions."
        )
    else:
        params["top_n"] = None
        st.caption(
            "This method keeps solutions that are not clearly "
            "outperformed within the selected enriched-indicator space."
        )

    return params


# =====================================================
# HELPER FUNCTIONS
# =====================================================


def _sanitize_criteria(
    df: pd.DataFrame, maximize: List[str], minimize: List[str]
) -> Tuple[List[str], List[str], List[str]]:
    """
    Validates criteria column existence within the input DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Target DataFrame to sanitize criteria against.
    maximize : List[str]
        List of target metric names to maximize.
    minimize : List[str]
        List of target metric names to minimize.

    Returns
    -------
    Tuple[List[str], List[str], List[str]]
        Sanitized maximize, minimize, and combined criteria lists.
    """
    valid_max = [m for m in maximize if m in df.columns]
    valid_min = [
        m for m in minimize if m in df.columns and m not in valid_max
    ]
    criteria = valid_max + valid_min
    return valid_max, valid_min, criteria


def _build_group_labels_from_count(
    result: pd.DataFrame, count_column: str
) -> pd.DataFrame:
    """
    Generates categorical grouping labels based on indicator match counts.

    Parameters
    ----------
    result : pd.DataFrame
        DataFrame containing match counts.
    count_column : str
        Target column containing numerical match count.

    Returns
    -------
    pd.DataFrame
        Enriched DataFrame with `group_base` and `group_label` columns.
    """
    result["group_base"] = result[count_column].apply(
        lambda count: f"Matches = {count}"
    )
    group_sizes = result["group_base"].value_counts().to_dict()

    result["group_label"] = result["group_base"].apply(
        lambda grp: f"{grp} (n={group_sizes[grp]})"
    )
    return result


# =====================================================
# METHOD ENGINES
# =====================================================


def _apply_top_n_matches(
    df: pd.DataFrame, maximize: List[str], minimize: List[str], top_n: int
) -> pd.DataFrame:
    """Computes Top-N match counts per solution across selected criteria."""
    result = df.copy()
    criteria = maximize + minimize

    if not criteria:
        return result

    effective_top_n = min(top_n, len(result))
    ranked_subsets: List[pd.DataFrame] = []

    for metric in maximize:
        sub = (
            result.sort_values(metric, ascending=False)
            .head(effective_top_n)[["id"]]
            .assign(matched_metric=metric, goal="Maximize")
        )
        ranked_subsets.append(sub)

    for metric in minimize:
        sub = (
            result.sort_values(metric, ascending=True)
            .head(effective_top_n)[["id"]]
            .assign(matched_metric=metric, goal="Minimize")
        )
        ranked_subsets.append(sub)

    if not ranked_subsets:
        return result

    matches = pd.concat(ranked_subsets, ignore_index=True)

    counts = (
        matches.groupby("id")
        .size()
        .reset_index(name="domain_match_count")
    )

    matched_metrics = (
        matches.groupby("id")["matched_metric"]
        .apply(lambda vals: ", ".join(sorted(set(vals))))
        .reset_index(name="domain_matched_metrics")
    )

    result = result.merge(counts, on="id", how="left").merge(
        matched_metrics, on="id", how="left"
    )

    result["domain_match_count"] = (
        result["domain_match_count"].fillna(0).astype(int)
    )
    result["domain_matched_metrics"] = result[
        "domain_matched_metrics"
    ].fillna("")

    result = result[result["domain_match_count"] > 0].copy()

    if result.empty:
        return result

    result = _build_group_labels_from_count(result, "domain_match_count")
    result = result.sort_values(
        ["domain_match_count", "id"], ascending=[False, True]
    ).copy()

    result["domain_rank"] = range(1, len(result) + 1)
    result["indicator_method"] = "Top-N Matches"
    result["indicator_top_n"] = effective_top_n

    return result


def _apply_non_dominated(
    df: pd.DataFrame, maximize: List[str], minimize: List[str]
) -> pd.DataFrame:
    """Filters solutions to retain only Pareto non-dominated candidates."""
    result = df.copy()
    criteria = maximize + minimize

    if not criteria:
        return result

    work = result[criteria].copy()

    # Invert minimize metrics to convert problem strictly to maximization
    for metric in minimize:
        work[metric] = -work[metric]

    values = work.to_numpy()
    n_samples = len(values)
    is_nondominated = np.ones(n_samples, dtype=bool)

    # Pairwise non-dominance check
    for i in range(n_samples):
        current = values[i]
        for j in range(n_samples):
            if i == j:
                continue
            challenger = values[j]

            # Dominance test: challenger is >= in all and > in at least one
            if np.all(challenger >= current) and np.any(challenger > current):
                is_nondominated[i] = False
                break

    result["indicator_nondominated"] = is_nondominated
    result = result[result["indicator_nondominated"]].copy()

    if result.empty:
        return result

    result["indicator_method"] = "Non-dominated"
    result["domain_match_count"] = len(criteria)
    result["domain_matched_metrics"] = ", ".join(criteria)
    result["group_base"] = "Non-dominated"
    result["group_label"] = f"Non-dominated (n={len(result)})"

    result = result.sort_values("id", ascending=True).copy()
    result["domain_rank"] = range(1, len(result) + 1)

    return result


# =====================================================
# MAIN PIPELINE ENTRY POINT
# =====================================================


def apply(
    df: pd.DataFrame, params: Dict[str, Any], dataset: Dict[str, Any]
) -> pd.DataFrame:
    """
    Applies selected indicator lens method to isolate solutions.

    Parameters
    ----------
    df : pd.DataFrame
        Input working solution space DataFrame.
    params : Dict[str, Any]
        Indicator lens setup parameters.
    dataset : Dict[str, Any]
        Global context dataset metadata.

    Returns
    -------
    pd.DataFrame
        Filtered and metadata-enriched solution space DataFrame.
    """
    if df is None or df.empty:
        return df

    result = df.copy()
    maximize, minimize, criteria = _sanitize_criteria(
        result,
        params.get("maximize", []),
        params.get("minimize", []),
    )

    if not criteria:
        return result

    method = params.get("method", "Top-N Matches")

    if method == "Top-N Matches":
        top_n = params.get("top_n", min(5, len(result)))
        return _apply_top_n_matches(result, maximize, minimize, top_n)

    if method == "Non-dominated":
        return _apply_non_dominated(result, maximize, minimize)

    return result


# =====================================================
# FEEDBACK UI
# =====================================================


def render_feedback(lens_df: pd.DataFrame) -> None:
    """
    Displays UI summary metadata when indicator lens filtering is active.

    Parameters
    ----------
    lens_df : pd.DataFrame
        Filtered DataFrame output from the active indicator lens.
    """
    if lens_df is None or lens_df.empty:
        st.warning("No indicator matches found.")
        return

    if "indicator_method" in lens_df.columns:
        method = lens_df["indicator_method"].dropna().iloc[0]
        st.info(f"Indicator method: {method}")

    if "domain_match_count" in lens_df.columns:
        max_matches = lens_df["domain_match_count"].max()
        st.caption(f"Maximum indicator matches: {int(max_matches)}")

    if "domain_matched_metrics" in lens_df.columns:
        st.caption("Solutions are grouped by matched indicators.")

    if "indicator_nondominated" in lens_df.columns:
        st.caption(f"Non-dominated solutions: {len(lens_df)}")


# =====================================================
# BACKWARD COMPATIBILITY
# =====================================================


def apply_domain_lens(
    df: pd.DataFrame, maximize: List[str], minimize: List[str], top_n: int
) -> pd.DataFrame:
    """
    Legacy entry point for Top-N Domain match filtering.

    Parameters
    ----------
    df : pd.DataFrame
        Input solution space DataFrame.
    maximize : List[str]
        Metrics to maximize.
    minimize : List[str]
        Metrics to minimize.
    top_n : int
        Top N cut-off per metric.

    Returns
    -------
    pd.DataFrame
        Ranked and filtered DataFrame.
    """
    if df is None or df.empty:
        return df

    valid_max, valid_min, criteria = _sanitize_criteria(
        df, maximize, minimize
    )

    if not criteria:
        return df.copy()

    return _apply_top_n_matches(df, valid_max, valid_min, top_n)