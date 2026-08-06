"""
Consensus Lens Module.

Aggregates multiple saved Sets of Interest (SOIs) into a unified consensus 
model using threshold-based voting logic, unions, majorities, or intersections.
"""

from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st


# =====================================================
# UI RENDERING
# =====================================================


def render_params(
    dataset: Dict[str, Any], working_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Renders Streamlit UI controls for selecting and combining saved SOIs.

    Parameters
    ----------
    dataset : Dict[str, Any]
        Global dataset configuration metadata.
    working_df : pd.DataFrame
        Current working solution space DataFrame.

    Returns
    -------
    Dict[str, Any]
        Dictionary of selected consensus methods, source SOIs, and thresholds.
    """
    params: Dict[str, Any] = {}
    saved_sois: List[Dict[str, Any]] = st.session_state.get("saved_sois", [])

    if len(saved_sois) < 2:
        st.info(
            "At least two saved SOIs are required to build a consensus SOI."
        )
        params["method"] = "Consensus Threshold"
        params["selected_sois"] = []
        params["threshold"] = 0.5
        return params

    soi_names = [soi["name"] for soi in saved_sois if "name" in soi]

    params["method"] = st.selectbox(
        "Consensus Method",
        [
            "Consensus Threshold",
            "Union",
            "Majority",
            "Intersection",
        ],
        key="consensus_method",
    )

    params["selected_sois"] = st.multiselect(
        "SOIs to Combine",
        soi_names,
        default=soi_names[: min(2, len(soi_names))],
        key="consensus_selected_sois",
    )

    n_selected = len(params["selected_sois"])

    if params["method"] == "Union":
        threshold = 1.0 / max(n_selected, 1)
        params["threshold"] = threshold
        st.caption(
            "Union keeps solutions supported by at least one selected SOI."
        )

    elif params["method"] == "Majority":
        params["threshold"] = 0.5
        st.caption(
            "Majority keeps solutions supported by at least half of the selected SOIs."
        )

    elif params["method"] == "Intersection":
        params["threshold"] = 1.0
        st.caption(
            "Intersection keeps only solutions supported by every selected SOI."
        )

    else:
        params["threshold"] = st.slider(
            "Consensus Level",
            0.0,
            1.0,
            0.5,
            0.05,
            key="consensus_threshold",
        )

        if params["threshold"] >= 0.75:
            st.caption("Mode: consensus core.")
        elif params["threshold"] >= 0.50:
            st.caption("Mode: consensus pool.")
        else:
            st.caption("Mode: broad exploratory pool.")

    st.caption(
        "This lens treats saved SOIs as expert opinions and combines them into one consensus SOI."
    )

    return params


# =====================================================
# HELPER FUNCTIONS
# =====================================================


def _get_selected_sois(selected_names: List[str]) -> List[Dict[str, Any]]:
    """
    Retrieves saved SOI dictionaries matching target selection names.

    Parameters
    ----------
    selected_names : List[str]
        List of target SOI names to fetch from session state.

    Returns
    -------
    List[Dict[str, Any]]
        Matching list of SOI data objects.
    """
    saved_sois: List[Dict[str, Any]] = st.session_state.get("saved_sois", [])
    return [soi for soi in saved_sois if soi.get("name") in selected_names]


def _build_support_table(selected_sois: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Computes solution support counts and consensus scores across selected SOIs.

    Parameters
    ----------
    selected_sois : List[Dict[str, Any]]
        List of selected SOI configuration dictionaries.

    Returns
    -------
    pd.DataFrame
        Support table mapping solution IDs to support count, score, and source list.
    """
    support: Dict[Any, int] = {}
    support_names: Dict[Any, List[str]] = {}

    for soi in selected_sois:
        soi_name = soi.get("name", "Unnamed SOI")
        unique_ids = set(soi.get("ids", []))

        for solution_id in unique_ids:
            support[solution_id] = support.get(solution_id, 0) + 1
            support_names.setdefault(solution_id, []).append(soi_name)

    rows = []
    n_sois = len(selected_sois)

    for solution_id, support_count in support.items():
        consensus_score = support_count / max(n_sois, 1)
        rows.append(
            {
                "id": solution_id,
                "consensus_support_count": support_count,
                "consensus_score": consensus_score,
                "consensus_supporting_sois": ", ".join(
                    sorted(support_names.get(solution_id, []))
                ),
            }
        )

    return pd.DataFrame(rows)


def _add_consensus_labels(
    result: pd.DataFrame, n_sois: int
) -> pd.DataFrame:
    """
    Appends consensus group base and group count label metadata to DataFrame.

    Parameters
    ----------
    result : pd.DataFrame
        DataFrame containing consensus support counts.
    n_sois : int
        Total number of evaluated source SOIs.

    Returns
    -------
    pd.DataFrame
        Enriched DataFrame with visual categorical label columns.
    """
    result["group_base"] = result["consensus_support_count"].apply(
        lambda count: f"Support = {int(count)}/{n_sois}"
    )

    group_sizes = result["group_base"].value_counts().to_dict()

    result["group_label"] = result["group_base"].apply(
        lambda grp: f"{grp} (n={group_sizes[grp]})"
    )

    return result


# =====================================================
# MAIN PIPELINE ENTRY POINT
# =====================================================


def apply(
    df: pd.DataFrame, params: Dict[str, Any], dataset: Dict[str, Any]
) -> pd.DataFrame:
    """
    Applies consensus voting filters across selected SOIs to retain valid solutions.

    Parameters
    ----------
    df : pd.DataFrame
        Input working solution space DataFrame.
    params : Dict[str, Any]
        Consensus algorithm configuration parameters.
    dataset : Dict[str, Any]
        Global context dataset metadata.

    Returns
    -------
    pd.DataFrame
        Filtered and metadata-enriched consensus DataFrame.
    """
    if df is None or df.empty:
        return df

    result = df.copy()
    selected_names = params.get("selected_sois", [])
    selected_sois = _get_selected_sois(selected_names)

    if len(selected_sois) < 2:
        result["consensus_warning"] = (
            "At least two SOIs are required for combination."
        )
        return result

    support_table = _build_support_table(selected_sois)

    if support_table.empty:
        result["consensus_warning"] = (
            "Selected SOIs do not contain any solution IDs."
        )
        return result

    threshold = params.get("threshold", 0.5)
    support_table = support_table[
        support_table["consensus_score"] >= threshold
    ].copy()

    if support_table.empty:
        empty_result = result.iloc[0:0].copy()
        empty_result["consensus_warning"] = (
            "No solutions satisfy the selected consensus threshold."
        )
        return empty_result

    result = result.merge(support_table, on="id", how="inner")
    n_sois = len(selected_sois)

    result = _add_consensus_labels(result, n_sois)
    result["consensus_method"] = params.get("method", "Consensus Threshold")
    result["consensus_threshold"] = threshold
    result["consensus_source_sois"] = ", ".join(selected_names)

    result = result.sort_values(
        ["consensus_score", "consensus_support_count", "id"],
        ascending=[False, False, True],
    ).copy()

    result["consensus_rank"] = range(1, len(result) + 1)

    return result


# =====================================================
# FEEDBACK UI
# =====================================================


def _safe_first_value(df: pd.DataFrame, column: str) -> Optional[Any]:
    """Extracts first valid scalar value from a target DataFrame column."""
    if column not in df.columns:
        return None
    values = df[column].dropna()
    if values.empty:
        return None
    return values.iloc[0]


def render_feedback(lens_df: Optional[pd.DataFrame]) -> None:
    """
    Displays UI summary indicators when the consensus lens is active.

    Parameters
    ----------
    lens_df : Optional[pd.DataFrame]
        Output DataFrame containing consensus evaluation metadata.
    """
    if lens_df is None:
        st.warning("No consensus result is available.")
        return

    warning_value = _safe_first_value(lens_df, "consensus_warning")
    if warning_value is not None:
        st.warning(warning_value)
        return

    if lens_df.empty:
        st.warning("The consensus SOI is empty.")
        return

    method = _safe_first_value(lens_df, "consensus_method")
    if method is not None:
        st.info(f"Consensus method: {method}")

    threshold = _safe_first_value(lens_df, "consensus_threshold")
    if threshold is not None:
        st.caption(f"Consensus threshold: {float(threshold):.2f}")

    if "consensus_score" in lens_df.columns:
        max_score = lens_df["consensus_score"].max()
        st.caption(f"Maximum consensus score: {float(max_score):.2f}")

    st.caption(f"Consensus SOI size: {len(lens_df)} solutions")