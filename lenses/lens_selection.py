"""
Lens Selection & Solution Grouping Module.

Provides state management, group filtering, and persistence mechanisms
for candidate Solutions of Interest (SOIs) extracted through analytical lenses.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

logger = logging.getLogger(__name__)


def ensure_soi_state() -> None:
    """
    Ensures that the state variable for saved SOIs exists in the Streamlit session.
    """
    if "saved_sois" not in st.session_state:
        st.session_state.saved_sois = []


def get_group_column(lens_df: Optional[pd.DataFrame]) -> Optional[str]:
    """
    Identifies the primary grouping or clustering column present in a DataFrame.

    Parameters
    ----------
    lens_df : Optional[pd.DataFrame]
        DataFrame transformed by an analytical lens.

    Returns
    -------
    Optional[str]
        Column name used for grouping ('group_label' or 'cluster_str'), or None if missing.
    """
    if lens_df is None:
        return None

    if "group_label" in lens_df.columns:
        return "group_label"

    if "cluster_str" in lens_df.columns:
        return "cluster_str"

    return None


def get_group_options(
    lens_df: pd.DataFrame, group_column: Optional[str]
) -> List[str]:
    """
    Extracts sorted unique string values from the specified grouping column.

    Parameters
    ----------
    lens_df : pd.DataFrame
        DataFrame containing solution data.
    group_column : Optional[str]
        Target grouping column name.

    Returns
    -------
    List[str]
        List of unique string representations of available group labels.
    """
    if group_column is None or group_column not in lens_df.columns:
        return []

    return sorted(
        lens_df[group_column].dropna().astype(str).unique().tolist()
    )


def filter_by_group(
    lens_df: Optional[pd.DataFrame],
    group_column: Optional[str],
    group_value: str,
) -> Optional[pd.DataFrame]:
    """
    Filters a DataFrame by a specified group label or value.

    Parameters
    ----------
    lens_df : Optional[pd.DataFrame]
        Input dataset to filter.
    group_column : Optional[str]
        Column name used for filtering.
    group_value : str
        Selected value to filter by ('All groups' returns an unmodified copy).

    Returns
    -------
    Optional[pd.DataFrame]
        Filtered copy of the input DataFrame.
    """
    if lens_df is None:
        return None

    if group_column is None or group_value == "All groups":
        return lens_df.copy()

    return lens_df[
        lens_df[group_column].astype(str) == str(group_value)
    ].copy()


def get_lens_label(active_lens: str) -> str:
    """
    Resolves the human-readable label for the currently active lens.

    Parameters
    ----------
    active_lens : str
        Active lens identifier.

    Returns
    -------
    str
        'Exploratory' if active_lens is "None", otherwise returns active_lens.
    """
    return "Exploratory" if active_lens == "None" else active_lens


def reset_soi_name_if_needed(active_lens: str, group_value: str) -> None:
    """
    Updates the session state's target SOI name when the active context changes.

    Parameters
    ----------
    active_lens : str
        Identifier of the current lens.
    group_value : str
        Selected group filtering value.
    """
    lens_label = get_lens_label(active_lens)
    suffix = group_value if group_value != "All groups" else "Current set"
    default_name = f"{lens_label} - {suffix} #{len(st.session_state.saved_sois) + 1}"

    name_context: Tuple[str, str] = (lens_label, group_value)

    if st.session_state.get("soi_name_context") != name_context:
        st.session_state["soi_name"] = default_name
        st.session_state["soi_name_context"] = name_context


def render_group_selector_and_save(
    placeholder: Optional[DeltaGenerator],
    active_lens: str,
    lens_df: Optional[pd.DataFrame],
    lens_params: Dict[str, Any],
) -> Optional[pd.DataFrame]:
    """
    Renders group filtering controls and save buttons for persisting SOIs.

    Parameters
    ----------
    placeholder : Optional[DeltaGenerator]
        Streamlit container placeholder for UI layout placement.
    active_lens : str
        Identifier of the active analytical lens.
    lens_df : Optional[pd.DataFrame]
        Transformed DataFrame containing candidate solutions.
    lens_params : Dict[str, Any]
        Configuration parameters associated with the current lens.

    Returns
    -------
    Optional[pd.DataFrame]
        The subset DataFrame selected by user interaction, or None if input is invalid.
    """
    ensure_soi_state()

    if placeholder is None or lens_df is None or lens_df.empty:
        return lens_df

    with placeholder.container():
        lens_label = get_lens_label(active_lens)
        group_column = get_group_column(lens_df)
        group_value = "All groups"

        if group_column is not None:
            group_options = get_group_options(lens_df, group_column)
            options = ["All groups"] + group_options

            selector_key = (
                f"soi_group_selector_{lens_label.replace(' ', '_')}"
            )

            if st.session_state.get(selector_key) not in options:
                st.session_state[selector_key] = "All groups"

            group_value = st.selectbox(
                "SOI group",
                options,
                key=selector_key,
                help=(
                    "Choose the group to promote as the current "
                    "Solution of Interest."
                ),
            )

        current_df = filter_by_group(lens_df, group_column, group_value)

        if current_df is None or current_df.empty:
            st.warning("Selected group contains no valid candidate solutions.")
            return lens_df

        st.caption(f"Current SOI candidate size: **{len(current_df)}** solutions")

        if active_lens == "None":
            st.caption("Source: exploratory current set.")

        st.markdown("---")

        reset_soi_name_if_needed(active_lens, group_value)

        soi_name = st.text_input("Name", key="soi_name")

        if st.button(
            "💾 Save Current Set",
            use_container_width=True,
            key="save_current_soi",
        ):
            # Safe extraction of IDs fallback to DataFrame index if 'id' column is missing
            solution_ids = (
                current_df["id"].tolist()
                if "id" in current_df.columns
                else current_df.index.tolist()
            )

            st.session_state.pending_save_soi = {
                "name": soi_name,
                "lens": lens_label,
                "method": lens_params.get("method", "Exploratory"),
                "params": lens_params,
                "ids": solution_ids,
                "group": group_value,
                "group_column": group_column,
                "source_size": len(lens_df),
                "soi_size": len(current_df),
            }

        return current_df