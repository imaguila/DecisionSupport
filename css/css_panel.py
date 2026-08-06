"""
Candidate Solution Set (CSS) Panel Module.

Provides sidebar controls and session state management for filtering, locking, 
and highlighting Candidate Solution Sets (CSS) across the visual workspace.
"""

from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st


# =====================================================
# SESSION STATE MANAGEMENT
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


def sanitize_ids(ids: List[Any], valid_ids: List[Any]) -> List[Any]:
    """
    Filters an ID list to retain only IDs present in valid_ids.

    Parameters
    ----------
    ids : List[Any]
        List of candidate IDs to validate.
    valid_ids : List[Any]
        List of active valid solution IDs.

    Returns
    -------
    List[Any]
        Filtered list of valid solution IDs.
    """
    if not ids or not valid_ids:
        return []

    valid_set = set(valid_ids)
    return [solution_id for solution_id in ids if solution_id in valid_set]


# =====================================================
# SIDEBAR PANEL RENDERER
# =====================================================


def render_css_panel(
    current_df: Optional[pd.DataFrame],
    dataset: Optional[Dict[str, Any]] = None,
) -> Optional[pd.DataFrame]:
    """
    Renders sidebar controls for managing the Candidate Solution Set (CSS).

    Allows users to lock the active solution space or manually select 
    solutions, dynamically tagging highlighted items.

    Parameters
    ----------
    current_df : Optional[pd.DataFrame]
        Active solution space DataFrame.
    dataset : Optional[Dict[str, Any]], optional
        Global dataset context dictionary, by default None.

    Returns
    -------
    Optional[pd.DataFrame]
        Filtered DataFrame representing the active Candidate Solution Set (CSS).
    """
    ensure_css_state()

    if current_df is None or current_df.empty:
        return current_df

    css_df = current_df.copy()
    valid_ids = (
        css_df["id"].dropna().tolist() if "id" in css_df.columns else []
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
            help=(
                "Create a Candidate Solution Set from current filtered set "
                "or manual selection."
            ),
        )

        if not st.session_state.css_enabled:
            st.caption(f"Current set available: {len(current_df)} solutions")
            css_df["highlight"] = False
            return css_df

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
            css_df = current_df[
                current_df["id"].isin(st.session_state.css_manual_ids)
            ].copy()
        else:
            css_df = current_df.copy()

        st.info(f"CSS size: {len(css_df)} solutions")

        st.session_state.show_css_comparison = st.checkbox(
            "Open detailed comparison",
            value=st.session_state.show_css_comparison,
            help="Open detailed visual comparison section for the current CSS.",
        )

    if "id" in css_df.columns:
        css_df["highlight"] = css_df["id"].isin(
            st.session_state.css_highlight_ids
        )
    else:
        css_df["highlight"] = False

    return css_df