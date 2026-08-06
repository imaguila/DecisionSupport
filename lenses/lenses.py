"""
Lens Panel UI Component.

Renders sidebar user interface controls for analytical lens selection,
dynamic parameter configuration, and container placeholders within the
Decision Space Explorer framework.
"""

import logging
from typing import Any, Dict, Tuple
import pandas as pd
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from lenses.lens_registry import get_lens_module, get_lens_names

logger = logging.getLogger(__name__)


# =====================================================
# HEADER
# =====================================================

def render_lens_header(active_lens: str) -> None:
    """
    Renders the visual header and session context indicator for the selected lens.

    Parameters
    ----------
    active_lens : str
        The unique identifier of the currently active analytical lens.
    """
    if "active_soi_name" in st.session_state and st.session_state.active_soi_name:
        st.caption(
            f"Working on loaded SOI: **{st.session_state.active_soi_name}**"
        )

    if active_lens != "None":
        st.markdown(
            f"""
            <div style="
                color:#E63946;
                font-size:12px;
                font-weight:600;
                text-align:center;
                margin:0.3rem 0 0.8rem 0;
            ">
                ───── {active_lens} lens ─────
            </div>
            """,
            unsafe_allow_html=True,
        )


# =====================================================
# ACTIVE LENS PARAMS
# =====================================================

def render_active_lens_params(
    active_lens: str, dataset: Dict[str, Any], working_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Retrieves and renders the UI parameter controls for the active lens.

    Parameters
    ----------
    active_lens : str
        Identifier of the active analytical lens.
    dataset : Dict[str, Any]
        Global dataset context containing objective space definitions and metadata.
    working_df : pd.DataFrame
        The current active dataset of candidate solutions.

    Returns
    -------
    Dict[str, Any]
        Dictionary of parameter values collected from user interaction in the UI.
    """
    if active_lens == "None":
        return {}

    lens_module = get_lens_module(active_lens)

    if lens_module is None:
        st.warning(f"No module registered for lens: '{active_lens}'")
        return {}

    if not hasattr(lens_module, "render_params"):
        st.warning(
            f"Lens module '{active_lens}' does not define 'render_params()'."
        )
        return {}

    try:
        return lens_module.render_params(dataset, working_df)
    except Exception as e:
        logger.error(f"Error rendering parameters for lens '{active_lens}': {str(e)}")
        st.error(f"Error loading parameters for '{active_lens}': {str(e)}")
        return {}


# =====================================================
# MAIN LENS PANEL
# =====================================================

def render_lens_panel(
    dataset: Dict[str, Any], working_df: pd.DataFrame
) -> Tuple[str, Dict[str, Any], DeltaGenerator, DeltaGenerator]:
    """
    Renders the main sidebar panel for selecting lenses and managing state.

    Parameters
    ----------
    dataset : Dict[str, Any]
        Global dataset dictionary.
    working_df : pd.DataFrame
        Current working solution space DataFrame.

    Returns
    -------
    Tuple[str, Dict[str, Any], DeltaGenerator, DeltaGenerator]
        A tuple containing:
        - active_lens (str): Selected lens name.
        - params (Dict[str, Any]): Dictionary of collected lens parameters.
        - feedback_placeholder (DeltaGenerator): Streamlit UI container for feedback.
        - selection_placeholder (DeltaGenerator): Streamlit UI container for grouping/saving.
    """
    params: Dict[str, Any] = {}

    with st.sidebar.expander("🧭 Solution of Interest", expanded=False):
        active_lens = st.selectbox(
            "Select an analytical lens",
            get_lens_names(),
            key="active_lens",
        )

        render_lens_header(active_lens)

        params = render_active_lens_params(
            active_lens, dataset, working_df
        )

        # Container reserved for lens output metrics / feedback
        feedback_placeholder = st.empty()

        # Container reserved for group selection & candidate saving controls
        selection_placeholder = st.empty()

    return (
        active_lens,
        params,
        feedback_placeholder,
        selection_placeholder,
    )