## --------------------------------------------------------------------------------------
## lenses.py
## --------------------------------------------------------------------------------------

import streamlit as st

from lenses.lens_registry import (
    get_lens_names,
    get_lens_module
)


# =====================================================
# HEADER
# =====================================================

def render_lens_header(
    active_lens
):

    if "active_soi_name" in st.session_state:

        st.caption(
            f"Working on loaded SOI: "
            f"{st.session_state.active_soi_name}"
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
            unsafe_allow_html=True
        )


# =====================================================
# ACTIVE LENS PARAMS
# =====================================================

def render_active_lens_params(
    active_lens,
    dataset,
    working_df
):

    if active_lens == "None":

        return {}

    lens_module = get_lens_module(
        active_lens
    )

    if lens_module is None:

        st.warning(
            f"No module registered for lens: {active_lens}"
        )

        return {}

    if not hasattr(
        lens_module,
        "render_params"
    ):

        st.warning(
            f"Lens module '{active_lens}' does not define render_params()."
        )

        return {}

    return lens_module.render_params(
        dataset,
        working_df
    )


# =====================================================
# MAIN LENS PANEL
# =====================================================

def render_lens_panel(
    dataset,
    working_df
):

    params = {}

    with st.sidebar.expander(
        "🧭 Solution of Interest",
        expanded=False
    ):

        active_lens = st.selectbox(
            "Select an analytical lens",
            get_lens_names(),
            key="active_lens"
        )

        render_lens_header(
            active_lens
        )

        params = render_active_lens_params(
            active_lens,
            dataset,
            working_df
        )

        # Feedback goes here after the lens is applied.
        feedback_placeholder = st.empty()

        # Group selection and save controls go here after feedback.
        selection_placeholder = st.empty()

    return (
        active_lens,
        params,
        feedback_placeholder,
        selection_placeholder
    )