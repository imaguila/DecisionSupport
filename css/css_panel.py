## --------------------------------------------------------------------------------------
## css/css_panel.py
## --------------------------------------------------------------------------------------

import streamlit as st

def ensure_css_state():
    if "css_enabled" not in st.session_state:
        st.session_state.css_enabled = False
    if "css_source" not in st.session_state:
        st.session_state.css_source = "Current set"
    if "css_manual_ids" not in st.session_state:
        st.session_state.css_manual_ids = []
    if "css_highlight_ids" not in st.session_state:
        st.session_state.css_highlight_ids = []
    if "show_css_comparison" not in st.session_state:
        st.session_state.show_css_comparison = False


def sanitize_ids(ids, valid_ids):
    valid_set = set(valid_ids)
    return [solution_id for solution_id in ids if solution_id in valid_set]


def render_css_panel(current_df, dataset):
    ensure_css_state()

    if current_df is None:
        return current_df

    css_df = current_df.copy()
    valid_ids = css_df["id"].tolist() if "id" in css_df.columns else []

    st.session_state.css_manual_ids = sanitize_ids(st.session_state.css_manual_ids, valid_ids)
    st.session_state.css_highlight_ids = sanitize_ids(st.session_state.css_highlight_ids, valid_ids)

    with st.sidebar.expander("🎯 Candidate Solution Set", expanded=False):
        st.session_state.css_enabled = st.checkbox(
            "Lock current set as CSS",
            value=st.session_state.css_enabled,
            help="Create a Candidate Solution Set from current filtered set or manual selection."
        )

        if not st.session_state.css_enabled:
            st.caption(f"Current set available: {len(current_df)} solutions")
            current_df["highlight"] = False
            return current_df

        st.session_state.css_source = st.radio(
            "CSS source",
            ["Current set", "Manual selection"],
            index=["Current set", "Manual selection"].index(st.session_state.css_source),
            horizontal=True
        )

        if st.session_state.css_source == "Manual selection":
            st.session_state.css_manual_ids = st.multiselect(
                "Solutions included in CSS",
                options=valid_ids,
                default=st.session_state.css_manual_ids,
                key="css_manual_ids_widget",
                help="Select the exact solutions that form the Candidate Solution Set."
            )
            css_df = current_df[current_df["id"].isin(st.session_state.css_manual_ids)].copy()
        else:
            css_df = current_df.copy()

        st.info(f"CSS size: {len(css_df)} solutions")

        st.session_state.show_css_comparison = st.checkbox(
            "Open detailed comparison",
            value=st.session_state.show_css_comparison,
            help="Open detailed visual comparison section for the current CSS."
        )

    css_df["highlight"] = css_df["id"].isin(st.session_state.css_highlight_ids)

    return css_df