"""
Input Panel UI Module.

Streamlit view component for dataset acquisition and metric selection.
Delegates all data operations to `core.dataset_loader`.
"""

from typing import Optional, Dict, Any
import streamlit as st

from config import CASES
from core.dataset_loader import (
    ProblemContext,
    load_raw_dataframe,
    create_problem_context,
    infer_numeric_metrics,
)
from ui.phase_help import render_help_icon, render_phase_help_icon


def render_domain_configuration_input() -> Optional[ProblemContext]:
    """Renders controls for pre-configured domain datasets."""
    dataset_names = ["-- No Data --"] + list(CASES.keys())
    
    col_dataset, col_help = st.columns([0.85, 0.15], vertical_alignment="bottom")
    with col_dataset:
        dataset_name = st.selectbox(
            "Domain Configuration", dataset_names, key="input_domain_configuration"
        )

    if dataset_name == "-- No Data --":
        st.info("Select data to continue.")
        return None

    cfg = CASES[dataset_name]
    
    with col_help:
        render_help_icon(
            cfg.get("help", "No description available."), 
            key="help_domain_configuration"
        )

    try:
        df = load_raw_dataframe(cfg["path_sol"])
    except Exception as exc:
        st.error(f"Unable to load dataset: {cfg.get('path_sol')}")
        st.exception(exc)
        return None

    # UI allows user to filter/choose objective metrics
    available_metrics = cfg.get("metrics") or infer_numeric_metrics(
        df, cfg.get("var_prefix", "x_"), cfg.get("exclude_cols")
    )
    selected_metrics = st.multiselect(
        "Objective Columns", available_metrics, default=available_metrics
    )

    return create_problem_context(df, cfg, selected_metrics=selected_metrics)


def render_uploaded_csv_input() -> Optional[ProblemContext]:
    """Renders controls for uploading custom CSV files."""
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is None:
        return None

    var_prefix = st.text_input("Decision-variable prefix", value="var_")

    try:
        df = load_raw_dataframe(uploaded_file)
    except Exception as exc:
        st.error("Unable to load uploaded CSV.")
        st.exception(exc)
        return None

    cfg: Dict[str, Any] = {
        "plugin": None,
        "metrics": [],
        "var_prefix": var_prefix,
        "exclude_cols": [],
        "default_indicators": [],
        "help": "Custom uploaded CSV.",
    }

    available_metrics = infer_numeric_metrics(df, var_prefix=var_prefix)
    selected_metrics = st.multiselect(
        "Objective Columns", available_metrics, default=available_metrics
    )

    return create_problem_context(df, cfg, selected_metrics=selected_metrics)


def render_input_panel() -> Optional[ProblemContext]:
    """Main input panel component for the Streamlit sidebar."""
    with st.sidebar.expander("🏷️ Input and Preparation", expanded=True):
        col_label, col_help = st.columns([0.85, 0.15], vertical_alignment="center")
        with col_label:
            st.markdown("**Data Source**")
        with col_help:
            render_phase_help_icon("input", key="help_input_phase")

        mode = st.radio(
            "Data Source",
            ["1. Domain Configuration", "2. Upload Enriched CSV"],
            horizontal=True,
            label_visibility="collapsed",
        )

        if mode == "1. Domain Configuration":
            return render_domain_configuration_input()

        return render_uploaded_csv_input()