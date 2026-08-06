"""
Input Panel UI and Data Loading Module.

Handles dataset selection, CSV file uploads, plugin initialization, and dynamic 
detection of objective metrics and decision variables.
"""

from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from config import CASES
from plugins import PLUGIN_REGISTRY
from ui.phase_help import render_help_icon, render_phase_help_icon


# =====================================================
# DETECTION & INFERENCE HELPERS
# =====================================================


def detect_decision_variables(df: pd.DataFrame, prefix: str) -> List[str]:
    """
    Identifies columns representing decision variables based on a string prefix.

    Parameters
    ----------
    df : pd.DataFrame
        Candidate solution space DataFrame.
    prefix : str
        Variable column prefix (e.g., 'x_', 'var_').

    Returns
    -------
    List[str]
        List of matching decision variable column names.
    """
    if df is None or df.empty or not prefix:
        return []

    return [col for col in df.columns if col.startswith(prefix)]


def infer_numeric_metrics(df: pd.DataFrame, cfg: Dict[str, Any]) -> List[str]:
    """
    Infers candidate objective metric columns by excluding system and variable columns.

    Parameters
    ----------
    df : pd.DataFrame
        Candidate solution space DataFrame.
    cfg : Dict[str, Any]
        Domain configuration dictionary.

    Returns
    -------
    List[str]
        List of numeric column names valid for objective analysis.
    """
    if df is None or df.empty:
        return []

    var_prefix = cfg.get("var_prefix", "x_")
    excluded = set(cfg.get("exclude_cols", []))

    system_cols = {
        "id",
        "ID",
        "cluster",
        "cluster_str",
        "group_label",
        "group_base",
        "label",
        "highlight",
        "highlight_label",
        "score",
        "preference_score",
        "preference_rank",
        "efficiency_score",
        "efficiency_rank",
        "domain_match_count",
        "domain_rank",
        "selected",
    }

    metrics: List[str] = []
    for col in df.columns:
        if col.startswith(var_prefix):
            continue
        if col in excluded or col in system_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            metrics.append(col)

    return metrics


# =====================================================
# PLUGIN & DATASET CONSTRUCTION
# =====================================================


def build_plugin(cfg: Dict[str, Any]) -> Optional[Any]:
    """
    Instantiates the analytical domain plugin specified in the dataset configuration.

    Parameters
    ----------
    cfg : Dict[str, Any]
        Domain configuration parameters.

    Returns
    -------
    Optional[Any]
        Initialized plugin instance, or None if not configured.
    """
    plugin_name = cfg.get("plugin")
    if not plugin_name:
        return None

    plugin_class = PLUGIN_REGISTRY.get(plugin_name)
    if plugin_class is not None:
        return plugin_class(var_prefix=cfg.get("var_prefix", "x_"))

    return None


def build_dataset(df: pd.DataFrame, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Assembles the complete dataset context dictionary including selected objective metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Loaded solution space DataFrame.
    cfg : Dict[str, Any]
        Domain dataset configuration metadata.

    Returns
    -------
    Dict[str, Any]
        Global dataset context dictionary.
    """
    plugin = build_plugin(cfg)

    all_metrics = cfg.get("metrics", [])
    if not all_metrics:
        all_metrics = infer_numeric_metrics(df, cfg)

    selected_metrics = st.multiselect(
        "Objective Columns",
        all_metrics,
        default=all_metrics,
    )

    decision_variables = detect_decision_variables(
        df, cfg.get("var_prefix", "x_")
    )

    return {
        "df": df,
        "config": cfg,
        "plugin": plugin,
        "metrics": selected_metrics,
        "selected_indicators": [],
        "decision_variables": decision_variables,
    }


# =====================================================
# DATA LOADERS
# =====================================================


def load_builtin_dataset(cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Loads a predefined domain solution dataset from CSV storage.

    Parameters
    ----------
    cfg : Dict[str, Any]
        Domain configuration object containing `path_sol`.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame containing an assigned 'id' column.
    """
    path = cfg["path_sol"]
    df = pd.read_csv(path).reset_index(drop=True)

    if "id" not in df.columns:
        df["id"] = range(1, len(df) + 1)

    return df


def load_uploaded_dataset(uploaded_file: Any) -> pd.DataFrame:
    """
    Loads a user-uploaded CSV file into a solution space DataFrame.

    Parameters
    ----------
    uploaded_file : Any
        Streamlit UploadedFile object.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame containing an assigned 'id' column.
    """
    df = pd.read_csv(uploaded_file).reset_index(drop=True)

    if "id" not in df.columns:
        df["id"] = range(1, len(df) + 1)

    return df


# =====================================================
# UI RENDERING & DATA SOURCE INPUT
# =====================================================


def render_domain_configuration_input() -> Optional[Dict[str, Any]]:
    """
    Renders UI controls for selecting built-in domain configurations.

    Returns
    -------
    Optional[Dict[str, Any]]
        Assembled dataset context dictionary, or None if unselected/failed.
    """
    dataset_names = ["-- No Data --"] + list(CASES.keys())

    col_dataset, col_help = st.columns(
        [0.85, 0.15], vertical_alignment="bottom"
    )

    with col_dataset:
        dataset_name = st.selectbox(
            "Domain Configuration",
            dataset_names,
            key="input_domain_configuration",
        )

    if dataset_name == "-- No Data --":
        dataset_help = (
            "No domain configuration selected yet.\n\n"
            "Choose a predefined case to load its dataset, objectives, "
            "decision-variable prefix, and optional plugin."
        )
    else:
        dataset_help = CASES[dataset_name].get(
            "help",
            "No additional description is available for this domain configuration.",
        )

    with col_help:
        render_help_icon(dataset_help, key="help_domain_configuration")

    if dataset_name == "-- No Data --":
        st.info("Select data to continue.")
        return None

    cfg = CASES[dataset_name]

    try:
        df = load_builtin_dataset(cfg)
    except Exception as exc:
        st.error(f"Unable to load dataset: {cfg.get('path_sol')}")
        st.exception(exc)
        return None

    return build_dataset(df, cfg)


def render_uploaded_csv_input() -> Optional[Dict[str, Any]]:
    """
    Renders UI controls for custom CSV file uploading.

    Returns
    -------
    Optional[Dict[str, Any]]
        Assembled dataset context dictionary, or None if unselected/failed.
    """
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is None:
        return None

    var_prefix = st.text_input("Decision-variable prefix", value="var_")

    try:
        df = load_uploaded_dataset(uploaded_file)
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
        "help": "Uploaded enriched CSV.",
    }

    return build_dataset(df, cfg)


def render_input_panel() -> Optional[Dict[str, Any]]:
    """
    Renders the main Input and Preparation sidebar expander panel.

    Returns
    -------
    Optional[Dict[str, Any]]
        Selected and processed dataset context dictionary, or None if empty.
    """
    with st.sidebar.expander("🏷️ Input and Preparation", expanded=True):
        col_label, col_help = st.columns(
            [0.85, 0.15], vertical_alignment="center"
        )

        with col_label:
            st.markdown("**Data Source**")

        with col_help:
            render_phase_help_icon("input", key="help_input_phase")

        mode = st.radio(
            "Data Source",
            [
                "1. Domain Configuration",
                "2. Upload Enriched CSV",
            ],
            horizontal=True,
            label_visibility="collapsed",
        )

        if mode == "1. Domain Configuration":
            return render_domain_configuration_input()

        return render_uploaded_csv_input()