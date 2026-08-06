"""
Data Enrichment Module.

Provides functionality to detect, select, and compute derived indicators 
for candidate solutions based on domain-specific plugin requirements.
"""

from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from ui.phase_help import render_phase_help_icon


# =====================================================
# HELPER FUNCTIONS
# =====================================================


def get_available_indicators(
    plugin: Any, selected_metrics: List[str]
) -> List[str]:
    """
    Identifies compatible indicators based on selected metrics and plugin requirements.

    Parameters
    ----------
    plugin : Any
        Domain plugin instance offering requirement checks.
    selected_metrics : List[str]
        List of currently active dataset metric names.

    Returns
    -------
    List[str]
        List of indicator names whose metric requirements are fully satisfied.
    """
    if not plugin or not hasattr(plugin, "requirements"):
        return []

    available_indicators: List[str] = []
    requirements: Dict[str, List[str]] = plugin.requirements()

    for indicator, reqs in requirements.items():
        if all(metric in selected_metrics for metric in reqs):
            available_indicators.append(indicator)

    return available_indicators


# =====================================================
# UI RENDERING & COMPUTATION ENTRY POINT
# =====================================================


def render_enrichment(dataset: Dict[str, Any]) -> Dict[str, Any]:
    """
    Renders the Data Enrichment sidebar UI section and computes selected indicators.

    Parameters
    ----------
    dataset : Dict[str, Any]
        Global dataset context dictionary containing plugin, df, and metadata.

    Returns
    -------
    Dict[str, Any]
        Updated dataset dictionary enriched with computed indicator features.
    """
    if not dataset:
        return {}

    plugin = dataset.get("plugin")
    if plugin is None:
        dataset["selected_indicators"] = []
        return dataset

    selected_metrics: List[str] = dataset.get("metrics", [])
    available_indicators = get_available_indicators(plugin, selected_metrics)

    config = dataset.get("config", {})
    default_indicators = [
        indicator
        for indicator in config.get("default_indicators", [])
        if indicator in available_indicators
    ]

    with st.sidebar.expander("⚙️ Data Enrichment", expanded=False):
        col_label, col_help = st.columns(
            [0.85, 0.15], vertical_alignment="center"
        )

        with col_label:
            st.markdown("**Derived Indicators**")

        with col_help:
            render_phase_help_icon("enrichment", key="help_enrichment_phase")

        st.caption(
            f"Detected {len(available_indicators)} compatible indicators."
        )

        selected_indicators = st.multiselect(
            "Available indicators",
            sorted(available_indicators),
            default=default_indicators,
        )

    df: Optional[pd.DataFrame] = dataset.get("df")
    if df is not None and hasattr(plugin, "compute_indicators"):
        dataset["df"] = plugin.compute_indicators(df, selected_indicators)

    dataset["selected_indicators"] = selected_indicators

    return dataset