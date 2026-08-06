"""
Workspace Summary Module.

Provides summary metrics, lens attribute inspection, executive report generation, 
and file export capabilities (Markdown/CSV) for the visual workspace.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from core.workspace_dataset import render_dataset_table
from soi.soi_registry import render_soi_tab


# =====================================================
# DERIVED / LENS COLUMNS
# =====================================================


def get_lens_columns(df: pd.DataFrame) -> List[str]:
    """
    Identifies structural and analytical lens columns present in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Active solution space DataFrame.

    Returns
    -------
    List[str]
        List of identified structural and derived lens column names.
    """
    if df is None or df.empty:
        return []

    lens_prefixes = (
        "preference_",
        "efficiency_",
        "diversity_",
        "domain_",
        "indicator_",
        "consensus_",
    )

    lens_columns = [
        col
        for col in df.columns
        if any(col.startswith(prefix) for prefix in lens_prefixes)
    ]

    structural_columns = [
        col
        for col in [
            "cluster",
            "cluster_str",
            "group_label",
            "group_base",
            "highlight",
        ]
        if col in df.columns
    ]

    return structural_columns + lens_columns


# =====================================================
# REPORT GENERATOR HELPER
# =====================================================


def generate_markdown_report(df: pd.DataFrame, dataset: Dict[str, Any]) -> str:
    """
    Generates an executive decision report formatted in Markdown for export.

    Parameters
    ----------
    df : pd.DataFrame
        Active solution space DataFrame.
    dataset : Dict[str, Any]
        Global dataset context dictionary.

    Returns
    -------
    str
        Formatted Markdown report text.
    """
    if df is None:
        df = pd.DataFrame()

    if dataset is None:
        dataset = {}

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    dataset_name = dataset.get("config", {}).get(
        "name", "Pareto Optimization Dataset"
    )
    saved_sois = st.session_state.get("saved_sois", [])
    highlight_ids = st.session_state.get("css_highlight_ids", [])

    report = []

    # 1. Header
    report.append(f"# 📊 Executive Decision Report: {dataset_name}")
    report.append(f"**Generated on:** {timestamp}\n")
    report.append("---")

    # 2. Executive Overview
    report.append("## 1. Executive Overview")
    css_active = st.session_state.get("css_enabled", False)
    report.append(f"- **Current Set Size:** {len(df)} solutions")
    report.append(f"- **CSS Lock Status:** {'Active' if css_active else 'Inactive'}")
    report.append(f"- **Saved Sets of Interest (SOIs):** {len(saved_sois)} sets")
    report.append(
        f"- **Highlighted Solutions:** {len(highlight_ids)} solutions\n"
    )

    # 3. Saved SOIs Summary
    report.append("## 2. Analyzed Sets of Interest (SOIs)")
    if saved_sois:
        report.append("| SOI Name | Lens / Type | Size | Source Group |")
        report.append("| :--- | :--- | :--- | :--- |")
        for soi in saved_sois:
            name = soi.get("name", "Unnamed")
            lens = soi.get("lens", soi.get("type", "Manual"))
            size = soi.get("soi_size", len(soi.get("ids", [])))
            group = soi.get("group", "N/A")
            report.append(f"| {name} | {lens} | {size} | {group} |")
    else:
        report.append("_No SOIs were explicitly saved during this session._")
    report.append("\n")

    # 4. Highlighted Solutions Comparison
    report.append("## 3. Highlighted Solutions Comparison")
    if highlight_ids and "id" in df.columns and not df.empty:
        high_df = df[df["id"].isin(highlight_ids)].copy()
        if not high_df.empty:
            metrics = dataset.get("metrics", []) + dataset.get(
                "selected_indicators", []
            )
            show_cols = ["id"] + [m for m in metrics if m in high_df.columns]
            report.append(high_df[show_cols].to_markdown(index=False))
        else:
            report.append(
                "_No matching highlighted solutions found in current set._"
            )
    else:
        report.append(
            "_No specific solutions are currently highlighted for comparison._"
        )

    report.append("\n---")
    report.append(
        "*Report generated automatically by Pareto Framework Decision Tool.*"
    )

    return "\n".join(report)


# =====================================================
# SUMMARY METRICS & EXPORTS
# =====================================================


def render_summary_metrics(df: pd.DataFrame, dataset: Dict[str, Any]) -> None:
    """
    Renders metric card indicators summarizing current dataset dimensions.

    Parameters
    ----------
    df : pd.DataFrame
        Active solution space DataFrame.
    dataset : Dict[str, Any]
        Global dataset context dictionary.
    """
    if df is None:
        return

    if dataset is None:
        dataset = {}

    c1, c2, c3, c4 = st.columns(4)

    var_prefix = dataset.get("config", {}).get("var_prefix", "x_")
    if "decision_variables" in dataset and isinstance(
        dataset["decision_variables"], list
    ):
        num_vars = len(dataset["decision_variables"])
    else:
        num_vars = len([col for col in df.columns if col.startswith(var_prefix)])

    with c1:
        st.metric("Solutions", len(df))
    with c2:
        st.metric("Attributes", len(df.columns))
    with c3:
        st.metric("Decision Variables", num_vars)
    with c4:
        css_status = (
            "Active" if st.session_state.get("css_enabled", False) else "Inactive"
        )
        st.metric("CSS", css_status)


def render_lens_summary(df: pd.DataFrame) -> None:
    """
    Displays a caption summarizing derived analytical lens attributes.

    Parameters
    ----------
    df : pd.DataFrame
        Active solution space DataFrame.
    """
    lens_columns = get_lens_columns(df)

    if not lens_columns:
        st.caption("No derived lens columns in the current set.")
        return

    st.caption("Derived columns: " + ", ".join(lens_columns))


def render_export_section(df: pd.DataFrame, dataset: Dict[str, Any]) -> None:
    """
    Renders export controls for downloading executive reports and CSV data.

    Parameters
    ----------
    df : pd.DataFrame
        Active solution space DataFrame.
    dataset : Dict[str, Any]
        Global dataset context dictionary.
    """
    if df is None or dataset is None:
        return

    st.markdown("---")
    st.markdown("##### 📥 Export Options")
    col_report, col_csv = st.columns(2)

    config_name = dataset.get("config", {}).get("name", "pareto")

    with col_report:
        report_md = generate_markdown_report(df, dataset)
        st.download_button(
            label="📄 Export Executive Report (.md)",
            data=report_md,
            file_name=f"executive_report_{config_name}.md",
            mime="text/markdown",
            use_container_width=True,
            type="primary",
        )

    with col_csv:
        st.download_button(
            label="📊 Export Current Set (.csv)",
            data=df.to_csv(index=False),
            file_name="current_set.csv",
            mime="text/csv",
            use_container_width=True,
        )


def get_summary_label() -> str:
    """
    Returns dynamic title string for summary container based on CSS state.

    Returns
    -------
    str
        Summary section title.
    """
    if st.session_state.get("css_enabled", False):
        return "Summary / Current CSS / Saved SOIs"
    return "Summary / Current Set / Saved SOIs"


# =====================================================
# MAIN RENDERER
# =====================================================


def render_summary(df: pd.DataFrame, dataset: Dict[str, Any]) -> None:
    """
    Main entry point for rendering the workspace summary, data table, and SOI tabs.

    Parameters
    ----------
    df : pd.DataFrame
        Active solution space DataFrame.
    dataset : Dict[str, Any]
        Global dataset context dictionary.
    """
    if df is None or df.empty:
        st.error(
            "Dataset summary cannot be rendered "
            "because the current dataframe is empty."
        )
        return

    label = get_summary_label()

    with st.expander(f"📊 {label}", expanded=False):
        tab_overview, tab_current, tab_saved_soi = st.tabs(
            [
                "**| Overview |**",
                "**| Current Set |**",
                "**| Saved SOIs |**",
            ]
        )

        with tab_overview:
            render_summary_metrics(df, dataset)
            config = dataset.get("config", {}) if dataset else {}
            st.caption(
                f"Decision-variable prefix: {config.get('var_prefix', 'x_')}"
            )
            render_lens_summary(df)
            render_export_section(df, dataset)

        with tab_current:
            render_dataset_table(df, dataset)

        with tab_saved_soi:
            render_soi_tab()