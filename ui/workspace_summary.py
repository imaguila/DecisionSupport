"""
Workspace Summary Module.
"""

from datetime import datetime
from typing import Any, Dict, List
import pandas as pd
import streamlit as st

from core.pipeline import ParetoExplorerEngine
from ui.soi_panel import render_saved_sois_view
from ui.workspace_dataset import render_dataset_table


def get_lens_columns(df: pd.DataFrame) -> List[str]:
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
    lens_columns = [col for col in df.columns if any(col.startswith(prefix) for prefix in lens_prefixes)]
    structural_columns = [
        col for col in ["cluster", "cluster_str", "group_label", "group_base", "highlight"]
        if col in df.columns
    ]
    return structural_columns + lens_columns


def generate_markdown_report(df: pd.DataFrame, dataset_config: Dict[str, Any], engine: ParetoExplorerEngine) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    dataset_name = dataset_config.get("name", "Pareto Optimization Workspace")
    saved_sois = engine.soi_registry.list_all() if engine else []

    report = []
    report.append(f"# 📊 Executive Decision Report: {dataset_name}")
    report.append(f"**Generated on:** {timestamp}\n")
    report.append("---")
    report.append("## 1. Executive Overview")
    report.append(f"- **Active Set Size:** {len(df)} solutions")
    report.append(f"- **Saved Sets of Interest (SOIs):** {len(saved_sois)} sets\n")

    report.append("## 2. Analyzed Sets of Interest (SOIs)")
    if saved_sois:
        report.append("| SOI Name | Lens / Method | Size | Created At |")
        report.append("| :--- | :--- | :--- | :--- |")
        for soi in saved_sois:
            method = soi.method_name or "N/A"
            report.append(f"| {soi.name} | {soi.lens_name} ({method}) | {soi.size} | {soi.created_at} |")
    else:
        report.append("_No SOIs were explicitly saved during this session._")

    report.append("\n---")
    report.append("*Report generated automatically by Pareto Framework Decision Space Explorer.*")
    return "\n".join(report)


def render_summary_metrics(df: pd.DataFrame, dataset_config: Dict[str, Any]) -> None:
    c1, c2, c3, c4 = st.columns(4)
    var_prefix = dataset_config.get("var_prefix", "x_")
    num_vars = len([col for col in df.columns if col.startswith(var_prefix)])

    with c1:
        st.metric("Solutions", len(df))
    with c2:
        st.metric("Attributes", len(df.columns))
    with c3:
        st.metric("Decision Variables", num_vars)
    with c4:
        st.metric("Status", "Active Workspace")


def render_export_section(df: pd.DataFrame, dataset_config: Dict[str, Any], engine: ParetoExplorerEngine) -> None:
    st.markdown("##### 📥 Export Options")
    col_report, col_csv = st.columns(2)

    config_name = dataset_config.get("name", "pareto_workspace")

    with col_report:
        report_md = generate_markdown_report(df, dataset_config, engine)
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


def render_summary(engine: ParetoExplorerEngine, dataset_config: Dict[str, Any]) -> None:
    """Main expander container rendering Overview, Current Set, and Saved SOIs via Radio controls."""
    df = engine.active_df

    if df is None or df.empty:
        st.error("Dataset summary cannot be rendered because the current dataframe is empty.")
        return

    with st.expander("📊 Workspace Summary & Data Inspection", expanded=False):
        view_mode = st.radio(
            "Summary View Selector",
            ["📊 Overview", "📋 Current Set", "🔖 Saved SOIs"],
            horizontal=True,
            label_visibility="collapsed",
            key="summary_view_mode",
        )

        if view_mode == "📊 Overview":
            render_summary_metrics(df, dataset_config)
            lens_cols = get_lens_columns(df)
            if lens_cols:
                st.caption("Derived analytical columns: " + ", ".join(lens_cols))
            render_export_section(df, dataset_config, engine)

        elif view_mode == "📋 Current Set":
            render_dataset_table(df, dataset_config)

        elif view_mode == "🔖 Saved SOIs":
            render_saved_sois_view(engine)