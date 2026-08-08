"""
Workspace Summary Module.
"""

from typing import Any, Dict
import pandas as pd
import streamlit as st

from core.pipeline import ParetoExplorerEngine
from ui.soi_panel import render_saved_sois_view
from ui.workspace_dataset import render_dataset_table


def render_summary_metrics(
    engine: ParetoExplorerEngine, dataset_config: Dict[str, Any]
) -> None:
    """Renders the 5 core dataset dimension metrics using the engine's problem context."""
    df = engine.active_df
    ctx = getattr(engine, "context", None)

    # 1. Variables de Decisión (X)
    if ctx and ctx.decision_variables:
        var_cols = set(ctx.decision_variables)
    else:
        var_prefix = dataset_config.get("var_prefix", "x_")
        var_cols = {col for col in df.columns if str(col).startswith(var_prefix)}

    # 2. Objetivos Principales (Y)
    if ctx and ctx.metrics:
        obj_cols = set(ctx.metrics)
    else:
        obj_cols = set(dataset_config.get("metrics", []) or [])

    # Filtrar solo las columnas presentes en el DataFrame activo
    vars_in_df = {col for col in var_cols if col in df.columns}
    objs_in_df = {col for col in obj_cols if col in df.columns}
    id_cols = {col for col in df.columns if str(col).lower() in ("id", "solution_id")}

    # 3. Atributos Enriquecidos / Analíticos (Z)
    enriched_cols = set(df.columns) - vars_in_df - objs_in_df - id_cols

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.metric("Solutions", len(df))
    with c2:
        st.metric("Decision Vars", len(vars_in_df))
    with c3:
        st.metric("Objectives", len(objs_in_df))
    with c4:
        st.metric("Enriched Attribs", len(enriched_cols))
    with c5:
        st.metric("Total Attribs", len(df.columns))


def render_export_section(df: pd.DataFrame) -> None:
    """Renders single CSV export control."""
    st.download_button(
        label="📊 Export Current Set (.csv)",
        data=df.to_csv(index=False),
        file_name="current_set.csv",
        mime="text/csv",
        use_container_width=True,
    )


def render_summary(
    engine: ParetoExplorerEngine, dataset_config: Dict[str, Any]
) -> None:
    """Main expander container rendering Overview, Current Set, and Saved SOIs via Radio controls."""
    df = engine.active_df

    if df is None or df.empty:
        st.error(
            "Dataset summary cannot be rendered because the current dataframe is empty."
        )
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
            render_summary_metrics(engine, dataset_config)
            st.divider()
            render_export_section(df)

        elif view_mode == "📋 Current Set":
            render_dataset_table(df, dataset_config)

        elif view_mode == "🔖 Saved SOIs":
            render_saved_sois_view(engine)