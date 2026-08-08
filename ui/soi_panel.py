"""
SOI (Set of Interest) UI Components.
"""

from typing import Any, Dict, Optional
import pandas as pd
import streamlit as st

from core.pipeline import ParetoExplorerEngine
from core.soi import SOI


def render_soi_saver(
    engine: ParetoExplorerEngine,
    active_lens_name: str,
    lens_params: Dict[str, Any],
) -> None:
    """Renders the Save SOI box inside the Lens Panel with dynamic updates."""
    active_df = engine.active_df

    if active_df is None or active_df.empty:
        return

    st.markdown("---")
    st.markdown("##### 💾 Save Current Set as SOI")

    # Detección de columnas de grupo/cluster
    group_col = None
    if "group_label" in active_df.columns:
        group_col = "group_label"
    elif "cluster_str" in active_df.columns:
        group_col = "cluster_str"
    elif "group_base" in active_df.columns:
        group_col = "group_base"

    working_set = active_df.copy()
    selected_group_name: Optional[str] = None

    if group_col:
        groups = ["All groups"] + sorted(active_df[group_col].dropna().astype(str).unique().tolist())
        # Clave dinámica para el filtro de grupo por Lente
        selected_group = st.selectbox(
            "Filter Group to Save", 
            groups, 
            key=f"soi_group_filter_{active_lens_name}"
        )
        if selected_group != "All groups":
            working_set = active_df[active_df[group_col].astype(str) == str(selected_group)].copy()
            selected_group_name = str(selected_group)

    st.caption(f"Candidate SOI size: **{len(working_set)}** solutions")

    total_sois = len(engine.soi_registry.list_all())

    # 1. Nombre predeterminado más descriptivo (incluye sub-grupo/cluster si aplica)
    group_suffix = f" ({selected_group_name})" if selected_group_name else ""
    default_name = f"{active_lens_name}{group_suffix} Set #{total_sois + 1}"

    # 2. CLAVE DINÁMICA: Cambia según la Lente y el Grupo activo.
    # Esto fuerza a Streamlit a refrescar el 'value' cuando cambia el contexto.
    input_key = f"soi_name_input_{active_lens_name}_{selected_group_name or 'all'}"

    soi_name = st.text_input("SOI Name", value=default_name, key=input_key)

    if st.button("💾 Save SOI", use_container_width=True, type="primary", key=f"btn_save_soi_{active_lens_name}"):
        solution_ids = working_set["id"].tolist() if "id" in working_set.columns else working_set.index.tolist()
        soi_id = f"soi_{total_sois + 1}_{pd.Timestamp.now().strftime('%H%M%S')}"
        method_name = lens_params.get("method", "Exploratory")

        new_soi = SOI(
            id=soi_id,
            name=soi_name,
            solution_ids=solution_ids,
            lens_name=active_lens_name,
            method_name=method_name,
            group=selected_group_name,
            params=lens_params,
        )

        engine.soi_registry.add(new_soi)
        st.success(f"Saved '{soi_name}' ({len(solution_ids)} solutions)!")
        st.rerun()


def render_saved_sois_view(engine: ParetoExplorerEngine) -> None:
    """Renders the Saved SOIs management view inside the Summary tab."""
    # 1. Banner si hay un Sub-Dataset activo
    if engine.active_soi:
        active = engine.active_soi
        st.info(
            f"📌 **Active Sub-Dataset:** `{active.name}` ({active.size} solutions)\n\n"
            "All framing bounds and analytical lenses are currently filtered to this subset."
        )
        if st.button("✖️ Clear Active Sub-Dataset Filter", use_container_width=True):
            engine.clear_active_soi()
            engine.apply_framing(engine.framing_bounds)
            engine.apply_lens("None", {})
            st.rerun()
        st.divider()

    saved_sois = engine.soi_registry.list_all()

    if not saved_sois:
        st.caption("No Sets of Interest (SOIs) saved yet. Save a set using the active Lens controls.")
        return

    # 2. Tabla / Tarjetas de SOIs guardados
    for soi in saved_sois:
        is_active = engine.active_soi_id == soi.id
        badge = " 📍 *(Active Sub-Dataset)*" if is_active else ""

        with st.expander(f"**{soi.name}** [{soi.size} solutions] · {soi.lens_name}{badge}", expanded=is_active):
            st.markdown(f"- **Lens / Method:** `{soi.lens_name}` / `{soi.method_name or 'N/A'}`")
            if soi.group:
                st.markdown(f"- **Group Filter:** `{soi.group}`")
            st.markdown(f"- **Created:** `{soi.created_at}`")

            col_load, col_del = st.columns([0.6, 0.4])

            with col_load:
                if is_active:
                    st.button("Active", key=f"btn_act_{soi.id}", disabled=True, use_container_width=True)
                else:
                    if st.button("⚡ Load as Sub-Dataset", key=f"btn_load_{soi.id}", use_container_width=True):
                        engine.load_soi_as_dataset(soi.id)
                        st.rerun()

            with col_del:
                if st.button("🗑️ Delete", key=f"btn_del_{soi.id}", use_container_width=True):
                    if is_active:
                        engine.clear_active_soi()
                    engine.soi_registry.delete(soi.id)
                    st.rerun()