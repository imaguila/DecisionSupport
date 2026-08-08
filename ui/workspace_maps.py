"""
Workspace Maps UI Module.

Renders multi-panel decision space visualizations driven by st.session_state.maps.
Includes dual 2D scatter plots (X vs Y & X vs Z), 3D visualization, Color mapping,
and multi-objective distribution analytics.
"""

from typing import List
import pandas as pd
import plotly.express as px
import streamlit as st


# =====================================================
# UTILITIES
# =====================================================

def _get_color_options(df: pd.DataFrame, dimensions: List[str]) -> List[str]:
    """Generates a list of available columns for color mapping."""
    options = ["None"]
    lens_cols = [c for c in ["cluster_str", "match_count", "group_label"] if c in df.columns]
    options.extend(lens_cols)
    for dim in dimensions:
        if dim not in options:
            options.append(dim)
    return options


def _get_default_color_index(options: List[str]) -> int:
    """Selects cluster_str as the default color column if present."""
    for preferred in ["cluster_str", "group_label", "match_count"]:
        if preferred in options:
            return options.index(preferred)
    return 0


# =====================================================
# INDIVIDUAL CHART RENDERERS
# =====================================================

def render_scatter_chart(
    df: pd.DataFrame, dimensions: List[str], show_ids: bool, map_id: int, default_x: str, default_y: str
) -> None:
    """
    Renders Scatter Map with top 3 axes (X, Y, Z=None), Color axis (Auto-Cluster default), 
    Dual 2D plots (X-Y & X-Z) or 3D plot.
    """
    c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 0.8])

    x_index = dimensions.index(default_x) if default_x in dimensions else 0
    y_index = dimensions.index(default_y) if default_y in dimensions else min(1, len(dimensions) - 1)

    x_dim = c1.selectbox("X Axis", dimensions, index=x_index, key=f"scatter_x_{map_id}")
    y_dim = c2.selectbox("Y Axis", dimensions, index=y_index, key=f"scatter_y_{map_id}")
    
    z_options = ["None"] + [d for d in dimensions if d not in [x_dim, y_dim]]
    z_dim = c3.selectbox("Z Axis", z_options, index=0, key=f"scatter_z_{map_id}")

    color_options = _get_color_options(df, dimensions)
    color_key = f"scatter_color_{map_id}"

    # Auto-forzar cluster_str en session_state si se detectan clusters y el estado actual es "None"
    if "cluster_str" in df.columns:
        if color_key not in st.session_state or st.session_state[color_key] == "None":
            st.session_state[color_key] = "cluster_str"

    color_dim = c4.selectbox("Color Axis", color_options, key=color_key)
    active_color = None if color_dim == "None" else color_dim

    use_3d = c5.checkbox("Enable 3D", value=False, key=f"scatter_3d_{map_id}", disabled=(z_dim == "None"))

    # Configuración base de argumentos
    base_kwargs = {
        "data_frame": df,
        "color": active_color,
        "hover_data": ["id"] if "id" in df.columns else None,
        "text": "id" if show_ids and "id" in df.columns else None,
        "color_discrete_sequence": px.colors.qualitative.G10,
    }

    if active_color == "cluster_str":
        base_kwargs["category_orders"] = {active_color: sorted(df[active_color].unique())}

    # 1. Modo 3D Habilitado
    if use_3d and z_dim != "None":
        fig = px.scatter_3d(**base_kwargs, x=x_dim, y=y_dim, z=z_dim)
        fig.update_traces(marker=dict(size=6, opacity=0.85))
        fig.update_layout(margin=dict(l=10, r=10, t=20, b=10), height=520)
        st.plotly_chart(fig, use_container_width=True, key=f"plotly_scatter_3d_{map_id}")

    # 2. Modo 2D Dual (Z no es None, pero no está en 3D) -> Dos gráficos X-Y y X-Z
    elif z_dim != "None":
        pcol1, pcol2 = st.columns(2)

        with pcol1:
            fig1 = px.scatter(**base_kwargs, x=x_dim, y=y_dim)
            fig1.update_traces(marker=dict(size=8, opacity=0.85))
            fig1.update_layout(margin=dict(l=10, r=10, t=20, b=10), height=440)
            st.plotly_chart(fig1, use_container_width=True, key=f"plotly_scatter_xy_{map_id}")

        with pcol2:
            fig2 = px.scatter(**base_kwargs, x=x_dim, y=z_dim)
            fig2.update_traces(marker=dict(size=8, opacity=0.85))
            fig2.update_layout(margin=dict(l=10, r=10, t=20, b=10), height=440)
            st.plotly_chart(fig2, use_container_width=True, key=f"plotly_scatter_xz_{map_id}")

    # 3. Modo 2D Estándar (Z es None)
    else:
        fig = px.scatter(**base_kwargs, x=x_dim, y=y_dim)
        fig.update_traces(marker=dict(size=8, opacity=0.85))
        fig.update_layout(margin=dict(l=10, r=10, t=20, b=10), height=460)
        st.plotly_chart(fig, use_container_width=True, key=f"plotly_scatter_{map_id}")


def render_bubble_chart(
    df: pd.DataFrame, dimensions: List[str], show_ids: bool, map_id: int, default_x: str, default_y: str
) -> None:
    """Renders Bubble Chart following the aligned header controls (X, Y, Size, Color)."""
    c1, c2, c3, c4 = st.columns(4)

    x_index = dimensions.index(default_x) if default_x in dimensions else 0
    y_index = dimensions.index(default_y) if default_y in dimensions else min(1, len(dimensions) - 1)

    x_dim = c1.selectbox("X Axis", dimensions, index=x_index, key=f"bubble_x_{map_id}")
    y_dim = c2.selectbox("Y Axis", dimensions, index=y_index, key=f"bubble_y_{map_id}")
    size_dim = c3.selectbox("Bubble Size Metric", dimensions, index=0, key=f"bubble_size_{map_id}")
    
    color_options = _get_color_options(df, dimensions)
    bubble_color_key = f"bubble_color_{map_id}"

    if "cluster_str" in df.columns:
        if bubble_color_key not in st.session_state or st.session_state[bubble_color_key] == "None":
            st.session_state[bubble_color_key] = "cluster_str"

    color_dim = c4.selectbox("Color Axis", color_options, key=bubble_color_key)
    active_color = None if color_dim == "None" else color_dim

    df_bubble = df.copy()
    min_val = df_bubble[size_dim].min()
    if min_val <= 0:
        df_bubble["_bubble_size"] = df_bubble[size_dim] - min_val + 1
    else:
        df_bubble["_bubble_size"] = df_bubble[size_dim]

    fig = px.scatter(
        df_bubble,
        x=x_dim,
        y=y_dim,
        size="_bubble_size",
        color=active_color,
        color_discrete_sequence=px.colors.qualitative.G10,
        hover_data=[size_dim] + (["id"] if "id" in df_bubble.columns else []),
        text="id" if show_ids and "id" in df_bubble.columns else None,
        category_orders={active_color: sorted(df_bubble[active_color].unique())}
        if active_color == "cluster_str"
        else None,
    )
    fig.update_layout(margin=dict(l=10, r=10, t=20, b=10), height=460)
    st.plotly_chart(fig, use_container_width=True, key=f"plotly_bubble_{map_id}")



def render_violin_chart(
    df: pd.DataFrame, dimensions: List[str], map_id: int
) -> None:
    """Renders Violin / Box distribution plot across clusters or groups."""
    c1, c2 = st.columns(2)

    target_dim = c1.selectbox("Dimension to Analyze", dimensions, key=f"violin_dim_{map_id}")
    plot_type = c2.radio("Plot Style", ["Violin", "Box"], horizontal=True, key=f"violin_style_{map_id}")

    group_col = "cluster_str" if "cluster_str" in df.columns else None

    if group_col:
        category_orders = {group_col: sorted(df[group_col].unique())}
        if plot_type == "Violin":
            fig = px.violin(
                df,
                x=group_col,
                y=target_dim,
                color=group_col,
                color_discrete_sequence=px.colors.qualitative.G10,
                box=True,
                points="all",
                hover_data=["id"] if "id" in df.columns else None,
                category_orders=category_orders,
            )
        else:
            fig = px.box(
                df,
                x=group_col,
                y=target_dim,
                color=group_col,
                color_discrete_sequence=px.colors.qualitative.G10,
                points="all",
                hover_data=["id"] if "id" in df.columns else None,
                category_orders=category_orders,
            )
    else:
        if plot_type == "Violin":
            fig = px.violin(df, y=target_dim, box=True, points="all")
        else:
            fig = px.box(df, y=target_dim, points="all")

    fig.update_layout(margin=dict(l=10, r=10, t=20, b=10), height=450)
    st.plotly_chart(fig, use_container_width=True, key=f"plotly_violin_{map_id}")


def render_parallel_coordinates(
    df: pd.DataFrame, dimensions: List[str], map_id: int
) -> None:
    """Renders Parallel Coordinates plot for multi-objective trade-offs."""
    selected_dims = st.multiselect(
        "Dimensions to include",
        dimensions,
        default=dimensions[: min(5, len(dimensions))],
        key=f"parallel_dims_{map_id}",
    )

    if len(selected_dims) < 2:
        st.info("Select at least 2 dimensions to render parallel coordinates.")
        return

    df_pc = df.copy()

    if "cluster_str" in df_pc.columns:
        df_pc["_color_code"] = pd.Categorical(df_pc["cluster_str"]).codes
        color_var = "_color_code"
    else:
        color_var = None

    fig = px.parallel_coordinates(
        df_pc,
        dimensions=selected_dims,
        color=color_var,
        color_continuous_scale=px.colors.diverging.Tealrose,
    )
    fig.update_layout(margin=dict(l=10, r=10, t=20, b=10), height=450)
    st.plotly_chart(fig, use_container_width=True, key=f"plotly_parcoords_{map_id}")


# =====================================================
# MAIN ENTRY POINT FOR DYNAMIC MAP PANELS
# =====================================================

def render_maps(
    df: pd.DataFrame, dimensions: List[str], show_ids: bool = False
) -> None:
    """
    Renders all dynamic Decision Space Map panels created via st.session_state.maps.
    """
    if df is None or df.empty or not dimensions:
        st.info("No data or dimensions available to render decision maps.")
        return

    # Garantizar presencia de la clave st.session_state.maps
    if "maps" not in st.session_state or not st.session_state.maps:
        st.session_state.maps = [
            {
                "id": 1,
                "x": dimensions[0],
                "y": dimensions[1] if len(dimensions) > 1 else dimensions[0],
                "chart_type": "Scatter Map",
            }
        ]

    # Iterar y renderizar cada mapa activo en session_state.maps
    for idx, map_item in enumerate(list(st.session_state.maps)):
        map_id = map_item.get("id", idx + 1)
        default_x = map_item.get("x", dimensions[0])
        default_y = map_item.get("y", dimensions[1] if len(dimensions) > 1 else dimensions[0])

        with st.expander(f"🗺️ Decision Map Panel #{idx + 1}", expanded=True):
            col_tabs, col_del = st.columns([0.88, 0.12])

            with col_del:
                if len(st.session_state.maps) > 1:
                    if st.button("🗑️ Remove", key=f"remove_map_{map_id}", use_container_width=True):
                        st.session_state.maps = [
                            m for m in st.session_state.maps if m.get("id", None) != map_id
                        ]
                        st.rerun()

            with col_tabs:
                chart_type = st.radio(
                    "Chart View Mode",
                    [
                        "Scatter Map",
                        "Bubble Chart",
                        "Violin / Distribution",
                        "Parallel Coordinates",
                    ],
                    horizontal=True,
                    key=f"map_type_radio_{map_id}",
                )

            st.markdown("---")

            # Renderizar el tipo de gráfico activo
            if chart_type == "Scatter Map":
                render_scatter_chart(df, dimensions, show_ids, map_id, default_x, default_y)
            elif chart_type == "Bubble Chart":
                render_bubble_chart(df, dimensions, show_ids, map_id, default_x, default_y)
            elif chart_type == "Violin / Distribution":
                render_violin_chart(df, dimensions, map_id)
            elif chart_type == "Parallel Coordinates":
                render_parallel_coordinates(df, dimensions, map_id)