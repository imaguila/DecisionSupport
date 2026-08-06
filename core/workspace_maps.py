"""
Workspace Maps Module.

Provides layout and rendering controls for decision-space visualization maps, 
supporting scatter plots, coordinated dual-maps, bubble charts, and 
distribution views (violin/box plots).
"""

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from ui.visualization import (
    render_coordinated_maps,
    render_distribution,
    render_scatter,
)


# =====================================================
# MAP STATE MANAGEMENT
# =====================================================


def ensure_valid_map_state(
    current_map: Dict[str, Any], dimensions: List[str]
) -> Dict[str, Any]:
    """
    Validates and updates map dimension keys against available dataset dimensions.

    Parameters
    ----------
    current_map : Dict[str, Any]
        Dictionary storing state (x, y, z, color) for a specific map instance.
    dimensions : List[str]
        List of currently available dimension column names.

    Returns
    -------
    Dict[str, Any]
        Validated map state dictionary.
    """
    if not dimensions:
        return current_map

    if current_map.get("x") not in dimensions:
        current_map["x"] = dimensions[0]

    y_options = [dim for dim in dimensions if dim != current_map["x"]]
    if not y_options:
        y_options = dimensions

    if current_map.get("y") not in y_options:
        current_map["y"] = y_options[0]

    z_options = [None] + [
        dim
        for dim in dimensions
        if dim not in (current_map["x"], current_map["y"])
    ]

    if current_map.get("z") not in z_options:
        current_map["z"] = None

    if "color" not in current_map:
        current_map["color"] = None

    return current_map


# =====================================================
# AXIS & UI CONTROLS
# =====================================================


def render_axis_controls(
    idx: int,
    current_map: Dict[str, Any],
    dimensions: List[str],
    map_mode: str,
) -> Tuple[str, str, Optional[str], Optional[str]]:
    """
    Renders column selectors for assigning axes and encodings based on active map mode.

    Parameters
    ----------
    idx : int
        Index of the active decision-space map.
    current_map : Dict[str, Any]
        Active map state dictionary.
    dimensions : List[str]
        List of available metric and indicator dimensions.
    map_mode : str
        Active visualization mode ("🗺️ Scatter", "🫧 Bubble", etc.).

    Returns
    -------
    Tuple[str, str, Optional[str], Optional[str]]
        Selected (x, y, z, color) dimension names.
    """
    if map_mode == "🗺️ Scatter":
        col1, col2, col3 = st.columns(3)
    else:
        col1, col2, col3, col4 = st.columns(4)

    current_x = (
        current_map["x"]
        if current_map.get("x") in dimensions
        else dimensions[0]
    )

    with col1:
        x = st.selectbox(
            "X Axis",
            dimensions,
            index=dimensions.index(current_x),
            key=f"x_{idx}",
        )

    y_options = [dim for dim in dimensions if dim != x]
    if not y_options:
        y_options = dimensions

    current_y = (
        current_map["y"]
        if current_map.get("y") in y_options
        else y_options[0]
    )

    with col2:
        y = st.selectbox(
            "Y Axis",
            y_options,
            index=y_options.index(current_y),
            key=f"y_{idx}",
        )

    z_options = [None] + [dim for dim in dimensions if dim not in (x, y)]

    current_z = (
        current_map["z"] if current_map.get("z") in z_options else None
    )

    with col3:
        z = st.selectbox(
            "Third Dimension",
            z_options,
            index=z_options.index(current_z),
            key=f"z_{idx}",
        )

    color = current_map.get("color")

    if map_mode == "🫧 Bubble":
        with col4:
            color_options = [None] + dimensions
            current_color = color if color in color_options else None

            color = st.selectbox(
                "Color",
                color_options,
                index=color_options.index(current_color),
                key=f"color_{idx}",
            )
    else:
        color = None

    return x, y, z, color


def render_distribution_controls(
    df: pd.DataFrame, idx: int, dimensions: List[str]
) -> None:
    """
    Renders dimension selection and view mode toggles for statistical distribution plots.

    Parameters
    ----------
    df : pd.DataFrame
        Active solution space DataFrame.
    idx : int
        Index of the active decision-space map.
    dimensions : List[str]
        List of available metric and indicator dimensions.
    """
    if not dimensions:
        st.warning("No dimensions available for distribution analysis.")
        return

    view_type = st.radio(
        "View",
        ["Violin", "Box"],
        horizontal=True,
        key=f"dist_mode_{idx}",
    )

    distribution_metric = st.selectbox(
        "Dimension",
        dimensions,
        key=f"distribution_{idx}",
    )

    render_distribution(
        df,
        metric=distribution_metric,
        mode=view_type,
        key=f"distribution_plot_{idx}",
    )


# =====================================================
# SCATTER & BUBBLE RENDERERS
# =====================================================


def render_scatter_or_bubble(
    df: pd.DataFrame,
    idx: int,
    x: str,
    y: str,
    z: Optional[str],
    color: Optional[str],
    map_mode: str,
    show_ids: bool,
) -> None:
    """
    Routes rendering calls to single scatter, coordinated dual scatter, or bubble charts.

    Parameters
    ----------
    df : pd.DataFrame
        Active solution space DataFrame.
    idx : int
        Index of the active decision-space map.
    x : str
        Target x-axis dimension.
    y : str
        Target y-axis dimension.
    z : Optional[str]
        Target z-axis or bubble size dimension.
    color : Optional[str]
        Target color encoding column.
    map_mode : str
        Active map mode ("🗺️ Scatter" or "🫧 Bubble").
    show_ids : bool
        Whether to display solution ID labels.
    """
    if map_mode == "🗺️ Scatter":
        if z is None:
            render_scatter(
                df,
                x=x,
                y=y,
                color=None,
                show_ids=show_ids,
                key=f"single_{idx}",
            )
        else:
            render_coordinated_maps(
                df,
                x=x,
                y=y,
                z=z,
                key_prefix=f"coord_{idx}",
                show_ids=show_ids,
            )
    else:
        render_scatter(
            df,
            x=x,
            y=y,
            size=z,
            color=color,
            show_ids=show_ids,
            key=f"bubble_{idx}",
        )


# =====================================================
# MAIN PIPELINE ENTRY POINT
# =====================================================


def render_maps(
    df: pd.DataFrame,
    dataset: Dict[str, Any],
    dimensions: List[str],
    show_ids: bool = False,
) -> None:
    """
    Main entry point for rendering active decision-space maps stored in session state.

    Parameters
    ----------
    df : pd.DataFrame
        Active solution space DataFrame.
    dataset : Dict[str, Any]
        Global dataset context dictionary.
    dimensions : List[str]
        List of filterable and renderable dataset dimensions.
    show_ids : bool, default=False
        Whether to display solution ID text labels.
    """
    if "maps" not in st.session_state:
        st.session_state.maps = []

    if len(st.session_state.maps) == 0:
        st.info(
            "No decision maps have been created yet. "
            "Use 'New Map' in the Visual Workspace panel."
        )
        return

    if not dimensions or len(dimensions) < 2:
        st.warning("At least two dimensions are required to display maps.")
        return

    for idx in range(len(st.session_state.maps)):
        current_map = st.session_state.maps[idx]
        current_map = ensure_valid_map_state(current_map, dimensions)

        with st.expander(
            f"🗺️ Decision-Space Map {idx + 1}",
            expanded=(idx == 0),
        ):
            map_mode = st.radio(
                "Visualization Mode",
                [
                    "🗺️ Scatter",
                    "🫧 Bubble",
                    "📈 Distribution",
                ],
                horizontal=True,
                key=f"map_mode_{idx}",
            )

            if map_mode in ["🗺️ Scatter", "🫧 Bubble"]:
                x, y, z, color = render_axis_controls(
                    idx, current_map, dimensions, map_mode
                )
                render_scatter_or_bubble(
                    df, idx, x, y, z, color, map_mode, show_ids
                )
            else:
                x = current_map["x"]
                y = current_map["y"]
                z = None
                color = None
                render_distribution_controls(df, idx, dimensions)

            st.session_state.maps[idx] = {
                "x": x,
                "y": y,
                "z": z,
                "color": color,
            }