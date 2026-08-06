## --------------------------------------------------------------------------------------
## core/workspace_maps.py
## --------------------------------------------------------------------------------------

import streamlit as st

from ui.visualization import (
    render_scatter,
    render_coordinated_maps,
    render_distribution
)


# =====================================================
# MAP STATE
# =====================================================

def ensure_valid_map_state(
    current_map,
    dimensions
):

    if current_map.get(
        "x"
    ) not in dimensions:

        current_map[
            "x"
        ] = dimensions[0]

    y_options = [
        dimension
        for dimension in dimensions
        if dimension != current_map["x"]
    ]

    if current_map.get(
        "y"
    ) not in y_options:

        current_map[
            "y"
        ] = y_options[0]

    z_options = [
        None
    ] + [
        dimension
        for dimension in dimensions
        if dimension not in [
            current_map["x"],
            current_map["y"]
        ]
    ]

    if current_map.get(
        "z"
    ) not in z_options:

        current_map[
            "z"
        ] = None

    if "color" not in current_map:

        current_map[
            "color"
        ] = None

    return current_map


# =====================================================
# AXIS CONTROLS
# =====================================================

def render_axis_controls(
    idx,
    current_map,
    dimensions,
    map_mode
):

    if map_mode == "🗺️ Scatter":

        col1, col2, col3 = st.columns(
            3
        )

    else:

        col1, col2, col3, col4 = st.columns(
            4
        )

    with col1:

        current_x = (
            current_map["x"]
            if current_map["x"] in dimensions
            else dimensions[0]
        )

        x = st.selectbox(
            "X Axis",
            dimensions,
            index=dimensions.index(
                current_x
            ),
            key=f"x_{idx}"
        )

    y_options = [
        dimension
        for dimension in dimensions
        if dimension != x
    ]

    with col2:

        current_y = (
            current_map["y"]
            if current_map["y"] in y_options
            else y_options[0]
        )

        y = st.selectbox(
            "Y Axis",
            y_options,
            index=y_options.index(
                current_y
            ),
            key=f"y_{idx}"
        )




    z_options = [
        None
    ] + [
        dimension
        for dimension in dimensions
        if dimension not in [
            x,
            y
        ]
    ]

    with col3:

        current_z = (
            current_map["z"]
            if current_map["z"] in z_options
            else None
        )

        z = st.selectbox(
            "Third Dimension",
            z_options,
            index=z_options.index(
                current_z
            ),
            key=f"z_{idx}"
        )

    color = current_map.get(
        "color"
    )

    if map_mode == "🫧 Bubble":

        with col4:

            color_options = [
                None
            ] + dimensions

            current_color = (
                color
                if color in color_options
                else None
            )

            color = st.selectbox(
                "Color",
                color_options,
                index=color_options.index(
                    current_color
                ),
                key=f"color_{idx}"
            )

    else:

        color = None

    return x, y, z, color





# =====================================================
# RENDER SCATTER / BUBBLE / DISTRIBUTION
# =====================================================

def render_scatter_or_bubble(
    df,
    idx,
    x,
    y,
    z,
    color,
    map_mode,
    show_ids
):

    if map_mode == "🗺️ Scatter":

        if z is None:

            render_scatter(
                df,
                x=x,
                y=y,
                color=None,
                show_ids=show_ids,
                key=f"single_{idx}"
            )

        else:

            render_coordinated_maps(
                df,
                x=x,
                y=y,
                z=z,
                key_prefix=f"coord_{idx}",
                show_ids=show_ids
            )

    else:

        render_scatter(
            df,
            x=x,
            y=y,
            size=(
                z
                if z is not None
                else None
            ),
            color=color,
            show_ids=show_ids,
            key=f"bubble_{idx}"
        )


def render_distribution_controls(
    df,
    idx,
    dimensions
):

    view_type = st.radio(
        "View",
        [
            "Violin",
            "Box"
        ],
        horizontal=True,
        key=f"dist_mode_{idx}"
    )

    distribution_metric = st.selectbox(
        "Dimension",
        dimensions,
        key=f"distribution_{idx}"
    )

    render_distribution(
        df,
        metric=distribution_metric,
        mode=view_type,
        key=f"distribution_plot_{idx}"
    )


# =====================================================
# MAIN MAP RENDERER
# =====================================================

def render_maps(
    df,
    dataset,
    dimensions,
    show_ids
):

    if "maps" not in st.session_state:

        st.session_state.maps = []

    if len(st.session_state.maps) == 0:

        st.info(
            "No decision maps have been created yet. "
            "Use 'New Map' in the Visual Workspace panel."
        )

        return

    if len(dimensions) < 2:

        st.warning(
            "At least two dimensions are required."
        )

        return

    for idx in range(
        len(st.session_state.maps)
    ):

        current_map = st.session_state.maps[
            idx
        ]

        current_map = ensure_valid_map_state(
            current_map,
            dimensions
        )

        with st.expander(
            f"🗺️ Decision-Space Map {idx + 1}",
            expanded=(
                idx == 0
            )
        ):

            map_mode = st.radio(
                "Visualization Mode",
                [
                    "🗺️ Scatter",
                    "🫧 Bubble",
                    "📈 Distribution"
                ],
                horizontal=True,
                key=f"map_mode_{idx}"
            )

            if map_mode in [
                "🗺️ Scatter",
                "🫧 Bubble"
            ]:

                x, y, z, color = render_axis_controls(
                    idx,
                    current_map,
                    dimensions,
                    map_mode
                )

                render_scatter_or_bubble(
                    df,
                    idx,
                    x,
                    y,
                    z,
                    color,
                    map_mode,
                    show_ids
                )

            else:

                x = current_map["x"]
                y = current_map["y"]
                z = None
                color = None

                render_distribution_controls(
                    df,
                    idx,
                    dimensions
                )

            st.session_state.maps[idx] = {
                "x": x,
                "y": y,
                "z": z,
                "color": color
            }

            