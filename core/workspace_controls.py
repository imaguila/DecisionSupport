## --------------------------------------------------------------------------------------
## core/workspace_controls.py
## --------------------------------------------------------------------------------------

import streamlit as st

def render_workspace_controls( dimensions ):

    with st.sidebar.expander(
        "🗺️ Visual Workspace", expanded=False ):

        if "maps" not in st.session_state:
            st.session_state.maps = []

        can_create_map = ( len(dimensions) >= 2 )
        col1, col2 = st.columns( [ 0.50, 0.50 ] )

        with col1:
            if st.button(
                "🔄 Reset Maps", use_container_width=True,
                disabled=not can_create_map ):

                st.session_state.maps = [ {
                    "x": dimensions[0], "y": dimensions[1], "z": None,
                    "color": None
                    }
                ]
                st.rerun()

        with col2:
            if st.button(
                "New Map",
                use_container_width=True,
                disabled=not can_create_map ):

                st.session_state.maps.append(
                    { "x": dimensions[0], "y": dimensions[1], "z": None,
                        "color": None
                    }
                )
                st.rerun()

        if not can_create_map:
            st.info(  "At least two dimensions are required to create maps." )
        show_ids = st.checkbox( "Show solution IDs",  value=False )
        st.caption(  f"Active maps: {len(st.session_state.maps)}" )

    return show_ids