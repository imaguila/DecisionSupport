## --------------------------------------------------------------------------------------
## core/workspace.py
## --------------------------------------------------------------------------------------

import streamlit as st

from core.workspace_summary import ( render_summary )
from core.workspace_maps import ( render_maps )

def get_workspace_dimensions( dataset ):
    return ( dataset["metrics"] + dataset["selected_indicators"] )

def render_empty_workspace_message():
    st.error( "No dataset is available for the workspace." )

def render_no_map_message():
    st.warning(
        "At least two dimensions are required "
        "to render decision-space maps." )

def render_workspace( df, dataset, show_ids ):
    if df is None:
        render_empty_workspace_message()
        return

    dimensions = get_workspace_dimensions( dataset )

    # ==================== SUMMARY + CURRENT SET =============================

    render_summary( df, dataset )

    # ====================== MAPS============================

    if len(dimensions) < 2:
        render_no_map_message()
    else:
        render_maps( df,  dataset, dimensions, show_ids )

