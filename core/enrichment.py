## --------------------------------------------------------------------------------------
## core/enrichment.py
## --------------------------------------------------------------------------------------

import streamlit as st
from ui.phase_help import ( render_phase_help_icon )

def get_available_indicators( plugin, selected_metrics ):
    available_indicators = []
    requirements = plugin.requirements()
    for indicator, reqs in requirements.items():
        if all( metric in selected_metrics for metric in reqs ):
            available_indicators.append( indicator )
    return available_indicators

def render_enrichment( dataset ):
    plugin = dataset[ "plugin" ]
    if plugin is None:
        dataset[ "selected_indicators" ] = []
        return dataset

    selected_metrics = dataset[ "metrics" ]
    available_indicators = get_available_indicators(
        plugin, selected_metrics )

    with st.sidebar.expander( "⚙️ Data Enrichment", expanded=False ):

        col_label, col_help = st.columns( [ 0.85, 0.15 ],
            vertical_alignment="center"
        )

        with col_label:
            st.markdown( "**Derived Indicators**" )

        with col_help:
            render_phase_help_icon("enrichment", key="help_enrichment_phase" )

        st.caption( f"Detected {len(available_indicators)} "
            "compatible indicators."
        )

        selected_indicators = st.multiselect(
            "Available indicators",
            sorted(
                available_indicators
            ),
            default=[
                indicator
                for indicator in dataset[
                    "config"
                ].get(
                    "default_indicators",
                    []
                )
                if indicator in available_indicators
            ]
        )

    dataset[ "df"] = plugin.compute_indicators( dataset[ "df" ],
        selected_indicators
    )
    
    dataset[ "selected_indicators" ] = selected_indicators

    return dataset