## --------------------------------------------------------------------------------------
## core/framing.py
## --------------------------------------------------------------------------------------

import streamlit as st
import pandas as pd
from ui.phase_help import ( render_phase_help_icon )

def get_framing_dimensions( dataset ):
    return (
        dataset[ "metrics" ]
        +
        dataset[ "selected_indicators" ]
    )

def is_valid_numeric_dimension( df, column ):
    if column not in df.columns:
        return False
    if not pd.api.types.is_numeric_dtype( df[column] ):
        return False
    return True

def apply_dimension_filter( filtered_df, metric, selected_range ):
    return filtered_df[
        ( filtered_df[metric] >= selected_range[0] )
        &
        ( filtered_df[metric] <= selected_range[1] )
    ]

def render_framing_summary( original_df, filtered_df ):
    total_solutions = len( original_df )
    remaining_solutions = len( filtered_df )

    ratio = ( remaining_solutions /
        max( total_solutions, 1 )
    )
    st.progress( ratio )

    st.markdown(
        f"""
        <div style="text-align:center">
            <div style="font-size:0.9rem;color:gray;">
                Remaining Solutions
            </div>
            <div style="font-size:1.8rem;font-weight:bold;">
                {remaining_solutions}/{total_solutions}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.caption( f"{ratio:.0%} of the decision space is visible." )

def apply_framing( dataset ):

    df = dataset[ "df" ].copy()
    filtered_df = df.copy()
    dimensions = get_framing_dimensions( dataset )

    with st.sidebar.expander(
        "🎛️ Context Framing", expanded=False ):

        col_label, col_help = st.columns(  [ 0.85, 0.15 ],
            vertical_alignment="center"
        )
        
        with col_help:
            render_phase_help_icon( "framing",  key="help_input_phase" )

        for metric in dimensions:
            if not is_valid_numeric_dimension( df, metric ):
                continue
            min_v = float( df[metric].min() )
            max_v = float( df[metric].max() )

            if min_v == max_v:
                continue

            selected_range = st.slider(
                metric,
                min_value=min_v,
                max_value=max_v,
                value=( min_v,  max_v ),
                step=(  max_v -  min_v  ) / 1000,
                key=f"framing_{metric}"
            )

            unchanged = (
                abs( selected_range[0] - min_v ) < 1e-6
                and
                abs( selected_range[1] - max_v ) < 1e-6
            )
            if unchanged:
                continue
            filtered_df = apply_dimension_filter(  filtered_df,  metric,  selected_range )
        render_framing_summary( df, filtered_df )

    return filtered_df