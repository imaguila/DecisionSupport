## --------------------------------------------------------------------------------------
## input_panel.py
## --------------------------------------------------------------------------------------

import streamlit as st
import pandas as pd

from config import CASES
from plugins import PLUGIN_REGISTRY
from ui.phase_help import (
    render_phase_help_icon, render_help_icon
)

# =====================================================
# DETECTION / INFERENCE
# =====================================================

def detect_decision_variables( df, prefix ):
    return [ col for col in df.columns if col.startswith( prefix ) ]

def infer_numeric_metrics( df, cfg ):
    var_prefix = cfg.get( "var_prefix", "x_" )
    excluded = set( cfg.get( "exclude_cols", [] ) )

    system_cols = {
        "id",
        "ID",
        "cluster",
        "cluster_str",
        "group_label",
        "group_base",
        "label",
        "highlight",
        "highlight_label",
        "score",
        "preference_score",
        "preference_rank",
        "efficiency_score",
        "efficiency_rank",
        "domain_match_count",
        "domain_rank",
        "selected"
    }

    metrics = []

    for col in df.columns:
        if col.startswith( var_prefix ):
            continue
        if col in excluded:
            continue
        if col in system_cols:
            continue
        if pd.api.types.is_numeric_dtype( df[col] ):
            metrics.append( col )
            
    return metrics

# ==================== PLUGIN / DATASET BUILDING ========================

def build_plugin( cfg ):
    plugin = None
    plugin_name = cfg.get(  "plugin"  )
    if plugin_name:
        plugin_class = PLUGIN_REGISTRY.get( plugin_name )
        if plugin_class is not None:
            plugin = plugin_class( var_prefix=cfg.get( "var_prefix",  "x_" ) )
    return plugin

def build_dataset( df, cfg ):
    plugin = build_plugin( cfg )

    all_metrics = cfg.get( "metrics", [] )

    if not all_metrics:
        all_metrics = infer_numeric_metrics( df, cfg )

    selected_metrics = st.multiselect(
        "Objective Columns",
        all_metrics,
        default=all_metrics
    )

    decision_variables = detect_decision_variables(
        df,
        cfg.get( "var_prefix", "x_" )
    )

    return {
        "df": df,
        "config": cfg,
        "plugin": plugin,
        "metrics": selected_metrics,
        "selected_indicators": [],
        "decision_variables": decision_variables
    }

# =================== LOADERS ===================

def load_builtin_dataset( cfg ):
    df = pd.read_csv( cfg["path_sol"] )
    df = df.reset_index( drop=True )
    df["id"] = range( 1,  len(df) + 1 )

    return df

def load_uploaded_dataset( uploaded_file ):
    df = pd.read_csv( uploaded_file )
    df = df.reset_index( drop=True )
    df["id"] = range( 1, len(df) + 1 )

    return df


# =================== MAIN INPUT PANEL ================

def render_input_panel():

    with st.sidebar.expander(
        "🏷️ Input and Preparation", expanded=True
    ):

        col_label, col_help = st.columns(  [ 0.85, 0.15 ],
            vertical_alignment="center"
        )

        with col_label:
            st.markdown( "**Data Source**" )

        with col_help:
            render_phase_help_icon( "input",  key="help_input_phase" )

        mode = st.radio(
            "Data Source",
            [
                "1. Domain Configuration",
                "2. Upload Enriched CSV"
            ],
            horizontal=True,
            label_visibility="collapsed"
        )

        if mode == "1. Domain Configuration":
            return render_domain_configuration_input()
        return render_uploaded_csv_input()
    
# ======================== DOMAIN CONFIGURATION INPUT ==================

def render_domain_configuration_input():

    dataset_names = [ "-- No Data --" ] + list( CASES.keys() )

    col_dataset, col_help = st.columns( [ 0.85, 0.15 ],
        vertical_alignment="bottom"
    )

    with col_dataset:
        dataset_name = st.selectbox(
            "Domain Configuration",
            dataset_names,
            key="input_domain_configuration"
        )

    if dataset_name == "-- No Data --":
        dataset_help = (
            "No domain configuration selected yet.\n\n"
            "Choose a predefined case to load its dataset, objectives, "
            "decision-variable prefix, and optional plugin."
        )

    else:
        dataset_help = CASES[
            dataset_name
        ].get(
            "help",
            "No additional description is available for this domain configuration."
        )

    with col_help:
        render_help_icon( dataset_help,  key="help_domain_configuration" )

    if dataset_name == "-- No Data --":
        st.info( "Select data to continue." )

        return None

    cfg = CASES[ dataset_name ]

    try:
        df = load_builtin_dataset( cfg )

    except Exception as exc:
        st.error( f"Unable to load dataset: {cfg.get('path_sol')}" )
        st.exception( exc )

        return None

    return build_dataset( df, cfg )

# ===================== UPLOADED CSV INPUT ====================

def render_uploaded_csv_input():
    uploaded_file = st.file_uploader( "Upload CSV", type=[ "csv" ] )

    if uploaded_file is None:
        return None

    var_prefix = st.text_input(  "Decision-variable prefix",  value="var_" )

    try:
        df = load_uploaded_dataset( uploaded_file )

    except Exception as exc:
        st.error( "Unable to load uploaded CSV." )
        st.exception( exc )

        return None

    cfg = {
        "plugin": None,
        "metrics": [],
        "var_prefix": var_prefix,
        "exclude_cols": [],
        "default_indicators": [],
        "help": "Uploaded enriched CSV."
    }

    return build_dataset(  df,  cfg )