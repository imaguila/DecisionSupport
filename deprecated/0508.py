
# --- ARCHIVO: __init__.py ---

# plugins/__init__.py

from .nrp_plugin import NRPPlugin
from .aerospace_plugin import AerospacePlugin

PLUGIN_REGISTRY = {
    "nrp": NRPPlugin,
    "aerospace": AerospacePlugin
}

# --- ARCHIVO: aerospace_plugin.py ---

## --------------------------------------------------------------------------------------
## plugins/aerospace_plugin.py


import numpy as np

EPS = 1e-9


class AerospacePlugin:
    """
    Demo aerospace plugin.

    Provides synthetic engineering indicators
    derived from aerodynamic objectives and
    design variables.

    The goal is not physical accuracy but
    demonstrating how a new domain can extend
    the framework through plugins.
    """

    def __init__(self, var_prefix="var_"):

        self.var_prefix = var_prefix

    # --------------------------------------------------
    # Available indicators
    # --------------------------------------------------

    def available_indicators(self):

        return {
            "density",
            "lift_to_drag_ratio",
            "structural_efficiency",
        }

    # --------------------------------------------------
    # Indicator dependencies
    # --------------------------------------------------

    def requirements(self):

        return {

            "density": [
                "weight"
            ],

            "lift_to_drag_ratio": [
                "drag",
                "weight"
            ],

            "structural_efficiency": [
                "drag",
                "weight"
            ]
        }

    # --------------------------------------------------
    # Design variables
    # --------------------------------------------------

    def decision_variables(self, df):

        return [
            c
            for c in df.columns
            if c.startswith(self.var_prefix)
        ]

    # --------------------------------------------------
    # Indicator computation
    # --------------------------------------------------

    def compute_indicators(
        self,
        df,
        selected_indicators
    ):

        result = df.copy()

        vars_cols = self.decision_variables(
            result
        )

        n_vars = max(
            len(vars_cols),
            1
        )

        for indicator in selected_indicators:

            try:

                # ==================================
                # Density
                # ==================================

                if indicator == "density":

                    result[indicator] = (
                        result["weight"]
                        / n_vars
                    )

                # ==================================
                # Lift-to-drag ratio
                # ==================================

                elif (
                    indicator
                    == "lift_to_drag_ratio"
                ):

                    pseudo_lift = (
                        result["weight"]
                        * 0.25
                    )

                    result[indicator] = (
                        pseudo_lift
                        /
                        np.maximum(
                            result["drag"],
                            EPS
                        )
                    )

                # ==================================
                # Structural efficiency
                # ==================================

                elif (
                    indicator
                    == "structural_efficiency"
                ):

                    result[indicator] = (

                        1.0

                        /

                        (
                            (
                                result["drag"]
                                /
                                result["drag"].max()
                            )

                            *

                            (
                                result["weight"]
                                /
                                result["weight"].max()
                            )

                            +

                            EPS
                        )
                    )

            except Exception as exc:

                print(
                    "[AerospacePlugin] "
                    f"Unable to compute "
                    f"{indicator}: {exc}"
                )

        return result

# --- ARCHIVO: base_plugin.py ---

# plugins/base_plugin.py

from abc import ABC, abstractmethod


class DomainPlugin(ABC):

    @abstractmethod
    def available_indicators(self):
        pass

    @abstractmethod
    def compute_indicators(self, df, indicators):
        pass

# --- ARCHIVO: column_classifier.py ---

## --------------------------------------------------------------------------------------
## column_classifier.py

import pandas as pd

class ColumnClassifier:
    """
    Handles dynamic column classification and exclusions based on problem configuration.
    Categorizes dataset attributes into Decision Variables, Base Metrics, and Derived Indicators.
    """
    
    def __init__(self, config: dict):
        self.metrics = set(config.get("metrics", []))
        self.var_prefix = config.get("var_prefix", "x_")
        self.user_excludes = set(config.get("exclude_cols", []))
        
        # Internal system-level columns generated dynamically by the framework
        self.system_excludes = {"highlight", "label", "highlight_label", "score", "cluster", "selected"}

    def get_decision_variables(self, df: pd.DataFrame) -> list:
        """Extracts decision variable columns (X) using the configured prefix."""
        return [col for col in df.columns if col.startswith(self.var_prefix)]

    def get_metrics(self, df: pd.DataFrame) -> list:
        """Extracts base optimization metrics (M) defined in the configuration."""
        return [col for col in df.columns if col in self.metrics]

    def get_derived_indicators(self, df: pd.DataFrame) -> list:
        """
        Extracts derived/enrichment indicators (I).
        Identifies numeric columns that are neither base metrics, decision variables, nor excluded attributes.
        """
        all_excluded = self.system_excludes | self.user_excludes | self.metrics
        
        indicators = []
        for col in df.columns:
            if col in all_excluded or col.startswith(self.var_prefix):
                continue
            # If the column is numeric and passed all exclusion filters, treat it as a derived lens/indicator
            if pd.api.types.is_numeric_dtype(df[col]):
                indicators.append(col)
                
        return indicators

# --- ARCHIVO: config.py ---

## --------------------------------------------------------------------------------------
## config.py
## --------------------------------------------------------------------------------------

"""
Example configurations shipped with the framework.

A configuration describes:

1. How a Pareto front is loaded.
2. Which columns correspond to optimization objectives.
3. How decision variables are identified.
4. Which domain plugin should be used.
5. Which enrichment indicators are available by default.

Users may:
- use an existing configuration,
- define their own configuration,
- or provide an already enriched dataset with no plugin.
"""

CASES = {

    # =====================================================================
    # CASE 1
    # Software Release Planning - CLASSIC Dataset
    # =====================================================================

    "CLASSIC Dataset": {
        "plugin": "nrp",
        "path_sol": "data/bagnallsoluciones.csv",
        "metrics": [
            "satisfaction",
            "effort"
        ],
        "var_prefix": "req_",
        "num_x": 18,
        "exclude_cols": [
            "id",
            "run_id",
            "timestamp"
        ],

        "default_indicators": [
            "scope",
            "productivity",
            "squandering"
        ],

        "help": (
            "Greer, D., & Ruhe, G. (2004). "
            "Software release planning: an evolutionary "
            "and iterative approach. Information and Software "
            "Technology, 46(4), 243-253."
        )
    },

    # =====================================================================
    # CASE 2
    # Software Release Planning - MSLite System
    # =====================================================================

    "MSLite System": {
        "plugin": "nrp",
        "path_sol": "data/mslitesoluciones.csv",
        "metrics": [
            "satisfaction",
            "effort",
            "dissatisfaction"
        ],
        "var_prefix": "req_",
        "num_x": 16,
        "exclude_cols": [
            "id",
            "run_id",
            "timestamp"
        ],

        "default_indicators": [
            "scope",
            "productivity",
            "squandering",
            "annoyance",
            "dirtiness"
        ],

        "help": (
            "Sangwan, R. S., Negahban, A., Nord, R. L., "
            "& Ozkaya, I. (2020). Optimization of software "
            "release planning considering architectural "
            "dependencies, cost, and value. "
            "IEEE Transactions on Software Engineering, "
            "48(4), 1369-1384."
        )
    },

    # =====================================================================
    # CASE 3
    # Replacement Access, Library and ID Card - RALIC
    # =====================================================================

    "Replacement Access, Library and ID Card (RALIC)": {
        "plugin": "nrp",
        "path_sol": "data/ralic.csv",
        "metrics": [
            "satisfaction",
            "effort"
        ],
        "var_prefix": "req_",
        "num_x": 83,
        "exclude_cols": [
            "id",
            "run_id",
            "timestamp"
        ],
        "default_indicators": [
            "scope",
            "productivity",
            "squandering"
        ],
        "help": (
            "Lim, S. L., & Finkelstein, A. (2011). "
            "StakeRare: using social networks and collaborative "
            "filtering for large-scale requirements elicitation. "
            "IEEE Transactions on Software Engineering, "
            "38(3), 707-735."
        )
    },

    # =====================================================================
    # CASE 4
    # Word Processing Software Project
    # =====================================================================

    "Word Processing Software Project": {
        "plugin": "nrp",
        "path_sol": "data/wordprocsoluciones.csv",
        "metrics": [
            "satisfaction",
            "effort",
            "time"
        ],
        "var_prefix": "req_",
        "num_x": 42,
        "exclude_cols": [
            "id",
            "run_id",
            "timestamp"
        ],
        "default_indicators": [
            "scope",
            "productivity",
            "squandering",
            "response",
            "opportunity"
        ],
        "help": (
            "Agarwal, N., Karimpour, R., & Ruhe, G. (2014). "
            "Theme-based product release planning: An analytical "
            "approach. In 2014 47th Hawaii International Conference "
            "on System Sciences, pp. 4739-4748. IEEE."
        )
    },

    # =====================================================================
    # CASE 5
    # Large Dataset - REQ100
    # =====================================================================

    "Large Dataset": {
        "plugin": "nrp",
        "path_sol": "data/req100frente.csv",
        "metrics": [
            "satisfaction",
            "effort"
        ],
        "var_prefix": "req_",
        "num_x": 96,
        "exclude_cols": [
            "id",
            "run_id",
            "timestamp"
        ],
        "default_indicators": [
            "scope",
            "productivity",
            "squandering"
        ],

        "help": (
            "Del Sagrado, J., Del Águila, I. M., "
            "& Orellana, F. J. (2015). Multi-objective "
            "ant colony optimization for requirements selection. "
            "Empirical Software Engineering, 20(3), 577-610."
        )
    },

    # =====================================================================
    # CASE 6
    # ReleasePlanner Dataset - THEME
    # =====================================================================

    "ReleasePlanner™ Dataset": {
        "plugin": "nrp",
        "path_sol": "data/themesoluciones.csv",
        "metrics": [
            "satisfaction",
            "prevalence",
            "cost",
            "dissatisfaction",
            "inestability",
            "effort"
        ],
        "var_prefix": "req_",
        "num_x": 22,
        "exclude_cols": [
            "id",
            "run_id",
            "timestamp"
        ],
        "default_indicators": [
            "scope",
            "productivity",
            "squandering",
            "effectiveness",
            "dirtiness",
            "annoyance",
            "stickiness",
            "fragility",
            "robustness",
            "usage_efficiency"
        ],

        "help": (
            "Karim, M. R., & Ruhe, G. (2014). "
            "Bi-objective genetic search for release planning "
            "in support of themes. In International Symposium "
            "on Search Based Software Engineering, pp. 123-137. "
            "Springer International Publishing."
        )
    },

    # =====================================================================
    # CASE 7
    # Motorola Dataset
    # =====================================================================

    "Motorola Dataset": {
        "plugin": "nrp",
        "path_sol": "data/motorolasoluciones.csv",
        "metrics": [
            "satisfaction",
            "effort"
        ],
        "var_prefix": "req_",
        "num_x": 35,
        "exclude_cols": [
            "id",
            "run_id",
            "timestamp"
        ],
        "default_indicators": [
            "scope",
            "productivity",
            "squandering"
        ],
        "help": (
            "Baker, P., Harman, M., Steinhofel, K., "
            "& Skaliotis, A. (2006). Search based approaches "
            "to component selection and prioritization for the "
            "next release problem. In 2006 22nd IEEE International "
            "Conference on Software Maintenance, pp. 176-185. IEEE."
        )
    },

    # =====================================================================
    # CASE 8
    # Generic Engineering Design - Aerospace Wing Design
    # =====================================================================

    "Aerospace Wing Design": {
        "plugin": "aerospace",
        "path_sol": "data/wing_pareto_front.csv",
        "metrics": [
            "drag",
            "weight"
        ],
        "var_prefix": "var_",
        "num_x": 10,
        "exclude_cols": [
            "sim_time",
            "solver_status"
        ],
        "default_indicators": [
            "density",
            "lift_to_drag_ratio",
            "structural_efficiency"
        ],
        "help": (
            "Example, A. et al. (2025). "
            "Multi-objective aerodynamic design optimization "
            "of aircraft wings. Journal of Aircraft, "
            "62(1), 100-115."
        )
    }
}

# --- ARCHIVO: css_comparison.py ---

## --------------------------------------------------------------------------------------
## css/css_comparison.py
## --------------------------------------------------------------------------------------

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# =====================================================
# BASIC HELPERS
# =====================================================

def get_numeric_dimensions(
    df,
    dataset
):

    dimensions = (
        dataset["metrics"]
        +
        dataset["selected_indicators"]
    )

    return [
        col
        for col in dimensions
        if (
            col in df.columns
            and pd.api.types.is_numeric_dtype(
                df[col]
            )
        )
    ]


def get_decision_variable_columns(
    df,
    dataset
):

    var_prefix = (
        dataset["config"]
        .get(
            "var_prefix",
            "x_"
        )
    )

    return [
        col
        for col in df.columns
        if col.startswith(
            var_prefix
        )
    ]


def normalize_metric(
    series,
    goal
):

    min_v = series.min()
    max_v = series.max()

    if max_v <= min_v:

        return pd.Series(
            0.5,
            index=series.index
        )

    normalized = (
        series
        -
        min_v
    ) / (
        max_v
        -
        min_v
    )

    if goal == "Minimize":

        normalized = (
            1.0
            -
            normalized
        )

    return normalized

# =====================================================
# TRADE-OFF RADAR
# =====================================================

def render_tradeoff_radar(
    compare_df,
    css_df,
    dataset
):

    numeric_dimensions = get_numeric_dimensions(
        css_df,
        dataset
    )

    if len(numeric_dimensions) < 3:

        st.info(
            "At least three numeric objectives or indicators are required "
            "to create a radar chart."
        )

        return

    selected_metrics = st.multiselect(
        "Objectives and indicators for radar profile",
        numeric_dimensions,
        default=numeric_dimensions[
            :min(
                5,
                len(numeric_dimensions)
            )
        ],
        key="css_tradeoff_metrics"
    )

    if len(selected_metrics) < 3:

        st.warning(
            "Select at least three objectives or indicators."
        )

        return

    metric_goals = {}

    cols = st.columns(
        len(selected_metrics)
    )

    for idx, metric in enumerate(
        selected_metrics
    ):

        col = cols[idx]

        with col:

            metric_goals[
                metric
            ] = st.selectbox(
                metric,
                [
                    "Maximize",
                    "Minimize"
                ],
                key=f"css_goal_{metric}"
            )

    radar_df = compare_df.copy()

    for metric in selected_metrics:

        radar_df[
            metric
        ] = normalize_metric(
            radar_df[metric],
            metric_goals[metric]
        )

    fig = go.Figure()

    for _, row in radar_df.iterrows():

        values = row[
            selected_metrics
        ].tolist()

        values.append(
            values[0]
        )

        theta = (
            selected_metrics
            +
            [
                selected_metrics[0]
            ]
        )

        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=theta,
                mode="lines+markers",
                name=f"ID {int(row['id'])}"
            )
        )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[
                    0,
                    1
                ]
            )
        ),
        showlegend=True,
        template="plotly_white",
        height=520
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# =====================================================
# DECISION-VARIABLE MATRIX
# =====================================================

def render_decision_variable_matrix(
    compare_df,
    dataset
):

    variable_cols = get_decision_variable_columns(
        compare_df,
        dataset
    )

    var_prefix = (
        dataset["config"]
        .get(
            "var_prefix",
            "x_"
        )
    )

    if not variable_cols:

        st.info(
            f"No decision-variable columns with prefix "
            f"'{var_prefix}' found in the current CSS."
        )

        return

    matrix_df = (
        compare_df
        .set_index(
            "id"
        )[variable_cols]
        .copy()
    )

    matrix_df.index = [
        f"ID {int(idx)}"
        for idx in matrix_df.index
    ]

    fig = px.imshow(
        matrix_df,
        labels=dict(
            x="Decision variables",
            y="Solutions",
            color="Value"
        ),
        color_continuous_scale=[
            [
                0,
                "#e0e0e0"
            ],
            [
                1,
                "#00e676"
            ]
        ]
    )

    fig.update_layout(
        template="plotly_white",
        coloraxis_showscale=False,
        xaxis=dict(
            tickangle=-45,
            showgrid=False
        ),
        yaxis=dict(
            autorange="reversed",
            showgrid=False
        ),
        height=520
    )

    fig.update_traces(
        xgap=3,
        ygap=3,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Variable: %{x}<br>"
            "Value: %{z}"
            "<extra></extra>"
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # =====================================================
# DECISION-VARIABLE DISTRIBUTION
# =====================================================

def render_decision_variable_distribution(
    css_df,
    dataset
):

    variable_cols = get_decision_variable_columns(
        css_df,
        dataset
    )

    var_prefix = (
        dataset["config"]
        .get(
            "var_prefix",
            "x_"
        )
    )

    if not variable_cols:

        st.info(
            f"No decision-variable columns with prefix "
            f"'{var_prefix}' found in the current CSS."
        )

        return

    variable_summary = (
        css_df[variable_cols]
        .mean()
        .reset_index()
    )

    variable_summary.columns = [
        "decision_variable",
        "selection_rate"
    ]

    variable_summary = variable_summary.sort_values(
        "selection_rate",
        ascending=False
    )

    max_variables = min(
        50,
        len(variable_summary)
    )

    if max_variables < 1:

        st.info(
            "No decision variables can be summarized."
        )

        return

    top_n = st.slider(
        "Decision variables to show",
        min_value=1,
        max_value=max_variables,
        value=min(
            20,
            max_variables
        ),
        key="css_decision_variable_top_n"
    )

    plot_df = variable_summary.head(
        top_n
    )

    fig = px.bar(
        plot_df,
        x="decision_variable",
        y="selection_rate",
        labels={
            "decision_variable": "Decision variable",
            "selection_rate": "Selection rate in CSS"
        }
    )

    fig.update_layout(
        template="plotly_white",
        height=420,
        xaxis_tickangle=-45
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# =====================================================
# MAIN CSS COMPARISON

def render_css_comparison(
    css_df,
    dataset
):

    if not st.session_state.get(
        "show_css_comparison",
        False
    ):

        return

    with st.expander(
        "🆚 Detailed comparison",
        expanded=True
    ):

        if css_df is None or css_df.empty:

            st.info(
                "No Candidate Solution Set is available for comparison."
            )

            return

        if "id" not in css_df.columns:

            st.warning(
                "The current CSS does not contain an 'id' column."
            )

            return

        css_ids = (
            css_df["id"]
            .dropna()
            .astype(int)
            .tolist()
        )

        default_ids = st.session_state.get(
            "css_highlight_ids",
            []
        )

        default_ids = [
            solution_id
            for solution_id in default_ids
            if solution_id in css_ids
        ]

        compare_ids = st.multiselect(
            "Pick solutions to compare",
            css_ids,
            default=default_ids,
            key="css_compare_ids"
        )

        if len(compare_ids) < 2:

            st.info(
                "Select at least 2 solutions to compare."
            )

            return

        compare_df = css_df[
            css_df["id"].isin(
                compare_ids
            )
        ].copy()

        tab1, tab2, tab3 = st.tabs(
            [
                "📊 Objectives and indicators",
                "📋 Decision-variable matrix",
                "📈 Decision-variable distribution"
            ]
        )

        with tab1:

            render_tradeoff_radar(
                compare_df,
                css_df,
                dataset
            )

        with tab2:

            render_decision_variable_matrix(
                compare_df,
                dataset
            )

        with tab3:

            render_decision_variable_distribution(
                css_df,
                dataset
            )

# --- ARCHIVO: css_panel.py ---

## --------------------------------------------------------------------------------------
## css/css_panel.py
## --------------------------------------------------------------------------------------

import streamlit as st

def ensure_css_state():

    if "css_enabled" not in st.session_state:

        st.session_state.css_enabled = False

    if "css_source" not in st.session_state:

        st.session_state.css_source = "Current set"

    if "css_manual_ids" not in st.session_state:

        st.session_state.css_manual_ids = []

    if "css_highlight_ids" not in st.session_state:

        st.session_state.css_highlight_ids = []

    if "show_css_comparison" not in st.session_state:

        st.session_state.show_css_comparison = False


def sanitize_ids(
    ids,
    valid_ids
):

    valid_set = set(
        valid_ids
    )

    return [
        solution_id
        for solution_id in ids
        if solution_id in valid_set
    ]


def render_css_panel(
    current_df,
    dataset
):

    ensure_css_state()

    if current_df is None:

        return current_df

    css_df = current_df.copy()

    valid_ids = (
        css_df["id"]
        .tolist()
        if "id" in css_df.columns
        else []
    )

    st.session_state.css_manual_ids = sanitize_ids(
        st.session_state.css_manual_ids,
        valid_ids
    )

    st.session_state.css_highlight_ids = sanitize_ids(
        st.session_state.css_highlight_ids,
        valid_ids
    )

    with st.sidebar.expander(
        "🎯 Candidate Solution Set",
        expanded=False
    ):

        st.session_state.css_enabled = st.checkbox(
            "Lock current set as CSS",
            value=st.session_state.css_enabled,
            help=(
                "Create a Candidate Solution Set from the current "
                "decision subset or from manually selected solutions."
            )
        )

        if not st.session_state.css_enabled:

            st.caption(
                f"Current set available: {len(current_df)} solutions"
            )

            current_df[
                "highlight"
            ] = False

            return current_df

        st.session_state.css_source = st.radio(
            "CSS source",
            [
                "Current set",
                "Manual selection"
            ],
            index=[
                "Current set",
                "Manual selection"
            ].index(
                st.session_state.css_source
            ),
            horizontal=True
        )

        if st.session_state.css_source == "Manual selection":

            st.session_state.css_manual_ids = st.multiselect(
                "Solutions included in CSS",
                options=valid_ids,
                default=st.session_state.css_manual_ids,
                key="css_manual_ids_widget",
                help=(
                    "Select the exact solutions that should form "
                    "the Candidate Solution Set."
                )
            )

            css_df = current_df[
                current_df["id"].isin(
                    st.session_state.css_manual_ids
                )
            ].copy()

        else:

            css_df = current_df.copy()

        st.info(
            f"CSS size: {len(css_df)} solutions"
        )

        css_ids = (
            css_df["id"]
            .tolist()
            if "id" in css_df.columns
            else []
        )

        st.session_state.css_highlight_ids = sanitize_ids(
            st.session_state.css_highlight_ids,
            css_ids
        )

        st.session_state.css_highlight_ids = st.multiselect(
            "Highlight solutions",
            options=css_ids,
            default=st.session_state.css_highlight_ids,
            key="css_highlight_ids_widget",
            help=(
                "Highlighted solutions remain visible in maps "
                "and can be used for detailed comparison."
            )
        )

        st.session_state.show_css_comparison = st.checkbox(
            "Open detailed comparison",
            value=st.session_state.show_css_comparison,
            help=(
                "Open the detailed visual comparison section "
                "for the current CSS."
            )
        )

    css_df = css_df.copy()

    css_df[
        "highlight"
    ] = css_df[
        "id"
    ].isin(
        st.session_state.css_highlight_ids
    )

    return css_df

# --- ARCHIVO: enrichment.py ---

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

# --- ARCHIVO: framing.py ---

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

# --- ARCHIVO: input_panel.py ---

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

# --- ARCHIVO: lens_consensus.py ---

## --------------------------------------------------------------------------------------
## lens_consensus.py
## --------------------------------------------------------------------------------------

import pandas as pd
import streamlit as st


# =====================================================
# UI
# =====================================================

def render_params(
    dataset,
    working_df
):

    params = {}

    saved_sois = st.session_state.get(
        "saved_sois",
        []
    )

    if len(saved_sois) < 2:

        st.info(
            "At least two saved SOIs are required "
            "to build a consensus SOI."
        )

        params["method"] = "Consensus Threshold"
        params["selected_sois"] = []
        params["threshold"] = 0.5

        return params

    soi_names = [
        soi["name"]
        for soi in saved_sois
    ]

    params["method"] = st.selectbox(
        "Consensus Method",
        [
            "Consensus Threshold",
            "Union",
            "Majority",
            "Intersection"
        ],
        key="consensus_method"
    )

    params["selected_sois"] = st.multiselect(
        "SOIs to Combine",
        soi_names,
        default=soi_names[:min(
            2,
            len(soi_names)
        )],
        key="consensus_selected_sois"
    )

    n_selected = len(
        params["selected_sois"]
    )

    if params["method"] == "Union":

        threshold = (
            1.0
            /
            max(
                n_selected,
                1
            )
        )

        params["threshold"] = threshold

        st.caption(
            "Union keeps solutions supported by at least one selected SOI."
        )

    elif params["method"] == "Majority":

        params["threshold"] = 0.5

        st.caption(
            "Majority keeps solutions supported by at least half "
            "of the selected SOIs."
        )

    elif params["method"] == "Intersection":

        params["threshold"] = 1.0

        st.caption(
            "Intersection keeps only solutions supported by every selected SOI."
        )

    else:

        params["threshold"] = st.slider(
            "Consensus Level",
            0.0,
            1.0,
            0.5,
            0.05,
            key="consensus_threshold"
        )

        if params["threshold"] >= 0.75:

            st.caption(
                "Mode: consensus core."
            )

        elif params["threshold"] >= 0.50:

            st.caption(
                "Mode: consensus pool."
            )

        else:

            st.caption(
                "Mode: broad exploratory pool."
            )

    st.caption(
        "This lens treats saved SOIs as expert opinions "
        "and combines them into one consensus SOI."
    )

    return params

# =====================================================
# HELPERS
# =====================================================

def _get_selected_sois(
    selected_names
):

    saved_sois = st.session_state.get(
        "saved_sois",
        []
    )

    return [
        soi
        for soi in saved_sois
        if soi.get(
            "name"
        )
        in selected_names
    ]


def _build_support_table(
    selected_sois
):

    support = {}
    support_names = {}

    for soi in selected_sois:

        soi_name = soi.get(
            "name",
            "Unnamed SOI"
        )

        unique_ids = set(
            soi.get(
                "ids",
                []
            )
        )

        for solution_id in unique_ids:

            support[
                solution_id
            ] = (
                support.get(
                    solution_id,
                    0
                )
                +
                1
            )

            support_names.setdefault(
                solution_id,
                []
            ).append(
                soi_name
            )

    rows = []

    n_sois = len(
        selected_sois
    )

    for solution_id, support_count in support.items():

        consensus_score = (
            support_count
            /
            max(
                n_sois,
                1
            )
        )

        rows.append(
            {
                "id": solution_id,
                "consensus_support_count": support_count,
                "consensus_score": consensus_score,
                "consensus_supporting_sois": ", ".join(
                    sorted(
                        support_names.get(
                            solution_id,
                            []
                        )
                    )
                )
            }
        )

    return pd.DataFrame(
        rows
    )


def _add_consensus_labels(
    result,
    n_sois
):

    result[
        "group_base"
    ] = result[
        "consensus_support_count"
    ].apply(
        lambda count: (
            f"Support = {int(count)}/{n_sois}"
        )
    )

    group_sizes = (
        result["group_base"]
        .value_counts()
        .to_dict()
    )

    result[
        "group_label"
    ] = result[
        "group_base"
    ].apply(
        lambda group: (
            f"{group} "
            f"(n={group_sizes[group]})"
        )
    )

    return result

# =====================================================
# APPLY
# =====================================================

def apply(
    df,
    params,
    dataset
):

    result = df.copy()

    selected_names = params.get(
        "selected_sois",
        []
    )

    selected_sois = _get_selected_sois(
        selected_names
    )

    if len(selected_sois) < 2:

        result[
            "consensus_warning"
        ] = (
            "At least two SOIs are required for combination."
        )

        return result

    support_table = _build_support_table(
        selected_sois
    )

    if support_table.empty:

        result[
            "consensus_warning"
        ] = (
            "Selected SOIs do not contain any solution IDs."
        )

        return result

    threshold = params.get(
        "threshold",
        0.5
    )

    support_table = support_table[
        support_table["consensus_score"] >= threshold
    ].copy()

    if support_table.empty:

        result = result.iloc[
            0:0
        ].copy()

        result[
            "consensus_warning"
        ] = (
            "No solutions satisfy the selected consensus threshold."
        )

        return result

    result = result.merge(
        support_table,
        on="id",
        how="inner"
    )

    n_sois = len(
        selected_sois
    )

    result = _add_consensus_labels(
        result,
        n_sois
    )

    result[
        "consensus_method"
    ] = params.get(
        "method",
        "Consensus Threshold"
    )

    result[
        "consensus_threshold"
    ] = threshold

    result[
        "consensus_source_sois"
    ] = ", ".join(
        selected_names
    )

    result = result.sort_values(
        [
            "consensus_score",
            "consensus_support_count",
            "id"
        ],
        ascending=[
            False,
            False,
            True
        ]
    ).copy()

    result[
        "consensus_rank"
    ] = range(
        1,
        len(result) + 1
    )

    return result


# =====================================================
# FEEDBACK
# =====================================================

def _safe_first_value(
    df,
    column
):

    if column not in df.columns:

        return None

    values = (
        df[column]
        .dropna()
    )

    if values.empty:

        return None

    return values.iloc[0]


def render_feedback(
    lens_df
):

    if lens_df is None:

        st.warning(
            "No consensus result is available."
        )

        return

    warning_value = _safe_first_value(
        lens_df,
        "consensus_warning"
    )

    if warning_value is not None:

        st.warning(
            warning_value
        )

        return

    if lens_df.empty:

        st.warning(
            "The consensus SOI is empty."
        )

        return

    method = _safe_first_value(
        lens_df,
        "consensus_method"
    )

    if method is not None:

        st.info(
            f"Consensus method: {method}"
        )

    threshold = _safe_first_value(
        lens_df,
        "consensus_threshold"
    )

    if threshold is not None:

        st.caption(
            f"Consensus threshold: {float(threshold):.2f}"
        )

    max_score = (
        lens_df[
            "consensus_score"
        ].max()
        if "consensus_score" in lens_df.columns
        else None
    )

    if max_score is not None:

        st.caption(
            f"Maximum consensus score: {float(max_score):.2f}"
        )

    st.caption(
        f"Consensus SOI size: {len(lens_df)} solutions"
    )



# --- ARCHIVO: lens_diversity.py ---

## --------------------------------------------------------------------------------------
## lens_diversity.py
## --------------------------------------------------------------------------------------

import pandas as pd
import streamlit as st

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

try:
    from sklearn_extra.cluster import KMedoids
except Exception:
    KMedoids = None

try:
    from sklearn.cluster import HDBSCAN
except Exception:
    HDBSCAN = None


# =====================================================
# UI
# =====================================================

def render_params(
    dataset,
    working_df
):

    dimensions = (
        dataset["metrics"]
        +
        dataset["selected_indicators"]
    )

    params = {}

    max_n = max(
        len(working_df),
        1
    )

    if len(dimensions) < 2:

        st.info(
            "At least two dimensions are required for clustering."
        )

        params["method"] = "K-Medoids"
        params["cluster_metrics"] = []

        return params

    params["method"] = st.selectbox(
        "Clustering Method",
        [
            "K-Medoids",
            "K-Means",
            "Agglomerative",
            "HDBSCAN"
        ],
        key="div_method"
    )

    default_cluster_metrics = dimensions[
        :min(
            2,
            len(dimensions)
        )
    ]

    params["cluster_metrics"] = st.multiselect(
        "Metrics for Clustering",
        dimensions,
        default=default_cluster_metrics,
        key="div_cluster_metrics"
    )

    if params["method"] in [
        "K-Medoids",
        "K-Means"
    ]:

        params["k_mode"] = st.radio(
            "Number of Groups",
            [
                "Auto",
                "Manual"
            ],
            horizontal=True,
            key="div_k_mode"
        )

        if params["k_mode"] == "Manual":

            max_k = max(
                2,
                min(
                    10,
                    max_n
                )
            )

            default_k = min(
                3,
                max_k
            )

            params["k"] = st.slider(
                "k Groups",
                2,
                max_k,
                default_k,
                key="div_k"
            )

        else:

            st.caption(
                "Auto mode selects k using silhouette score."
            )

    elif params["method"] == "Agglomerative":

        params["agglomerative_mode"] = st.radio(
            "Hierarchy Cut Mode",
            [
                "Number of Groups",
                "Distance Cut"
            ],
            horizontal=True,
            key="div_agglomerative_mode"
        )

        if params["agglomerative_mode"] == "Number of Groups":

            params["k_mode"] = st.radio(
                "Number of Groups",
                [
                    "Auto",
                    "Manual"
                ],
                horizontal=True,
                key="div_agg_k_mode"
            )

            if params["k_mode"] == "Manual":

                max_k = max(
                    2,
                    min(
                        10,
                        max_n
                    )
                )

                default_k = min(
                    3,
                    max_k
                )

                params["k"] = st.slider(
                    "k Groups",
                    2,
                    max_k,
                    default_k,
                    key="div_agg_k"
                )

            else:

                st.caption(
                    "Auto mode selects the dendrogram cut "
                    "that produces the best silhouette score."
                )

        else:

            params["distance_threshold"] = st.slider(
                "Distance Threshold",
                0.10,
                10.00,
                2.00,
                0.10,
                key="div_agg_distance_threshold"
            )

            st.caption(
                "Distance Cut builds the hierarchy and cuts it at the selected "
                "distance. The number of groups is determined automatically."
            )


    elif params["method"] == "HDBSCAN":

        params["cluster_size_mode"] = st.radio(
            "Cluster Size",
            [
                "Auto",
                "Manual"
            ],
            horizontal=True,
            key="div_hdbscan_size_mode"
        )

        if params["cluster_size_mode"] == "Auto":

            params["granularity"] = st.selectbox(
                "Cluster Granularity",
                [
                    "Small (~5%)",
                    "Medium (~10%)",
                    "Large (~20%)"
                ],
                index=1,
                key="div_hdbscan_granularity"
            )


        else:

            default_min_cluster_size = max(
                2,
                int(
                    0.10 * max_n
                )
            )

            params["min_cluster_size"] = st.slider(
                "Minimum Cluster Size",
                2,
                max(
                    2,
                    max_n
                ),
                default_min_cluster_size,
                key="div_hdbscan_min_cluster_size"
            )

        params["exclude_noise"] = st.checkbox(
            "Exclude noise solutions",
            value=True,
            key="div_hdbscan_exclude_noise"
        )

        st.caption(
            "If HDBSCAN returns mostly noise, try Small or Medium "
            "granularity, or disable noise exclusion."
        )

    st.caption(
        "Diversity structures the current subset into clusters "
        "instead of applying a preference score."
    )

    return params


# =====================================================
# HELPERS
# =====================================================

def _valid_numeric_metrics(
    df,
    metrics
):

    return [
        metric
        for metric in metrics
        if (
            metric in df.columns
            and pd.api.types.is_numeric_dtype(
                df[metric]
            )
        )
    ]


def _prepare_matrix(
    df,
    metrics
):

    x = df[
        metrics
    ].copy()

    x = x.fillna(
        x.median(
            numeric_only=True
        )
    )

    scaler = StandardScaler()

    x_scaled = scaler.fit_transform(
        x
    )

    return x_scaled



def _build_partition_model(
    method,
    k
):

    if method == "K-Medoids":

        if KMedoids is not None:

            return KMedoids(
                n_clusters=k,
                method="pam",
                random_state=123
            )

        return KMeans(
            n_clusters=k,
            random_state=123,
            n_init=10
        )

    if method == "K-Means":

        return KMeans(
            n_clusters=k,
            random_state=123,
            n_init=10
        )

    if method == "Agglomerative":

        return AgglomerativeClustering(
            n_clusters=k
        )

    return KMeans(
        n_clusters=k,
        random_state=123,
        n_init=10
    )


def _compute_auto_k(
    x_scaled,
    method,
    max_k=10
):

    n = len(
        x_scaled
    )

    if n < 3:

        return 1, None

    best_k = 2
    best_score = -1

    upper_k = min(
        max_k,
        n - 1
    )

    for k in range(
        2,
        upper_k + 1
    ):

        try:

            model = _build_partition_model(
                method,
                k
            )

            labels = model.fit_predict(
                x_scaled
            )


            unique_labels = set(
                labels
            )

            if (
                len(unique_labels) > 1
                and
                len(unique_labels) < n
            ):

                score = silhouette_score(
                    x_scaled,
                    labels
                )

                if score > best_score:

                    best_score = score
                    best_k = k

        except Exception:

            pass

    return best_k, best_score


def _fit_partition_clustering(
    x_scaled,
    method,
    k
):

    model = _build_partition_model(
        method,
        k
    )

    labels = model.fit_predict(
        x_scaled
    )

    if (
        method == "K-Medoids"
        and
        KMedoids is None
    ):

        method_used = "K-Means fallback"

    else:

        method_used = method

    return labels, method_used


def _fit_hdbscan(
    x_scaled,
    min_cluster_size
):

    if HDBSCAN is None:

        labels = [
            0
            for _ in range(
                len(x_scaled)
            )
        ]

        method_used = "HDBSCAN unavailable"

        return labels, method_used

    model = HDBSCAN(
        min_cluster_size=min_cluster_size
    )

    labels = model.fit_predict(
        x_scaled
    )

    method_used = "HDBSCAN"

    return labels, method_used

def _fit_agglomerative_distance_cut(
    x_scaled,
    distance_threshold
):

    model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        compute_full_tree=True
    )

    labels = model.fit_predict(
        x_scaled
    )

    method_used = "Agglomerative distance cut"

    return labels, method_used


def _compute_silhouette_if_valid(
    x_scaled,
    labels
):

    unique_labels = set(
        labels
    )

    n = len(
        labels
    )

    if (
        len(unique_labels) <= 1
        or
        len(unique_labels) >= n
    ):

        return None

    try:

        return silhouette_score(
            x_scaled,
            labels
        )

    except Exception:

        return None


def _add_cluster_labels(
    result,
    labels,
    method_used,
    metrics_used
):

    result = result.copy()

    result[
        "cluster"
    ] = labels

    result[
        "cluster_str"
    ] = result[
        "cluster"
    ].astype(
        str
    )

    result[
        "cluster_str"
    ] = result[
        "cluster_str"
    ].replace(
        "-1",
        "Noise"
    )

    cluster_sizes = (
        result
        .groupby(
            "cluster_str"
        )["id"]
        .transform(
            "size"
        )
    )

    result[
        "group_label"
    ] = (
        "Cluster "
        +
        result["cluster_str"]
        +
        " (n="
        +
        cluster_sizes.astype(
            str
        )
        +
        ")"
    )

    n_clusters = (
        result["cluster"]
        .dropna()
        .astype(int)
        .loc[
            lambda values: values != -1
        ]
        .nunique()
    )

    noise_count = (
        result["cluster"]
        .eq(-1)
        .sum()
    )

    result[
        "diversity_method"
    ] = method_used

    result[
        "diversity_metrics"
    ] = ", ".join(
        metrics_used
    )

    result[
        "diversity_n_clusters"
    ] = n_clusters

    result[
        "diversity_noise_count"
    ] = noise_count

    return result




# =====================================================
# APPLY
# =====================================================

def apply(
    df,
    params,
    dataset
):

    result = df.copy()

    dimensions = (
        dataset["metrics"]
        +
        dataset["selected_indicators"]
    )

    method = params.get(
        "method",
        "K-Medoids"
    )

    cluster_metrics = params.get(
        "cluster_metrics",
        dimensions
    )

    cluster_metrics = _valid_numeric_metrics(
        result,
        cluster_metrics
    )

    if len(cluster_metrics) < 2:

        return result

    if len(result) < 2:

        return result

    x_scaled = _prepare_matrix(
        result,
        cluster_metrics
    )

    # ==================================================
    # K-MEDOIDS / K-MEANS / AGGLOMERATIVE
    # ==================================================

    if method in [
        "K-Medoids",
        "K-Means"
    ]:
        


        k_mode = params.get(
            "k_mode",
            "Auto"
        )

        if k_mode == "Manual":

            k = params.get(
                "k",
                2
            )

            k = max(
                2,
                min(
                    k,
                    len(result)
                )
            )

            silhouette = None

        else:

            k, silhouette = _compute_auto_k(
                x_scaled,
                method
            )

            if k < 2:

                return result

        labels, method_used = _fit_partition_clustering(
            x_scaled,
            method,
            k
        )

        result = _add_cluster_labels(
            result,
            labels,
            method_used,
            cluster_metrics
        )

        result[
            "diversity_k"
        ] = k

        if silhouette is not None:

            result[
                "diversity_silhouette"
            ] = silhouette

        return result




    if method == "Agglomerative":

        agglomerative_mode = params.get(
            "agglomerative_mode",
            "Number of Groups"
        )

        if agglomerative_mode == "Distance Cut":

            distance_threshold = params.get(
                "distance_threshold",
                2.0
            )

            labels, method_used = _fit_agglomerative_distance_cut(
                x_scaled,
                distance_threshold
            )

            result = _add_cluster_labels(
                result,
                labels,
                method_used,
                cluster_metrics
            )

            result[
                "diversity_distance_threshold"
            ] = distance_threshold

            silhouette = _compute_silhouette_if_valid(
                x_scaled,
                labels
            )

            if silhouette is not None:

                result[
                    "diversity_silhouette"
                ] = silhouette

            return result

        k_mode = params.get(
            "k_mode",
            "Auto"
        )

        if k_mode == "Manual":

            k = params.get(
                "k",
                2
            )

            k = max(
                2,
                min(
                    k,
                    len(result)
                )
            )

            silhouette = None

        else:

            k, silhouette = _compute_auto_k(
                x_scaled,
                method
            )

            if k < 2:

                return result

        labels, method_used = _fit_partition_clustering(
            x_scaled,
            method,
            k
        )

        result = _add_cluster_labels(
            result,
            labels,
            method_used,
            cluster_metrics
        )

        result[
            "diversity_k"
        ] = k

        if silhouette is not None:

            result[
                "diversity_silhouette"
            ] = silhouette

        return result




    # ==================================================
    # HDBSCAN
    # ==================================================

    if method == "HDBSCAN":

        n = len(
            result
        )

        size_mode = params.get(
            "cluster_size_mode",
            "Auto"
        )

        if size_mode == "Manual":

            min_cluster_size = params.get(
                "min_cluster_size",
                max(
                    2,
                    int(
                        0.1 * n
                    )
                )
            )

        else:

            granularity = params.get(
                "granularity",
                "Medium (~10%)"
            )

            if granularity == "Small (~5%)":

                min_cluster_size = max(
                    2,
                    int(
                        0.05 * n
                    )
                )

            elif granularity == "Large (~20%)":

                min_cluster_size = max(
                    2,
                    int(
                        0.20 * n
                    )
                )

            else:

                min_cluster_size = max(
                    2,
                    int(
                        0.10 * n
                    )
                )

        labels, method_used = _fit_hdbscan(
            x_scaled,
            min_cluster_size
        )

        result = _add_cluster_labels(
            result,
            labels,
            method_used,
            cluster_metrics
        )

        result[
            "diversity_min_cluster_size"
        ] = min_cluster_size

        exclude_noise = params.get(
            "exclude_noise",
            True
        )

        if exclude_noise:

            filtered_result = result[
                result["cluster"] != -1
            ].copy()

            if filtered_result.empty:

                result[
                    "diversity_warning"
                ] = (
                    "All solutions were classified as noise. "
                    "Noise exclusion was not applied."
                )

                return result

            return filtered_result

        return result

    return result


# =====================================================
# FEEDBACK
# =====================================================

def _safe_first_value(
    df,
    column
):

    if column not in df.columns:

        return None

    values = (
        df[column]
        .dropna()
    )

    if values.empty:

        return None

    return values.iloc[0]


def render_feedback(
    lens_df
):

    if lens_df is None:

        st.warning(
            "No clustering result is available."
        )

        return

    if lens_df.empty:

        st.warning(
            "The clustering lens returned an empty subset. "
            "Try reducing the HDBSCAN cluster size or disabling noise exclusion."
        )

        return

    warning_value = _safe_first_value(
        lens_df,
        "diversity_warning"
    )

    if warning_value is not None:

        st.warning(
            warning_value
        )

    n_clusters = _safe_first_value(
        lens_df,
        "diversity_n_clusters"
    )

    if n_clusters is not None:

        st.info(
            f"Clusters detected: {int(n_clusters)}"
        )

    k_value = _safe_first_value(
        lens_df,
        "diversity_k"
    )

    if k_value is not None:

        st.caption(
            f"Selected k: {int(k_value)}"
        )

    silhouette_value = _safe_first_value(
        lens_df,
        "diversity_silhouette"
    )

    if silhouette_value is not None:

        st.caption(
            f"Silhouette score: {silhouette_value:.3f}"
        )

    min_cluster_size = _safe_first_value(
        lens_df,
        "diversity_min_cluster_size"
    )

    if min_cluster_size is not None:

        st.caption(
            f"Minimum cluster size: {int(min_cluster_size)}"
        )

    distance_threshold = _safe_first_value(
        lens_df,
        "diversity_distance_threshold"
    )

    if distance_threshold is not None:

        st.caption(
            f"Distance threshold: {float(distance_threshold):.2f}"
        )

    noise_count = _safe_first_value(
        lens_df,
        "diversity_noise_count"
    )

    if noise_count is not None:

        if int(noise_count) > 0:

            st.caption(
                f"Noise solutions: {int(noise_count)}"
            )



# --- ARCHIVO: lens_efficiency.py ---

## --------------------------------------------------------------------------------------
## lens_efficiency.py
## --------------------------------------------------------------------------------------

import pandas as pd
import streamlit as st


EPS = 1e-9


# =====================================================
# UI
# =====================================================

def render_params(
    dataset,
    working_df
):

    dimensions = (
        dataset["metrics"]
        +
        dataset["selected_indicators"]
    )

    params = {}

    max_n = max(
        len(working_df),
        1
    )

    default_n = min(
        5,
        max_n
    )

    if len(dimensions) < 2:

        st.info(
            "At least two dimensions are required "
            "for the Efficiency lens."
        )

        params["method"] = "Benefit/Cost Ratio"
        params["benefit"] = None
        params["cost"] = None
        params["top_n"] = default_n

        return params

    params["method"] = st.selectbox(
        "Efficiency Method",
        [
            "Benefit/Cost Ratio",
            "Normalized Ratio",
            "Distance to Ideal",
            "Composite Cost Ratio"
        ],
        key="eff_method"
    )

    params["benefit"] = st.selectbox(
        "Benefit Metric",
        dimensions,
        key="eff_benefit"
    )

    cost_options = [
        d
        for d in dimensions
        if d != params["benefit"]
    ]

    if params["method"] == "Composite Cost Ratio":

        params["cost"] = st.multiselect(
            "Cost Metrics",
            cost_options,
            default=cost_options[
                :min(
                    2,
                    len(cost_options)
                )
            ],
            key="eff_costs"
        )

    else:

        params["cost"] = st.selectbox(
            "Cost Metric",
            cost_options,
            key="eff_cost"
        )

    params["top_n"] = st.slider(
        "Top N Solutions",
        1,
        max_n,
        default_n,
        key="eff_top_n"
    )

    st.caption(
        "Efficiency methods rank solutions by benefit-cost trade-off."
    )

    return params


# =====================================================
# HELPERS
# =====================================================

def _normalize_series(
    series
):

    min_v = series.min()
    max_v = series.max()

    if max_v > min_v:

        return (
            series
            -
            min_v
        ) / (
            max_v
            -
            min_v
        )

    return pd.Series(
        0.0,
        index=series.index
    )


def _resolve_cost_metrics(
    result,
    benefit,
    cost
):

    if cost is None:

        return []

    if isinstance(
        cost,
        str
    ):

        cost_metrics = [
            cost
        ]

    else:

        cost_metrics = [
            c
            for c in cost
            if c in result.columns
        ]

    cost_metrics = [
        c
        for c in cost_metrics
        if c != benefit
    ]

    return cost_metrics

# =====================================================
# SCORE METHODS
# =====================================================

def _benefit_cost_ratio(
    result,
    benefit,
    cost_metrics
):

    cost_metric = cost_metrics[0]

    safe_cost = result[
        cost_metric
    ].replace(
        0,
        EPS
    )

    return (
        result[benefit]
        /
        safe_cost
    )


def _normalized_ratio(
    result,
    benefit,
    cost_metrics
):

    cost_metric = cost_metrics[0]

    benefit_norm = _normalize_series(
        result[benefit]
    )

    cost_norm = _normalize_series(
        result[cost_metric]
    )

    return (
        benefit_norm
        /
        (
            cost_norm
            +
            EPS
        )
    )


def _distance_to_ideal(
    result,
    benefit,
    cost_metrics
):

    cost_metric = cost_metrics[0]

    benefit_norm = _normalize_series(
        result[benefit]
    )

    cost_norm = _normalize_series(
        result[cost_metric]
    )

    distance_to_ideal = (
        (
            1.0
            -
            benefit_norm
        ) ** 2
        +
        (
            cost_norm
        ) ** 2
    ) ** 0.5

    max_distance = (
        2 ** 0.5
    )

    return (
        1.0
        -
        distance_to_ideal
        /
        max_distance
    )


def _composite_cost_ratio(
    result,
    benefit,
    cost_metrics
):

    benefit_norm = _normalize_series(
        result[benefit]
    )

    composite_cost = pd.Series(
        0.0,
        index=result.index
    )

    for cost_metric in cost_metrics:

        composite_cost = (
            composite_cost
            +
            _normalize_series(
                result[cost_metric]
            )
        )

    composite_cost = (
        composite_cost
        /
        len(cost_metrics)
    )

    return (
        benefit_norm
        /
        (
            composite_cost
            +
            EPS
        )
    )

# =====================================================
# APPLY
# =====================================================

def apply(
    df,
    params,
    dataset
):

    result = df.copy()

    method = params.get(
        "method",
        "Benefit/Cost Ratio"
    )

    benefit = params.get(
        "benefit"
    )

    cost = params.get(
        "cost"
    )

    if (
        benefit is None
        or benefit not in result.columns
    ):

        return result

    cost_metrics = _resolve_cost_metrics(
        result,
        benefit,
        cost
    )

    if len(cost_metrics) == 0:

        return result

    top_n = min(
        params.get(
            "top_n",
            len(result)
        ),
        len(result)
    )

    if method == "Benefit/Cost Ratio":

        score = _benefit_cost_ratio(
            result,
            benefit,
            cost_metrics
        )

    elif method == "Normalized Ratio":

        score = _normalized_ratio(
            result,
            benefit,
            cost_metrics
        )

    elif method == "Distance to Ideal":

        score = _distance_to_ideal(
            result,
            benefit,
            cost_metrics
        )

    elif method == "Composite Cost Ratio":

        score = _composite_cost_ratio(
            result,
            benefit,
            cost_metrics
        )

        result[
            "efficiency_costs"
        ] = ", ".join(
            cost_metrics
        )

    else:

        return result

    result[
        "efficiency_score"
    ] = score

    result = result.sort_values(
        "efficiency_score",
        ascending=False
    ).copy()

    result[
        "efficiency_rank"
    ] = range(
        1,
        len(result) + 1
    )

    result[
        "efficiency_method"
    ] = method

    result[
        "efficiency_benefit"
    ] = benefit

    result[
        "efficiency_primary_cost"
    ] = cost_metrics[0]

    return result.head(
        top_n
    )


# =====================================================
# FEEDBACK
# =====================================================

def render_feedback(
    lens_df
):

    if "efficiency_method" in lens_df.columns:

        method = (
            lens_df["efficiency_method"]
            .dropna()
            .iloc[0]
        )

        st.info(
            f"Efficiency method: {method}"
        )

    if "efficiency_benefit" in lens_df.columns:

        benefit = (
            lens_df["efficiency_benefit"]
            .dropna()
            .iloc[0]
        )

        st.caption(
            f"Benefit metric: {benefit}"
        )

    if "efficiency_costs" in lens_df.columns:

        costs = (
            lens_df["efficiency_costs"]
            .dropna()
            .iloc[0]
        )

        st.caption(
            f"Composite costs: {costs}"
        )

    elif "efficiency_primary_cost" in lens_df.columns:

        cost = (
            lens_df["efficiency_primary_cost"]
            .dropna()
            .iloc[0]
        )

        st.caption(
            f"Cost metric: {cost}"
        )


# --- ARCHIVO: lens_engine.py ---

## --------------------------------------------------------------------------------------
## lens_engine.py
## --------------------------------------------------------------------------------------

from lenses.lens_registry import (
    get_lens_module
)


def apply_lens(
    df,
    lens_name,
    params,
    dataset
):

    if df is None:

        return df

    if lens_name == "None":

        return df.copy()

    lens_module = get_lens_module(
        lens_name
    )

    if lens_module is None:

        return df.copy()

    return lens_module.apply(
        df,
        params,
        dataset
    )

# --- ARCHIVO: lens_feedback.py ---

## --------------------------------------------------------------------------------------
## lens_feedback.py
## --------------------------------------------------------------------------------------


from lenses.lens_registry import (
    get_lens_module
)


def render_lens_feedback(
    placeholder,
    active_lens,
    lens_df
):

    if placeholder is None:
        return

    if lens_df is None:

        return

    if active_lens == "None":

        return

    lens_module = get_lens_module(
        active_lens
    )

    if lens_module is None:

        return

    with placeholder.container():

        if hasattr(
            lens_module,
            "render_feedback"
        ):

            lens_module.render_feedback(
                lens_df
            )

# --- ARCHIVO: lens_indicator.py ---

        
## --------------------------------------------------------------------------------------
## lens_indicator.py
## --------------------------------------------------------------------------------------

import pandas as pd
import streamlit as st

# =====================================================
# UI

def render_params( dataset, working_df ):

    dimensions = (
        dataset["metrics"]
        +
        dataset["selected_indicators"]
    )

    indicators = dataset[ "selected_indicators"]

    params = {}

    max_n = max( len(working_df), 1)

    default_n = min( 5,  max_n)

    if len(dimensions) == 0:

        st.info(
            "No dimensions are currently available. "
            "Select objectives or enable indicators first."
        )

        params["method"] = "Top-N Matches"
        params["maximize"] = []
        params["minimize"] = []
        params["top_n"] = default_n

        return params

    params["method"] = st.selectbox(
        "Indicator Method",
        [
            "Top-N Matches",
            "Non-dominated"
        ],
        key="indicator_method"
    )

    if params["method"] == "Top-N Matches":

        available_criteria = dimensions

        st.caption(
            "Top-N Matches can use both original objectives "
            "and enriched indicators."
        )

    else:

        available_criteria = indicators

        if len(available_criteria) == 0:

            st.info(
                "Non-dominated analysis currently uses enriched indicators. "
                "Enable indicators in Data Enrichment first."
            )

            params["maximize"] = []
            params["minimize"] = []
            params["top_n"] = None

            return params

        st.caption(
            "Non-dominated analysis uses enriched indicators."
        )

    params["maximize"] = st.multiselect(
        "Dimensions to Maximize",
        available_criteria,
        key="indicator_maximize"
    )

    minimize_options = [
        criterion
        for criterion in available_criteria
        if criterion not in params["maximize"]
    ]

    params["minimize"] = st.multiselect(
        "Dimensions to Minimize",
        minimize_options,
        key="indicator_minimize"
    )

    if params["method"] == "Top-N Matches":

        params["top_n"] = st.slider(
            "Top N per Dimension",
            1,
            max_n,
            default_n,
            key="indicator_top_n"
        )

        st.caption(
            "This method counts how often each solution appears "
            "among the best candidates for the selected dimensions."
        )

    else:

        params["top_n"] = None

        st.caption(
            "This method keeps solutions that are not clearly "
            "outperformed within the selected enriched-indicator space."
        )

    return params



# =====================================================
# HELPERS
# =====================================================

def _sanitize_criteria( df, maximize, minimize ):

    maximize = [
        metric
        for metric in maximize
        if metric in df.columns
    ]

    minimize = [
        metric
        for metric in minimize
        if (
            metric in df.columns
            and metric not in maximize
        )
    ]

    criteria = (  maximize +  minimize )

    return maximize, minimize, criteria


def _build_group_labels_from_count(
    result,
    count_column
):

    result[
        "group_base"
    ] = result[
        count_column
    ].apply(
        lambda count: f"Matches = {count}"
    )

    group_sizes = (
        result["group_base"]
        .value_counts()
        .to_dict()
    )

    result[
        "group_label"
    ] = result[
        "group_base"
    ].apply(
        lambda group: (
            f"{group} "
            f"(n={group_sizes[group]})"
        )
    )

    return result


# =====================================================
# METHOD 1: TOP-N MATCHES
# =====================================================

def _apply_top_n_matches(
    df,
    maximize,
    minimize,
    top_n
):

    result = df.copy()

    criteria = (
        maximize
        +
        minimize
    )

    if not criteria:

        return result

    top_n = min(
        top_n,
        len(result)
    )

    ranked_subsets = []

    for metric in maximize:

        ranked_subsets.append(
            result
            .sort_values(
                metric,
                ascending=False
            )
            .head(top_n)
            [["id"]]
            .assign(
                matched_metric=metric,
                goal="Maximize"
            )
        )

    for metric in minimize:

        ranked_subsets.append(
            result
            .sort_values(
                metric,
                ascending=True
            )
            .head(top_n)
            [["id"]]
            .assign(
                matched_metric=metric,
                goal="Minimize"
            )
        )

    if not ranked_subsets:

        return result

    matches = pd.concat(
        ranked_subsets,
        ignore_index=True
    )

    counts = (
        matches
        .groupby("id")
        .size()
        .reset_index(
            name="domain_match_count"
        )
    )

    matched_metrics = (
        matches
        .groupby("id")["matched_metric"]
        .apply(
            lambda values: ", ".join(
                sorted(
                    set(values)
                )
            )
        )
        .reset_index(
            name="domain_matched_metrics"
        )
    )

    result = result.merge(
        counts,
        on="id",
        how="left"
    )

    result = result.merge(
        matched_metrics,
        on="id",
        how="left"
    )

    result[
        "domain_match_count"
    ] = result[
        "domain_match_count"
    ].fillna(
        0
    ).astype(
        int
    )

    result[
        "domain_matched_metrics"
    ] = result[
        "domain_matched_metrics"
    ].fillna(
        ""
    )

    result = result[
        result["domain_match_count"] > 0
    ].copy()

    if result.empty:

        return result

    result = _build_group_labels_from_count(
        result,
        "domain_match_count"
    )

    result = result.sort_values(
        [
            "domain_match_count",
            "id"
        ],
        ascending=[
            False,
            True
        ]
    ).copy()

    result[
        "domain_rank"
    ] = range(
        1,
        len(result) + 1
    )

    result[
        "indicator_method"
    ] = "Top-N Matches"

    result[
        "indicator_top_n"
    ] = top_n

    return result


# =====================================================
# METHOD 2: NON-DOMINATED
# =====================================================

def _apply_non_dominated(
    df,
    maximize,
    minimize
):

    result = df.copy()

    criteria = (
        maximize
        +
        minimize
    )

    if not criteria:

        return result

    work = result[
        criteria
    ].copy()

    for metric in minimize:

        work[
            metric
        ] = -work[
            metric
        ]

    values = work.to_numpy()

    is_nondominated = []

    for i in range(
        len(values)
    ):

        current = values[i]

        dominated = False

        for j in range(
            len(values)
        ):

            if i == j:

                continue

            challenger = values[j]

            better_or_equal_all = (
                challenger >= current
            ).all()

            strictly_better_one = (
                challenger > current
            ).any()

            if (
                better_or_equal_all
                and
                strictly_better_one
            ):

                dominated = True

                break

        is_nondominated.append(
            not dominated
        )

    result[
        "indicator_nondominated"
    ] = is_nondominated

    result = result[
        result["indicator_nondominated"]
    ].copy()

    if result.empty:

        return result

    result[
        "indicator_method"
    ] = "Non-dominated"

    result[
        "domain_match_count"
    ] = len(criteria)

    result[
        "domain_matched_metrics"
    ] = ", ".join(
        criteria
    )

    result[
        "group_base"
    ] = "Non-dominated"

    result[
        "group_label"
    ] = (
        "Non-dominated "
        f"(n={len(result)})"
    )

    result = result.sort_values(
        "id",
        ascending=True
    ).copy()

    result[
        "domain_rank"
    ] = range(
        1,
        len(result) + 1
    )

    return result


# =====================================================
# APPLY
# =====================================================

def apply(
    df,
    params,
    dataset
):

    result = df.copy()

    maximize, minimize, criteria = _sanitize_criteria(
        result,
        params.get(
            "maximize",
            []
        ),
        params.get(
            "minimize",
            []
        )
    )

    if not criteria:

        return result

    method = params.get(
        "method",
        "Top-N Matches"
    )

    if method == "Top-N Matches":

        return _apply_top_n_matches(
            result,
            maximize,
            minimize,
            params.get(
                "top_n",
                min(
                    5,
                    len(result)
                )
            )
        )

    if method == "Non-dominated":

        return _apply_non_dominated(
            result,
            maximize,
            minimize
        )

    return result


# =====================================================
# FEEDBACK
# =====================================================

def render_feedback(
    lens_df
):

    if "indicator_method" in lens_df.columns:

        method = (
            lens_df["indicator_method"]
            .dropna()
            .iloc[0]
        )

        st.info(
            f"Indicator method: {method}"
        )

    if "domain_match_count" in lens_df.columns:

        max_matches = (
            lens_df["domain_match_count"]
            .max()
        )

        st.caption(
            f"Maximum indicator matches: {int(max_matches)}"
        )

    if "domain_matched_metrics" in lens_df.columns:

        st.caption(
            "Solutions are grouped by matched indicators."
        )

    if "indicator_nondominated" in lens_df.columns:

        st.caption(
            f"Non-dominated solutions: {len(lens_df)}"
        )


# =====================================================
# BACKWARD COMPATIBILITY
# =====================================================

def apply_domain_lens(
    df,
    maximize,
    minimize,
    top_n
):

    maximize, minimize, criteria = _sanitize_criteria(
        df,
        maximize,
        minimize
    )

    if not criteria:

        return df.copy()

    return _apply_top_n_matches(
        df,
        maximize,
        minimize,
        top_n
    )

# --- ARCHIVO: lens_preference.py ---

## --------------------------------------------------------------------------------------
## lens_preference.py
## --------------------------------------------------------------------------------------

import pandas as pd
import streamlit as st


def render_params(
    dataset,
    working_df
):

    dimensions = (
        dataset["metrics"]
        +
        dataset["selected_indicators"]
    )

    max_n = max(
        len(working_df),
        1
    )

    default_n = min(
        5,
        max_n
    )

    params = {}

    params["method"] = st.selectbox(
        "Scoring Method",
        [
            "Weighted Sum",
            "TOPSIS",
            "VIKOR",
            "Reference Point"
        ],
        key="pref_method"
    )

    st.caption(
        "All preference methods currently use equal weights."
    )

    params["maximize"] = st.multiselect(
        "Metrics to Maximize",
        dimensions,
        key="pref_maximize"
    )

    minimize_options = [
        d
        for d in dimensions
        if d not in params["maximize"]
    ]

    params["minimize"] = st.multiselect(
        "Metrics to Minimize",
        minimize_options,
        key="pref_minimize"
    )

    params["top_n"] = st.slider(
        "Top N Solutions",
        1,
        max_n,
        default_n,
        key="pref_top_n"
    )

    return params


def _sanitize_criteria(
    df,
    maximize,
    minimize
):

    maximize = [
        m
        for m in maximize
        if m in df.columns
    ]

    minimize = [
        m
        for m in minimize
        if (
            m in df.columns
            and m not in maximize
        )
    ]

    return (
        maximize,
        minimize,
        maximize + minimize
    )


def _minmax_normalize(
    df,
    criteria
):

    norm = pd.DataFrame(
        index=df.index
    )

    for metric in criteria:

        min_v = df[metric].min()
        max_v = df[metric].max()

        if max_v > min_v:

            norm[metric] = (
                df[metric]
                -
                min_v
            ) / (
                max_v
                -
                min_v
            )

        else:

            norm[metric] = 0.0

    return norm


def _weighted_sum(
    df,
    maximize,
    minimize
):

    criteria = (
        maximize
        +
        minimize
    )

    norm = _minmax_normalize(
        df,
        criteria
    )

    score = pd.Series(
        0.0,
        index=df.index
    )

    weight = (
        1.0
        /
        len(criteria)
    )

    for metric in criteria:

        if metric in maximize:

            value = norm[metric]

        else:

            value = (
                1.0
                -
                norm[metric]
            )

        score = (
            score
            +
            weight
            *
            value
        )

    return score


def _topsis(
    df,
    maximize,
    minimize
):

    criteria = (
        maximize
        +
        minimize
    )

    norm = df[
        criteria
    ].copy()

    weight = (
        1.0
        /
        len(criteria)
    )

    for metric in criteria:

        denom = (
            norm[metric] ** 2
        ).sum() ** 0.5

        if denom != 0:

            norm[metric] = (
                norm[metric]
                /
                denom
            )

        else:

            norm[metric] = 0.0

        norm[metric] = (
            norm[metric]
            *
            weight
        )

    ideal = {}
    anti_ideal = {}

    for metric in criteria:

        if metric in maximize:

            ideal[metric] = (
                norm[metric].max()
            )

            anti_ideal[metric] = (
                norm[metric].min()
            )

        else:

            ideal[metric] = (
                norm[metric].min()
            )

            anti_ideal[metric] = (
                norm[metric].max()
            )

    scores = []

    for _, row in norm.iterrows():

        d_plus = sum(
            (
                row[metric]
                -
                ideal[metric]
            ) ** 2
            for metric in criteria
        ) ** 0.5

        d_minus = sum(
            (
                row[metric]
                -
                anti_ideal[metric]
            ) ** 2
            for metric in criteria
        ) ** 0.5

        if (
            d_plus
            +
            d_minus
        ) != 0:

            score = (
                d_minus
                /
                (
                    d_plus
                    +
                    d_minus
                )
            )

        else:

            score = 0.0

        scores.append(
            score
        )

    return pd.Series(
        scores,
        index=df.index
    )


def _vikor(
    df,
    maximize,
    minimize,
    v=0.5
):

    criteria = (
        maximize
        +
        minimize
    )

    weight = (
        1.0
        /
        len(criteria)
    )

    regret = pd.DataFrame(
        index=df.index
    )

    for metric in criteria:

        if metric in maximize:

            best = df[metric].max()
            worst = df[metric].min()

        else:

            best = df[metric].min()
            worst = df[metric].max()

        denom = abs(
            best
            -
            worst
        )

        if denom == 0:

            regret[metric] = 0.0

        else:

            regret[metric] = (
                weight
                *
                abs(
                    best
                    -
                    df[metric]
                )
                /
                denom
            )

    s_value = regret.sum(
        axis=1
    )

    r_value = regret.max(
        axis=1
    )

    if s_value.max() > s_value.min():

        s_norm = (
            s_value
            -
            s_value.min()
        ) / (
            s_value.max()
            -
            s_value.min()
        )

    else:

        s_norm = 0.0

    if r_value.max() > r_value.min():

        r_norm = (
            r_value
            -
            r_value.min()
        ) / (
            r_value.max()
            -
            r_value.min()
        )

    else:

        r_norm = 0.0

    q_value = (
        v
        *
        s_norm
        +
        (
            1.0
            -
            v
        )
        *
        r_norm
    )

    return (
        1.0
        -
        q_value
    )


def _reference_point(
    df,
    maximize,
    minimize
):

    criteria = (
        maximize
        +
        minimize
    )

    norm = _minmax_normalize(
        df,
        criteria
    )

    oriented = pd.DataFrame(
        index=df.index
    )

    for metric in criteria:

        if metric in maximize:

            oriented[metric] = norm[metric]

        else:

            oriented[metric] = (
                1.0
                -
                norm[metric]
            )

    distances = []

    for _, row in oriented.iterrows():

        distance = sum(
            (
                1.0
                -
                row[metric]
            ) ** 2
            for metric in criteria
        ) ** 0.5

        distances.append(
            distance
        )

    distances = pd.Series(
        distances,
        index=df.index
    )

    max_distance = distances.max()

    if max_distance > 0:

        return (
            1.0
            -
            distances
            /
            max_distance
        )

    return pd.Series(
        1.0,
        index=df.index
    )


def apply(
    df,
    params,
    dataset
):

    result = df.copy()

    maximize, minimize, criteria = _sanitize_criteria(
        result,
        params.get(
            "maximize",
            []
        ),
        params.get(
            "minimize",
            []
        )
    )

    if not criteria:

        return result

    method = params.get(
        "method",
        "Weighted Sum"
    )

    top_n = min(
        params.get(
            "top_n",
            len(result)
        ),
        len(result)
    )

    if method == "Weighted Sum":

        score = _weighted_sum(
            result,
            maximize,
            minimize
        )

    elif method == "TOPSIS":

        score = _topsis(
            result,
            maximize,
            minimize
        )

    elif method == "VIKOR":

        score = _vikor(
            result,
            maximize,
            minimize
        )

    elif method == "Reference Point":

        score = _reference_point(
            result,
            maximize,
            minimize
        )

    else:

        return result

    result[
        "preference_score"
    ] = score

    result = result.sort_values(
        "preference_score",
        ascending=False
    ).copy()

    result[
        "preference_rank"
    ] = range(
        1,
        len(result) + 1
    )

    result[
        "preference_method"
    ] = method

    return result.head(
        top_n
    )


def render_feedback(
    lens_df
):

    if "preference_method" in lens_df.columns:

        method = (
            lens_df["preference_method"]
            .dropna()
            .iloc[0]
        )

        st.info(
            f"Preference method: {method}"
        )

    if "preference_score" in lens_df.columns:

        st.caption(
            "Solutions are ranked by preference_score."
        )

# --- ARCHIVO: lens_registry.py ---

## --------------------------------------------------------------------------------------
## lens_registry.py
## --------------------------------------------------------------------------------------

from lenses import lens_preference
from lenses import lens_diversity
from lenses import lens_efficiency
from lenses import lens_indicator
from lenses import lens_consensus


LENS_REGISTRY = {
    "Preference": lens_preference,
    "Diversity": lens_diversity,
    "Efficiency": lens_efficiency,
    "Indicator Dominance": lens_indicator,
    "SOI Consensus": lens_consensus
}


def get_lens_names():

    return [
        "None"
    ] + list(
        LENS_REGISTRY.keys()
    )


def get_lens_module(
    lens_name
):

    return LENS_REGISTRY.get(
        lens_name
    )

# --- ARCHIVO: lens_selection.py ---

## --------------------------------------------------------------------------------------
## lens_selection.py
## --------------------------------------------------------------------------------------

import streamlit as st


def ensure_soi_state():

    if "saved_sois" not in st.session_state:

        st.session_state.saved_sois = []


def get_group_column(
    lens_df
):

    if lens_df is None:

        return None

    if "group_label" in lens_df.columns:

        return "group_label"

    if "cluster_str" in lens_df.columns:

        return "cluster_str"

    return None


def get_group_options(
    lens_df,
    group_column
):

    if group_column is None:

        return []

    return sorted(
        lens_df[group_column]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )


def filter_by_group(
    lens_df,
    group_column,
    group_value
):

    if lens_df is None:

        return None

    if group_column is None:

        return lens_df.copy()

    if group_value == "All groups":

        return lens_df.copy()

    return lens_df[
        lens_df[group_column].astype(str)
        ==
        str(group_value)
    ].copy()


def get_lens_label(
    active_lens
):

    if active_lens == "None":

        return "Exploratory"

    return active_lens


def reset_soi_name_if_needed(
    active_lens,
    group_value
):

    lens_label = get_lens_label(
        active_lens
    )

    suffix = (
        group_value
        if group_value != "All groups"
        else "Current set"
    )

    default_name = (
        f"{lens_label} - {suffix} "
        f"#{len(st.session_state.saved_sois) + 1}"
    )

    name_context = (
        lens_label,
        group_value
    )

    if (
        st.session_state.get(
            "soi_name_context"
        )
        != name_context
    ):

        st.session_state[
            "soi_name"
        ] = default_name

        st.session_state[
            "soi_name_context"
        ] = name_context


def render_group_selector_and_save(
    placeholder,
    active_lens,
    lens_df,
    lens_params
):

    ensure_soi_state()

    if lens_df is None:
        return lens_df

    with placeholder.container():
        lens_label = get_lens_label(
            active_lens
        )

        group_column = get_group_column(
            lens_df
        )

        group_value = "All groups"
        
        if group_column is not None:

            group_options = get_group_options(
                lens_df,
                group_column
            )

            options = (
                ["All groups"]
                +
                group_options
            )

            selector_key = (
                f"soi_group_selector_"
                f"{lens_label.replace(' ', '_')}"
            )

            if (
                st.session_state.get(
                    selector_key
                )
                not in options
            ):

                st.session_state[
                    selector_key
                ] = "All groups"

            group_value = st.selectbox(
                "SOI group",
                options,
                key=selector_key,
                help=(
                    "Choose the group to promote as the current "
                    "Solution of Interest."
                )
            )

        current_df = filter_by_group(
            lens_df,
            group_column,
            group_value
        )

        if current_df is None:

            return lens_df

        st.caption(
            f"Current SOI candidate size: "
            f"{len(current_df)} solutions"
        )

        if active_lens == "None":

            st.caption(
                "Source: exploratory current set."
            )

        st.markdown(
            "---"
        )

        reset_soi_name_if_needed(
            active_lens,
            group_value
        )

        soi_name = st.text_input(
            "Name",
            key="soi_name"
        )

        if st.button(
            "💾 Save Current Set",
            use_container_width=True,
            key="save_current_soi"
        ):

            st.session_state.pending_save_soi = {
                "name": soi_name,
                "lens": lens_label,
                "method": lens_params.get(
                    "method",
                    "Exploratory"
                ),
                "params": lens_params,
                "ids": current_df["id"].tolist(),
                "group": group_value,
                "group_column": group_column,
                "source_size": len(lens_df),
                "soi_size": len(current_df)
            }

        return current_df

# --- ARCHIVO: lenses.py ---

## --------------------------------------------------------------------------------------
## lenses.py
## --------------------------------------------------------------------------------------

import streamlit as st

from lenses.lens_registry import (
    get_lens_names,
    get_lens_module
)


# =====================================================
# HEADER
# =====================================================

def render_lens_header(
    active_lens
):

    if "active_soi_name" in st.session_state:

        st.caption(
            f"Working on loaded SOI: "
            f"{st.session_state.active_soi_name}"
        )

    if active_lens != "None":

        st.markdown(
            f"""
            <div style="
                color:#E63946;
                font-size:12px;
                font-weight:600;
                text-align:center;
                margin:0.3rem 0 0.8rem 0;
            ">
            ───── {active_lens} lens ─────
            </div>
            """,
            unsafe_allow_html=True
        )


# =====================================================
# ACTIVE LENS PARAMS
# =====================================================

def render_active_lens_params(
    active_lens,
    dataset,
    working_df
):

    if active_lens == "None":

        return {}

    lens_module = get_lens_module(
        active_lens
    )

    if lens_module is None:

        st.warning(
            f"No module registered for lens: {active_lens}"
        )

        return {}

    if not hasattr(
        lens_module,
        "render_params"
    ):

        st.warning(
            f"Lens module '{active_lens}' does not define render_params()."
        )

        return {}

    return lens_module.render_params(
        dataset,
        working_df
    )


# =====================================================
# MAIN LENS PANEL
# =====================================================

def render_lens_panel(
    dataset,
    working_df
):

    params = {}

    with st.sidebar.expander(
        "🧭 Solution of Interest",
        expanded=False
    ):

        active_lens = st.selectbox(
            "Select an analytical lens",
            get_lens_names(),
            key="active_lens"
        )

        render_lens_header(
            active_lens
        )

        params = render_active_lens_params(
            active_lens,
            dataset,
            working_df
        )

        # Feedback goes here after the lens is applied.
        feedback_placeholder = st.empty()

        # Group selection and save controls go here after feedback.
        selection_placeholder = st.empty()

    return (
        active_lens,
        params,
        feedback_placeholder,
        selection_placeholder
    )

# --- ARCHIVO: nrp_plugin copy.py ---

## --------------------------------------------------------------------------------------
## plugins/nrp_plugin.py

import numpy as np

EPS = 1e-9

class NRPPlugin:
    """
    Minimal NRP plugin.

    Supports only the indicators required by the
    MSLite example configuration shipped with the framework.
    """

    def __init__(self, var_prefix="req_"):
        self.var_prefix = var_prefix

    # --------------------------------------------------
    # Available indicators
    # --------------------------------------------------

    def available_indicators(self):
        return {
            "scope",
            "productivity",
            "squandering",
            "annoyance",
            "dirtiness",
        }

    # --------------------------------------------------
    # Decision variables
    # --------------------------------------------------
    def decision_variables(self, df):
        return [
            c
            for c in df.columns
            if c.startswith(self.var_prefix)
        ]
    
    # --------------------------------------------------
    # Dependencies between indicators 
    # --------------------------------------------------
    def requirements(self):
        return {
            "scope": [],
            "productivity": [
                "satisfaction",
                "effort"
            ],
            "squandering": [
                "effort"
            ],
            "annoyance": [
                "dissatisfaction",
                "satisfaction"
            ],
            "dirtiness": [
                "dissatisfaction",
                "effort"
            ],
        }
    # --------------------------------------------------
    # Indicator computation
    # --------------------------------------------------

    def compute_indicators( self, df, selected_indicators ):
        result = df.copy()
        req_cols = self.decision_variables(result)
        for indicator in selected_indicators:
            try:
                # --------------------------------------
                # Productivity
                # --------------------------------------
                if indicator == "productivity":
                    result[indicator] = ( 
                        result["satisfaction"]
                        / np.maximum(
                            result["effort"],
                            EPS
                        )
                    )
                # --------------------------------------
                # Dirtiness
                # --------------------------------------
                elif indicator == "dirtiness":
                    result[indicator] = np.where(
                        result["dissatisfaction"] == 0,
                        0,
                        result["dissatisfaction"]
                        / np.maximum(
                            result["effort"],
                            EPS
                        )
                    )
                # --------------------------------------
                # Annoyance
                # --------------------------------------
                elif indicator == "annoyance":
                    result[indicator] = np.where(
                        result["dissatisfaction"] == 0,
                        0,
                        result["dissatisfaction"]
                        / np.maximum(
                            result["satisfaction"],
                            EPS
                        )
                    )
                # --------------------------------------
                # Squandering
                # --------------------------------------
                elif indicator == "squandering":
                    effort_max = result["effort"].max()
                    result[indicator] = (
                        effort_max
                        - result["effort"]
                    ) / np.maximum(
                        effort_max,
                        EPS
                    )
                # --------------------------------------
                # Scope
                # --------------------------------------
                elif indicator == "scope":
                    if len(req_cols) > 0:
                        result[indicator] = (
                            result[req_cols]
                            .sum(axis=1)
                            / len(req_cols)
                        )
            except Exception as exc:
                print(
                    f"[NRPPlugin] "
                    f"Unable to compute "
                    f"{indicator}: {exc}"
                )
        return result

# --- ARCHIVO: nrp_plugin.py ---

import numpy as np

EPS = 1e-9


class NRPPlugin:

    """
    Next Release Problem (NRP) domain plugin.

    Provides derived indicators commonly used in
    software release planning.
    """

    def __init__(self, var_prefix="req_"):
        self.var_prefix = var_prefix

    # --------------------------------------------------
    # Indicator registry
    # --------------------------------------------------

    def available_indicators(self):

        return {
            "scope",
            "productivity",
            "squandering",
            "annoyance",
            "dirtiness",
            "effectiveness",
            "stickiness",
            "robustness",
            "fragility",
            "response",
            "opportunity",
            "usage_efficiency",
        }

    # --------------------------------------------------
    # Required columns
    # --------------------------------------------------

    def requirements(self):

        return {

            "productivity":
                ["satisfaction", "effort"],

            "effectiveness":
                ["satisfaction", "cost"],

            "dirtiness":
                ["dissatisfaction", "effort"],

            "annoyance":
                ["dissatisfaction", "satisfaction"],

            "stickiness":
                ["prevalence", "effort"],

            "robustness":
                ["satisfaction", "inestability"],

            "fragility":
                ["prevalence", "inestability", "effort"],

            "response":
                ["time", "effort"],

            "opportunity":
                ["satisfaction", "time"],

            "usage_efficiency":
                ["prevalence", "cost"],

            "scope":
                [],

            "squandering":
                ["effort"],
        }

    # --------------------------------------------------
    # Decision variables
    # --------------------------------------------------

    def decision_variables(self, df):

        return [
            c
            for c in df.columns
            if c.startswith(self.var_prefix)
        ]

    # --------------------------------------------------
    # Indicator computation
    # --------------------------------------------------

    def compute_indicators(self, df, indicators):

        result = df.copy()

        req_cols = self.decision_variables(result)

        for indicator in indicators:

            try:

                # ----------------------------------
                # Productivity
                # ----------------------------------

                if indicator == "productivity":

                    result[indicator] = (
                        result["satisfaction"]
                        / np.maximum(result["effort"], EPS)
                    )

                # ----------------------------------
                # Effectiveness
                # ----------------------------------

                elif indicator == "effectiveness":

                    result[indicator] = (
                        result["satisfaction"]
                        / np.maximum(result["cost"], EPS)
                    )

                # ----------------------------------
                # Squandering
                # ----------------------------------

                elif indicator == "squandering":

                    effort_max = result["effort"].max()

                    result[indicator] = (
                        effort_max - result["effort"]
                    ) / np.maximum(effort_max, EPS)

                # ----------------------------------
                # Dirtiness
                # ----------------------------------

                elif indicator == "dirtiness":

                    result[indicator] = np.where(
                        result["dissatisfaction"] == 0,
                        0,
                        result["dissatisfaction"]
                        / np.maximum(result["effort"], EPS)
                    )

                # ----------------------------------
                # Annoyance
                # ----------------------------------

                elif indicator == "annoyance":

                    result[indicator] = np.where(
                        result["dissatisfaction"] == 0,
                        0,
                        result["dissatisfaction"]
                        / np.maximum(
                            result["satisfaction"],
                            EPS
                        )
                    )

                # ----------------------------------
                # Stickiness
                # ----------------------------------

                elif indicator == "stickiness":

                    result[indicator] = (
                        result["prevalence"]
                        / np.maximum(result["effort"], EPS)
                    )

                # ----------------------------------
                # Robustness
                # ----------------------------------

                elif indicator == "robustness":

                    result[indicator] = (
                        result["satisfaction"]
                        / np.maximum(
                            result["inestability"],
                            EPS
                        )
                    )

                # ----------------------------------
                # Fragility
                # ----------------------------------

                elif indicator == "fragility":

                    result[indicator] = (
                        result["prevalence"]
                        * result["inestability"]
                    ) / np.maximum(
                        result["effort"],
                        EPS
                    )

                # ----------------------------------
                # Response
                # ----------------------------------

                elif indicator == "response":

                    result[indicator] = np.where(
                        result["time"] == 0,
                        0,
                        result["time"]
                        / np.maximum(
                            result["effort"],
                            EPS
                        )
                    )

                # ----------------------------------
                # Opportunity
                # ----------------------------------

                elif indicator == "opportunity":

                    result[indicator] = np.where(
                        result["satisfaction"] == 0,
                        0,
                        result["satisfaction"]
                        / np.maximum(
                            result["time"],
                            EPS
                        )
                    )

                # ----------------------------------
                # Usage efficiency
                # ----------------------------------

                elif indicator == "usage_efficiency":

                    result[indicator] = (
                        result["prevalence"]
                        / result["cost"].replace(
                            0,
                            np.nan
                        )
                    ).fillna(0)

                # ----------------------------------
                # Scope
                # ----------------------------------

                elif indicator == "scope":

                    if len(req_cols) > 0:

                        result[indicator] = (
                            result[req_cols].sum(axis=1)
                            / len(req_cols)
                        )

            except Exception as e:

                print(
                    f"[PLUGIN][NRP] "
                    f"Unable to compute {indicator}: {e}"
                )

        return result

# --- ARCHIVO: nrpfull_plugin copy.py ---

import numpy as np

EPS = 1e-9


class NRPPlugin:

    """
    Next Release Problem (NRP) domain plugin.

    Provides derived indicators commonly used in
    software release planning.
    """

    def __init__(self, var_prefix="req_"):
        self.var_prefix = var_prefix

    # --------------------------------------------------
    # Indicator registry
    # --------------------------------------------------

    def available_indicators(self):

        return {
            "scope",
            "productivity",
            "squandering",
            "annoyance",
            "dirtiness",
            "effectiveness",
            "stickiness",
            "robustness",
            "fragility",
            "response",
            "opportunity",
            "usage_efficiency",
        }

    # --------------------------------------------------
    # Required columns
    # --------------------------------------------------

    def requirements(self):

        return {

            "productivity":
                ["satisfaction", "effort"],

            "effectiveness":
                ["satisfaction", "cost"],

            "dirtiness":
                ["dissatisfaction", "effort"],

            "annoyance":
                ["dissatisfaction", "satisfaction"],

            "stickiness":
                ["prevalence", "effort"],

            "robustness":
                ["satisfaction", "inestability"],

            "fragility":
                ["prevalence", "inestability", "effort"],

            "response":
                ["time", "effort"],

            "opportunity":
                ["satisfaction", "time"],

            "usage_efficiency":
                ["prevalence", "cost"],

            "scope":
                [],

            "squandering":
                ["effort"],
        }

    # --------------------------------------------------
    # Decision variables
    # --------------------------------------------------

    def decision_variables(self, df):

        return [
            c
            for c in df.columns
            if c.startswith(self.var_prefix)
        ]

    # --------------------------------------------------
    # Indicator computation
    # --------------------------------------------------

    def compute_indicators(self, df, indicators):

        result = df.copy()

        req_cols = self.decision_variables(result)

        for indicator in indicators:

            try:

                # ----------------------------------
                # Productivity
                # ----------------------------------

                if indicator == "productivity":

                    result[indicator] = (
                        result["satisfaction"]
                        / np.maximum(result["effort"], EPS)
                    )

                # ----------------------------------
                # Effectiveness
                # ----------------------------------

                elif indicator == "effectiveness":

                    result[indicator] = (
                        result["satisfaction"]
                        / np.maximum(result["cost"], EPS)
                    )

                # ----------------------------------
                # Squandering
                # ----------------------------------

                elif indicator == "squandering":

                    effort_max = result["effort"].max()

                    result[indicator] = (
                        effort_max - result["effort"]
                    ) / np.maximum(effort_max, EPS)

                # ----------------------------------
                # Dirtiness
                # ----------------------------------

                elif indicator == "dirtiness":

                    result[indicator] = np.where(
                        result["dissatisfaction"] == 0,
                        0,
                        result["dissatisfaction"]
                        / np.maximum(result["effort"], EPS)
                    )

                # ----------------------------------
                # Annoyance
                # ----------------------------------

                elif indicator == "annoyance":

                    result[indicator] = np.where(
                        result["dissatisfaction"] == 0,
                        0,
                        result["dissatisfaction"]
                        / np.maximum(
                            result["satisfaction"],
                            EPS
                        )
                    )

                # ----------------------------------
                # Stickiness
                # ----------------------------------

                elif indicator == "stickiness":

                    result[indicator] = (
                        result["prevalence"]
                        / np.maximum(result["effort"], EPS)
                    )

                # ----------------------------------
                # Robustness
                # ----------------------------------

                elif indicator == "robustness":

                    result[indicator] = (
                        result["satisfaction"]
                        / np.maximum(
                            result["inestability"],
                            EPS
                        )
                    )

                # ----------------------------------
                # Fragility
                # ----------------------------------

                elif indicator == "fragility":

                    result[indicator] = (
                        result["prevalence"]
                        * result["inestability"]
                    ) / np.maximum(
                        result["effort"],
                        EPS
                    )

                # ----------------------------------
                # Response
                # ----------------------------------

                elif indicator == "response":

                    result[indicator] = np.where(
                        result["time"] == 0,
                        0,
                        result["time"]
                        / np.maximum(
                            result["effort"],
                            EPS
                        )
                    )

                # ----------------------------------
                # Opportunity
                # ----------------------------------

                elif indicator == "opportunity":

                    result[indicator] = np.where(
                        result["satisfaction"] == 0,
                        0,
                        result["satisfaction"]
                        / np.maximum(
                            result["time"],
                            EPS
                        )
                    )

                # ----------------------------------
                # Usage efficiency
                # ----------------------------------

                elif indicator == "usage_efficiency":

                    result[indicator] = (
                        result["prevalence"]
                        / result["cost"].replace(
                            0,
                            np.nan
                        )
                    ).fillna(0)

                # ----------------------------------
                # Scope
                # ----------------------------------

                elif indicator == "scope":

                    if len(req_cols) > 0:

                        result[indicator] = (
                            result[req_cols].sum(axis=1)
                            / len(req_cols)
                        )

            except Exception as e:

                print(
                    f"[PLUGIN][NRP] "
                    f"Unable to compute {indicator}: {e}"
                )

        return result

# --- ARCHIVO: phase_help.py ---

## --------------------------------------------------------------------------------------
## ui/phase_help.py
## --------------------------------------------------------------------------------------

import streamlit as st


# =====================================================
# PHASE HELP TEXTS
# =====================================================

PHASE_HELP = {
    "input": """
Load the base decision space.

**1. Domain Configuration**

Use this option when you want to load a predefined case already configured in the library.

A domain configuration usually provides:

- the solution dataset
- the default optimization objectives
- the decision-variable prefix
- optional plugin logic
- optional default indicators

This is the recommended option when the decision problem has a known structure.

**2. Upload Enriched CSV**

Use this option when you already have a standalone CSV.

The uploaded CSV should contain:

- one row per solution
- numeric objective or indicator columns
- decision-variable columns sharing a common prefix

Examples of decision-variable prefixes:

- `x_`
- `var_`
- `req_`
- `feature_`
- `design_`

After loading the data, refine the Objective Column (optimization objectives) that define the base decision space.

> **User Purpose:** Load the base Pareto-optimal alternatives to start transforming them into an interpretable decision space.

""",

    "enrichment": """
The goal of this stage is to **expand the descriptive layer of the solutions** by adding domain-specific quality and semantic indicators.

Available indicators are provided by the active domain plugin. The app checks
which indicators can be computed from the currently selected base objectives.

Only compatible indicators are shown. An indicator is compatible when all
required input columns are available in the current dataset.

Each indicator must also have its calculation logic defined in the plugin.
If the plugin does not define how an indicator is computed, the app cannot
generate that indicator.

Derived indicators enrich the decision space with additional analytical views,
such as:

- productivity
- scope
- quality
- efficiency
- domain-specific measures

> **User Purpose:** Uncover implicit properties in solutions to project, filter, and compare the decision space under richer analytical perspectives.


""",

    "framing": """
The goal of this stage is to **restrict the global decision space to current operational conditions**.

Use constraints in this section to delimit the active set by applying:
- Maximum effort (budget) thresholds.
- Risk limits or minimum productivity/scope requirements.

> **User Purpose:** Reduce analytical complexity by isolating only those solutions that are viable under the current strategic context before applying analytical lenses.
""",

    "workspace_controls": """
Create and manage decision-space maps.

Maps visualize the current decision set, SOI, or CSS using selected objectives and indicators.

The goal of Maps is to **project and visualize alternatives across multiple dimensions simultaneously**.

Configure the axes (X, Y) and marker size to inspect how objectives and derived indicators behave across the active solution set.

> **User Purpose:** Detect visual patterns, trade-offs, and spatial distributions without committing to a single ranking upfront.
""",

    "soi": """
Generate or load a Solution of Interest.

A SOI is a candidate subset of solutions. It can come from:

- an analytical lens
- a saved SOI
- a consensus of saved SOIs
- the exploratory current set
The goal of this stage is to **identify Solutions of Interest (SOIs)**: sub-sets with strategic coherence identified under a specific analytical perspective (Lens).

Rather than evaluating isolated solutions, apply different Analytical Lenses:
- **Preference (MCDA):** Discover top-N alternatives based on TOPSIS or Weighted Sum.
- **Diversity:** Group structurally representative solutions using clustering (K-Medoids, HDBSCAN).
- **Efficiency:** Identify solutions offering the best benefit-cost trade-offs.
- **Dominance:** Highlight alternatives that repeatedly excel across multiple quality indicators.

> **User Purpose:** Extract latent strategic insights and patterns from the decision space using intermediate analytical units (SOIs).

Examples of Lens include:

- a cluster from Diversity
- a Top-N set from Preference
- an efficient subset
- a non-dominated subset
- a manually saved current set
""",

    "saved_sois": """
Review, load, or delete saved Solutions of Interest.

Saved SOIs store:

- solution IDs
- source lens
- method
- selected group
- parameters
- creation context

Use saved SOIs to return to previous interesting subsets or combine them later.
""",

    "css": """
Define the Candidate Solution Set.

The CSS is the final subset that will be studied in detail.

You can build it from:

- the current decision set
- the current SOI
- a manual selection of specific solutions

Once a CSS is active, it can be used for detailed visual comparison.
""",

    "summary": """
Summarize the current decision set or CSS.

This section shows:

- number of solutions
- number of attributes
- number of decision variables
- CSS status
- derived columns
- export options
- the current data table
""",

    "maps": """
Explore the current decision set or CSS visually.

Maps help reveal:

- trade-offs
- clusters
- consensus groups
- preference scores
- efficiency scores
- highlighted candidates
""",

    "comparison": """
Compare selected candidate solutions in detail.

The detailed comparison includes:

- radar profiles for objectives and indicators
- decision-variable composition matrix
- decision-variable distribution inside the CSS
"""
}


# =====================================================
# HELP ACCESS
# =====================================================

def get_phase_help(
    phase_key
):

    return PHASE_HELP.get(
        phase_key,
        ""
    )


def get_help_text(
    phase_key
):

    return get_phase_help(
        phase_key
    )


# =====================================================
# GENERIC HELP POPOVER
# =====================================================

def render_help_icon(
    help_text,
    key=None,
    label="ⓘ"
):

    if not help_text:

        return

    with st.popover(
        label
    ):

        st.markdown(
            help_text
        )


# =====================================================
# PHASE HELP POPOVER
# =====================================================

def render_phase_help_icon(
    phase_key,
    key=None,
    label="ⓘ"
):

    help_text = get_phase_help(
        phase_key
    )

    if not help_text:

        return

    render_help_icon(
        help_text,
        key=key,
        label=label
    )

# --- ARCHIVO: soi_registry.py ---

## --------------------------------------------------------------------------------------
## soi/soi_registry.py
## --------------------------------------------------------------------------------------

import html
import streamlit as st


# =====================================================
# SESSION STATE
# =====================================================

def ensure_soi_state():

    if "saved_sois" not in st.session_state:
        st.session_state.saved_sois = []


def has_loaded_soi():

    return (
        "active_soi_name"
        in st.session_state
        and
        "active_soi_ids"
        in st.session_state
    )


def clear_loaded_soi():

    if "active_soi_ids" in st.session_state:

        del st.session_state[
            "active_soi_ids"
        ]

    if "active_soi_name" in st.session_state:

        del st.session_state[
            "active_soi_name"
        ]

    if "active_soi_metadata" in st.session_state:

        del st.session_state[
            "active_soi_metadata"
        ]

    st.session_state[
        "pending_lens_reset"
    ] = True


def load_soi(
    soi
):

    st.session_state[
        "active_soi_ids"
    ] = soi[
        "ids"
    ]

    st.session_state[
        "active_soi_name"
    ] = soi[
        "name"
    ]

    st.session_state[
        "active_soi_metadata"
    ] = {
        "lens": soi.get(
            "lens"
        ),
        "method": soi.get(
            "method"
        ),
        "group": soi.get(
            "group"
        ),
        "group_column": soi.get(
            "group_column"
        ),
        "source_size": soi.get(
            "source_size"
        ),
        "soi_size": soi.get(
            "soi_size"
        ),
        "created_at": soi.get(
            "created_at"
        ),
        "params": soi.get(
            "params",
            {}
        )
    }

    st.session_state[
        "pending_lens_reset"
    ] = True


def delete_soi(
    idx
):

    deleted_name = (
        st.session_state
        .saved_sois[idx]["name"]
    )

    st.session_state.saved_sois.pop(
        idx
    )

    if (
        st.session_state.get(
            "active_soi_name"
        )
        ==
        deleted_name
    ):

        clear_loaded_soi()


# =====================================================
# LABEL HELPERS
# =====================================================

def normalize_method_label(
    soi
):

    method = soi.get(
        "method"
    )

    if method is None or method == "None":

        lens = soi.get(
            "lens",
            "Unknown"
        )

        if lens == "Exploratory":

            return "Exploratory"

        return None

    return method


def is_informative_group(
    group
):

    return (
        group is not None
        and group != ""
        and group != "All groups"
    )


def build_soi_main_label(
    soi
):

    name = soi.get(
        "name",
        "Unnamed SOI"
    )

    size = len(
        soi.get(
            "ids",
            []
        )
    )

    lens = soi.get(
        "lens",
        "Unknown"
    )

    method = normalize_method_label(
        soi
    )

    if method:

        return (
            f"{name} "
            f"[{size}] · "
            f"{lens} / {method}"
        )

    return (
        f"{name} "
        f"[{size}] · "
        f"{lens}"
    )


def build_compact_trace_label(
    soi
):

    parts = []

    group = soi.get(
        "group"
    )

    if is_informative_group(
        group
    ):

        parts.append(
            f"Group: {group}"
        )

    source_size = soi.get(
        "source_size"
    )

    soi_size = soi.get(
        "soi_size",
        len(
            soi.get(
                "ids",
                []
            )
        )
    )

    if source_size is not None:

        parts.append(
            f"{soi_size}/{source_size} solutions"
        )

    else:

        parts.append(
            f"{soi_size} solutions"
        )

    created_at = soi.get(
        "created_at"
    )

    if created_at:

        parts.append(
            created_at
        )

    return " · ".join(
        parts
    )


def build_tooltip_text(
    soi
):

    lens = soi.get(
        "lens",
        "Unknown"
    )

    method = normalize_method_label(
        soi
    )

    group = soi.get(
        "group"
    )

    source_size = soi.get(
        "source_size"
    )

    soi_size = soi.get(
        "soi_size",
        len(
            soi.get(
                "ids",
                []
            )
        )
    )

    created_at = soi.get(
        "created_at"
    )

    params = soi.get(
        "params",
        {}
    )

    lines = [
        f"Lens: {lens}",
        f"Method: {method if method else 'N/A'}",
        f"SOI size: {soi_size}"
    ]

    if source_size is not None:

        lines.append(
            f"Source size: {source_size}"
        )

    if is_informative_group(
        group
    ):

        lines.append(
            f"Group: {group}"
        )

    if created_at:

        lines.append(
            f"Created: {created_at}"
        )

    if params:

        compact_params = ", ".join(
            [
                f"{key}={value}"
                for key, value in params.items()
                if key not in [
                    "selected_sois",
                    "params"
                ]
            ]
        )

        if compact_params:

            lines.append(
                f"Params: {compact_params}"
            )

    return "\n".join(
        lines
    )


def render_trace_tooltip(
    soi
):

    compact_label = build_compact_trace_label(
        soi
    )

    tooltip = build_tooltip_text(
        soi
    )

    safe_label = html.escape(
        compact_label
    )

    safe_tooltip = html.escape(
        tooltip
    )

    st.markdown(
        (
            "<span "
            f"title=\"{safe_tooltip}\" "
            "style=\""
            "font-size:0.82rem;"
            "color:#6b7280;"
            "line-height:1.2;"
            "cursor:help;"
            "\">"
            f"ⓘ {safe_label}"
            "</span>"
        ),
        unsafe_allow_html=True
    )


# =====================================================
# RENDER LOADED SOI
# =====================================================

def render_loaded_soi_status():
    if not has_loaded_soi():

        return

    st.success(
        f"Active SOI: "
        f"{st.session_state.active_soi_name} "
        f"({len(st.session_state.active_soi_ids)} solutions)"
    )

    metadata = st.session_state.get(
        "active_soi_metadata",
        {}
    )

    if metadata:

        lens = metadata.get(
            "lens"
        )

        method = metadata.get(
            "method"
        )

        group = metadata.get(
            "group"
        )

        label_parts = []

        if lens:

            label_parts.append(
                lens
            )

        if method:

            label_parts.append(
                method
            )

        if is_informative_group(
            group
        ):

            label_parts.append(
                group
            )

        if label_parts:

            st.caption(
                " · ".join(
                    label_parts
                )
            )

    if st.button(
        "Clear Loaded SOI",
        use_container_width=True,
        key="clear_loaded_soi"
    ):

        clear_loaded_soi()

        st.rerun()

    st.markdown(
        "---"
    )


# =====================================================
# RENDER SAVED SOI ROW
# =====================================================
def render_saved_soi_row(
    soi,
    idx
):

    col_info, col_help, col_load, col_delete = st.columns(
        [
            0.68,
            0.08,
            0.14,
            0.10
        ]
    )

    with col_info:

        st.caption(
            f"• {build_soi_main_label(soi)}"
        )

    with col_help:

        tooltip = build_tooltip_text(
            soi
        )

        if st.button(
            "ⓘ",
            key=f"info_soi_{idx}",
            help=tooltip,
            use_container_width=True
        ):

            pass

    with col_load:

        if st.button(
            "Load",
            key=f"load_soi_{idx}",
            use_container_width=True
        ):

            load_soi(
                soi
            )

            st.rerun()

    with col_delete:

        if st.button(
            "🗑️",
            key=f"delete_soi_{idx}",
            use_container_width=True
        ):

            delete_soi(
                idx
            )

            st.rerun()


# =====================================================
# MAIN RENDERER
# =====================================================


def render_soi_tab():
    ensure_soi_state()
    if not st.session_state.saved_sois:

            st.info(
                "No saved SOIs."
            )

            return

    render_loaded_soi_status()

    for idx, soi in enumerate(
            st.session_state.saved_sois
    ):

        render_saved_soi_row(
                soi,
                idx
        )

# --- ARCHIVO: streamlit_app.py ---

## --------------------------------------------------------------------------------------
## streamlit_app.py
## --------------------------------------------------------------------------------------

import streamlit as st
from datetime import datetime

from deprecated.input_panel import render_input_panel
from core.enrichment import render_enrichment
from core.framing import apply_framing
from core.workspace_controls import render_workspace_controls
from core.workspace import render_workspace
from lenses.lenses import render_lens_panel
from lenses.lens_engine import apply_lens
from lenses.lens_feedback import render_lens_feedback
from lenses.lens_selection import ( render_group_selector_and_save )
from css.css_panel import ( render_css_panel )
from css.css_comparison import ( render_css_comparison )



st.set_page_config(
    page_title="Decision Space Explorer",
    layout="wide"
)

st.markdown(
    """
    <style>
    [data-testid="stExpander"] details summary p {
        font-size: 1.2rem;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title(  "Decision Space Explorer" )

# ==================================================
# INPUT

dataset = render_input_panel()
if dataset is None:
    st.info( "Select a domain configuration to begin.")
    st.stop()

# ==================================================
# ENRICHMENT

dataset = render_enrichment( dataset )

# ==================================================
# WORKSPACE CONTROLS

dimensions = ( dataset["metrics"] + dataset["selected_indicators"] )

show_ids = render_workspace_controls( dimensions )

# ==================================================
# FRAMING

framed_df = apply_framing( dataset )

# ==================================================
# WORKING DATASET

working_df = framed_df.copy()

if "active_soi_ids" in st.session_state:
    working_df = working_df[
        working_df["id"].isin(  st.session_state.active_soi_ids )
    ].copy()

# ==================================================
# RESET LENS AFTER LOADING / CLEARING SOI

if st.session_state.get( "pending_lens_reset", False ):
    st.session_state[ "active_lens" ] = "None"
    st.session_state["pending_lens_reset" ] = False

# ==================================================
# LENSES / SOI IDENTIFICATION

(
    active_lens,
    lens_params,
    feedback_placeholder,
    selection_placeholder
) = render_lens_panel(
    dataset,
    working_df
)

lens_df = apply_lens(
    working_df,
    active_lens,
    lens_params,
    dataset
)

if lens_df is None:

    st.sidebar.warning(
        "The selected lens returned no dataset. "
        "Reverting to the current working dataset."
    )

    lens_df = working_df.copy()

# ==================================================
# LENS FEEDBACK

render_lens_feedback( feedback_placeholder,  active_lens, lens_df )

# ==================================================
# GROUP SELECTION / CURRENT SOI CANDIDATE SET

current_df = render_group_selector_and_save(
    selection_placeholder,
    active_lens,
    lens_df,
    lens_params
)

if current_df is None:
    current_df = lens_df.copy()

# ==================================================
# SAVE CURRENT SOI

if "pending_save_soi" in st.session_state:
    if "saved_sois" not in st.session_state:
        st.session_state.saved_sois = []

    pending = ( st.session_state.pending_save_soi )
    existing_names = [
        soi["name"]
        for soi in st.session_state.saved_sois
    ]

    if pending["name"] in existing_names:
        st.sidebar.warning( "A SOI with this name already exists." )
    else:
        st.session_state.saved_sois.append(
            {
                "name": pending["name"],  "lens": pending["lens"],
                "params": pending.get( "params", {} ),
                "ids": pending.get( "ids", current_df["id"].tolist() ),
                "group": pending.get( "group",  "All groups" ),
                "group_column": pending.get( "group_column" )
            }
        )
        st.sidebar.success( f"Saved SOI: {pending['name']}" )
    del st.session_state[  "pending_save_soi" ]


# ==================== CANDIDATE SOLUTION SET ===================
 
css_df = render_css_panel( current_df,  dataset )

# ==================== WORKSPACE ==============================
    
render_workspace(  css_df, dataset, show_ids )

# ==================== DETAILED COMPARISON =======================

render_css_comparison( css_df, dataset )

# --- ARCHIVO: visualization.py ---

## --------------------------------------------------------------------------------------
## ui/visualization.py
## --------------------------------------------------------------------------------------

import plotly.express as px
import streamlit as st
import pandas as pd

# =====================================================
# COLOR SELECTION
# =====================================================

def infer_lens_color_column( df, user_color=None ):

    # --------------------------------------------------
    # Priority order:
    # 1. Group labels from clustering / indicator dominance
    # 2. Cluster labels
    # 3. Preference score
    # 4. Efficiency score
    # 5. Indicator dominance score
    # 6. User-selected color
    # --------------------------------------------------

    if "group_label" in df.columns:
        return "group_label"

    if "cluster_str" in df.columns:
        return "cluster_str"

    if "preference_score" in df.columns:
        return "preference_score"

    if "efficiency_score" in df.columns:
        return "efficiency_score"

    if "consensus_score" in df.columns:
        return "consensus_score"

    if "domain_match_count" in df.columns:
        return "domain_match_count"

    return user_color


def is_discrete_color( df, color_column ):

    if color_column is None:
        return False

    if color_column not in df.columns:
        return False

    if color_column in [
        "group_label",
        "cluster_str",
        "preference_method",
        "efficiency_method",
        "domain_matched_metrics"
    ]:

        return True

    if pd.api.types.is_object_dtype( df[color_column] ):
        return True

    return False


def build_hover_columns( df ):

    excluded_prefixes = ( "req_", "var_", "x_" )

    excluded_cols = {
        "label",
        "highlight",
        "highlight_label"
    }

    hover_cols = []

    for col in df.columns:

        if col in excluded_cols:
            continue

        if col.startswith( excluded_prefixes ):
            continue

        hover_cols.append( col)

    return hover_cols


# =====================================================
# SCATTER
# =====================================================

def render_scatter(
    df,
    x,
    y,
    size=None,
    color=None,
    show_ids=False,
    key=None
):

    df = df.copy()

    if x not in df.columns or y not in df.columns:
        st.warning( "Selected axes are not available in the current dataset." )
        return

    text_column = None

    if show_ids:

        if "id" in df.columns:
            text_column = "id"

        elif "ID" in df.columns:
            text_column = "ID"

    plot_color = infer_lens_color_column(
        df,
        user_color=color
    )

    discrete_color = is_discrete_color(
        df,
        plot_color
    )

    hover_cols = build_hover_columns(
        df
    )

    if discrete_color and plot_color is not None:

        df[
            plot_color
        ] = df[
            plot_color
        ].astype(
            str
        )

        fig = px.scatter(
            df,
            x=x,
            y=y,
            size=size,
            color=plot_color,
            text=text_column,
            hover_data=hover_cols
        )

    else:

        fig = px.scatter(
            df,
            x=x,
            y=y,
            size=size,
            color=plot_color,
            color_continuous_scale=px.colors.sequential.Viridis,
            text=text_column,
            hover_data=hover_cols
        )

    fig.update_traces(
        textposition="top center",
        textfont=dict(
            size=10
        )
    )

    if (
        "highlight"
        in df.columns
        and df["highlight"].any()
    ):

        marker_opacity = df[
            "highlight"
        ].apply(
            lambda value: 1.0 if value else 0.25
        )

        fig.update_traces(
            marker=dict(
                opacity=marker_opacity
            )
        )

        

    fig.update_layout(
        height=500,
        template="plotly_white",
        legend_title_text=(
            plot_color
            if plot_color is not None
            else ""
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key=key
    )


# =====================================================
# COORDINATED MAPS
# =====================================================

def render_coordinated_maps(
    df,
    x,
    y,
    z,
    key_prefix,
    show_ids=False
):

    col1, col2 = st.columns(
        2
    )

    with col1:

        st.caption(
            f"{x} vs {y}"
        )

        render_scatter(
            df,
            x=x,
            y=y,
            show_ids=show_ids,
            key=f"{key_prefix}_left"
        )

    with col2:

        st.caption(
            f"{x} vs {z}"
        )

        render_scatter(
            df,
            x=x,
            y=z,
            show_ids=show_ids,
            key=f"{key_prefix}_right"
        )


# =====================================================
# DISTRIBUTION
# =====================================================

def render_distribution(
    df,
    metric,
    mode="Violin",
    key=None
):

    if metric not in df.columns:

        st.warning(
            "Selected metric is not available in the current dataset."
        )

        return

    if mode == "Violin":

        fig = px.violin(
            df,
            y=metric,
            box=True,
            points="all"
        )

    else:

        fig = px.box(
            df,
            y=metric,
            points="all"
        )

    fig.update_layout(
        title=f"Distribution of {metric}",
        height=550,
        showlegend=False,
        template="plotly_white"
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key=key
    )

# --- ARCHIVO: workspace.py ---

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



# --- ARCHIVO: workspace_controls.py ---

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

# --- ARCHIVO: workspace_dataset.py ---

## --------------------------------------------------------------------------------------
## core/workspace_dataset.py
## --------------------------------------------------------------------------------------

import streamlit as st

def get_ordered_columns( df, dataset ):

    var_prefix = (
        dataset["config"] .get( "var_prefix", "x_" )
    )

    objective_cols = ( dataset["metrics"] )
    indicator_cols = ( dataset["selected_indicators"] )
    decision_cols = [ col for col in df.columns
        if col.startswith(
            var_prefix ) ]

    control_cols = [ "highlight", "highlight_label", "label" ]

    other_cols = [
        col
        for col in df.columns
        if (
            col not in objective_cols
            and col not in indicator_cols
            and col not in decision_cols
            and col not in control_cols
            and col != "id"
        )
    ]

    ordered_cols = (
        ["id"] + objective_cols + indicator_cols +
        other_cols + decision_cols )

    ordered_cols = [ col for col in ordered_cols
        if col in df.columns ]

    return ordered_cols


def get_current_set_label():
    if st.session_state.get( "css_enabled", False ):
        return "Current CSS"
    return "Current Decision Set"


def render_dataset_table( df, dataset ):
    label = get_current_set_label()

    st.markdown( f"#### 📋 {label}" )
    ordered_cols = get_ordered_columns( df, dataset )

    st.dataframe(
        df[ordered_cols],
        use_container_width=True,
        height=420,
        hide_index=True
    )


def render_dataset_preview( df, dataset ):
    label = get_current_set_label()

    with st.expander(
        f"📋 {label} "
        f"(prefix: "
        f"{dataset['config'].get('var_prefix')})",
        expanded=False
    ):

        render_dataset_table( df, dataset )

# --- ARCHIVO: workspace_maps.py ---

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

            

# --- ARCHIVO: workspace_summary.py ---

## --------------------------------------------------------------------------------------
## core/workspace_summary.py
## --------------------------------------------------------------------------------------

import streamlit as st

from core.workspace_dataset import (
    render_dataset_table
)
from soi.soi_registry import(
    render_soi_tab
)


# =====================================================
# DERIVED / LENS COLUMNS
# =====================================================

def get_lens_columns(
    df
):

    lens_prefixes = [
        "preference_",
        "efficiency_",
        "diversity_",
        "domain_",
        "indicator_",
        "consensus_"
    ]

    lens_columns = [
        col
        for col in df.columns
        if any(
            col.startswith(
                prefix
            )
            for prefix in lens_prefixes
        )
    ]

    structural_columns = [
        col
        for col in [
            "cluster",
            "cluster_str",
            "group_label",
            "group_base",
            "highlight"
        ]
        if col in df.columns
    ]

    return (
        structural_columns
        +
        lens_columns
    )


# =====================================================
# SUMMARY METRICS
# =====================================================

def render_summary_metrics(
    df,
    dataset
):

    c1, c2, c3, c4 = st.columns(
        4
    )

    with c1:

        st.metric(
            "Solutions",
            len(df)
        )

    with c2:

        st.metric(
            "Attributes",
            len(df.columns)
        )

    with c3:

        st.metric(
            "Decision Variables",
            len(
                dataset[
                    "decision_variables"
                ]
            )
        )

    with c4:

        css_status = (
            "Active"
            if st.session_state.get(
                "css_enabled",
                False
            )
            else "Inactive"
        )

        st.metric(
            "CSS",
            css_status
        )


def render_lens_summary(
    df
):

    lens_columns = get_lens_columns(
        df
    )

    if len(lens_columns) == 0:

        st.caption(
            "No derived lens columns in the current set."
        )

        return

    st.caption(
        "Derived columns: "
        +
        ", ".join(
            lens_columns
        )
    )


#def render_export_button(
#    df ):

#    st.download_button(
#        label="⬇️ Export Current Set",
#        data=df.to_csv(
#            index=False
#        ),
#        file_name="current_set.csv",
#        mime="text/csv",
#        use_container_width=True
#    )


def get_summary_label():

    if st.session_state.get(
        "css_enabled",
        False
    ):

        return "Dataset Summary / Current CSS"

    return "Dataset Summary / Current Set"


# =====================================================
# MAIN RENDERER
# =====================================================

def render_summary(df, dataset):
    if df is None:
        st.error(
            "Dataset summary cannot be rendered "
            "because the current dataframe is empty." )
        return

    label = get_summary_label()

    with st.expander(f"📊 {label}", expanded=False):
        tab_overview, tab_current, tab_saved_soi = st.tabs(
            [
                "**| Overview |**",
                "**| Current Set |**",
                "**| Saved SOIs |**"
            ]
        )

        with tab_overview:
            render_summary_metrics(df, dataset)
            st.caption(
                f"Decision-variable prefix: "
                f"{dataset['config'].get('var_prefix')}" )
            render_lens_summary(df)
        with tab_current:
            render_dataset_table(df, dataset)
        with tab_saved_soi:
            render_soi_tab()
        
