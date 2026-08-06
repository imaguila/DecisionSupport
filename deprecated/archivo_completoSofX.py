
# --- ARCHIVO: __init__.py ---

"""
Plugins Package Initializer.

Exports registered domain plugins and provides lookup and factory mechanisms
for dynamic plugin instantiation across the framework.
"""

from typing import Any, Dict, List, Optional, Type

from .aerospace_plugin import AerospacePlugin
from .nrp_plugin import NRPPlugin

# =====================================================
# PLUGIN REGISTRY
# =====================================================

PLUGIN_REGISTRY: Dict[str, Type[Any]] = {
    "nrp": NRPPlugin,
    "aerospace": AerospacePlugin,
}

__all__ = [
    "NRPPlugin",
    "AerospacePlugin",
    "PLUGIN_REGISTRY",
    "get_plugin",
    "list_plugins",
]

# =====================================================
# HELPER & FACTORY FUNCTIONS
# =====================================================


def list_plugins() -> List[str]:
    """
    Retrieves a list of all registered domain plugin identifiers.

    Returns
    -------
    List[str]
        List of registered plugin name strings.
    """
    return list(PLUGIN_REGISTRY.keys())


def get_plugin(plugin_name: str, **kwargs: Any) -> Optional[Any]:
    """
    Instantiates and returns a domain plugin by identifier name.

    Parameters
    ----------
    plugin_name : str
        Identifier key of the requested plugin (e.g., 'nrp', 'aerospace').
    **kwargs : Any
        Keyword arguments passed directly to the target plugin's `__init__`.

    Returns
    -------
    Optional[Any]
        An instance of the requested plugin class, or None if key is unrecognized.
    """
    plugin_cls = PLUGIN_REGISTRY.get(plugin_name)
    if plugin_cls is None:
        return None
    return plugin_cls(**kwargs)

# --- ARCHIVO: aerospace_plugin.py ---

"""
Aerospace Domain Plugin Module.

Provides domain-specific quality and engineering indicators for aerodynamic 
and structural evaluation of aerospace design space solutions.
"""

import logging
from typing import Dict, Iterable, List, Set

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

EPS: float = 1e-9


class AerospacePlugin:
    """
    Aerospace domain plugin for multi-objective solution space enrichment.

    Provides synthetic engineering indicators derived from aerodynamic
    objectives and decision variables.

    Parameters
    ----------
    var_prefix : str, default="var_"
        Prefix used to identify decision-variable columns in datasets.
    """

    def __init__(self, var_prefix: str = "var_") -> None:
        self.var_prefix: str = var_prefix

    def available_indicators(self) -> Set[str]:
        """
        Retrieves the set of indicators supported by this plugin.

        Returns
        -------
        Set[str]
            Set of indicator column names.
        """
        return {
            "density",
            "lift_to_drag_ratio",
            "structural_efficiency",
        }

    def requirements(self) -> Dict[str, List[str]]:
        """
        Retrieves objective column dependencies required for each indicator.

        Returns
        -------
        Dict[str, List[str]]
            Mapping of indicator names to required input DataFrame columns.
        """
        return {
            "density": ["weight"],
            "lift_to_drag_ratio": ["drag", "weight"],
            "structural_efficiency": ["drag", "weight"],
        }

    def decision_variables(self, df: pd.DataFrame) -> List[str]:
        """
        Identifies decision-variable columns matching the configured prefix.

        Parameters
        ----------
        df : pd.DataFrame
            Dataset containing solution decision variables and objectives.

        Returns
        -------
        List[str]
            List of column names starting with `var_prefix`.
        """
        return [c for c in df.columns if str(c).startswith(self.var_prefix)]

    def compute_indicators(
        self,
        df: pd.DataFrame,
        selected_indicators: Iterable[str],
    ) -> pd.DataFrame:
        """
        Computes requested domain indicators and appends them to a DataFrame copy.

        Parameters
        ----------
        df : pd.DataFrame
            Source solution dataset.
        selected_indicators : Iterable[str]
            Names of indicators to calculate.

        Returns
        -------
        pd.DataFrame
            Enriched DataFrame containing requested indicator columns.
        """
        result = df.copy()
        vars_cols = self.decision_variables(result)
        n_vars = max(len(vars_cols), 1)

        for indicator in selected_indicators:
            try:
                if indicator == "density":
                    result[indicator] = result["weight"] / n_vars

                elif indicator == "lift_to_drag_ratio":
                    pseudo_lift = result["weight"] * 0.25
                    denom = np.maximum(result["drag"].values, EPS)
                    result[indicator] = pseudo_lift / denom

                elif indicator == "structural_efficiency":
                    max_drag = max(float(result["drag"].max()), EPS)
                    max_weight = max(float(result["weight"].max()), EPS)

                    norm_drag = result["drag"] / max_drag
                    norm_weight = result["weight"] / max_weight

                    denom = (norm_drag * norm_weight) + EPS
                    result[indicator] = 1.0 / denom

            except Exception as exc:
                logger.warning(
                    "[AerospacePlugin] Unable to compute '%s': %s",
                    indicator,
                    exc,
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

"""
Column Classifier Module.

Provides dynamic column classification and exclusion logic based on problem 
configuration metadata, categorizing dataset features into Decision Variables (X), 
Base Optimization Metrics (M), and Derived Indicators (I).
"""

from typing import Any, Dict, List, Set

import pandas as pd


class ColumnClassifier:
    """
    Handles dynamic column classification and exclusions based on problem configuration.

    Categorizes dataset attributes into Decision Variables, Base Metrics, 
    and Derived Indicators while filtering out framework metadata columns.

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary containing 'metrics', 'var_prefix', and 'exclude_cols'.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        if config is None:
            config = {}

        self.metrics: Set[str] = set(config.get("metrics", []))
        self.var_prefix: str = str(config.get("var_prefix", "x_"))
        self.user_excludes: Set[str] = set(config.get("exclude_cols", []))

        # Internal system-level columns generated dynamically by the framework
        self.system_excludes: Set[str] = {
            "id",
            "ID",
            "highlight",
            "label",
            "highlight_label",
            "score",
            "cluster",
            "cluster_str",
            "group_label",
            "group_base",
            "selected",
            "preference_score",
            "preference_rank",
            "efficiency_score",
            "efficiency_rank",
            "domain_match_count",
            "domain_rank",
            "consensus_score",
            "consensus_support_count",
            "consensus_rank",
        }

    def get_decision_variables(self, df: pd.DataFrame) -> List[str]:
        """
        Extracts decision variable columns (X) matching the configured variable prefix.

        Parameters
        ----------
        df : pd.DataFrame
            Target solution space DataFrame.

        Returns
        -------
        List[str]
            List of matching decision variable column names.
        """
        if df is None or df.empty or not self.var_prefix:
            return []

        return [col for col in df.columns if col.startswith(self.var_prefix)]

    def get_metrics(self, df: pd.DataFrame) -> List[str]:
        """
        Extracts base optimization metrics (M) defined in the configuration.

        Parameters
        ----------
        df : pd.DataFrame
            Target solution space DataFrame.

        Returns
        -------
        List[str]
            List of present metric column names.
        """
        if df is None or df.empty:
            return []

        return [col for col in df.columns if col in self.metrics]

    def get_derived_indicators(self, df: pd.DataFrame) -> List[str]:
        """
        Extracts derived enrichment indicators (I).

        Identifies numeric columns that are neither base metrics, decision variables, 
        nor framework-excluded system attributes.

        Parameters
        ----------
        df : pd.DataFrame
            Target solution space DataFrame.

        Returns
        -------
        List[str]
            List of identified derived indicator column names.
        """
        if df is None or df.empty:
            return []

        all_excluded = self.system_excludes | self.user_excludes | self.metrics
        has_prefix = bool(self.var_prefix)

        indicators: List[str] = []
        for col in df.columns:
            if col in all_excluded:
                continue
            if has_prefix and col.startswith(self.var_prefix):
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                indicators.append(col)

        return indicators

# --- ARCHIVO: config.py ---

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

from typing import Dict, List, Optional, TypedDict


class CaseConfig(TypedDict, total=False):
    """Schema definition for domain case configurations."""

    plugin: str
    path_sol: str
    metrics: List[str]
    var_prefix: str
    num_x: int
    exclude_cols: List[str]
    default_indicators: List[str]
    help: str


# =====================================================================
# CONFIGURATION REGISTRY
# =====================================================================

CASES: Dict[str, CaseConfig] = {
    # -----------------------------------------------------------------
    # CASE 1: Software Release Planning - CLASSIC Dataset
    # -----------------------------------------------------------------
    "CLASSIC Dataset": {
        "plugin": "nrp",
        "path_sol": "data/bagnallsoluciones.csv",
        "metrics": ["satisfaction", "effort"],
        "var_prefix": "req_",
        "num_x": 18,
        "exclude_cols": ["id", "run_id", "timestamp"],
        "default_indicators": ["scope", "productivity", "squandering"],
        "help": (
            "Greer, D., & Ruhe, G. (2004). Software release planning: an evolutionary "
            "and iterative approach. Information and Software Technology, 46(4), 243-253."
        ),
    },
    # -----------------------------------------------------------------
    # CASE 2: Software Release Planning - MSLite System
    # -----------------------------------------------------------------
    "MSLite System": {
        "plugin": "nrp",
        "path_sol": "data/mslitesoluciones.csv",
        "metrics": ["satisfaction", "effort", "dissatisfaction"],
        "var_prefix": "req_",
        "num_x": 16,
        "exclude_cols": ["id", "run_id", "timestamp"],
        "default_indicators": [
            "scope",
            "productivity",
            "squandering",
            "annoyance",
            "dirtiness",
        ],
        "help": (
            "Sangwan, R. S., Negahban, A., Nord, R. L., & Ozkaya, I. (2020). "
            "Optimization of software release planning considering architectural "
            "dependencies, cost, and value. IEEE Transactions on Software Engineering, "
            "48(4), 1369-1384."
        ),
    },
    # -----------------------------------------------------------------
    # CASE 3: Replacement Access, Library and ID Card (RALIC)
    # -----------------------------------------------------------------
    "Replacement Access, Library and ID Card (RALIC)": {
        "plugin": "nrp",
        "path_sol": "data/ralic.csv",
        "metrics": ["satisfaction", "effort"],
        "var_prefix": "req_",
        "num_x": 83,
        "exclude_cols": ["id", "run_id", "timestamp"],
        "default_indicators": ["scope", "productivity", "squandering"],
        "help": (
            "Lim, S. L., & Finkelstein, A. (2011). StakeRare: using social networks "
            "and collaborative filtering for large-scale requirements elicitation. "
            "IEEE Transactions on Software Engineering, 38(3), 707-735."
        ),
    },
    # -----------------------------------------------------------------
    # CASE 4: Word Processing Software Project
    # -----------------------------------------------------------------
    "Word Processing Software Project": {
        "plugin": "nrp",
        "path_sol": "data/wordprocsoluciones.csv",
        "metrics": ["satisfaction", "effort", "time"],
        "var_prefix": "req_",
        "num_x": 42,
        "exclude_cols": ["id", "run_id", "timestamp"],
        "default_indicators": [
            "scope",
            "productivity",
            "squandering",
            "response",
            "opportunity",
        ],
        "help": (
            "Agarwal, N., Karimpour, R., & Ruhe, G. (2014). Theme-based product "
            "release planning: An analytical approach. In 2014 47th Hawaii International "
            "Conference on System Sciences, pp. 4739-4748. IEEE."
        ),
    },
    # -----------------------------------------------------------------
    # CASE 5: Large Dataset - REQ100
    # -----------------------------------------------------------------
    "Large Dataset": {
        "plugin": "nrp",
        "path_sol": "data/req100frente.csv",
        "metrics": ["satisfaction", "effort"],
        "var_prefix": "req_",
        "num_x": 96,
        "exclude_cols": ["id", "run_id", "timestamp"],
        "default_indicators": ["scope", "productivity", "squandering"],
        "help": (
            "Del Sagrado, J., Del Águila, I. M., & Orellana, F. J. (2015). Multi-objective "
            "ant colony optimization for requirements selection. Empirical Software "
            "Engineering, 20(3), 577-610."
        ),
    },
    # -----------------------------------------------------------------
    # CASE 6: ReleasePlanner Dataset - THEME
    # -----------------------------------------------------------------
    "ReleasePlanner™ Dataset": {
        "plugin": "nrp",
        "path_sol": "data/themesoluciones.csv",
        "metrics": [
            "satisfaction",
            "prevalence",
            "cost",
            "dissatisfaction",
            "inestability",
            "effort",
        ],
        "var_prefix": "req_",
        "num_x": 22,
        "exclude_cols": ["id", "run_id", "timestamp"],
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
            "usage_efficiency",
        ],
        "help": (
            "Karim, M. R., & Ruhe, G. (2014). Bi-objective genetic search for release "
            "planning in support of themes. In International Symposium on Search Based "
            "Software Engineering, pp. 123-137. Springer International Publishing."
        ),
    },
    # -----------------------------------------------------------------
    # CASE 7: Motorola Dataset
    # -----------------------------------------------------------------
    "Motorola Dataset": {
        "plugin": "nrp",
        "path_sol": "data/motorolasoluciones.csv",
        "metrics": ["satisfaction", "effort"],
        "var_prefix": "req_",
        "num_x": 35,
        "exclude_cols": ["id", "run_id", "timestamp"],
        "default_indicators": ["scope", "productivity", "squandering"],
        "help": (
            "Baker, P., Harman, M., Steinhofel, K., & Skaliotis, A. (2006). Search based "
            "approaches to component selection and prioritization for the next release "
            "problem. In 2006 22nd IEEE International Conference on Software Maintenance, "
            "pp. 176-185. IEEE."
        ),
    },
    # -----------------------------------------------------------------
    # CASE 8: Generic Engineering Design - Aerospace Wing Design
    # -----------------------------------------------------------------
    "Aerospace Wing Design": {
        "plugin": "aerospace",
        "path_sol": "data/wing_pareto_front.csv",
        "metrics": ["drag", "weight"],
        "var_prefix": "var_",
        "num_x": 10,
        "exclude_cols": ["sim_time", "solver_status"],
        "default_indicators": [
            "density",
            "lift_to_drag_ratio",
            "structural_efficiency",
        ],
        "help": (
            "Example, A. et al. (2025). Multi-objective aerodynamic design optimization "
            "of aircraft wings. Journal of Aircraft, 62(1), 100-115."
        ),
    },
}


# =====================================================================
# ACCESSOR UTILITIES
# =====================================================================


def get_available_cases() -> List[str]:
    """
    Returns a list of all pre-configured domain case names.

    Returns
    -------
    List[str]
        Names of all available cases in the registry.
    """
    return list(CASES.keys())


def get_case_config(case_name: str) -> Optional[CaseConfig]:
    """
    Safely retrieves the configuration dictionary for a target case name.

    Parameters
    ----------
    case_name : str
        Name of the case entry.

    Returns
    -------
    Optional[CaseConfig]
        Configuration metadata dictionary, or None if case name does not exist.
    """
    return CASES.get(case_name)

# --- ARCHIVO: css_comparison.py ---

"""
Candidate Solution Set (CSS) Comparison Module.

Provides visual trade-off analysis, structural decision-variable inspection, 
parallel coordinate mapping, baseline difference metrics, and X -> Y 
correlation heatmaps for candidate solution subsets.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# =====================================================
# HELPER FUNCTIONS
# =====================================================


def get_numeric_dimensions(
    df: pd.DataFrame, dataset: Dict[str, Any]
) -> List[str]:
    """
    Extracts numeric objective and indicator column names present in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Active solution space DataFrame.
    dataset : Dict[str, Any]
        Global dataset context dictionary.

    Returns
    -------
    List[str]
        List of verified numeric metric and indicator column names.
    """
    if df is None or df.empty or not dataset:
        return []

    metrics = dataset.get("metrics", []) or []
    indicators = dataset.get("selected_indicators", []) or []
    dimensions = list(metrics) + list(indicators)

    return [
        col
        for col in dimensions
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
    ]


def get_decision_variable_columns(
    df: pd.DataFrame, dataset: Dict[str, Any]
) -> List[str]:
    """
    Retrieves decision variable column names matching the configured variable prefix.

    Parameters
    ----------
    df : pd.DataFrame
        Active solution space DataFrame.
    dataset : Dict[str, Any]
        Global dataset context dictionary.

    Returns
    -------
    List[str]
        List of numeric decision variable column names.
    """
    if df is None or df.empty or not dataset:
        return []

    config = dataset.get("config", {}) if dataset else {}
    var_prefix = config.get("var_prefix", "x_")

    return [
        col
        for col in df.columns
        if var_prefix
        and col.startswith(var_prefix)
        and pd.api.types.is_numeric_dtype(df[col])
    ]


def normalize_metric(series: pd.Series, goal: str) -> pd.Series:
    """
    Normalizes a numeric Pandas Series to the range [0.0, 1.0] based on optimization goal.

    Parameters
    ----------
    series : pd.Series
        Numeric metric values to normalize.
    goal : str
        Optimization goal ("Maximize" or "Minimize").

    Returns
    -------
    pd.Series
        Normalized metric values scaled from 0.0 to 1.0.
    """
    if series.empty:
        return series

    min_v = series.min()
    max_v = series.max()

    if max_v <= min_v:
        return pd.Series(0.5, index=series.index)

    normalized = (series - min_v) / (max_v - min_v)

    if goal == "Minimize":
        normalized = 1.0 - normalized

    return normalized


# =====================================================
# TRADE-OFF RADAR CHART
# =====================================================


def render_tradeoff_radar(
    compare_df: pd.DataFrame, css_df: pd.DataFrame, dataset: Dict[str, Any]
) -> None:
    """
    Renders a polar radar chart comparing normalized solution profiles across objectives.

    Parameters
    ----------
    compare_df : pd.DataFrame
        DataFrame containing selected solutions to compare.
    css_df : pd.DataFrame
        Full Candidate Solution Set DataFrame used for dimension discovery.
    dataset : Dict[str, Any]
        Global dataset context dictionary.
    """
    numeric_dimensions = get_numeric_dimensions(css_df, dataset)

    if len(numeric_dimensions) < 3:
        st.info(
            "At least three numeric objectives or indicators are required to create a radar chart."
        )
        return

    selected_metrics = st.multiselect(
        "Objectives and indicators for radar profile",
        numeric_dimensions,
        default=numeric_dimensions[: min(5, len(numeric_dimensions))],
        key="css_tradeoff_metrics",
    )

    if len(selected_metrics) < 3:
        st.warning("Select at least three objectives or indicators.")
        return

    metric_goals = {}
    cols = st.columns(len(selected_metrics))

    for idx, metric in enumerate(selected_metrics):
        with cols[idx]:
            metric_goals[metric] = st.selectbox(
                metric,
                ["Maximize", "Minimize"],
                key=f"css_goal_{metric}",
            )

    radar_df = compare_df.copy()

    for metric in selected_metrics:
        radar_df[metric] = normalize_metric(
            radar_df[metric], metric_goals[metric]
        )

    fig = go.Figure()

    for _, row in radar_df.iterrows():
        values = row[selected_metrics].tolist()
        values.append(values[0])
        theta = selected_metrics + [selected_metrics[0]]

        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=theta,
                mode="lines+markers",
                name=f"ID {int(row['id'])}",
            )
        )

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        template="plotly_white",
        height=450,
        margin=dict(t=40, b=40),
    )

    st.plotly_chart(fig, use_container_width=True)


# =====================================================
# PARALLEL COORDINATES
# =====================================================


def render_parallel_coordinates(
    compare_df: pd.DataFrame, dataset: Dict[str, Any]
) -> None:
    """
    Renders a Parallel Coordinates plot mapping multi-dimensional solution tradeoffs.

    Parameters
    ----------
    compare_df : pd.DataFrame
        DataFrame containing selected solutions to compare.
    dataset : Dict[str, Any]
        Global dataset context dictionary.
    """
    numeric_dims = get_numeric_dimensions(compare_df, dataset)

    if len(numeric_dims) < 2:
        st.info(
            "At least two numerical metrics are required for Parallel Coordinates."
        )
        return

    selected_dims = st.multiselect(
        "Metrics for Parallel Coordinates",
        numeric_dims,
        default=numeric_dims[: min(6, len(numeric_dims))],
        key="css_parcoords_dims",
    )

    if not selected_dims:
        return

    dimensions_config = [
        dict(
            range=[compare_df[col].min(), compare_df[col].max()],
            label=col,
            values=compare_df[col],
        )
        for col in selected_dims
    ]

    fig = go.Figure(
        data=go.Parcoords(
            line=dict(
                color=compare_df["id"],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Solution ID"),
            ),
            dimensions=dimensions_config,
        )
    )

    fig.update_layout(
        template="plotly_white", height=400, margin=dict(t=40, b=40)
    )
    st.plotly_chart(fig, use_container_width=True)


# =====================================================
# BASELINE DIFFERENCE / GAP ANALYSIS
# =====================================================


def render_baseline_difference_chart(
    compare_df: pd.DataFrame, dataset: Dict[str, Any]
) -> None:
    """
    Renders a relative percentage difference bar chart relative to a selected baseline solution.

    Parameters
    ----------
    compare_df : pd.DataFrame
        DataFrame containing selected solutions to compare.
    dataset : Dict[str, Any]
        Global dataset context dictionary.
    """
    numeric_dims = get_numeric_dimensions(compare_df, dataset)
    if not numeric_dims or "id" not in compare_df.columns:
        return

    col_base, col_metrics = st.columns([1, 2])

    with col_base:
        baseline_id = st.selectbox(
            "Select Baseline Solution",
            options=compare_df["id"].tolist(),
            format_func=lambda x: f"ID {int(x)}",
            key="css_baseline_id",
        )

    with col_metrics:
        selected_metrics = st.multiselect(
            "Metrics to compare vs Baseline",
            numeric_dims,
            default=numeric_dims[: min(4, len(numeric_dims))],
            key="css_baseline_metrics",
        )

    if not selected_metrics or baseline_id is None:
        return

    baseline_row = compare_df[compare_df["id"] == baseline_id].iloc[0]
    other_df = compare_df[compare_df["id"] != baseline_id].copy()

    if other_df.empty:
        st.info(
            "Select at least one additional solution to compare against the Baseline."
        )
        return

    diff_data = []
    for _, row in other_df.iterrows():
        for metric in selected_metrics:
            base_val = baseline_row[metric]
            curr_val = row[metric]

            if base_val != 0:
                pct_change = ((curr_val - base_val) / abs(base_val)) * 100
            else:
                pct_change = 0.0 if curr_val == 0 else np.nan

            diff_data.append(
                {
                    "Solution": f"ID {int(row['id'])}",
                    "Metric": metric,
                    "Relative Change (%)": pct_change,
                    "Absolute Difference": curr_val - base_val,
                }
            )

    diff_df = pd.DataFrame(diff_data)

    fig = px.bar(
        diff_df,
        x="Metric",
        y="Relative Change (%)",
        color="Solution",
        barmode="group",
        hover_data=["Absolute Difference"],
    )

    fig.add_hline(y=0, line_dash="dash", line_color="black")
    fig.update_layout(template="plotly_white", height=400)
    st.plotly_chart(fig, use_container_width=True)


# =====================================================
# SOLUTION SIMILARITY MATRIX
# =====================================================


def render_solution_similarity_matrix(
    compare_df: pd.DataFrame, dataset: Dict[str, Any]
) -> None:
    """
    Computes and displays pairwise decision-variable similarity correlation between solutions.

    Parameters
    ----------
    compare_df : pd.DataFrame
        DataFrame containing selected solutions to compare.
    dataset : Dict[str, Any]
        Global dataset context dictionary.
    """
    var_cols = get_decision_variable_columns(compare_df, dataset)
    if not var_cols or len(compare_df) < 2:
        st.info(
            "Requires decision variables and at least 2 solutions to compute similarity."
        )
        return

    matrix_df = compare_df.set_index("id")[var_cols]
    sim_matrix = matrix_df.T.corr().fillna(0.0)

    sim_matrix.index = [f"ID {int(i)}" for i in sim_matrix.index]
    sim_matrix.columns = [f"ID {int(c)}" for c in sim_matrix.columns]

    fig = px.imshow(
        sim_matrix,
        text_auto=".2f",
        color_continuous_scale="RdBu",
        zmin=-1.0,
        zmax=1.0,
        labels=dict(color="Correlation"),
    )

    fig.update_layout(template="plotly_white", height=450)
    st.plotly_chart(fig, use_container_width=True)


# =====================================================
# DECISION-VARIABLE MATRIX
# =====================================================


def render_decision_variable_matrix(
    compare_df: pd.DataFrame, dataset: Dict[str, Any]
) -> None:
    """
    Renders a structural heatmap matrix of decision variable values per candidate solution.

    Parameters
    ----------
    compare_df : pd.DataFrame
        DataFrame containing selected solutions to compare.
    dataset : Dict[str, Any]
        Global dataset context dictionary.
    """
    variable_cols = get_decision_variable_columns(compare_df, dataset)
    config = dataset.get("config", {}) if dataset else {}
    var_prefix = config.get("var_prefix", "x_")

    if not variable_cols:
        st.info(
            f"No numeric decision-variable columns with prefix '{var_prefix}' found."
        )
        return

    matrix_df = compare_df.set_index("id")[variable_cols].copy()
    matrix_df.index = [f"ID {int(idx)}" for idx in matrix_df.index]

    fig = px.imshow(
        matrix_df,
        labels=dict(x="Decision variables", y="Solutions", color="Value"),
        color_continuous_scale=[[0, "#e0e0e0"], [1, "#00e676"]],
    )

    fig.update_layout(
        template="plotly_white",
        coloraxis_showscale=False,
        xaxis=dict(tickangle=-45, showgrid=False),
        yaxis=dict(autorange="reversed", showgrid=False),
        height=520,
    )

    fig.update_traces(
        xgap=3,
        ygap=3,
        hovertemplate="<b>%{y}</b><br>Variable: %{x}<br>Value: %{z}<extra></extra>",
    )

    st.plotly_chart(fig, use_container_width=True)


# =====================================================
# DECISION-VARIABLE DISTRIBUTION
# =====================================================


def render_decision_variable_distribution(
    css_df: pd.DataFrame, dataset: Dict[str, Any]
) -> None:
    """
    Renders a bar chart summarizing average activation/selection rates across decision variables.

    Parameters
    ----------
    css_df : pd.DataFrame
        Full Candidate Solution Set DataFrame.
    dataset : Dict[str, Any]
        Global dataset context dictionary.
    """
    variable_cols = get_decision_variable_columns(css_df, dataset)
    config = dataset.get("config", {}) if dataset else {}
    var_prefix = config.get("var_prefix", "x_")

    if not variable_cols:
        st.info(
            f"No numeric decision-variable columns with prefix '{var_prefix}' found."
        )
        return

    variable_summary = css_df[variable_cols].mean().reset_index()
    variable_summary.columns = ["decision_variable", "selection_rate"]
    variable_summary = variable_summary.sort_values(
        "selection_rate", ascending=False
    )

    max_variables = min(50, len(variable_summary))
    if max_variables < 1:
        st.info("No decision variables can be summarized.")
        return

    top_n = st.slider(
        "Decision variables to show",
        min_value=1,
        max_value=max_variables,
        value=min(20, max_variables),
        key="css_decision_variable_top_n",
    )

    plot_df = variable_summary.head(top_n)

    fig = px.bar(
        plot_df,
        x="decision_variable",
        y="selection_rate",
        labels={
            "decision_variable": "Decision variable",
            "selection_rate": "Mean Value / Selection rate",
        },
    )

    fig.update_layout(template="plotly_white", height=420, xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)


# =====================================================
# VARIABLE TO METRIC MAPPING (X vs Y Correlation)
# =====================================================


def render_variable_metric_correlation(
    compare_df: pd.DataFrame, dataset: Dict[str, Any]
) -> None:
    """
    Renders an X -> Y correlation matrix heatmap (Decision Variables vs. Metrics).

    Parameters
    ----------
    compare_df : pd.DataFrame
        DataFrame containing selected solutions to compare.
    dataset : Dict[str, Any]
        Global dataset context dictionary.
    """
    var_cols = get_decision_variable_columns(compare_df, dataset)
    metric_cols = get_numeric_dimensions(compare_df, dataset)

    if not var_cols or not metric_cols:
        st.info(
            "Both decision variables and numeric metrics are required to compute mapping."
        )
        return

    if len(compare_df) < 2:
        st.info("Select at least 2 solutions to calculate correlation.")
        return

    combined_df = compare_df[var_cols + metric_cols]
    corr_matrix = combined_df.corr()

    xy_corr = (
        corr_matrix.loc[var_cols, metric_cols].dropna(how="all").fillna(0.0)
    )

    if xy_corr.empty:
        st.info(
            "Could not calculate variance/correlation for the selected subset."
        )
        return

    fig = px.imshow(
        xy_corr,
        labels=dict(
            x="Metrics / Objectives (Y)",
            y="Decision Variables (X)",
            color="Correlation",
        ),
        color_continuous_scale="RdBu",
        zmin=-1.0,
        zmax=1.0,
        aspect="auto",
    )

    fig.update_layout(
        template="plotly_white",
        height=max(400, len(var_cols) * 20),
        xaxis=dict(tickangle=-45),
    )

    st.plotly_chart(fig, use_container_width=True)


# =====================================================
# MAIN CSS COMPARISON PIPELINE
# =====================================================


def render_css_comparison(
    css_df: pd.DataFrame, dataset: Dict[str, Any]
) -> None:
    """
    Main entry point for rendering the detailed Candidate Solution Set (CSS) comparison panel.

    Parameters
    ----------
    css_df : pd.DataFrame
        Active Candidate Solution Set DataFrame.
    dataset : Dict[str, Any]
        Global dataset context dictionary.
    """
    if not st.session_state.get("show_css_comparison", False):
        return

    with st.expander("🆚 Detailed comparison", expanded=True):
        if css_df is None or css_df.empty:
            st.info("No Candidate Solution Set is available for comparison.")
            return

        if "id" not in css_df.columns:
            st.warning("The current CSS does not contain an 'id' column.")
            return

        css_ids = css_df["id"].dropna().astype(int).tolist()
        default_ids = st.session_state.get("css_highlight_ids", [])
        default_ids = [sid for sid in default_ids if sid in css_ids]

        compare_ids = st.multiselect(
            "Pick solutions to compare & highlight",
            css_ids,
            default=default_ids,
            key="css_compare_ids",
        )

        st.session_state.css_highlight_ids = compare_ids

        if len(compare_ids) < 2:
            st.info("Select at least 2 solutions to compare.")
            return

        compare_df = css_df[css_df["id"].isin(compare_ids)].copy()

        tab_metrics, tab_vars, tab_sim, tab_mapping = st.tabs(
            [
                "📊 Metrics & Trade-offs",
                "📋 Decision Variables",
                "🔀 Structural Similarity",
                "🔗 X → Y Mapping",
            ]
        )

        with tab_metrics:
            render_tradeoff_radar(compare_df, css_df, dataset)
            st.divider()
            render_parallel_coordinates(compare_df, dataset)
            st.divider()
            render_baseline_difference_chart(compare_df, dataset)

        with tab_vars:
            render_decision_variable_matrix(compare_df, dataset)
            st.divider()
            render_decision_variable_distribution(css_df, dataset)

        with tab_sim:
            render_solution_similarity_matrix(compare_df, dataset)

        with tab_mapping:
            render_variable_metric_correlation(compare_df, dataset)

# --- ARCHIVO: css_panel.py ---

"""
Candidate Solution Set (CSS) Panel Module.

Provides sidebar controls and session state management for filtering, locking, 
and highlighting Candidate Solution Sets (CSS) across the visual workspace.
"""

from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st


# =====================================================
# SESSION STATE MANAGEMENT
# =====================================================


def ensure_css_state() -> None:
    """Ensures all session state keys required for CSS management are initialized."""
    defaults: Dict[str, Any] = {
        "css_enabled": False,
        "css_source": "Current set",
        "css_manual_ids": [],
        "css_highlight_ids": [],
        "show_css_comparison": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def sanitize_ids(ids: List[Any], valid_ids: List[Any]) -> List[Any]:
    """
    Filters an ID list to retain only IDs present in valid_ids.

    Parameters
    ----------
    ids : List[Any]
        List of candidate IDs to validate.
    valid_ids : List[Any]
        List of active valid solution IDs.

    Returns
    -------
    List[Any]
        Filtered list of valid solution IDs.
    """
    if not ids or not valid_ids:
        return []

    valid_set = set(valid_ids)
    return [solution_id for solution_id in ids if solution_id in valid_set]


# =====================================================
# SIDEBAR PANEL RENDERER
# =====================================================


def render_css_panel(
    current_df: Optional[pd.DataFrame],
    dataset: Optional[Dict[str, Any]] = None,
) -> Optional[pd.DataFrame]:
    """
    Renders sidebar controls for managing the Candidate Solution Set (CSS).

    Allows users to lock the active solution space or manually select 
    solutions, dynamically tagging highlighted items.

    Parameters
    ----------
    current_df : Optional[pd.DataFrame]
        Active solution space DataFrame.
    dataset : Optional[Dict[str, Any]], optional
        Global dataset context dictionary, by default None.

    Returns
    -------
    Optional[pd.DataFrame]
        Filtered DataFrame representing the active Candidate Solution Set (CSS).
    """
    ensure_css_state()

    if current_df is None or current_df.empty:
        return current_df

    css_df = current_df.copy()
    valid_ids = (
        css_df["id"].dropna().tolist() if "id" in css_df.columns else []
    )

    st.session_state.css_manual_ids = sanitize_ids(
        st.session_state.css_manual_ids, valid_ids
    )
    st.session_state.css_highlight_ids = sanitize_ids(
        st.session_state.css_highlight_ids, valid_ids
    )

    with st.sidebar.expander("🎯 Candidate Solution Set", expanded=False):
        st.session_state.css_enabled = st.checkbox(
            "Lock current set as CSS",
            value=st.session_state.css_enabled,
            help=(
                "Create a Candidate Solution Set from current filtered set "
                "or manual selection."
            ),
        )

        if not st.session_state.css_enabled:
            st.caption(f"Current set available: {len(current_df)} solutions")
            css_df["highlight"] = False
            return css_df

        sources = ["Current set", "Manual selection"]
        source_idx = (
            sources.index(st.session_state.css_source)
            if st.session_state.css_source in sources
            else 0
        )

        st.session_state.css_source = st.radio(
            "CSS source",
            sources,
            index=source_idx,
            horizontal=True,
        )

        if st.session_state.css_source == "Manual selection":
            st.session_state.css_manual_ids = st.multiselect(
                "Solutions included in CSS",
                options=valid_ids,
                default=st.session_state.css_manual_ids,
                key="css_manual_ids_widget",
                help="Select the exact solutions that form the Candidate Solution Set.",
            )
            css_df = current_df[
                current_df["id"].isin(st.session_state.css_manual_ids)
            ].copy()
        else:
            css_df = current_df.copy()

        st.info(f"CSS size: {len(css_df)} solutions")

        st.session_state.show_css_comparison = st.checkbox(
            "Open detailed comparison",
            value=st.session_state.show_css_comparison,
            help="Open detailed visual comparison section for the current CSS.",
        )

    if "id" in css_df.columns:
        css_df["highlight"] = css_df["id"].isin(
            st.session_state.css_highlight_ids
        )
    else:
        css_df["highlight"] = False

    return css_df

# --- ARCHIVO: enrichment.py ---

"""
Data Enrichment Module.

Provides functionality to detect, select, and compute derived indicators 
for candidate solutions based on domain-specific plugin requirements.
"""

from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from ui.phase_help import render_phase_help_icon


# =====================================================
# HELPER FUNCTIONS
# =====================================================


def get_available_indicators(
    plugin: Any, selected_metrics: List[str]
) -> List[str]:
    """
    Identifies compatible indicators based on selected metrics and plugin requirements.

    Parameters
    ----------
    plugin : Any
        Domain plugin instance offering requirement checks.
    selected_metrics : List[str]
        List of currently active dataset metric names.

    Returns
    -------
    List[str]
        List of indicator names whose metric requirements are fully satisfied.
    """
    if not plugin or not hasattr(plugin, "requirements"):
        return []

    available_indicators: List[str] = []
    requirements: Dict[str, List[str]] = plugin.requirements()

    for indicator, reqs in requirements.items():
        if all(metric in selected_metrics for metric in reqs):
            available_indicators.append(indicator)

    return available_indicators


# =====================================================
# UI RENDERING & COMPUTATION ENTRY POINT
# =====================================================


def render_enrichment(dataset: Dict[str, Any]) -> Dict[str, Any]:
    """
    Renders the Data Enrichment sidebar UI section and computes selected indicators.

    Parameters
    ----------
    dataset : Dict[str, Any]
        Global dataset context dictionary containing plugin, df, and metadata.

    Returns
    -------
    Dict[str, Any]
        Updated dataset dictionary enriched with computed indicator features.
    """
    if not dataset:
        return {}

    plugin = dataset.get("plugin")
    if plugin is None:
        dataset["selected_indicators"] = []
        return dataset

    selected_metrics: List[str] = dataset.get("metrics", [])
    available_indicators = get_available_indicators(plugin, selected_metrics)

    config = dataset.get("config", {})
    default_indicators = [
        indicator
        for indicator in config.get("default_indicators", [])
        if indicator in available_indicators
    ]

    with st.sidebar.expander("⚙️ Data Enrichment", expanded=False):
        col_label, col_help = st.columns(
            [0.85, 0.15], vertical_alignment="center"
        )

        with col_label:
            st.markdown("**Derived Indicators**")

        with col_help:
            render_phase_help_icon("enrichment", key="help_enrichment_phase")

        st.caption(
            f"Detected {len(available_indicators)} compatible indicators."
        )

        selected_indicators = st.multiselect(
            "Available indicators",
            sorted(available_indicators),
            default=default_indicators,
        )

    df: Optional[pd.DataFrame] = dataset.get("df")
    if df is not None and hasattr(plugin, "compute_indicators"):
        dataset["df"] = plugin.compute_indicators(df, selected_indicators)

    dataset["selected_indicators"] = selected_indicators

    return dataset

# --- ARCHIVO: framing.py ---

"""
Context Framing Module.

Provides range-based bounding filters on numeric metrics and derived indicators 
to dynamically reduce the visible solution decision space.
"""

from typing import Any, Dict, List, Tuple

import pandas as pd
import streamlit as st

from ui.phase_help import render_phase_help_icon


# =====================================================
# HELPER FUNCTIONS
# =====================================================


def get_framing_dimensions(dataset: Dict[str, Any]) -> List[str]:
    """
    Extracts all filterable metric and indicator dimension names from dataset context.

    Parameters
    ----------
    dataset : Dict[str, Any]
        Global dataset configuration metadata.

    Returns
    -------
    List[str]
        List of dimension column names available for range filtering.
    """
    if not dataset:
        return []

    metrics = dataset.get("metrics", [])
    indicators = dataset.get("selected_indicators", [])

    return list(metrics) + list(indicators)


def is_valid_numeric_dimension(df: pd.DataFrame, column: str) -> bool:
    """
    Checks whether a column exists in the DataFrame and contains numeric data.

    Parameters
    ----------
    df : pd.DataFrame
        Target solution space DataFrame.
    column : str
        Column name to evaluate.

    Returns
    -------
    bool
        True if column exists and is numeric; False otherwise.
    """
    if df is None or column not in df.columns:
        return False

    return pd.api.types.is_numeric_dtype(df[column])


def apply_dimension_filter(
    filtered_df: pd.DataFrame,
    metric: str,
    selected_range: Tuple[float, float],
) -> pd.DataFrame:
    """
    Filters a DataFrame based on a closed numeric interval [min, max].

    Parameters
    ----------
    filtered_df : pd.DataFrame
        DataFrame to be filtered.
    metric : str
        Target column name to apply bounding condition.
    selected_range : Tuple[float, float]
        Lower and upper bounds for filtering.

    Returns
    -------
    pd.DataFrame
        Filtered solution space DataFrame.
    """
    if filtered_df is None or filtered_df.empty:
        return filtered_df

    return filtered_df[
        (filtered_df[metric] >= selected_range[0])
        & (filtered_df[metric] <= selected_range[1])
    ]


# =====================================================
# UI RENDERING & SUMMARY
# =====================================================


def render_framing_summary(
    original_df: pd.DataFrame, filtered_df: pd.DataFrame
) -> None:
    """
    Renders progress bar and ratio metrics summarizing solution space reduction.

    Parameters
    ----------
    original_df : pd.DataFrame
        Original un-filtered solution space DataFrame.
    filtered_df : pd.DataFrame
        Filtered active solution space DataFrame.
    """
    total_solutions = len(original_df) if original_df is not None else 0
    remaining_solutions = len(filtered_df) if filtered_df is not None else 0

    ratio = remaining_solutions / max(total_solutions, 1)
    st.progress(ratio)

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
        unsafe_allow_html=True,
    )

    st.caption(f"{ratio:.0%} of the decision space is visible.")


# =====================================================
# MAIN PIPELINE ENTRY POINT
# =====================================================


def apply_framing(dataset: Dict[str, Any]) -> pd.DataFrame:
    """
    Renders UI sliders for context framing and returns the filtered solution space.

    Parameters
    ----------
    dataset : Dict[str, Any]
        Global dataset context dictionary containing working DataFrame and metrics.

    Returns
    -------
    pd.DataFrame
        Range-bounded solution space DataFrame.
    """
    if not dataset or "df" not in dataset or dataset["df"] is None:
        return pd.DataFrame()

    df = dataset["df"].copy()
    if df.empty:
        return df

    filtered_df = df.copy()
    dimensions = get_framing_dimensions(dataset)

    with st.sidebar.expander("🎛️ Context Framing", expanded=False):
        col_label, col_help = st.columns(
            [0.85, 0.15], vertical_alignment="center"
        )

        with col_label:
            st.markdown("**Bounded Range Filters**")

        with col_help:
            render_phase_help_icon("framing", key="help_input_phase")

        for metric in dimensions:
            if not is_valid_numeric_dimension(df, metric):
                continue

            min_v = float(df[metric].min())
            max_v = float(df[metric].max())

            if pd.isna(min_v) or pd.isna(max_v) or min_v >= max_v:
                continue

            step_val = (max_v - min_v) / 1000.0

            selected_range = st.slider(
                metric,
                min_value=min_v,
                max_value=max_v,
                value=(min_v, max_v),
                step=step_val,
                key=f"framing_{metric}",
            )

            unchanged = (
                abs(selected_range[0] - min_v) < 1e-6
                and abs(selected_range[1] - max_v) < 1e-6
            )
            if unchanged:
                continue

            filtered_df = apply_dimension_filter(
                filtered_df, metric, selected_range
            )

        render_framing_summary(df, filtered_df)

    return filtered_df

# --- ARCHIVO: input_panel.py ---

"""
Input Panel UI and Data Loading Module.

Handles dataset selection, CSV file uploads, plugin initialization, and dynamic 
detection of objective metrics and decision variables.
"""

from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from config import CASES
from plugins import PLUGIN_REGISTRY
from ui.phase_help import render_help_icon, render_phase_help_icon


# =====================================================
# DETECTION & INFERENCE HELPERS
# =====================================================


def detect_decision_variables(df: pd.DataFrame, prefix: str) -> List[str]:
    """
    Identifies columns representing decision variables based on a string prefix.

    Parameters
    ----------
    df : pd.DataFrame
        Candidate solution space DataFrame.
    prefix : str
        Variable column prefix (e.g., 'x_', 'var_').

    Returns
    -------
    List[str]
        List of matching decision variable column names.
    """
    if df is None or df.empty or not prefix:
        return []

    return [col for col in df.columns if col.startswith(prefix)]


def infer_numeric_metrics(df: pd.DataFrame, cfg: Dict[str, Any]) -> List[str]:
    """
    Infers candidate objective metric columns by excluding system and variable columns.

    Parameters
    ----------
    df : pd.DataFrame
        Candidate solution space DataFrame.
    cfg : Dict[str, Any]
        Domain configuration dictionary.

    Returns
    -------
    List[str]
        List of numeric column names valid for objective analysis.
    """
    if df is None or df.empty:
        return []

    var_prefix = cfg.get("var_prefix", "x_")
    excluded = set(cfg.get("exclude_cols", []))

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
        "selected",
    }

    metrics: List[str] = []
    for col in df.columns:
        if col.startswith(var_prefix):
            continue
        if col in excluded or col in system_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            metrics.append(col)

    return metrics


# =====================================================
# PLUGIN & DATASET CONSTRUCTION
# =====================================================


def build_plugin(cfg: Dict[str, Any]) -> Optional[Any]:
    """
    Instantiates the analytical domain plugin specified in the dataset configuration.

    Parameters
    ----------
    cfg : Dict[str, Any]
        Domain configuration parameters.

    Returns
    -------
    Optional[Any]
        Initialized plugin instance, or None if not configured.
    """
    plugin_name = cfg.get("plugin")
    if not plugin_name:
        return None

    plugin_class = PLUGIN_REGISTRY.get(plugin_name)
    if plugin_class is not None:
        return plugin_class(var_prefix=cfg.get("var_prefix", "x_"))

    return None


def build_dataset(df: pd.DataFrame, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Assembles the complete dataset context dictionary including selected objective metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Loaded solution space DataFrame.
    cfg : Dict[str, Any]
        Domain dataset configuration metadata.

    Returns
    -------
    Dict[str, Any]
        Global dataset context dictionary.
    """
    plugin = build_plugin(cfg)

    all_metrics = cfg.get("metrics", [])
    if not all_metrics:
        all_metrics = infer_numeric_metrics(df, cfg)

    selected_metrics = st.multiselect(
        "Objective Columns",
        all_metrics,
        default=all_metrics,
    )

    decision_variables = detect_decision_variables(
        df, cfg.get("var_prefix", "x_")
    )

    return {
        "df": df,
        "config": cfg,
        "plugin": plugin,
        "metrics": selected_metrics,
        "selected_indicators": [],
        "decision_variables": decision_variables,
    }


# =====================================================
# DATA LOADERS
# =====================================================


def load_builtin_dataset(cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Loads a predefined domain solution dataset from CSV storage.

    Parameters
    ----------
    cfg : Dict[str, Any]
        Domain configuration object containing `path_sol`.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame containing an assigned 'id' column.
    """
    path = cfg["path_sol"]
    df = pd.read_csv(path).reset_index(drop=True)

    if "id" not in df.columns:
        df["id"] = range(1, len(df) + 1)

    return df


def load_uploaded_dataset(uploaded_file: Any) -> pd.DataFrame:
    """
    Loads a user-uploaded CSV file into a solution space DataFrame.

    Parameters
    ----------
    uploaded_file : Any
        Streamlit UploadedFile object.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame containing an assigned 'id' column.
    """
    df = pd.read_csv(uploaded_file).reset_index(drop=True)

    if "id" not in df.columns:
        df["id"] = range(1, len(df) + 1)

    return df


# =====================================================
# UI RENDERING & DATA SOURCE INPUT
# =====================================================


def render_domain_configuration_input() -> Optional[Dict[str, Any]]:
    """
    Renders UI controls for selecting built-in domain configurations.

    Returns
    -------
    Optional[Dict[str, Any]]
        Assembled dataset context dictionary, or None if unselected/failed.
    """
    dataset_names = ["-- No Data --"] + list(CASES.keys())

    col_dataset, col_help = st.columns(
        [0.85, 0.15], vertical_alignment="bottom"
    )

    with col_dataset:
        dataset_name = st.selectbox(
            "Domain Configuration",
            dataset_names,
            key="input_domain_configuration",
        )

    if dataset_name == "-- No Data --":
        dataset_help = (
            "No domain configuration selected yet.\n\n"
            "Choose a predefined case to load its dataset, objectives, "
            "decision-variable prefix, and optional plugin."
        )
    else:
        dataset_help = CASES[dataset_name].get(
            "help",
            "No additional description is available for this domain configuration.",
        )

    with col_help:
        render_help_icon(dataset_help, key="help_domain_configuration")

    if dataset_name == "-- No Data --":
        st.info("Select data to continue.")
        return None

    cfg = CASES[dataset_name]

    try:
        df = load_builtin_dataset(cfg)
    except Exception as exc:
        st.error(f"Unable to load dataset: {cfg.get('path_sol')}")
        st.exception(exc)
        return None

    return build_dataset(df, cfg)


def render_uploaded_csv_input() -> Optional[Dict[str, Any]]:
    """
    Renders UI controls for custom CSV file uploading.

    Returns
    -------
    Optional[Dict[str, Any]]
        Assembled dataset context dictionary, or None if unselected/failed.
    """
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is None:
        return None

    var_prefix = st.text_input("Decision-variable prefix", value="var_")

    try:
        df = load_uploaded_dataset(uploaded_file)
    except Exception as exc:
        st.error("Unable to load uploaded CSV.")
        st.exception(exc)
        return None

    cfg: Dict[str, Any] = {
        "plugin": None,
        "metrics": [],
        "var_prefix": var_prefix,
        "exclude_cols": [],
        "default_indicators": [],
        "help": "Uploaded enriched CSV.",
    }

    return build_dataset(df, cfg)


def render_input_panel() -> Optional[Dict[str, Any]]:
    """
    Renders the main Input and Preparation sidebar expander panel.

    Returns
    -------
    Optional[Dict[str, Any]]
        Selected and processed dataset context dictionary, or None if empty.
    """
    with st.sidebar.expander("🏷️ Input and Preparation", expanded=True):
        col_label, col_help = st.columns(
            [0.85, 0.15], vertical_alignment="center"
        )

        with col_label:
            st.markdown("**Data Source**")

        with col_help:
            render_phase_help_icon("input", key="help_input_phase")

        mode = st.radio(
            "Data Source",
            [
                "1. Domain Configuration",
                "2. Upload Enriched CSV",
            ],
            horizontal=True,
            label_visibility="collapsed",
        )

        if mode == "1. Domain Configuration":
            return render_domain_configuration_input()

        return render_uploaded_csv_input()

# --- ARCHIVO: lens_consensus.py ---

"""
Consensus Lens Module.

Aggregates multiple saved Sets of Interest (SOIs) into a unified consensus 
model using threshold-based voting logic, unions, majorities, or intersections.
"""

from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st


# =====================================================
# UI RENDERING
# =====================================================


def render_params(
    dataset: Dict[str, Any], working_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Renders Streamlit UI controls for selecting and combining saved SOIs.

    Parameters
    ----------
    dataset : Dict[str, Any]
        Global dataset configuration metadata.
    working_df : pd.DataFrame
        Current working solution space DataFrame.

    Returns
    -------
    Dict[str, Any]
        Dictionary of selected consensus methods, source SOIs, and thresholds.
    """
    params: Dict[str, Any] = {}
    saved_sois: List[Dict[str, Any]] = st.session_state.get("saved_sois", [])

    if len(saved_sois) < 2:
        st.info(
            "At least two saved SOIs are required to build a consensus SOI."
        )
        params["method"] = "Consensus Threshold"
        params["selected_sois"] = []
        params["threshold"] = 0.5
        return params

    soi_names = [soi["name"] for soi in saved_sois if "name" in soi]

    params["method"] = st.selectbox(
        "Consensus Method",
        [
            "Consensus Threshold",
            "Union",
            "Majority",
            "Intersection",
        ],
        key="consensus_method",
    )

    params["selected_sois"] = st.multiselect(
        "SOIs to Combine",
        soi_names,
        default=soi_names[: min(2, len(soi_names))],
        key="consensus_selected_sois",
    )

    n_selected = len(params["selected_sois"])

    if params["method"] == "Union":
        threshold = 1.0 / max(n_selected, 1)
        params["threshold"] = threshold
        st.caption(
            "Union keeps solutions supported by at least one selected SOI."
        )

    elif params["method"] == "Majority":
        params["threshold"] = 0.5
        st.caption(
            "Majority keeps solutions supported by at least half of the selected SOIs."
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
            key="consensus_threshold",
        )

        if params["threshold"] >= 0.75:
            st.caption("Mode: consensus core.")
        elif params["threshold"] >= 0.50:
            st.caption("Mode: consensus pool.")
        else:
            st.caption("Mode: broad exploratory pool.")

    st.caption(
        "This lens treats saved SOIs as expert opinions and combines them into one consensus SOI."
    )

    return params


# =====================================================
# HELPER FUNCTIONS
# =====================================================


def _get_selected_sois(selected_names: List[str]) -> List[Dict[str, Any]]:
    """
    Retrieves saved SOI dictionaries matching target selection names.

    Parameters
    ----------
    selected_names : List[str]
        List of target SOI names to fetch from session state.

    Returns
    -------
    List[Dict[str, Any]]
        Matching list of SOI data objects.
    """
    saved_sois: List[Dict[str, Any]] = st.session_state.get("saved_sois", [])
    return [soi for soi in saved_sois if soi.get("name") in selected_names]


def _build_support_table(selected_sois: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Computes solution support counts and consensus scores across selected SOIs.

    Parameters
    ----------
    selected_sois : List[Dict[str, Any]]
        List of selected SOI configuration dictionaries.

    Returns
    -------
    pd.DataFrame
        Support table mapping solution IDs to support count, score, and source list.
    """
    support: Dict[Any, int] = {}
    support_names: Dict[Any, List[str]] = {}

    for soi in selected_sois:
        soi_name = soi.get("name", "Unnamed SOI")
        unique_ids = set(soi.get("ids", []))

        for solution_id in unique_ids:
            support[solution_id] = support.get(solution_id, 0) + 1
            support_names.setdefault(solution_id, []).append(soi_name)

    rows = []
    n_sois = len(selected_sois)

    for solution_id, support_count in support.items():
        consensus_score = support_count / max(n_sois, 1)
        rows.append(
            {
                "id": solution_id,
                "consensus_support_count": support_count,
                "consensus_score": consensus_score,
                "consensus_supporting_sois": ", ".join(
                    sorted(support_names.get(solution_id, []))
                ),
            }
        )

    return pd.DataFrame(rows)


def _add_consensus_labels(
    result: pd.DataFrame, n_sois: int
) -> pd.DataFrame:
    """
    Appends consensus group base and group count label metadata to DataFrame.

    Parameters
    ----------
    result : pd.DataFrame
        DataFrame containing consensus support counts.
    n_sois : int
        Total number of evaluated source SOIs.

    Returns
    -------
    pd.DataFrame
        Enriched DataFrame with visual categorical label columns.
    """
    result["group_base"] = result["consensus_support_count"].apply(
        lambda count: f"Support = {int(count)}/{n_sois}"
    )

    group_sizes = result["group_base"].value_counts().to_dict()

    result["group_label"] = result["group_base"].apply(
        lambda grp: f"{grp} (n={group_sizes[grp]})"
    )

    return result


# =====================================================
# MAIN PIPELINE ENTRY POINT
# =====================================================


def apply(
    df: pd.DataFrame, params: Dict[str, Any], dataset: Dict[str, Any]
) -> pd.DataFrame:
    """
    Applies consensus voting filters across selected SOIs to retain valid solutions.

    Parameters
    ----------
    df : pd.DataFrame
        Input working solution space DataFrame.
    params : Dict[str, Any]
        Consensus algorithm configuration parameters.
    dataset : Dict[str, Any]
        Global context dataset metadata.

    Returns
    -------
    pd.DataFrame
        Filtered and metadata-enriched consensus DataFrame.
    """
    if df is None or df.empty:
        return df

    result = df.copy()
    selected_names = params.get("selected_sois", [])
    selected_sois = _get_selected_sois(selected_names)

    if len(selected_sois) < 2:
        result["consensus_warning"] = (
            "At least two SOIs are required for combination."
        )
        return result

    support_table = _build_support_table(selected_sois)

    if support_table.empty:
        result["consensus_warning"] = (
            "Selected SOIs do not contain any solution IDs."
        )
        return result

    threshold = params.get("threshold", 0.5)
    support_table = support_table[
        support_table["consensus_score"] >= threshold
    ].copy()

    if support_table.empty:
        empty_result = result.iloc[0:0].copy()
        empty_result["consensus_warning"] = (
            "No solutions satisfy the selected consensus threshold."
        )
        return empty_result

    result = result.merge(support_table, on="id", how="inner")
    n_sois = len(selected_sois)

    result = _add_consensus_labels(result, n_sois)
    result["consensus_method"] = params.get("method", "Consensus Threshold")
    result["consensus_threshold"] = threshold
    result["consensus_source_sois"] = ", ".join(selected_names)

    result = result.sort_values(
        ["consensus_score", "consensus_support_count", "id"],
        ascending=[False, False, True],
    ).copy()

    result["consensus_rank"] = range(1, len(result) + 1)

    return result


# =====================================================
# FEEDBACK UI
# =====================================================


def _safe_first_value(df: pd.DataFrame, column: str) -> Optional[Any]:
    """Extracts first valid scalar value from a target DataFrame column."""
    if column not in df.columns:
        return None
    values = df[column].dropna()
    if values.empty:
        return None
    return values.iloc[0]


def render_feedback(lens_df: Optional[pd.DataFrame]) -> None:
    """
    Displays UI summary indicators when the consensus lens is active.

    Parameters
    ----------
    lens_df : Optional[pd.DataFrame]
        Output DataFrame containing consensus evaluation metadata.
    """
    if lens_df is None:
        st.warning("No consensus result is available.")
        return

    warning_value = _safe_first_value(lens_df, "consensus_warning")
    if warning_value is not None:
        st.warning(warning_value)
        return

    if lens_df.empty:
        st.warning("The consensus SOI is empty.")
        return

    method = _safe_first_value(lens_df, "consensus_method")
    if method is not None:
        st.info(f"Consensus method: {method}")

    threshold = _safe_first_value(lens_df, "consensus_threshold")
    if threshold is not None:
        st.caption(f"Consensus threshold: {float(threshold):.2f}")

    if "consensus_score" in lens_df.columns:
        max_score = lens_df["consensus_score"].max()
        st.caption(f"Maximum consensus score: {float(max_score):.2f}")

    st.caption(f"Consensus SOI size: {len(lens_df)} solutions")

# --- ARCHIVO: lens_diversity.py ---

"""
Diversity Lens Module.

Structures candidate solution sets into clusters using distance-based or
density-based unsupervised learning algorithms (K-Medoids, K-Means,
Agglomerative Hierarchical Clustering, or HDBSCAN).
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Optional dependencies for enhanced clustering algorithms
try:
    from sklearn_extra.cluster import KMedoids
except ImportError:
    KMedoids = None

try:
    from sklearn.cluster import HDBSCAN
except ImportError:
    HDBSCAN = None

logger = logging.getLogger(__name__)


# =====================================================
# UI RENDERING
# =====================================================


def render_params(
    dataset: Dict[str, Any], working_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Renders Streamlit UI controls for diversity clustering algorithms.

    Parameters
    ----------
    dataset : Dict[str, Any]
        Global dataset configuration containing metric and indicator keys.
    working_df : pd.DataFrame
        Current working solution space DataFrame.

    Returns
    -------
    Dict[str, Any]
        Dictionary of selected clustering parameters.
    """
    dimensions = dataset.get("metrics", []) + dataset.get(
        "selected_indicators", []
    )
    params: Dict[str, Any] = {}
    max_n = max(len(working_df), 1)

    if len(dimensions) < 2:
        st.info("At least two dimensions are required for clustering.")
        params["method"] = "K-Medoids"
        params["cluster_metrics"] = []
        return params

    params["method"] = st.selectbox(
        "Clustering Method",
        ["K-Medoids", "K-Means", "Agglomerative", "HDBSCAN"],
        key="div_method",
    )

    default_cluster_metrics = dimensions[: min(2, len(dimensions))]

    params["cluster_metrics"] = st.multiselect(
        "Metrics for Clustering",
        dimensions,
        default=default_cluster_metrics,
        key="div_cluster_metrics",
    )

    if params["method"] in ["K-Medoids", "K-Means"]:
        params["k_mode"] = st.radio(
            "Number of Groups",
            ["Auto", "Manual"],
            horizontal=True,
            key="div_k_mode",
        )

        if params["k_mode"] == "Manual":
            max_k = max(2, min(10, max_n))
            default_k = min(3, max_k)
            params["k"] = st.slider(
                "k Groups", 2, max_k, default_k, key="div_k"
            )
        else:
            st.caption(
                "Auto mode selects k using silhouette score maximization."
            )

    elif params["method"] == "Agglomerative":
        params["agglomerative_mode"] = st.radio(
            "Hierarchy Cut Mode",
            ["Number of Groups", "Distance Cut"],
            horizontal=True,
            key="div_agglomerative_mode",
        )

        if params["agglomerative_mode"] == "Number of Groups":
            params["k_mode"] = st.radio(
                "Number of Groups",
                ["Auto", "Manual"],
                horizontal=True,
                key="div_agg_k_mode",
            )

            if params["k_mode"] == "Manual":
                max_k = max(2, min(10, max_n))
                default_k = min(3, max_k)
                params["k"] = st.slider(
                    "k Groups", 2, max_k, default_k, key="div_agg_k"
                )
            else:
                st.caption(
                    "Auto mode selects the dendrogram cut with the best silhouette score."
                )
        else:
            params["distance_threshold"] = st.slider(
                "Distance Threshold",
                0.10,
                10.00,
                2.00,
                0.10,
                key="div_agg_distance_threshold",
            )
            st.caption(
                "Distance Cut builds the hierarchy and cuts it at the selected distance threshold."
            )

    elif params["method"] == "HDBSCAN":
        params["cluster_size_mode"] = st.radio(
            "Cluster Size",
            ["Auto", "Manual"],
            horizontal=True,
            key="div_hdbscan_size_mode",
        )

        if params["cluster_size_mode"] == "Auto":
            params["granularity"] = st.selectbox(
                "Cluster Granularity",
                ["Small (~5%)", "Medium (~10%)", "Large (~20%)"],
                index=1,
                key="div_hdbscan_granularity",
            )
        else:
            default_min_size = max(2, int(0.10 * max_n))
            params["min_cluster_size"] = st.slider(
                "Minimum Cluster Size",
                2,
                max(2, max_n),
                default_min_size,
                key="div_hdbscan_min_cluster_size",
            )

        params["exclude_noise"] = st.checkbox(
            "Exclude noise solutions",
            value=True,
            key="div_hdbscan_exclude_noise",
        )
        st.caption(
            "If HDBSCAN returns mostly noise, reduce cluster size or disable noise exclusion."
        )

    st.caption(
        "Diversity structures candidate solutions into spatial clusters instead of preference ranking."
    )

    return params


# =====================================================
# HELPER FUNCTIONS
# =====================================================


def _valid_numeric_metrics(df: pd.DataFrame, metrics: List[str]) -> List[str]:
    """
    Filters metrics present in DataFrame that are strictly numeric.
    """
    return [
        m
        for m in metrics
        if m in df.columns and pd.api.types.is_numeric_dtype(df[m])
    ]


def _prepare_matrix(df: pd.DataFrame, metrics: List[str]) -> np.ndarray:
    """
    Imputes missing values and standardizes features using Z-score scaling.
    """
    x = df[metrics].copy()
    x = x.fillna(x.median(numeric_only=True)).fillna(0.0)
    scaler = StandardScaler()
    return scaler.fit_transform(x)


def _build_partition_model(
    method: str, k: int
) -> Union[KMeans, AgglomerativeClustering, Any]:
    """
    Instantiates specified partition clustering model instance.
    """
    if method == "K-Medoids":
        if KMedoids is not None:
            return KMedoids(n_clusters=k, method="pam", random_state=123)
        logger.warning(
            "scikit-learn-extra KMedoids not installed. Falling back to KMeans."
        )
        return KMeans(n_clusters=k, random_state=123, n_init=10)

    if method == "K-Means":
        return KMeans(n_clusters=k, random_state=123, n_init=10)

    if method == "Agglomerative":
        return AgglomerativeClustering(n_clusters=k)

    return KMeans(n_clusters=k, random_state=123, n_init=10)


def _compute_auto_k(
    x_scaled: np.ndarray, method: str, max_k: int = 10
) -> Tuple[int, Optional[float]]:
    """
    Determines optimal number of clusters k via silhouette score maximization.
    """
    n = len(x_scaled)
    if n < 3:
        return 1, None

    best_k = 2
    best_score = -1.0
    upper_k = min(max_k, n - 1)

    for k in range(2, upper_k + 1):
        try:
            model = _build_partition_model(method, k)
            labels = model.fit_predict(x_scaled)
            unique_labels = set(labels)

            if 1 < len(unique_labels) < n:
                score = silhouette_score(x_scaled, labels)
                if score > best_score:
                    best_score = score
                    best_k = k
        except Exception as err:
            logger.debug("Silhouette evaluation failed for k=%d: %s", k, err)

    return best_k, (best_score if best_score != -1.0 else None)


def _fit_partition_clustering(
    x_scaled: np.ndarray, method: str, k: int
) -> Tuple[np.ndarray, str]:
    """
    Fits partition clustering model and returns assigned cluster labels.
    """
    model = _build_partition_model(method, k)
    labels = model.fit_predict(x_scaled)

    method_used = (
        "K-Means fallback"
        if (method == "K-Medoids" and KMedoids is None)
        else method
    )
    return labels, method_used


def _fit_hdbscan(
    x_scaled: np.ndarray, min_cluster_size: int
) -> Tuple[np.ndarray, str]:
    """
    Fits HDBSCAN density model if available.
    """
    if HDBSCAN is None:
        logger.warning("HDBSCAN module not installed.")
        labels = np.zeros(len(x_scaled), dtype=int)
        return labels, "HDBSCAN unavailable"

    model = HDBSCAN(min_cluster_size=min_cluster_size)
    labels = model.fit_predict(x_scaled)
    return labels, "HDBSCAN"


def _fit_agglomerative_distance_cut(
    x_scaled: np.ndarray, distance_threshold: float
) -> Tuple[np.ndarray, str]:
    """
    Fits Agglomerative clustering cut at fixed distance threshold.
    """
    model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        compute_full_tree=True,
    )
    labels = model.fit_predict(x_scaled)
    return labels, "Agglomerative distance cut"


def _compute_silhouette_if_valid(
    x_scaled: np.ndarray, labels: np.ndarray
) -> Optional[float]:
    """
    Safely calculates silhouette score if valid number of clusters exist.
    """
    unique_labels = set(labels)
    n = len(labels)

    if len(unique_labels) <= 1 or len(unique_labels) >= n:
        return None

    try:
        return float(silhouette_score(x_scaled, labels))
    except Exception:
        return None


def _add_cluster_labels(
    result: pd.DataFrame,
    labels: np.ndarray,
    method_used: str,
    metrics_used: List[str],
) -> pd.DataFrame:
    """
    Attaches cluster IDs, labels, sizes, and metadata to output DataFrame.
    """
    res = result.copy()
    res["cluster"] = labels
    res["cluster_str"] = res["cluster"].astype(str).replace("-1", "Noise")

    cluster_sizes = res.groupby("cluster_str")["id"].transform("size")
    res["group_label"] = (
        "Cluster " + res["cluster_str"] + " (n=" + cluster_sizes.astype(str) + ")"
    )

    n_clusters = (
        res["cluster"]
        .dropna()
        .astype(int)
        .loc[lambda v: v != -1]
        .nunique()
    )
    noise_count = int(res["cluster"].eq(-1).sum())

    res["diversity_method"] = method_used
    res["diversity_metrics"] = ", ".join(metrics_used)
    res["diversity_n_clusters"] = n_clusters
    res["diversity_noise_count"] = noise_count

    return res


# =====================================================
# MAIN PIPELINE ENTRY POINT
# =====================================================


def apply(
    df: pd.DataFrame, params: Dict[str, Any], dataset: Dict[str, Any]
) -> pd.DataFrame:
    """
    Applies selected diversity lens method to structure DataFrame into clusters.

    Parameters
    ----------
    df : pd.DataFrame
        Input working solution space DataFrame.
    params : Dict[str, Any]
        Clustering configuration parameters.
    dataset : Dict[str, Any]
        Global context dataset metadata.

    Returns
    -------
    pd.DataFrame
        Enriched DataFrame with cluster assignments and metadata.
    """
    if df is None or df.empty or len(df) < 2:
        return df

    result = df.copy()
    dimensions = dataset.get("metrics", []) + dataset.get(
        "selected_indicators", []
    )
    method = params.get("method", "K-Medoids")
    cluster_metrics = params.get("cluster_metrics", dimensions)
    cluster_metrics = _valid_numeric_metrics(result, cluster_metrics)

    if len(cluster_metrics) < 2:
        return result

    x_scaled = _prepare_matrix(result, cluster_metrics)

    # ----------------------------------------------------
    # K-Medoids / K-Means
    # ----------------------------------------------------
    if method in ["K-Medoids", "K-Means"]:
        k_mode = params.get("k_mode", "Auto")

        if k_mode == "Manual":
            k = max(2, min(params.get("k", 2), len(result)))
            silhouette = None
        else:
            k, silhouette = _compute_auto_k(x_scaled, method)
            if k < 2:
                return result

        labels, method_used = _fit_partition_clustering(x_scaled, method, k)
        result = _add_cluster_labels(
            result, labels, method_used, cluster_metrics
        )
        result["diversity_k"] = k
        if silhouette is not None:
            result["diversity_silhouette"] = silhouette

        return result

    # ----------------------------------------------------
    # Agglomerative
    # ----------------------------------------------------
    if method == "Agglomerative":
        agglomerative_mode = params.get(
            "agglomerative_mode", "Number of Groups"
        )

        if agglomerative_mode == "Distance Cut":
            dist_thresh = params.get("distance_threshold", 2.0)
            labels, method_used = _fit_agglomerative_distance_cut(
                x_scaled, dist_thresh
            )
            result = _add_cluster_labels(
                result, labels, method_used, cluster_metrics
            )
            result["diversity_distance_threshold"] = dist_thresh

            silhouette = _compute_silhouette_if_valid(x_scaled, labels)
            if silhouette is not None:
                result["diversity_silhouette"] = silhouette

            return result

        k_mode = params.get("k_mode", "Auto")
        if k_mode == "Manual":
            k = max(2, min(params.get("k", 2), len(result)))
            silhouette = None
        else:
            k, silhouette = _compute_auto_k(x_scaled, method)
            if k < 2:
                return result

        labels, method_used = _fit_partition_clustering(x_scaled, method, k)
        result = _add_cluster_labels(
            result, labels, method_used, cluster_metrics
        )
        result["diversity_k"] = k
        if silhouette is not None:
            result["diversity_silhouette"] = silhouette

        return result

    # ----------------------------------------------------
    # HDBSCAN
    # ----------------------------------------------------
    if method == "HDBSCAN":
        n = len(result)
        size_mode = params.get("cluster_size_mode", "Auto")

        if size_mode == "Manual":
            min_cluster_size = params.get(
                "min_cluster_size", max(2, int(0.1 * n))
            )
        else:
            granularity = params.get("granularity", "Medium (~10%)")
            if granularity == "Small (~5%)":
                min_cluster_size = max(2, int(0.05 * n))
            elif granularity == "Large (~20%)":
                min_cluster_size = max(2, int(0.20 * n))
            else:
                min_cluster_size = max(2, int(0.10 * n))

        labels, method_used = _fit_hdbscan(x_scaled, min_cluster_size)
        result = _add_cluster_labels(
            result, labels, method_used, cluster_metrics
        )
        result["diversity_min_cluster_size"] = min_cluster_size

        if params.get("exclude_noise", True):
            filtered = result[result["cluster"] != -1].copy()
            if filtered.empty:
                result["diversity_warning"] = (
                    "All solutions were classified as noise. "
                    "Noise exclusion was not applied."
                )
                return result
            return filtered

        return result

    return result


# =====================================================
# FEEDBACK UI
# =====================================================


def _safe_first_value(df: pd.DataFrame, column: str) -> Any:
    """
    Extracts first non-null value from given DataFrame column if present.
    """
    if column not in df.columns:
        return None
    values = df[column].dropna()
    return values.iloc[0] if not values.empty else None


def render_feedback(lens_df: pd.DataFrame) -> None:
    """
    Displays UI summary metrics and feedback for applied clustering lens.
    """
    if lens_df is None:
        st.warning("No clustering result is available.")
        return

    if lens_df.empty:
        st.warning(
            "The clustering lens returned an empty subset. "
            "Try reducing HDBSCAN minimum cluster size or disabling noise exclusion."
        )
        return

    warning_value = _safe_first_value(lens_df, "diversity_warning")
    if warning_value is not None:
        st.warning(warning_value)

    n_clusters = _safe_first_value(lens_df, "diversity_n_clusters")
    if n_clusters is not None:
        st.info(f"Clusters detected: **{int(n_clusters)}**")

    k_value = _safe_first_value(lens_df, "diversity_k")
    if k_value is not None:
        st.caption(f"Selected k: **{int(k_value)}**")

    silhouette_val = _safe_first_value(lens_df, "diversity_silhouette")
    if silhouette_val is not None:
        st.caption(f"Silhouette score: **{silhouette_val:.3f}**")

    min_cluster_size = _safe_first_value(lens_df, "diversity_min_cluster_size")
    if min_cluster_size is not None:
        st.caption(f"Minimum cluster size: **{int(min_cluster_size)}**")

    dist_thresh = _safe_first_value(lens_df, "diversity_distance_threshold")
    if dist_thresh is not None:
        st.caption(f"Distance threshold: **{float(dist_thresh):.2f}**")

    noise_count = _safe_first_value(lens_df, "diversity_noise_count")
    if noise_count is not None and int(noise_count) > 0:
        st.caption(f"Noise solutions detected: **{int(noise_count)}**")

# --- ARCHIVO: lens_efficiency.py ---

"""
Efficiency Lens Module.

Ranks candidate solutions based on benefit-cost trade-offs using raw ratios,
min-max normalized efficiency, composite cost aggregation, or Euclidean distance
to ideal target states in objective space.
"""

from typing import Any, Dict, List, Optional, Union

import pandas as pd
import streamlit as st

# Global small constant to prevent division by zero
EPS: float = 1e-9


# =====================================================
# UI RENDERING
# =====================================================


def render_params(
    dataset: Dict[str, Any], working_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Renders Streamlit UI controls for efficiency ranking parameters.

    Parameters
    ----------
    dataset : Dict[str, Any]
        Global dataset context containing metric and indicator keys.
    working_df : pd.DataFrame
        Current working solution space DataFrame.

    Returns
    -------
    Dict[str, Any]
        Dictionary of user-selected efficiency parameters.
    """
    dimensions = dataset.get("metrics", []) + dataset.get(
        "selected_indicators", []
    )
    params: Dict[str, Any] = {}
    max_n = max(len(working_df), 1)
    default_n = min(5, max_n)

    if len(dimensions) < 2:
        st.info(
            "At least two dimensions are required for the Efficiency lens."
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
            "Composite Cost Ratio",
        ],
        key="eff_method",
    )

    params["benefit"] = st.selectbox(
        "Benefit Metric", dimensions, key="eff_benefit"
    )

    cost_options = [d for d in dimensions if d != params["benefit"]]

    if params["method"] == "Composite Cost Ratio":
        params["cost"] = st.multiselect(
            "Cost Metrics",
            cost_options,
            default=cost_options[: min(2, len(cost_options))],
            key="eff_costs",
        )
    else:
        params["cost"] = st.selectbox(
            "Cost Metric", cost_options, key="eff_cost"
        )

    params["top_n"] = st.slider(
        "Top N Solutions", 1, max_n, default_n, key="eff_top_n"
    )

    st.caption(
        "Efficiency methods rank solutions by benefit-cost trade-off."
    )

    return params


# =====================================================
# HELPER FUNCTIONS
# =====================================================


def _normalize_series(series: pd.Series) -> pd.Series:
    """
    Normalizes a numeric pandas Series to the range [0.0, 1.0] via Min-Max scaling.

    Parameters
    ----------
    series : pd.Series
        Numeric input series to normalize.

    Returns
    -------
    pd.Series
        Min-Max normalized series, or zeros if min equals max.
    """
    min_v = series.min()
    max_v = series.max()

    if max_v > min_v:
        return (series - min_v) / (max_v - min_v)

    return pd.Series(0.0, index=series.index)


def _resolve_cost_metrics(
    result: pd.DataFrame,
    benefit: str,
    cost: Optional[Union[str, List[str]]],
) -> List[str]:
    """
    Resolves and validates cost metric column names present in the DataFrame.

    Parameters
    ----------
    result : pd.DataFrame
        Input working solution space DataFrame.
    benefit : str
        Selected benefit metric name.
    cost : Optional[Union[str, List[str]]]
        Single cost metric name or list of cost metric names.

    Returns
    -------
    List[str]
        Filtered list of valid cost column names excluding the benefit metric.
    """
    if cost is None:
        return []

    if isinstance(cost, str):
        cost_metrics = [cost]
    else:
        cost_metrics = [c for c in cost if c in result.columns]

    return [c for c in cost_metrics if c != benefit]


# =====================================================
# SCORE METHOD ENGINES
# =====================================================


def _benefit_cost_ratio(
    result: pd.DataFrame, benefit: str, cost_metrics: List[str]
) -> pd.Series:
    """Calculates unnormalized Benefit / Cost ratio."""
    cost_metric = cost_metrics[0]
    safe_cost = result[cost_metric].replace(0, EPS)
    return result[benefit] / safe_cost


def _normalized_ratio(
    result: pd.DataFrame, benefit: str, cost_metrics: List[str]
) -> pd.Series:
    """Calculates Min-Max normalized Benefit / Normalized Cost ratio."""
    cost_metric = cost_metrics[0]
    benefit_norm = _normalize_series(result[benefit])
    cost_norm = _normalize_series(result[cost_metric])
    return benefit_norm / (cost_norm + EPS)


def _distance_to_ideal(
    result: pd.DataFrame, benefit: str, cost_metrics: List[str]
) -> pd.Series:
    """
    Calculates proximity score based on Euclidean distance to ideal state (1.0 benefit, 0.0 cost).
    """
    cost_metric = cost_metrics[0]
    benefit_norm = _normalize_series(result[benefit])
    cost_norm = _normalize_series(result[cost_metric])

    distance_to_ideal = (
        (1.0 - benefit_norm) ** 2 + (cost_norm) ** 2
    ) ** 0.5
    max_distance = 2.0**0.5

    return 1.0 - (distance_to_ideal / max_distance)


def _composite_cost_ratio(
    result: pd.DataFrame, benefit: str, cost_metrics: List[str]
) -> pd.Series:
    """Calculates normalized Benefit / Average Composite Normalized Costs ratio."""
    benefit_norm = _normalize_series(result[benefit])
    composite_cost = pd.Series(0.0, index=result.index)

    for cost_metric in cost_metrics:
        composite_cost += _normalize_series(result[cost_metric])

    composite_cost /= len(cost_metrics)
    return benefit_norm / (composite_cost + EPS)


# =====================================================
# MAIN PIPELINE ENTRY POINT
# =====================================================


def apply(
    df: pd.DataFrame, params: Dict[str, Any], dataset: Dict[str, Any]
) -> pd.DataFrame:
    """
    Applies selected efficiency lens method to calculate score and rank solutions.

    Parameters
    ----------
    df : pd.DataFrame
        Input solution space DataFrame.
    params : Dict[str, Any]
        Efficiency configuration parameters.
    dataset : Dict[str, Any]
        Global context dataset metadata.

    Returns
    -------
    pd.DataFrame
        Ranked top N subset of solutions enriched with efficiency scores.
    """
    if df is None or df.empty:
        return df

    result = df.copy()
    method = params.get("method", "Benefit/Cost Ratio")
    benefit = params.get("benefit")
    cost = params.get("cost")

    if benefit is None or benefit not in result.columns:
        return result

    cost_metrics = _resolve_cost_metrics(result, benefit, cost)
    if not cost_metrics:
        return result

    top_n = min(params.get("top_n", len(result)), len(result))

    if method == "Benefit/Cost Ratio":
        score = _benefit_cost_ratio(result, benefit, cost_metrics)
    elif method == "Normalized Ratio":
        score = _normalized_ratio(result, benefit, cost_metrics)
    elif method == "Distance to Ideal":
        score = _distance_to_ideal(result, benefit, cost_metrics)
    elif method == "Composite Cost Ratio":
        score = _composite_cost_ratio(result, benefit, cost_metrics)
        result["efficiency_costs"] = ", ".join(cost_metrics)
    else:
        return result

    result["efficiency_score"] = score
    result = result.sort_values(
        "efficiency_score", ascending=False
    ).copy()

    result["efficiency_rank"] = range(1, len(result) + 1)
    result["efficiency_method"] = method
    result["efficiency_benefit"] = benefit
    result["efficiency_primary_cost"] = cost_metrics[0]

    return result.head(top_n)


# =====================================================
# FEEDBACK UI
# =====================================================


def render_feedback(lens_df: pd.DataFrame) -> None:
    """
    Displays UI summary metadata and indicators for applied efficiency ranking.

    Parameters
    ----------
    lens_df : pd.DataFrame
        Filtered/ranked output DataFrame containing efficiency metadata.
    """
    if lens_df is None or lens_df.empty:
        st.warning("No efficiency results available.")
        return

    if "efficiency_method" in lens_df.columns:
        method = lens_df["efficiency_method"].dropna().iloc[0]
        st.info(f"Efficiency method: {method}")

    if "efficiency_benefit" in lens_df.columns:
        benefit = lens_df["efficiency_benefit"].dropna().iloc[0]
        st.caption(f"Benefit metric: {benefit}")

    if "efficiency_costs" in lens_df.columns:
        costs = lens_df["efficiency_costs"].dropna().iloc[0]
        st.caption(f"Composite costs: {costs}")
    elif "efficiency_primary_cost" in lens_df.columns:
        cost = lens_df["efficiency_primary_cost"].dropna().iloc[0]
        st.caption(f"Cost metric: {cost}")

# --- ARCHIVO: lens_engine.py ---

"""
Lens Engine Module.

Provides execution and orchestration services for dynamic analytical lenses
in multi-objective decision space exploration frameworks.
"""

import logging
from typing import Any, Dict, Optional
import pandas as pd

from lenses.lens_registry import get_lens_module

logger = logging.getLogger(__name__)


def apply_lens(
    df: Optional[pd.DataFrame],
    lens_name: str,
    params: Dict[str, Any],
    dataset: Dict[str, Any],
) -> Optional[pd.DataFrame]:
    """
    Applies a selected analytical lens to a decision space DataFrame.

    Parameters
    ----------
    df : Optional[pd.DataFrame]
        The input dataset representing candidate solutions in the working set.
    lens_name : str
        Identifier of the lens to apply (e.g., 'ParetoFilter', 'KneePoint').
        Passing "None" or an empty string returns an unmodified copy of `df`.
    params : Dict[str, Any]
        User-defined parameters required by the specific lens implementation.
    dataset : Dict[str, Any]
        Global dataset context containing domain metadata and configurations.

    Returns
    -------
    Optional[pd.DataFrame]
        Transformed DataFrame after applying the lens, or a copy of the input
        DataFrame if no transformation is applied or if execution fails safely.
    """
    if df is None or df.empty:
        return df

    if lens_name == "None" or not lens_name:
        return df.copy()

    lens_module = get_lens_module(lens_name)

    if lens_module is None:
        logger.warning(
            f"Lens '{lens_name}' not found in registry. Returning original DataFrame."
        )
        return df.copy()

    try:
        # 1. Functional approach: Module implements .apply(df, params, dataset)
        if hasattr(lens_module, "apply") and callable(lens_module.apply):
            return lens_module.apply(df, params, dataset)

        # 2. Object-Oriented approach: Class instance with .transform(df)
        elif hasattr(lens_module, "transform") and callable(
            lens_module.transform
        ):
            return lens_module.transform(df)

        else:
            raise AttributeError(
                f"Lens '{lens_name}' does not implement a valid 'apply()' or 'transform()' interface."
            )

    except Exception as e:
        logger.error(f"Error executing lens '{lens_name}': {str(e)}")
        # Safe fallback: prevent GUI/Pipeline crash by returning input data copy
        return df.copy()

# --- ARCHIVO: lens_feedback.py ---

"""
Lens Feedback Component.

Renders analytical feedback, summary statistics, and visual indicators
generated by active lenses into designated Streamlit container placeholders.
"""

import logging
from typing import Optional
import pandas as pd
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from lenses.lens_registry import get_lens_module

logger = logging.getLogger(__name__)


def render_lens_feedback(
    placeholder: Optional[DeltaGenerator],
    active_lens: str,
    lens_df: Optional[pd.DataFrame],
) -> None:
    """
    Renders analytical feedback provided by the active lens into a UI placeholder.

    Parameters
    ----------
    placeholder : Optional[DeltaGenerator]
        Streamlit container placeholder where feedback UI elements will be attached.
    active_lens : str
        Identifier of the active analytical lens (e.g., 'Preference', 'Diversity').
    lens_df : Optional[pd.DataFrame]
        DataFrame resulting from the lens transformation.
    """
    if placeholder is None or lens_df is None or lens_df.empty:
        return

    if active_lens == "None" or not active_lens:
        return

    lens_module = get_lens_module(active_lens)

    if lens_module is None:
        return

    if hasattr(lens_module, "render_feedback") and callable(
        lens_module.render_feedback
    ):
        try:
            with placeholder.container():
                lens_module.render_feedback(lens_df)
        except Exception as e:
            logger.error(
                f"Error rendering feedback for lens '{active_lens}': {str(e)}"
            )
            placeholder.error(
                f"Failed to display feedback for '{active_lens}': {str(e)}"
            )

# --- ARCHIVO: lens_indicator.py ---

"""
Indicator Lens Module.

Provides multi-criteria selection methods based on domain indicators:
1. Top-N Matches: Aggregates top solutions across individual target dimensions.
2. Non-Dominated Sorting: Identifies Pareto-optimal solutions within the enriched 
   indicator space.
"""

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st


# =====================================================
# UI RENDERING
# =====================================================


def render_params(
    dataset: Dict[str, Any], working_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Renders Streamlit UI controls for indicator lens options.

    Parameters
    ----------
    dataset : Dict[str, Any]
        Global dataset configuration metadata containing metrics and indicators.
    working_df : pd.DataFrame
        Current working solution space DataFrame.

    Returns
    -------
    Dict[str, Any]
        Dictionary of user-selected criteria and algorithm parameters.
    """
    dimensions = dataset.get("metrics", []) + dataset.get(
        "selected_indicators", []
    )
    indicators = dataset.get("selected_indicators", [])

    params: Dict[str, Any] = {}
    max_n = max(len(working_df) if working_df is not None else 0, 1)
    default_n = min(5, max_n)

    if not dimensions:
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
        ["Top-N Matches", "Non-dominated"],
        key="indicator_method",
    )

    if params["method"] == "Top-N Matches":
        available_criteria = dimensions
        st.caption(
            "Top-N Matches can use both original objectives and enriched indicators."
        )
    else:
        available_criteria = indicators
        if not available_criteria:
            st.info(
                "Non-dominated analysis currently uses enriched indicators. "
                "Enable indicators in Data Enrichment first."
            )
            params["maximize"] = []
            params["minimize"] = []
            params["top_n"] = None
            return params

        st.caption("Non-dominated analysis uses enriched indicators.")

    params["maximize"] = st.multiselect(
        "Dimensions to Maximize", available_criteria, key="indicator_maximize"
    )

    minimize_options = [
        c for c in available_criteria if c not in params["maximize"]
    ]

    params["minimize"] = st.multiselect(
        "Dimensions to Minimize", minimize_options, key="indicator_minimize"
    )

    if params["method"] == "Top-N Matches":
        params["top_n"] = st.slider(
            "Top N per Dimension", 1, max_n, default_n, key="indicator_top_n"
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
# HELPER FUNCTIONS
# =====================================================


def _sanitize_criteria(
    df: pd.DataFrame, maximize: List[str], minimize: List[str]
) -> Tuple[List[str], List[str], List[str]]:
    """
    Validates criteria column existence within the input DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Target DataFrame to sanitize criteria against.
    maximize : List[str]
        List of target metric names to maximize.
    minimize : List[str]
        List of target metric names to minimize.

    Returns
    -------
    Tuple[List[str], List[str], List[str]]
        Sanitized maximize, minimize, and combined criteria lists.
    """
    valid_max = [m for m in maximize if m in df.columns]
    valid_min = [
        m for m in minimize if m in df.columns and m not in valid_max
    ]
    criteria = valid_max + valid_min
    return valid_max, valid_min, criteria


def _build_group_labels_from_count(
    result: pd.DataFrame, count_column: str
) -> pd.DataFrame:
    """
    Generates categorical grouping labels based on indicator match counts.

    Parameters
    ----------
    result : pd.DataFrame
        DataFrame containing match counts.
    count_column : str
        Target column containing numerical match count.

    Returns
    -------
    pd.DataFrame
        Enriched DataFrame with `group_base` and `group_label` columns.
    """
    result["group_base"] = result[count_column].apply(
        lambda count: f"Matches = {count}"
    )
    group_sizes = result["group_base"].value_counts().to_dict()

    result["group_label"] = result["group_base"].apply(
        lambda grp: f"{grp} (n={group_sizes[grp]})"
    )
    return result


# =====================================================
# METHOD ENGINES
# =====================================================


def _apply_top_n_matches(
    df: pd.DataFrame, maximize: List[str], minimize: List[str], top_n: int
) -> pd.DataFrame:
    """Computes Top-N match counts per solution across selected criteria."""
    result = df.copy()
    criteria = maximize + minimize

    if not criteria:
        return result

    effective_top_n = min(top_n, len(result))
    ranked_subsets: List[pd.DataFrame] = []

    for metric in maximize:
        sub = (
            result.sort_values(metric, ascending=False)
            .head(effective_top_n)[["id"]]
            .assign(matched_metric=metric, goal="Maximize")
        )
        ranked_subsets.append(sub)

    for metric in minimize:
        sub = (
            result.sort_values(metric, ascending=True)
            .head(effective_top_n)[["id"]]
            .assign(matched_metric=metric, goal="Minimize")
        )
        ranked_subsets.append(sub)

    if not ranked_subsets:
        return result

    matches = pd.concat(ranked_subsets, ignore_index=True)

    counts = (
        matches.groupby("id")
        .size()
        .reset_index(name="domain_match_count")
    )

    matched_metrics = (
        matches.groupby("id")["matched_metric"]
        .apply(lambda vals: ", ".join(sorted(set(vals))))
        .reset_index(name="domain_matched_metrics")
    )

    result = result.merge(counts, on="id", how="left").merge(
        matched_metrics, on="id", how="left"
    )

    result["domain_match_count"] = (
        result["domain_match_count"].fillna(0).astype(int)
    )
    result["domain_matched_metrics"] = result[
        "domain_matched_metrics"
    ].fillna("")

    result = result[result["domain_match_count"] > 0].copy()

    if result.empty:
        return result

    result = _build_group_labels_from_count(result, "domain_match_count")
    result = result.sort_values(
        ["domain_match_count", "id"], ascending=[False, True]
    ).copy()

    result["domain_rank"] = range(1, len(result) + 1)
    result["indicator_method"] = "Top-N Matches"
    result["indicator_top_n"] = effective_top_n

    return result


def _apply_non_dominated(
    df: pd.DataFrame, maximize: List[str], minimize: List[str]
) -> pd.DataFrame:
    """Filters solutions to retain only Pareto non-dominated candidates."""
    result = df.copy()
    criteria = maximize + minimize

    if not criteria:
        return result

    work = result[criteria].copy()

    # Invert minimize metrics to convert problem strictly to maximization
    for metric in minimize:
        work[metric] = -work[metric]

    values = work.to_numpy()
    n_samples = len(values)
    is_nondominated = np.ones(n_samples, dtype=bool)

    # Pairwise non-dominance check
    for i in range(n_samples):
        current = values[i]
        for j in range(n_samples):
            if i == j:
                continue
            challenger = values[j]

            # Dominance test: challenger is >= in all and > in at least one
            if np.all(challenger >= current) and np.any(challenger > current):
                is_nondominated[i] = False
                break

    result["indicator_nondominated"] = is_nondominated
    result = result[result["indicator_nondominated"]].copy()

    if result.empty:
        return result

    result["indicator_method"] = "Non-dominated"
    result["domain_match_count"] = len(criteria)
    result["domain_matched_metrics"] = ", ".join(criteria)
    result["group_base"] = "Non-dominated"
    result["group_label"] = f"Non-dominated (n={len(result)})"

    result = result.sort_values("id", ascending=True).copy()
    result["domain_rank"] = range(1, len(result) + 1)

    return result


# =====================================================
# MAIN PIPELINE ENTRY POINT
# =====================================================


def apply(
    df: pd.DataFrame, params: Dict[str, Any], dataset: Dict[str, Any]
) -> pd.DataFrame:
    """
    Applies selected indicator lens method to isolate solutions.

    Parameters
    ----------
    df : pd.DataFrame
        Input working solution space DataFrame.
    params : Dict[str, Any]
        Indicator lens setup parameters.
    dataset : Dict[str, Any]
        Global context dataset metadata.

    Returns
    -------
    pd.DataFrame
        Filtered and metadata-enriched solution space DataFrame.
    """
    if df is None or df.empty:
        return df

    result = df.copy()
    maximize, minimize, criteria = _sanitize_criteria(
        result,
        params.get("maximize", []),
        params.get("minimize", []),
    )

    if not criteria:
        return result

    method = params.get("method", "Top-N Matches")

    if method == "Top-N Matches":
        top_n = params.get("top_n", min(5, len(result)))
        return _apply_top_n_matches(result, maximize, minimize, top_n)

    if method == "Non-dominated":
        return _apply_non_dominated(result, maximize, minimize)

    return result


# =====================================================
# FEEDBACK UI
# =====================================================


def render_feedback(lens_df: pd.DataFrame) -> None:
    """
    Displays UI summary metadata when indicator lens filtering is active.

    Parameters
    ----------
    lens_df : pd.DataFrame
        Filtered DataFrame output from the active indicator lens.
    """
    if lens_df is None or lens_df.empty:
        st.warning("No indicator matches found.")
        return

    if "indicator_method" in lens_df.columns:
        method = lens_df["indicator_method"].dropna().iloc[0]
        st.info(f"Indicator method: {method}")

    if "domain_match_count" in lens_df.columns:
        max_matches = lens_df["domain_match_count"].max()
        st.caption(f"Maximum indicator matches: {int(max_matches)}")

    if "domain_matched_metrics" in lens_df.columns:
        st.caption("Solutions are grouped by matched indicators.")

    if "indicator_nondominated" in lens_df.columns:
        st.caption(f"Non-dominated solutions: {len(lens_df)}")


# =====================================================
# BACKWARD COMPATIBILITY
# =====================================================


def apply_domain_lens(
    df: pd.DataFrame, maximize: List[str], minimize: List[str], top_n: int
) -> pd.DataFrame:
    """
    Legacy entry point for Top-N Domain match filtering.

    Parameters
    ----------
    df : pd.DataFrame
        Input solution space DataFrame.
    maximize : List[str]
        Metrics to maximize.
    minimize : List[str]
        Metrics to minimize.
    top_n : int
        Top N cut-off per metric.

    Returns
    -------
    pd.DataFrame
        Ranked and filtered DataFrame.
    """
    if df is None or df.empty:
        return df

    valid_max, valid_min, criteria = _sanitize_criteria(
        df, maximize, minimize
    )

    if not criteria:
        return df.copy()

    return _apply_top_n_matches(df, valid_max, valid_min, top_n)

# --- ARCHIVO: lens_manual.py ---

"""
Manual Selection Lens Module.

Enables manual filtering and isolation of specific candidate solutions from 
the active solution space using explicit identifier selection.
"""

from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st


# =====================================================
# UI RENDERING
# =====================================================


def render_params(
    dataset: Dict[str, Any], working_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Renders Streamlit UI controls for picking candidate solutions by ID.

    Parameters
    ----------
    dataset : Dict[str, Any]
        Global dataset configuration metadata.
    working_df : pd.DataFrame
        Current working solution space DataFrame.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing selected manual solution identifiers.
    """
    params: Dict[str, Any] = {"method": "Manual Selection"}

    if (
        working_df is None
        or working_df.empty
        or "id" not in working_df.columns
    ):
        st.warning("No solutions available for manual selection.")
        params["selected_ids"] = []
        return params

    valid_ids: List[int] = working_df["id"].dropna().astype(int).tolist()

    params["selected_ids"] = st.multiselect(
        "Pick solutions one by one",
        options=valid_ids,
        default=[],
        key="manual_lens_selected_ids",
        help="Manually pick the exact solutions you want to isolate.",
    )

    return params


# =====================================================
# MAIN PIPELINE ENTRY POINT
# =====================================================


def apply(
    df: pd.DataFrame, params: Dict[str, Any], dataset: Dict[str, Any]
) -> pd.DataFrame:
    """
    Filters the DataFrame to retain only manually selected solution IDs.

    Parameters
    ----------
    df : pd.DataFrame
        Input working solution space DataFrame.
    params : Dict[str, Any]
        Manual selection parameters containing target IDs.
    dataset : Dict[str, Any]
        Global context dataset metadata.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing only selected solution records.
    """
    if df is None or df.empty:
        return df

    selected_ids: List[int] = params.get("selected_ids", [])

    if not selected_ids:
        # Return an empty DataFrame with preserved schema if no selection is made
        return df.iloc[0:0].copy()

    return df[df["id"].isin(selected_ids)].copy()


# =====================================================
# FEEDBACK UI
# =====================================================


def render_feedback(lens_df: Optional[pd.DataFrame]) -> None:
    """
    Displays UI summary indicators when the manual selection lens is active.

    Parameters
    ----------
    lens_df : Optional[pd.DataFrame]
        Filtered output DataFrame containing active manual selection.
    """
    if lens_df is None:
        return

    count = len(lens_df)
    if count == 0:
        st.caption("No solutions selected in manual lens.")
    else:
        st.info(f"📌 Manual selection: {count} solution(s) active.")

# --- ARCHIVO: lens_preference.py ---

"""
Preference Lens Module.

Implements Multi-Criteria Decision Making (MCDM) ranking algorithms to score
and order Pareto-optimal solutions based on Decision Maker (DM) preferences.
Supported methods: Weighted Sum, TOPSIS, VIKOR, and Reference Point.
"""

import logging
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)


def render_params(
    dataset: Dict[str, Any], working_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Renders UI controls for MCDM method selection and metric optimization goals.

    Parameters
    ----------
    dataset : Dict[str, Any]
        Global dataset configuration containing metric metadata.
    working_df : pd.DataFrame
        Current working solution space DataFrame.

    Returns
    -------
    Dict[str, Any]
        Dictionary of selected preference configuration parameters.
    """
    dimensions = dataset.get("metrics", []) + dataset.get(
        "selected_indicators", []
    )
    max_n = max(len(working_df), 1)
    default_n = min(5, max_n)

    params: Dict[str, Any] = {}

    params["method"] = st.selectbox(
        "Scoring Method",
        ["Weighted Sum", "TOPSIS", "VIKOR", "Reference Point"],
        key="pref_method",
    )

    st.caption("All preference methods currently assign equal weight to selected criteria.")

    params["maximize"] = st.multiselect(
        "Metrics to Maximize", dimensions, key="pref_maximize"
    )

    minimize_options = [
        d for d in dimensions if d not in params.get("maximize", [])
    ]

    params["minimize"] = st.multiselect(
        "Metrics to Minimize", minimize_options, key="pref_minimize"
    )

    params["top_n"] = st.slider(
        "Top N Solutions", 1, max_n, default_n, key="pref_top_n"
    )

    return params


def _sanitize_criteria(
    df: pd.DataFrame, maximize: List[str], minimize: List[str]
) -> Tuple[List[str], List[str], List[str]]:
    """
    Validates and cleans user-selected criteria columns against DataFrame schema.
    """
    valid_max = [m for m in maximize if m in df.columns]
    valid_min = [m for m in minimize if m in df.columns and m not in valid_max]
    return valid_max, valid_min, valid_max + valid_min


def _minmax_normalize(df: pd.DataFrame, criteria: List[str]) -> pd.DataFrame:
    """
    Applies Min-Max normalization across selected evaluation criteria.
    """
    norm = pd.DataFrame(index=df.index)
    for metric in criteria:
        min_v = df[metric].min()
        max_v = df[metric].max()
        if max_v > min_v:
            norm[metric] = (df[metric] - min_v) / (max_v - min_v)
        else:
            norm[metric] = 0.0
    return norm


def _weighted_sum(
    df: pd.DataFrame, maximize: List[str], minimize: List[str]
) -> pd.Series:
    """
    Calculates score via Weighted Sum Model (WSM).
    """
    criteria = maximize + minimize
    norm = _minmax_normalize(df, criteria)
    score = pd.Series(0.0, index=df.index)
    weight = 1.0 / len(criteria)

    for metric in criteria:
        if metric in maximize:
            val = norm[metric]
        else:
            val = 1.0 - norm[metric]
        score += weight * val

    return score


def _topsis(
    df: pd.DataFrame, maximize: List[str], minimize: List[str]
) -> pd.Series:
    """
    Vectorized TOPSIS (Technique for Order Preference by Similarity to Ideal Solution).
    """
    criteria = maximize + minimize
    vals = df[criteria].to_numpy(dtype=float)

    # Vectorized L2 Normalization
    norms = np.linalg.norm(vals, axis=0)
    norms[norms == 0] = 1.0
    norm_vals = (vals / norms) * (1.0 / len(criteria))

    ideal = np.zeros(len(criteria))
    anti_ideal = np.zeros(len(criteria))

    for idx, metric in enumerate(criteria):
        if metric in maximize:
            ideal[idx] = norm_vals[:, idx].max()
            anti_ideal[idx] = norm_vals[:, idx].min()
        else:
            ideal[idx] = norm_vals[:, idx].min()
            anti_ideal[idx] = norm_vals[:, idx].max()

    # Vectorized Euclidean Distances
    d_plus = np.linalg.norm(norm_vals - ideal, axis=1)
    d_minus = np.linalg.norm(norm_vals - anti_ideal, axis=1)

    denom = d_plus + d_minus
    scores = np.where(denom != 0, d_minus / denom, 0.0)

    return pd.Series(scores, index=df.index)


def _vikor(
    df: pd.DataFrame, maximize: List[str], minimize: List[str], v: float = 0.5
) -> pd.Series:
    """
    Calculates VIKOR compromise ranking index (Q).
    """
    criteria = maximize + minimize
    weight = 1.0 / len(criteria)
    regret = pd.DataFrame(index=df.index)

    for metric in criteria:
        if metric in maximize:
            best, worst = df[metric].max(), df[metric].min()
        else:
            best, worst = df[metric].min(), df[metric].max()

        denom = abs(best - worst)
        if denom == 0:
            regret[metric] = 0.0
        else:
            regret[metric] = weight * abs(best - df[metric]) / denom

    s_value = regret.sum(axis=1)
    r_value = regret.max(axis=1)

    s_range = s_value.max() - s_value.min()
    s_norm = (s_value - s_value.min()) / s_range if s_range > 0 else 0.0

    r_range = r_value.max() - r_value.min()
    r_norm = (r_value - r_value.min()) / r_range if r_range > 0 else 0.0

    q_value = v * s_norm + (1.0 - v) * r_norm
    return 1.0 - q_value  # Higher score implies better rank


def _reference_point(
    df: pd.DataFrame, maximize: List[str], minimize: List[str]
) -> pd.Series:
    """
    Vectorized distance to ideal reference point (1.0 in normalized objective space).
    """
    criteria = maximize + minimize
    norm = _minmax_normalize(df, criteria)
    oriented = pd.DataFrame(index=df.index)

    for metric in criteria:
        if metric in maximize:
            oriented[metric] = norm[metric]
        else:
            oriented[metric] = 1.0 - norm[metric]

    # Vectorized Euclidean Distance to Ideal Point [1, 1, ..., 1]
    oriented_vals = oriented.to_numpy(dtype=float)
    distances = np.linalg.norm(1.0 - oriented_vals, axis=1)

    max_dist = distances.max()
    if max_dist > 0:
        scores = 1.0 - (distances / max_dist)
    else:
        scores = np.ones(len(df))

    return pd.Series(scores, index=df.index)


def apply(
    df: pd.DataFrame, params: Dict[str, Any], dataset: Dict[str, Any]
) -> pd.DataFrame:
    """
    Applies the specified preference lens and ranks solutions accordingly.

    Parameters
    ----------
    df : pd.DataFrame
        Input decision space candidate solutions.
    params : Dict[str, Any]
        Configuration mapping including optimization directions and method.
    dataset : Dict[str, Any]
        Global context dataset metadata.

    Returns
    -------
    pd.DataFrame
        Ranked and truncated DataFrame containing top N solutions.
    """
    if df is None or df.empty:
        return df

    result = df.copy()
    maximize, minimize, criteria = _sanitize_criteria(
        result, params.get("maximize", []), params.get("minimize", [])
    )

    if not criteria:
        return result

    method = params.get("method", "Weighted Sum")
    top_n = min(params.get("top_n", len(result)), len(result))

    if method == "Weighted Sum":
        score = _weighted_sum(result, maximize, minimize)
    elif method == "TOPSIS":
        score = _topsis(result, maximize, minimize)
    elif method == "VIKOR":
        score = _vikor(result, maximize, minimize)
    elif method == "Reference Point":
        score = _reference_point(result, maximize, minimize)
    else:
        return result

    result["preference_score"] = score
    result = result.sort_values("preference_score", ascending=False).copy()
    result["preference_rank"] = range(1, len(result) + 1)
    result["preference_method"] = method

    return result.head(top_n)


def render_feedback(lens_df: pd.DataFrame) -> None:
    """
    Displays UI feedback summarizing the applied preference scoring.
    """
    if lens_df is None or lens_df.empty:
        return

    if "preference_method" in lens_df.columns:
        method = lens_df["preference_method"].dropna().iloc[0]
        st.info(f"Preference method applied: **{method}**")

    if "preference_score" in lens_df.columns:
        st.caption("Solutions ranked and sorted by highest `preference_score`.")

# --- ARCHIVO: lens_registry.py ---

"""
Lens Registry Module.

Provides dynamic registration and lookup services for analytical lenses
within the Decision Space Explorer architecture.
"""

import logging
from typing import Any, Dict, List, Optional

# Core default analytical lenses
from lenses import (
    lens_consensus,
    lens_diversity,
    lens_efficiency,
    lens_indicator,
    lens_manual,
    lens_preference,
)

logger = logging.getLogger(__name__)

# Private dictionary storing name-to-module/class mappings
_LENS_REGISTRY: Dict[str, Any] = {
    "Manual Selection": lens_manual,
    "Preference": lens_preference,
    "Diversity": lens_diversity,
    "Efficiency": lens_efficiency,
    "Indicator Dominance": lens_indicator,
    "SOI Consensus": lens_consensus,
}


def register_lens(name: str, lens_module: Any, override: bool = False) -> None:
    """
    Dynamically registers a new analytical lens in the framework.

    Parameters
    ----------
    name : str
        The unique display identifier for the lens.
    lens_module : Any
        Module or class implementing the lens logic.
    override : bool, optional
        Whether to overwrite an existing lens entry with the same name.
        Defaults to False.

    Raises
    ------
    ValueError
        If the lens name is already registered and `override` is False.
    """
    if name in _LENS_REGISTRY and not override:
        raise ValueError(
            f"Lens '{name}' is already registered. Set override=True to replace."
        )

    _LENS_REGISTRY[name] = lens_module
    logger.info(f"Lens '{name}' successfully registered.")


def get_lens_names() -> List[str]:
    """
    Retrieves the list of available analytical lens names.

    Returns
    -------
    List[str]
        Ordered list of lens identifiers, prefixed with 'None' as default selection.
    """
    return ["None"] + list(_LENS_REGISTRY.keys())


def get_lens_module(lens_name: str) -> Optional[Any]:
    """
    Retrieves the module or object associated with a registered lens.

    Parameters
    ----------
    lens_name : str
        The unique display name of the target lens.

    Returns
    -------
    Optional[Any]
        The corresponding lens module or class instance, or None if not found.
    """
    return _LENS_REGISTRY.get(lens_name)

# --- ARCHIVO: lens_selection.py ---

"""
Lens Selection & Solution Grouping Module.

Provides state management, group filtering, and persistence mechanisms
for candidate Solutions of Interest (SOIs) extracted through analytical lenses.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

logger = logging.getLogger(__name__)


def ensure_soi_state() -> None:
    """
    Ensures that the state variable for saved SOIs exists in the Streamlit session.
    """
    if "saved_sois" not in st.session_state:
        st.session_state.saved_sois = []


def get_group_column(lens_df: Optional[pd.DataFrame]) -> Optional[str]:
    """
    Identifies the primary grouping or clustering column present in a DataFrame.

    Parameters
    ----------
    lens_df : Optional[pd.DataFrame]
        DataFrame transformed by an analytical lens.

    Returns
    -------
    Optional[str]
        Column name used for grouping ('group_label' or 'cluster_str'), or None if missing.
    """
    if lens_df is None:
        return None

    if "group_label" in lens_df.columns:
        return "group_label"

    if "cluster_str" in lens_df.columns:
        return "cluster_str"

    return None


def get_group_options(
    lens_df: pd.DataFrame, group_column: Optional[str]
) -> List[str]:
    """
    Extracts sorted unique string values from the specified grouping column.

    Parameters
    ----------
    lens_df : pd.DataFrame
        DataFrame containing solution data.
    group_column : Optional[str]
        Target grouping column name.

    Returns
    -------
    List[str]
        List of unique string representations of available group labels.
    """
    if group_column is None or group_column not in lens_df.columns:
        return []

    return sorted(
        lens_df[group_column].dropna().astype(str).unique().tolist()
    )


def filter_by_group(
    lens_df: Optional[pd.DataFrame],
    group_column: Optional[str],
    group_value: str,
) -> Optional[pd.DataFrame]:
    """
    Filters a DataFrame by a specified group label or value.

    Parameters
    ----------
    lens_df : Optional[pd.DataFrame]
        Input dataset to filter.
    group_column : Optional[str]
        Column name used for filtering.
    group_value : str
        Selected value to filter by ('All groups' returns an unmodified copy).

    Returns
    -------
    Optional[pd.DataFrame]
        Filtered copy of the input DataFrame.
    """
    if lens_df is None:
        return None

    if group_column is None or group_value == "All groups":
        return lens_df.copy()

    return lens_df[
        lens_df[group_column].astype(str) == str(group_value)
    ].copy()


def get_lens_label(active_lens: str) -> str:
    """
    Resolves the human-readable label for the currently active lens.

    Parameters
    ----------
    active_lens : str
        Active lens identifier.

    Returns
    -------
    str
        'Exploratory' if active_lens is "None", otherwise returns active_lens.
    """
    return "Exploratory" if active_lens == "None" else active_lens


def reset_soi_name_if_needed(active_lens: str, group_value: str) -> None:
    """
    Updates the session state's target SOI name when the active context changes.

    Parameters
    ----------
    active_lens : str
        Identifier of the current lens.
    group_value : str
        Selected group filtering value.
    """
    lens_label = get_lens_label(active_lens)
    suffix = group_value if group_value != "All groups" else "Current set"
    default_name = f"{lens_label} - {suffix} #{len(st.session_state.saved_sois) + 1}"

    name_context: Tuple[str, str] = (lens_label, group_value)

    if st.session_state.get("soi_name_context") != name_context:
        st.session_state["soi_name"] = default_name
        st.session_state["soi_name_context"] = name_context


def render_group_selector_and_save(
    placeholder: Optional[DeltaGenerator],
    active_lens: str,
    lens_df: Optional[pd.DataFrame],
    lens_params: Dict[str, Any],
) -> Optional[pd.DataFrame]:
    """
    Renders group filtering controls and save buttons for persisting SOIs.

    Parameters
    ----------
    placeholder : Optional[DeltaGenerator]
        Streamlit container placeholder for UI layout placement.
    active_lens : str
        Identifier of the active analytical lens.
    lens_df : Optional[pd.DataFrame]
        Transformed DataFrame containing candidate solutions.
    lens_params : Dict[str, Any]
        Configuration parameters associated with the current lens.

    Returns
    -------
    Optional[pd.DataFrame]
        The subset DataFrame selected by user interaction, or None if input is invalid.
    """
    ensure_soi_state()

    if placeholder is None or lens_df is None or lens_df.empty:
        return lens_df

    with placeholder.container():
        lens_label = get_lens_label(active_lens)
        group_column = get_group_column(lens_df)
        group_value = "All groups"

        if group_column is not None:
            group_options = get_group_options(lens_df, group_column)
            options = ["All groups"] + group_options

            selector_key = (
                f"soi_group_selector_{lens_label.replace(' ', '_')}"
            )

            if st.session_state.get(selector_key) not in options:
                st.session_state[selector_key] = "All groups"

            group_value = st.selectbox(
                "SOI group",
                options,
                key=selector_key,
                help=(
                    "Choose the group to promote as the current "
                    "Solution of Interest."
                ),
            )

        current_df = filter_by_group(lens_df, group_column, group_value)

        if current_df is None or current_df.empty:
            st.warning("Selected group contains no valid candidate solutions.")
            return lens_df

        st.caption(f"Current SOI candidate size: **{len(current_df)}** solutions")

        if active_lens == "None":
            st.caption("Source: exploratory current set.")

        st.markdown("---")

        reset_soi_name_if_needed(active_lens, group_value)

        soi_name = st.text_input("Name", key="soi_name")

        if st.button(
            "💾 Save Current Set",
            use_container_width=True,
            key="save_current_soi",
        ):
            # Safe extraction of IDs fallback to DataFrame index if 'id' column is missing
            solution_ids = (
                current_df["id"].tolist()
                if "id" in current_df.columns
                else current_df.index.tolist()
            )

            st.session_state.pending_save_soi = {
                "name": soi_name,
                "lens": lens_label,
                "method": lens_params.get("method", "Exploratory"),
                "params": lens_params,
                "ids": solution_ids,
                "group": group_value,
                "group_column": group_column,
                "source_size": len(lens_df),
                "soi_size": len(current_df),
            }

        return current_df

# --- ARCHIVO: lenses.py ---

"""
Lens Panel UI Component.

Renders sidebar user interface controls for analytical lens selection,
dynamic parameter configuration, and container placeholders within the
Decision Space Explorer framework.
"""

import logging
from typing import Any, Dict, Tuple
import pandas as pd
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from lenses.lens_registry import get_lens_module, get_lens_names

logger = logging.getLogger(__name__)


# =====================================================
# HEADER
# =====================================================

def render_lens_header(active_lens: str) -> None:
    """
    Renders the visual header and session context indicator for the selected lens.

    Parameters
    ----------
    active_lens : str
        The unique identifier of the currently active analytical lens.
    """
    if "active_soi_name" in st.session_state and st.session_state.active_soi_name:
        st.caption(
            f"Working on loaded SOI: **{st.session_state.active_soi_name}**"
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
            unsafe_allow_html=True,
        )


# =====================================================
# ACTIVE LENS PARAMS
# =====================================================

def render_active_lens_params(
    active_lens: str, dataset: Dict[str, Any], working_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Retrieves and renders the UI parameter controls for the active lens.

    Parameters
    ----------
    active_lens : str
        Identifier of the active analytical lens.
    dataset : Dict[str, Any]
        Global dataset context containing objective space definitions and metadata.
    working_df : pd.DataFrame
        The current active dataset of candidate solutions.

    Returns
    -------
    Dict[str, Any]
        Dictionary of parameter values collected from user interaction in the UI.
    """
    if active_lens == "None":
        return {}

    lens_module = get_lens_module(active_lens)

    if lens_module is None:
        st.warning(f"No module registered for lens: '{active_lens}'")
        return {}

    if not hasattr(lens_module, "render_params"):
        st.warning(
            f"Lens module '{active_lens}' does not define 'render_params()'."
        )
        return {}

    try:
        return lens_module.render_params(dataset, working_df)
    except Exception as e:
        logger.error(f"Error rendering parameters for lens '{active_lens}': {str(e)}")
        st.error(f"Error loading parameters for '{active_lens}': {str(e)}")
        return {}


# =====================================================
# MAIN LENS PANEL
# =====================================================

def render_lens_panel(
    dataset: Dict[str, Any], working_df: pd.DataFrame
) -> Tuple[str, Dict[str, Any], DeltaGenerator, DeltaGenerator]:
    """
    Renders the main sidebar panel for selecting lenses and managing state.

    Parameters
    ----------
    dataset : Dict[str, Any]
        Global dataset dictionary.
    working_df : pd.DataFrame
        Current working solution space DataFrame.

    Returns
    -------
    Tuple[str, Dict[str, Any], DeltaGenerator, DeltaGenerator]
        A tuple containing:
        - active_lens (str): Selected lens name.
        - params (Dict[str, Any]): Dictionary of collected lens parameters.
        - feedback_placeholder (DeltaGenerator): Streamlit UI container for feedback.
        - selection_placeholder (DeltaGenerator): Streamlit UI container for grouping/saving.
    """
    params: Dict[str, Any] = {}

    with st.sidebar.expander("🧭 Solution of Interest", expanded=False):
        active_lens = st.selectbox(
            "Select an analytical lens",
            get_lens_names(),
            key="active_lens",
        )

        render_lens_header(active_lens)

        params = render_active_lens_params(
            active_lens, dataset, working_df
        )

        # Container reserved for lens output metrics / feedback
        feedback_placeholder = st.empty()

        # Container reserved for group selection & candidate saving controls
        selection_placeholder = st.empty()

    return (
        active_lens,
        params,
        feedback_placeholder,
        selection_placeholder,
    )

# --- ARCHIVO: nrp_plugin.py ---

"""
Next Release Problem (NRP) Domain Plugin Module.

Provides mathematical indicator calculations and requirement dependency 
mappings for software release planning optimization models.
"""

from typing import Dict, List, Set, Union

import numpy as np
import pandas as pd

EPS: float = 1e-9


class NRPPlugin:
    """
    Next Release Problem (NRP) domain plugin.

    Provides derived indicators and attribute dependency mapping commonly 
    utilized in multi-objective software release planning problems.

    Parameters
    ----------
    var_prefix : str, default="req_"
        Prefix string identifying decision variable columns in the DataFrame.
    """

    def __init__(self, var_prefix: str = "req_") -> None:
        self.var_prefix = var_prefix

    # --------------------------------------------------
    # Indicator registry
    # --------------------------------------------------

    def available_indicators(self) -> Set[str]:
        """
        Retrieves the set of indicators supported by the NRP plugin.

        Returns
        -------
        Set[str]
            Set of available indicator names.
        """
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

    def requirements(self) -> Dict[str, List[str]]:
        """
        Maps each indicator to its required DataFrame column dependencies.

        Returns
        -------
        Dict[str, List[str]]
            Dictionary mapping indicator names to lists of required column names.
        """
        return {
            "productivity": ["satisfaction", "effort"],
            "effectiveness": ["satisfaction", "cost"],
            "dirtiness": ["dissatisfaction", "effort"],
            "annoyance": ["dissatisfaction", "satisfaction"],
            "stickiness": ["prevalence", "effort"],
            "robustness": ["satisfaction", "instability"],
            "fragility": ["prevalence", "instability", "effort"],
            "response": ["time", "effort"],
            "opportunity": ["satisfaction", "time"],
            "usage_efficiency": ["prevalence", "cost"],
            "scope": [],
            "squandering": ["effort"],
        }

    # --------------------------------------------------
    # Decision variables
    # --------------------------------------------------

    def decision_variables(self, df: pd.DataFrame) -> List[str]:
        """
        Identifies decision variable columns in the input DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Input solution space DataFrame.

        Returns
        -------
        List[str]
            List of column names matching the decision variable prefix.
        """
        if df is None or df.empty:
            return []

        return [c for c in df.columns if c.startswith(self.var_prefix)]

    # --------------------------------------------------
    # Indicator computation
    # --------------------------------------------------

    def compute_indicators(
        self, df: pd.DataFrame, indicators: Union[Set[str], List[str]]
    ) -> pd.DataFrame:
        """
        Computes specified software engineering indicators for the given DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Input solution space DataFrame.
        indicators : Union[Set[str], List[str]]
            Collection of indicator names to compute.

        Returns
        -------
        pd.DataFrame
            DataFrame augmented with computed indicator columns.
        """
        if df is None or df.empty:
            return df

        result = df.copy()
        req_cols = self.decision_variables(result)

        for indicator in indicators:
            try:
                if indicator == "productivity":
                    result[indicator] = result["satisfaction"] / np.maximum(
                        result["effort"], EPS
                    )

                elif indicator == "effectiveness":
                    result[indicator] = result["satisfaction"] / np.maximum(
                        result["cost"], EPS
                    )

                elif indicator == "squandering":
                    effort_max = result["effort"].max()
                    result[indicator] = (effort_max - result["effort"]) / np.maximum(
                        effort_max, EPS
                    )

                elif indicator == "dirtiness":
                    result[indicator] = np.where(
                        result["dissatisfaction"] == 0,
                        0.0,
                        result["dissatisfaction"]
                        / np.maximum(result["effort"], EPS),
                    )

                elif indicator == "annoyance":
                    result[indicator] = np.where(
                        result["dissatisfaction"] == 0,
                        0.0,
                        result["dissatisfaction"]
                        / np.maximum(result["satisfaction"], EPS),
                    )

                elif indicator == "stickiness":
                    result[indicator] = result["prevalence"] / np.maximum(
                        result["effort"], EPS
                    )

                elif indicator == "robustness":
                    result[indicator] = result["satisfaction"] / np.maximum(
                        result["instability"], EPS
                    )

                elif indicator == "fragility":
                    result[indicator] = (
                        result["prevalence"] * result["instability"]
                    ) / np.maximum(result["effort"], EPS)

                elif indicator == "response":
                    result[indicator] = np.where(
                        result["time"] == 0,
                        0.0,
                        result["effort"] / np.maximum(result["time"], EPS),
                    )

                elif indicator == "opportunity":
                    result[indicator] = np.where(
                        result["satisfaction"] == 0,
                        0.0,
                        result["satisfaction"] / np.maximum(result["time"], EPS),
                    )

                elif indicator == "usage_efficiency":
                    result[indicator] = (
                        result["prevalence"]
                        / result["cost"].replace(0, np.nan)
                    ).fillna(0.0)

                elif indicator == "scope":
                    if req_cols:
                        result[indicator] = (
                            result[req_cols].sum(axis=1) / len(req_cols)
                        )

            except Exception as e:
                print(f"[PLUGIN][NRP] Unable to compute '{indicator}': {e}")

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

"""
Sets of Interest (SOI) Registry Module.

Provides session state lifecycle management, formatting utilities, and 
Streamlit UI components for saving, inspecting, loading, and deleting candidate 
Sets of Interest (SOIs).
"""

import html
from typing import Any, Dict, List, Optional

import streamlit as st


# =====================================================
# SESSION STATE MANAGEMENT
# =====================================================


def ensure_soi_state() -> None:
    """Ensures session state contains required keys for SOI registry operations."""
    if "saved_sois" not in st.session_state:
        st.session_state.saved_sois = []


def has_loaded_soi() -> bool:
    """
    Checks whether an active SOI is currently loaded in session state.

    Returns
    -------
    bool
        True if both active SOI name and active SOI IDs exist in session state.
    """
    return (
        "active_soi_name" in st.session_state
        and "active_soi_ids" in st.session_state
    )


def clear_loaded_soi() -> None:
    """Clears the currently active SOI from session state and triggers lens reset."""
    for key in ["active_soi_ids", "active_soi_name", "active_soi_metadata"]:
        if key in st.session_state:
            del st.session_state[key]

    st.session_state["pending_lens_reset"] = True


def load_soi(soi: Dict[str, Any]) -> None:
    """
    Loads a saved SOI dictionary into active session state.

    Parameters
    ----------
    soi : Dict[str, Any]
        Dictionary representing the SOI configuration and solution IDs.
    """
    st.session_state["active_soi_ids"] = soi.get("ids", [])
    st.session_state["active_soi_name"] = soi.get("name", "Unnamed SOI")
    st.session_state["active_soi_metadata"] = {
        "lens": soi.get("lens"),
        "method": soi.get("method"),
        "group": soi.get("group"),
        "group_column": soi.get("group_column"),
        "source_size": soi.get("source_size"),
        "soi_size": soi.get("soi_size", len(soi.get("ids", []))),
        "created_at": soi.get("created_at"),
        "params": soi.get("params", {}),
    }

    st.session_state["pending_lens_reset"] = True


def delete_soi(idx: int) -> None:
    """
    Deletes an SOI entry from the registry by list index.

    Parameters
    ----------
    idx : int
        Index of the SOI to remove from `st.session_state.saved_sois`.
    """
    ensure_soi_state()
    saved_sois: List[Dict[str, Any]] = st.session_state.saved_sois

    if 0 <= idx < len(saved_sois):
        deleted_soi = saved_sois.pop(idx)
        deleted_name = deleted_soi.get("name")

        if st.session_state.get("active_soi_name") == deleted_name:
            clear_loaded_soi()


# =====================================================
# LABEL & FORMATTING HELPERS
# =====================================================


def normalize_method_label(soi: Dict[str, Any]) -> Optional[str]:
    """
    Normalizes the method label extracted from an SOI dictionary.

    Parameters
    ----------
    soi : Dict[str, Any]
        SOI metadata dictionary.

    Returns
    -------
    Optional[str]
        Normalized method label string or None if uninformative.
    """
    method = soi.get("method")

    if method is None or method == "None":
        lens = soi.get("lens", "Unknown")
        if lens == "Exploratory":
            return "Exploratory"
        return None

    return str(method)


def is_informative_group(group: Any) -> bool:
    """
    Determines if a group identifier represents a non-trivial subgroup filter.

    Parameters
    ----------
    group : Any
        Group name or filter value.

    Returns
    -------
    bool
        True if group is non-empty and distinct from default global indicators.
    """
    return (
        group is not None
        and group != ""
        and group != "All groups"
    )


def build_soi_main_label(soi: Dict[str, Any]) -> str:
    """
    Constructs the primary label string displayed in registry lists.

    Parameters
    ----------
    soi : Dict[str, Any]
        SOI metadata dictionary.

    Returns
    -------
    str
        Formatted main label string.
    """
    name = soi.get("name", "Unnamed SOI")
    size = len(soi.get("ids", []))
    lens = soi.get("lens", "Unknown")
    method = normalize_method_label(soi)

    if method:
        return f"{name} [{size}] · {lens} / {method}"

    return f"{name} [{size}] · {lens}"


def build_compact_trace_label(soi: Dict[str, Any]) -> str:
    """
    Constructs a compact single-line provenance trace label.

    Parameters
    ----------
    soi : Dict[str, Any]
        SOI metadata dictionary.

    Returns
    -------
    str
        Compact provenance trace summary string.
    """
    parts = []
    group = soi.get("group")

    if is_informative_group(group):
        parts.append(f"Group: {group}")

    source_size = soi.get("source_size")
    soi_size = soi.get("soi_size", len(soi.get("ids", [])))

    if source_size is not None:
        parts.append(f"{soi_size}/{source_size} solutions")
    else:
        parts.append(f"{soi_size} solutions")

    created_at = soi.get("created_at")
    if created_at:
        parts.append(str(created_at))

    return " · ".join(parts)


def build_tooltip_text(soi: Dict[str, Any]) -> str:
    """
    Generates plain text tooltip information describing complete SOI parameters.

    Parameters
    ----------
    soi : Dict[str, Any]
        SOI metadata dictionary.

    Returns
    -------
    str
        Multi-line plain text summary for tooltip rendering.
    """
    lens = soi.get("lens", "Unknown")
    method = normalize_method_label(soi)
    group = soi.get("group")
    source_size = soi.get("source_size")
    soi_size = soi.get("soi_size", len(soi.get("ids", [])))
    created_at = soi.get("created_at")
    params = soi.get("params", {})

    lines = [
        f"Lens: {lens}",
        f"Method: {method if method else 'N/A'}",
        f"SOI size: {soi_size}",
    ]

    if source_size is not None:
        lines.append(f"Source size: {source_size}")

    if is_informative_group(group):
        lines.append(f"Group: {group}")

    if created_at:
        lines.append(f"Created: {created_at}")

    if params and isinstance(params, dict):
        compact_params = ", ".join(
            [
                f"{k}={v}"
                for k, v in params.items()
                if k not in ["selected_sois", "params"]
            ]
        )
        if compact_params:
            lines.append(f"Params: {compact_params}")

    return "\n".join(lines)


def render_trace_tooltip(soi: Dict[str, Any]) -> None:
    """
    Renders an HTML-escaped inline provenance trace tag with hover tooltip.

    Parameters
    ----------
    soi : Dict[str, Any]
        SOI metadata dictionary.
    """
    compact_label = build_compact_trace_label(soi)
    tooltip = build_tooltip_text(soi)

    safe_label = html.escape(compact_label)
    safe_tooltip = html.escape(tooltip)

    st.markdown(
        f'<span title="{safe_tooltip}" style="'
        f'font-size:0.82rem; color:#6b7280; line-height:1.2; cursor:help;'
        f'">ⓘ {safe_label}</span>',
        unsafe_allow_html=True,
    )


# =====================================================
# UI RENDER COMPONENTS
# =====================================================


def render_loaded_soi_status() -> None:
    """Renders status header and control buttons for the currently active loaded SOI."""
    if not has_loaded_soi():
        return

    active_name = st.session_state.active_soi_name
    active_ids = st.session_state.active_soi_ids

    st.success(f"Active SOI: {active_name} ({len(active_ids)} solutions)")

    metadata = st.session_state.get("active_soi_metadata", {})
    if metadata:
        lens = metadata.get("lens")
        method = metadata.get("method")
        group = metadata.get("group")

        label_parts = []
        if lens:
            label_parts.append(str(lens))
        if method:
            label_parts.append(str(method))
        if is_informative_group(group):
            label_parts.append(str(group))

        if label_parts:
            st.caption(" · ".join(label_parts))

    if st.button(
        "Clear Loaded SOI",
        use_container_width=True,
        key="clear_loaded_soi",
    ):
        clear_loaded_soi()
        st.rerun()

    st.divider()


def render_saved_soi_row(soi: Dict[str, Any], idx: int) -> None:
    """
    Renders a single row entry for a saved SOI in the registry panel.

    Parameters
    ----------
    soi : Dict[str, Any]
        Target SOI metadata dictionary.
    idx : int
        Registry list index.
    """
    col_info, col_help, col_load, col_delete = st.columns([0.68, 0.08, 0.14, 0.10])

    with col_info:
        st.caption(f"• {build_soi_main_label(soi)}")

    with col_help:
        tooltip = build_tooltip_text(soi)
        st.button(
            "ⓘ",
            key=f"info_soi_{idx}",
            help=tooltip,
            use_container_width=True,
        )

    with col_load:
        if st.button(
            "Load",
            key=f"load_soi_{idx}",
            use_container_width=True,
        ):
            load_soi(soi)
            st.rerun()

    with col_delete:
        if st.button(
            "🗑️",
            key=f"delete_soi_{idx}",
            use_container_width=True,
        ):
            delete_soi(idx)
            st.rerun()


# =====================================================
# MAIN TAB RENDERER
# =====================================================


def render_soi_tab() -> None:
    """Renders the complete Sets of Interest (SOI) registry tab interface."""
    ensure_soi_state()

    if not st.session_state.saved_sois:
        st.info("No saved SOIs.")
        return

    render_loaded_soi_status()

    for idx, soi in enumerate(st.session_state.saved_sois):
        render_saved_soi_row(soi, idx)

# --- ARCHIVO: streamlit_app.py ---

from datetime import datetime
import streamlit as st

from css.css_comparison import render_css_comparison
from css.css_panel import render_css_panel
from core.enrichment import render_enrichment
from core.framing import apply_framing
from core.input_panel import render_input_panel
from core.workspace import render_workspace
from core.workspace_controls import render_workspace_controls
from lenses.lens_engine import apply_lens
from lenses.lens_feedback import render_lens_feedback
from lenses.lens_selection import render_group_selector_and_save
from lenses.lenses import render_lens_panel

# --------------------------------------------------------------------------------------
# Page Configuration & Global Styling
# --------------------------------------------------------------------------------------
st.set_page_config(page_title="Decision Space Explorer", layout="wide")

st.markdown(
    """
    <style>
    [data-testid="stExpander"] details summary p {
        font-size: 1.2rem;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Decision Space Explorer")

# --------------------------------------------------------------------------------------
# 1. Input Panel
# --------------------------------------------------------------------------------------
dataset = render_input_panel()
if dataset is None:
    st.info("Select a domain configuration to begin.")
    st.stop()

# --------------------------------------------------------------------------------------
# 2. Enrichment Step (Domain indicators computation)
# --------------------------------------------------------------------------------------
dataset = render_enrichment(dataset)

# --------------------------------------------------------------------------------------
# 3. Workspace Controls & Framing
# --------------------------------------------------------------------------------------
dimensions = dataset["metrics"] + dataset["selected_indicators"]
show_ids = render_workspace_controls(dimensions)

framed_df = apply_framing(dataset)

# --------------------------------------------------------------------------------------
# 4. Working Dataset Construction (Active SOI Filtering)
# --------------------------------------------------------------------------------------
working_df = framed_df.copy()

if "active_soi_ids" in st.session_state:
    working_df = working_df[
        working_df["id"].isin(st.session_state.active_soi_ids)
    ].copy()

# Reset lens if requested by state
if st.session_state.get("pending_lens_reset", False):
    st.session_state["active_lens"] = "None"
    st.session_state["pending_lens_reset"] = False

# --------------------------------------------------------------------------------------
# 5. Lenses Processing & Engine
# --------------------------------------------------------------------------------------
(
    active_lens,
    lens_params,
    feedback_placeholder,
    selection_placeholder,
) = render_lens_panel(dataset, working_df)

lens_df = apply_lens(working_df, active_lens, lens_params, dataset)

if lens_df is None:
    st.sidebar.warning(
        "The selected lens returned no dataset. Reverting to the current working dataset."
    )
    lens_df = working_df.copy()

# Render Feedback & Group Selector
render_lens_feedback(feedback_placeholder, active_lens, lens_df)

current_df = render_group_selector_and_save(
    selection_placeholder, active_lens, lens_df, lens_params
)

if current_df is None:
    current_df = lens_df.copy()

# --------------------------------------------------------------------------------------
# 6. Save State of Interest (SOI)
# --------------------------------------------------------------------------------------
if "pending_save_soi" in st.session_state:
    if "saved_sois" not in st.session_state:
        st.session_state.saved_sois = []

    pending = st.session_state.pending_save_soi
    existing_names = [soi["name"] for soi in st.session_state.saved_sois]

    if pending["name"] in existing_names:
        st.sidebar.warning("A SOI with this name already exists.")
    else:
        st.session_state.saved_sois.append(
            {
                "name": pending["name"],
                "lens": pending["lens"],
                "params": pending.get("params", {}),
                "ids": pending.get("ids", current_df["id"].tolist()),
                "group": pending.get("group", "All groups"),
                "group_column": pending.get("group_column"),
            }
        )
        st.sidebar.success(f"Saved SOI: {pending['name']}")
    del st.session_state["pending_save_soi"]

# --------------------------------------------------------------------------------------
# 7. Candidate Solution Set (CSS) & Workspace Rendering
# --------------------------------------------------------------------------------------
css_df = render_css_panel(current_df, dataset)

render_workspace(css_df, dataset, show_ids)

render_css_comparison(css_df, dataset)

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

"""
Workspace Module.

Serves as the main orchestrator for rendering the visual workspace layout, 
combining executive summary views, dataset previews, and interactive 
decision-space maps.
"""

from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from core.workspace_maps import render_maps
from core.workspace_summary import render_summary


# =====================================================
# HELPER FUNCTIONS
# =====================================================


def get_workspace_dimensions(dataset: Dict[str, Any]) -> List[str]:
    """
    Extracts active optimization metrics and selected indicator dimensions.

    Parameters
    ----------
    dataset : Dict[str, Any]
        Global dataset context dictionary.

    Returns
    -------
    List[str]
        List of active dimension column names.
    """
    if not dataset:
        return []

    metrics = dataset.get("metrics", []) or []
    indicators = dataset.get("selected_indicators", []) or []

    return list(metrics) + list(indicators)


def render_empty_workspace_message() -> None:
    """Renders an error message when no valid dataset DataFrame is available."""
    st.error("No dataset is available for the workspace.")


def render_no_map_message() -> None:
    """Renders a warning message when insufficient dimensions exist for mapping."""
    st.warning(
        "At least two dimensions are required to render decision-space maps."
    )


# =====================================================
# MAIN WORKSPACE ENTRY POINT
# =====================================================


def render_workspace(
    df: Optional[pd.DataFrame],
    dataset: Dict[str, Any],
    show_ids: bool = False,
) -> None:
    """
    Renders the primary visual workspace UI.

    Integrates high-level summary metrics, solution set preview tables, and 
    interactive decision-space maps.

    Parameters
    ----------
    df : Optional[pd.DataFrame]
        Active solution space DataFrame.
    dataset : Dict[str, Any]
        Global dataset context dictionary.
    show_ids : bool, default=False
        Whether to display solution ID text labels across workspace maps.
    """
    if df is None or df.empty:
        render_empty_workspace_message()
        return

    dimensions = get_workspace_dimensions(dataset)

    # ==================== SUMMARY & CURRENT SET ====================
    render_summary(df, dataset)

    # ====================== DECISION MAPS ==========================
    if len(dimensions) < 2:
        render_no_map_message()
    else:
        render_maps(df, dataset, dimensions, show_ids)

# --- ARCHIVO: workspace_controls.py ---

"""
Workspace Controls Module.

Provides sidebar UI controls for managing interactive visual workspace maps, 
allowing users to dynamically create, reset, and configure scatter plot layouts.
"""

from typing import List, Optional

import streamlit as st


# =====================================================
# UI RENDERING & WORKSPACE CONTROL
# =====================================================


def render_workspace_controls(dimensions: Optional[List[str]] = None) -> bool:
    """
    Renders UI controls in the sidebar for managing visual map instances.

    Parameters
    ----------
    dimensions : Optional[List[str]], default=None
        List of available metric and indicator dimension names.

    Returns
    -------
    bool
        State of the 'Show solution IDs' checkbox toggle.
    """
    if dimensions is None:
        dimensions = []

    with st.sidebar.expander("🗺️ Visual Workspace", expanded=False):
        if "maps" not in st.session_state:
            st.session_state.maps = []

        can_create_map = len(dimensions) >= 2
        col1, col2 = st.columns([0.50, 0.50])

        with col1:
            if st.button(
                "🔄 Reset Maps",
                use_container_width=True,
                disabled=not can_create_map,
            ):
                st.session_state.maps = [
                    {
                        "x": dimensions[0],
                        "y": dimensions[1],
                        "z": None,
                        "color": None,
                    }
                ]
                st.rerun()

        with col2:
            if st.button(
                "New Map",
                use_container_width=True,
                disabled=not can_create_map,
            ):
                st.session_state.maps.append(
                    {
                        "x": dimensions[0],
                        "y": dimensions[1],
                        "z": None,
                        "color": None,
                    }
                )
                st.rerun()

        if not can_create_map:
            st.info("At least two dimensions are required to create maps.")

        show_ids = st.checkbox("Show solution IDs", value=False)
        st.caption(f"Active maps: {len(st.session_state.maps)}")

    return show_ids

# --- ARCHIVO: workspace_dataset.py ---

"""
Workspace Dataset Module.

Provides utilities for column ordering, dynamic labeling, and rendering 
interactive data table previews for active solution sets.
"""

from typing import Any, Dict, List

import pandas as pd
import streamlit as st


# =====================================================
# HELPER FUNCTIONS
# =====================================================


def get_ordered_columns(df: pd.DataFrame, dataset: Dict[str, Any]) -> List[str]:
    """
    Orders DataFrame columns logically by category: ID, objectives, indicators, 
    miscellaneous metadata, and decision variables.

    Parameters
    ----------
    df : pd.DataFrame
        Active solution space DataFrame.
    dataset : Dict[str, Any]
        Global dataset context dictionary.

    Returns
    -------
    List[str]
        List of column names in prioritized ordering.
    """
    if df is None or df.empty:
        return []

    if not dataset:
        dataset = {}

    config = dataset.get("config", {})
    var_prefix = config.get("var_prefix", "x_")

    objective_cols = dataset.get("metrics", [])
    indicator_cols = dataset.get("selected_indicators", [])

    decision_cols = [
        col for col in df.columns if var_prefix and col.startswith(var_prefix)
    ]
    control_cols = {"highlight", "highlight_label", "label"}

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

    raw_ordered_cols = (
        (["id"] if "id" in df.columns else [])
        + objective_cols
        + indicator_cols
        + other_cols
        + decision_cols
    )

    # Deduplicate while preserving order and ensuring columns exist in df
    seen = set()
    ordered_cols: List[str] = []
    for col in raw_ordered_cols:
        if col in df.columns and col not in seen:
            seen.add(col)
            ordered_cols.append(col)

    return ordered_cols


def get_current_set_label() -> str:
    """
    Retrieves UI label corresponding to the active solution set state.

    Returns
    -------
    str
        Human-readable label for the current set.
    """
    if st.session_state.get("css_enabled", False):
        return "Current CSS"
    return "Current Decision Set"


# =====================================================
# UI RENDERING COMPONENTS
# =====================================================


def render_dataset_table(df: pd.DataFrame, dataset: Dict[str, Any]) -> None:
    """
    Renders an interactive Streamlit DataFrame table with prioritized column ordering.

    Parameters
    ----------
    df : pd.DataFrame
        Solution set DataFrame to display.
    dataset : Dict[str, Any]
        Global dataset context dictionary.
    """
    if df is None or df.empty:
        st.info("No solutions available in the current dataset.")
        return

    label = get_current_set_label()
    st.markdown(f"#### 📋 {label}")

    ordered_cols = get_ordered_columns(df, dataset)
    display_df = df[ordered_cols] if ordered_cols else df

    st.dataframe(
        display_df,
        use_container_width=True,
        height=420,
        hide_index=True,
    )


def render_dataset_preview(df: pd.DataFrame, dataset: Dict[str, Any]) -> None:
    """
    Renders a collapsible expander containing the interactive dataset table preview.

    Parameters
    ----------
    df : pd.DataFrame
        Solution set DataFrame to display.
    dataset : Dict[str, Any]
        Global dataset context dictionary.
    """
    if df is None or df.empty:
        return

    label = get_current_set_label()
    config = dataset.get("config", {}) if dataset else {}
    var_prefix = config.get("var_prefix", "x_")

    with st.expander(f"📋 {label} (prefix: {var_prefix})", expanded=False):
        render_dataset_table(df, dataset)

# --- ARCHIVO: workspace_maps.py ---

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

# --- ARCHIVO: workspace_summary.py ---

"""
Workspace Summary Module.

Provides summary metrics, lens attribute inspection, executive report generation, 
and file export capabilities (Markdown/CSV) for the visual workspace.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from core.workspace_dataset import render_dataset_table
from soi.soi_registry import render_soi_tab


# =====================================================
# DERIVED / LENS COLUMNS
# =====================================================


def get_lens_columns(df: pd.DataFrame) -> List[str]:
    """
    Identifies structural and analytical lens columns present in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Active solution space DataFrame.

    Returns
    -------
    List[str]
        List of identified structural and derived lens column names.
    """
    if df is None or df.empty:
        return []

    lens_prefixes = (
        "preference_",
        "efficiency_",
        "diversity_",
        "domain_",
        "indicator_",
        "consensus_",
    )

    lens_columns = [
        col
        for col in df.columns
        if any(col.startswith(prefix) for prefix in lens_prefixes)
    ]

    structural_columns = [
        col
        for col in [
            "cluster",
            "cluster_str",
            "group_label",
            "group_base",
            "highlight",
        ]
        if col in df.columns
    ]

    return structural_columns + lens_columns


# =====================================================
# REPORT GENERATOR HELPER
# =====================================================


def generate_markdown_report(df: pd.DataFrame, dataset: Dict[str, Any]) -> str:
    """
    Generates an executive decision report formatted in Markdown for export.

    Parameters
    ----------
    df : pd.DataFrame
        Active solution space DataFrame.
    dataset : Dict[str, Any]
        Global dataset context dictionary.

    Returns
    -------
    str
        Formatted Markdown report text.
    """
    if df is None:
        df = pd.DataFrame()

    if dataset is None:
        dataset = {}

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    dataset_name = dataset.get("config", {}).get(
        "name", "Pareto Optimization Dataset"
    )
    saved_sois = st.session_state.get("saved_sois", [])
    highlight_ids = st.session_state.get("css_highlight_ids", [])

    report = []

    # 1. Header
    report.append(f"# 📊 Executive Decision Report: {dataset_name}")
    report.append(f"**Generated on:** {timestamp}\n")
    report.append("---")

    # 2. Executive Overview
    report.append("## 1. Executive Overview")
    css_active = st.session_state.get("css_enabled", False)
    report.append(f"- **Current Set Size:** {len(df)} solutions")
    report.append(f"- **CSS Lock Status:** {'Active' if css_active else 'Inactive'}")
    report.append(f"- **Saved Sets of Interest (SOIs):** {len(saved_sois)} sets")
    report.append(
        f"- **Highlighted Solutions:** {len(highlight_ids)} solutions\n"
    )

    # 3. Saved SOIs Summary
    report.append("## 2. Analyzed Sets of Interest (SOIs)")
    if saved_sois:
        report.append("| SOI Name | Lens / Type | Size | Source Group |")
        report.append("| :--- | :--- | :--- | :--- |")
        for soi in saved_sois:
            name = soi.get("name", "Unnamed")
            lens = soi.get("lens", soi.get("type", "Manual"))
            size = soi.get("soi_size", len(soi.get("ids", [])))
            group = soi.get("group", "N/A")
            report.append(f"| {name} | {lens} | {size} | {group} |")
    else:
        report.append("_No SOIs were explicitly saved during this session._")
    report.append("\n")

    # 4. Highlighted Solutions Comparison
    report.append("## 3. Highlighted Solutions Comparison")
    if highlight_ids and "id" in df.columns and not df.empty:
        high_df = df[df["id"].isin(highlight_ids)].copy()
        if not high_df.empty:
            metrics = dataset.get("metrics", []) + dataset.get(
                "selected_indicators", []
            )
            show_cols = ["id"] + [m for m in metrics if m in high_df.columns]
            report.append(high_df[show_cols].to_markdown(index=False))
        else:
            report.append(
                "_No matching highlighted solutions found in current set._"
            )
    else:
        report.append(
            "_No specific solutions are currently highlighted for comparison._"
        )

    report.append("\n---")
    report.append(
        "*Report generated automatically by Pareto Framework Decision Tool.*"
    )

    return "\n".join(report)


# =====================================================
# SUMMARY METRICS & EXPORTS
# =====================================================


def render_summary_metrics(df: pd.DataFrame, dataset: Dict[str, Any]) -> None:
    """
    Renders metric card indicators summarizing current dataset dimensions.

    Parameters
    ----------
    df : pd.DataFrame
        Active solution space DataFrame.
    dataset : Dict[str, Any]
        Global dataset context dictionary.
    """
    if df is None:
        return

    if dataset is None:
        dataset = {}

    c1, c2, c3, c4 = st.columns(4)

    var_prefix = dataset.get("config", {}).get("var_prefix", "x_")
    if "decision_variables" in dataset and isinstance(
        dataset["decision_variables"], list
    ):
        num_vars = len(dataset["decision_variables"])
    else:
        num_vars = len([col for col in df.columns if col.startswith(var_prefix)])

    with c1:
        st.metric("Solutions", len(df))
    with c2:
        st.metric("Attributes", len(df.columns))
    with c3:
        st.metric("Decision Variables", num_vars)
    with c4:
        css_status = (
            "Active" if st.session_state.get("css_enabled", False) else "Inactive"
        )
        st.metric("CSS", css_status)


def render_lens_summary(df: pd.DataFrame) -> None:
    """
    Displays a caption summarizing derived analytical lens attributes.

    Parameters
    ----------
    df : pd.DataFrame
        Active solution space DataFrame.
    """
    lens_columns = get_lens_columns(df)

    if not lens_columns:
        st.caption("No derived lens columns in the current set.")
        return

    st.caption("Derived columns: " + ", ".join(lens_columns))


def render_export_section(df: pd.DataFrame, dataset: Dict[str, Any]) -> None:
    """
    Renders export controls for downloading executive reports and CSV data.

    Parameters
    ----------
    df : pd.DataFrame
        Active solution space DataFrame.
    dataset : Dict[str, Any]
        Global dataset context dictionary.
    """
    if df is None or dataset is None:
        return

    st.markdown("---")
    st.markdown("##### 📥 Export Options")
    col_report, col_csv = st.columns(2)

    config_name = dataset.get("config", {}).get("name", "pareto")

    with col_report:
        report_md = generate_markdown_report(df, dataset)
        st.download_button(
            label="📄 Export Executive Report (.md)",
            data=report_md,
            file_name=f"executive_report_{config_name}.md",
            mime="text/markdown",
            use_container_width=True,
            type="primary",
        )

    with col_csv:
        st.download_button(
            label="📊 Export Current Set (.csv)",
            data=df.to_csv(index=False),
            file_name="current_set.csv",
            mime="text/csv",
            use_container_width=True,
        )


def get_summary_label() -> str:
    """
    Returns dynamic title string for summary container based on CSS state.

    Returns
    -------
    str
        Summary section title.
    """
    if st.session_state.get("css_enabled", False):
        return "Summary / Current CSS / Saved SOIs"
    return "Summary / Current Set / Saved SOIs"


# =====================================================
# MAIN RENDERER
# =====================================================


def render_summary(df: pd.DataFrame, dataset: Dict[str, Any]) -> None:
    """
    Main entry point for rendering the workspace summary, data table, and SOI tabs.

    Parameters
    ----------
    df : pd.DataFrame
        Active solution space DataFrame.
    dataset : Dict[str, Any]
        Global dataset context dictionary.
    """
    if df is None or df.empty:
        st.error(
            "Dataset summary cannot be rendered "
            "because the current dataframe is empty."
        )
        return

    label = get_summary_label()

    with st.expander(f"📊 {label}", expanded=False):
        tab_overview, tab_current, tab_saved_soi = st.tabs(
            [
                "**| Overview |**",
                "**| Current Set |**",
                "**| Saved SOIs |**",
            ]
        )

        with tab_overview:
            render_summary_metrics(df, dataset)
            config = dataset.get("config", {}) if dataset else {}
            st.caption(
                f"Decision-variable prefix: {config.get('var_prefix', 'x_')}"
            )
            render_lens_summary(df)
            render_export_section(df, dataset)

        with tab_current:
            render_dataset_table(df, dataset)

        with tab_saved_soi:
            render_soi_tab()
