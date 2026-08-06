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