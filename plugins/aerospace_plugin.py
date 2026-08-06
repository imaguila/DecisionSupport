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