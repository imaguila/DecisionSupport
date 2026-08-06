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