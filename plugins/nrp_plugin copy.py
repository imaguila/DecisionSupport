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