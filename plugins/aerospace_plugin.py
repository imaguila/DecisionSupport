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