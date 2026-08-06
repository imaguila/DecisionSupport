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