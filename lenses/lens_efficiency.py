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
