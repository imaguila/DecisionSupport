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