        
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