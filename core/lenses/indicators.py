"""
Indicator Lens Module (Headless Core).

Provides multi-criteria selection methods based on domain indicators:
1. Top-N Matches: Aggregates top solutions across individual target dimensions
   and allows filtering by exact or minimum match counts.
2. Non-Dominated Sorting: Identifies Pareto-optimal solutions within the enriched 
   indicator space.

Pure Python & Pandas module — Framework agnostic.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


# =====================================================
# HELPER FUNCTIONS
# =====================================================


def _sanitize_criteria(
    df: pd.DataFrame, maximize: List[str], minimize: List[str]
) -> Tuple[List[str], List[str], List[str]]:
    """Validates criteria column existence within the input DataFrame."""
    valid_max = [m for m in maximize if m in df.columns]
    valid_min = [
        m for m in minimize if m in df.columns and m not in valid_max
    ]
    criteria = valid_max + valid_min
    return valid_max, valid_min, criteria


def _build_group_labels_from_count(
    result: pd.DataFrame, count_column: str
) -> pd.DataFrame:
    """Generates categorical grouping labels based on indicator match counts."""
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
    df: pd.DataFrame,
    maximize: List[str],
    minimize: List[str],
    top_n: int,
    match_filter: str = "All",
    target_matches: int = 1,
) -> pd.DataFrame:
    """
    Computes Top-N match counts per solution across selected criteria
    and filters by user-defined match count groups.
    """
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

    # Base filter: Must have at least 1 match
    result = result[result["domain_match_count"] > 0].copy()

    if result.empty:
        return result

    # ------------------------------------------------------------------
    # SELECCIÓN/FILTRADO POR GRUPO DE COINCIDENCIAS (N-Matches)
    # ------------------------------------------------------------------
    if match_filter == "Highest":
        max_c = result["domain_match_count"].max()
        result = result[result["domain_match_count"] == max_c].copy()
    elif match_filter == "At least":
        result = result[result["domain_match_count"] >= target_matches].copy()
    elif match_filter == "Exact":
        result = result[result["domain_match_count"] == target_matches].copy()

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
# MAIN ENTRY POINT
# =====================================================


def apply(
    df: pd.DataFrame,
    params: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Applies selected indicator lens method to isolate solutions.

    Parameters
    ----------
    df : pd.DataFrame
        Input working solution space DataFrame.
    params : Dict[str, Any]
        Indicator lens setup parameters:
        - method: 'Top-N Matches' | 'Non-dominated'
        - maximize: List[str]
        - minimize: List[str]
        - top_n: int
        - match_filter: 'All' | 'Highest' | 'At least' | 'Exact'
        - target_matches: int
    context : Dict[str, Any], optional
        Engine context dataset metadata.

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
        match_filter = params.get("match_filter", "All")
        target_matches = params.get("target_matches", 1)

        return _apply_top_n_matches(
            result, maximize, minimize, top_n, match_filter, target_matches
        )

    if method == "Non-dominated":
        return _apply_non_dominated(result, maximize, minimize)

    return result


def apply_domain_lens(
    df: pd.DataFrame, maximize: List[str], minimize: List[str], top_n: int
) -> pd.DataFrame:
    """Legacy entry point for Top-N Domain match filtering."""
    if df is None or df.empty:
        return df

    valid_max, valid_min, criteria = _sanitize_criteria(
        df, maximize, minimize
    )

    if not criteria:
        return df.copy()

    return _apply_top_n_matches(df, valid_max, valid_min, top_n)