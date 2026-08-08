"""
Efficiency Lens Module (Headless Core).

Ranks candidate solutions based on benefit-cost trade-offs using raw ratios,
min-max normalized efficiency, composite cost aggregation, or Euclidean distance
to ideal target states in objective space.

Pure Python & Pandas module — Framework agnostic.
"""

from typing import Any, Dict, List, Optional, Union
import pandas as pd

EPS: float = 1e-9


# =====================================================
# HELPER FUNCTIONS
# =====================================================


def _normalize_series(series: pd.Series) -> pd.Series:
    """Normalizes a numeric pandas Series to range [0.0, 1.0] via Min-Max scaling."""
    min_v = series.min()
    max_v = series.max()

    if max_v > min_v:
        return (series - min_v) / (max_v - min_v)

    return pd.Series(0.0, index=series.index)


def _resolve_cost_metrics(
    result: pd.DataFrame,
    benefit: str,
    cost: Optional[Union[str, List[str]]],
) -> List[str]:
    """Resolves and validates cost metric column names present in the DataFrame."""
    if cost is None:
        return []

    if isinstance(cost, str):
        cost_metrics = [cost]
    else:
        cost_metrics = [c for c in cost if c in result.columns]

    return [c for c in cost_metrics if c != benefit]


# =====================================================
# SCORE METHOD ENGINES
# =====================================================


def _benefit_cost_ratio(
    result: pd.DataFrame, benefit: str, cost_metrics: List[str]
) -> pd.Series:
    """Calculates unnormalized Benefit / Cost ratio."""
    cost_metric = cost_metrics[0]
    safe_cost = result[cost_metric].replace(0, EPS)
    return result[benefit] / safe_cost


def _normalized_ratio(
    result: pd.DataFrame, benefit: str, cost_metrics: List[str]
) -> pd.Series:
    """Calculates Min-Max normalized Benefit / Normalized Cost ratio."""
    cost_metric = cost_metrics[0]
    benefit_norm = _normalize_series(result[benefit])
    cost_norm = _normalize_series(result[cost_metric])
    return benefit_norm / (cost_norm + EPS)


def _distance_to_ideal(
    result: pd.DataFrame, benefit: str, cost_metrics: List[str]
) -> pd.Series:
    """Calculates proximity score based on Euclidean distance to ideal state."""
    cost_metric = cost_metrics[0]
    benefit_norm = _normalize_series(result[benefit])
    cost_norm = _normalize_series(result[cost_metric])

    distance_to_ideal = ((1.0 - benefit_norm) ** 2 + (cost_norm) ** 2) ** 0.5
    max_distance = 2.0**0.5

    return 1.0 - (distance_to_ideal / max_distance)


def _composite_cost_ratio(
    result: pd.DataFrame, benefit: str, cost_metrics: List[str]
) -> pd.Series:
    """Calculates normalized Benefit / Average Composite Normalized Costs ratio."""
    benefit_norm = _normalize_series(result[benefit])
    composite_cost = pd.Series(0.0, index=result.index)

    for cost_metric in cost_metrics:
        composite_cost += _normalize_series(result[cost_metric])

    composite_cost /= len(cost_metrics)
    return benefit_norm / (composite_cost + EPS)


# =====================================================
# MAIN ENTRY POINT
# =====================================================


def apply(
    df: pd.DataFrame, params: Dict[str, Any], context: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Applies selected efficiency lens method to calculate score and rank solutions.

    Parameters
    ----------
    df : pd.DataFrame
        Input solution space DataFrame.
    params : Dict[str, Any]
        Efficiency configuration parameters:
        - method: str ('Benefit/Cost Ratio', 'Normalized Ratio', 'Distance to Ideal', 'Composite Cost Ratio')
        - benefit: str (Column name)
        - cost: str | List[str] (Column name(s))
        - top_n: int
    context : Dict[str, Any], optional
        Engine context metadata.

    Returns
    -------
    pd.DataFrame
        Ranked top N subset of solutions enriched with efficiency scores.
    """
    if df is None or df.empty:
        return df

    result = df.copy()
    method = params.get("method", "Benefit/Cost Ratio")
    benefit = params.get("benefit")
    cost = params.get("cost")

    if benefit is None or benefit not in result.columns:
        return result

    cost_metrics = _resolve_cost_metrics(result, benefit, cost)
    if not cost_metrics:
        return result

    top_n = min(params.get("top_n", len(result)), len(result))

    if method == "Benefit/Cost Ratio":
        score = _benefit_cost_ratio(result, benefit, cost_metrics)
    elif method == "Normalized Ratio":
        score = _normalized_ratio(result, benefit, cost_metrics)
    elif method == "Distance to Ideal":
        score = _distance_to_ideal(result, benefit, cost_metrics)
    elif method == "Composite Cost Ratio":
        score = _composite_cost_ratio(result, benefit, cost_metrics)
        result["efficiency_costs"] = ", ".join(cost_metrics)
    else:
        return result

    result["efficiency_score"] = score
    result = result.sort_values("efficiency_score", ascending=False).copy()

    result["efficiency_rank"] = range(1, len(result) + 1)
    result["efficiency_method"] = method
    result["efficiency_benefit"] = benefit
    result["efficiency_primary_cost"] = cost_metrics[0]

    return result.head(top_n)
