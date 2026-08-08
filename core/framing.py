"""
Core Context Framing Logic.

Pure Python module for range-based bounding and DataFrame filtering.
Contains ZERO GUI dependencies.
"""

from typing import Dict, List, Tuple
import pandas as pd


def get_framing_dimensions(metrics: List[str], indicators: List[str]) -> List[str]:
    """Combines metrics and selected semantic indicators into framing dimensions."""
    return list(metrics) + list(indicators)


def get_dimension_bounds(
    df: pd.DataFrame, dimensions: List[str]
) -> Dict[str, Tuple[float, float]]:
    """
    Computes valid min and max range bounds for each numeric dimension in the DataFrame.
    """
    bounds: Dict[str, Tuple[float, float]] = {}
    if df is None or df.empty:
        return bounds

    for dim in dimensions:
        if dim in df.columns and pd.api.types.is_numeric_dtype(df[dim]):
            min_v = float(df[dim].min())
            max_v = float(df[dim].max())

            if not (pd.isna(min_v) or pd.isna(max_v) or min_v >= max_v):
                bounds[dim] = (min_v, max_v)

    return bounds


def apply_framing_bounds(
    df: pd.DataFrame, bounds: Dict[str, Tuple[float, float]]
) -> pd.DataFrame:
    """
    Filters a DataFrame based on lower and upper range boundaries.
    """
    if df is None or df.empty or not bounds:
        return df

    filtered_df = df.copy()
    for dim, (min_v, max_v) in bounds.items():
        if dim in filtered_df.columns:
            filtered_df = filtered_df[
                (filtered_df[dim] >= min_v) & (filtered_df[dim] <= max_v)
            ]

    return filtered_df