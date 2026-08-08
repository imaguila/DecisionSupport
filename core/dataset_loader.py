"""
Core Dataset Loading and Problem Context Factory.

Pure Python module with ZERO GUI dependencies. Handles CSV parsing,
objective/variable inference, and domain plugin initialization.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, BinaryIO
import pandas as pd

from config import CASES, CaseConfig, get_case_config
from plugins import get_plugin, DomainPlugin


@dataclass
class ProblemContext:
    """
    Data container representing a loaded optimization problem instance.
    """
    df: pd.DataFrame
    config: Dict[str, Any]
    plugin: Optional[DomainPlugin]
    metrics: List[str]
    decision_variables: List[str]
    selected_indicators: List[str] = field(default_factory=list)


def detect_decision_variables(df: pd.DataFrame, prefix: str) -> List[str]:
    """Identifies columns representing decision variables based on prefix."""
    if df is None or df.empty or not prefix:
        return []
    return [str(col) for col in df.columns if str(col).startswith(prefix)]


def infer_numeric_metrics(
    df: pd.DataFrame, 
    var_prefix: str = "x_", 
    exclude_cols: Optional[List[str]] = None
) -> List[str]:
    """Infers candidate objective columns by excluding system and variable columns."""
    if df is None or df.empty:
        return []

    excluded = set(exclude_cols or [])
    system_cols = {
        "id", "ID", "cluster", "cluster_str", "group_label", "group_base",
        "label", "highlight", "highlight_label", "score", "preference_score",
        "preference_rank", "efficiency_score", "efficiency_rank",
        "domain_match_count", "domain_rank", "selected"
    }

    metrics: List[str] = []
    for col in df.columns:
        col_str = str(col)
        if col_str.startswith(var_prefix):
            continue
        if col_str in excluded or col_str in system_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            metrics.append(col_str)

    return metrics


def load_raw_dataframe(source: Union[str, BinaryIO]) -> pd.DataFrame:
    """Reads a CSV file or buffer and ensures a standard 'id' column exists."""
    df = pd.read_csv(source).reset_index(drop=True)
    if "id" not in df.columns:
        df["id"] = range(1, len(df) + 1)
    return df


def create_problem_context(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
    selected_metrics: Optional[List[str]] = None,
) -> ProblemContext:
    """
    Assembles a complete ProblemContext instance.

    Parameters
    ----------
    df : pd.DataFrame
        Loaded solution space DataFrame.
    cfg : Dict[str, Any]
        Domain configuration parameters.
    selected_metrics : Optional[List[str]]
        User-chosen objective metrics. If None, uses config or infers them.
    """
    var_prefix = cfg.get("var_prefix", "x_")
    
    # 1. Initialize Plugin (if configured)
    plugin_name = cfg.get("plugin")
    plugin = get_plugin(plugin_name, var_prefix=var_prefix) if plugin_name else None

    # 2. Determine Objective Metrics
    if selected_metrics is None:
        selected_metrics = cfg.get("metrics", [])
        if not selected_metrics:
            selected_metrics = infer_numeric_metrics(
                df, var_prefix=var_prefix, exclude_cols=cfg.get("exclude_cols")
            )

    # 3. Detect Decision Variables
    decision_variables = detect_decision_variables(df, var_prefix)

    return ProblemContext(
        df=df,
        config=cfg,
        plugin=plugin,
        metrics=selected_metrics,
        decision_variables=decision_variables,
    )


def load_problem_from_case(
    case_name: str, 
    override_metrics: Optional[List[str]] = None
) -> Optional[ProblemContext]:
    """High-level function to load a pre-configured domain case."""
    cfg = get_case_config(case_name)
    if not cfg:
        return None

    path = cfg.get("path_sol")
    if not path:
        return None

    df = load_raw_dataframe(path)
    return create_problem_context(df, cfg, selected_metrics=override_metrics)