"""
Lens Engine Module.

Provides execution and orchestration services for dynamic analytical lenses
in multi-objective decision space exploration frameworks.
"""

import logging
from typing import Any, Dict, Optional
import pandas as pd

from lenses.lens_registry import get_lens_module

logger = logging.getLogger(__name__)


def apply_lens(
    df: Optional[pd.DataFrame],
    lens_name: str,
    params: Dict[str, Any],
    dataset: Dict[str, Any],
) -> Optional[pd.DataFrame]:
    """
    Applies a selected analytical lens to a decision space DataFrame.

    Parameters
    ----------
    df : Optional[pd.DataFrame]
        The input dataset representing candidate solutions in the working set.
    lens_name : str
        Identifier of the lens to apply (e.g., 'ParetoFilter', 'KneePoint').
        Passing "None" or an empty string returns an unmodified copy of `df`.
    params : Dict[str, Any]
        User-defined parameters required by the specific lens implementation.
    dataset : Dict[str, Any]
        Global dataset context containing domain metadata and configurations.

    Returns
    -------
    Optional[pd.DataFrame]
        Transformed DataFrame after applying the lens, or a copy of the input
        DataFrame if no transformation is applied or if execution fails safely.
    """
    if df is None or df.empty:
        return df

    if lens_name == "None" or not lens_name:
        return df.copy()

    lens_module = get_lens_module(lens_name)

    if lens_module is None:
        logger.warning(
            f"Lens '{lens_name}' not found in registry. Returning original DataFrame."
        )
        return df.copy()

    try:
        # 1. Functional approach: Module implements .apply(df, params, dataset)
        if hasattr(lens_module, "apply") and callable(lens_module.apply):
            return lens_module.apply(df, params, dataset)

        # 2. Object-Oriented approach: Class instance with .transform(df)
        elif hasattr(lens_module, "transform") and callable(
            lens_module.transform
        ):
            return lens_module.transform(df)

        else:
            raise AttributeError(
                f"Lens '{lens_name}' does not implement a valid 'apply()' or 'transform()' interface."
            )

    except Exception as e:
        logger.error(f"Error executing lens '{lens_name}': {str(e)}")
        # Safe fallback: prevent GUI/Pipeline crash by returning input data copy
        return df.copy()