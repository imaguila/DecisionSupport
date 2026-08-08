"""
Base Lens Module.

Defines the abstract interface for analytical lenses in the Pareto Explorer framework.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import pandas as pd


class BaseLens(ABC):
    """Abstract Base Class that all analytical lenses must implement."""

    name: str = "Base Lens"
    category: str = "General"  # Options: 'Manual', 'Preference', 'Efficiency', 'Diversity'
    description: str = "Base analytical lens abstraction."

    @abstractmethod
    def apply(
        self,
        df: pd.DataFrame,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """
        Applies the analytical transformation or filtering to the solution set.

        Parameters
        ----------
        df : pd.DataFrame
            The input candidate solutions (typically framed_df).
        params : Dict[str, Any]
            User-configured parameters for this lens instance.
        context : Optional[Dict[str, Any]]
            Global context/metadata (e.g. objectives, active indicators).

        Returns
        -------
        pd.DataFrame
            Transformed/filtered DataFrame (the resulting active SOI).
        """
        pass
    