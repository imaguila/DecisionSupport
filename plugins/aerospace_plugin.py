# plugins/aerospace_plugin.py
import logging
from typing import Dict, Iterable, List, Set
import numpy as np
import pandas as pd
from .base_plugin import DomainPlugin

logger = logging.getLogger(__name__)
EPS: float = 1e-9


class AerospacePlugin(DomainPlugin):

    def __init__(self, var_prefix: str = "var_") -> None:
        super().__init__(var_prefix=var_prefix)

    def available_indicators(self) -> Set[str]:
        return {
            "density",
            "lift_to_drag_ratio",
            "structural_efficiency",
        }

    def requirements(self) -> Dict[str, List[str]]:
        return {
            "density": ["weight"],
            "lift_to_drag_ratio": ["drag", "weight"],
            "structural_efficiency": ["drag", "weight"],
        }

    def compute_indicators(
        self, df: pd.DataFrame, indicators: Iterable[str]
    ) -> pd.DataFrame:
        if df is None or df.empty:
            return df

        result = df.copy()
        vars_cols = self.decision_variables(result)
        n_vars = max(len(vars_cols), 1)

        for indicator in indicators:
            try:
                if indicator == "density":
                    result[indicator] = result["weight"] / n_vars
                # ... resto de tus cálculos ...
            except Exception as exc:
                logger.warning("[AerospacePlugin] Error en '%s': %s", indicator, exc)

        return result
    
    