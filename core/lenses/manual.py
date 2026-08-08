"""
Manual Selection Lens (Core).

Pure analytical filter for isolating candidate solutions by identifier.
"""

from typing import Any, Dict, List, Optional
import pandas as pd
from core.lenses.base import BaseLens


class ManualLens(BaseLens):
    name: str = "Manual Selection"
    category: str = "Manual"
    description: str = "Isolates specific candidate solutions using explicit ID selection."

    def apply(
        self,
        df: pd.DataFrame,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        if df is None or df.empty:
            return df

        selected_ids: List[int] = params.get("selected_ids", [])

        if not selected_ids or "id" not in df.columns:
            # Retorna DataFrame vacío manteniendo el esquema
            return df.iloc[0:0].copy()

        return df[df["id"].isin(selected_ids)].copy()


# ===================================================================
# ENTRADA A NIVEL DE MÓDULO (Para compatibilidad con pipeline.py)
# ===================================================================

_lens_instance = ManualLens()


def apply(
    df: pd.DataFrame,
    params: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Punto de entrada invocado directamente por el pipeline."""
    return _lens_instance.apply(df, params, context)