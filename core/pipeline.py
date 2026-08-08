"""
Core Framework Pipeline Engine.

Pure Python module holding the state, operations, and Set of Interest (SOI) 
registry for the Pareto analysis workspace. Completely decoupled from UI frameworks.
"""

from typing import Dict, List, Optional, Tuple, Any
import pandas as pd

from core.dataset_loader import ProblemContext
from core.framing import apply_framing_bounds, get_framing_dimensions
from core.lenses.registry import get_lens
from core.soi import SOI, SOIRegistry  # Asegúrate de importar tus clases puras de SOI
from plugins import DomainPlugin


class ParetoExplorerEngine:
    """Central engine managing the lifecycle of the decision space analysis."""

    def __init__(self) -> None:
        self.context: Optional[ProblemContext] = None
        self.enriched_df: Optional[pd.DataFrame] = None
        self.framed_df: Optional[pd.DataFrame] = None
        self._active_df: Optional[pd.DataFrame] = None
        self.framing_bounds: Dict[str, Tuple[float, float]] = {}

            # --- REGISTRO Y ESTADO SOI ---
        self.soi_registry: SOIRegistry = SOIRegistry()
        self.active_soi_id: Optional[str] = None

    def _get_context_id(self, ctx: Optional[ProblemContext]) -> Optional[str]:
        """Genera un identificador único y estable para el contexto actual."""
        if ctx is None:
            return None
        if hasattr(ctx, "case_name") and ctx.case_name:
            return str(ctx.case_name)
        if hasattr(ctx, "config") and isinstance(ctx.config, dict):
            path = ctx.config.get("path_sol")
            if path:
                return str(path)
        if hasattr(ctx, "df") and ctx.df is not None:
            return f"df_{id(ctx.df)}"
        return "unknown_context"

    def load_problem(self, context: ProblemContext) -> None:
        """
        Carga o actualiza el contexto del problema sin perder el estado de enriquecimiento
        o los SOIs guardados cuando Streamlit se re-ejecuta sobre el mismo dataset.
        """
        if context is None:
            return

        current_id = self._get_context_id(self.context)
        new_id = self._get_context_id(context)

        # SI ES EL MISMO DATASET: Preservamos indicadores y SOIs
        if current_id is not None and current_id == new_id:
            prev_indicators = getattr(self.context, "selected_indicators", [])
            self.context = context
            setattr(self.context, "selected_indicators", prev_indicators)
            return

        # SI ES UN DATASET NUEVO: Reiniciamos el estado completo
        self.context = context
        setattr(self.context, "selected_indicators", [])
        self.enriched_df = None
        self.framed_df = None
        self._active_df = None
        self.framing_bounds = {}
        
        # Reiniciar registro de SOIs y filtro activo
        self.soi_registry.clear()
        self.active_soi_id = None

    def enrich(self, indicators: Optional[List[str]] = None) -> pd.DataFrame:
        """Enriquece el dataset calculando los indicadores con el plugin activo."""
        if not self.has_data:
            raise ValueError("Cannot enrich: No problem context loaded.")

        indicators = indicators or []
        setattr(self.context, "selected_indicators", indicators)

        if self.plugin is None or not indicators:
            self.enriched_df = self.raw_df.copy()
        else:
            self.enriched_df = self.plugin.compute_indicators(
                df=self.raw_df, indicators=indicators
            )

        # Resetear el filtro de lente al recalcular indicadores
        self._active_df = None
        self.apply_framing(self.framing_bounds)
        return self.enriched_df

    def apply_framing(
        self, bounds: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> pd.DataFrame:
        """Aplica filtros de rango a todo el espacio de soluciones de trabajo."""
        if not self.has_data:
            raise ValueError("Cannot frame: No problem context loaded.")

        if bounds is not None:
            self.framing_bounds = bounds

        # working_df ya incluye el filtro del SOI activo si existe
        source_df = self.working_df

        if source_df is None or source_df.empty:
            self.framed_df = pd.DataFrame()
        elif not self.framing_bounds:
            self.framed_df = source_df.copy()
        else:
            self.framed_df = apply_framing_bounds(source_df, self.framing_bounds)

        # Resetear la lente al cambiar los límites de framing
        self._active_df = None
        return self.framed_df

    def apply_lens(self, lens_name: str, params: dict) -> pd.DataFrame:
        """Applies the selected analytical lens over the framed solution space."""
        if self.framed_df is None or self.framed_df.empty:
            self._active_df = pd.DataFrame()
            return self._active_df

        if lens_name == "None" or not lens_name:
            self._active_df = self.framed_df.copy()
            return self._active_df

        lens = get_lens(lens_name)
        if lens:
            # Contexto extendido con el registro de SOIs
            context = {
                "metrics": self.metrics,
                "selected_indicators": self.selected_indicators,
                "soi_registry": self.soi_registry,
            }

            # Si la lente es de combinación/consenso, extraemos los objetos SOI reales
            lens_params = dict(params)
            if lens_name == "Consensus" or getattr(lens, "category", "") == "Combination":
                selected_names = lens_params.get("selected_sois", [])
                target_sois = [
                    self.soi_registry.get_by_name(name)
                    for name in selected_names
                    if self.soi_registry.get_by_name(name) is not None
                ]
                lens_params["target_sois"] = target_sois

            self._active_df = lens.apply(self.framed_df, lens_params, context)
        else:
            self._active_df = self.framed_df.copy()

        return self._active_df

    # ------------------------------------------------------------------
    # OPERACIONES Y GESTIÓN DE SOIs (MÉTODO & DATASET ACTIVO)
    # ------------------------------------------------------------------
    def save_current_soi(
        self,
        name: str,
        lens_name: str = "Manual",
        method_name: Optional[str] = None,
        group: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Optional[SOI]:
        """
        Guarda el conjunto de soluciones actualmente visible/activo como un SOI en el registro.
        """
        source_df = self.active_df
        if source_df is None or source_df.empty or "id" not in source_df.columns:
            return None

        solution_ids = source_df["id"].tolist()
        soi_id = f"soi_{id(solution_ids)}_{len(self.soi_registry.list_all()) + 1}"

        soi = SOI(
            id=soi_id,
            name=name,
            solution_ids=solution_ids,
            lens_name=lens_name,
            method_name=method_name,
            group=group,
            params=params or {},
        )
        self.soi_registry.add(soi)
        return soi

    def load_soi_as_dataset(self, soi_id: str) -> None:
        """
        Establece un SOI guardado como el sub-dataset activo.
        Esto provoca que las operaciones subsecuentes de Framing y Lentes
        trabajen únicamente sobre este subconjunto de soluciones.
        """
        if self.soi_registry.get(soi_id):
            self.active_soi_id = soi_id
            # Recalcular framing y resetea la lente para refrescar la tubería
            self.apply_framing(self.framing_bounds)

    def clear_active_soi(self) -> None:
        """Remueve el filtro de SOI activo y restaura el dataset base completo."""
        if self.active_soi_id is not None:
            self.active_soi_id = None
            self.apply_framing(self.framing_bounds)

    # ------------------------------------------------------------------
    # PROPIEDADES Y ATRIBUTOS DE ACCESO RÁPIDO
    # ------------------------------------------------------------------
    @property
    def has_data(self) -> bool:
        return self.context is not None and self.context.df is not None

    @property
    def raw_df(self) -> Optional[pd.DataFrame]:
        return self.context.df if self.context else None

    @property
    def working_df(self) -> Optional[pd.DataFrame]:
        """
        Devuelve enriched_df (o raw_df) FILTRADO por el SOI activo si existe.
        Sirve como punto de entrada base para el Framing y las Lentes.
        """
        base = self.enriched_df if self.enriched_df is not None else self.raw_df
        if base is None:
            return None

        # Si hay un SOI activo cargado, filtramos el DataFrame base por sus IDs
        if self.active_soi_id:
            active_soi = self.soi_registry.get(self.active_soi_id)
            if active_soi and "id" in base.columns:
                return base[base["id"].isin(active_soi.solution_ids)].copy()

        return base

    @property
    def active_df(self) -> Optional[pd.DataFrame]:
        """Devuelve el DF filtrado por la Lente si existe, si no framed_df, si no working_df."""
        if self._active_df is not None:
            return self._active_df
        if self.framed_df is not None:
            return self.framed_df
        return self.working_df

    @property
    def active_soi(self) -> Optional[SOI]:
        """Devuelve el objeto SOI actualmente cargado como sub-dataset activo."""
        if self.active_soi_id:
            return self.soi_registry.get(self.active_soi_id)
        return None

    @property
    def metrics(self) -> List[str]:
        """Devuelve la lista de métricas del problema cargado."""
        if self.context and hasattr(self.context, "metrics"):
            return self.context.metrics
        return []

    @property
    def selected_indicators(self) -> List[str]:
        """Devuelve los indicadores actualmente calculados."""
        return getattr(self.context, "selected_indicators", []) if self.context else []

    @property
    def framing_dimensions(self) -> List[str]:
        """Devuelve la lista combinada de objetivos activos + indicadores calculados."""
        if not self.context:
            return []
        return get_framing_dimensions(self.metrics, self.selected_indicators)

    @property
    def plugin(self) -> Optional[DomainPlugin]:
        return self.context.plugin if self.context else None
    
    @property
    def active_soi(self) -> Optional[SOI]:
        """Devuelve el objeto SOI actualmente cargado como sub-dataset activo."""
        if self.active_soi_id:
            return self.soi_registry.get(self.active_soi_id)
        return None