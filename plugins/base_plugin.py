# plugins/base_plugin.py
from abc import ABC, abstractmethod
from typing import Dict, Iterable, List, Set
import pandas as pd


class DomainPlugin(ABC):
    """
    Clase Base Abstracta para todos los plugins de dominio.
    Cualquier nuevo plugin DEBE heredar de esta clase.
    """

    def __init__(self, var_prefix: str) -> None:
        self.var_prefix = var_prefix

    @abstractmethod
    def available_indicators(self) -> Set[str]:
        """Devuelve el conjunto de indicadores soportados por el plugin."""
        pass

    @abstractmethod
    def requirements(self) -> Dict[str, List[str]]:
        """Devuelve las columnas del DataFrame requeridas para cada indicador."""
        pass

    def decision_variables(self, df: pd.DataFrame) -> List[str]:
        """
        Identifica las variables de decisión basadas en el prefijo.
        (Implementación por defecto reutilizable para todos los plugins).
        """
        if df is None or df.empty:
            return []
        return [c for c in df.columns if str(c).startswith(self.var_prefix)]

    @abstractmethod
    def compute_indicators(
        self, df: pd.DataFrame, indicators: Iterable[str]
    ) -> pd.DataFrame:
        """Calcula los indicadores seleccionados y devuelve el DataFrame enriquecido."""
        pass
