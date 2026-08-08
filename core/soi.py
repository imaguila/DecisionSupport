"""
Core SOI Models and Registry.
Zero Streamlit dependencies. Usable in Jupyter, scripts, or APIs.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import pandas as pd


@dataclass
class SOI:
    """Representa un Conjunto de Interés (Set of Interest)."""
    id: str
    name: str
    solution_ids: List[Any]
    lens_name: str
    method_name: Optional[str] = None
    group: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M"))

    @property
    def size(self) -> int:
        return len(self.solution_ids)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "ids": self.solution_ids,
            "lens": self.lens_name,
            "method": self.method_name,
            "group": self.group,
            "soi_size": self.size,
            "created_at": self.created_at,
            "params": self.params,
        }


class SOIRegistry:
    """Gestor en memoria de SOIs guardados."""

    def __init__(self):
        self._sois: Dict[str, SOI] = {}

    def add(self, soi: SOI) -> None:
        self._sois[soi.id] = soi

    def get(self, soi_id: str) -> Optional[SOI]:
        return self._sois.get(soi_id)

    def get_by_name(self, name: str) -> Optional[SOI]:
        for soi in self._sois.values():
            if soi.name == name:
                return soi
        return None

    def list_all(self) -> List[SOI]:
        return list(self._sois.values())

    def delete(self, soi_id: str) -> bool:
        if soi_id in self._sois:
            del self._sois[soi_id]
            return True
        return False

    def clear(self) -> None:
        self._sois.clear()

