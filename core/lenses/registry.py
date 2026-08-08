"""
Analytical Lenses Registry.

Maps lens names to their respective processing modules.
"""

from typing import Any, Dict, List, Optional
import core.lenses.manual as manual_lens
import core.lenses.preference as preference_lens
import core.lenses.efficiency as efficiency_lens
import core.lenses.indicators as indicators_lens
import core.lenses.diversity as diversity_lens
import core.lenses.consensus as consensus_core

# Registro centralizado con los 5 tipos de lentes
LENSES: Dict[str, Any] = {
    "Manual Selection": manual_lens,
    "Preference": preference_lens,
    "Efficiency": efficiency_lens,
    "Indicators": indicators_lens,
    "Diversity": diversity_lens,
    "Consensus": consensus_core,
}


def get_lens_names() -> List[str]:
    """Retorna la lista de nombres de Lentes disponibles para el selectbox."""
    return ["None"] + list(LENSES.keys())


def get_lens(name: str) -> Optional[Any]:
    """Obtiene el módulo de la Lente seleccionada por su nombre."""
    return LENSES.get(name)

