"""
Enrichment Panel UI Component.

Streamlit view component for selecting domain enrichment indicators.
"""

from typing import Any, List
import streamlit as st
from core.pipeline import ParetoExplorerEngine


def get_plugin_indicators(plugin: Any) -> List[str]:
    """
    Inspecciona el plugin de forma segura para obtener la lista de indicadores,
    soportando métodos, propiedades, listas o diccionarios.
    """
    if plugin is None:
        return []

    # Nombres habituales con los que se definen los indicadores en el plugin
    candidate_attrs = [
        "get_available_indicators",
        "available_indicators",
        "get_indicators",
        "indicators",
        "AVAILABLE_INDICATORS",
    ]

    for attr in candidate_attrs:
        if hasattr(plugin, attr):
            val = getattr(plugin, attr)

            # Si es un método (ej. plugin.get_indicators())
            if callable(val):
                try:
                    res = val()
                    if isinstance(res, (list, tuple, set)):
                        return list(res)
                except Exception:
                    pass

            # Si es una lista o tupla (ej. plugin.available_indicators)
            elif isinstance(val, (list, tuple, set)):
                return list(val)

            # Si es un diccionario con las funciones de cálculo
            elif isinstance(val, dict):
                return list(val.keys())

    return []


def render_enrichment_panel(engine: ParetoExplorerEngine) -> List[str]:
    """
    Renderiza los controles de la interfaz para seleccionar indicadores semánticos.
    """
    if not engine.has_data:
        st.info("Load a dataset first to enable enrichment.")
        return []

    if engine.plugin is None:
        st.info("The current dataset has no domain plugin attached. Step skipped.")
        return []

    # 1. Obtener indicadores ofrecidos por la clase del Plugin
    plugin_indicators = get_plugin_indicators(engine.plugin)

    # 2. Obtener indicadores por defecto configurados en config.py
    config_defaults = engine.context.config.get("default_indicators", [])

    # Combinar ambas listas evitando duplicados conservando el orden
    all_options = list(dict.fromkeys(plugin_indicators + config_defaults))

    if not all_options:
        st.warning("No enrichment indicators provided by this plugin or configuration.")
        return []

    # Determinar qué opciones aparecen preseleccionadas por defecto
    default_selection = [ind for ind in config_defaults if ind in all_options]
    if not default_selection:
        default_selection = all_options

    selected_indicators = st.multiselect(
        "Semantic Indicators",
        options=all_options,
        default=default_selection,
        help="Select domain indicators to enrich the dataset solution space.",
    )

    return selected_indicators