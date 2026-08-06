"""
Plugins Package Initializer.

Exports registered domain plugins and provides lookup and factory mechanisms
for dynamic plugin instantiation across the framework.
"""

from typing import Any, Dict, List, Optional, Type

from .aerospace_plugin import AerospacePlugin
from .nrp_plugin import NRPPlugin

# =====================================================
# PLUGIN REGISTRY
# =====================================================

PLUGIN_REGISTRY: Dict[str, Type[Any]] = {
    "nrp": NRPPlugin,
    "aerospace": AerospacePlugin,
}

__all__ = [
    "NRPPlugin",
    "AerospacePlugin",
    "PLUGIN_REGISTRY",
    "get_plugin",
    "list_plugins",
]

# =====================================================
# HELPER & FACTORY FUNCTIONS
# =====================================================


def list_plugins() -> List[str]:
    """
    Retrieves a list of all registered domain plugin identifiers.

    Returns
    -------
    List[str]
        List of registered plugin name strings.
    """
    return list(PLUGIN_REGISTRY.keys())


def get_plugin(plugin_name: str, **kwargs: Any) -> Optional[Any]:
    """
    Instantiates and returns a domain plugin by identifier name.

    Parameters
    ----------
    plugin_name : str
        Identifier key of the requested plugin (e.g., 'nrp', 'aerospace').
    **kwargs : Any
        Keyword arguments passed directly to the target plugin's `__init__`.

    Returns
    -------
    Optional[Any]
        An instance of the requested plugin class, or None if key is unrecognized.
    """
    plugin_cls = PLUGIN_REGISTRY.get(plugin_name)
    if plugin_cls is None:
        return None
    return plugin_cls(**kwargs)