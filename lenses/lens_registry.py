"""
Lens Registry Module.

Provides dynamic registration and lookup services for analytical lenses
within the Decision Space Explorer architecture.
"""

import logging
from typing import Any, Dict, List, Optional

# Core default analytical lenses
from lenses import (
    lens_consensus,
    lens_diversity,
    lens_efficiency,
    lens_indicator,
    lens_manual,
    lens_preference,
)

logger = logging.getLogger(__name__)

# Private dictionary storing name-to-module/class mappings
_LENS_REGISTRY: Dict[str, Any] = {
    "Manual Selection": lens_manual,
    "Preference": lens_preference,
    "Diversity": lens_diversity,
    "Efficiency": lens_efficiency,
    "Indicator Dominance": lens_indicator,
    "SOI Consensus": lens_consensus,
}


def register_lens(name: str, lens_module: Any, override: bool = False) -> None:
    """
    Dynamically registers a new analytical lens in the framework.

    Parameters
    ----------
    name : str
        The unique display identifier for the lens.
    lens_module : Any
        Module or class implementing the lens logic.
    override : bool, optional
        Whether to overwrite an existing lens entry with the same name.
        Defaults to False.

    Raises
    ------
    ValueError
        If the lens name is already registered and `override` is False.
    """
    if name in _LENS_REGISTRY and not override:
        raise ValueError(
            f"Lens '{name}' is already registered. Set override=True to replace."
        )

    _LENS_REGISTRY[name] = lens_module
    logger.info(f"Lens '{name}' successfully registered.")


def get_lens_names() -> List[str]:
    """
    Retrieves the list of available analytical lens names.

    Returns
    -------
    List[str]
        Ordered list of lens identifiers, prefixed with 'None' as default selection.
    """
    return ["None"] + list(_LENS_REGISTRY.keys())


def get_lens_module(lens_name: str) -> Optional[Any]:
    """
    Retrieves the module or object associated with a registered lens.

    Parameters
    ----------
    lens_name : str
        The unique display name of the target lens.

    Returns
    -------
    Optional[Any]
        The corresponding lens module or class instance, or None if not found.
    """
    return _LENS_REGISTRY.get(lens_name)