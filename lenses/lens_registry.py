## --------------------------------------------------------------------------------------
## lenses/lens_registry.py
## --------------------------------------------------------------------------------------

from lenses import lens_preference
from lenses import lens_diversity
from lenses import lens_efficiency
from lenses import lens_indicator
from lenses import lens_consensus
from lenses import lens_manual  # <--- 1. Importar módulo manual


LENS_REGISTRY = {
    "Manual Selection": lens_manual,  
    "Preference": lens_preference,
    "Diversity": lens_diversity,
    "Efficiency": lens_efficiency,
    "Indicator Dominance": lens_indicator,
    "SOI Consensus": lens_consensus
}


def get_lens_names():

    return [
        "None"
    ] + list(
        LENS_REGISTRY.keys()
    )


def get_lens_module(
    lens_name
):

    return LENS_REGISTRY.get(
        lens_name
    )