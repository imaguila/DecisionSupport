# plugins/__init__.py

from .nrp_plugin import NRPPlugin
from .aerospace_plugin import AerospacePlugin

PLUGIN_REGISTRY = {
    "nrp": NRPPlugin,
    "aerospace": AerospacePlugin
}