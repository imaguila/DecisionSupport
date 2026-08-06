# plugins/base_plugin.py

from abc import ABC, abstractmethod


class DomainPlugin(ABC):

    @abstractmethod
    def available_indicators(self):
        pass

    @abstractmethod
    def compute_indicators(self, df, indicators):
        pass