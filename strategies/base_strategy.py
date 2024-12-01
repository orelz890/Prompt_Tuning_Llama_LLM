
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from abc import ABC, abstractmethod


class BasePipelineStrategy(ABC):
    @abstractmethod
    def execute(self):
        pass
