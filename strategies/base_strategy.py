
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from abc import ABC, abstractmethod


class BasePipelineStrategy(ABC):
    """
    Abstract base class for defining pipeline strategies in the application.

    This class provides the structure for implementing different pipeline strategies,
    such as training, inference, visualization, and debugging. Subclasses must override
    the `execute` method to provide the specific logic for the strategy.
    
    Methods:
        execute: Abstract method that must be implemented by subclasses to perform the strategy's core functionality.
    """
    
    @abstractmethod
    def execute(self):
        """
        Perform the core functionality of the strategy.

        This method must be implemented by subclasses.
        """
        
        pass
