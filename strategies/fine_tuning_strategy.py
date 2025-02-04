import os
import sys
from typing import Type

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from strategies.base_strategy import BasePipelineStrategy
from strategies.training_strategy import TrainingStrategy
from utils.DatasetProcessor import DatasetProcessor
from managers.model_manager import ModelManager
from configuration.config import Config

conf = Config()

# Fine-Tuning Strategy
class FineTuningStrategy(BasePipelineStrategy):
    def __init__(self, model_manager: ModelManager, dataset_path: str, dataset_processor: Type[DatasetProcessor], output_dir: str):
        self.model_manager = model_manager
        self.dataset_path = dataset_path
        self.output_dir = output_dir + "_fine_tuned"
        self.dataset_processor = dataset_processor

    def execute(self, **kwargs):
        print("[INFO] Starting fine-tuning on new dataset.")
      
        # Merge defaults with provided kwargs
        args = {**conf.TRAIN_HYPER_PARAMETERS, **conf.SCHEDULER, **conf.OPTIMIZER, **conf.DATASET, **conf.DEBUG, **kwargs}

        # Train
        train_strategy = TrainingStrategy(
            model_manager=self.model_manager, 
            dataset_path=self.dataset_path, 
            output_dir=self.output_dir,
            dataset_processor=self.dataset_processor
            )

        train_strategy.execute(**args)
        