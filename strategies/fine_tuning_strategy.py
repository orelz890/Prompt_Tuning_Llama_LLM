import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from strategies.base_strategy import BasePipelineStrategy
from managers.model_manager import ModelManager

# Fine-Tuning Strategy
class FineTuningStrategy(BasePipelineStrategy):
    def __init__(self, model_manager: ModelManager, dataset_path: str, output_dir: str):
        self.model_manager = model_manager
        self.dataset_path = dataset_path
        self.output_dir = output_dir

    def execute(self):
        print("[INFO] Starting fine-tuning on new dataset.")
        # TrainingStrategy(self.model_manager, self.dataset_path, self.output_dir).execute()
