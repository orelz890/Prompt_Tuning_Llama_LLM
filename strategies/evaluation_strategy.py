import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import TrainingArguments, Trainer

from strategies.base_strategy import BasePipelineStrategy
from managers.model_manager import ModelManager
from utils.Aviman1DatasetProcessor import Aviman1DatasetProcessor
from utils.CustomDataCollatorSameSize import CustomDataCollatorSameSize


# Evaluation Strategy
class EvaluationStrategy(BasePipelineStrategy):
    def __init__(self, model_manager: ModelManager, dataset_path: str):
        self.model_manager = model_manager
        self.dataset_path = dataset_path
        self.device = model_manager.device

    def execute(self):
        print("[INFO] Evaluating the model on the test dataset.")

        # # Preprocess datasets
        # dp = Aviman1DatasetProcessor(data_collator=self.model_manager.tokenizer,
        #                         dataset_path= kwargs.get("dataset_path")
        #                         ,**kwargs)
        # _, eval_dataset, test_dataset =  dp.preprocess()
    
        # #  = preprocess_and_create_dataloaders(self.dataset_path, self.model_manager.tokenizer)

        # # Use the custom collator for consistent padding
        # data_collator = CustomDataCollatorSameSize(tokenizer=self.model_manager.tokenizer, device=self.device)

        # # Evaluate the model
        # print("[INFO] Evaluating...")
        # trainer = Trainer(
        #     model=self.model_manager.model,
        #     args=TrainingArguments(
        #         output_dir="./evaluation_results",
        #         per_device_eval_batch_size=4,
        #         logging_dir="./evaluation_logs"
        #     ),
        #     eval_dataset=test_dataset,  # Use the test dataset for evaluation
        #     data_collator=data_collator,  # Use the custom data collator
        # )

        # # Evaluate and log results
        # results = trainer.evaluate()
        # print(f"[RESULTS]: {results}")

