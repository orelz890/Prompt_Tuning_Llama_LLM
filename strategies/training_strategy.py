import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import TrainingArguments, Trainer, DataCollatorWithPadding

from strategies.base_strategy import BasePipelineStrategy
from managers.model_manager import ModelManager
from utils.dataset_utils import preprocess_and_tokenize
from utils.custom_data_collator import CustomDataCollatorSameSize


# Training Strategy
class TrainingStrategy(BasePipelineStrategy):
    def __init__(self, model_manager: ModelManager, dataset_path: str, output_dir: str):
        self.model_manager = model_manager
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.train_dataset = None
        self.test_dataset = None
        self.device = model_manager.device

    def preprocess_dataset(self) -> tuple:
        return preprocess_and_tokenize(self.dataset_path, self.model_manager.tokenizer)

    def execute(self, lr, per_device_train_batch_size, save_steps, eval_steps, save_total_limit, epochs):
        print("[INFO] Starting training...")
        train_dataset, eval_dataset,_= self.preprocess_dataset()
        
        data_collator = CustomDataCollatorSameSize(tokenizer=self.model_manager.tokenizer, device=self.device)

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            eval_strategy="steps",
            eval_steps=eval_steps,
            save_steps=save_steps,
            save_total_limit=save_total_limit,
            learning_rate=lr,
            num_train_epochs=epochs,
            logging_dir=f"{self.output_dir}/logs",
            fp16=True,
            max_grad_norm=1.0, # prevents the model from making large, unstable updates 
            # label_smoothing_factor=0.1, # preventing it from becoming too confident in specific token probabilities
            load_best_model_at_end=True,
        )

        trainer = Trainer(
            model=self.model_manager.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.model_manager.tokenizer,
            data_collator=data_collator,  # Dynamic padding happens here
        )
        trainer.train()

        print(f"[INFO] Saving trained model to {self.output_dir}")
        self.model_manager.model.save_pretrained(self.output_dir)
        self.model_manager.tokenizer.save_pretrained(self.output_dir)

