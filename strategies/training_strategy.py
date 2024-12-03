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
        
        # Prepare the datasets
        train_dataset, eval_dataset,_= self.preprocess_dataset()
        
        # Processes the batches of data for input into the model.
        data_collator = CustomDataCollatorSameSize( 
                                                   tokenizer=self.model_manager.tokenizer, 
                                                   device=self.device,
                                                   )
        # Training args
        training_args = self.create_training_arguments(
            learning_rate=lr,
            epochs=epochs,
            eval_steps=eval_steps,
            save_steps=save_steps,
            save_total_limit=save_total_limit
        )
        
        # Trainer
        trainer = self.create_trainer(
            training_args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,  # Dynamic padding happens here
        )
        
        trainer.train()

        self.model_manager.save_model(
            output_dir=self.output_dir,
            model_type="peft"
        )


    def create_trainer(self, training_args, train_dataset, eval_dataset, data_collator):
        trainer = Trainer(
            model=self.model_manager.peft_model_prompt,  # We pass in the PEFT version of the foundation model, bloomz-560M
            args=training_args,  # The args for the training.
            train_dataset=train_dataset,  # The dataset used to tyrain the model.
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            # tokenizer=self.model_manager.tokenizer,
        )
        return trainer

    def create_training_arguments(self, learning_rate, epochs, eval_steps, save_steps, save_total_limit):
        training_args = TrainingArguments(
            output_dir=self.output_dir,  # Where the model predictions and checkpoints will be written
            # use_cpu=True,  # This is necessary for CPU clusters.
            auto_find_batch_size=True,  # Find a suitable batch size that will fit into memory automatically
            learning_rate=learning_rate,  # Higher learning rate than full Fine-Tuning
            num_train_epochs=epochs,
            eval_strategy="steps",
            eval_steps=eval_steps,
            save_steps=save_steps,
            metric_for_best_model="eval_loss",
            save_total_limit=save_total_limit,
            logging_dir=f"{self.output_dir}/logs",
            fp16=True,
            max_grad_norm=1.0, # prevents the model from making large, unstable updates 
            # label_smoothing_factor=0.1, # preventing it from becoming too confident in specific token probabilities
            load_best_model_at_end=True,
        )
        
        return training_args
