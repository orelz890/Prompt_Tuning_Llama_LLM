from configuration.config import Config

conf = Config()

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from transformers import TrainingArguments, Trainer, DataCollatorWithPadding, AdamW, get_scheduler

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
        self.optimizer = None
        self.scheduler = None
        self.device = model_manager.device

    def preprocess_dataset(self) -> tuple:
        return preprocess_and_tokenize(self.dataset_path, self.model_manager.tokenizer)

    def execute(self, **kwargs):
        print("[INFO] Starting training...")
        
        # Merge defaults with provided kwargs
        args = {**conf.TRAIN_HYPER_PARAMETERS, **kwargs}
        
        # Prepare the datasets
        train_dataset, eval_dataset,_= self.preprocess_dataset()
        
        # Processes the batches of data for input into the model.
        data_collator = CustomDataCollatorSameSize( 
                                                   tokenizer=self.model_manager.tokenizer, 
                                                   device=self.device,
                                                   )
        
        # Training args
        training_args = self.create_training_arguments(**args)
        
        # Initialize optimizer and scheduler
        self.setup_optimizer_and_scheduler(lr=args.get("learning_rate"))
        
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
        print("[INFO] Finished Training...")


    def create_trainer(self, training_args, train_dataset, eval_dataset, data_collator):
        trainer = Trainer(
            model=self.model_manager.peft_model_prompt,  # We pass in the PEFT version of the foundation model, bloomz-560M
            args=training_args,  # The args for the training.
            train_dataset=train_dataset,  # The dataset used to tyrain the model.
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            optimizers=(self.optimizer, self.scheduler),  # Custom optimizer and scheduler
            # tokenizer=self.model_manager.tokenizer,
        )
        return trainer

    def create_training_arguments(self, **kwargs):
        
        training_args = TrainingArguments(
            output_dir=self.output_dir,  # Where the model predictions and checkpoints will be written
            logging_dir=f"{self.output_dir}/logs",
            auto_find_batch_size=True,  # Find a suitable batch size that will fit into memory automatically
            load_best_model_at_end=True,
            learning_rate=kwargs.get("learning_rate"),  # Higher learning rate than full Fine-Tuning
            num_train_epochs=kwargs.get("epochs"),
            eval_strategy=kwargs.get("eval_strategy"),
            eval_steps=kwargs.get("eval_steps"),
            save_steps=kwargs.get("save_steps"),
            metric_for_best_model=kwargs.get("metric_for_best_model"),
            save_total_limit=kwargs.get("save_total_limit"),
            fp16=kwargs.get("fp16"),
            max_grad_norm=kwargs.get("max_grad_norm"),        
        )
        
        return training_args

    def setup_optimizer_and_scheduler(self, lr):
        """
        Initializes the Adam optimizer and ReduceLROnPlateau scheduler.
        """
        if self.optimizer is None:
            self.optimizer = Adam(self.model_manager.peft_model_prompt.parameters(), lr=lr)

        self.model_manager.peft_model_prompt.print_trainable_parameters()
        
        # raise("Stop here")
    
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=1,
            threshold=0.01,
            verbose=True
        )
