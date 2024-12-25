from configuration.config import Config

conf = Config()

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import Type

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from transformers import TrainingArguments, Trainer, DataCollatorWithPadding, AdamW, get_scheduler

from strategies.base_strategy import BasePipelineStrategy
from strategies.debugging_strategy import DebuggingStrategy
from managers.model_manager import ModelManager
# from utils.dataset_utils import preprocess_and_create_dataloaders
from utils.DatasetProcessor import DatasetProcessor
from utils.CustomDataCollatorSameSize import CustomDataCollatorSameSize


# Training Strategy
class TrainingStrategy(BasePipelineStrategy):
    """
    A strategy for training prompt-tuned using a preprocessed dataset
    and customizable training configurations.

    Attributes:
        model_manager (ModelManager): Manages the foundational or prompt-tuned model.
        dataset_path (str): Path to the training dataset.
        output_dir (str): Directory for saving the model outputs and checkpoints.
        dataset_processor (Type[DatasetProcessor]): Processor for preparing datasets.
        train_dataset: Training dataset.
        test_dataset: Testing dataset.
        optimizer: Optimizer used during training.
        scheduler: Scheduler for adjusting the learning rate.
        data_collator: Collator for dynamic data padding during training.
        device (str): The device (e.g., 'cuda', 'cpu') used for computations.
    """
    
    def __init__(self, model_manager: ModelManager,
                 dataset_path: str,
                 output_dir: str,
                 dataset_processor: Type[DatasetProcessor] # Reference
                 ):
        """
        Initialize the training strategy with model manager, dataset path, and output configurations.

        Args:
            model_manager (ModelManager): Handles the model and tokenizer.
            dataset_path (str): Path to the dataset used for training.
            output_dir (str): Directory for saving outputs and checkpoints.
            dataset_processor (Type[DatasetProcessor]): Processor class for preparing datasets.
        """
        
        self.model_manager = model_manager
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.device = model_manager.device
        self.dataset_processor = dataset_processor
        self.train_dataset = None
        self.test_dataset = None
        self.optimizer = None
        self.scheduler = None
        self.data_collator = None

    def preprocess_dataset(self, **kwargs) -> tuple:
        """
        Preprocesses the dataset for training and evaluation.

        Returns:
            tuple: Preprocessed train, evaluation, and test datasets.
        """
        
        # Processes the batches of data for input into the model.
        self.data_collator = CustomDataCollatorSameSize( 
            tokenizer=self.model_manager.tokenizer, 
            device=self.device,
        )
        
        dp = self.dataset_processor(
            tokenizer=self.model_manager.tokenizer,
            **kwargs
        )
        
        return dp.preprocess()

    def execute(self, **kwargs):
        """
        Execute the training process, including dataset preparation, trainer initialization, and training.

        Args:
            **kwargs: Additional training configurations, including:

                DATASET:
                - batch_size (int): Batch size for training and evaluation. Default is 16.
                - seed (int): Random seed for reproducibility. Default is 42.
                - test_size (float): Fraction of the dataset to use for testing. Default is 0.2.

                DATA_PATH:
                - dataset_path (str): Path to the dataset. Default is "google/Synthetic-Persona-Chat".

                TOKENIZER:
                - max_length (int): Maximum tokenization length. Default is 512.

        """
        
        print("[INFO] Starting training...")
        
        # Prepare the datasets
        train_dataset, eval_dataset,_= self.preprocess_dataset(
            **{**conf.DATA_PATH, **conf.DATASET, **conf.TOKENIZER}
        )
                
        # Training args
        training_args = self.create_training_arguments( **kwargs)
        
        print(f"len(train_dataset) = {len(train_dataset)}")
        
        # Initialize optimizer and scheduler
        self.setup_optimizer_and_scheduler(data_size=len(train_dataset), **kwargs)
        
        # Train
        trainer = self.create_trainer(
            training_args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=self.data_collator,  # Dynamic padding happens here
        )
        
        trainer.train()

        # Save the trained model
        self.model_manager.save_model(
            output_dir=self.output_dir,
            model_type="peft"
        )
        print("[INFO] Finished Training...")
        

    def create_trainer(self, training_args, train_dataset, eval_dataset, data_collator):
        """
        Creates a Hugging Face Trainer instance for managing the training loop.

        Args:
            training_args (TrainingArguments): Training configuration.
            train_dataset: Dataset used for training.
            eval_dataset: Dataset used for evaluation.
            data_collator: Collator for handling data padding.

        Returns:
            Trainer: A configured Trainer instance.
        """
        
        # Add the custom callback
        test_callback = DebuggingStrategy(
            model=self.model_manager.peft_model_prompt,
            tokenizer=self.model_manager.tokenizer,
            device=self.device
        )
        
        print(train_dataset)
        
        trainer = Trainer(
            model=self.model_manager.peft_model_prompt,  # We pass in the PEFT version of the foundation model
            args=training_args,  # The args for the training.
            train_dataset=train_dataset,  # The dataset used to tyrain the model.
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            optimizers=(self.optimizer, self.scheduler),  # Custom optimizer and scheduler
            callbacks=[test_callback]
            # tokenizer=self.model_manager.tokenizer,
        )
        return trainer

    def create_training_arguments(self, **kwargs):
        """
        Create training arguments using Hugging Face's TrainingArguments.

        Args:
            **kwargs: Additional training configurations.

        Returns:
            TrainingArguments: Configured training arguments.
        """
        
        training_args = TrainingArguments(
            output_dir=self.output_dir,  # Where the model predictions and checkpoints will be written
            logging_dir=f"{self.output_dir}/logs",
            auto_find_batch_size=True,  # Find a suitable batch size that will fit into memory automatically
            load_best_model_at_end=True,
            learning_rate=kwargs.get("learning_rate"),  # Higher learning rate than full Fine-Tuning
            num_train_epochs=kwargs.get("num_train_epochs"),
            eval_strategy=kwargs.get("eval_strategy"),
            eval_steps=kwargs.get("eval_steps"),
            save_steps=kwargs.get("save_steps"),
            logging_steps=kwargs.get("eval_steps"),
            # metric_for_best_model=kwargs.get("metric_for_best_model"),
            # save_total_limit=kwargs.get("save_total_limit"),
            fp16=kwargs.get("fp16"),
            max_grad_norm=kwargs.get("max_grad_norm"),
            per_device_train_batch_size=kwargs.get("batch_size"),
            per_device_eval_batch_size=kwargs.get("batch_size"),
            # gradient_accumulation_steps=kwargs.get("gradient_accumulation_steps"),
            do_train=True,
            do_eval=True
        )
        
        # Print training arguments
        print("\n[INFO] Training Arguments:")
        for key, value in training_args.to_dict().items():
            print(f"{key}: {value}")
    
        return training_args

    def setup_optimizer_and_scheduler(self, data_size, **kwargs):
        """
        Initializes the optimizer and learning rate scheduler.

        Args:
            data_size (int): Size of the training dataset.
            **kwargs: Additional configurations for optimizer and scheduler.
        """
        
        if self.optimizer is None:
            # self.optimizer = Adam(self.model_manager.peft_model_prompt.parameters(), lr=lr)
            optimizer_grouped_parameters = [
                {
                    "params": self.model_manager.peft_model_prompt.parameters(),
                    "weight_decay": kwargs.get("weight_decay")
                }
            ]
            self.optimizer = AdamW(optimizer_grouped_parameters, lr=kwargs.get("learning_rate"))
      
        num_training_steps = (kwargs.get("batch_size") * kwargs.get("epochs") * kwargs.get("num_train_epochs")) * data_size
        
        print("num_training_steps: ", num_training_steps)
        
        # self.scheduler = get_scheduler(
        #     name=kwargs.get("lr_scheduler_type"),
        #     optimizer=self.optimizer,
        #     # num_warmup_steps=0.1 * num_training_steps,
        #     num_warmup_steps=5,
        #     num_training_steps=num_training_steps,
        # )
        
        self.scheduler = ReduceLROnPlateau(
            optimizer = self.optimizer,
            mode = kwargs.get("mode"),
            factor = kwargs.get("factor"),
            patience = kwargs.get("patience"),
            threshold = kwargs.get("threshold"),
            verbose = kwargs.get("verbose"),
            # num_training_steps = num_training_steps,
        )
        
        print("\n[INFO] Optimizer Arguments:")
        print(self.optimizer.state_dict())
        
        print("\n[INFO] Scheduler Arguments:")
        print(self.scheduler.state_dict())