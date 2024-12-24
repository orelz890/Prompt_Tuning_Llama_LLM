from configuration.config import Config

conf = Config()

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from managers.model_manager import ModelManager

from strategies.training_strategy import TrainingStrategy
from strategies.inference_strategy import InferenceStrategy
from strategies.evaluation_strategy import EvaluationStrategy
from strategies.visualization_strategy import VisualizationStrategy
from strategies.debugging_strategy import DebuggingStrategy
from strategies.fine_tuning_strategy import FineTuningStrategy 

from utils.Aviman1DatasetProcessor import Aviman1DatasetProcessor
from utils.GooglePersonaDatasetProcessor import GooglePersonaDatasetProcessor


# Pipeline
class PromptTuningPipeline:
    """
    A pipeline to handle various stages of prompt tuning, including training, inference,
    evaluation, visualization, and debugging. Integrates multiple strategies and a model manager.

    Attributes:
        dataset_path (str): Path to the dataset used for training/inference.
        output_dir (str): Directory to save outputs, such as model checkpoints or visualizations.
        device (str): The device (e.g., 'cuda', 'cpu') on which computations are performed.
        model_manager (ModelManager): Manages the foundational model and its configurations.
    """
    
    def __init__(self, **kwargs):
        
        """
        Initializes the PromptTuningPipeline with configurations and custom arguments.

        Args:
            **kwargs: Additional arguments that override default configurations, such as:
                - foundational_model (str): Name of the foundational model to use.
                - base_output_dir (str): Directory for saving outputs.
                - base_local_model_dir (str): Directory for storing the local model.
                - device (str): Device for computation (e.g., 'cuda', 'cpu').
        """
        
        # Merge defaults with provided kwargs
        args = {**conf.MODELS, "device": conf.DEVICE, **conf.PATHS, **kwargs}
        
        self.dataset_path = args.get("dataset_path")
        self.output_dir = args.get("output_dir")
        self.device = args.get("device")
        self.task_type = args.get("task_type")
        
        print("\n Using: \n", self.dataset_path, self.output_dir, self.device)
        
        self.model_manager = ModelManager(model_name = args.get("foundational_model"), 
                                          device = self.device, 
                                          local_model_dir = args.get("base_local_model_dir"),
                                          auto_tokenizer = args.get("auto_tokenizer"),
                                          model_loader = args.get("model_loader"),
                                          task_type = args.get("task_type"),
                                          )

    def train(self, strategy_class=TrainingStrategy, **kwargs):

        """
        Execute the training process for prompt tuning.

        Args:
            dataset_processor: A callable or object that preprocesses the dataset.
            
            **kwargs: Additional arguments for training configurations, including:
                TRAINING:
                - num_virtual_tokens (int): Number of virtual tokens for prompt tuning. Default is 30.
                - learning_rate (float): Learning rate for optimization. Default is 5e-5.
                - epochs (int): Number of epochs for training. Default is 1.
                - num_train_epochs (int): Total number of training epochs. Default is 20.
                - save_steps (int): Number of steps between model checkpoints. Default is 100.
                - eval_steps (int): Number of steps between evaluations. Default is 100.
                - save_total_limit (int): Maximum number of checkpoints to keep. Default is 2.
                - eval_strategy (str): Evaluation strategy (e.g., 'steps'). Default is "steps".
                - max_grad_norm (float): Maximum gradient norm for gradient clipping. Default is 1.0.
                - fp16 (bool): Whether to use 16-bit floating point precision. Default is True.
                
                SCHEDULER:
                - lr_scheduler_type (str): Type of learning rate scheduler. Default is "linear".
                - init_from_vocab (bool): Whether to initialize from vocabulary. Default is True.
                - mode (str): Scheduler mode (e.g., 'min'). Default is "min".
                - factor (float): Scheduler reduction factor. Default is 0.5.
                - patience (int): Patience for scheduler. Default is 1.
                - threshold (float): Threshold for scheduler. Default is 0.01.
                - verbose (bool): Whether to log scheduler events. Default is True.
                - num_warmup_steps (int): Number of warmup steps for scheduler. Default is 500.
                
                OPTIMIZER:
                - weight_decay (float): Weight decay for optimizer. Default is 0.01.
                
                DATASET:
                - batch_size (int): Batch size for training. Default is 16.
                - seed (int): Random seed for reproducibility. Default is 42.
                - test_size (float): Fraction of dataset to use for testing. Default is 0.2.
        """
        
        self.setup(type="train", **kwargs)

        # Merge defaults with provided kwargs
        args = {**conf.TRAIN_HYPER_PARAMETERS, **conf.SCHEDULER, **conf.OPTIMIZER, **conf.DATASET, **kwargs}

        # Train
        train_strategy = strategy_class(
            model_manager=self.model_manager, 
            dataset_path=self.dataset_path, 
            output_dir=self.output_dir,
            dataset_processor=self.get_specific_data_processor()
            )

        train_strategy.execute(**args)
        
    def infer(self, model_type="peft", strategy_class=InferenceStrategy, **kwargs):
        """
        Perform inference using a pre-trained model (interactive chat).

        Args:
            model_type (str): Type of model to use for inference (e.g., "peft").
            
            **kwargs: Additional arguments for inference configurations, including:

                TOKENIZER:
                - max_length (int): Maximum length for tokenization. Default is 512.
                
                INFERRING:
                - temperature (float): Sampling temperature for inference. Default is 1.
                - top_p (float): Probability threshold for nucleus sampling. Default is 0.9.
                - top_k (int): Number of highest probability vocabulary tokens to keep for top-k filtering. Default is 3.
                - max_tokens_length (int): Maximum length of generated tokens. Default is 20.
                - min_tokens_length (int): Minimum length of generated tokens. Default is 1.
                - num_beams (int): Number of beams for beam search. Default is 3.
                - length_penalty (float): Length penalty for beam search. Default is 5.0.
                - repetition_penalty (float): Penalty for repeated sequences. Default is 1.5.
                - early_stopping (bool): Whether to stop early in beam search. Default is True.
                - do_sample (bool): Whether to sample from the distribution. Default is True.
        """
        
        self.setup(type="infer")
        
        # Merge defaults with provided kwargs
        args = {**conf.INFER_HYPER_PARAMETERS, **conf.TOKENIZER, **kwargs}

        # Infer
        infer_strategy = strategy_class(model_manager=self.model_manager)
        
        infer_strategy.execute(
            model_type=model_type,
            **args
        )
    
    def visualize(self, output_dir, strategy_class=VisualizationStrategy):
        """
        Visualize the outputs or results using a visualization strategy.

        Args:
            output_dir (str): Directory containing results to visualize.
        """
        
        visual_strategy = strategy_class(output_dir)
        visual_strategy.execute()
        
    def setup(self, type: str, **kwargs):
        """
        Setup the pipeline for a specific task type (training or inference).

        Args:
            type (str): Task type, either "train" or "infer".
            **kwargs: Additional setup arguments.

        Raises:
            ValueError: If the task type is invalid.
        """
        
        if type == "train":
            
            self.model_manager.load_model_and_tokenizer()
            
            self.model_manager.configure_prompt_tuning(
                num_virtual_tokens = kwargs.get("num_virtual_tokens") or conf.TRAIN_HYPER_PARAMETERS["num_virtual_tokens"],
            )
        elif type == "infer":
            self.model_manager.load_prompt_tuned_model(self.output_dir)
        else:
            raise ValueError("Invalid PromptTuningPipeline Setup Type")
    
    def get_specific_data_processor(self):
    
        if self.dataset_path == "google/Synthetic-Persona-Chat":
            return GooglePersonaDatasetProcessor
        
        elif self.dataset_path == "Aviman1/Bot_Human_Prompt_Tuning_Dataset":
            return Aviman1DatasetProcessor

        else:
            raise ValueError("You need to implement a data processor for your specific dataset that extends DataProcessor")
    