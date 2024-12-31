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
    A pipeline designed to handle various stages of prompt tuning for machine learning models. 
    This includes tasks such as training, inference, evaluation, visualization, and debugging.

    The pipeline integrates multiple strategies and leverages a `ModelManager` to streamline the 
    management of models and associated components like tokenizers, foundational model and prompt tuning model.

    Attributes:
        dataset_path (str): Path to the dataset used for training and evaluation.
        task_type (TaskType): The type of task the pipeline is handling, defined by the `TaskType` enumeration.
                             Examples: TaskType.CAUSAL_LM, TaskType.SEQ2SEQ_LM, etc.
        output_dir (str): Directory where outputs such as checkpoints and visualizations are saved.
        device (str): The device used for computation (e.g., "cuda" or "cpu").
        model_manager (ModelManager): Manages model-related operations such as loading, configuration and generate output.
    """
    def __init__(self, 
                 model_loader,
                 task_type,
                 tokenizer_class,
                 model_name,
                 dataset_path,
                 local_model_dir,
                 **kwargs):
        
        """
        Initializes the PromptTuningPipeline with configurations and user-defined arguments.

        Args:
            model_loader (class or callable): A class or callable object for loading the model.
                                              Examples: AutoModelForCausalLM, AutoModelForSeq2SeqLM.
            task_type (TaskType): The type of task this pipeline will handle, defined by the `TaskType` enumeration.
                                  Examples: TaskType.CAUSAL_LM, TaskType.SEQ2SEQ_LM, etc.
            tokenizer_class (class): The tokenizer class associated with the model.
                                     Examples: AutoTokenizer.
            model_name (str): Name of the foundational model. (e.g., "unsloth/Llama-3.2-1B-Instruct").
            dataset_path (str): Path to the dataset used for training or evaluation.
            local_model_dir (str): Path to the directory where the model is stored locally.
            **kwargs: Additional arguments to override default configurations. Common keys include:
                - output_dir (str): Directory to save outputs, such as model checkpoints or visualizations.
                - device (str): Device for computation, e.g., "cuda" or "cpu".
        """
        
        # Merge defaults with provided kwargs
        args = {"device": conf.DEVICE, **conf.PATHS, **kwargs}
        
        self.dataset_path = dataset_path
        self.task_type = task_type
        self.output_dir = args.get("output_dir")
        self.device = args.get("device")
        
        print("\n Using: \n", self.dataset_path, self.output_dir, self.device)
        
        self.model_manager = ModelManager(model_name = model_name,
                                          local_model_dir = local_model_dir,
                                          device = self.device, 
                                          tokenizer_class = tokenizer_class,
                                          model_loader = model_loader,
                                          task_type = task_type,
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
            output_dir (str): Directory containing results to visualize (check points).
        """
        
        visual_strategy = strategy_class(output_dir)
        visual_strategy.execute()
        
    def setup(self, type: str, **kwargs):
        """
            Configures the pipeline for a specific task type, such as training or inference.

            Args:
                type (str): The type of task to set up the pipeline for. Supported values:
                    - "train": Configures the pipeline for training.
                    - "infer": Configures the pipeline for inference using a prompt-tuned model.
                **kwargs: Additional parameters for task-specific configurations, such as:
                    - `num_virtual_tokens` (int): The number of virtual tokens to use for prompt tuning during training. 
                        Defaults to the value specified in `conf.TRAIN_HYPER_PARAMETERS["num_virtual_tokens"]`.
            Raises:
                ValueError: If the provided `type` is neither "train" nor "infer".

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
        """
            Returns the appropriate data processor class based on the dataset path.

            This method selects and returns a specific data processor class that corresponds to the 
            dataset being used. Each data processor class is responsible for handling preprocessing, 
            formatting, and loading of the dataset for the pipeline.

            Returns:
                class: The data processor class corresponding to the dataset path.

            Raises:
                ValueError: If the dataset path does not match a known dataset, prompting the user to 
                            implement a custom data processor that extends `DataProcessor`.

        """
        if self.dataset_path == "google/Synthetic-Persona-Chat":
            return GooglePersonaDatasetProcessor
        
        elif self.dataset_path == "Aviman1/Bot_Human_Prompt_Tuning_Dataset":
            return Aviman1DatasetProcessor

        else:
            raise ValueError("You need to implement a data processor for your specific dataset that extends DataProcessor")
    