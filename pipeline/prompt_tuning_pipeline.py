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


# Pipeline
class PromptTuningPipeline:
    def __init__(self, model_name: str,
                 dataset_path: str, 
                 output_dir: str, 
                 local_model_dir: str,
                 device='cpu'):
                
        self.model_manager = ModelManager(model_name, device=device, 
                                          local_model_dir=local_model_dir)
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.device = device

    def train(self, 
              num_virtual_tokens: int = 100, 
              lr: float = 5e-5, 
              per_device_train_batch_size: int = 1,
              save_steps: int = 50, 
              eval_steps: int = 50, 
              save_total_limit: int = 2, 
              epochs: int = 20
              ):
        
        print("\n====================== load_model_and_tokenizer ======================\n")
        self.model_manager.load_model_and_tokenizer()
        
        print("\n====================== configure_prompt_tuning ======================\n")
        self.model_manager.configure_prompt_tuning(num_virtual_tokens)
        
        print("\n====================== train ======================\n")
        ts = TrainingStrategy(self.model_manager, self.dataset_path, self.output_dir)
        ts.execute(
            lr=lr, 
            per_device_train_batch_size=per_device_train_batch_size,
            save_steps=save_steps, 
            eval_steps=eval_steps, 
            save_total_limit=save_total_limit, 
            epochs=epochs
        )
        print("\n====================== finished training ======================\n")

    def infer(self,
        max_length: int = 128,
        temperature: float = 0.2,
        max_new_tokens: int = 20,
        top_k=40,
        top_p=0.95,
        model_type="peft"
    ):
        
        self.model_manager.load_prompt_tuned_model(self.output_dir)
        
        inf = InferenceStrategy(self.model_manager)
        inf.execute(
            max_length = max_length,
            temperature = temperature,
            max_new_tokens = max_new_tokens,
            top_k=top_k,
            top_p=top_p,
            model_type=model_type
        )

    def evaluate(self):
        self.model_manager.load_prompt_tuned_model(self.output_dir)
        EvaluationStrategy(self.model_manager, self.dataset_path).execute()

    def visualize(self):
        VisualizationStrategy(f"{self.output_dir}/logs").execute()

    def debug(self):
        self.model_manager.load_prompt_tuned_model(self.output_dir)
        DebuggingStrategy(self.model_manager).execute()

    def fine_tune(self, new_dataset_path: str):
        self.model_manager.load_prompt_tuned_model(self.output_dir)
        FineTuningStrategy(self.model_manager, new_dataset_path, self.output_dir).execute()

    # def to(self, device):
    #     self.device = device
    #     self.model_manager.device = device