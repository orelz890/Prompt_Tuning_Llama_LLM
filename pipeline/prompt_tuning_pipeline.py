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


# Pipeline
class PromptTuningPipeline:
    def __init__(self, **kwargs):
        
        args = {**conf.MODELS, "device": conf.DEVICE, **conf.PATHS, **kwargs}
        
        self.dataset_path = args.get("dataset_path")
        self.output_dir = args.get("output_dir")
        self.device = args.get("device")
        
        print(self.dataset_path, self.output_dir,self.device)
        
        self.model_manager = ModelManager(args.get("foundational_model"), 
                                          device = self.device, 
                                          local_model_dir = args.get("local_model_dir"))

    def train(self, dataset_processor, **kwargs):

        self.setup(type="train", **kwargs)
        
        ts = TrainingStrategy(
            model_manager=self.model_manager, 
            dataset_path=self.dataset_path, 
            output_dir=self.output_dir,
            dataset_processor=dataset_processor
            )
        
        ts.execute(**kwargs)
        
    def infer(self, model_type="peft", **kwargs):
        
        self.setup(type="infer")
        
        inf = InferenceStrategy(model_manager=self.model_manager)
        
        inf.execute(
            model_type=model_type,
            **kwargs
        )
    
    def visualize(self, output_dir):
        
        VisualizationStrategy(output_dir).execute()
        
    def setup(self, type: str, **kwargs):
        if type == "train":
            
            self.model_manager.load_model_and_tokenizer()
            
            self.model_manager.configure_prompt_tuning(
                num_virtual_tokens = kwargs.get("num_virtual_tokens") or conf.TRAIN_HYPER_PARAMETERS["num_virtual_tokens"]
            )
        elif type == "infer":
            self.model_manager.load_prompt_tuned_model(self.output_dir)
        else:
            raise ValueError("Invalid PromptTuningPipeline Setup Type")