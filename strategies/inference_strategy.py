from configuration.config import Config

conf = Config()

import os
import sys
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from strategies.base_strategy import BasePipelineStrategy
from managers.model_manager import ModelManager

# Suppress the position IDs warning
import warnings

warnings.filterwarnings(
    "ignore",
    message="Position ids are not supported for parameter efficient tuning. Ignoring position ids."
)

# Inference Strategy
class InferenceStrategy(BasePipelineStrategy):
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.device = model_manager.device
        
    def execute(self, model_type: str, **kwargs):
        """
            Execute the inference process in an interactive chat mode.

            Args:
                model_type (str): Specifies the type of model to use ('foundational' or 'peft').
                
                **kwargs: Additional inference configurations, including:
                
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
        
        print("[INFO] Entering interactive chat mode. Type 'exit' to quit.")

        with torch.no_grad():
            messages = [
                # {"role": "system", "content": "You are a bot that responds to weather queries. You should reply with the unit used in the queried location."},
            ]

            while True:
                user_input = str(input("\n[USER]: "))
                if user_input.lower() in {"exit", "quit"}:
                    print("[INFO] Exiting chat.")
                    break
                
                messages.append({"role": "user", "content": user_input})

                inputs = self.model_manager.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_dict=True, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # tokens = [self.model_manager.tokenizer.convert_ids_to_tokens(x) for x in inputs['input_ids']]
                # print(inputs['input_ids']," = ", tokens) 
            
                # Generate output
                outputs = self.model_manager.get_output(
                    inputs=inputs,
                    model_type=model_type,
                    **kwargs
                )

                # Print info about the model inputs and outputs
                self.print_info(inputs=inputs, outputs=outputs)
    
    def print_info(self, inputs, outputs):
        """
            Print detailed information about inputs and outputs during inference.

            Args:
                inputs (dict): Tokenized inputs for the model.
                outputs (torch.Tensor): Generated outputs from the model.
        """
        
        # Convert input IDs to tokens for the entire batch
        response = self.model_manager.tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
        print(f"[MODEL]: {response}\n")
