import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import TrainerCallback
import torch
from transformers import pipeline

from strategies.base_strategy import BasePipelineStrategy
from managers.model_manager import ModelManager

# Debugging Strategy
class DebuggingStrategy(BasePipelineStrategy, TrainerCallback):
    """
    A strategy for debugging model outputs at the end of each training epoch.
    Integrates with the Hugging Face Trainer as a callback.

    Attributes:
        model: The model being trained or debugged.
        tokenizer: Tokenizer used for encoding inputs and decoding outputs.
        device (str): The device (e.g., 'cuda', 'cpu') used for computations.
    """
    
    def __init__(self, model, tokenizer, device):
        """
        Initialize the DebuggingStrategy with the model, tokenizer, and device.

        Args:
            model: The model to debug.
            tokenizer: Tokenizer for processing inputs and outputs.
            device (str): Device to run computations ('cuda' or 'cpu').
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def on_epoch_end(self, args, state, control, **kwargs):
        """
        Executes the debugging process at the end of each epoch.

        Args:
            args: Training arguments.
            state: Trainer state.
            control: Trainer control object.
            **kwargs: Additional arguments.
        """
        self.execute()

    def execute(self):
        """
        Perform model debugging by generating outputs for predefined test inputs.
        """
        self.model.eval()
        print("\n[INFO] Debugging model output...")

        # Predefined test inputs
        test_inputs = ["how are you?", "HOW ARE YOU", "what are you doing?", "where are you from", "Are you a bot?"]

        messages = [
            # {"role": "system", "content": "You are a bot that responds to weather queries. You should reply with the unit used in the queried location."},
        ]
        
        for input_text in test_inputs:
            
            # Format the input as an instruct-style message
            messages.append({"role": "user", "content": str(input_text)})
                        
            inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_dict=True, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Generate model response
            outputs = self.model.generate(
                **inputs,
                do_sample=True,             # Enable sampling for varied responses
                # temperature=0.7,            # Control randomness
                top_p=0.9,                  # Nucleus sampling
                max_new_tokens=100,         # Maximum new tokens to generate
                repetition_penalty=1.2,     # Penalize repetition
                # assistant_early_exit=4,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                # cache_implementation="quantized", 
                # cache_config={"nbits": 4, "backend": "quanto"}
            )
            response = self.tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
            print("[INPUT]: ", input_text)
            print("[MODEL]: ", response)

        self.model.train()
        print("\n[INFO] Finished Debugging model output...")
