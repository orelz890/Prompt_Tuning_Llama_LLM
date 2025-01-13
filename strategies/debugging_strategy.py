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
    
    def __init__(self, model_manager, test_inputs):
        """
            Initialize the DebuggingStrategy with the model_manager.

            Args:
                model: The model_manager containing the model to debug.
        """
        self.model_manager: ModelManager = model_manager
        self.test_inputs = test_inputs

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
        model = self.model_manager.peft_model_prompt
        tokenizer = self.model_manager.tokenizer
        
        model.eval()
        
        print("\n[INFO] Debugging model output...")

        messages = []
        
        print("self.test_inputs = ", self.test_inputs)

        for input_text in self.test_inputs:
            
            # Format the input as an instruct-style message
            messages.append({"role": "user", "content": str(input_text)})
                        
            inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_dict=True, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Generate model response
            outputs = self.model_manager.get_output(
                inputs=inputs,
                model_type="peft",
            )
            
            response = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
            print("[INPUT]: ", input_text)
            print("[MODEL]: ", response)

        model.train()
        print("\n[INFO] Finished Debugging model output...")
