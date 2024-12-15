import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from transformers import TrainerCallback
import torch

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
        Perform model debugging by generating outputs for a predefined set of test inputs.
        Prints the model's predictions, input-output details, and ensures the model
        is returned to training mode after debugging.
        """
        
        self.model.eval()
        
        print("\n[INFO] Debugging model output...")
        test_inputs = ["Hello!", "How are you?", "What is the capital of France?", "Are you a bot?"]
        for input_text in test_inputs:
            inputs = self.tokenizer(input_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512  # Cap input length).to(self.device)
            ).to(self.device)
            
            # Generate output
            with torch.no_grad():  # Disable gradient calculation for inference
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask= inputs["attention_mask"],
                    max_length=20,
                    min_length=1,
                    do_sample=True,
                    eos_token_id= self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.5,
                    # length_penalty=5.0,
                    temperature=1,
                    top_p=0.9,
                    top_k=3,
                    num_beams=3,  # Use beam search
                    early_stopping=True,  # Stop generation on EOS
                )
            
            # Decode and display results
            print(f"\n[INPUT]: {input_text}")
            print(f"[OUTPUT]: {self.tokenizer.decode(outputs[0], skip_special_tokens=True)}")
            input_size = inputs["input_ids"].size(1)
            outputs_size = outputs.size(1)
            print(f"DETAILS: input_size: {input_size}, output_size: {outputs_size}\n")
        
        self.model.train()

