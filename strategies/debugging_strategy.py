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
        Perform model debugging by generating outputs for predefined test inputs.
        """
        self.model.eval()
        print("\n[INFO] Debugging model output...")
        test_inputs = ["Hello!", "How are you?", "What is the capital of France?", "Are you a bot?"]

        # Conversation history
        conversation_history = ""

        for input_text in test_inputs:
            # Append user input to conversation history
            conversation_history += f" {input_text}" if conversation_history else input_text

            # Tokenize the conversation history
            inputs = self.tokenizer(
                conversation_history,
                return_tensors="pt",
                truncation=True,
                max_length=256
            ).to(self.device)

            print("[DEBUG] Tokenized Input:", inputs)

            # Generate model response
            with torch.no_grad():
                # Ensure only `input_ids` and `attention_mask` are passed
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],        # Pass tokenized input IDs
                    # attention_mask=inputs["attention_mask"],  # Pass attention mask
                    do_sample=True,             # Enable sampling for varied responses
                    temperature=0.7,            # Control randomness
                    top_p=0.9,                  # Nucleus sampling
                    top_k=50,                   # Top-K sampling
                    max_new_tokens=50,          # Limit generated tokens
                    repetition_penalty=1.2,     # Penalize repetition
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

                # Decode and display the response
                response = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
                print(f"[MODEL]: {response}")

        self.model.train()
        print("\n[INFO] Finished Debugging model output...")
