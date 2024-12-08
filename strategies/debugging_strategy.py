import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from transformers import TrainerCallback
import torch

from strategies.base_strategy import BasePipelineStrategy
from managers.model_manager import ModelManager


# Debugging Strategy
class DebuggingStrategy(BasePipelineStrategy, TrainerCallback):
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def on_epoch_end(self, args, state, control, **kwargs):
        self.execute()
        
    def execute(self):
        
        self.model.eval()
        
        print("[INFO] Debugging model output...")
        test_inputs = ["Hello!", "How are you?", "What is the capital of France?"]
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
                    max_length=50,
                    do_sample=True,
                    eos_token_id= self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.5
                )
            
            # Decode and display results
            print(f"\n[INPUT]: {input_text}")
            print(f"[OUTPUT]: {self.tokenizer.decode(outputs[0], skip_special_tokens=True)}\n")
        
        self.model.train()

