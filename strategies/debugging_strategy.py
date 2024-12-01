import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from strategies.base_strategy import BasePipelineStrategy
from managers.model_manager import ModelManager


# Debugging Strategy
class DebuggingStrategy(BasePipelineStrategy):
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager

    def execute(self):
        print("[INFO] Debugging model output...")
        test_inputs = ["Hello!", "How are you?", "What is the capital of France?"]
        for input_text in test_inputs:
            inputs = self.model_manager.tokenizer(input_text, return_tensors="pt")
            outputs = self.model_manager.model.generate(
                input_ids=inputs["input_ids"], max_length=50, do_sample=False
            )
            print(f"[INPUT]: {input_text}")
            print(f"[OUTPUT]: {self.model_manager.tokenizer.decode(outputs[0], skip_special_tokens=True)}")
